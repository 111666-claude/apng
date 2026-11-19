// Copyright 2018 kts of kettek / Ketchetwahmeegwun Tecumseh Southall. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package apng

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/color"
	"io"
	"os"
	"testing"
)

// --- helpers for building PNG/APNG fixtures in memory ---

func chunk(typ string, data []byte) []byte {
	b := make([]byte, 0, 12+len(data))
	var n [4]byte
	binary.BigEndian.PutUint32(n[:], uint32(len(data)))
	b = append(b, n[:]...)
	b = append(b, typ...)
	b = append(b, data...)
	crc := crc32.NewIEEE()
	crc.Write([]byte(typ))
	crc.Write(data)
	binary.BigEndian.PutUint32(n[:], crc.Sum32())
	return append(b, n[:]...)
}

func ihdrChunk(w, h int) []byte {
	d := make([]byte, 13)
	binary.BigEndian.PutUint32(d[0:4], uint32(w))
	binary.BigEndian.PutUint32(d[4:8], uint32(h))
	d[8] = 8  // bit depth
	d[9] = 6  // truecolor + alpha
	d[10] = 0 // compression
	d[11] = 0 // filter
	d[12] = 0 // interlace
	return chunk("IHDR", d)
}

func actlChunk(frames, loops uint32) []byte {
	d := make([]byte, 8)
	binary.BigEndian.PutUint32(d[0:4], frames)
	binary.BigEndian.PutUint32(d[4:8], loops)
	return chunk("acTL", d)
}

func fctlChunk(seq uint32, w, h, x, y int, delayNum, delayDen uint16, dispose, blend byte) []byte {
	d := make([]byte, 26)
	binary.BigEndian.PutUint32(d[0:4], seq)
	binary.BigEndian.PutUint32(d[4:8], uint32(w))
	binary.BigEndian.PutUint32(d[8:12], uint32(h))
	binary.BigEndian.PutUint32(d[12:16], uint32(x))
	binary.BigEndian.PutUint32(d[16:20], uint32(y))
	binary.BigEndian.PutUint16(d[20:22], delayNum)
	binary.BigEndian.PutUint16(d[22:24], delayDen)
	d[24] = dispose
	d[25] = blend
	return chunk("fcTL", d)
}

// rawFrame renders an NRGBA image as non-interlaced 8-bit RGBA scanlines with
// no filtering, then zlib-compresses it.
func rawFrame(m *image.NRGBA) []byte {
	var raw bytes.Buffer
	b := m.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		raw.WriteByte(0)
		for x := b.Min.X; x < b.Max.X; x++ {
			c := m.NRGBAAt(x, y)
			raw.Write([]byte{c.R, c.G, c.B, c.A})
		}
	}
	var z bytes.Buffer
	zw := zlib.NewWriter(&z)
	zw.Write(raw.Bytes())
	zw.Close()
	return z.Bytes()
}

// splitN splits b into n roughly equal parts, imitating how encoders emit
// several IDAT/fdAT chunks for one frame.
func splitN(b []byte, n int) [][]byte {
	if n < 1 {
		n = 1
	}
	parts := make([][]byte, 0, n)
	size := (len(b) + n - 1) / n
	for len(b) > 0 {
		k := size
		if k > len(b) {
			k = len(b)
		}
		parts = append(parts, b[:k])
		b = b[k:]
	}
	return parts
}

// imageFixture describes one frame of a synthetic APNG.
type imageFixture struct {
	img  *image.NRGBA
	x, y int
	// delay in hundredths of a second, as fcTL stores it
	delayNum, delayDen uint16
	dispose, blend     byte
}

// buildAPNG assembles an APNG. If defaultImage is non-nil and not part of the
// animation, it becomes the IDAT stream and the first animation frame starts
// at fdAT seq 1. chunks controls how many IDAT/fdAT chunks each frame is split
// into.
func buildAPNG(t *testing.T, canvasW, canvasH int, loops uint32, frames []imageFixture, defaultImage *image.NRGBA, chunks int) []byte {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString(pngHeader)
	buf.Write(ihdrChunk(canvasW, canvasH))
	buf.Write(actlChunk(uint32(len(frames)), loops))

	seq := uint32(0)
	var idat []byte
	if defaultImage != nil {
		idat = rawFrame(defaultImage)
	} else if len(frames) > 0 {
		f := frames[0]
		idat = rawFrame(f.img)
		buf.Write(fctlChunk(seq, f.img.Bounds().Dx(), f.img.Bounds().Dy(), f.x, f.y, f.delayNum, f.delayDen, f.dispose, f.blend))
		seq++
	}
	if idat != nil {
		for _, p := range splitN(idat, chunks) {
			buf.Write(chunk("IDAT", p))
		}
	}

	start := 0
	if defaultImage == nil && len(frames) > 0 {
		start = 1
	}
	for i := start; i < len(frames); i++ {
		f := frames[i]
		buf.Write(fctlChunk(seq, f.img.Bounds().Dx(), f.img.Bounds().Dy(), f.x, f.y, f.delayNum, f.delayDen, f.dispose, f.blend))
		seq++
		for _, p := range splitN(rawFrame(f.img), chunks) {
			p = append([]byte{byte(seq >> 24), byte(seq >> 16), byte(seq >> 8), byte(seq)}, p...)
			buf.Write(chunk("fdAT", p))
			seq++
		}
	}
	buf.Write(chunk("IEND", nil))
	return buf.Bytes()
}

func solidNRGBA(w, h int, c color.NRGBA) *image.NRGBA {
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.SetNRGBA(x, y, c)
		}
	}
	return m
}

// readAllStream drains a FrameReader, returning the frames and the terminal
// error.
func readAllStream(t *testing.T, r io.Reader) ([]Frame, *FrameReader, error) {
	t.Helper()
	fr, err := NewFrameReader(r)
	if err != nil {
		return nil, nil, err
	}
	var frames []Frame
	for {
		f, err := fr.Next()
		if err == io.EOF {
			return frames, fr, nil
		}
		if err != nil {
			return frames, fr, err
		}
		frames = append(frames, *f)
	}
}

func sameImage(a, b image.Image) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if a.Bounds() != b.Bounds() {
		return false
	}
	bd := a.Bounds()
	for y := bd.Min.Y; y < bd.Max.Y; y++ {
		for x := bd.Min.X; x < bd.Max.X; x++ {
			r1, g1, b1, a1 := a.At(x, y).RGBA()
			r2, g2, b2, a2 := b.At(x, y).RGBA()
			if r1 != r2 || g1 != g2 || b1 != b2 || a1 != a2 {
				return false
			}
		}
	}
	return true
}

// --- tests against the checked-in fixtures ---

// TestFrameReaderMatchesDecodeAll streams each checked-in APNG and compares
// every frame, pixel for pixel, and all frame metadata against DecodeAll.
func TestFrameReaderMatchesDecodeAll(t *testing.T) {
	for _, path := range []string{
		"tests/WithDefaultFrame.png",
		"tests/WithoutDefaultFrame.png",
		"tests/MultipleIDATs.png",
	} {
		t.Run(path, func(t *testing.T) {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			want, err := DecodeAll(bytes.NewReader(data))
			if err != nil {
				t.Fatalf("DecodeAll: %v", err)
			}
			got, fr, err := readAllStream(t, bytes.NewReader(data))
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			if fr.IsAnimation() != (len(want.Frames) > 0) {
				t.Errorf("IsAnimation = %v", fr.IsAnimation())
			}
			if fr.LoopCount() != want.LoopCount {
				t.Errorf("LoopCount = %d, want %d", fr.LoopCount(), want.LoopCount)
			}
			if len(got) != len(want.Frames) {
				t.Fatalf("got %d frames, want %d", len(got), len(want.Frames))
			}
			anim := 0
			for i := range got {
				if !want.Frames[i].IsDefault {
					anim++
				}
			}
			if fr.NumFrames() != uint32(anim) {
				t.Errorf("NumFrames = %d, want %d", fr.NumFrames(), anim)
			}
			for i := range got {
				g, w := got[i], want.Frames[i]
				if g.IsDefault != w.IsDefault {
					t.Errorf("frame %d: IsDefault = %v, want %v", i, g.IsDefault, w.IsDefault)
				}
				if g.XOffset != w.XOffset || g.YOffset != w.YOffset {
					t.Errorf("frame %d: offset = (%d,%d), want (%d,%d)", i, g.XOffset, g.YOffset, w.XOffset, w.YOffset)
				}
				if g.BlendOp != w.BlendOp || g.DisposeOp != w.DisposeOp {
					t.Errorf("frame %d: ops = %d/%d, want %d/%d", i, g.DisposeOp, g.BlendOp, w.DisposeOp, w.BlendOp)
				}
				if g.DelayNumerator != w.DelayNumerator || g.DelayDenominator != w.DelayDenominator {
					t.Errorf("frame %d: delay = %d/%d, want %d/%d", i, g.DelayNumerator, g.DelayDenominator, w.DelayNumerator, w.DelayDenominator)
				}
				if !sameImage(g.Image, w.Image) {
					t.Errorf("frame %d: pixels differ", i)
				}
			}
		})
	}
}

// TestFrameReaderStopsEarly verifies that a caller can stop mid-animation and
// that the frames it does not ask for are never decoded. The fixture is
// truncated at a frame boundary, which only a reader that stops early can
// accept.
func TestFrameReaderStopsEarly(t *testing.T) {
	full, err := os.ReadFile("tests/WithDefaultFrame.png")
	if err != nil {
		t.Fatal(err)
	}
	// Cut the file just after the second frame's first fdAT chunk group ends.
	// We do this by parsing chunks and truncating after the first fcTL/fdAT
	// pair of the second animation frame.
	cut := len(full)
	seen := 0
	for off := 8; off+8 <= len(full); {
		n := int(binary.BigEndian.Uint32(full[off : off+4]))
		typ := string(full[off+4 : off+8])
		if typ == "fcTL" {
			seen++
			if seen == 3 { // default image fcTL(1), frame 1 fcTL(2), frame 2 fcTL(3)
				cut = off
				break
			}
		}
		off += 12 + n
	}
	truncated := full[:cut]

	// DecodeAll sees a truncated stream and fails.
	if _, err := DecodeAll(bytes.NewReader(truncated)); err == nil {
		t.Fatal("DecodeAll accepted a truncated file")
	}

	// The streaming reader can read the first two frames and stop.
	fr, err := NewFrameReader(bytes.NewReader(truncated))
	if err != nil {
		t.Fatalf("NewFrameReader: %v", err)
	}
	defer fr.Close()
	// The default image plus the first two animation frames are intact.
	for i := 0; i < 3; i++ {
		if _, err := fr.Next(); err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
	}
	// Continuing fails, because the rest of the file is missing.
	if _, err := fr.Next(); err == nil {
		t.Fatal("expected an error reading the truncated rest")
	}
}

// TestFrameReaderReleasesPreviousFrame verifies that Next drops the decoder's
// reference to the previously returned frame, so a long animation does not
// accumulate frames inside the reader.
func TestFrameReaderReleasesPreviousFrame(t *testing.T) {
	f, err := os.Open("tests/WithDefaultFrame.png")
	if err != nil {
		t.Fatal(err)
	}
	fr, err := NewFrameReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer fr.Close()

	first, err := fr.Next()
	if err != nil {
		t.Fatal(err)
	}
	if fr.d.cur != first {
		t.Fatal("decoder should hold the frame last returned")
	}
	if _, err := fr.Next(); err != nil {
		t.Fatal(err)
	}
	if fr.d.cur == first {
		t.Fatal("decoder still holds the previous frame after Next")
	}
	if len(fr.d.a.Frames) > 1 {
		t.Fatalf("FrameReader accumulated %d frames", len(fr.d.a.Frames))
	}
}

// TestFrameReaderDefaultImageNotInAnimation checks the IsDefault reporting for
// a default image that is separate from the animation.
func TestFrameReaderDefaultImageNotInAnimation(t *testing.T) {
	def := solidNRGBA(8, 8, color.NRGBA{10, 20, 30, 255})
	frames := []imageFixture{
		{img: solidNRGBA(8, 8, color.NRGBA{200, 0, 0, 255}), delayNum: 1, delayDen: 10},
		{img: solidNRGBA(4, 4, color.NRGBA{0, 200, 0, 255}), x: 2, y: 3, delayNum: 1, delayDen: 10},
	}
	data := buildAPNG(t, 8, 8, 3, frames, def, 1)

	all, err := DecodeAll(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("DecodeAll: %v", err)
	}
	got, fr, err := readAllStream(t, bytes.NewReader(data))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if !fr.IsAnimation() {
		t.Error("IsAnimation = false, want true")
	}
	if fr.LoopCount() != 3 {
		t.Errorf("LoopCount = %d, want 3", fr.LoopCount())
	}
	if fr.NumFrames() != 2 {
		t.Errorf("NumFrames = %d, want 2", fr.NumFrames())
	}
	if len(got) != 3 {
		t.Fatalf("got %d frames, want 3 (default image + 2 animation frames)", len(got))
	}
	if !got[0].IsDefault {
		t.Error("first frame should be the default image")
	}
	if got[1].IsDefault || got[2].IsDefault {
		t.Error("animation frames must not be marked default")
	}
	// fcTL and fdAT chunks share one sequence counter, so the animation
	// frames' fcTL chunks carry sequence numbers 0 and 2 here.
	if got[1].Sequence != 0 || got[2].Sequence != 2 {
		t.Errorf("sequence numbers = %d,%d, want 0,2", got[1].Sequence, got[2].Sequence)
	}
	for i := range got {
		if !sameImage(got[i].Image, all.Frames[i].Image) {
			t.Errorf("frame %d differs from DecodeAll", i)
		}
	}
}

// TestFrameReaderZeroDelay checks that a zero delay is reported faithfully
// rather than rejected, and that a zero denominator means 1/100 second.
func TestFrameReaderZeroDelay(t *testing.T) {
	frames := []imageFixture{
		{img: solidNRGBA(2, 2, color.NRGBA{1, 2, 3, 255}), delayNum: 0, delayDen: 0},
		{img: solidNRGBA(2, 2, color.NRGBA{4, 5, 6, 255}), delayNum: 7, delayDen: 0},
	}
	data := buildAPNG(t, 2, 2, 0, frames, nil, 1)
	got, _, err := readAllStream(t, bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d frames", len(got))
	}
	if d := got[0].GetDelay(); d != 0 {
		t.Errorf("zero delay = %v, want 0", d)
	}
	if d := got[1].GetDelay(); d != 0.07 {
		t.Errorf("7/0 delay = %v, want 0.07", d)
	}
}

// TestFrameReaderPlainPNG checks that a PNG without acTL yields its single
// image and then EOF, with IsAnimation false.
func TestFrameReaderPlainPNG(t *testing.T) {
	img := solidNRGBA(3, 5, color.NRGBA{9, 8, 7, 255})
	var buf bytes.Buffer
	buf.WriteString(pngHeader)
	buf.Write(ihdrChunk(3, 5))
	for _, p := range splitN(rawFrame(img), 3) {
		buf.Write(chunk("IDAT", p))
	}
	buf.Write(chunk("IEND", nil))

	fr, err := NewFrameReader(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if fr.IsAnimation() {
		t.Error("IsAnimation = true for plain PNG")
	}
	if fr.LoopCount() != 0 || fr.NumFrames() != 0 {
		t.Errorf("LoopCount/NumFrames = %d/%d, want 0/0", fr.LoopCount(), fr.NumFrames())
	}
	f, err := fr.Next()
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if !f.IsDefault {
		t.Error("plain PNG frame should be IsDefault")
	}
	if !sameImage(f.Image, img) {
		t.Error("pixels differ")
	}
	if _, err := fr.Next(); err != io.EOF {
		t.Errorf("second Next = %v, want io.EOF", err)
	}
}

// TestFrameReaderTooLarge verifies that the memory limit rejects an
// oversized frame before decoding it, that the limit is applied to each
// frame's own dimensions, and that DecodeAll is unaffected.
func TestFrameReaderTooLarge(t *testing.T) {
	frames := []imageFixture{
		{img: solidNRGBA(4, 4, color.NRGBA{1, 1, 1, 255}), delayNum: 1, delayDen: 10},
		{img: solidNRGBA(16, 16, color.NRGBA{2, 2, 2, 255}), delayNum: 1, delayDen: 10},
	}
	data := buildAPNG(t, 16, 16, 0, frames, nil, 1)

	if _, err := DecodeAll(bytes.NewReader(data)); err != nil {
		t.Fatalf("DecodeAll must not apply the streaming limit: %v", err)
	}

	fr, err := NewFrameReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer fr.Close()
	// The first frame is 4x4 RGBA (64 bytes) and fits; the second is 16x16
	// RGBA (1024 bytes) and does not.
	fr.MaxFrameBytes = 128
	if _, err := fr.Next(); err != nil {
		t.Fatalf("first frame should fit: %v", err)
	}
	_, err = fr.Next()
	if !errors.Is(err, ErrFrameTooLarge) {
		t.Fatalf("second frame = %v, want ErrFrameTooLarge", err)
	}
	// The failure is terminal and does not return a partial frame.
	if _, err := fr.Next(); !errors.Is(err, ErrFrameTooLarge) {
		t.Fatalf("Next after failure = %v, want ErrFrameTooLarge", err)
	}
}

// TestFrameReaderHugeDimensions checks that a frame far larger than the limit
// is rejected from its header alone, without allocating its pixel buffer.
func TestFrameReaderHugeDimensions(t *testing.T) {
	const dim = 1 << 20 // 1048576 x 1048576 RGBA would be 4 TiB.
	var buf bytes.Buffer
	buf.WriteString(pngHeader)
	buf.Write(ihdrChunk(dim, dim))
	buf.Write(actlChunk(1, 0))
	buf.Write(fctlChunk(0, dim, dim, 0, 0, 1, 10, 0, 0))
	// The payload is intentionally tiny; it must never be decompressed.
	zw := zlib.NewWriter(&bytes.Buffer{})
	zw.Close()
	buf.Write(chunk("IDAT", []byte{0x78, 0x9c, 0x63, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01}))
	buf.Write(chunk("IEND", nil))

	fr, err := NewFrameReader(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("NewFrameReader: %v", err)
	}
	defer fr.Close()
	// Well below the default 1 GiB, but still a hard stop.
	fr.MaxFrameBytes = 64 << 20
	if _, err := fr.Next(); !errors.Is(err, ErrFrameTooLarge) {
		t.Fatalf("Next = %v, want ErrFrameTooLarge", err)
	}
}

// TestFrameReaderOutOfOrderChunks checks that an fdAT that does not follow an
// fcTL, and a sequence number that skips, are rejected.
func TestFrameReaderOutOfOrderChunks(t *testing.T) {
	img := solidNRGBA(4, 4, color.NRGBA{1, 2, 3, 255})
	raw := rawFrame(img)

	t.Run("fdAT without fcTL", func(t *testing.T) {
		var buf bytes.Buffer
		buf.WriteString(pngHeader)
		buf.Write(ihdrChunk(4, 4))
		buf.Write(actlChunk(2, 0))
		buf.Write(fctlChunk(0, 4, 4, 0, 0, 1, 10, 0, 0))
		buf.Write(chunk("IDAT", raw))
		fd := append([]byte{0, 0, 0, 1}, raw...)
		buf.Write(chunk("fdAT", fd))
		buf.Write(chunk("IEND", nil))

		fr, err := NewFrameReader(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatalf("NewFrameReader: %v", err)
		}
		defer fr.Close()
		if _, err := fr.Next(); err != nil {
			t.Fatalf("first frame: %v", err)
		}
		if _, err := fr.Next(); err == nil {
			t.Fatal("accepted an fdAT with no fcTL")
		}
	})

	t.Run("sequence gap", func(t *testing.T) {
		var buf bytes.Buffer
		buf.WriteString(pngHeader)
		buf.Write(ihdrChunk(4, 4))
		buf.Write(actlChunk(2, 0))
		buf.Write(fctlChunk(0, 4, 4, 0, 0, 1, 10, 0, 0))
		buf.Write(chunk("IDAT", raw))
		buf.Write(fctlChunk(1, 4, 4, 0, 0, 1, 10, 0, 0))
		fd := append([]byte{0, 0, 0, 7}, raw...) // skips 2..6
		buf.Write(chunk("fdAT", fd))
		buf.Write(chunk("IEND", nil))

		fr, err := NewFrameReader(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatalf("NewFrameReader: %v", err)
		}
		defer fr.Close()
		if _, err := fr.Next(); err != nil {
			t.Fatalf("first frame: %v", err)
		}
		if _, err := fr.Next(); err == nil {
			t.Fatal("accepted a sequence number gap")
		}
	})
}

// TestFrameReaderChunkOrder checks additional ordering rules that the
// streaming reader enforces while DecodeAll stays permissive.
func TestFrameReaderChunkOrder(t *testing.T) {
	img := solidNRGBA(4, 4, color.NRGBA{1, 2, 3, 255})
	raw := rawFrame(img)

	t.Run("IDAT after fdAT", func(t *testing.T) {
		var buf bytes.Buffer
		buf.WriteString(pngHeader)
		buf.Write(ihdrChunk(4, 4))
		buf.Write(actlChunk(2, 0))
		buf.Write(fctlChunk(0, 4, 4, 0, 0, 1, 10, 0, 0))
		buf.Write(chunk("IDAT", raw))
		buf.Write(fctlChunk(1, 4, 4, 0, 0, 1, 10, 0, 0))
		buf.Write(chunk("fdAT", append([]byte{0, 0, 0, 2}, raw...)))
		buf.Write(chunk("IDAT", raw))
		buf.Write(chunk("IEND", nil))

		fr, err := NewFrameReader(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatal(err)
		}
		defer fr.Close()
		var last error
		for i := 0; i < 3; i++ {
			if _, last = fr.Next(); last != nil {
				break
			}
		}
		var ferr FormatError
		if !errors.As(last, &ferr) {
			t.Fatalf("error = %v, want FormatError", last)
		}
	})

	t.Run("frame count mismatch", func(t *testing.T) {
		var buf bytes.Buffer
		buf.WriteString(pngHeader)
		buf.Write(ihdrChunk(4, 4))
		buf.Write(actlChunk(3, 0)) // promises three frames
		buf.Write(fctlChunk(0, 4, 4, 0, 0, 1, 10, 0, 0))
		buf.Write(chunk("IDAT", raw))
		buf.Write(chunk("IEND", nil))

		fr, err := NewFrameReader(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatal(err)
		}
		defer fr.Close()
		if _, err := fr.Next(); err != nil {
			t.Fatalf("first frame: %v", err)
		}
		if _, err := fr.Next(); err == nil || err == io.EOF {
			t.Fatalf("Next = %v, want a frame count error", err)
		}
	})
}

// TestFrameReaderAnimationOnlyChunks checks chunks that only make sense in an
// APNG.
func TestFrameReaderAnimationOnlyChunks(t *testing.T) {
	img := solidNRGBA(4, 4, color.NRGBA{1, 2, 3, 255})
	raw := rawFrame(img)

	// An fcTL without acTL.
	var noACTL bytes.Buffer
	noACTL.WriteString(pngHeader)
	noACTL.Write(ihdrChunk(4, 4))
	noACTL.Write(fctlChunk(0, 4, 4, 0, 0, 1, 10, 0, 0))
	noACTL.Write(chunk("IDAT", raw))
	noACTL.Write(chunk("IEND", nil))
	if _, err := NewFrameReader(bytes.NewReader(noACTL.Bytes())); err == nil {
		t.Error("accepted an fcTL without acTL")
	}

	// A zero length fdAT chunk cannot hold a sequence number.
	var shortFDAT bytes.Buffer
	shortFDAT.WriteString(pngHeader)
	shortFDAT.Write(ihdrChunk(4, 4))
	shortFDAT.Write(actlChunk(2, 0))
	shortFDAT.Write(fctlChunk(0, 4, 4, 0, 0, 1, 10, 0, 0))
	shortFDAT.Write(chunk("IDAT", raw))
	shortFDAT.Write(fctlChunk(1, 4, 4, 0, 0, 1, 10, 0, 0))
	shortFDAT.Write(chunk("fdAT", nil))
	shortFDAT.Write(chunk("IEND", nil))
	fr, err := NewFrameReader(bytes.NewReader(shortFDAT.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	defer fr.Close()
	if _, err := fr.Next(); err != nil {
		t.Fatalf("first frame: %v", err)
	}
	if _, err := fr.Next(); err == nil {
		t.Error("accepted a zero length fdAT")
	}
}

// TestFrameReaderBadCRC checks that a corrupt chunk checksum is reported.
func TestFrameReaderBadCRC(t *testing.T) {
	frames := []imageFixture{
		{img: solidNRGBA(4, 4, color.NRGBA{1, 2, 3, 255}), delayNum: 1, delayDen: 10},
		{img: solidNRGBA(4, 4, color.NRGBA{4, 5, 6, 255}), delayNum: 1, delayDen: 10},
	}
	data := buildAPNG(t, 4, 4, 0, frames, nil, 1)
	// Flip a bit in the last fdAT chunk's payload (before its 4 byte CRC).
	idx := bytes.LastIndex(data, []byte("fdAT"))
	if idx < 0 {
		t.Fatal("no fdAT chunk found")
	}
	data[idx+20] ^= 0xff

	fr, err := NewFrameReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer fr.Close()
	if _, err := fr.Next(); err != nil {
		t.Fatalf("first frame: %v", err)
	}
	_, err = fr.Next()
	var ferr FormatError
	if !errors.As(err, &ferr) {
		t.Fatalf("second frame = %v, want FormatError", err)
	}
}

// TestFrameReaderTruncated frames checks that a stream cut inside a frame
// reports an error rather than a partial frame, both in Next and NewFrameReader.
func TestFrameReaderTruncated(t *testing.T) {
	frames := []imageFixture{
		{img: solidNRGBA(32, 32, color.NRGBA{1, 2, 3, 255}), delayNum: 1, delayDen: 10},
		{img: solidNRGBA(32, 32, color.NRGBA{4, 5, 6, 255}), delayNum: 1, delayDen: 10},
	}
	data := buildAPNG(t, 32, 32, 0, frames, nil, 1)
	for _, cut := range []int{8, 20, 40, len(data) / 2, len(data) - 12} {
		if cut >= len(data) {
			continue
		}
		_, err := DecodeAll(bytes.NewReader(data[:cut]))
		if err == nil {
			t.Errorf("cut %d: DecodeAll accepted truncated data", cut)
		}
		fr, err := NewFrameReader(bytes.NewReader(data[:cut]))
		if err != nil {
			continue // failing at construction is fine
		}
		for {
			_, err = fr.Next()
			if err != nil {
				break
			}
		}
		if err == io.EOF {
			t.Errorf("cut %d: got io.EOF, want a truncation error", cut)
		}
		fr.Close()
	}
}

// TestFrameReaderNotPNG checks that a non-PNG stream is rejected up front.
func TestFrameReaderNotPNG(t *testing.T) {
	if _, err := NewFrameReader(bytes.NewReader([]byte("not a png at all"))); err == nil {
		t.Fatal("accepted a non-PNG stream")
	}
	var ferr FormatError
	if _, err := NewFrameReader(bytes.NewReader([]byte("not a png at all"))); !errors.As(err, &ferr) {
		t.Fatalf("error = %v, want FormatError", err)
	}
}

// TestFrameReaderClose checks that Close is idempotent, that Next fails after
// it, and that the underlying reader is not closed.
func TestFrameReaderClose(t *testing.T) {
	data, err := os.ReadFile("tests/WithoutDefaultFrame.png")
	if err != nil {
		t.Fatal(err)
	}
	fr, err := NewFrameReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fr.Next(); err != nil {
		t.Fatal(err)
	}
	if err := fr.Close(); err != nil {
		t.Fatal(err)
	}
	if err := fr.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := fr.Next(); !errors.Is(err, ErrFrameReaderClosed) {
		t.Fatalf("Next after Close = %v, want ErrFrameReaderClosed", err)
	}
}

// TestFrameReaderMultiChunkFrames checks a frame whose compressed data is
// spread over many IDAT and fdAT chunks, comparing against DecodeAll.
func TestFrameReaderMultiChunkFrames(t *testing.T) {
	// Use a larger, less compressible image so the frame spans many chunks.
	mk := func(seed uint8) *image.NRGBA {
		m := image.NewNRGBA(image.Rect(0, 0, 64, 64))
		for y := 0; y < 64; y++ {
			for x := 0; x < 64; x++ {
				m.SetNRGBA(x, y, color.NRGBA{uint8(x * 4), uint8(y * 4), seed, 255})
			}
		}
		return m
	}
	frames := []imageFixture{
		{img: mk(1), delayNum: 1, delayDen: 10},
		{img: mk(2), delayNum: 1, delayDen: 10},
		{img: mk(3), delayNum: 1, delayDen: 10},
	}
	data := buildAPNG(t, 64, 64, 0, frames, nil, 7)

	want, err := DecodeAll(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	got, _, err := readAllStream(t, bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want.Frames) {
		t.Fatalf("got %d frames, want %d", len(got), len(want.Frames))
	}
	for i := range got {
		if !sameImage(got[i].Image, want.Frames[i].Image) {
			t.Errorf("frame %d differs", i)
		}
	}
}

// countingReader records how many bytes were consumed, so a test can tell
// whether the reader stopped early or read to the end.
type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// TestFrameReaderDoesNotReadPastStoppedFrame checks that once the caller stops,
// no further input is consumed. This is what makes it usable on a stream that
// stays open while frames are displayed.
func TestFrameReaderDoesNotReadPastStoppedFrame(t *testing.T) {
	data, err := os.ReadFile("tests/MultipleIDATs.png")
	if err != nil {
		t.Fatal(err)
	}
	cr := &countingReader{r: bytes.NewReader(data)}
	fr, err := NewFrameReader(cr)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fr.Next(); err != nil {
		t.Fatal(err)
	}
	afterFirst := cr.n
	if afterFirst >= int64(len(data)) {
		t.Fatalf("first frame consumed the entire stream (%d bytes)", afterFirst)
	}
	if err := fr.Close(); err != nil {
		t.Fatal(err)
	}
	if cr.n != afterFirst {
		t.Fatalf("Close read %d more bytes", cr.n-afterFirst)
	}
}

// TestFrameReaderMetadataBeforePixels checks that loop count and frame count
// are available without decoding any frame, as an animation preview needs.
func TestFrameReaderMetadataBeforePixels(t *testing.T) {
	data, err := os.ReadFile("tests/WithoutDefaultFrame.png")
	if err != nil {
		t.Fatal(err)
	}
	cr := &countingReader{r: bytes.NewReader(data)}
	fr, err := NewFrameReader(cr)
	if err != nil {
		t.Fatal(err)
	}
	if !fr.IsAnimation() || fr.NumFrames() != 4 || fr.LoopCount() != 0 {
		t.Fatalf("metadata = %v/%d/%d", fr.IsAnimation(), fr.NumFrames(), fr.LoopCount())
	}
	// acTL is near the start; the frame data is not.
	if cr.n > int64(len(data))/2 {
		t.Fatalf("NewFrameReader consumed %d of %d bytes", cr.n, len(data))
	}
}

// TestFrameReaderMaxFrameBytesAllowance checks that the limit can be relaxed
// between frames and disabled entirely with a negative value.
func TestFrameReaderMaxFrameBytesAllowance(t *testing.T) {
	frames := []imageFixture{
		{img: solidNRGBA(4, 4, color.NRGBA{1, 1, 1, 255}), delayNum: 1, delayDen: 10},
		{img: solidNRGBA(16, 16, color.NRGBA{2, 2, 2, 255}), delayNum: 1, delayDen: 10},
	}
	data := buildAPNG(t, 16, 16, 0, frames, nil, 1)

	fr, err := NewFrameReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer fr.Close()
	fr.MaxFrameBytes = 128
	if _, err := fr.Next(); err != nil {
		t.Fatal(err)
	}
	// Raise the limit before asking for the larger frame.
	fr.MaxFrameBytes = 1024
	if _, err := fr.Next(); err != nil {
		t.Fatalf("frame within raised limit: %v", err)
	}

	// A negative value disables the limit.
	fr2, err := NewFrameReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer fr2.Close()
	fr2.MaxFrameBytes = -1
	for {
		if _, err := fr2.Next(); err == io.EOF {
			break
		} else if err != nil {
			t.Fatalf("unlimited: %v", err)
		}
	}
}
