// Original PNG code Copyright 2009 The Go Authors.
// Additional APNG enhancements Copyright 2018 Ketchetwahmeegwun
// Tecumseh Southall / kts of kettek.
// All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package apng implements an APNG image decoder.
//
// The PNG specification is at https://www.w3.org/TR/PNG/.
// The APNG specification is at https://wiki.mozilla.org/APNG_Specification
package apng

import (
	"compress/zlib"
	"encoding/binary"
	"fmt"
	"hash"
	"hash/crc32"
	"image"
	"image/color"
	"io"
)

// Color type, as per the PNG spec.
const (
	ctGrayscale      = 0
	ctTrueColor      = 2
	ctPaletted       = 3
	ctGrayscaleAlpha = 4
	ctTrueColorAlpha = 6
)

// A cb is a combination of color type and bit depth.
const (
	cbInvalid = iota
	cbG1
	cbG2
	cbG4
	cbG8
	cbGA8
	cbTC8
	cbP1
	cbP2
	cbP4
	cbP8
	cbTCA8
	cbG16
	cbGA16
	cbTC16
	cbTCA16
)

func cbPaletted(cb int) bool {
	return cbP1 <= cb && cb <= cbP8
}

// pixelStorageBytes returns the number of bytes per pixel used by the in-memory
// image that readImagePass allocates for the given color type. It is generally
// larger than the on-disk size; a 24-bit truecolor image, for example, is
// decoded into a 4 byte per pixel image.RGBA.
func pixelStorageBytes(cb int, useTransparent bool) int64 {
	switch cb {
	case cbG1, cbG2, cbG4, cbG8:
		if useTransparent {
			return 4
		}
		return 1
	case cbP1, cbP2, cbP4, cbP8:
		return 1
	case cbGA8, cbTC8, cbTCA8:
		return 4
	case cbG16:
		if useTransparent {
			return 8
		}
		return 2
	case cbGA16, cbTC16, cbTCA16:
		return 8
	}
	return 8
}

// Filter type, as per the PNG spec.
const (
	ftNone    = 0
	ftSub     = 1
	ftUp      = 2
	ftAverage = 3
	ftPaeth   = 4
	nFilter   = 5
)

// Interlace type.
const (
	itNone  = 0
	itAdam7 = 1
)

// interlaceScan defines the placement and size of a pass for Adam7 interlacing.
type interlaceScan struct {
	xFactor, yFactor, xOffset, yOffset int
}

// interlacing defines Adam7 interlacing, with 7 passes of reduced images.
// See https://www.w3.org/TR/PNG/#8Interlace
var interlacing = []interlaceScan{
	{8, 8, 0, 0},
	{8, 8, 4, 0},
	{4, 8, 0, 4},
	{4, 4, 2, 0},
	{2, 4, 0, 2},
	{2, 2, 1, 0},
	{1, 2, 0, 1},
}

// Decoding stage.
// The PNG specification says that the IHDR, PLTE (if present), tRNS (if
// present), IDAT and IEND chunks must appear in that order. There may be
// multiple IDAT chunks, and IDAT chunks must be sequential (i.e. they may not
// have any other chunks between them).
// https://www.w3.org/TR/PNG/#5ChunkOrdering
const (
	dsStart = iota
	dsSeenIHDR
	dsSeenPLTE
	dsSeentRNS
	dsSeenIDAT
	dsSeenfdAT
	dsSeenIEND
)

const pngHeader = "\x89PNG\r\n\x1a\n"

type decoder struct {
	r             io.Reader
	numFrames     uint32
	a             APNG
	frameIndex    int
	cur           *Frame
	crc           hash.Hash32
	width, height int
	depth         int
	palette       color.Palette
	cb            int
	stage         int
	idatLength    uint32
	tmp           [3 * 256]byte
	interlace     int

	// maxFrameBytes bounds the memory used for a single frame's decoded
	// pixel buffer. A non-positive value disables the limit; this is the
	// case for DecodeAll and Decode, whose behavior must not change.
	maxFrameBytes int64

	// strict enables the additional format checks that FrameReader performs
	// on top of what DecodeAll has historically accepted: APNG sequence
	// numbers, frame dimensions and frame placement on the canvas.
	strict bool
	// seqExpected is the next expected APNG sequence number in strict mode.
	seqExpected uint32
	// stream indicates that only the frame in cur is retained. It is set by
	// NewFrameReader; DecodeAll and Decode leave it false.
	stream bool
	// pendingFcTL reports whether cur was described by an fcTL chunk that has
	// not yet produced a complete frame.
	pendingFcTL bool
	// seenACTL reports whether an acTL chunk has been read.
	seenACTL bool
	// fcTLCount is the number of fcTL chunks read in stream mode.
	fcTLCount uint32

	// useTransparent and transparent are used for grayscale and truecolor
	// transparency, as opposed to palette transparency.
	useTransparent bool
	transparent    [6]byte
}

// A chunkHeader is the 8 byte prefix of a PNG chunk: a 4 byte big-endian
// length followed by the 4 byte chunk type.
type chunkHeader struct {
	length uint32
	typ    string
}

// readChunkHeader reads a chunk header and initializes the CRC state with the
// chunk type, leaving d.r positioned at the chunk data.
func (d *decoder) readChunkHeader() (chunkHeader, error) {
	if _, err := io.ReadFull(d.r, d.tmp[:8]); err != nil {
		return chunkHeader{}, err
	}
	d.crc.Reset()
	d.crc.Write(d.tmp[4:8])
	return chunkHeader{
		length: binary.BigEndian.Uint32(d.tmp[:4]),
		typ:    string(d.tmp[4:8]),
	}, nil
}

// skipChunk consumes and verifies the data and checksum of a chunk whose
// header has already been read by readChunkHeader.
func (d *decoder) skipChunk(length uint32) error {
	if length > 0x7fffffff {
		return FormatError(fmt.Sprintf("Bad chunk length: %d", length))
	}
	var ignored [4096]byte
	for length > 0 {
		n, err := io.ReadFull(d.r, ignored[:min(len(ignored), int(length))])
		if err != nil {
			return err
		}
		d.crc.Write(ignored[:n])
		length -= uint32(n)
	}
	return d.verifyChecksum()
}

// A FormatError reports that the input is not a valid PNG.
type FormatError string

func (e FormatError) Error() string { return "apng: invalid format: " + string(e) }

var chunkOrderError = FormatError("chunk out of order")

// An UnsupportedError reports that the input uses a valid but unimplemented PNG feature.
type UnsupportedError string

func (e UnsupportedError) Error() string { return "apng: unsupported feature: " + string(e) }

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (d *decoder) parseIHDR(length uint32) error {
	if length != 13 {
		return FormatError("bad IHDR length")
	}
	if _, err := io.ReadFull(d.r, d.tmp[:13]); err != nil {
		return err
	}
	d.crc.Write(d.tmp[:13])
	if d.tmp[10] != 0 {
		return UnsupportedError("compression method")
	}
	if d.tmp[11] != 0 {
		return UnsupportedError("filter method")
	}
	if d.tmp[12] != itNone && d.tmp[12] != itAdam7 {
		return FormatError("invalid interlace method")
	}
	d.interlace = int(d.tmp[12])

	w := int32(binary.BigEndian.Uint32(d.tmp[0:4]))
	h := int32(binary.BigEndian.Uint32(d.tmp[4:8]))
	if w <= 0 || h <= 0 {
		return FormatError("non-positive dimension")
	}
	nPixels64 := int64(w) * int64(h)
	nPixels := int(nPixels64)
	if nPixels64 != int64(nPixels) {
		return UnsupportedError("dimension overflow")
	}
	// There can be up to 8 bytes per pixel, for 16 bits per channel RGBA.
	if nPixels != (nPixels*8)/8 {
		return UnsupportedError("dimension overflow")
	}

	d.cb = cbInvalid
	d.depth = int(d.tmp[8])
	switch d.depth {
	case 1:
		switch d.tmp[9] {
		case ctGrayscale:
			d.cb = cbG1
		case ctPaletted:
			d.cb = cbP1
		}
	case 2:
		switch d.tmp[9] {
		case ctGrayscale:
			d.cb = cbG2
		case ctPaletted:
			d.cb = cbP2
		}
	case 4:
		switch d.tmp[9] {
		case ctGrayscale:
			d.cb = cbG4
		case ctPaletted:
			d.cb = cbP4
		}
	case 8:
		switch d.tmp[9] {
		case ctGrayscale:
			d.cb = cbG8
		case ctTrueColor:
			d.cb = cbTC8
		case ctPaletted:
			d.cb = cbP8
		case ctGrayscaleAlpha:
			d.cb = cbGA8
		case ctTrueColorAlpha:
			d.cb = cbTCA8
		}
	case 16:
		switch d.tmp[9] {
		case ctGrayscale:
			d.cb = cbG16
		case ctTrueColor:
			d.cb = cbTC16
		case ctGrayscaleAlpha:
			d.cb = cbGA16
		case ctTrueColorAlpha:
			d.cb = cbTCA16
		}
	}
	if d.cb == cbInvalid {
		return UnsupportedError(fmt.Sprintf("bit depth %d, color type %d", d.tmp[8], d.tmp[9]))
	}
	d.a.Frames[0].width, d.a.Frames[0].height = int(w), int(h)
	d.width, d.height = int(w), int(h)
	return d.verifyChecksum()
}

func (d *decoder) parsePLTE(length uint32) error {
	np := int(length / 3) // The number of palette entries.
	if length%3 != 0 || np <= 0 || np > 256 || np > 1<<uint(d.depth) {
		return FormatError("bad PLTE length")
	}
	n, err := io.ReadFull(d.r, d.tmp[:3*np])
	if err != nil {
		return err
	}
	d.crc.Write(d.tmp[:n])
	switch d.cb {
	case cbP1, cbP2, cbP4, cbP8:
		d.palette = make(color.Palette, 256)
		for i := 0; i < np; i++ {
			d.palette[i] = color.RGBA{d.tmp[3*i+0], d.tmp[3*i+1], d.tmp[3*i+2], 0xff}
		}
		for i := np; i < 256; i++ {
			// Initialize the rest of the palette to opaque black. The spec (section
			// 11.2.3) says that "any out-of-range pixel value found in the image data
			// is an error", but some real-world PNG files have out-of-range pixel
			// values. We fall back to opaque black, the same as libpng 1.5.13;
			// ImageMagick 6.5.7 returns an error.
			d.palette[i] = color.RGBA{0x00, 0x00, 0x00, 0xff}
		}
		d.palette = d.palette[:np]
	case cbTC8, cbTCA8, cbTC16, cbTCA16:
		// As per the PNG spec, a PLTE chunk is optional (and for practical purposes,
		// ignorable) for the ctTrueColor and ctTrueColorAlpha color types (section 4.1.2).
	default:
		return FormatError("PLTE, color type mismatch")
	}
	return d.verifyChecksum()
}

func (d *decoder) parsetRNS(length uint32) error {
	switch d.cb {
	case cbG1, cbG2, cbG4, cbG8, cbG16:
		if length != 2 {
			return FormatError("bad tRNS length")
		}
		n, err := io.ReadFull(d.r, d.tmp[:length])
		if err != nil {
			return err
		}
		d.crc.Write(d.tmp[:n])

		copy(d.transparent[:], d.tmp[:length])
		switch d.cb {
		case cbG1:
			d.transparent[1] *= 0xff
		case cbG2:
			d.transparent[1] *= 0x55
		case cbG4:
			d.transparent[1] *= 0x11
		}
		d.useTransparent = true

	case cbTC8, cbTC16:
		if length != 6 {
			return FormatError("bad tRNS length")
		}
		n, err := io.ReadFull(d.r, d.tmp[:length])
		if err != nil {
			return err
		}
		d.crc.Write(d.tmp[:n])

		copy(d.transparent[:], d.tmp[:length])
		d.useTransparent = true

	case cbP1, cbP2, cbP4, cbP8:
		if length > 256 {
			return FormatError("bad tRNS length")
		}
		n, err := io.ReadFull(d.r, d.tmp[:length])
		if err != nil {
			return err
		}
		d.crc.Write(d.tmp[:n])

		if len(d.palette) < n {
			d.palette = d.palette[:n]
		}
		for i := 0; i < n; i++ {
			rgba := d.palette[i].(color.RGBA)
			d.palette[i] = color.NRGBA{rgba.R, rgba.G, rgba.B, d.tmp[i]}
		}

	default:
		return FormatError("tRNS, color type mismatch")
	}
	return d.verifyChecksum()
}

// Read presents one or more IDAT chunks as one continuous stream (minus the
// intermediate chunk headers and footers). If the PNG data looked like:
//
//	... len0 IDAT xxx crc0 len1 IDAT yy crc1 len2 IEND crc2
//
// then this reader presents xxxyy. For well-formed PNG data, the decoder state
// immediately before the first Read call is that d.r is positioned between the
// first IDAT and xxx, and the decoder state immediately after the last Read
// call is that d.r is positioned between yy and crc1.
func (d *decoder) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	for d.idatLength == 0 {
		// We have exhausted an IDAT chunk. Verify the checksum of that chunk.
		if err := d.verifyChecksum(); err != nil {
			return 0, err
		}
		// Read the length and chunk type of the next chunk, and check that
		// it is an IDAT chunk.
		if _, err := io.ReadFull(d.r, d.tmp[:8]); err != nil {
			return 0, err
		}

		if d.stage < dsSeenfdAT {
			d.idatLength = binary.BigEndian.Uint32(d.tmp[:4])
			if string(d.tmp[4:8]) != "IDAT" {
				return 0, FormatError(fmt.Sprintf("expected IDAT, found %s", string(d.tmp[4:8])))
			}
		} else {
			length := binary.BigEndian.Uint32(d.tmp[:4])
			if d.strict && length < 4 {
				return 0, FormatError("bad fdAT length")
			}
			d.idatLength = length - 4
			if string(d.tmp[4:8]) != "fdAT" {
				return 0, FormatError(fmt.Sprintf("expected fdAT, found %s", string(d.tmp[4:8])))
			}
		}
		d.crc.Reset()
		d.crc.Write(d.tmp[4:8])
		if d.stage >= dsSeenfdAT {
			if _, err := io.ReadFull(d.r, d.tmp[:4]); err != nil {
				return 0, err
			}
			if err := d.checkSequence(binary.BigEndian.Uint32(d.tmp[:4])); err != nil {
				return 0, err
			}
			d.crc.Write(d.tmp[:4])
		}
	}
	if int(d.idatLength) < 0 {
		return 0, UnsupportedError("IDAT/fdAT chunk length overflow")
	}
	n, err := d.r.Read(p[:min(len(p), int(d.idatLength))])
	d.crc.Write(p[:n])
	d.idatLength -= uint32(n)
	return n, err
}

// decode decodes the IDAT data into an image.
func (d *decoder) decode() (image.Image, error) {
	r, err := zlib.NewReader(d)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var img image.Image
	if d.interlace == itNone {
		img, err = d.readImagePass(r, 0, false)
		if err != nil {
			return nil, err
		}
	} else if d.interlace == itAdam7 {
		// Allocate a blank image of the full size.
		img, err = d.readImagePass(nil, 0, true)
		if err != nil {
			return nil, err
		}
		for pass := 0; pass < 7; pass++ {
			imagePass, err := d.readImagePass(r, pass, false)
			if err != nil {
				return nil, err
			}
			if imagePass != nil {
				d.mergePassInto(img, imagePass, pass)
			}
		}
	}

	// Check for EOF, to verify the zlib checksum.
	n := 0
	for i := 0; n == 0 && err == nil; i++ {
		if i == 100 {
			return nil, io.ErrNoProgress
		}
		n, err = r.Read(d.tmp[:1])
	}
	if err != nil && err != io.EOF {
		return nil, FormatError(err.Error())
	}
	if n != 0 || d.idatLength != 0 {
		return nil, FormatError("too much pixel data")
	}

	return img, nil
}

// targetFrame returns the Frame that the decoder is currently filling. For
// DecodeAll and Decode this is the frame at frameIndex in the accumulated
// APNG; a FrameReader sets stream and instead uses cur, so that frames which
// have already been consumed are not retained.
func (d *decoder) targetFrame() *Frame {
	if d.stream {
		if d.cur == nil {
			d.cur = &Frame{}
		}
		return d.cur
	}
	if d.frameIndex >= len(d.a.Frames) {
		d.a.Frames = append(d.a.Frames, Frame{})
	}
	return &d.a.Frames[d.frameIndex]
}

// readImagePass reads a single image pass, sized according to the pass number.
func (d *decoder) readImagePass(r io.Reader, pass int, allocateOnly bool) (image.Image, error) {
	bitsPerPixel := 0
	pixOffset := 0
	var (
		gray     *image.Gray
		rgba     *image.RGBA
		paletted *image.Paletted
		nrgba    *image.NRGBA
		gray16   *image.Gray16
		rgba64   *image.RGBA64
		nrgba64  *image.NRGBA64
		img      image.Image
		width    int
		height   int
	)
	f := d.targetFrame()
	width = f.width
	height = f.height
	if d.interlace == itAdam7 && !allocateOnly {
		p := interlacing[pass]
		// Add the multiplication factor and subtract one, effectively rounding up.
		width = (width - p.xOffset + p.xFactor - 1) / p.xFactor
		height = (height - p.yOffset + p.yFactor - 1) / p.yFactor
		// A PNG image can't have zero width or height, but for an interlaced
		// image, an individual pass might have zero width or height. If so, we
		// shouldn't even read a per-row filter type byte, so return early.
		if width == 0 || height == 0 {
			return nil, nil
		}
	}
	if d.maxFrameBytes > 0 {
		// Reject before allocating anything. The division form avoids an
		// overflow in width*height*bpp.
		if width <= 0 || height <= 0 {
			return nil, FormatError("non-positive frame dimension")
		}
		bpp := pixelStorageBytes(d.cb, d.useTransparent)
		// An interlaced frame allocates a full-size image plus one image per
		// Adam7 pass; the passes together cover every pixel once, so double
		// the per-pixel figure to bound the total.
		if d.interlace == itAdam7 {
			bpp *= 2
		}
		if int64(width) > d.maxFrameBytes/int64(height)/bpp {
			return nil, fmt.Errorf("%w: %dx%d frame needs more than %d bytes of pixel storage",
				ErrFrameTooLarge, width, height, d.maxFrameBytes)
		}
	}
	switch d.cb {
	case cbG1, cbG2, cbG4, cbG8:
		bitsPerPixel = d.depth
		if d.useTransparent {
			nrgba = image.NewNRGBA(image.Rect(0, 0, width, height))
			img = nrgba
		} else {
			gray = image.NewGray(image.Rect(0, 0, width, height))
			img = gray
		}
	case cbGA8:
		bitsPerPixel = 16
		nrgba = image.NewNRGBA(image.Rect(0, 0, width, height))
		img = nrgba
	case cbTC8:
		bitsPerPixel = 24
		if d.useTransparent {
			nrgba = image.NewNRGBA(image.Rect(0, 0, width, height))
			img = nrgba
		} else {
			rgba = image.NewRGBA(image.Rect(0, 0, width, height))
			img = rgba
		}
	case cbP1, cbP2, cbP4, cbP8:
		bitsPerPixel = d.depth
		paletted = image.NewPaletted(image.Rect(0, 0, width, height), d.palette)
		img = paletted
	case cbTCA8:
		bitsPerPixel = 32
		nrgba = image.NewNRGBA(image.Rect(0, 0, width, height))
		img = nrgba
	case cbG16:
		bitsPerPixel = 16
		if d.useTransparent {
			nrgba64 = image.NewNRGBA64(image.Rect(0, 0, width, height))
			img = nrgba64
		} else {
			gray16 = image.NewGray16(image.Rect(0, 0, width, height))
			img = gray16
		}
	case cbGA16:
		bitsPerPixel = 32
		nrgba64 = image.NewNRGBA64(image.Rect(0, 0, width, height))
		img = nrgba64
	case cbTC16:
		bitsPerPixel = 48
		if d.useTransparent {
			nrgba64 = image.NewNRGBA64(image.Rect(0, 0, width, height))
			img = nrgba64
		} else {
			rgba64 = image.NewRGBA64(image.Rect(0, 0, width, height))
			img = rgba64
		}
	case cbTCA16:
		bitsPerPixel = 64
		nrgba64 = image.NewNRGBA64(image.Rect(0, 0, width, height))
		img = nrgba64
	}
	if allocateOnly {
		return img, nil
	}
	bytesPerPixel := (bitsPerPixel + 7) / 8

	// The +1 is for the per-row filter type, which is at cr[0].
	rowSize := 1 + (int64(bitsPerPixel)*int64(width)+7)/8
	if rowSize != int64(int(rowSize)) {
		return nil, UnsupportedError("dimension overflow")
	}
	// cr and pr are the bytes for the current and previous row.
	cr := make([]uint8, rowSize)
	pr := make([]uint8, rowSize)

	for y := 0; y < height; y++ {
		// Read the decompressed bytes.
		_, err := io.ReadFull(r, cr)
		if err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return nil, FormatError("not enough pixel data")
			}
			return nil, err
		}

		// Apply the filter.
		cdat := cr[1:]
		pdat := pr[1:]
		switch cr[0] {
		case ftNone:
			// No-op.
		case ftSub:
			for i := bytesPerPixel; i < len(cdat); i++ {
				cdat[i] += cdat[i-bytesPerPixel]
			}
		case ftUp:
			for i, p := range pdat {
				cdat[i] += p
			}
		case ftAverage:
			// The first column has no column to the left of it, so it is a
			// special case. We know that the first column exists because we
			// check above that width != 0, and so len(cdat) != 0.
			for i := 0; i < bytesPerPixel; i++ {
				cdat[i] += pdat[i] / 2
			}
			for i := bytesPerPixel; i < len(cdat); i++ {
				cdat[i] += uint8((int(cdat[i-bytesPerPixel]) + int(pdat[i])) / 2)
			}
		case ftPaeth:
			filterPaeth(cdat, pdat, bytesPerPixel)
		default:
			return nil, FormatError("bad filter type")
		}

		// Convert from bytes to colors.
		switch d.cb {
		case cbG1:
			if d.useTransparent {
				ty := d.transparent[1]
				for x := 0; x < width; x += 8 {
					b := cdat[x/8]
					for x2 := 0; x2 < 8 && x+x2 < width; x2++ {
						ycol := (b >> 7) * 0xff
						acol := uint8(0xff)
						if ycol == ty {
							acol = 0x00
						}
						nrgba.SetNRGBA(x+x2, y, color.NRGBA{ycol, ycol, ycol, acol})
						b <<= 1
					}
				}
			} else {
				for x := 0; x < width; x += 8 {
					b := cdat[x/8]
					for x2 := 0; x2 < 8 && x+x2 < width; x2++ {
						gray.SetGray(x+x2, y, color.Gray{(b >> 7) * 0xff})
						b <<= 1
					}
				}
			}
		case cbG2:
			if d.useTransparent {
				ty := d.transparent[1]
				for x := 0; x < width; x += 4 {
					b := cdat[x/4]
					for x2 := 0; x2 < 4 && x+x2 < width; x2++ {
						ycol := (b >> 6) * 0x55
						acol := uint8(0xff)
						if ycol == ty {
							acol = 0x00
						}
						nrgba.SetNRGBA(x+x2, y, color.NRGBA{ycol, ycol, ycol, acol})
						b <<= 2
					}
				}
			} else {
				for x := 0; x < width; x += 4 {
					b := cdat[x/4]
					for x2 := 0; x2 < 4 && x+x2 < width; x2++ {
						gray.SetGray(x+x2, y, color.Gray{(b >> 6) * 0x55})
						b <<= 2
					}
				}
			}
		case cbG4:
			if d.useTransparent {
				ty := d.transparent[1]
				for x := 0; x < width; x += 2 {
					b := cdat[x/2]
					for x2 := 0; x2 < 2 && x+x2 < width; x2++ {
						ycol := (b >> 4) * 0x11
						acol := uint8(0xff)
						if ycol == ty {
							acol = 0x00
						}
						nrgba.SetNRGBA(x+x2, y, color.NRGBA{ycol, ycol, ycol, acol})
						b <<= 4
					}
				}
			} else {
				for x := 0; x < width; x += 2 {
					b := cdat[x/2]
					for x2 := 0; x2 < 2 && x+x2 < width; x2++ {
						gray.SetGray(x+x2, y, color.Gray{(b >> 4) * 0x11})
						b <<= 4
					}
				}
			}
		case cbG8:
			if d.useTransparent {
				ty := d.transparent[1]
				for x := 0; x < width; x++ {
					ycol := cdat[x]
					acol := uint8(0xff)
					if ycol == ty {
						acol = 0x00
					}
					nrgba.SetNRGBA(x, y, color.NRGBA{ycol, ycol, ycol, acol})
				}
			} else {
				copy(gray.Pix[pixOffset:], cdat)
				pixOffset += gray.Stride
			}
		case cbGA8:
			for x := 0; x < width; x++ {
				ycol := cdat[2*x+0]
				nrgba.SetNRGBA(x, y, color.NRGBA{ycol, ycol, ycol, cdat[2*x+1]})
			}
		case cbTC8:
			if d.useTransparent {
				pix, i, j := nrgba.Pix, pixOffset, 0
				tr, tg, tb := d.transparent[1], d.transparent[3], d.transparent[5]
				for x := 0; x < width; x++ {
					r := cdat[j+0]
					g := cdat[j+1]
					b := cdat[j+2]
					a := uint8(0xff)
					if r == tr && g == tg && b == tb {
						a = 0x00
					}
					pix[i+0] = r
					pix[i+1] = g
					pix[i+2] = b
					pix[i+3] = a
					i += 4
					j += 3
				}
				pixOffset += nrgba.Stride
			} else {
				pix, i, j := rgba.Pix, pixOffset, 0
				for x := 0; x < width; x++ {
					pix[i+0] = cdat[j+0]
					pix[i+1] = cdat[j+1]
					pix[i+2] = cdat[j+2]
					pix[i+3] = 0xff
					i += 4
					j += 3
				}
				pixOffset += rgba.Stride
			}
		case cbP1:
			for x := 0; x < width; x += 8 {
				b := cdat[x/8]
				for x2 := 0; x2 < 8 && x+x2 < width; x2++ {
					idx := b >> 7
					if len(paletted.Palette) <= int(idx) {
						paletted.Palette = paletted.Palette[:int(idx)+1]
					}
					paletted.SetColorIndex(x+x2, y, idx)
					b <<= 1
				}
			}
		case cbP2:
			for x := 0; x < width; x += 4 {
				b := cdat[x/4]
				for x2 := 0; x2 < 4 && x+x2 < width; x2++ {
					idx := b >> 6
					if len(paletted.Palette) <= int(idx) {
						paletted.Palette = paletted.Palette[:int(idx)+1]
					}
					paletted.SetColorIndex(x+x2, y, idx)
					b <<= 2
				}
			}
		case cbP4:
			for x := 0; x < width; x += 2 {
				b := cdat[x/2]
				for x2 := 0; x2 < 2 && x+x2 < width; x2++ {
					idx := b >> 4
					if len(paletted.Palette) <= int(idx) {
						paletted.Palette = paletted.Palette[:int(idx)+1]
					}
					paletted.SetColorIndex(x+x2, y, idx)
					b <<= 4
				}
			}
		case cbP8:
			if len(paletted.Palette) != 256 {
				for x := 0; x < width; x++ {
					if len(paletted.Palette) <= int(cdat[x]) {
						paletted.Palette = paletted.Palette[:int(cdat[x])+1]
					}
				}
			}
			copy(paletted.Pix[pixOffset:], cdat)
			pixOffset += paletted.Stride
		case cbTCA8:
			copy(nrgba.Pix[pixOffset:], cdat)
			pixOffset += nrgba.Stride
		case cbG16:
			if d.useTransparent {
				ty := uint16(d.transparent[0])<<8 | uint16(d.transparent[1])
				for x := 0; x < width; x++ {
					ycol := uint16(cdat[2*x+0])<<8 | uint16(cdat[2*x+1])
					acol := uint16(0xffff)
					if ycol == ty {
						acol = 0x0000
					}
					nrgba64.SetNRGBA64(x, y, color.NRGBA64{ycol, ycol, ycol, acol})
				}
			} else {
				for x := 0; x < width; x++ {
					ycol := uint16(cdat[2*x+0])<<8 | uint16(cdat[2*x+1])
					gray16.SetGray16(x, y, color.Gray16{ycol})
				}
			}
		case cbGA16:
			for x := 0; x < width; x++ {
				ycol := uint16(cdat[4*x+0])<<8 | uint16(cdat[4*x+1])
				acol := uint16(cdat[4*x+2])<<8 | uint16(cdat[4*x+3])
				nrgba64.SetNRGBA64(x, y, color.NRGBA64{ycol, ycol, ycol, acol})
			}
		case cbTC16:
			if d.useTransparent {
				tr := uint16(d.transparent[0])<<8 | uint16(d.transparent[1])
				tg := uint16(d.transparent[2])<<8 | uint16(d.transparent[3])
				tb := uint16(d.transparent[4])<<8 | uint16(d.transparent[5])
				for x := 0; x < width; x++ {
					rcol := uint16(cdat[6*x+0])<<8 | uint16(cdat[6*x+1])
					gcol := uint16(cdat[6*x+2])<<8 | uint16(cdat[6*x+3])
					bcol := uint16(cdat[6*x+4])<<8 | uint16(cdat[6*x+5])
					acol := uint16(0xffff)
					if rcol == tr && gcol == tg && bcol == tb {
						acol = 0x0000
					}
					nrgba64.SetNRGBA64(x, y, color.NRGBA64{rcol, gcol, bcol, acol})
				}
			} else {
				for x := 0; x < width; x++ {
					rcol := uint16(cdat[6*x+0])<<8 | uint16(cdat[6*x+1])
					gcol := uint16(cdat[6*x+2])<<8 | uint16(cdat[6*x+3])
					bcol := uint16(cdat[6*x+4])<<8 | uint16(cdat[6*x+5])
					rgba64.SetRGBA64(x, y, color.RGBA64{rcol, gcol, bcol, 0xffff})
				}
			}
		case cbTCA16:
			for x := 0; x < width; x++ {
				rcol := uint16(cdat[8*x+0])<<8 | uint16(cdat[8*x+1])
				gcol := uint16(cdat[8*x+2])<<8 | uint16(cdat[8*x+3])
				bcol := uint16(cdat[8*x+4])<<8 | uint16(cdat[8*x+5])
				acol := uint16(cdat[8*x+6])<<8 | uint16(cdat[8*x+7])
				nrgba64.SetNRGBA64(x, y, color.NRGBA64{rcol, gcol, bcol, acol})
			}
		}

		// The current row for y is the previous row for y+1.
		pr, cr = cr, pr
	}

	return img, nil
}

// mergePassInto merges a single pass into a full sized image.
func (d *decoder) mergePassInto(dst image.Image, src image.Image, pass int) {
	p := interlacing[pass]
	var (
		srcPix        []uint8
		dstPix        []uint8
		stride        int
		rect          image.Rectangle
		bytesPerPixel int
	)
	switch target := dst.(type) {
	case *image.Alpha:
		srcPix = src.(*image.Alpha).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 1
	case *image.Alpha16:
		srcPix = src.(*image.Alpha16).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 2
	case *image.Gray:
		srcPix = src.(*image.Gray).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 1
	case *image.Gray16:
		srcPix = src.(*image.Gray16).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 2
	case *image.NRGBA:
		srcPix = src.(*image.NRGBA).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 4
	case *image.NRGBA64:
		srcPix = src.(*image.NRGBA64).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 8
	case *image.Paletted:
		source := src.(*image.Paletted)
		srcPix = source.Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 1
		if len(target.Palette) < len(source.Palette) {
			// readImagePass can return a paletted image whose implicit palette
			// length (one more than the maximum Pix value) is larger than the
			// explicit palette length (what's in the PLTE chunk). Make the
			// same adjustment here.
			target.Palette = source.Palette
		}
	case *image.RGBA:
		srcPix = src.(*image.RGBA).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 4
	case *image.RGBA64:
		srcPix = src.(*image.RGBA64).Pix
		dstPix, stride, rect = target.Pix, target.Stride, target.Rect
		bytesPerPixel = 8
	}
	s, bounds := 0, src.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		dBase := (y*p.yFactor+p.yOffset-rect.Min.Y)*stride + (p.xOffset-rect.Min.X)*bytesPerPixel
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			d := dBase + x*p.xFactor*bytesPerPixel
			copy(dstPix[d:], srcPix[s:s+bytesPerPixel])
			s += bytesPerPixel
		}
	}
}

func (d *decoder) parseacTL(length uint32) (err error) {
	if length != 8 {
		return FormatError("bad acTL length")
	}
	if _, err := io.ReadFull(d.r, d.tmp[:8]); err != nil {
		return err
	}

	d.numFrames = binary.BigEndian.Uint32(d.tmp[:4])
	d.a.LoopCount = uint(binary.BigEndian.Uint32(d.tmp[4:8]))

	d.crc.Write(d.tmp[:8])
	return d.verifyChecksum()
}

// checkSequence verifies an APNG sequence number when the decoder is running
// in strict mode. Sequence numbers must start at zero and increase by one for
// every fcTL and fdAT chunk.
func (d *decoder) checkSequence(seq uint32) error {
	if !d.strict {
		return nil
	}
	if seq != d.seqExpected {
		return FormatError(fmt.Sprintf("unexpected sequence number %d, expected %d", seq, d.seqExpected))
	}
	d.seqExpected++
	return nil
}

// validateFrame checks that frame metadata read from fcTL is usable and stays
// within the image canvas. DecodeAll and Decode do not enforce these checks,
// matching their historical behavior.
func (d *decoder) validateFrame(f *Frame) error {
	if !d.strict {
		return nil
	}
	if f.width <= 0 || f.height <= 0 {
		return FormatError("non-positive frame dimension")
	}
	if f.XOffset < 0 || f.YOffset < 0 ||
		int64(f.XOffset)+int64(f.width) > int64(d.width) ||
		int64(f.YOffset)+int64(f.height) > int64(d.height) {
		return FormatError("frame outside image bounds")
	}
	return nil
}

func (d *decoder) parsefcTL(length uint32) (err error) {
	if length != 26 {
		return FormatError("bad fcTL length")
	}
	if _, err := io.ReadFull(d.r, d.tmp[:26]); err != nil {
		return err
	}

	// The frame control chunk is the only place frame metadata is stored;
	// the image data chunks that follow carry pixels only.
	if err := d.checkSequence(binary.BigEndian.Uint32(d.tmp[0:4])); err != nil {
		return err
	}

	f := d.targetFrame()
	f.IsDefault = false
	f.Sequence = binary.BigEndian.Uint32(d.tmp[0:4])
	f.width = int(int32(binary.BigEndian.Uint32(d.tmp[4:8])))
	f.height = int(int32(binary.BigEndian.Uint32(d.tmp[8:12])))
	f.XOffset = int(binary.BigEndian.Uint32(d.tmp[12:16]))
	f.YOffset = int(binary.BigEndian.Uint32(d.tmp[16:20]))
	f.DelayNumerator = binary.BigEndian.Uint16(d.tmp[20:22])
	f.DelayDenominator = binary.BigEndian.Uint16(d.tmp[22:24])
	f.DisposeOp = d.tmp[24]
	f.BlendOp = d.tmp[25]
	if err := d.validateFrame(f); err != nil {
		return err
	}

	d.crc.Write(d.tmp[:26])
	return d.verifyChecksum()
}

func (d *decoder) parsefdAT(length uint32) (err error) {
	if d.strict && length < 4 {
		return FormatError("bad fdAT length")
	}
	if _, err := io.ReadFull(d.r, d.tmp[:4]); err != nil {
		return err
	}
	if err := d.checkSequence(binary.BigEndian.Uint32(d.tmp[:4])); err != nil {
		return err
	}
	d.crc.Write(d.tmp[:4])
	d.idatLength = length - 4
	f := d.targetFrame()
	f.Image, err = d.decode()
	if err != nil {
		return err
	}
	return d.verifyChecksum()
}

func (d *decoder) parseIDAT(length uint32) (err error) {
	d.idatLength = length
	f := d.targetFrame()
	f.Image, err = d.decode()
	if err != nil {
		return err
	}
	return d.verifyChecksum()
}

func (d *decoder) parseIEND(length uint32) error {
	if length != 0 {
		return FormatError("bad IEND length")
	}
	return d.verifyChecksum()
}

// parseChunk reads and processes one chunk.
func (d *decoder) parseChunk() error {
	_, err := d.parseChunkType()
	return err
}

// parseChunkType reads and processes one chunk and returns its type. It is
// used by FrameReader to drive the stream one chunk at a time.
func (d *decoder) parseChunkType() (string, error) {
	h, err := d.readChunkHeader()
	if err != nil {
		return "", err
	}
	return d.processChunk(h)
}

// processChunk processes a chunk whose header has already been read by
// readChunkHeader, which also initialized the CRC with the chunk type.
func (d *decoder) processChunk(h chunkHeader) (string, error) {
	length, typ := h.length, h.typ

	// Read the chunk data.
	switch typ {
	case "IHDR":
		if d.stage != dsStart {
			return typ, chunkOrderError
		}
		d.stage = dsSeenIHDR
		return typ, d.parseIHDR(length)
	case "PLTE":
		if d.stage != dsSeenIHDR {
			return typ, chunkOrderError
		}
		d.stage = dsSeenPLTE
		return typ, d.parsePLTE(length)
	case "tRNS":
		if cbPaletted(d.cb) {
			if d.stage != dsSeenPLTE {
				return typ, chunkOrderError
			}
		} else if d.stage != dsSeenIHDR {
			return typ, chunkOrderError
		}
		d.stage = dsSeentRNS
		return typ, d.parsetRNS(length)
	case "acTL":
		if d.stage >= dsSeenIDAT {
			return typ, chunkOrderError
		}
		if d.strict && d.seenACTL {
			return typ, FormatError("duplicate acTL chunk")
		}
		if err := d.parseacTL(length); err != nil {
			return typ, err
		}
		d.seenACTL = true
		return typ, nil
	case "fcTL":
		if d.strict && !d.seenACTL {
			return typ, FormatError("fcTL without acTL")
		}
		if d.stage >= dsSeenIDAT {
			d.frameIndex = d.frameIndex + 1
		}
		if d.stream {
			// A frame control chunk always begins a new frame. If the previous
			// frame control chunk never received image data, the input is
			// truncated.
			if d.pendingFcTL {
				return typ, FormatError("truncated frame: fcTL without image data")
			}
			d.cur = nil
		}
		if err := d.parsefcTL(length); err != nil {
			return typ, err
		}
		if d.stream {
			d.fcTLCount++
			d.pendingFcTL = true
		}
		return typ, nil
	case "fdAT":
		if d.stage < dsSeenIDAT {
			return typ, chunkOrderError
		}
		if d.stream && !d.pendingFcTL {
			return typ, FormatError("fdAT without preceding fcTL")
		}
		d.stage = dsSeenfdAT
		err := d.parsefdAT(length)
		if err == nil && d.stream {
			d.pendingFcTL = false
		}
		return typ, err
	case "IDAT":
		if d.stage < dsSeenIHDR || d.stage > dsSeenIDAT || (d.stage == dsSeenIHDR && cbPaletted(d.cb)) {
			return typ, chunkOrderError
		} else if d.stage == dsSeenIDAT {
			// Ignore trailing zero-length or garbage IDAT chunks.
			//
			// This does not affect valid PNG images that contain multiple IDAT
			// chunks, since the first call to parseIDAT below will consume all
			// consecutive IDAT chunks required for decoding the image.
			return typ, d.skipChunk(length)
		}
		if d.stream && d.cur == nil {
			// The image data is not preceded by an fcTL chunk, so it is the
			// default image and does not form part of the animation.
			d.cur = &Frame{width: d.width, height: d.height, IsDefault: true}
		}
		d.stage = dsSeenIDAT
		err := d.parseIDAT(length)
		if err == nil && d.stream {
			d.pendingFcTL = false
		}
		return typ, err
	case "IEND":
		if d.stage < dsSeenIDAT {
			return typ, chunkOrderError
		}
		if d.stream {
			if d.pendingFcTL {
				return typ, FormatError("truncated frame: no image data before IEND")
			}
			if d.seenACTL && int(d.fcTLCount) != int(d.numFrames) {
				return typ, FormatError(fmt.Sprintf("expected %d animation frames, found %d", d.numFrames, d.fcTLCount))
			}
		}
		d.stage = dsSeenIEND
		return typ, d.parseIEND(length)
	}
	return typ, d.skipChunk(length)
}

func (d *decoder) verifyChecksum() error {
	if _, err := io.ReadFull(d.r, d.tmp[:4]); err != nil {
		return err
	}
	if binary.BigEndian.Uint32(d.tmp[:4]) != d.crc.Sum32() {
		return FormatError("invalid checksum")
	}
	return nil
}

func (d *decoder) checkHeader() error {
	_, err := io.ReadFull(d.r, d.tmp[:len(pngHeader)])
	if err != nil {
		return err
	}
	if string(d.tmp[:len(pngHeader)]) != pngHeader {
		return FormatError("not a PNG file")
	}
	return nil
}

// DecodeAll reads an APNG file from r and returns it as an APNG
// Type. If the first frame returns true for IsDefault(), that
// frame should not be part of the result.
// The type of Image returned depends on the PNG contents.
func DecodeAll(r io.Reader) (APNG, error) {
	d := &decoder{
		r:          r,
		crc:        crc32.NewIEEE(),
		frameIndex: 0,
		a:          APNG{Frames: make([]Frame, 1)},
	}
	d.a.Frames[0].IsDefault = true
	if err := d.checkHeader(); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return d.a, err
	}
	for d.stage != dsSeenIEND {
		if err := d.parseChunk(); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return d.a, err
		}
	}
	return d.a, nil
}

// Decode reads an APNG file from r and returns the default image.
func Decode(r io.Reader) (image.Image, error) {
	a, err := DecodeAll(r)
	if err != nil {
		return nil, err
	}
	return a.Frames[0].Image, nil
}

// DecodeConfig returns the color model and dimensions of a PNG image without
// decoding the entire image.
func DecodeConfig(r io.Reader) (image.Config, error) {
	d := &decoder{
		r:   r,
		crc: crc32.NewIEEE(),
		a:   APNG{Frames: make([]Frame, 1)},
	}
	if err := d.checkHeader(); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return image.Config{}, err
	}
	for {
		if err := d.parseChunk(); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return image.Config{}, err
		}
		paletted := cbPaletted(d.cb)
		if d.stage == dsSeenIHDR && !paletted {
			break
		}
		if d.stage == dsSeenPLTE && paletted {
			break
		}
	}
	var cm color.Model
	switch d.cb {
	case cbG1, cbG2, cbG4, cbG8:
		cm = color.GrayModel
	case cbGA8:
		cm = color.NRGBAModel
	case cbTC8:
		cm = color.RGBAModel
	case cbP1, cbP2, cbP4, cbP8:
		cm = d.palette
	case cbTCA8:
		cm = color.NRGBAModel
	case cbG16:
		cm = color.Gray16Model
	case cbGA16:
		cm = color.NRGBA64Model
	case cbTC16:
		cm = color.RGBA64Model
	case cbTCA16:
		cm = color.NRGBA64Model
	}
	return image.Config{
		ColorModel: cm,
		Width:      d.a.Frames[0].width,
		Height:     d.a.Frames[0].height,
	}, nil
}

func init() {
	image.RegisterFormat("apng", pngHeader, Decode, DecodeConfig)
}
