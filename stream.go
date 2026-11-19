// Copyright 2018 kts of kettek / Ketchetwahmeegwun Tecumseh Southall. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package apng

import (
	"errors"
	"hash/crc32"
	"io"
)

// DefaultMaxFrameBytes is the default per-frame memory limit used by
// FrameReader. It bounds the decoded pixel buffer of one frame, not the
// encoded size of the input.
const DefaultMaxFrameBytes int64 = 1 << 30 // 1 GiB

var (
	// ErrFrameTooLarge reports that decoding a frame would require more than
	// FrameReader.MaxFrameBytes of pixel memory. It is returned by Next before
	// any pixel buffer is allocated. Use errors.Is to test for it.
	ErrFrameTooLarge = errors.New("apng: decoded frame exceeds the size limit")

	// ErrFrameReaderClosed reports that Next was called after Close. Use
	// errors.Is to test for it.
	ErrFrameReaderClosed = errors.New("apng: frame reader is closed")
)

// FrameReader incrementally decodes the frames of an APNG from an io.Reader.
//
// Unlike DecodeAll, it does not retain the frames it has already returned. A
// frame's pixel buffer becomes eligible for garbage collection as soon as the
// caller drops it and asks for the next frame. Only the frame being decoded is
// kept in memory, bounded by MaxFrameBytes.
//
// NewFrameReader reads up to and including the acTL chunk, so LoopCount,
// NumFrames and IsAnimation are available before the first call to Next.
//
// The first frame returned for an APNG whose default image is not part of the
// animation has IsDefault set to true, exactly as DecodeAll reports it. A
// non-APNG PNG yields a single frame with IsDefault set to true followed by
// io.EOF; IsAnimation reports false and LoopCount is 0.
//
// Frame.Sequence holds the APNG sequence number of the frame's fcTL chunk.
// The default image has no fcTL chunk, so its Sequence is zero; use IsDefault
// to distinguish it from the first animation frame.
//
// A frame's delay is reported without modification: GetDelay returns zero for
// a zero delay numerator, and treats a zero denominator as 100 as the APNG
// specification requires. Zero delays are not an error.
//
// Next reports io.EOF after the last frame. A truncated stream, a chunk out of
// order, an invalid checksum or sequence number, or a frame larger than
// MaxFrameBytes all produce a non-nil error, and no partial frame is returned.
// Stopping early is done simply by not calling Next again; Close releases the
// most recently returned frame and makes any further Next call fail with
// ErrFrameReaderClosed.
type FrameReader struct {
	// MaxFrameBytes limits the decoded pixel buffer of a single frame. Zero
	// selects DefaultMaxFrameBytes; a negative value removes the limit.
	// The limit is enforced before allocating the pixel buffer. It may be
	// changed between calls to Next.
	MaxFrameBytes int64

	d         *decoder
	pending   *chunkHeader
	err       error
	done      bool
	closed    bool
	delivered bool
}

// NewFrameReader creates a FrameReader that reads an APNG (or a plain PNG)
// from r. It parses the PNG header and any acTL chunk needed to report the
// loop count, but does not decode a frame. It returns a FormatError if r does
// not begin with a PNG signature and io.ErrUnexpectedEOF if the stream ends
// before the first frame's image data.
func NewFrameReader(r io.Reader) (*FrameReader, error) {
	d := &decoder{
		r:             r,
		crc:           crc32.NewIEEE(),
		a:             APNG{Frames: make([]Frame, 1)},
		stream:        true,
		strict:        true,
		maxFrameBytes: DefaultMaxFrameBytes,
	}
	fr := &FrameReader{d: d, MaxFrameBytes: DefaultMaxFrameBytes}
	if err := d.checkHeader(); err != nil {
		return nil, normalizeStreamError(err)
	}
	for {
		h, err := d.readChunkHeader()
		if err != nil {
			return nil, normalizeStreamError(err)
		}
		switch h.typ {
		case "IDAT", "fdAT":
			// Leave the image data for Next, so that no pixels are decoded
			// until the caller asks for them.
			fr.pending = &h
			return fr, nil
		case "IEND":
			if _, err := d.processChunk(h); err != nil {
				return nil, normalizeStreamError(err)
			}
			fr.done = true
			return fr, nil
		}
		if _, err := d.processChunk(h); err != nil {
			return nil, normalizeStreamError(err)
		}
		if d.seenACTL {
			return fr, nil
		}
		if d.stage == dsSeenIEND {
			fr.done = true
			return fr, nil
		}
	}
}

// IsAnimation reports whether the input contained an acTL chunk, which is what
// makes a PNG an APNG.
func (fr *FrameReader) IsAnimation() bool {
	return fr.d.seenACTL
}

// NumFrames returns the number of animation frames declared by the acTL chunk,
// or 0 if the input is not an APNG.
func (fr *FrameReader) NumFrames() uint32 {
	return fr.d.numFrames
}

// LoopCount returns the number of times the animation restarts during display.
// A value of 0 means to loop forever. It is 0 for a non-APNG PNG.
func (fr *FrameReader) LoopCount() uint {
	return fr.d.a.LoopCount
}

// frameLimit resolves MaxFrameBytes into a decoder limit, where 0 means no
// limit.
func (fr *FrameReader) frameLimit() int64 {
	switch {
	case fr.MaxFrameBytes == 0:
		return DefaultMaxFrameBytes
	case fr.MaxFrameBytes < 0:
		return 0
	default:
		return fr.MaxFrameBytes
	}
}

// Next decodes and returns the next frame.
//
// The returned Frame, including its Image, is owned by the caller. Calling
// Next again releases the decoder's reference to it, so the caller only needs
// to keep the frames it actually uses. Next returns io.EOF once the last frame
// has been returned. After any other error it returns the same error on every
// subsequent call.
func (fr *FrameReader) Next() (*Frame, error) {
	if fr.closed {
		return nil, ErrFrameReaderClosed
	}
	if fr.err != nil {
		return nil, fr.err
	}
	d := fr.d
	d.maxFrameBytes = fr.frameLimit()
	// Release the frame handed to the caller by the previous call. A frame
	// that has been described by an fcTL chunk but not yet decoded is kept.
	if fr.delivered {
		d.cur = nil
		fr.delivered = false
	}

	for {
		if fr.done || d.stage == dsSeenIEND {
			fr.err = io.EOF
			return nil, io.EOF
		}
		var err error
		if fr.pending != nil {
			h := *fr.pending
			fr.pending = nil
			_, err = d.processChunk(h)
		} else {
			_, err = d.parseChunkType()
		}
		if err != nil {
			fr.err = normalizeStreamError(err)
			return nil, fr.err
		}
		if d.cur != nil && d.cur.Image != nil {
			fr.delivered = true
			return d.cur, nil
		}
		if d.stage == dsSeenIEND {
			fr.err = io.EOF
			return nil, io.EOF
		}
	}
}

// Close releases the most recently returned frame and prevents further reads.
// It does not close the underlying io.Reader, which remains the caller's
// responsibility.
func (fr *FrameReader) Close() error {
	fr.d.cur = nil
	fr.d.pendingFcTL = false
	fr.closed = true
	return nil
}

// normalizeStreamError maps a premature end of input to io.ErrUnexpectedEOF so
// that io.EOF is reserved for the normal end of the frame sequence.
func normalizeStreamError(err error) error {
	if err == io.EOF {
		return io.ErrUnexpectedEOF
	}
	return err
}
