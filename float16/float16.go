package float16

import (
	"errors"
	"math"
)

const (
	NaN          Float16 = 0x7FFF
	InfPos       Float16 = 0x7C00
	InfNeg       Float16 = 0xFC00
	expMask      Float16 = 0x7C00
	mantissaMask Float16 = 0x03FF
	signMask     Float16 = 0x8000
	signExpMask  Float16 = 0x0010
)

var ErrOverflow = errors.New("float16 overflow")

type Float16 uint16

func FF32(value float32) Float16 {
	bits := math.Float32bits(value)
	sign := bits >> 31
	exp := (bits >> 23) & 0xff
	frac := bits & 0x7fffff
	// NaN or Inf
	if exp == 0xff {
		if frac != 0 {
			return NaN
		}
		return Float16(sign<<15) | InfPos
	}
	//Denormalized
	if exp == 0 {
		frac >>= 13
		return Float16(sign<<15) | Float16(frac)
	}
	sexp := int32(exp) - 127 + 15
	if sexp >= 0x1f { //Too large, return Inf
		return Float16(sign<<15) | InfPos
	}
	if sexp <= 0 { //Too small, return 0
		frac >>= -sexp + 1
		return Float16(sign<<15) | Float16(frac>>13)
	}
	exp = uint32(sexp)
	return Float16(sign<<15) | Float16(exp<<10) | Float16(frac>>13)
}

func FF64(value float64) Float16 {
	bits := math.Float64bits(value)
	sign := bits >> 63
	exp := (bits >> 52) & 0x7ff
	frac := bits & 0xfffffffffffff
	// NaN or Inf
	if exp == 0x7ff {
		if frac != 0 {
			return NaN
		}
		return Float16(sign<<15) | InfPos
	}
	// Denormalized
	if exp == 0 {
		frac >>= 42
		return Float16(sign<<15) | Float16(frac)
	}
	sexp := int32(exp) - 1023 + 15
	if sexp >= 0x1f { // Too large, return Inf
		return Float16(sign<<15) | InfPos
	}
	if sexp <= 0 { // Too small, return 0
		frac >>= -sexp + 1
		return Float16(sign<<15) | Float16(frac>>13)
	}
	exp = uint64(sexp)
	return Float16(sign<<15) | Float16(exp<<10) | Float16(frac>>42)
}

func (f16 Float16) ToF32() float32 {
	sign := uint32(f16 >> 15 & 0x1)
	exp := uint32(f16 >> 10 & 0x1f)
	frac := uint32(f16 & 0x3ff)

	if exp == 0x1f { //NaN or Inf
		if frac != 0 {
			return math.Float32frombits(uint32(0xff<<23 | frac<<13 | 0x1))
		}
		return math.Float32frombits(uint32(sign<<31 | 0xff<<23))
	}
	if frac == 0 && exp == 0 {
		return 0.0
	}
	if exp == 0 { //Denormalized
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3ff
	}
	exp += 127 - 15
	return math.Float32frombits(uint32(sign<<31 | exp<<23 | frac<<13))
}

func (f16 Float16) ToF64() float64 {
	sign := uint64(f16 >> 15 & 0x1)
	exp := uint64(f16 >> 10 & 0x1f)
	frac := uint64(f16 & 0x3ff)

	if exp == 0x1f { // NaN or Inf
		if frac != 0 {
			return math.Float64frombits(uint64(0x7ff<<52 | frac<<42 | 0x1))
		}
		return math.Float64frombits(uint64(sign<<63 | 0x7ff<<52))
	}
	if frac == 0 && exp == 0 {
		return 0.0
	}
	if exp == 0 { // Denormalized
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3ff
	}
	exp += 1023 - 15
	return math.Float64frombits(uint64(sign<<63 | exp<<52 | frac<<42))
}
