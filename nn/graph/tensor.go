package graph

import (
	"errors"
	"fmt"

	"github.com/stellviaproject/go-ia/float16"
)

var (
	ErrInvalidShape    error = errors.New("invalid shape")
	ErrDimMismatch     error = errors.New("dimension mismatch")
	ErrIndexOutOfRange error = errors.New("index out of range")
	ErrInvalidData     error = errors.New("tensor invalid data")
	ErrTypeMismatch    error = errors.New("type mismatch")
)

// Tensor shape representation
type Shape []int

func NewShape(lens ...int) Shape {
	return Shape(lens)
}

// Dimension of shape
func (sh Shape) Dim() int {
	return len(sh)
}

// Stride of dimension
func (sh Shape) StrideOf(dim int) int {
	stride := 1
	for i, c := 1, len(sh); i < c; i++ {
		stride *= sh[i]
	}
	return stride
}

// Strides of every dimension
func (sh Shape) Strides() []int {
	strides := make([]int, len(sh))
	strides[0] = 1
	for i, c := 1, len(strides); i < c; i++ {
		strides[i] = strides[i-1] * sh[i-1]
	}
	return strides
}

func (sh Shape) Key() []int {
	index := make([]int, sh.Dim())
	for i := 0; i < len(index); i++ {
		index[i] = -1
	}
	return index
}

// Len of shape
func (sh Shape) Len() int {
	ln := 1
	for i := 0; i < len(sh); i++ {
		ln *= sh[i]
	}
	return ln
}

func (sh Shape) Equal(other Shape) bool {
	if sh.Len() != other.Len() {
		return false
	}
	for i := 0; i < sh.Dim(); i++ {
		if sh[i] != other[i] {
			return false
		}
	}
	return true
}

type Type int

const (
	Float16 Type = iota + 1
	Float32
	Float64
)

// Represents a tensor
type Tensor struct {
	rank    int
	data    any
	shape   Shape
	typ     Type
	strides []int
}

// Create a tensor with given data, type and shape
//
// data may be []float16.Float16, []float32, []float64 or nil. If data is nil a slice of given type will be created
//
// type may be Float16, Float32, Float64
//
// shape of tensor, it says the number of elements of tensor and panics if len(data) is not equal to shape.Len()
func NewTensor(data any, typ Type, shape Shape) *Tensor {
	// validate type values
	if typ != Float16 && typ != Float32 && typ != Float64 {
		panic(ErrTypeMismatch)
	}
	// validate length of shape dimensions
	for i := range shape {
		if shape[i] <= 0 {
			panic(ErrInvalidShape)
		}
	}
	// create a slice if data is nil
	if data == nil {
		if typ == Float16 {
			data = make([]float16.Float16, shape.Len())
		} else if typ == Float64 {
			data = make([]float32, shape.Len())
		} else {
			data = make([]float64, shape.Len())
		}
	}
	// convert given slice to given type
	switch v := data.(type) {
	case []float16.Float16:
		// validate slice len with shape len
		if len(v) != shape.Len() {
			panic(ErrInvalidShape)
		}
		if typ == Float32 {
			// convert float16 to float32
			aux := make([]float32, len(v))
			for i, c := 0, len(aux); i < c; i++ {
				aux[i] = v[i].ToF32()
			}
			data = aux
		} else if typ == Float64 {
			// convert float16 to float64
			aux := make([]float64, len(v))
			for i, c := 0, len(aux); i < c; i++ {
				aux[i] = v[i].ToF64()
			}
			data = aux
		}
	case []float64:
		// validate slice len with shape len
		if len(v) != shape.Len() {
			panic(ErrInvalidShape)
		}
		if typ == Float16 {
			// convert float64 to float16
			aux := make([]float16.Float16, len(v))
			for i, c := 0, len(aux); i < c; i++ {
				aux[i] = float16.FF64(v[i])
			}
			data = aux
		} else if typ == Float32 {
			// convert float64 to float32
			aux := make([]float32, len(v))
			for i := range v {
				aux[i] = float32(v[i])
			}
			data = aux
		}
	case []float32:
		// validate slice len with shape len
		if len(v) != shape.Len() {
			panic(ErrInvalidShape)
		}
		if typ == Float16 {
			// convert float32 to float16
			aux := make([]float16.Float16, len(v))
			for i, c := 0, len(aux); i < c; i++ {
				aux[i] = float16.FF32(v[i])
			}
			data = aux
		} else if typ == Float64 {
			// convert float32 to float64
			aux := make([]float64, len(v))
			for i := range v {
				aux[i] = float64(v[i])
			}
			data = aux
		}
	default:
		// slice data not valid for a tensor
		panic(ErrInvalidData)
	}
	// init tensor
	tensor := new(Tensor)
	tensor.data = data
	tensor.shape = shape
	tensor.strides = shape.Strides()
	tensor.rank = len(shape)
	tensor.typ = typ
	return tensor
}

// Rank of tensor
func (ts *Tensor) Rank() int {
	return ts.rank
}

// Get tensor float16 slice
//
// panics if type is not Float16
func (ts *Tensor) F16Slice() []float16.Float16 {
	if ts.typ != Float16 {
		panic(ErrTypeMismatch)
	}
	return ts.data.([]float16.Float16)
}

// Get tensor float32 slice
//
// panics if type is not float32
func (ts *Tensor) F32Slice() []float32 {
	if ts.typ != Float32 {
		panic(ErrTypeMismatch)
	}
	return ts.data.([]float32)
}

// Get tensor float64 slice
//
// panics if type is not float64
func (ts *Tensor) F64Slice() []float64 {
	if ts.typ != Float64 {
		panic(ErrTypeMismatch)
	}
	return ts.data.([]float64)
}

// Get tensor shape
func (ts *Tensor) Shape() Shape {
	sh := make(Shape, len(ts.shape))
	copy(sh, ts.shape)
	return sh
}

// Reshape tensor
//
// panics if the length of new shape doesn't equal to length of slice of tensor
// or if shape has not valid length of dimensions
func (ts *Tensor) Reshape(shape Shape) {
	// validate dimension lengths
	for i := range shape {
		if shape[i] <= 0 {
			panic(ErrInvalidShape)
		}
	}
	// validate shape len with length of slice of tensor
	switch ts.typ {
	case Float16:
		// validate as float16 slice
		if len(ts.data.([]float16.Float16)) != shape.Len() {
			panic(ErrInvalidShape)
		}
	case Float32:
		// validate as float32 slice
		if len(ts.data.([]float32)) != shape.Len() {
			panic(ErrInvalidShape)
		}
	case Float64:
		// validate as float64 slice
		if len(ts.data.([]float64)) != shape.Len() {
			panic(ErrInvalidShape)
		}
	}
	// set new shape
	ts.shape = shape
	ts.strides = shape.Strides()
	ts.rank = len(shape)
}

// get multidimensional index for some offset
func (ts *Tensor) index(offset int) []int {
	// index for tensor shape
	index := make([]int, len(ts.shape))
	// get index value using strides and offsets in strides
	for i := len(ts.strides) - 1; i >= 0; i-- {
		index[i] = offset / ts.strides[i]
		offset %= ts.strides[i]
	}
	return index
}

// validate offset and panics if offset is lesser than zero o greater than length of shape
func (ts *Tensor) testOffset(offset int) {
	if offset < 0 || offset > ts.shape.Len() {
		panic(ErrIndexOutOfRange)
	}
}

// get offset for some index
func (ts *Tensor) offset(index []int) int {
	offset := 0 //offset of given index
	for i := range index {
		offset += ts.strides[i] * index[i] //offset is the product of index by stride in its dimension
	}
	return offset
}

// validate index and panics if length of index is not equal to length of shape or if elements of index are out of dimension range in shape
func (ts *Tensor) testIndex(index []int) {
	if len(index) != len(ts.shape) {
		panic(ErrDimMismatch)
	}
	for i, ln := 0, len(index); i < ln; i++ {
		if index[i] < 0 || index[i] >= ts.shape[i] {
			panic(ErrIndexOutOfRange)
		}
	}
}

// get element by index and panics if index is not in range
func (ts *Tensor) Get(index []int) any {
	ts.testIndex(index)  // validate index
	return ts.get(index) // get element by index
}

// get element by index without validation, used for more speed in internal operations
func (ts *Tensor) get(index []int) any {
	offset := ts.offset(index) //get offset
	// get element by type
	switch ts.typ {
	case Float16:
		return ts.data.([]float16.Float16)[offset] // get float16 element
	case Float32:
		return ts.data.([]float32)[offset] // get float32 element
	case Float64:
		return ts.data.([]float64)[offset] // get float64 element
	default:
		panic(ErrInvalidData)
	}
}

// Get element by offset and panics if offset is not in range
func (ts *Tensor) GetAt(offset int) any {
	ts.testOffset(offset) // validate offset
	return ts.at(offset)  // get element by offset
}

// get element at offset
func (ts *Tensor) at(offset int) any {
	// get element by offset with corresponding types
	switch ts.typ {
	case Float16:
		return ts.data.([]float16.Float16)[offset] // get float16 element
	case Float32:
		return ts.data.([]float32)[offset] // get float32 element
	case Float64:
		return ts.data.([]float64)[offset] // get float64 element
	default:
		panic(ErrInvalidData)
	}
}

// Set element at offset and panics if offset is not in range, or if value is not a valid type
func (ts *Tensor) SetAt(offset int, value any) {
	ts.testOffset(offset)   // validate offset
	ts.setAt(offset, value) // set value at offset
}

// set element at offset
func (ts *Tensor) setAt(offset int, value any) {
	// set element at offset by type
	switch ts.typ {
	case Float16:
		v := ts.data.([]float16.Float16)
		// validate element type
		if in, ok := value.(float16.Float16); ok {
			v[offset] = in // set float16 at offset
		} else {
			panic(ErrTypeMismatch)
		}
	case Float32:
		v := ts.data.([]float32)
		// validate element type
		if in, ok := value.(float32); ok {
			v[offset] = in // set float32 at offset
		} else {
			panic(ErrTypeMismatch)
		}
	case Float64:
		v := ts.data.([]float64)
		// validate element type
		if in, ok := value.(float64); ok {
			v[offset] = in // set float64 at offset
		} else {
			panic(ErrTypeMismatch)
		}
	default:
		panic(ErrInvalidData)
	}
}

// validate index-key
func (ts *Tensor) testKey(index []int) {
	// validate index length
	if len(index) != len(ts.shape) {
		panic(ErrDimMismatch)
	}
	// validate index elements with shape dimensions
	for i, ln := 0, len(index); i < ln; i++ {
		// -1 is accepted for telling the dimension for iter
		if index[i] < -1 || index[i] >= ts.shape[i] {
			panic(ErrIndexOutOfRange)
		}
	}
}

// Get subtensor with a given index that select a dimension to copy
//
// the shape of the new tensor will be the shape of excluding index with values -1
//
// the values of the new tensor will be the values of the tensor in the dimensions with values -1
//
// Example:
// Tensor is with shape{4, 3}
// | 0 3 6 9  |
// | 1 4 7 10 |
// | 2 5 8 11 |
// If indexKey is [-1, 1] you will get a tensor with shape{4, 1}
func (ts *Tensor) GetSub(indexKey []int) *Tensor {
	ts.testKey(indexKey)
	shape := Shape{}
	for i := 0; i < len(indexKey); i++ {
		if indexKey[i] == -1 {
			// shape dimension of new tensor is the same dimension of the parent tensor
			shape = append(shape, ts.shape[i])
		} else {
			// shape dimension of new tensor of selected dimension is one
			shape = append(shape, 1)
		}
	}
	tensor := NewTensor(nil, ts.typ, shape) // create the new tensor
	srcIndex := make([]int, len(indexKey))
	// interate in every element of new tensor
	for offset, length := 0, shape.Len(); offset < length; offset++ {
		dstIndex := tensor.index(offset) // get index for offset in new tensor
		copy(srcIndex, dstIndex)         //copy dst index to src index
		// set index key
		for i := 0; i < len(indexKey); i++ {
			if indexKey[i] != -1 {
				srcIndex[i] = indexKey[i]
			}
		}
		// set element at indexKey to destination index
		tensor.set(dstIndex, ts.get(srcIndex))
	}
	return tensor
}

// Get a float16 element at index location in tensor
//
// panics if index is out of range or element doesn't match with tensor type
func (ts *Tensor) GetF16At(index []int) float16.Float16 {
	if ts.typ != Float16 {
		panic(ErrTypeMismatch)
	}
	ts.testIndex(index)
	return ts.data.([]float16.Float16)[ts.offset(index)]
}

// Get a float32 element at index location in tensor
//
// panics if index is out of range or element doesn't match with tensor type
func (ts *Tensor) GetF32At(index []int) float32 {
	if ts.typ != Float32 {
		panic(ErrTypeMismatch)
	}
	ts.testIndex(index)
	return ts.data.([]float32)[ts.offset(index)]
}

// Get a float64 element at index location in tensor
//
// panics if index is out of range or element doesn't match with tensor type
func (ts *Tensor) GetF64At(index []int) float64 {
	if ts.typ != Float64 {
		panic(ErrTypeMismatch)
	}
	ts.testIndex(index)
	return ts.data.([]float64)[ts.offset(index)]
}

// Set value at index in tensor
//
// panics if index is out of range or type doesn't match
func (ts *Tensor) Set(index []int, value any) {
	ts.testIndex(index)
	ts.set(index, value)
}

// set value at index in tensor
func (ts *Tensor) set(index []int, value any) {
	// get index offset
	offset := ts.offset(index)
	// set value with corresponding type
	switch ts.typ {
	case Float16:
		// get float16 slice
		v := ts.data.([]float16.Float16)
		if in, ok := value.(float16.Float16); ok {
			v[offset] = in // set value as float16
		} else {
			panic(ErrTypeMismatch)
		}
	case Float32:
		// get float32 slice
		v := ts.data.([]float32)
		if in, ok := value.(float32); ok {
			v[offset] = in // set value as float32
		} else {
			panic(ErrTypeMismatch)
		}
	case Float64:
		// get float64 slice
		v := ts.data.([]float64)
		if in, ok := value.(float64); ok {
			v[offset] = in // set value as float64
		} else {
			panic(ErrTypeMismatch)
		}
	default:
		panic(ErrInvalidData)
	}
}

// Set float16 value at index
//
// panics if type doesn't match or index is out of range
func (ts *Tensor) SetF16(index []int, value float16.Float16) {
	if ts.typ != Float16 {
		panic(ErrTypeMismatch)
	}
	ts.testIndex(index)
	ts.data.([]float16.Float16)[ts.offset(index)] = value
}

// Set float32 value at index
//
// panics if type doesn't match or index is out of range
func (ts *Tensor) SetF32(index []int, value float32) {
	if ts.typ != Float32 {
		panic(ErrTypeMismatch)
	}
	ts.testIndex(index)
	ts.data.([]float32)[ts.offset(index)] = value
}

// Set float64 value at index
//
// panics if type doesn't match or index is out of range
func (ts *Tensor) SetF64(index []int, value float64) {
	if ts.typ != Float64 {
		panic(ErrTypeMismatch)
	}
	ts.testIndex(index)
	ts.data.([]float64)[ts.offset(index)] = value
}

// Compare the tensor with other tensor
//
// Return true or false if a tensor is equal to other
func (ts *Tensor) Equal(other *Tensor) bool {
	// validate the same shape
	if !ts.shape.Equal(other.shape) {
		return false
	}
	// validate the same type
	if ts.typ != other.typ {
		return false
	}
	// validate the same data
	switch ts.typ {
	case Float16:
		v, o := ts.data.([]float16.Float16), other.data.([]float16.Float16)
		for i, length := 0, ts.shape.Len(); i < length; i++ {
			if v[i] != o[i] {
				return false
			}
		}
	case Float32:
		v, o := ts.data.([]float32), other.data.([]float32)
		for i, length := 0, ts.shape.Len(); i < length; i++ {
			if v[i] != o[i] {
				return false
			}
		}
	case Float64:
		v, o := ts.data.([]float64), other.data.([]float64)
		for i, length := 0, ts.shape.Len(); i < length; i++ {
			if v[i] != o[i] {
				return false
			}
		}
	}
	return true
}

func (ts *Tensor) String() string {
	var toString func(d int, index []int) string
	toString = func(d int, index []int) string {
		s := "["
		if d == ts.shape.Dim()-1 {
			for i := 0; i < ts.shape[d]-1; i++ {
				index[d] = i
				s += fmt.Sprintf("%v, ", ts.get(index))
			}
			if ts.shape[d]-1 > 0 {
				index[d] = ts.shape[d] - 1
				s += fmt.Sprintf("%v", ts.get(index))
			}
		} else {
			for i := 0; i < ts.shape[d]-1; i++ {
				index[d] = i
				s += toString(d+1, index) + ", "
			}
			if ts.shape[d]-1 > 0 {
				index[d] = ts.shape[d] - 1
				s += toString(d+1, index)
			}
		}
		return s + "]"
	}
	return toString(0, make([]int, ts.shape.Dim()))
}
