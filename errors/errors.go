package errors

import "errors"

var ShapeHasNoDimErr = errors.New("shape has not dimension")
var ShapeDimNotValidErr = errors.New("shape dimension could not be zero or negative")
