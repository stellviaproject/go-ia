package knn

import (
	"math"
	"testing"
)

func TestWithPoint(t *testing.T) {
	p := WithPoint(1.0, 2.0, 3.0)
	if len(p) != 3 || p[0] != 1.0 || p[1] != 2.0 || p[2] != 3.0 {
		t.Errorf("WithPoint failed. Expected [1.0, 2.0, 3.0], but got %v", p)
	}
}

func TestEuclideanEval(t *testing.T) {
	p1 := WithPoint(0.0, 0.0)
	p2 := WithPoint(3.0, 4.0)
	euclidean := NewEuclideanDist()
	d := euclidean.Eval(p1, p2)
	if d != 5.0 {
		t.Errorf("EuclideanEval failed. Expected 5.0, but got %v", d)
	}
}

func TestManhattanEval(t *testing.T) {
	ma := NewManhattanDist()
	p1 := WithPoint(1.0, 2.0)
	p2 := WithPoint(3.0, 4.0)
	d := ma.Eval(p1, p2)
	expected := math.Abs(1.0-3.0) + math.Abs(2.0-4.0)
	if d != expected {
		t.Errorf("ManhattanEval failed. Expected %v, but got %v", expected, d)
	}
}

func TestMinkowskiEval(t *testing.T) {
	mi := NewMinkowskiDist(2.0)
	p1 := WithPoint(1.0, 2.0)
	p2 := WithPoint(3.0, 4.0)
	d := mi.Eval(p1, p2)
	expected := math.Sqrt(math.Pow(1.0-3.0, 2) + math.Pow(2.0-4.0, 2))
	if d != expected {
		t.Errorf("MinkowskiEval failed. Expected %v, but got %v", expected, d)
	}
}

func TestChebyshevEval(t *testing.T) {
	ch := NewChebyshevDist()
	p1 := WithPoint(1.0, 2.0)
	p2 := WithPoint(3.0, 4.0)
	d := ch.Eval(p1, p2)
	expected := math.Max(math.Abs(1.0-3.0), math.Abs(2.0-4.0))
	if d != expected {
		t.Errorf("ChebyshevEval failed. Expected %v, but got %v", expected, d)
	}
}

func TestHammingEval(t *testing.T) {
	ha := NewHammingDist()
	p1 := WithPoint(1.0, 0.0, 1.0)
	p2 := WithPoint(0.0, 1.0, 1.0)
	d := ha.Eval(p1, p2)
	expected := 2.0
	if d != expected {
		t.Errorf("HammingEval failed. Expected %v, but got %v", expected, d)
	}
}

func TestPearsonCorrelationEval(t *testing.T) {
	pe := NewPearsonCorrelationDist()
	p1 := WithPoint(1.0, 2.0, 3.0)
	p2 := WithPoint(2.0, 4.0, 6.0)
	d := pe.Eval(p1, p2)
	expected := 0.0
	if d != expected {
		t.Errorf("PearsonCorrelationEval failed. Expected %v, but got %v", expected, d)
	}
}

func TestBinarySelectorLabel(t *testing.T) {
	kset := []DataDist{
		newDataDist(1.0, &dataPoint{WithPoint(1.0), true}),
		newDataDist(2.0, &dataPoint{WithPoint(2.0), false}),
		newDataDist(3.0, &dataPoint{WithPoint(3.0), true}),
	}
	binary := NewBinarySelector()
	l := binary.Label(kset)
	if l != true {
		t.Errorf("BinarySelectorLabel failed. Expected true, but got %v", l)
	}
}

func TestMultiClassSelectorLabel(t *testing.T) {
	mu := NewMultiClassSelector()
	p1 := NewDataPoint("A", WithPoint(1.0, 2.0))
	p2 := NewDataPoint("B", WithPoint(2.0, 3.0))
	p3 := NewDataPoint("A", WithPoint(3.0, 4.0))
	p4 := NewDataPoint("A", WithPoint(4.0, 5.0))
	d1 := newDataDist(0.5, p1)
	d2 := newDataDist(0.3, p2)
	d3 := newDataDist(0.2, p3)
	d4 := newDataDist(0.7, p4)
	kset := []DataDist{d1, d2, d3, d4}
	label := mu.Label(kset)
	expected := "A"
	if label != expected {
		t.Errorf("MultiClassSelectorLabel failed. Expected %v, but got %v", expected, label)
	}
}

func TestRegressionSelectorLabel(t *testing.T) {
	re := NewRegressionSelector()
	p1 := NewDataPoint(3.0, WithPoint(1.0, 2.0))
	p2 := NewDataPoint(4.0, WithPoint(2.0, 3.0))
	p3 := NewDataPoint(5.0, WithPoint(3.0, 4.0))
	p4 := NewDataPoint(6.0, WithPoint(4.0, 5.0))
	d1 := newDataDist(0.5, p1)
	d2 := newDataDist(0.3, p2)
	d3 := newDataDist(0.2, p3)
	d4 := newDataDist(0.7, p4)
	kset := []DataDist{d1, d2, d3, d4}
	label := re.Label(kset)
	expected := 4.5
	if label != expected {
		t.Errorf("RegressionSelectorLabel failed. Expected %v, but got %v", expected, label)
	}
}

func TestWeightedVotingSelectorLabel(t *testing.T) {
	we := NewWeightedVotingSelector()
	we.Set("A", 0.5)
	we.Set("B", 0.3)
	we.Set("C", 0.2)
	kset := []DataDist{
		newDataDist(1.0, NewDataPoint("A", WithPoint(1.0, 2.0))),
		newDataDist(2.0, NewDataPoint("B", WithPoint(2.0, 3.0))),
		newDataDist(3.0, NewDataPoint("A", WithPoint(3.0, 4.0))),
		newDataDist(4.0, NewDataPoint("A", WithPoint(4.0, 5.0))),
	}
	label := we.Label(kset)
	expected := "A"
	if label != expected {
		t.Errorf("WeightedVotingSelectorLabel failed. Expected %v, but got %v", expected, label)
	}
}

func TestKNNFit(t *testing.T) {
	dataPoints := []DataPoint{
		&dataPoint{WithPoint(0.0, 0.0), true},
		&dataPoint{WithPoint(1.0, 1.0), true},
		&dataPoint{WithPoint(2.0, 2.0), false},
		&dataPoint{WithPoint(3.0, 3.0), false},
	}
	knn := NewKNN(3, NewEuclideanDist(), NewBinarySelector(), dataPoints)
	testData := WithPoint(1.5, 1.5)
	l := knn.Fit(testData)
	if l != true {
		t.Errorf("KNNFit failed. Expected true, but got %v", l)
	}
}
