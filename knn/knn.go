package knn

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

var (
	ErrParallelLevelIsNotValid = fmt.Errorf("parallelism level is not greater or equal to 1")
	ErrPointDimensionMismatch  = fmt.Errorf("point dimension is not the same")
	ErrKIsNotValid             = fmt.Errorf("value of k is not greater or equal to 1")
)

var prll chan int //control parallelism level
var plv int
var prllMtx sync.RWMutex //control access to parallelism

// Set the numbers of gorutines used in every function of knn
func SetParallelLv(lv int) error {
	prllMtx.Lock()
	defer prllMtx.Unlock()
	if lv < 1 {
		panic(ErrParallelLevelIsNotValid)
	}
	prll = make(chan int, lv)
	plv = lv
	return nil
}

// Get the numbers of gorutines used in every function of knn
func GetParallelLv() int {
	prllMtx.RLock()
	defer prllMtx.RUnlock()
	return plv
}

//Types used in knn

type RuleKind int //A rule kind used to select a class

type Distance interface {
	Eval(point1, point2 Point) float64
}

type Point []float64

func WithPoint(xs ...float64) Point {
	return Point(xs)
}

func NewPoint(dim int) Point {
	return make(Point, dim)
}

func (p Point) Dim() int {
	return len(p)
}

type euclidean struct{}

func NewEuclideanDist() Distance {
	return &euclidean{}
}

func (eu *euclidean) Eval(p1, p2 Point) float64 {
	if p1.Dim() != p2.Dim() {
		panic(ErrPointDimensionMismatch)
	}
	sum := 0.0
	for i, ln := 0, len(p1); i < ln; i++ {
		dif := p1[i] - p2[i]
		sum += dif * dif
	}
	return math.Sqrt(sum)
}

type manhattan struct{}

func NewManhattanDist() Distance {
	return &manhattan{}
}

func (ma *manhattan) Eval(p1, p2 Point) float64 {
	if p1.Dim() != p2.Dim() {
		panic(ErrPointDimensionMismatch)
	}
	sum := 0.0
	for i, ln := 0, len(p1); i < ln; i++ {
		sum += math.Abs(p1[i] - p2[i])
	}
	return sum
}

type minkowski struct {
	ratio float64
}

func NewMinkowskiDist(ratio float64) Distance {
	return &minkowski{
		ratio: ratio,
	}
}

func (mi *minkowski) Eval(p1, p2 Point) float64 {
	if p1.Dim() != p2.Dim() {
		panic(ErrPointDimensionMismatch)
	}
	sum := 0.0
	for i, ln := 0, len(p1); i < ln; i++ {
		sum += math.Pow(math.Abs(p1[i]-p2[i]), mi.ratio)
	}
	return math.Pow(sum, 1/mi.ratio)
}

type chebyshev struct{}

func NewChebyshevDist() Distance {
	return &chebyshev{}
}

func (ch *chebyshev) Eval(p1, p2 Point) float64 {
	if p1.Dim() != p2.Dim() {
		panic(ErrPointDimensionMismatch)
	}
	var max float64
	for i, ln := 0, len(p1); i < ln; i++ {
		if math.Abs(p1[i]-p2[i]) > max {
			max = math.Abs(p1[i] - p2[i])
		}
	}
	return max
}

type pearsonCorrelation struct{}

func NewPearsonCorrelationDist() Distance {
	return &pearsonCorrelation{}
}

func (pe *pearsonCorrelation) Eval(p1, p2 Point) float64 {
	var sum1, sum2, sum1Sq, sum2Sq, pSum float64
	n := float64(len(p1))
	for i, ln := 0, len(p1); i < ln; i++ {
		sum1 += p1[i]
		sum2 += p2[i]
		sum1Sq += p1[i] * p1[i]
		sum2Sq += p2[i] * p2[i]
		pSum += p1[i] * p2[i]
	}
	num := pSum - (sum1 * sum2 / n)
	den := math.Sqrt((sum1Sq - sum1*sum1/n) * (sum2Sq - sum2*sum2/n))
	if den == 0.0 {
		return 0.0
	}
	return 1 - num/den
}

type hamming struct{}

func NewHammingDist() Distance {
	return &hamming{}
}

func (ha *hamming) Eval(p1, p2 Point) float64 {
	if p1.Dim() != p2.Dim() {
		panic(ErrPointDimensionMismatch)
	}
	distance := 0
	for i, ln := 0, len(p1); i < ln; i++ {
		if p1[i] != p2[i] {
			distance++
		}
	}
	return float64(distance)
}

type DataDist interface {
	Dist() float64
	DataPoint() DataPoint
}

type datadist struct {
	dist float64
	data DataPoint
}

func newDataDist(dist float64, data DataPoint) DataDist {
	return &datadist{
		dist: dist,
		data: data,
	}
}

func (dd *datadist) Dist() float64 {
	return dd.dist
}

func (dd *datadist) DataPoint() DataPoint {
	return dd.data
}

type Selector interface {
	Label(kset []DataDist) any
}

type binarySelector struct{}

func NewBinarySelector() Selector {
	return &binarySelector{}
}

func (bi *binarySelector) Label(kset []DataDist) interface{} {
	var ones, zeros int
	for _, d := range kset {
		if d.DataPoint().Label().(bool) {
			ones++
		} else {
			zeros++
		}
	}
	return ones > zeros
}

type multiClassSelector struct{}

func NewMultiClassSelector() Selector {
	return &multiClassSelector{}
}

func (mu *multiClassSelector) Label(kset []DataDist) interface{} {
	counts := make(map[interface{}]int)
	for _, d := range kset {
		label := d.DataPoint().Label()
		if _, ok := counts[label]; !ok {
			counts[label] = 0
		}
		counts[label]++
	}
	maxCount := 0
	maxLabel := kset[0].DataPoint().Label()
	for label, count := range counts {
		if count > maxCount {
			maxCount = count
			maxLabel = label
		}
	}
	return maxLabel
}

type regressionSelector struct{}

func NewRegressionSelector() Selector {
	return &regressionSelector{}
}

func (re *regressionSelector) Label(kset []DataDist) interface{} {
	var sum float64
	for _, d := range kset {
		sum += d.DataPoint().Label().(float64)
	}
	return sum / float64(len(kset))
}

type WeightedVotingSelector interface {
	Selector
	Set(label any, weight float64)
	Get(label any) float64
	Del(label any)
	Has(label any) bool
	Len() int
	Clear()
}

type weightedVotingSelector struct {
	weights map[any]float64
}

func NewWeightedVotingSelector() WeightedVotingSelector {
	return &weightedVotingSelector{
		weights: make(map[any]float64),
	}
}

func (we *weightedVotingSelector) Set(label any, weight float64) {
	we.weights[label] = weight
}

func (we *weightedVotingSelector) Get(label any) float64 {
	return we.weights[label]
}

func (we *weightedVotingSelector) Del(label any) {
	delete(we.weights, label)
}

func (we *weightedVotingSelector) Has(label any) bool {
	_, ok := we.weights[label]
	return ok
}

func (we *weightedVotingSelector) Len() int {
	return len(we.weights)
}

func (we *weightedVotingSelector) Clear() {
	we.weights = make(map[any]float64)
}

func (we *weightedVotingSelector) Label(kset []DataDist) interface{} {
	counts := make(map[interface{}]float64)
	for _, d := range kset {
		label := d.DataPoint().Label()
		weight, ok := we.weights[label]
		if !ok {
			weight = 1e-6
		}
		if _, ok := counts[label]; !ok {
			counts[label] = 0
		}
		counts[label] += 1 / (d.Dist() + weight)
	}
	maxCount := 0.0
	maxLabel := kset[0].DataPoint().Label()
	for label, count := range counts {
		if count > maxCount {
			maxCount = count
			maxLabel = label
		}
	}
	return maxLabel
}

type inverseDistanceSelector struct{}

func NewInverseDistanceSelector() Selector {
	return &inverseDistanceSelector{}
}

func (in *inverseDistanceSelector) Label(kset []DataDist) interface{} {
	freq := make(map[interface{}]float64)
	for _, d := range kset {
		label := d.DataPoint().Label()
		freq[label] += 1 / d.Dist()
	}
	var maxLabel interface{}
	maxWeight := 0.0
	for label, weight := range freq {
		if weight > maxWeight {
			maxWeight = weight
			maxLabel = label
		}
	}
	return maxLabel
}

type smoothInverseDistanceSelector struct {
	WeightParam    float64
	SmoothingParam float64
}

func NewSmoothInverseDistanceSelector(weightParam, smoothingParam float64) Selector {
	return &smoothInverseDistanceSelector{WeightParam: weightParam, SmoothingParam: smoothingParam}
}

func (sm *smoothInverseDistanceSelector) Label(kset []DataDist) interface{} {
	freq := make(map[interface{}]float64)
	for _, d := range kset {
		label := d.DataPoint().Label()
		weight := 1 / (math.Pow(d.Dist(), sm.WeightParam) + sm.SmoothingParam)
		freq[label] += weight
	}
	var maxLabel interface{}
	maxWeight := 0.0
	for label, weight := range freq {
		if weight > maxWeight {
			maxWeight = weight
			maxLabel = label
		}
	}
	return maxLabel
}

type DataPoint interface {
	Point() Point
	Label() any
}

type KNN struct {
	data     []DataPoint
	k        int
	dist     Distance
	selector Selector
}

func NewKNN(k int, dist Distance, selector Selector, dataPoints []DataPoint) *KNN {
	if k <= 0 {
		panic(ErrKIsNotValid)
	}
	return &KNN{
		k:        k,
		dist:     dist,
		data:     dataPoints,
		selector: selector,
	}
}

func (knn *KNN) Append(dp DataPoint) *KNN {
	knn.data = append(knn.data, dp)
	return knn
}

func (knn *KNN) GetDataPoints() []DataPoint {
	return knn.data
}

func (knn *KNN) Fit(testData Point) any {

	distances := make([]DataDist, len(knn.data))
	if plv > 1 {
		wg := sync.WaitGroup{}
		for i, d := range knn.data {
			go func(i int, d DataPoint) {
				defer wg.Done()
				prll <- 0
				distances[i] = newDataDist(knn.dist.Eval(d.Point(), testData), d)
				<-prll
			}(i, d)
		}
		wg.Wait()
	} else {
		for i, d := range knn.data {
			distances[i] = newDataDist(knn.dist.Eval(d.Point(), testData), d)
		}
	}

	sort.Slice(distances, func(i, j int) bool {
		return distances[i].Dist() < distances[j].Dist()
	})

	kset := distances[:knn.k]

	return knn.selector.Label(kset)
}

type dataPoint struct {
	point Point
	label any
}

func NewDataPoint(label any, point Point) DataPoint {
	return &dataPoint{
		label: label,
		point: point,
	}
}

func (ap *dataPoint) Label() any {
	return ap.label
}

func (ap *dataPoint) Point() Point {
	return ap.point
}
