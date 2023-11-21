package graph

import (
	"fmt"
	"testing"
)

func TestTensor(t *testing.T) {
	ts := NewTensor(
		[]float64{
			0, 1, 2, 3,
			4, 5, 6, 7,
			8, 9, 10, 11,
			12, 13, 14, 15,
		},
		Float64,
		NewShape(4, 4),
	)
	s := [][]float64{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 10, 11},
		{12, 13, 14, 15},
	}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if s[j][i] != ts.GetF64At([]int{i, j}) {
				t.FailNow()
			}
		}
	}
	ts.Reshape(NewShape(2, 2, 2, 2))
	r := [2][2][2][2]float64{{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}}, {{{8, 9}, {10, 11}}, {{12, 13}, {14, 15}}}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			for k := 0; k < 2; k++ {
				for l := 0; l < 2; l++ {
					if r[l][k][j][i] != ts.GetF64At([]int{i, j, k, l}) {
						t.FailNow()
					}
				}
			}
		}
	}
	ts.Reshape(NewShape(2, 8))
	u := [8][2]float64{
		{0, 1},
		{2, 3},
		{4, 5},
		{6, 7},
		{8, 9},
		{10, 11},
		{12, 13},
		{14, 15},
	}
	sub := ts.GetSub([]int{1, -1})
	for i := 0; i < 8; i++ {
		if u[i][1] != sub.Get([]int{0, i}).(float64) {
			t.FailNow()
		}
	}
	slice := ts.F64Slice()
	a := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	for i := 0; i < 16; i++ {
		if slice[i] != a[i] {
			t.FailNow()
		}
	}
	sh := ts.Shape()
	for i, ln := 0, sh.Len(); i < ln; i++ {
		if a[i] != ts.GetAt(i) {
			t.FailNow()
		}
	}
	if !ts.Equal(ts) {
		t.FailNow()
	}
	fmt.Println(ts.String())
	ts.Reshape(NewShape(4, 4))
	ts.Set([]int{1, 2}, 100.0)
	fmt.Println(ts.String())
}
