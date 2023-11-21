package graph

import (
	"fmt"
	"testing"
)

func TestGraph(t *testing.T) {
	g := New("mygraph")
	g.AddNode("0", 0) //0
	g.AddNode("1", 0) //1
	g.AddNode("2", 0) //2
	g.AddNode("3", 0) //3
	g.AddNode("4", 0) //4
	g.AddNode("5", 0) //5
	g.AddNode("6", 0) //6
	g.AddEdge(0, 1)
	g.AddEdge(0, 2)
	g.AddEdge(0, 3)
	g.AddEdge(2, 4)
	g.AddEdge(2, 5)
	g.AddEdge(3, 6)
	g.AddEdge(6, 4)
	g.AddEdge(3, 3)
	g.AddEdge(4, 3)
	g.AddEdge(1, 5)
	g.AddEdge(3, 5)
	g.AddEdge(3, 5)
	g.AddEdge(3, 0)
	g.ToDot("./graph.dot")
	g.RemoveEdge(3, 5)
	g.RemoveEdge(3, 3)
	g.ToDot("./graph-remove.dot")
	//Test has when has
	if !g.HasEdge(3, 6) {
		t.FailNow()
	}
	//Test has when no has
	if g.HasEdge(2, 1) {
		t.FailNow()
	}
	//Show out
	out := New("out")
	out.AddNode("3", 0)
	outs := g.OutEdges(3)
	for i := 0; i < len(outs); i++ {
		out.AddNode(fmt.Sprintf("%d", outs[i]), 0)
		out.AddEdge(0, i+1)
	}
	out.ToDot("./graph-out.dot")
	in := New("out")
	in.AddNode("3", 0)
	ins := g.InEdges(3)
	for i := 0; i < len(ins); i++ {
		in.AddNode(fmt.Sprintf("%d", ins[i]), 0)
		in.AddEdge(i+1, 0)
	}
	in.ToDot("./graph-in.dot")
	g.RemoveNodeAt(3)
	g.ToDot("./graph-remove-3.dot")
}

func TestDFS(t *testing.T) {
	g := New("G")
	node_a := g.AddNode("node_a", 0)
	node_b := g.AddNode("node_b", 0)
	node_c := g.AddNode("node_c", 0)
	node_d := g.AddNode("node_d", 0)
	node_e := g.AddNode("node_e", 0)
	node_f := g.AddNode("node_f", 0)
	node_g := g.AddNode("node_g", 0)
	node_h := g.AddNode("node_h", 0)
	node_i := g.AddNode("node_i", 0)
	node_j := g.AddNode("node_j", 0)
	g.AddEdge(node_j, node_h)
	g.AddEdge(node_i, node_h)
	g.AddEdge(node_h, node_a)
	g.AddEdge(node_g, node_a)
	g.AddEdge(node_f, node_b)
	g.AddEdge(node_e, node_c)
	g.AddEdge(node_d, node_c)
	g.AddEdge(node_c, node_b)
	g.AddEdge(node_b, node_a)
	g.ToDot("./dfs.dot")
	result := g.DFS(node_a)
	if !assert(result, []int{node_a, node_b, node_c, node_d, node_e, node_f, node_g, node_h, node_i, node_j}) {
		t.FailNow()
	}
}

func TestBFS(t *testing.T) {
	g := New("G")
	node_a := g.AddNode("node_a", 0)
	node_b := g.AddNode("node_b", 0)
	node_c := g.AddNode("node_c", 0)
	node_d := g.AddNode("node_d", 0)
	node_e := g.AddNode("node_e", 0)
	node_f := g.AddNode("node_f", 0)
	node_g := g.AddNode("node_g", 0)
	node_h := g.AddNode("node_h", 0)
	node_i := g.AddNode("node_i", 0)
	node_j := g.AddNode("node_j", 0)
	g.AddEdge(node_d, node_c)
	g.AddEdge(node_e, node_c)
	g.AddEdge(node_c, node_b)
	g.AddEdge(node_b, node_a)
	g.AddEdge(node_g, node_a)
	g.AddEdge(node_h, node_a)
	g.AddEdge(node_f, node_b)
	g.AddEdge(node_i, node_h)
	g.AddEdge(node_j, node_h)
	g.ToDot("./bfs.dot")
	result := g.BFS(node_a)
	if !assert(result, []int{node_a, node_b, node_g, node_h, node_c, node_f, node_i, node_j, node_d, node_e}) {
		t.FailNow()
	}
}

func assert(t1, t2 []int) bool {
	if len(t1) != len(t2) {
		return false
	}
	for i := range t1 {
		if t1[i] != t2[i] {
			return false
		}
	}
	return true
}

func TestHasCycle(t *testing.T) {
	// Creamos un grafo con ciclo
	g1 := New("g1")
	g1.AddNode("A", nil)
	g1.AddNode("B", nil)
	g1.AddNode("C", nil)
	g1.AddEdge(0, 1)
	g1.AddEdge(1, 2)
	g1.AddEdge(2, 0)

	g1.ToDot("./cycle.dot")
	if !g1.HasCycle() {
		t.Errorf("g1 debería tener ciclo")
	}

	// Creamos un grafo sin ciclo
	g2 := New("g2")
	g2.AddNode("A", nil)
	g2.AddNode("B", nil)
	g2.AddNode("C", nil)
	g2.AddEdge(0, 1)
	g2.AddEdge(1, 2)

	g2.ToDot("./un-cycle.dot")
	if g2.HasCycle() {
		t.Errorf("g2 no debería tener ciclo")
	}
}
