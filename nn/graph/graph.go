package graph

import (
	"errors"
	"os"
	"unicode"
)

var ErrEdgeNoExist error = errors.New("edge no exist")
var ErrNodeNoExist error = errors.New("node no exist")

func prepareName(name string) string {
	outname := ""
	for i := 0; i < len(name); i++ {
		ch := rune(name[i])
		if !unicode.IsLetter(ch) && !unicode.IsDigit(ch) {
			outname += "_"
		} else {
			outname += string(name[i])
		}
	}
	if outname == "" {
		outname = "G"
	}
	return outname
}

// It Represents graph node
type Node struct {
	name  string
	value any
}

// Node name
func (n *Node) Name() string {
	return n.name
}

// Node value
func (n *Node) Value() any {
	return n.value
}

// Node string representation
func (n *Node) String() string {
	return prepareName(n.name)
}

// Graph (digraph)
type Graph struct {
	name     string  //graph name
	vertices []*Node //graph vertices
	edges    [][]int //gragh edges
}

// Create a graph
func New(name string) Graph {
	return Graph{
		name:     name,
		vertices: make([]*Node, 0, 100),
		edges:    make([][]int, 0, 100),
	}
}

// Get graph name
func (graph *Graph) Name() string {
	return graph.name
}

// Add a new node to graph
func (graph *Graph) AddNode(name string, value any) int {
	vid := len(graph.vertices)
	graph.vertices = append(graph.vertices, &Node{name: name, value: value})
	graph.edges = append(graph.edges, []int{})
	return vid
}

// Add edge to graph
func (graph *Graph) AddEdge(src, dst int) error {
	if src < 0 || dst < 0 || src >= len(graph.vertices) || dst >= len(graph.vertices) {
		return ErrNodeNoExist
	}
	graph.edges[dst] = append(graph.edges[dst], src)
	return nil
}

// Remove edge from graph
func (graph *Graph) RemoveEdge(src, dst int) bool {
	if src < 0 || src > len(graph.vertices) || dst < 0 || dst > len(graph.vertices) {
		return false
	}
	srcLs := graph.edges[dst]
	for i := 0; i < len(srcLs); i++ {
		if srcLs[i] == src {
			srcLs = append(srcLs[:i], srcLs[i+1:]...)
			graph.edges[dst] = srcLs
			return true
		}
	}
	return false
}

// Test if edge exist
func (graph *Graph) HasEdge(src, dst int) bool {
	if src < 0 || src >= len(graph.vertices) {
		return false
	}
	if dst < 0 || dst >= len(graph.vertices) {
		return false
	}
	edges := graph.edges[dst]
	for i := range edges {
		if edges[i] == src {
			return true
		}
	}
	return false
}

// Get childs of node edge
func (graph *Graph) OutEdges(node int) []int {
	if node < 0 || node >= len(graph.edges) {
		return []int{}
	}
	dstLs := make([]int, 0, 10)
	for edge, srcLs := range graph.edges {
		for _, dst := range srcLs {
			if dst == node {
				dstLs = append(dstLs, edge)
			}
		}
	}
	return dstLs
}

// Get parent nodes of edge
func (graph *Graph) InEdges(node int) []int {
	if node < 0 || node >= len(graph.edges) {
		return []int{}
	}
	return graph.edges[node]
}

func (graph *Graph) NodeAt(index int) *Node {
	if index < 0 || index >= len(graph.vertices) {
		return nil
	}
	return graph.vertices[index]
}

func (graph *Graph) RemoveNodeAt(index int) bool {
	if index < 0 || index >= len(graph.vertices) {
		return false
	}
	graph.edges = append(graph.edges[:index], graph.edges[index+1:]...)
	graph.vertices = append(graph.vertices[:index], graph.vertices[index+1:]...)
	for i := range graph.edges {
		srcLs := graph.edges[i]
		for j := 0; j < len(srcLs); {
			if srcLs[j] == index {
				srcLs = append(srcLs[:j], srcLs[j+1:]...)
			} else {
				if srcLs[j] > index {
					srcLs[j]--
				}
				j++
			}
		}
		graph.edges[i] = srcLs
	}
	return true
}

// Get nodes count
func (graph *Graph) LenNodes() int {
	return len(graph.vertices)
}

// Get edges count
func (graph *Graph) LenEdges() int {
	return len(graph.edges)
}

func (graph *Graph) ToDot(fileName string) error {
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()
	file.Write([]byte(graph.String()))
	return nil
}

// Dot representation
func (graph *Graph) String() string {
	digraph := "digraph " + prepareName(graph.name) + "{\n"
	for i := range graph.edges {
		srcLs := graph.edges[i]
		dst := graph.vertices[i]
		for j := range srcLs {
			src := graph.vertices[srcLs[j]]
			digraph += src.String() + " -> " + dst.String() + "\n"
		}
	}
	return digraph + "}"
}

// DFS (Depth-First Search)
func (graph *Graph) DFS(node int) []int {
	stack := make([]int, 0, 10)
	visited := make([]bool, len(graph.vertices))
	sequence := make([]int, 0, len(graph.vertices))
	stack = append(stack, node)
	for len(stack) != 0 {
		curr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if !visited[curr] {
			visited[curr] = true
			sequence = append(sequence, curr)
			stack = append(stack, graph.edges[curr]...)
		}
	}
	return sequence
}

// BFS (Breadth-First Search)
func (graph *Graph) BFS(node int) []int {
	queue := make([]int, 0, 10)
	visited := make([]bool, len(graph.vertices))
	sequence := make([]int, 0, len(graph.vertices))
	queue = append(queue, node)
	visited[node] = true
	for len(queue) != 0 {
		curr := queue[0]
		queue = queue[1:]
		sequence = append(sequence, curr)
		for _, neighbor := range graph.edges[curr] {
			if !visited[neighbor] {
				queue = append(queue, neighbor)
				visited[neighbor] = true
			}
		}
	}
	return sequence
}

// Reverse BFS (Reverse Breadth-First Search)
func (graph *Graph) ReverseBFS(node int) []int {
	sequence := graph.BFS(node)
	for i, j := 0, len(sequence)-1; i < j; i, j = i+1, j-1 {
		sequence[i], sequence[j] = sequence[j], sequence[i]
	}
	return sequence
}

func (graph *Graph) HasCycle() bool {
	visited := make([]bool, len(graph.vertices))
	recStack := make([]bool, len(graph.vertices))

	for i := range graph.vertices {
		if !visited[i] && hasCycleUtil(i, visited, recStack, graph) {
			return true
		}
	}
	return false
}

func hasCycleUtil(node int, visited, recStack []bool, graph *Graph) bool {
	visited[node] = true
	recStack[node] = true

	// Recorremos los nodos adyacentes al nodo actual
	for _, adj := range graph.InEdges(node) {
		if !visited[adj] && hasCycleUtil(adj, visited, recStack, graph) {
			return true
		} else if recStack[adj] {
			return true
		}
	}

	recStack[node] = false
	return false
}
