package graph

type Operation interface {
	Forward()
	Backward()
}
