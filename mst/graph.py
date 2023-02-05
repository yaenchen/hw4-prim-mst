import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        # instantiate mst
        mst = np.zeros(self.adj_mat.shape)
        # vertices are going to be an integer for each row/column
        # total number of vertices is the number of rows (or it could be columns)
        tol_vertices = np.shape(self.adj_mat)[0]
        # instantiate an empty list for visited nodes
        visited = []
        # instantiate a starting vertex at 0
        start_vertex = 0
        # instantiate an empty list for the priority queue, store outgoing edges from start
        priority_queue = []

        # for all vertices, add its weight, starting vertex, and current vertex to the priority queue if it has a cost
        for vertex in range(tol_vertices):
            if self.adj_mat[start_vertex, vertex] > 0:
                priority_queue.append((self.adj_mat[start_vertex, vertex], start_vertex, vertex))

        # now that we've visited a node, add it to visited
        visited.append(start_vertex)
        # sort the priority queue based on weight, which is the first element in each tuple
        heapq.heapify(priority_queue)

        while len(visited) < tol_vertices:
            # pop the lowest weight edge from the priority queue
            weight, start, end = heapq.heappop(priority_queue)
            if end not in visited:
                # add this edge to the MST
                mst[start, end] = weight
                mst[end, start] = weight
                # add the destination to visited
                visited.append(end)

                # add all outgoing edges from the destination vertex into priority queue
                for vertex in range(tol_vertices):
                    if self.adj_mat[end, vertex] > 0:
                        heapq.heappush(priority_queue, (self.adj_mat[end, vertex], end, vertex))

        # add mst as attribute
        self.mst = mst