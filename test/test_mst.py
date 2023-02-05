import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances
import networkx as nx


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < 0.0001

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # read in mst as networkx graph
    g_nx = nx.from_numpy_array(mst)
    # check that the mst is connected
    assert nx.is_connected(g_nx) == True, "MST should be connected!"

    # make sure that the weight of the mst is less than the graph used to construct it
    assert np.sum(adj_mat) > np.sum(mst)


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student(allowed_error: float = 0.000001):
    """
    
    MSTs should be symmetric. Here, I will test that the msts generated for the sample data are both symmetric.
    
    """
    # construct MST using small dataset
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    # check that they are symmetric
    assert np.all(np.abs(g.mst - g.mst.T) < allowed_error) == True

    # repeat for single cell dataset
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path)  # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords)  # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    # check that they are symmetric
    assert np.all(np.abs(g.mst - g.mst.T) < allowed_error) == True
