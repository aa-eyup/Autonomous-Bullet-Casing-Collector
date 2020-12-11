from itertools import combinations, permutations
from random import random
from math import exp

def evaluate(dist, path):
    score = dist[path[-1], path[0]]
    for i, j in zip(path[:-1], path[1:]):
        score += dist[i, j]
    return score

def neighborhood(dist, path):
    length = evaluate(dist, path)
    shuffled_combos = sorted(permutations(range(len(path)), 2), key=lambda x: random())
    for i, j in shuffled_combos:
        new_path = list(path)
        new_length = length \
                    - dist[new_path[i-1 if i > 0 else -1], new_path[i]] \
                    - dist[new_path[i], new_path[i+1 if i+1 < len(new_path) else 0]] \
                    - dist[new_path[j-1 if j > 0 else -1], new_path[j]] \
                    - dist[new_path[j], new_path[j+1 if j+1 < len(new_path) else 0]]
        new_path[i], new_path[j] = new_path[j], new_path[i]
        new_length += dist[new_path[i-1 if i > 0 else -1], new_path[i]] \
                    + dist[new_path[i], new_path[i+1 if i+1 < len(new_path) else 0]] \
                    + dist[new_path[j-1 if j > 0 else -1], new_path[j]] \
                    + dist[new_path[j], new_path[j+1 if j+1 < len(new_path) else 0]]
        #assert np.isclose(new_length, evaluate(dist, new_path)) # this technically approximates the new length, but it's close
        yield new_path, new_length

def construct(dist, vertex=0):
    """constructs initial solution using nearest neighbor heuristic
    
    Arguments
    ---------
    dist : 2 dimensional distance matrix.
    vertex : Vertex to start search on. Because this is a greedy search,
        the starting vertex can heavily affect the final solution.
        
    Returns
    -------
    path : list of vertices in order
    """
    vertices = range(dist.shape[0])
    path = [vertex]
    while len(path) < len(vertices):
        unvisited = set(vertices) - set(path)
        vertex = min(unvisited, key=lambda i: dist[vertex, i])
        path.append(vertex)
    return path

def improve(dist, path):
    """improves a solution using 2-opt improvement heuristic
    
    Arguments
    ---------
    dist : 2 dimensional distance matrix.
    path : list of vertices in order.
    
    Returns
    -------
    path: list of vertices in order.
    """
    length = evaluate(dist, path)
    new_path, new_length = min(neighborhood(dist, path), key=lambda p: p[1])
    if new_length < length:
        path = new_path
        length = new_length
    return path, length
    
def solve(dist, vertex=0):
    """searches for a suboptimal solution to the traveling salesman problem by hill climbing.
    
    Arguments
    ---------
    dist : 2 dimensional distance matrix.
    vertex : Vertex to start search on. Because this is a greedy search,
        the starting vertex can heavily affect the final solution.
        
    Returns
    -------
    best_path : list of vertices in order.
    """
    best_path = construct(dist, vertex=vertex)
    best_length = evaluate(dist, best_path)
    while True:
        path, length = improve(dist, best_path)
        if length < best_length:
            best_length = length
            best_path = path
        else:
            break
    return best_path, best_length

if __name__ == "__main__":
    import numpy as np
    from scipy.spatial.distance import cdist

    np.random.seed(1000)
    coords = np.random.uniform(low=0, high=100, size=(25, 2))
    dist = cdist(coords, coords)
    path, length = solve(dist)

    # plotting
    import matplotlib.pyplot as plt
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, j in zip(path[:-1], path[1:]):
        x, y = coords[i, 0], coords[i, 1]
        dx, dy = coords[j, 0] - coords[i, 0], coords[j, 1] - coords[i, 1]
        plt.arrow(x, y, dx, dy, shape='full', lw=1, length_includes_head=True, head_width=0)
    i, j = path[-1], path[0]
    x, y = coords[i, 0], coords[i, 1]
    dx, dy = coords[j, 0] - coords[i, 0], coords[j, 1] - coords[i, 1]
    plt.arrow(x, y, dx, dy, shape='full', lw=1, length_includes_head=True, head_width=0)
    plt.show()
    