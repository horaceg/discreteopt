import numpy as np
#import itertools as it


def bellman_ford(graph):
    neighbors, weights, node_pairs = graph
    n = len(neighbors)
    result = np.empty(len(node_pairs), dtype='float')
    permutation = node_pairs[:,0].argsort()
    st_pairs = node_pairs[permutation]  
    # sorts with regard to the sources s 
    # s.t. redundant computation is avoided

    for i, (s, t) in enumerate(st_pairs) :
        if i > 0 and st_pairs[i-1,0] == s:
            result[i] = dist[t]
        else:
            dist = np.empty(n)
            dist.fill(np.inf)
            # parent = [None]*n
            dist[s] = 0.
            for k in range(n-1):
                for u in range(n):
                    for v, w in zip(neighbors[u], weights[u]):
                        dist[v] = min(dist[v], w + dist[u])
            result[i] = dist[t]
    inv_perm = np.empty(len(st_pairs), dtype=np.int)
    inv_perm[permutation] = np.arange(len(st_pairs))

    return result[inv_perm]


def floyd_warshall(graph):
    neighbors, weights, node_pairs = graph
    n = len(neighbors)
    result = np.empty(len(node_pairs), dtype='float')
    dist = np.empty((n, n), dtype='float')
    dist.fill(np.inf)
    for u in range(n) :
        dist[u][neighbors[u]] = weights[u]
        dist[u][u] = 0

    #for k,i,j in it.product(range(n), repeat=3):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])

    for i,(s,t) in enumerate(node_pairs):
        result[i] = dist[s,t]

    return result


def bfs(graph):
    neighbors, weights, node_pairs = graph
    pass


def dijkstra(graph):
    neighbors, weights, node_pairs = graph
    g = next((w for x in weights for w in x if w < 0), 1)
    print("Poids négatifs ?", g < 0)
    assert g == 1

    n = len(neighbors)
    result = np.empty(len(node_pairs), dtype='float')

    for i, (s, t) in enumerate(node_pairs) :
        q = np.arange(n, dtype='int32')
        dist = np.empty(n, dtype='float')
        dist.fill(np.inf)
        u, dist[s] = s, 0
        while u != t :
            d, j = np.min(dist[q]), np.argmin(dist[q])
            u = q[j]
            q = np.delete(q, j)
            for v, w in zip(neighbors[u], weights[u]):
                dist[v] = min(dist[v], dist[u] + w)
        result[i] = dist[t]
    return result


def bfs(start, target, neighbors, num_nodes) :
    length = [float("inf")]*num_nodes
    vertQueue = [start]
    length[start] = 0
    while(len(vertQueue) > 0)and(length[target] == float("inf")) :
        u = vertQueue[0]
        vertQueue = vertQueue[1:len(vertQueue)]
        for v in neighbors[u] :
            if length[v] == float("inf"):
                length[v] = length[u] + 2
                vertQueue = vertQueue + [v]
    return length[target]


def solution_bfs(graph):
    neighbors, weights, node_pairs = graph
    num_pairs = len(node_pairs)
    answer = np.empty(num_pairs)
    for i in range(num_pairs):
        answer[i] = bfs(node_pairs[i][0], node_pairs[i][1], neighbors, len(neighbors))
    return answer
