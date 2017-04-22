import numpy as np
import itertools as it


def bellman_ford(node_pairs, neighbors, weights):
    n = len(neighbors), len(node_pairs)
    result = np.empty(len(node_pairs), dtype='float')
    # st_pairs = list(node_pairs).sort(key=lambda p:p[0])
    permutation = node_pairs[:,0].argsort()
    st_pairs = node_pairs[permutation]  # sorts with regard to the sources s

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

def floyd_warshall(node_pairs, neighbors, weights):
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

def bread_fsearch(node_pairs, neighbors, weights):
    pass

def dijkstra(node_pairs, neighbors, weights) :
    permutation = node_pairs[:,0].argsort()
    st_pairs = node_pairs[permutation]  # sorts with regard to the sources s
    n = len(neighbors)
    result = np.empty(len(node_pairs), dtype='float')

    for i, (s,t) in enumerate(st_pairs) :
        p, q = neighbors, np.copy(neighbors)
        dist = numpy.empty(n, dtype='float')
        dist.fill(np.inf)
        dist[s] = 0
        while len(p) >= 1 :




    inv_perm = np.empty(len(st_pairs), dtype=np.int)
    inv_perm[permutation] = np.arange(len(st_pairs))