from functools import lru_cache
import numpy as np
from collections import deque


class Solver:
    
    def __init__(self, value=[], size=[], capacity=0):
        self.value = value
        self.size = size
        self.capacity = capacity
        self.mask, self.inv_perm, self.ovalue, self.osize = self.dense_sort()

    def solve_it(self):
        if len(self.size) <= 200:
            name, val, items = ("dynamic programming",) + self.dynamic()
            items = np.array(items)
        elif len(self.size) <= 200:
            name, val, items = ("Branch and bound - depth-first",) + self.branch_bound()
        else:
            name, val, items = ("Branch and bound - limited discrepancy (1)",) + self.lds(errors=1)
            # name, val, items = ("Branch and bound - iterative dfs",) + self.dfs_iter()
        print("============ Solution found ===============", "Algorithm : {}".format(name), "value : {}".format(val),
              "solution items sorted by density", sep='\n')
        print(' '.join(map(str, items[self.mask])))

        return val, items

    def dense_sort(self):
        # returns density-sorted value, size arrays
        mask = (np.array(self.value) / np.array(self.size)).argsort()[::-1]
        ovalue, osize = self.value[mask], self.size[mask]  # order by density
        inv_perm = np.empty(len(self.size), dtype=np.int)
        inv_perm[mask] = np.arange(len(self.size))  # to retrieve the original order
        print("==== Density sort ====", "mask : ", list(mask), "ovalue : ", list(ovalue), "osize : ", list(osize), sep='\n')
        return mask, inv_perm, ovalue, osize

    def compute_val(self, dec_var):
        # computes total value of a solution dec_var = [x1...xn] , xi in {0,1}
        return np.dot(self.value.T, dec_var)

    def dynamic(self):
        @lru_cache(maxsize=None)  # memoization : stores computed values by function k()
        def k(i, s):
            if i == 0: return 0
            elif self.size[i - 1] <= s: return max(k(i-1, s), self.value[i-1] + k(i-1, s - self.size[i-1]))
            else: return k(i-1, s)

        def findpath(i, s, val):
            if i == 0: return []
            elif val == k(i-1, s): return findpath(i-1, s, val) + [0]
            else: return findpath(i-1, s - self.size[i-1], val - self.value[i-1]) + [1]

        sol = k(len(self.size), self.capacity)
        items = findpath(len(self.size), self.capacity, sol)
        return sol, items

    def heuristic(self, i, room):
        # Estimates additional value to gain from i to end, having items sorted by density
        # Is admissible, e.g. always >= to true value
        v, s, item = 0, 0, i
        while s < room and item < len(self.osize):
            if self.osize[item] + s <= room:
                s += self.osize[item]
                v += self.ovalue[item]
            else:
                v += self.ovalue[item] * (room - s) / self.osize[item]
                s = room
            item += 1
        return v

    def greedy(self):
        room, set, val = self.capacity, [], 0
        for v, s in zip(self.ovalue, self.osize):
            if s <= room:
                set += [1]
                room -= s
                val += v
            else:
                set += [0]
        return val, np.array(set)[self.inv_perm]

    def lowbound(self):
        print("======== Lower greedy bound ============")
        val, items = self.greedy()
        print("value : {}".format(val), "items by density", list(items[self.mask]), sep='\n')
        return val, items[self.mask]

    def branch_bound(self):
        # depth-first search, having items sorted by density.
        # produces stack overflow with test 5 because of recursion limits

        def dfs(item, v, room, set):
            nonlocal bestval, bestset
            if item == len(self.osize):
                if v > bestval:
                    bestset = set
                    bestval = v
            else:
                estim = v + self.heuristic(item, room)
                if estim > bestval:
                    if self.osize[item] <= room:
                        dfs(item+1, v+self.ovalue[item], room-self.osize[item], set+[1])
                    dfs(item+1, v, room, set+[0])

        bestval, bestset = self.lowbound()
        dfs(0, 0, self.capacity, [])
        fset = np.array(bestset)[self.inv_perm]

        assert self.compute_val(fset) == bestval
        return bestval, fset

    def dfs_iter(self):
        root = {'item': 0, 'val': 0, 'room': self.capacity, 'set': []}
        bestval, bestset = self.lowbound()
        bestnode = {'item': None, 'val': bestval, 'room': None, 'set': bestset}
        s = deque()
        s.append(root)
        while len(s) > 0:
            node = s.pop()
            if node['item'] == len(self.osize):
                if node['val'] > bestnode['val']:
                    bestnode = node
            else:
                estim = node['val'] + self.heuristic(node['item'], node['room'])
                if estim > bestnode['val']:
                    s.append({'item': node['item'] + 1, 'val': node['val'],
                              'room': node['room'], 'set': node['set']+[0]})
                    if self.osize[node['item']] <= node['room']:
                        s.append({'item': node['item']+1, 'val': node['val']+self.ovalue[node['item']],
                                  'room': node['room']-self.osize[node['item']], 'set': node['set']+[1]})
        return bestnode['val'], np.array(bestnode['set'])[self.inv_perm]

    def lds(self, errors=1):
        # limited discrepancy search : greedy algorithm with up to k discrepancies

        root = {'item': 0, 'val': 0, 'room': self.capacity, 'set': [], 'left':errors}
        bestval, bestset = self.lowbound()
        bestnode = {'item': None, 'val': bestval, 'room': None, 'set': bestset, 'left':0}
        s = deque()  # stack
        s.append(root)
        left = errors
        while len(s) > 0:
            node = s.pop()
            if node['item'] == len(self.osize):
                if node['val'] > bestnode['val']:
                    bestnode = node
            else:
                estim = node['val'] + self.heuristic(node['item'], node['room'])
                if estim > bestnode['val']:
                    if self.osize[node['item']] <= node['room']:
                        if node['left'] >= 1:
                            s.append({'item': node['item']+1, 'val': node['val'],
                                      'room': node['room'], 'set': node['set']+[0],
                                      'left': node['left']-1})
                        s.append({'item': node['item']+1, 'val': node['val']+self.ovalue[node['item']],
                                  'room': node['room']-self.osize[node['item']], 'set': node['set']+[1],
                                  'left':node['left']})
                    else:
                        s.append({'item': node['item']+1, 'val': node['val'],
                                  'room': node['room'], 'set': node['set']+[0],
                                  'left': node['left']})

        return bestnode['val'], np.array(bestnode['set'])[self.inv_perm]