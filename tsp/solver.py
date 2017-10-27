import numpy as np
import math, random
import matplotlib.pyplot as plt
from functools import lru_cache
import time


class TSP:

    def __init__(self, points):
        self.points = points
        self.size = len(points)
        self.ordered = np.arange(self.size)  # initial order of the nodes

    @staticmethod
    def dist(p1, p2):
        # compute distance by points (coordinates)
        return math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )

    @lru_cache(maxsize=100000)
    def dist_nb(self, i, j):
        # compute distance by initial IDs of the nodes
        # print("Computing distances btw nodes {} and {} ".format(i,j))
        return math.sqrt((self.points[i,0] - self.points[j,0])**2 + (self.points[i,1] - self.points[j,1])**2)

    def check_solution(self, solution):
        visited_nodes = np.zeros(self.size, dtype=bool)
        path_length = self.dist_nb(solution[0], solution[-1])
        visited_nodes[self.size - 1] = True
        for i in range(self.size - 1):
            visited_nodes[i] = True
            path_length += self.dist_nb(solution[i], solution[i+1])

        is_valid_solution = len(solution) == self.size and False not in visited_nodes
        return is_valid_solution, path_length

    def plot_solution(self, solution):
        is_valid_solution, path_length = self.check_solution(solution)

        x = np.hstack((self.points[solution][:, 0], self.points[solution[0]][0]))
        y = np.hstack((self.points[solution][:, 1], self.points[solution[0]][1]))
        plt.plot(x, y, "o-")
        plt.xlabel('x')
        plt.ylabel('y')
        solution_quality = ['Inconsistent', 'Valid']
        plt.title(
            '%s solution; %d points; length = %f' % (solution_quality[is_valid_solution], self.size, path_length))
        plt.show(block=True)

    def nearest_neighbor(self, start=None):
        if start is None:
            current = random.randint(0, self.size - 1)
        else:
            current = start
        left = np.ones(self.size, dtype=bool)  # which nodes have not been visited yet
        left[current] = False
        visited, cost = [current], 0
        while len(visited) < self.size:
            dist_cur = np.vectorize(lambda other: self.dist_nb(current, other))
            i = np.argmin(dist_cur(self.ordered[left]))
            nextp = self.ordered[left][i]
            cost += self.dist_nb(current, nextp)
            visited.append(nextp)
            left[nextp] = False
            current = nextp

        cost += self.dist_nb(visited[0], visited[-1])
        return cost, visited

    def random_path(self):
        remaining = list(range(self.size))
        tour = []
        for i in range(self.size):
            node = random.choice(remaining)
            remaining.remove(node)
            tour.append(node)

        _, cost = self.check_solution(tour)
        return cost, tour

    def iter_nn(self, nb_iter):
        cost, visited = self.nearest_neighbor(start=0)
        if nb_iter >= self.size:
            for i in range(1, self.size):
                cost2, visited2 = self.nearest_neighbor(start=i)
                print("computed nn with start ", i)
                if cost2 < cost:
                    cost = cost2
                    visited = visited2
        else:
            for _ in range(1, nb_iter):
                cost2, visited2 = self.nearest_neighbor()
                print("New nn computed")
                if cost2 < cost:
                    cost = cost2
                    visited = visited2
                    print("Length is now ", cost)

        return cost, visited

    def flip_cost(self, i, j, tour):
        # cost of switching arcs (i,i+1) and (j,j+1) by (i,j) and (i+1,j+1) in a tour
        if i in [(j-1)%self.size, j, (j+1)%self.size]:
            return 0
        a, b = tour[i], tour[(i + 1) % self.size]
        c, d = tour[j], tour[(j + 1) % self.size]
        delta = self.dist_nb(a, c) + self.dist_nb(b, d) - self.dist_nb(c, d) - self.dist_nb(a, b)
        return delta

    def flip(self, i, k, tour, cost):
        if i > k: i, k = k, i
        #print("flipping {}-{} & {}-{} | improvement :".format(tour[i], tour[(i+1)%self.size],
        #                                                      tour[k], tour[(k+1)%self.size]), self.flip_cost(i,k,tour))
        if i in [k-1, k]:
            return cost, tour
        else:
            j, l = i+1, k+1
            newtour = tour[:i+1] + tour[k:j-1:-1] + tour[l:]
            newcost = cost + self.flip_cost(i, k, tour)
            return newcost, newtour

    def two_opt_step(self, tour, cost, sort=True):
        newcost, newtour = cost, tour
        if sort:
            sorted_arcs = np.argsort([self.dist_nb(tour[i], tour[(i+1) % self.size]) for i in range(self.size)])
        else:
            sorted_arcs = [random.randint(0, self.size-1) for _ in range(1000)]
        for ii, i in enumerate(sorted_arcs):
            for j in sorted_arcs[ii:]:
                delta = self.flip_cost(i, j, newtour)
                if delta <= -1:
                    newcost, newtour = self.flip(i, j, newtour, newcost)
                    return newcost, newtour

        return newcost, newtour

    def two_opt(self, start, cost, max_time=60.):
        # max_time : maximum minutes elapsed to stop the search.
        # start = initial tour
        i, start_date = 0, time.time()
        best_cost, best_tour = cost, start
        new_cost, new_tour = self.two_opt_step(start, cost)
        while new_cost < best_cost and (time.time()-start_date)/60. < max_time:
            i += 1
            print("2-opt local search in progress, step {}. Path improved by {}. Length is now {}"
                  .format(i, -int(best_cost - new_cost), int(new_cost)))
            #print("cost improvement : {} to {}".format(cost, new_cost))
            best_tour, best_cost = new_tour, new_cost
            sort = random.random() < 1.
            new_cost, new_tour = self.two_opt_step(best_tour, best_cost, sort)
        if (time.time() - start_date) / 60. > max_time:
            print("Maximum time {} minutes elapsed".format(max_time))
        return best_cost, best_tour

    def localsearch_multistart(self, nb_iter, opt=2):
        best_cost, best_tour = self.nearest_neighbor()
        assert opt in [2,3]
        for i in range(nb_iter):
            print("\n========================Beginning {}th start========================".format(i))
            c, t = self.nearest_neighbor()  # random start NN heuristic
            if opt == 2:
                cost, tour = self.two_opt(t, c)
            elif opt == 3:
                cost, tour = self.three_opt(t, c)
            else:
                print("invalid argument")

            if cost < best_cost:
                best_cost = cost
                best_tour = tour
                print("improving in iteration {}. Path length is now approx {}".format(i, int(cost)))
        return best_cost, best_tour

    def three_opt_move(self, a, c, e, tour, cost):
        # a, c, e : 3 edges defined by their first node
        edges = [a,c,e]
        a, c, e = sorted([a, c, e])
        bb, dd, ff = (a + 1)%self.size, (c + 1)%self.size, (e + 1)%self.size  # indices to compute distances
        b, d, f = a+1, c+1, e+1  # indices to slice the tour

        old_cost = self.dist_nb(tour[a],tour[bb]) + self.dist_nb(tour[c],tour[dd]) + self.dist_nb(tour[e],tour[ff])

        choices = np.zeros(7, dtype=float)
        choices[0] = self.flip_cost(c,e,tour)
        choices[1] = self.flip_cost(a,c,tour)
        choices[2] = self.flip_cost(a,e,tour)
        if len(edges) == len(set(edges)):
            # if two nodes are identical among a,c and e, any possible move is actually a 2-opt
            choices[3] = self.dist_nb(tour[a],tour[dd]) + self.dist_nb(tour[e],tour[bb]) + self.dist_nb(tour[c],tour[ff]) - old_cost
            choices[4] = self.dist_nb(tour[a],tour[dd]) + self.dist_nb(tour[e],tour[c]) + self.dist_nb(tour[bb],tour[ff]) - old_cost
            choices[5] = self.dist_nb(tour[a],tour[e]) + self.dist_nb(tour[dd],tour[bb]) + self.dist_nb(tour[c],tour[ff]) - old_cost
            choices[6] = self.dist_nb(tour[a],tour[c]) + self.dist_nb(tour[bb],tour[e]) + self.dist_nb(tour[dd],tour[ff]) - old_cost
        amini = np.argmin(choices)
        mini = choices[amini]

        if mini <= -1. :
            if amini <= 2:
                if amini == 0:
                    newcost, sol = self.flip(c,e,tour,cost)
                elif amini == 1:
                    newcost, sol = self.flip(a,c,tour,cost)
                elif amini == 2:
                    newcost, sol = self.flip(a,e,tour,cost)
                print("2-opt move")
            else:
                if amini == 3:
                    sol = tour[:a+1] + tour[d:e+1] + tour[b:c+1] + tour[f:]  # 3-opt
                elif amini == 4:
                    sol = tour[:a+1] + tour[d:e+1] + tour[c:b-1:-1] + tour[f:]  # 3-opt
                elif amini == 5:
                    sol = tour[:a+1] + tour[e:d-1:-1] + tour[b:c+1] + tour[f:]  # 3-opt
                elif amini == 6:
                    sol = tour[:a+1] + tour[c:b-1:-1] + tour[e:d-1:-1] + tour[f:]  # 3-opt
                newcost = cost + mini
                print("3-opt move")
            #print("2|3-opt choices : \n{}\n min {}".format(choices, mini))
            print("Length improved by {}. \nNew length is approx. {}".format(int(10.*mini)/10., int(newcost)))
        else:
            sol = tour
            newcost = cost

        return newcost, sol

    def three_opt_step(self, tour, cost, sort=True):
        best_cost, best_tour = cost, tour
        if sort:
            sorted_arcs = np.argsort([self.dist_nb(tour[i], tour[(i+1) % self.size]) for i in range(self.size)])[::-1]
        else:
            sample = [random.randint(0, self.size - 1) for _ in range(200)]
            sorted_arcs = np.argsort([self.dist_nb(tour[i], tour[(i+1) % self.size]) for i in sample])[::-1]
        for ii, i in enumerate(sorted_arcs):
            for jj, j in enumerate(sorted_arcs[ii:]):
                for kk, k in enumerate(sorted_arcs[ii+jj:]):
                    newcost, newtour = self.three_opt_move(i,j,k,best_tour,best_cost)
                    if newcost < best_cost:
                        print("Improvement done by changing {}, {} and {} sorted by decreasing arc length"
                              .format(ii,ii+jj,ii+jj+kk))
                        return newcost, newtour

        return best_cost, best_tour

    def three_opt(self, start, cost, max_time=30.):
        new_cost, new_tour = self.three_opt_step(start, cost)
        best_cost, best_tour = cost, np.copy(start)
        i = 0
        start_date = time.time()
        while new_cost < best_cost and (start_date-time.time())/60. < max_time:
            i += 1
            print("\n ------3-opt local search in progress, step {}. Path improved by approx {}."
                  " Length is now approx {}\n".format(i, -int(best_cost - new_cost), int(new_cost)))
            best_tour, best_cost = new_tour, new_cost
            new_cost, new_tour = self.three_opt_step(best_tour, best_cost)
        if (start_date-time.time())/60. > max_time:
            print("Maximum time elapsed.")
        return best_cost, best_tour

    def fast_three_opt(self, start, cost, max_time=30.):
        new_cost, new_tour = self.three_opt_step(start, cost, sort=False)
        best_cost, best_tour = cost, np.copy(start)
        i = 0
        start_date = time.time()
        while (time.time()-start_date)/60. < max_time :
            i += 1
            print("\n ------3-opt local search in progress, step {}. Path improved by approx {}."
                  " Length is now approx {}\n".format(i, int(best_cost - new_cost), int(new_cost)))
            best_tour, best_cost = new_tour, new_cost
            new_cost, new_tour = self.three_opt_step(best_tour, best_cost, sort=False)
        return best_cost, best_tour

    def simu_two_opt_step(self, tour, cost, temp, tabu):
        newcost, newtour = cost, tour
        # simulated annealing
        #sorted_arcs = list(range(self.size))
        #random.shuffle(sorted_arcs)
        sorted_arcs = np.argsort([self.dist_nb(tour[i], tour[(i+1) % self.size]) for i in range(self.size)])[::-1]
        for ii, i in enumerate(sorted_arcs):
            for j in sorted_arcs[ii:]:
                delta = self.flip_cost(i, j, tour)
                if delta <= -1:
                    seq = sorted([tour[i], tour[j], tour[(i+1)%self.size], tour[(j+1)%self.size]])
                    if seq not in tabu:
                        newcost, newtour = self.flip(i, j, tour, cost)
                        return newcost, newtour, tabu, temp*0.99
                    else:
                        pass
                        #print("tabu")

                elif delta > 0:
                    p = math.exp(-delta/temp)
                    x = random.random()
                    if x < p:
                        seq = sorted([tour[i], tour[j], tour[(i + 1) % self.size], tour[(j + 1) % self.size]])
                        if seq not in tabu:
                            newcost, newtour = self.flip(i, j, tour, cost)
                            return newcost, newtour, tabu+[seq], temp*0.99
        if len(tabu) > 30:
            return newcost, newtour, [], temp * 0.8
        return newcost, newtour, tabu, temp*0.8

    def simu_two_opt(self, tour, cost, max_time=30., tempe=100):
        # simulated annealing
        i, start_date = 0, time.time()
        best_cost, best_tour = cost, tour
        temp, tabu = tempe, []
        new_cost, new_tour, tabu, temp = self.simu_two_opt_step(tour, cost, temp, tabu)
        while (time.time()-start_date)/60. < max_time and temp >= 0.1:
            i += 1
            oc = new_cost
            if new_cost < best_cost:
                best_tour, best_cost = new_tour, new_cost
            new_cost, new_tour, tabu, temp = self.simu_two_opt_step(new_tour, new_cost, temp, tabu)
            print("2-opt simulated annealing local search in progress, step {}.\n Path changed by {}. Length is {}."
                  "Best so far is {}. Temp is {}.".format(i, int(new_cost - oc), int(new_cost), int(best_cost), temp))
        if (time.time() - start_date) / 60. < max_time :
            print("Maximum time {} minutes elapsed".format(max_time))
        if temp < 0.1:
            new_cost, new_tour = self.two_opt(new_tour, new_cost, max_time= max_time - (time.time()-start_date)/60.)
            if new_cost < best_cost:
                best_tour, best_cost = new_tour, new_cost
        return best_cost, best_tour

    def chenille(self):
        # first we sort the nodes by x's
        mask = np.argsort(self.points.T[0])
        opoints = self.points[mask]
        rev = False
        uniquex, first_xs, counts_x = np.unique(opoints.T[0], return_index=True, return_counts=True)

        # sorting the nodes by lexicographic order [x, y]
        for n, i in enumerate(first_xs):
            if n == len(first_xs) - 1:
                sorted_chunk = np.argsort(opoints.T[1][i:])
                mask[i:][sorted_chunk] = mask[i:]
            else:
                sorted_chunk = np.argsort(opoints.T[1][i:first_xs[n + 1]])
                mask[i:first_xs[n + 1]][sorted_chunk] = mask[i:first_xs[n + 1]]

        cost = 0
        count = len(mask[:first_xs[1]])
        tour = list(mask[:first_xs[1]])
        for n, i in enumerate(first_xs[1:]):
            m = n + 1
            if m < len(first_xs) - 1:
                j = first_xs[m + 1] - 1
            else:
                j = self.size - 1
            p1 = mask[i - 1]
            if rev:
                chunk = mask[j:i - 1:-1]
            else:
                chunk = mask[i:j + 1]
            for k in chunk:
                p2 = k
                tour.append(k)
                cost += self.dist_nb(p1, p2)
                p1 = p2
                count += 1
            rev = not rev
            print(len(tour), count, len(mask[:j]))
        cost += self.dist_nb(tour[0], tour[-1])
        print(len(tour), self.size)
        print(count, len(first_xs))
        return cost, tour

    def chenille2(self):
        mask = np.argsort(self.points.T[0])
        opoints = self.points[mask]
        uniquex, first_xs, counts_x = np.unique(opoints.T[0], return_index=True, return_counts=True)
        for n, i in enumerate(first_xs):
            if n == len(first_xs) - 1:
                sorted_chunk = np.argsort(opoints.T[1][i:])
                mask[i:][sorted_chunk] = mask[i:]
            else:
                sorted_chunk = np.argsort(opoints.T[1][i:first_xs[n + 1]])
                mask[i:first_xs[n + 1]][sorted_chunk] = mask[i:first_xs[n + 1]]
                print(mask)
        valid, cost = self.check_solution(mask)
        print(valid, len(mask))
        return cost, mask