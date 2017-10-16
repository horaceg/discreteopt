#Author : O. Agier & H. Guy

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import random



def read_numbers(data_file):
    input_data_file = open(data_file, 'r')
    input_data = input_data_file.readlines()
    input_data_file.close()

    numbers = np.array([])
    for i_line in range(len(input_data)):
        entries = input_data[i_line].split()
        entries = filter(None, entries) # remove empty entries
        line_numbers = [ float(x) if x.lower != "inf" else float("inf") for x in entries ]
        numbers = np.append(numbers, line_numbers)
    return numbers


def read_data(data_file):
    numbers = read_numbers(data_file)
    cur_entry = 0

    # number of points
    num_points = int(numbers[cur_entry])
    cur_entry += 1

    # get data on the points
    points = np.zeros((num_points, 2))
    for i_point in range(num_points):
        points[i_point, 0] = float(numbers[cur_entry])
        cur_entry += 1
        points[i_point, 1] = float(numbers[cur_entry])
        cur_entry += 1

    return points


def dist(A, B):
    return math.sqrt( (A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) )


def check_tsp_solution( solution, points ):
    num_points = points.shape[0]
    visited_nodes = np.zeros(num_points, dtype=bool)
    path_length = dist( points[solution[0]], points[solution[-1]] )
    for i_point in range(num_points-1):
        visited_nodes[i_point] = True
        path_length += dist( points[solution[i_point]], points[solution[i_point+1]] )

    is_valid_solution = False in visited_nodes
    return is_valid_solution, path_length


def plot_tsp_solution(solution, points):
    is_valid_solution, path_length = check_tsp_solution( solution, points )

    x = np.hstack((points[solution][:,0], points[solution[0]][0]))
    y = np.hstack((points[solution][:,1], points[solution[0]][1]))
    plt.plot(x, y, "o-")
    plt.xlabel('x')
    plt.ylabel('y')
    solution_quality = ['Inconsistent', 'Valid']
    plt.title( '%s solution; %d points; length = %f'%(solution_quality[is_valid_solution], len(points), path_length) )
    plt.show(block=True)

##########################################################################################################

def dist_matrix(coord):
    n = len(coord)
    distances = {}
    for p in range(n-1) :
        for k in range(p+1,n):
            A = coord[p]
            B = coord[k]
            distances[p,k] = dist(A,B)
            distances[k,p] = distances[p,k]  
    return distances


def find_closest(distances,n):
    closest = []
    for p in range(n):
        l = []
        for r in range(n):
            if r != p :
                l = l + [(distances[p,r], r)]
        l.sort()
        closest.append(l)
    return closest
    


def length(cycle, distances):
    l = distances[cycle[-1], cycle[0]]
    for p in range(1,len(cycle)):
        l += distances[cycle[p], cycle[p-1]]
    return l

def cycle_rnd(n):
    cycle = []
    for i in range(n) :
        cycle += [i]
    random.shuffle(cycle)
    return cycle


def closest_node(last, unvisited, distances):
    next_node = unvisited[0]
    dist_min = distances[last, next_node]
    for i in unvisited[1:]:
        if distances[last,i] < dist_min:
            next_node = i
            dist_min = distances[last, next_node]
    return next_node

def closest_neighbor_tour(n, fst, distances):
    unvisited = []
    length = 0.
    for i in range(n) :
        unvisited += [i]
    unvisited.remove(fst)
    last = fst
    cycle = [fst]
    while unvisited != []:
        next_node = closest_node(last, unvisited, distances)
        length += distances[last, next_node]
        cycle.append(next_node)
        unvisited.remove(next_node)
        last = next_node
    length += distances[last,fst]
    return length, cycle

###############################################################################
## Améliorer le cycle##
###############################################################################


def exchange_cost(cycle, i, j, distances):
    n = len(cycle)
    a,b = cycle[i], cycle[(i+1)%n]
    c,d = cycle[j], cycle[(j+1)%n]
    return (distances[a,c] + distances[b,d]) - (distances[a,b]+ distances[c,d])


def exchange(cycle, tinv, i, j):
    n = len(cycle)
    if i >j :
        i,j = j,i
    assert i >=0 and i< j-1 and j<n
    path = cycle[i+1:j+1]
    path.reverse()
    cycle[i+1:j+1] = path
    for k in range(i+1,j+1):
        tinv[cycle[k]] = k
            

def improve(cycle, length, distances, closest):
    n = len(cycle)
    tinv = [0 for i in cycle]
    for p in range(n) :
        tinv[cycle[p]] = p
    
    for p in range(n):
        a,b = cycle[p],cycle[(p+1)%n]
        distance_ab = distances[a,b]
        better = False
        for distance_ac,c in closest[a]:
            if distance_ac >= distance_ab:
                break
            j = tinv[c]
            d = cycle[(j+1)%n]
            distance_cd = distances[c,d]
            distance_bd = distances[b,d]
            red = (distance_ac + distance_bd) - (distance_ab + distance_cd)
            if red <0 :
                exchange(cycle, tinv,p,j)
                length += red
                better = True
                break
        if better:
            continue
        for distance_bd, d in closest[b]:
            if distance_bd >= distance_ab:
                break
            j = tinv[d]-1
            if j == -1:
                j == n-1
            c = cycle[j]
            distance_cd = distances[c,d]
            distance_ac = distances[a,c]
            red = (distance_ac + distance_bd)-(distance_ab + distance_cd)
            if red <0 :
                exchange(cycle, tinv,p,j)
                length += red
                break
    return length, cycle    







#######################################################################################"


def make_dummy_solution(points):
    num_points = points.shape[0]
    solution = np.arange(num_points)
    solution_value = dist( points[0], points[-1] )
    for i_point in range(num_points-1):
        solution_value += dist( points[i_point], points[i_point+1] )
    return solution_value, solution


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1].strip()
    else:
        k = input("test sample number : ")
        path = "task3_test{}.txt".format(k)

    points = read_data(path)
    
    #solution_value, solution = make_dummy_solution(points)
    
    coord = read_data("task3_test1.txt")
    
    n = len(coord)
    dist_mat = dist_matrix(coord)
    clo = find_closest(dist_mat,n)
    tour = closest_neighbor_tour(n, 0,dist_mat)
    solution_value, solution = improve(tour[1],tour[0], dist_mat, clo)
    
    plot_tsp_solution(solution, points)

    print (solution_value)
    print (' '.join(map(str, solution)))

    outpath = path.split('.')[0] + "_solution.txt"
    with open(outpath, 'w') as outfile:
        outfile.write(str(solution_value)+'\n')
        outfile.write(" ".join(map(str, solution)))


