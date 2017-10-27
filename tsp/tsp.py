
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import solver


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
    print(numbers)
    return numbers


def read_sol(data_file, num_points):
    numbers = read_numbers(data_file)
    cur_entry = 0
    mycost = float(numbers[cur_entry])
    cur_entry += 1
    mytour = []
    for i in range(num_points):
        mytour.append(int(numbers[cur_entry]))
        cur_entry += 1
    print(mycost)
    print(mytour)
    return mycost, mytour


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
    visited_nodes = np.zeros(num_points-1, dtype=bool)
    path_length = dist( points[solution[0]], points[solution[-1]] )
    for i_point in range(num_points-1):
        visited_nodes[i_point] = True
        path_length += dist( points[solution[i_point]], points[solution[i_point+1]] )

    is_valid_solution = len(solution) == len(points) and False not in visited_nodes
    return is_valid_solution, path_length


def plot_tsp_solution(solution, points):
    is_valid_solution, path_length = check_tsp_solution( solution, points )

    fig = plt.figure()
    x = np.hstack((points[solution][:,0], points[solution[0]][0]))
    y = np.hstack((points[solution][:,1], points[solution[0]][1]))
    plt.plot(x, y, "o-")
    plt.xlabel('x')
    plt.ylabel('y')
    solution_quality = ['Inconsistent', 'Valid']
    plt.title( '%s solution; %d points; length = %f'
               %(solution_quality[is_valid_solution], len(points), path_length) )
    plt.show(block=True)
    return fig


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
    #plt.scatter(points.T[0], points.T[1])
    #plt.show()
    # solution_value, solution = make_dummy_solution(points)

    mytsp = solver.TSP(points=points)
    #solution_value, solution = mytsp.nearest_neighbor()
    if len(points) <= 51:
        solution_value, solution = mytsp.localsearch_multistart(20, opt=3)
        cost, tour = mytsp.localsearch_multistart(50, opt=2)
        if cost < solution_value:
            solution_value, solution = cost, tour
    elif len(points) <= 100:
        solution_value, solution = mytsp.localsearch_multistart(3, opt=2)
        for k in range(3):
            cost, tour = mytsp.localsearch_multistart(3, opt=2)
            cost, tour = mytsp.three_opt(tour, cost)
            if cost < solution_value:
                solution_value, solution = cost, tour
    elif len(points) <= 200:
        solution_value, solution = mytsp.localsearch_multistart(3, opt=2)
        solution_value, solution = mytsp.simu_two_opt(solution, solution_value, max_time=5., tempe=100)
    elif len(points) <= 574:
        solution_value, solution = mytsp.localsearch_multistart(3, opt=2)
        solution_value, solution = mytsp.simu_two_opt(solution, solution_value, max_time=5., tempe=10)
        solution_value, solution = mytsp.two_opt(solution, solution_value)
    elif len(points) <= 1889:
        # here, running time approx 25 minutes
        solution_value, solution = read_sol(path.split('.')[0] + "_solution.txt", len(points))
        #solution_value, solution = mytsp.nearest_neighbor()
        solution_value, solution = mytsp.simu_two_opt(solution, solution_value, max_time=1., tempe=10.)
        solution_value, solution = mytsp.two_opt(solution, solution_value)
        #solution_value, solution = mytsp.fast_three_opt(solution, solution_value, max_time=5.)
        solution_value, solution = mytsp.simu_two_opt(solution, solution_value, max_time=1., tempe=10.)
        solution_value, solution = mytsp.two_opt(solution, solution_value)
    else:
        # solution_value, solution = read_sol(path.split('.')[0] + "_solution.txt", len(points))
        solution_value, solution = mytsp.nearest_neighbor()
        solution_value, solution = mytsp.two_opt(solution, solution_value, max_time=30.)

    fig = plot_tsp_solution(solution, points)

    print(solution_value)
    print(' '.join(map(str, solution)))

    outpath = path.split('.')[0] + "_solution.txt"
    outpath_fig = path.split('.')[0] + "_solution.png"
    fig.savefig(outpath_fig)
    with open(outpath, 'w') as outfile:
        outfile.write(str(solution_value) + '\n')
        outfile.write(" ".join(map(str, solution)))
