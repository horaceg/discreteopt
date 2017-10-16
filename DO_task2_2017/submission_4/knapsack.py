
import numpy as np
from solver import Solver
import sys

sys.setrecursionlimit(2500)


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

    # number of nodes
    num_items = int(numbers[cur_entry])
    cur_entry += 1
    
    # maximum capacity of the knapsack
    capacity = float(numbers[cur_entry])
    cur_entry += 1
    
    # get data on the items
    value = np.zeros(num_items, dtype = 'float')
    size = np.zeros(num_items, dtype = 'float')
    for i_item in range(num_items):
        value[i_item] = float(numbers[cur_entry])
        cur_entry += 1
        size[i_item] = float(numbers[cur_entry])
        cur_entry += 1
        
    return value, size, capacity


def make_dummy_solution(value, size, capacity):
    num_items = len(value)
    solution_value = 0
    solution_items = np.zeros(num_items, 'int')
    return solution_value, solution_items


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1].strip()
    else:
        k = input("test sample number : ")
        path = "task2_test{}.txt".format(k)

    value, size, capacity = read_data(path)
    print("Nb items :", len(size), "\nCapacity :", capacity, "\nvalue :\n", list(value), "\nsize :\n", list(size))

    knapsack = Solver(value=value, size=size, capacity=capacity)
    solution_value, solution_items = knapsack.solve_it()
    #val_greedy, greedy_items = knapsack.greedy(value, size, capacity)
    print("solution items")
    print(' '.join(map(str, solution_items)))

    outpath = path.split('.')[0] + "_solution.txt"
    with open(outpath, 'w') as outfile:
        outfile.write(str(solution_value)+'\n')
        outfile.write(" ".join(map(str, solution_items)))