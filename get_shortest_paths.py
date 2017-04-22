
import numpy as np
import algos_path as shortpath
import sys

def read_numbers(data_file):
    input_data_file = open(data_file, 'r')
    input_data = input_data_file.readlines()
    input_data_file.close()

    numbers = np.array([])
    for i_line in range(len(input_data)):
        entries = input_data[i_line].split()
        entries = filter(None, entries)  # remove empty entries
        line_numbers = [float(x) if x.lower != "inf" else float("inf") for x in entries]
        numbers = np.append(numbers, line_numbers)
    print("numbers : \n", numbers)
    return numbers


def read_data(data_file):
    numbers = read_numbers(data_file)
    cur_entry = 0

    # number of nodes
    n = int(numbers[cur_entry])
    cur_entry += 1

    # init graph
    neighbors = [None] * n
    weights = [None] * n

    # construct the graph
    for i_node in range(n):
        num_neighbors = int(numbers[cur_entry])
        cur_entry += 1
        cur_neighbors = np.zeros(num_neighbors, dtype = 'int32')
        cur_weights = np.zeros(num_neighbors, dtype = 'float')
        for i_neighbor in range(num_neighbors):
            cur_neighbors[i_neighbor] = int(numbers[cur_entry])
            cur_entry += 1
            cur_weights[i_neighbor] = numbers[cur_entry]
            cur_entry += 1
        neighbors[i_node] = cur_neighbors
        weights[i_node] = cur_weights

    # get pairs of nodes to compute distances
    num_pairs_of_interest = int(numbers[cur_entry])
    cur_entry += 1

    node_pairs = np.zeros( (num_pairs_of_interest, 2), dtype = 'int32' )
    for i_pair in range(num_pairs_of_interest):
        node_pairs[i_pair][0] = int(numbers[cur_entry])
        cur_entry += 1
        node_pairs[i_pair][1] = int(numbers[cur_entry])
        cur_entry += 1

    #print("neighbors :", neighbors, "weights :", weights, "node_pairs", node_pairs, sep='\n')
    return np.array(neighbors), np.array(weights), node_pairs


def make_dummy_solution(neighbors, weights, node_pairs):
    num_pairs = len(node_pairs)
    asw = np.empty(num_pairs)
    asw.fill(float('inf'))
    return asw

def write_solution(result, path) :
    outfile = open(path + "_solution", 'w')
    outfile.write("\n".join(result))
    outfile.close()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1].strip()
    else:
        k = 1 # input("numéro du test ? ")
        path = "task1_test"+str(k)+".txt"

    neighbors, weights, node_pairs = read_data(path)
    ans_bf = shortpath.bellman_ford(node_pairs, neighbors, weights)
    #ans_fw = shortpath.floyd_warshall(node_pairs, neighbors, weights)
    print("Bellman-Ford :", ans_bf, "Floyd-Warshall :, ans_fw", sep='\n')

    outfile_pref = path.split('.')[0] + "_solution"
    with open(outfile_pref + "_bf.txt", 'w') as outfile:
        outfile.write("\n".join(map(str, ans_bf)))
    #with open(outfile_pref + "_fw.txt", 'w') as outfile:
    #    outfile.write("\n".join(map(str, ans_fw)))
    #print('\n'.join(map(str, answer)))