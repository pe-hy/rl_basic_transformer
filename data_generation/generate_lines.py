import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import pickle
from itertools import product
import os

with open(os.path.join("data_generation", "graphs.pkl"), "rb") as f:
    graphs = pickle.load(f)


def get_vectors(chains):
    states_per_chain = []
    for chain_id in range(len(chains)):
        states_per_chain.append(list(chains[chain_id].nodes()))

    return list(product(*states_per_chain))


vecs = get_vectors(graphs)


def get_sample(vector, graphs):
    num_states = len(graphs[0].nodes())
    vec = list(vector)  # [1,2,3,4]
    for i in range(num_states):
        if vec[0] != i:
            yield vec, i


indexes = list(range(len(vecs)))
samples = []
 
def filter_unique_paths(paths_lst):
    if not paths_lst:
        return []
    
    # Keep track of which paths should be excluded
    paths_to_exclude = set()
    
    # For each path, check if it contains any shorter path as a subsequence
    for path in paths_lst:
        for other_path in paths_lst:
            # Skip if same path or if other_path is not shorter
            if path == other_path or len(other_path) >= len(path):
                continue
                
            # Check if other_path is a subsequence of path
            j, k = 0, 0  # j for other_path, k for path
            while j < len(other_path) and k < len(path):
                if other_path[j] == path[k]:
                    j += 1
                k += 1
                
            # If other_path is a subsequence of path, exclude path
            if j == len(other_path):
                paths_to_exclude.add(tuple(path))
                break
    
    # Return all paths except those that contain shorter paths as subsequences
    return [path for path in paths_lst if tuple(path) not in paths_to_exclude]

def solve(curr_states, target_node, graphs, orig_state_len, log=[], depth=0):
    '''
    Solves the problem of reaching target node from current state in graphs by returning the resulting state and log of steps.
    Parameters:
        curr_states: list of current states of automatas; LIST of INTEGERS
        target_node: target node; INTEGER
        graphs: list of graphs; LIST of networkx graphs
        log: list of log messages; LIST of STRINGS
        depth: depth of recursion; INTEGER
    Returns:
        out: states of automatas after solving or None if no solution; LIST of INTEGERS/None
            out: log: list of log messages; LIST of STRINGS
    '''
    indent = ' '*depth*2
    # log.append(f"{'-'*50}")
    curr_states_cpy = curr_states.copy()
    graphs_cpy = graphs.copy()
    log.append(f"{indent}S {curr_states_cpy} TN {target_node}")
    # get all paths from current state to target node
    paths_lst = list(nx.all_simple_paths(graphs_cpy[0], curr_states_cpy[0], target_node))
    # check if a shorter path isn't already in a different longer path
    paths_lst = filter_unique_paths(paths_lst)
    log.append(f"{indent}PL {paths_lst}")
    
    # check if there is a path
    if len(paths_lst) == 0:
        log.append(f"{indent}NOPATH {curr_states} TN {target_node}")
        return None, log
    
    # check if current node equals target node
    if curr_states_cpy[0] == target_node:
        log.append(f"{indent}REND {curr_states_cpy}")
        return curr_states_cpy, log
    
    # path or paths exist and we are not in target node
    for path in paths_lst:
        log.append(f"{indent}P {path}")
        # projdu hrany po ceste (3-4), ziskam enabling
        curr_states_cpy = curr_states.copy()
        graphs_cpy = graphs.copy()
        rules = graphs_cpy[0].get_edge_data(curr_states_cpy[0], path[1])
        # pokud neexistuje enabling, projdu edge
        log.append(f"{indent}RLS {rules}")
        if rules == {}:
            log.append(f"{indent}NOR {curr_states_cpy} TN {target_node}")
            # pokud má path jeden node, tak je to konec
            curr_states_cpy[0] = path[1]
            log.append(f"{indent}NORCH {curr_states_cpy}")

            return solve(curr_states_cpy, target_node, graphs_cpy, orig_state_len, log, depth+1)

        # pokud existuje enabling, tak treba automat 1 ma byt ve stavu 2 a automat 2 ma byt ve stavu 3
        # prvni vyresime prvni pravidlo a pak druhe
        else:
            for rule_set in rules['enabling']:
                log.append(f"{indent}RS {rule_set}")

                curr_states_cpy = curr_states.copy()
                graphs_cpy = graphs.copy()
                found = True

                for rule in sorted(rule_set, key=lambda x: x['automata_id']): # sorted znamena setrizene od nejvice zavisleho po nejmene
                    log.append(f"{indent}RULE {rule}")
                    idx = max(rule['automata_id'] - (orig_state_len - len(curr_states_cpy)), 0)
                    log.append(f"{indent}IDX {idx} LEN {len(curr_states_cpy)}")
                    current_sub_state, log = solve(curr_states_cpy[idx:], rule['node_id'], graphs_cpy[idx:], orig_state_len, log, depth+1)

                    log.append(f"{indent}RCH {current_sub_state}")
                    if current_sub_state == None:
                        found = False
                        break
                    log.append(f"{indent}SUB {curr_states_cpy}")
                    curr_states_cpy[idx:] = current_sub_state
                    log.append(f"{indent}SUBCH {curr_states_cpy}")

                if found:
                    log.append(f"{indent}SUBF {curr_states_cpy}")
                    curr_states_cpy[0] = path[1]
                    log.append(f"{indent}SUBSOLVED {curr_states_cpy}")
                    break
                else:
                    continue
            
            if found:
                return solve(curr_states_cpy, target_node, graphs_cpy, orig_state_len, log, depth+1)
            else:
                continue
    log.append(f"{indent}UNSOL {curr_states_cpy}")            
    return None, log


# # in nx graph change direction from node 4 to 1 into 1 to 4 in graph with index 2 but save enabling rules on edge
# edge_data = graphs[2].get_edge_data(4, 1)

# # Remove the edge from 4 to 1
# graphs[2].remove_edge(4, 1)

# # Add the new edge from 1 to 4 with the same attributes
# graphs[2].add_edge(1, 4, **edge_data)

log = []
import pprint as pprint
sample = [3, 2, 2]
target_node = 1
curstate, log = solve(sample, target_node, list(graphs.values()), len(sample), log)

pprint.pprint(log)
print(curstate)

# Assuming 'graphs' is a list of NetworkX graphs
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first graph on the first subplot
nx.draw(graphs[0], with_labels=True, node_color='lightblue', arrows=True, ax=axs[0])
axs[0].set_title('Graph 1')

# Plot the second graph on the second subplot
nx.draw(graphs[1], with_labels=True, node_color='lightgreen', arrows=True, ax=axs[1])
axs[1].set_title('Graph 2')

# nx.draw(graphs[2], with_labels=True, node_color='orange', arrows=True, ax=axs[2])
# axs[2].set_title('Graph 3')

# Display the plots
plt.tight_layout()
plt.show()