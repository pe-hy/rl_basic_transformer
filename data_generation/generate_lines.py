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


def solve(curr_states, target_node, graphs, log=[], depth=0):
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
    log.append(f"{indent}S {curr_states_cpy} TN {target_node}")
    # get all paths from current state to target node
    paths_lst = list(nx.all_simple_paths(graphs[0], curr_states_cpy[0], target_node))
    
    # check if there is a path
    if len(paths_lst) == 0:
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
        rules = graphs[0].get_edge_data(curr_states_cpy[0], path[1])
        # pokud neexistuje enabling, projdu edge
        log.append(f"{indent}RLS {rules}")
        if rules == {}:
            log.append(f"{indent}NR {curr_states_cpy} TN {target_node}")
            # pokud má path jeden node, tak je to konec
            curr_states_cpy[0] = path[1]
            log.append(f"{indent}NRCH {curr_states_cpy}")

            return solve(curr_states_cpy, target_node, graphs, log, depth+1)

        # pokud existuje enabling, tak treba automat 1 ma byt ve stavu 2 a automat 2 ma byt ve stavu 3
        # prvni vyresime prvni pravidlo a pak druhe
        else:
            for rule_set in rules['enabling']:
                log.append(f"{indent}RS {rule_set}")

                curr_states_cpy = curr_states.copy()
                found = True

                for rule in sorted(rule_set, key=lambda x: x['automata_id']): # sorted znamena setrizene od nejvice zavisleho po nejmene
                    log.append(f"{indent}RULE {rule}")

                    current_sub_state, log = solve(curr_states_cpy[rule['automata_id']:], rule['node_id'], graphs[rule['automata_id']:], log, depth+1)

                    log.append(f"{indent}RCH {current_sub_state}")
                    if current_sub_state == None:
                        log.append(f"{indent}NPATH {curr_states_cpy[rule['automata_id']:]} TN {rule['node_id']}")
                        found = False
                        break
                    log.append(f"{indent}SUB {curr_states_cpy}")
                    curr_states_cpy[rule['automata_id']:] = current_sub_state
                    log.append(f"{indent}SUBCH {curr_states_cpy}")

                if found:
                    log.append(f"{indent}SUBF {curr_states_cpy}")
                    curr_states_cpy[0] = path[1]
                    log.append(f"{indent}SUBSOLVED {curr_states_cpy}")
                    break
                else:
                    continue
            
            if found:
                return solve(curr_states_cpy, target_node, graphs, log, depth+1)
            else:
                continue

    return None, log


log = []
import pprint as pprint
curstate, log = solve([9, 1], 1, list(graphs.values()), log, 0)

pprint.pprint(log)
print(curstate)

# # Assuming 'graphs' is a list of NetworkX graphs
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # Plot the first graph on the first subplot
# nx.draw(graphs[0], with_labels=True, node_color='lightblue', arrows=True, ax=axs[0])
# axs[0].set_title('Graph 1')

# # Plot the second graph on the second subplot
# nx.draw(graphs[1], with_labels=True, node_color='lightgreen', arrows=True, ax=axs[1])
# axs[1].set_title('Graph 2')

# # Display the plots
# plt.tight_layout()
# plt.show()