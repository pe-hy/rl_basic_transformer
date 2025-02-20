import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import pickle
from itertools import product

with open("graphs.pkl", "rb") as f:
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


def solve(sample, target_node, graphs, log=None):
    if log is None:
        log = []
    target_chain = 0
    current_state = sample.copy()
    paths_lst = list(nx.all_simple_paths(graphs[target_chain], current_state[target_chain], target_node))
    
    # check if there is a path
    if len(paths_lst) <= 1:
        return None, log
    
    for i, path in enumerate(paths_lst): # 3 4 5
        log.append(f"P {path}")
        # projdu hrany po ceste (3-4), ziskam enabling
        current_state = sample.copy()
        rules = graphs[target_chain].get_edge_data(current_state[target_chain], path[1:][0])

        # pokud neexistuje enabling, projdu edge
        log.append(f"RLS {rules}")
        if rules == {}:
            log.append(f"NR {current_state} TN {current_state[target_chain]}")
            current_state[target_chain] = path[1:][0]
            log.append(f"NRCH {current_state}")
            return solve(current_state, target_node, graphs, log)

        # pokud existuje enabling, tak treba automat 1 ma byt ve stavu 2 a automat 2 ma byt ve stavu 3
        # prvni vyresime prvni pravidlo a pak druhe
        else:
            for rule_set in rules.values():
                log.append(f"RS {rule_set}")
                for rule_group in rule_set:
                    current_state = sample.copy()
                    log.append(f"RG {rule_group}")
                    found = True
                    for rule in sorted(rule_group, key=lambda x: x['automata_id']): # sorted znamena setrizene od nejvice zavisleho po nejmene
                        log.append(f"RULE {rule}")
                        log.append(f"R {current_state} TN {current_state[target_chain]}")
                        current_sub_state, log = solve(current_state[target_chain:], rule['node_id'], graphs, log)
                        log.append(f"RCH {current_sub_state}")
                        if current_sub_state == None:
                            log.append(f"NPATH {current_state} TN {current_state[target_chain]}")
                            found = False
                            break
                        log.append(f"SUB {current_state}")
                        current_state[target_chain:] = current_sub_state
                        log.append(f"SUBCH {current_state}")
                        if found:
                            log.append(f"SUBF {current_state}")
                            break
                    log.append(f"REND {current_state}")
                    current_state[target_chain] = path[1:][0]
                    log.append(f"SUBSOLVED {current_state}")
                    return solve(current_state, target_node, graphs, log)
        return current_state, log


log = []

curstate, log = solve([9, 1], 1, graphs, log)
