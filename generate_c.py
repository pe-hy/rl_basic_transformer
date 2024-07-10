from itertools import combinations, permutations
import random
import pickle
from tqdm import tqdm
def get_star_transitions(first):
    train_core = []
    test_core = []
    # first 4 nodes (1,2,3,4) in the automaton are connected to the center node

    for i in range(1,5):
        for j in range(i,3+i):
            # start, center, end
            # if i = 2 and first = 0, then we have tuples:
            # (2,5,8),(2,5,9),(2,5,6)
            train_core.append((first+i,first+5,first+6+j%4))
        # for test we would have (2,5,7)
        test_core.append((first+i,first+5,first+i+5))
    return train_core,test_core

train_core = {}
test_core = {}

for i in range(0,3):
    train,test = get_star_transitions(i*9)
    train_core[i+1]=train
    test_core[i+1]= test

automaton2states = {}

for i in range(0,9*3):
    automaton2states[i+1] = (i*2+1,i*2+2)

def gen_pair_distractors_c(automaton2states):
    pair_distractors = {}
    for k, v in tqdm(automaton2states.items()):
        off, on = v 
        tmp = set(range(1,55)) - set([off, on])
        combs = list(combinations(tmp, 5))
        samples = random.sample(combs, k=100)
        pair_distractors[k] = samples
    return pair_distractors


def gen_c(automaton2states, pair_distractors):
    train_data = []
    test_data = []
    num_permutations_train = 50
    num_permutations_test = 10
    for k, v in tqdm(automaton2states.items()):
        distractors = list(pair_distractors[k])
        n_train = int(len(distractors)*0.9)
        for i in distractors[:n_train]:
            i = list(i)
            # prvních 6 stavů - permutace
            permutations_list = list(permutations(i + [v[0]]))
            # výběr 20
            selected_permutations = random.sample(permutations_list, num_permutations_train)
            # pro každou permutaci
            for perm in selected_permutations:
                # do listu dát tuto permutaci a výstupní stav
                input_list = list(perm) + [v[1]]
                # do out hodit index cílového stavu
                orig_idx = input_list.index(v[0])
                out = orig_idx+55
                # dát do dictu
                train_data.append({'input': input_list, 'out': out, 'orig_idx': orig_idx})
        for j in distractors[n_train:]:
            j = list(j)
            # prvních 6 stavů - permutace
            permutations_list = list(permutations(j + [v[0]]))
            # výběr 20
            selected_permutations = random.sample(permutations_list, num_permutations_test)
            # pro každou permutaci
            for perm in selected_permutations:
                # do listu dát tuto permutaci a výstupní stav
                input_list = list(perm) + [v[1]]
                # do out hodit index cílového stavu
                orig_idx = input_list.index(v[0])
                out = orig_idx+55
                # dát do dictu
                test_data.append({'input': input_list, 'out': out, 'orig_idx': orig_idx})
    return train_data, test_data

pair_distractors = gen_pair_distractors_c(automaton2states)

train_data_c, test_data_c = gen_c(automaton2states, pair_distractors)

data = {"train": train_data_c, "test": test_data_c}

with open ("data_c.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved: ", "data_c.pkl")



