from itertools import combinations, permutations
import random
import pickle
from tqdm import tqdm

def merge_cases(original_dict):
    merged_dict = {'train': [], 'test': []}
    
    for key in original_dict:
        if 'train' in key:
            merged_dict['train'].extend(original_dict[key])
        elif 'test' in key:
            merged_dict['test'].extend(original_dict[key])
    
    return merged_dict

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

tmp = [automaton2states[5], automaton2states[14], automaton2states[23]]
middle_states = []
for tup in tmp:
    middle_states.append(tup[0])
    middle_states.append(tup[1])

target_automatons = [6,7,8,9,15,16,17,18,24,25,26,27]
target_states_on = []
target_states_off = []

for index in target_automatons:
    value_tuple = automaton2states[index]
    target_states_off.append(value_tuple[0])
    target_states_on.append(value_tuple[1])

def get_middle_state(k):
    if k in [6,7,8,9]:
        return automaton2states[5][0]
    elif k in [15,16,17,18]:
        return automaton2states[14][0]
    elif k in [24,25,26,27]:
        return automaton2states[23][0] 
    return None 

def gen_b(pair_distractors, target_automatons, is_case_2=False):
    train_data = []
    test_data = []
    num_permutations_train = 50
    num_permutations_test = 20
    for k, v in automaton2states.items():
        if k in target_automatons:
            distractors = list(pair_distractors[k])
            n_train = int(len(distractors)*0.9)
            
            for i in distractors[:n_train]:
                i = list(i)
                permutations_list = list(permutations(i + [v[0]]))
                selected_permutations = random.sample(permutations_list, num_permutations_train)
                for perm in selected_permutations:
                    input_list = list(perm) + [v[1]]
                    
                    if is_case_2:
                        middle_state = get_middle_state(k)
                        orig_idx = input_list.index(middle_state)
                    else:  # Case 1 and 3
                        orig_idx = input_list.index(v[0])
                    
                    out = orig_idx + 55
                    train_data.append({'input': input_list, 'out': out, 'orig_idx': orig_idx})
            
            for j in distractors[n_train:]:
                j = list(j)
                permutations_list = list(permutations(j + [v[0]]))
                selected_permutations = random.sample(permutations_list, num_permutations_test)
                for perm in selected_permutations:
                    input_list = list(perm) + [v[1]]
                    
                    if is_case_2:
                        middle_state = get_middle_state(k)
                        orig_idx = input_list.index(middle_state)
                    else:  # Case 1 and 3
                        orig_idx = input_list.index(v[0])
                    
                    out = orig_idx + 55
                    test_data.append({'input': input_list, 'out': out, 'orig_idx': orig_idx})
    return train_data, test_data

def gen_pair_distractors_b_case_1(automaton2states):
    pair_distractors = {}
    for k, v in tqdm(automaton2states.items()):
        off, on = v
        tmp = set(range(1, 55)) - set([off, on])
        combs = list(combinations(tmp, 5))
        
        if k in [6,7,8,9]:
            combs = [comb for comb in combs if not any(state in comb for state in list(automaton2states[5]))]
        if k in [15,16,17,18]:
            combs = [comb for comb in combs if not any(state in comb for state in list(automaton2states[14]))]
        if k in [24,25,26,27]:
            combs = [comb for comb in combs if not any(state in comb for state in list(automaton2states[23]))]
    
        samples = random.sample(combs, k=100)
        pair_distractors[k] = samples
    return pair_distractors

def gen_pair_distractors_b_case_2(automaton2states):
    pair_distractors = {}
    for k, v in tqdm(automaton2states.items()):
        off, on = v
        tmp = set(range(1, 55)) - set([off, on])
        combs = list(combinations(tmp, 5))
        
        if k in [6,7,8,9]:
            combs = [comb for comb in combs if automaton2states[5][0] in comb and automaton2states[5][1] not in comb]
        elif k in [15,16,17,18]:
            combs = [comb for comb in combs if automaton2states[14][0] in comb and automaton2states[14][1] not in comb]
        elif k in [24,25,26,27]:
            combs = [comb for comb in combs if automaton2states[23][0] in comb and automaton2states[23][1] not in comb]
    
        samples = random.sample(combs, k=min(100, len(combs)))
        pair_distractors[k] = samples
    return pair_distractors

def gen_pair_distractors_b_case_3(automaton2states):
    pair_distractors = {}
    for k, v in tqdm(automaton2states.items()):
        off, on = v
        tmp = set(range(1, 55)) - set([off, on])
        combs = list(combinations(tmp, 5))
        
        if k in [6,7,8,9]:
            combs = [comb for comb in combs if automaton2states[5][1] in comb and automaton2states[5][0] not in comb]
        elif k in [15,16,17,18]:
            combs = [comb for comb in combs if automaton2states[14][1] in comb and automaton2states[14][0] not in comb]
        elif k in [24,25,26,27]:
            combs = [comb for comb in combs if automaton2states[23][1] in comb and automaton2states[23][0] not in comb]
    
        samples = random.sample(combs, k=min(100, len(combs)))
        pair_distractors[k] = samples
    return pair_distractors

# Case 1
pair_distractors_b_case1 = gen_pair_distractors_b_case_1(automaton2states)
train_data, test_data = gen_b(pair_distractors_b_case1, target_automatons)
data_01 = {"case1_train": train_data, "case1_test": test_data}

# Case 2
pair_distractors_b_case2 = gen_pair_distractors_b_case_2(automaton2states)
train_data, test_data = gen_b(pair_distractors_b_case2, target_automatons, True)
data_02 = {"case2_train": train_data, "case2_test": test_data}
print(data_02)
# Case 3
pair_distractors_b_case3 = gen_pair_distractors_b_case_3(automaton2states)
train_data, test_data = gen_b(pair_distractors_b_case3, target_automatons)
data_03 = {"case3_train": train_data, "case3_test": test_data}

# Merge
data_b = {**data_01, **data_02, **data_03}

# Remove outer keys, make it just train, test
data_b_merged = merge_cases(data_b)

with open("data_b.pkl", "wb") as f:
    pickle.dump(data_b_merged, f)

print("Saved: ", "data_b.pkl")

