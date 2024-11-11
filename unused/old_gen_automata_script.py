# %%
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
# create 3 star automata
for i in range(0,3):
    train,test = get_star_transitions(i*9)
    train_core[i+1]=train
    test_core[i+1]= test

automaton2states = {}
# for each automaton, we want to create two states
for i in range(0,9*3):
    automaton2states[i+1] = (i*2+1,i*2+2)
automaton2states

# %%
def generate_star():
    train = []
    test = []
    for i in range(1, 5):
        for j in range(1, 5):
            if i==j:
                test.append((i, 5, 5+j))
            else:
                train.append((i, 5, 5+j))
    return train, test

# %%
train, test = generate_star()
train

# %%
def add_new_automata(train, test):
    new_train = train.copy()
    new_test = test.copy()
    for i in range(9, 72, 9):
        for sample in train:
            a, b, c = sample
            new_train.append((a+i, b+i, c+i))
        for sample in test:
            a, b, c = sample
            new_test.append((a+i, b+i, c+i))
    return new_train, new_test

# %%
# napárovat train a test podle ID
# {id: (train), (test)}

# nejdříve vygenerovat 3 možné cesty pro každou hvězdu z trainu
# (1,5,6) - (1,5), (5,6)
# 1-5 nesmí být distractor
# samplovat dvojice a trojice (50/50) z jiných hvězd jako distractory
# když bude chybět, dát random distractor z jiných hvězd
# potom permutace
# poslední pozice = cílový automat (5 nebo 6)

# pair train and test by ID
# {id: (train), (test)}

# first generate 3 possible paths for each star from train
# one example: (1,5,6) - (1,5), (5,6)
# for this example, any automata 1-5 must not be a distractor
# sample pairs and triples (50/50) from other stars as distractors
# if missing, obtain random distractor from other stars
# then permutations
# last position = target automaton (in this example 5 for (1,5), 6 for (5,6), 6 for (1,5,6))
# total of 7 positions

# %%
train, test = add_new_automata(train, test)

# %%
def get_paths(train):
    train_paths = []
    for i in train:
        train_paths.append(i)
        tup1 = (i[0], i[1])
        train_paths.append(tup1)
        tup2 = (i[1], i[2])
        train_paths.append(tup2)
    return train_paths

train_paths = get_paths(train)

# %%
train_paths

# %% [markdown]
# target samplovat z target automatů (6789, 15-18, 24-27) - stavy 
# middle jsou 5, 14, 23
# (pro a. case jsou to vždy ty první 4 potom - 1-5)
# 
# 1) middle automat stav který náleží target stavu nebude v datech
# 2) middle automat stav bude v datech ale bude off
# 3) middle automat stav bude v datech ale bude on
# 
# 

# %%
from itertools import combinations, permutations
import random
# 3 typy distractors pro B, v tmp nebudou ty dva stavy prostředního automatu - zjistit id automatu pro daný cílový stav
# vytvořím samply pěti distractorů třikrát tak, že v prvním nejsou stavy odpovídající prostřednímu automatu
# potom na samples udělám 3 casy, u druhého casu změním 5. element na target (on) prostředního automatu
# potom u třetího casu změním 5. element na off prostředního automatu
# vědět který case je který u dat
def gen_pair_distractors_c(automaton2states):
    pair_distractors = {}
    for k, v in automaton2states.items():
        off, on = v 
        tmp = set(range(1,55)) - set([off, on])
        combs = list(combinations(tmp, 5))
        samples = random.sample(combs, k=100)
        pair_distractors[k] = samples
    return pair_distractors

# %%
pair_distractors = gen_pair_distractors_c(automaton2states)
pair_distractors

# %%
def gen_c(automaton2states, pair_distractors):
    train_data = []
    test_data = []
    num_permutations_train = 50
    num_permutations_test = 10
    for k, v in automaton2states.items():
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
train_data_c, test_data_c = gen_c(automaton2states, pair_distractors)

# %%
len(train_data_c)

# %%
data = {"train": train_data_c, "test": test_data_c}
import pickle
with open ("data.pkl", "wb") as f:
    pickle.dump(data, f)

# %%
import random
from itertools import permutations, combinations
tmp = [automaton2states[5], automaton2states[14], automaton2states[23]]
middle_states = []
for tup in tmp:
    middle_states.append(tup[0])
    middle_states.append(tup[1])

middle_states

target_automatons = [6,7,8,9,15,16,17,18,24,25,26,27]
target_states_on = []
target_states_off = []

for index in target_automatons:
    value_tuple = automaton2states[index]
    target_states_off.append(value_tuple[0])
    target_states_on.append(value_tuple[1])

print(target_states_off)
print(target_states_on)
len(target_automatons)

# %%
# target samplovat z target automatů (6-9, 15-18, 24-27) - stavy 
# middle jsou 5, 14, 23
# (pro a. case jsou to vždy ty první 4 potom - 1-5)

# 1) middle automat stav který náleží target stavu nebude v datech
# 2) middle automat stav bude v datech ale bude off
# 3) middle automat stav bude v datech ale bude on

def gen_pair_distractors_b_case_1(automaton2states):
    pair_distractors = {}
    for k, v in automaton2states.items():
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

pair_distractors_b_case1 = gen_pair_distractors_b_case_1(automaton2states)
pair_distractors_b_case1

# %%
def gen_b(pair_distractors, target_automatons):
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
train_data, test_data = gen_b(pair_distractors_b_case1, target_automatons)
data_01 = {"case1_train": train_data, "case1_test": test_data}

# %%
import random
from itertools import combinations

def gen_pair_distractors_b_case_2(automaton2states):
    pair_distractors = {}
    for k, v in automaton2states.items():
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

pair_distractors_b_case2 = gen_pair_distractors_b_case_2(automaton2states)
pair_distractors_b_case2

train_data, test_data = gen_b(pair_distractors_b_case2, target_automatons)
data_02 = {"case2_train": train_data, "case2_test": test_data}

# %%
data_02["case2_train"][-100:]

# %%
import random
from itertools import combinations

def gen_pair_distractors_b_case_3(automaton2states):
    pair_distractors = {}
    for k, v in automaton2states.items():
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

pair_distractors_b_case3 = gen_pair_distractors_b_case_3(automaton2states)
pair_distractors_b_case3

train_data, test_data = gen_b(pair_distractors_b_case3, target_automatons)
data_03 = {"case3_train": train_data, "case3_test": test_data}

# %%
data_03["case3_train"][-100:]

# %%
data_b = {**data_01, **data_02, **data_03}

# %%
def merge_cases(original_dict):
    merged_dict = {'train': [], 'test': []}
    
    for key in original_dict:
        if 'train' in key:
            merged_dict['train'].extend(original_dict[key])
        elif 'test' in key:
            merged_dict['test'].extend(original_dict[key])
    
    return merged_dict

data_b_merged = merge_cases(data_b)

# %%
len(data_b_merged["train"])

# %%
from itertools import combinations
import random
num_stars = 3
train_data = []
test_data = []
distractors = {}
for i in range(num_stars):
    choices = [i%27+1 for i in range((i+1)*9,(i+3)*9)]
    # generate all possible non-repeating triples from choices
    combinations_of_three = list(combinations(choices, 3))
    distractors[i+1]=combinations_of_three
train_dicts = []
for k,tcs in train_core.items():
    for tc in tcs:
        for dist in distractors[k]:
            train_dicts.append({'core':tc,'distractor':dist})

test_dicts = []
for k,tcs in test_core.items():
    for tc in tcs:
        rand_dists = random.sample(distractors[k], 5)
        for dist in rand_dists:
            test_dicts.append({'core':tc,'distractor':dist})
    

# %%
train_dicts[-100:]

# case 1: nebude 1 ani 5 - nahradit 2 random distractory (kromě prostředního (5) a prvního - automaty 1-4 a 7) 
# case 2: nebude 1 - nahradit 1 random distractorem (kromě 1-5 a 7), 5 bude ve stavu on a target idx bude ukazovat na cílový automat (stav off u 7)
# case 3: nebude 1 - nahradit 1 random distractorem (kromě 1-5 a 7), 5 bude ve stavu off a target idx bude ukazovat na off stav u 5
# case 4: 5 bude ve stavu off, 1 bude ve stavu off - target idx bude ukazovat na off stav 1 (nesmí být distractor 1-5 a 7)
# case 5: 5 bude ve stavu off, 1 bude ve stavu on - target idx bude ukazovat na off stav 5 (nesmí být distractor 1-5 a 7)

# %%
import random

def generate_samples(train_dicts, automaton2states):
    samples = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    for row in train_dicts:
        core = row['core']

        in1, middle, out = core
        in1_states = automaton2states[in1]
        middle_states = automaton2states[middle]
        out_states = automaton2states[out]
        

        exclude_states = set(in1_states + middle_states + out_states)

        distractor_states = []

        for automaton, states in automaton2states.items():
            for state in states:
                if state not in exclude_states:
                    distractor_states.append(state)

        def get_random_distractor_states(n):
            return random.sample(distractor_states, n)

        case1_distractors = get_random_distractor_states(5)
        case1_input = [
            *case1_distractors[:2],
            out_states[0],
            *case1_distractors[2:],
            out_states[1]
        ]
        random.shuffle(case1_input)
        #case1_input = case1_input + [out_states[1]]
        samples[1].append({"input": case1_input, "out": case1_input.index(out_states[0]) + 55,"orig_idx": case1_input.index(out_states[0])})

        # Case 2
        case2_distractors = get_random_distractor_states(4)
        case2_input = [
            case2_distractors[0],
            middle_states[1],  # on state of middle
            case2_distractors[1],
            case2_distractors[2],
            out_states[0],
            case2_distractors[3],
            out_states[1]
        ]
        random.shuffle(case2_input)
        #case2_input = case2_input + [out_states[1]]
        samples[2].append({"input": case2_input, "out": case2_input.index(out_states[0]) + 55, "orig_idx": case2_input.index(out_states[0])})

        # Case 3
        case3_distractors = get_random_distractor_states(4)
        case3_input = [
            case3_distractors[0],
            middle_states[0],  # off state of middle
            case3_distractors[1],
            case3_distractors[2],
            out_states[0],
            case3_distractors[3],
            out_states[1]
        ]
        random.shuffle(case3_input)
        #case3_input = case3_input + [out_states[1]]
        samples[3].append({"input": case3_input, "out": case3_input.index(middle_states[0]) + 55, "orig_idx": case3_input.index(middle_states[0])})

        # Case 4
        case4_distractors = get_random_distractor_states(4)
        case4_input = [
            in1_states[0],
            middle_states[0],  # off state of middle
            case4_distractors[1],
            case4_distractors[2],
            out_states[0],
            case4_distractors[3],
            out_states[1]
        ]
        random.shuffle(case4_input)
        #case4_input = case4_input + [out_states[1]]
        samples[4].append({"input": case4_input, "out": case4_input.index(in1_states[0]) + 55,"orig_idx": case4_input.index(in1_states[0])})

        # Case 5
        case5_distractors = get_random_distractor_states(4)
        case5_input = [
            in1_states[1],
            middle_states[0],  # off state of middle
            case5_distractors[1],
            case5_distractors[2],
            out_states[0],
            case5_distractors[3],
            out_states[1]
        ]
        random.shuffle(case5_input)
        #case5_input = case5_input + [out_states[1]]
        samples[5].append({"input": case5_input, "out": case5_input.index(middle_states[0]), "orig_idx": case5_input.index(middle_states[0])})

    min_samples = min(len(samples[case]) for case in samples)
    for case in samples:
        samples[case] = samples[case][:min_samples]
    
    return samples

# %%
samples = generate_samples(train_dicts, automaton2states)

# %%
samples[3][1510]

# %%
for k, v in samples.items():
    print(len(v))

# %%
train_c_data = {"train": []}
for key in samples:
    train_c_data["train"].extend(samples[key])

# %%
test_c_data = []
def get_test_data_c(test_dicts):
    test_c_data = []
    for item in test_dicts:
        core = item['core']
        distractor = item['distractor']
        s_in = automaton2states[core[0]][0]
        s_mid = automaton2states[core[1]][0]
        s_out_off = automaton2states[core[2]][0]
        s_out_on = automaton2states[core[2]][1]
        dis1 = automaton2states[distractor[0]][random.randint(0, 1)]
        dis2 = automaton2states[distractor[1]][random.randint(0, 1)]
        dis3 = automaton2states[distractor[2]][random.randint(0, 1)]
        input_list = [dis1, dis2, dis3, s_in, s_mid, s_out_off, s_out_on]

        random.shuffle(input_list)
        #input_list = input_list + [s_out_on]
        
        dic = {"input": input_list, "out": input_list.index(s_in) + 55, "orig_idx": input_list.index(s_in)}
        test_c_data.append(dic)

    for item in test_dicts:
        core = item['core']
        distractor = item['distractor']
        s_in = automaton2states[core[0]][1]
        s_mid = automaton2states[core[1]][0]
        s_out_off = automaton2states[core[2]][0]
        s_out_on = automaton2states[core[2]][1]
        dis1 = automaton2states[distractor[0]][random.randint(0, 1)]
        dis2 = automaton2states[distractor[1]][random.randint(0, 1)]
        dis3 = automaton2states[distractor[2]][random.randint(0, 1)]

        input_list = [dis1, dis2, dis3, s_in, s_mid, s_out_off, s_out_on]

        random.shuffle(input_list)
        #input_list = input_list + [s_out_on]
        dic = {"input": input_list, "out": input_list.index(s_mid) + 55, "orig_idx": input_list.index(s_mid)}
        test_c_data.append(dic)
    return test_c_data

# %%
test_dicts

# %%
test_c_data = get_test_data_c(test_dicts)
test_c_data

# %%
test_c_data = {"test": test_c_data}

# %%
data_c = {**train_c_data, **test_c_data}
data_c

with open("data_c.pkl", "wb") as f:
    pickle.dump(data_c, f)

print("Saved: ", "data_c.pkl")

# %%
len(test_c_data["test"])

# %%
# a) one set of transitions where 3 automata are needed
# b) one set of transitions where only 2 automata are needed
# c) one set of transition where only 1 automaton is needed

# a) input format: [22:0, 23:0, 27:0, 1:0, 5:0, 11:0, 27:1]
# b) input format: [22:1, 23:0, 27:0, 1:0, 5:0, 11:0, 27:1]
# c) input format: [22:1, 23:1, 27:0, 1:0, 5:0, 11:0, 27:1]

# next create training samples by permuting the 6 first elements of the input and storing the position of the relevant automaton as output



# %%
from collections import defaultdict
import random
# TODO: přidat d), kde první dva stavy - druhý je dobře nastavený ale první je špatně
def create_transition_sets(data, automaton2states):
    result = defaultdict(list)

    for example in data:
        core = example['core'] # (22, 23, 27)
        distractor = example['distractor'] # (1, 4, 10)
        
        # a)
        input_dict_a = {}
        for node in core:
            input_dict_a[node] = automaton2states[node][0] # na 22 dám 43 (0), na 23 dám 45 (0), na 27 dám 53 (0) 
        for node in distractor:
            input_dict_a[node] = random.choice(automaton2states[node]) # na 1, 4, 10 dám náhodným výběrem stav, tzn. na 4 dám buď 7 nebo 8 (0 nebo 1)
        
        output_dict = {core[2]: automaton2states[core[2]][1]} # 3 automat v pořadí je finální, na něj dám stav 54 (1).
        result['a'].append({'in': input_dict_a, 'out': output_dict})
        
        # b)
        input_dict_b = {}
        for node in core:
            input_dict_b[node] = automaton2states[node][0] # stejné jako předtím
        input_dict_b[core[0]] = automaton2states[core[0]][1] # tady ale na v pořadí první automat 22 dám stav 44 (1)
        for node in distractor:
            input_dict_b[node] = input_dict_a[node]  # distractory mají stejné stavy pro a, b, c
        result['b'].append({'in': input_dict_b, 'out': output_dict})
        
        # c)
        input_dict_c = {}
        for node in core:
            input_dict_c[node] = automaton2states[node][0]
        input_dict_c[core[0]] = automaton2states[core[0]][1] # tady dávám na 22 stav 44 (1)
        input_dict_c[core[1]] = automaton2states[core[1]][1] # tady dávám na 23 stav 46 (1)
        for node in distractor:
            input_dict_c[node] = input_dict_a[node]  # distractory mají stejné stavy pro a, b, c
        result['c'].append({'in': input_dict_c, 'out': output_dict})

    return result

# %%
test_transition_sets = create_transition_sets(test_dicts, automaton2states)
train_transition_sets = create_transition_sets(train_dicts, automaton2states)
#sample:
for i, case in enumerate(['a', 'b', 'c']):
    print(f"{case}) {i+1}:")
    print(test_transition_sets[case][59])
    print()

# %%
len(train_transition_sets["a"]) * 3

# %%
# {'in': {22: 43, 23: 45, 27: 53, 4: 8, 12: 24, 16: 31}, 'out': {27: 54}}

# dict {input: [43, 53, 45, 8, 24, 31, 54], out: 0}

# {'in': {22: 44, 23: 45, 27: 53, 4: 8, 12: 24, 16: 31}, 'out': {27: 54}}

# dict {input: [43, 53, 45, 8, 24, 31, 54], out: 2}

# {'in': {22: 44, 23: 46, 27: 53, 4: 8, 12: 24, 16: 31}, 'out': {27: 54}}

# dict {[input: 43, 45, 8, 24, 31, 54], out: 1}

# 20 * 88128

# %%
train_transition_sets["c"][0]

# %%
# vytvořit vektory
# 7 položek - poslední položka je cílové id_stavu 
# 6 předchozích jsou id stavů ale ty permutujeme (20 náhodných permutací)
# bude dict, kde 6 čísel bude permutovaných a 7 nechávám jak je - tj. dict {input: [vektor], output: [idx_pozice_automatu]}

# %%
import random
from itertools import permutations

def extract_state(transition_sets, num_permutations=20):
    result = []

    for case, t_set in transition_sets.items():
        for dicts in t_set:
            input_dict = dicts["in"]
            output_dict = dicts["out"]
            
            # prvních 6 stavů vytáhnu
            states = list(input_dict.values())[:6]
            # a vytáhnu výstupní stav
            output_state = list(output_dict.values())[0]
            
            # prozatím pro C, tady se vytáhne podle pozice stav, který je třeba změnit
            if case == 'c':
                target_state = states[2]
            else:
                continue
            # if case == 'b':
            #     target_state = states[1] 
            # if case == 'a':
            #     target_state = states[0] 
            
            # prvních 6 stavů - permutace
            permutations_list = list(permutations(states))
            # výběr 20
            selected_permutations = random.sample(permutations_list, num_permutations)
            # pro každou permutaci
            for perm in selected_permutations:
                # do listu dát tuto permutaci a výstupní stav
                input_list = list(perm) + [output_state]
                # do out hodit index cílového stavu
                orig_idx = input_list.index(target_state)
                out = orig_idx+55
                # dát do dictu
                result.append({'input': input_list, 'out': out, 'orig_idx': orig_idx})

    return result

# data_train = extract_state(train_data)
# data_test = extract_state(test_data)

# %%



