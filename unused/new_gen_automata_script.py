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

train, test = generate_star()

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

train, test = add_new_automata(train, test)

# %%
len(test)

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
print(len(train_paths))
train_paths = list(sorted(set(get_paths(train))))

# %%
# test_paths = get_paths(test)

# %%
test

# %%
# nejdříve vygenerovat 3 možné cesty pro každou hvězdu z trainu
# (1,5,6) - (1,5), (5,6)
# 1-5 nesmí být distractor
# samplovat dvojice a trojice (50/50) z jiných hvězd jako distractory
# když bude chybět, dát random distractor z jiných hvězd
# potom permutace
# poslední pozice = cílový automat (5 nebo 6)

import itertools
import random
from collections import defaultdict

automata_to_exluded_automata = {i: range(1 + 9*i, 6 + 9*i) for i in range(8)}
star_to_excluded_automata = {i: range(1 + 9*i, 10 + 9*i) for i in range(8)}

def get_distractors(paths, n, train=False):
    combinations = defaultdict(list)
    all_elements = set(element for path in paths for element in path)
    for i, path in enumerate(paths):
        if train:
            group = i // 20
        else:
            group = i // 4

        excluded = set(automata_to_exluded_automata[group])
        candidates = [c for c in paths if not set(c).intersection(excluded)]
        current_star_automata = set(star_to_excluded_automata[group])
        while len(combinations[path]) < n and candidates:
            candidate = random.choice(candidates)
            sample_size = min(len(candidate), random.randint(2, 3))
            inner_list = random.sample(candidate, sample_size)
            
            while len(path) + len(inner_list) < 20:
                available = all_elements - set(path) - set(inner_list) - excluded - current_star_automata
                inner_list.append(random.choice(list(available)))
            
            combinations[path].append(inner_list)
            # candidates.remove(candidate)
            # excluded.update(candidate)

    return combinations

# %%
combinations_train = get_distractors(train, 100, True)
combinations_test = get_distractors(test, 1, False)


# %%
combinations_train

# %%
combinations_test

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

# %%
import random
import itertools

def generate_train(combinations):
    l_of_d = []
    for path, inner_lists in combinations.items():
        for lst in inner_lists:
            # Generate a few random permutations instead of all possible permutations
            num_permutations = min(10, len(lst))  # Adjust this number as needed
            for _ in range(num_permutations):
                new_lst = lst.copy()
                random.shuffle(new_lst)
                
                new_lst += path[:-1]
                random.shuffle(new_lst)
                new_lst.append(path[-1])
                
                d = {"input": new_lst, "orig_idx": new_lst.index(path[0]), "target_idx": new_lst.index(path[0]) + 73}
                l_of_d.append(d)

    return l_of_d

# %%
def generate_test(combinations):
    l_of_d = []
    for path, inner_lists in combinations.items():
        for lst in inner_lists:
            new_lst = list(lst)
            new_lst += path[:-1]
            random.shuffle(new_lst)
            new_lst.append(path[-1])
            d = {"input": new_lst, "orig_idx": new_lst.index(path[0]), "target_idx": new_lst.index(path[0]) + 73}
            l_of_d.append(d)

    return l_of_d

# %%
result_test = generate_test(combinations_test)
result_train = generate_train(combinations_train)

# %%
len(result_train)

# %%
data = {"train": result_train, "test": result_test}
import pickle
with open ("new_data.pkl", "wb") as f:
    pickle.dump(data, f)

# %%
len(data["train"])

# %%
len(data["test"])

# %%
data["test"]

# %%
my_s = set()
for i in data["train"]:
    for k, v in i.items():
        if k == "target_idx":
            my_s.add(v)

# %%
my_s  # vocab size = 92

# %%



