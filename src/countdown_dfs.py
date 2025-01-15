import itertools

from countdown_utils import combine_nums, CountdownNode, sum_heuristic, mult_heuristic, metric_fn

def dfs(target, nums, heuristic=sum_heuristic, threshold=None, search_trace="", open_set=[]):
    if len(open_set) == 0:
        # Push the initial node with its index, heuristic value, and parent index
        open_set.append((heuristic(nums, target), CountdownNode(0, None, nums, [], heuristic(nums, target))))

    while open_set:
        # Sort open_set by heuristic value, then pop the best node (lowest heuristic)
        open_set.sort(key=lambda x: -x[0])
        _, current_node = open_set.pop()

        nums = str(current_node.nums).replace(",", "").replace("[", "").replace("]", "")
        search_trace += f"S {target} [ {nums} ] , "

        # Generate successors for the current node
        generated_nodes = []
        for i, j in itertools.combinations(range(len(current_node.nums)), 2):
            node_index = 0
            for result, operation in combine_nums(current_node.nums[i], current_node.nums[j]):
                new_nums = [current_node.nums[k] for k in range(len(current_node.nums)) if k != i and k != j] + [result]
                new_operations = current_node.operations + [operation]
                new_heuristic = heuristic(new_nums, target)
                new_node = CountdownNode(node_index, current_node, new_nums, new_operations, new_heuristic)
                generated_nodes.append((new_heuristic, new_node))  # Add to generated nodes

        kept_nodes = []
        for g in generated_nodes:
            if threshold is None or g[0] <= threshold:
                kept_nodes.append(g)
            else:
                continue
        generated_nodes = kept_nodes
        generated_nodes.sort()

        node_index = 0
        for g, (_, new_node) in enumerate(generated_nodes):
            new_node.idx = f"{new_node.parent.idx},{node_index}"
            operations = str(new_node.operations[-1]).replace("*", " * ").replace("/", " / ").replace("+", " + ").replace("-", " - ").replace("=", " = ")
            nums = str(new_node.nums).replace(",", "").replace("[", "").replace("]", "")
            search_trace += f"E {operations} R [ {nums} ] , "

            if len(new_node.nums) == 1 and new_node.nums[0] == target:
                search_trace += f"O {new_node.nums[0]} {target} ."
                return search_trace
            elif len(new_node.nums) == 1:
                search_trace += f"N {new_node.nums[0]} {target} ; "
            else:
                node = str(new_node.idx).replace(",", "")
                nums = str(current_node.nums).replace(",", "").replace("[", "").replace("]", "")
                search_trace += f"G #{node} {target} [ {nums} ] , "
                new_set = [(new_heuristic, new_node)]

                node = str(new_node.idx).replace(",", "")
                search_trace += f"M #{node} , "
                search_trace = dfs(target, nums, heuristic=heuristic, threshold=threshold, search_trace=search_trace, open_set=new_set)
                if "O" in search_trace:
                    return search_trace
            node_index += 1
            if g < len(generated_nodes) - 1:
                next_index = new_node.parent.idx
                node = str(next_index).replace(",", "")
                search_trace += f"M #{node} , "
                nums = str(new_node.parent.nums).replace(",", "").replace("[", "").replace("]", "")
                search_trace += f"S {target} [ {nums} ] , "

        # Backtracking trace
        if open_set:  # If there are still nodes to explore
            next_node = open_set[-1][1]  # Get the index of the next node to be explored
            next_index = next_node.idx
            node = str(next_index).replace(",", "")
            search_trace += f"M #{node} , "

    return search_trace


if __name__ == "__main__":
    # Example usage
    target = 24
    nums = [8, 2, 3, 2, 1]
    search_path = dfs(target, nums, heuristic=mult_heuristic, threshold=target)
    print(search_path)
    print(len(search_path))
    # print(metric_fn(search_path))
    # enc = tiktoken.get_encoding("cl100k_base")
    # tokens = enc.encode(search_path)
    # print(f"token length: {len(tokens)}")