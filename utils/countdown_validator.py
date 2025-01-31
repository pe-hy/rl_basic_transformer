import re
import networkx as nx
from collections import Counter


def validate_search_path(raw_search_path):
    """Processes a search path string and validates it, returning either 'Valid path' or an error message."""

    def parse_search_path(search_path):
        return list(map(str.strip, re.split(r",|;", search_path)))

    def strip_search_path(search_path):
        match_start = re.search(r"\bS\b", search_path)
        match_end = search_path.find(".") + 1  # Include the dot

        if not match_start:
            return "Invalid search_path: Missing standalone 'S'"
        if " O " not in search_path:
            return "Invalid search_path: No goal reached statement found."
        if match_end == 0:
            return "Invalid search_path: Missing '.' indicating the end of the goal statement."

        return search_path[match_start.start() : match_end]

    search_path = strip_search_path(raw_search_path)
    if not search_path.startswith("S"):
        return search_path  # Error message returned

    def build_tree_graph(search_path):
        """Build a tree graph from the search path."""
        actions = parse_search_path(search_path)
        graph = nx.DiGraph()
        current_node = "root"
        graph.add_node(current_node, numbers=None)
        node_map = {"#0": current_node}
        parent_stack = [current_node]

        for action in actions:
            if action.startswith("S"):
                # Set initial state numbers
                match = re.match(r"S (\d+) \[ ([\d\s]+) \]", action)
                if match:
                    nums = list(map(int, match.group(2).split()))
                    graph.nodes[current_node]["numbers"] = nums
            elif action.startswith("E"):
                # Handle operations
                match = re.match(
                    r"E\s*(\d+\s*[+\-*/]\s*\d+)\s*=\s*(\d+)\s*R\s*\[ ([^\]]+)\]", action
                )
                if match:
                    operation = match.group(1).strip()
                    result = int(match.group(2).strip())
                    operation_label = f"{operation} = {result}"
                    # Extract operands
                    left_operand, operator, right_operand = re.match(
                        r"(\d+)\s*([+\-*/])\s*(\d+)", operation
                    ).groups()
                    left_operand, right_operand = int(left_operand), int(right_operand)
                    # Update numbers
                    numbers_counter = Counter(graph.nodes[current_node]["numbers"])

                    # Reduce the count of each operand by 1 (remove one occurrence)
                    numbers_counter.subtract([left_operand, right_operand])

                    # Ensure non-negative counts (avoid negative values)
                    numbers_counter += (
                        Counter()
                    )  # This removes any keys with negative counts

                    # Construct the updated numbers list
                    updated_numbers = []
                    for num, count in numbers_counter.items():
                        updated_numbers.extend([num] * count)

                    # Construct the updated numbers list
                    updated_numbers = []
                    for num, count in numbers_counter.items():
                        updated_numbers.extend(
                            [num] * count
                        )  # Preserve duplicates correctly
                    updated_numbers.append(result)
                    updated_numbers.sort()
                    next_node = f"{current_node}-{result}"
                    if next_node not in graph:
                        graph.add_node(next_node, numbers=updated_numbers)
                    graph.add_edge(current_node, next_node, operation=operation_label)
                    parent_stack.append(current_node)
                    current_node = next_node
            elif action.startswith("G"):
                # Map node IDs
                match = re.match(r"G (#\d+)", action)
                if match:
                    node_id = match.group(1)
                    node_map[node_id] = current_node
            elif action.startswith("M"):
                # Move to an existing node
                match = re.match(r"M (#\d+)", action)
                if match:
                    node_id = match.group(1)
                    if node_id in node_map:
                        current_node = node_map[node_id]
                    else:
                        return (
                            f"Error: Invalid move to node ID {node_id}, node not found."
                        )

            elif action.startswith("N"):
                # Non-goal (failure) path
                match = re.match(r"N (\d+) (\d+)", action)
                if match:
                    generated = int(match.group(1))
                    target = int(match.group(2))
                    # Set the numbers for the current node before creating a failure node
                    graph.nodes[current_node]["numbers"] = [generated]
                    next_node = f"{current_node}-{generated}"
                    if next_node not in graph:
                        graph.add_node(next_node, numbers=[])  # Leaf node remains empty
                    graph.add_edge(
                        current_node, next_node, operation=f"N {generated} ≠ {target}"
                    )
                    current_node = parent_stack.pop()  # Move back up to the parent node
            elif action.startswith("O"):
                # Handle the goal state
                match = re.match(r"O (\d+) (\d+)", action)
                if match:
                    goal = int(match.group(1))
                    target = int(match.group(2))
                    # Update the second-to-last node with the goal value
                    graph.nodes[current_node]["numbers"] = [goal]
                    next_node = f"{current_node}-goal"
                    if next_node not in graph:
                        graph.add_node(next_node, numbers=[])  # Goal node is empty
                    graph.add_edge(
                        current_node, next_node, operation=f"Goal: {goal} = {target}"
                    )

        return graph

    graph = build_tree_graph(search_path)

    if isinstance(graph, str):
        return graph  # Stop execution

    def validate_path(graph):
        """Validate the path to the goal in the graph, including input-output arrays."""
        # Find goal node
        goal_node = None
        for node in graph.nodes:
            if "goal" in node:
                goal_node = node
                break

        if not goal_node:
            return "Invalid graph: No goal node found."

        # Traverse from goal to root and validate
        current_node = goal_node
        path_operations = []
        while current_node != "root":
            predecessors = list(graph.predecessors(current_node))
            if not predecessors:
                return "Invalid graph: Path to root is broken."
            parent_node = predecessors[0]
            edge_data = graph.edges[parent_node, current_node]
            operation = edge_data.get("operation")
            if not operation:
                return f"Invalid graph: Missing operation on edge {parent_node} -> {current_node}."

            path_operations.append((parent_node, operation, current_node))
            current_node = parent_node

        path_operations.reverse()

        # Validate each operation and the resulting numbers
        current_numbers = graph.nodes["root"]["numbers"]
        if current_numbers is None:

            return "Invalid graph: Root node has no numbers."

        for parent, operation, child in path_operations[
            :-1
        ]:  # Exclude the goal operation

            match = re.match(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)", operation)
            if not match:
                return f"Invalid operation: {operation}"

            left_operand = int(match.group(1))
            operator = match.group(2)
            right_operand = int(match.group(3))
            result = int(match.group(4))

            # Check if numbers used are available
            if (
                left_operand not in current_numbers
                or right_operand not in current_numbers
            ):
                return f"Not available numbers: {left_operand} or {right_operand} not in {current_numbers}"

            # Validate the operation
            valid_result = {
                "+": left_operand + right_operand,
                "-": left_operand - right_operand,
                "*": left_operand * right_operand,
                "/": left_operand / right_operand if right_operand != 0 else None,
            }.get(operator)

            if valid_result is None or valid_result != result:
                return f"Invalid operation: {operation} results in {valid_result}, not {result}"

            # Update current numbers
            # Use a Counter to track occurrences properly
            expected_counter = Counter(current_numbers)
            expected_counter.subtract(
                [left_operand, right_operand]
            )  # Remove one occurrence of each
            expected_counter += Counter()  # Remove negative counts
            expected_counter[result] += 1  # Add the result

            # Convert Counter back to a sorted list
            expected_numbers = sorted(
                sum([[num] * count for num, count in expected_counter.items()], [])
            )

            # Check the output array in the child node
            child_numbers = graph.nodes[child].get("numbers")
            if child_numbers is not None:
                if sorted(child_numbers) != expected_numbers:
                    return f"Wrong input output list: Expected {expected_numbers}, but got {child_numbers}"

            # Update current numbers for next iteration
            current_numbers = expected_numbers

        # Validate the final goal operation
        final_node = path_operations[-1][0]  # One-before-last node
        final_numbers = graph.nodes[final_node].get("numbers")
        goal_value = int(re.match(r"Goal: (\d+) =", path_operations[-1][1]).group(1))

        if final_numbers is None or goal_value not in final_numbers:
            return f"Wrong input output list: Final state numbers {final_numbers} do not contain goal {goal_value}."

        return "Valid path."

    return validate_path(graph)
