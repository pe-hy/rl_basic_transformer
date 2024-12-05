import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import copy

SOLVED_CONDITION_COLOR = 'grey'
SOLVED_CONDITION_FONTSIZE = 8
SOLVING_CONDITION_FONTSIZE = 42

class Visualize_search():
    def __init__(self, graphs, steps, numbers):
        self.graphs = graphs
        self.steps = steps
        print(len(self.steps))

        self.graphs_number = len(graphs)
        self.colors = [np.random.rand(3,) for _ in range(self.graphs_number)]
        self.pos_list = []
        self.enabling_conditions = []

        self.visualize_dict = {
        'shortest_paths_visualize': 
            lambda step, automaton_idx, graph, pos, ax: self.__highlight_shortest_path(step, automaton_idx, graph, pos, ax),
        'current_edges_visualize': 
            lambda step, automaton_idx, graph, pos, ax: self.__visualize_current_edge(step, automaton_idx, graph, pos, ax),
        'highlighted_states': 
            lambda step, automaton_idx, graph, pos, ax: self.__highlight_state(step, automaton_idx, graph, pos, ax),
        'edge_conditions_visualize': 
            lambda step, automaton_idx, graph, pos, ax: self.__visualize_edge_conditions(step, automaton_idx, graph, pos, ax)
        }

        for graph_idx in range(self.graphs_number):
            G = graphs[graph_idx]
            self.pos_list.append(nx.spring_layout(G))

        # Create a figure with custom gridspec
        self.fig = plt.figure(figsize=(20, 28))
        self.gs = plt.GridSpec(self.graphs_number, 2, figure=self.fig)

        self.numbers = numbers

    # Function to update the frame
    def __update(self, frame):
        current_step = self.steps[frame]
        print(current_step)
        # current_goal = current_step['goal']
        # target_state = current_goal['target_state']
        # target_automaton = current_goal['target_automaton']
        current_goal = 0
        target_state = 0
        target_automaton = 0

        numbers = self.numbers

        # Clear the figure
        self.fig.clear()

        # Create 5 random graphs in the first column
        for i in range(self.graphs_number):
            
            ax = self.fig.add_subplot(self.gs[i, 0])
            # Generate random graph
            G = self.graphs[i]
            
            # Draw the graph
            pos = self.pos_list[i]
            
            # Draw all nodes
            nx.draw(G, pos, ax=ax, with_labels=True,
                    node_color=self.colors[i], node_size=300,
                    font_size=8, font_weight='bold')
            
            # for item in current_step['visualize']:
            #     self.visualize_dict[item](current_step, i, G, pos, ax)

            #ax.set_title(f'Random Graph {i+1}')

        # Create 1 subplot spanning all rows in second column
        ax = self.fig.add_subplot(self.gs[:1, 1])

        ax.clear()
        ax.axis('off')
        # Create vector of 5 random numbers
        top_row = [str(target_state)]
        top_colors = [self.colors[target_automaton]]
        # Create sample data for tables
        table1 = ax.table(cellText=[numbers],
                        cellColours=[self.colors],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 0.5])

        table2 = ax.table(cellText=[top_row],
                        cellColours=[top_colors],
                        cellLoc='center',
                        loc='center',
                        bbox=[0.2*target_automaton, 0.6, 0.2, 0.5]) # prvni souradnice ovlivnuje umisteni vrchni bunky

        # Style the tables
        table1.set_fontsize(14)
        table2.set_fontsize(14)

    def animate(self, filename):

        # Create animation
        anim = animation.FuncAnimation(self.fig, self.__update, frames=len(self.steps),
                                    
                                    interval=1000, repeat=True)
        anim.save(f'{filename}.gif', writer='pillow')

    def __visualize_current_edge(self, step, automaton_idx, graph, pos, ax):
        
        current_edge = step['current_edges'][automaton_idx]

        if step['visualize']['current_edges_visualize'][automaton_idx] == True and current_edge != None:
            color = current_edge['color']
            width = current_edge['width']
            
            nx.draw_networkx_edges(graph, pos, ax=ax,
                                    edgelist=[(current_edge['edge'][0], current_edge['edge'][1])],
                                    edge_color=color,
                                    width=width)
            
    def __highlight_state(self, step, automaton_idx, graph, pos, ax):
        hightlighted_nodes = step['visualize']['highlighted_states'][automaton_idx]

        if hightlighted_nodes != None:

            for item in hightlighted_nodes:
                color = item['color']
                node_size = item['node_size']
                edge_color = item['edge_color']
                linewidth =  item['linewidth']
                nx.draw_networkx_nodes(graph, pos, ax=ax,
                                    nodelist=[item['state']],
                                    node_color=color,
                                    edgecolors=edge_color,
                                    node_size=node_size,
                                    linewidths=linewidth)
        
    def __highlight_shortest_path(self, step, automaton_idx, graph, pos, ax):
        if step['visualize']['shortest_paths_visualize'][automaton_idx] and step['shortest_paths'][automaton_idx] != None:
            shortest_path = step['shortest_paths'][automaton_idx]['path']

            for i in range(len(shortest_path)-1):
                nx.draw_networkx_edges(graph, pos, ax=ax,
                                    edgelist=[(shortest_path[i], shortest_path[i+1])],
                                    edge_color='blue',
                                    width=3)
                
    def __visualize_edge_conditions(self, step, automaton_idx, graph, pos, ax):
        if step['visualize']['edge_conditions_visualize'][automaton_idx] and step['edge_conditions'][automaton_idx] != None:
            for edge in graph.edges(data=True):
                    for m, condition in enumerate(step["edge_conditions"][automaton_idx][(edge[0], edge[1])]):
                        edge_pos = pos[edge[0]] + (pos[edge[1]] - pos[edge[0]]) * 0.5
                        
                        if condition['solved'] == True:
                            plt.text(edge_pos[0]+0.05*m, edge_pos[1]+0.05, f'{condition["state"]}', color=SOLVED_CONDITION_COLOR, fontsize=SOLVED_CONDITION_FONTSIZE)
                        elif condition['being_solved'] == True:
                            plt.text(edge_pos[0]+0.05*m, edge_pos[1]+0.05, f'{condition["state"]}', color=condition['color'], fontsize=SOLVING_CONDITION_FONTSIZE)
                        else:
                            plt.text(edge_pos[0]+0.05*m, edge_pos[1]+0.05, f'{condition["state"]}', color=condition['color'], fontsize=condition['fontsize'])
        