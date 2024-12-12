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

        self.graphs_number = len(graphs)
        self.colors = [np.random.rand(3,) for _ in range(self.graphs_number)]
        self.pos_list = []
        self.enabling_conditions = []

        self.visualize_dict = {
        'shortest_paths_visualize': 
            lambda step, automaton_idx, graph, pos, ax: self.__highlight_shortest_path(step, automaton_idx, graph, pos, ax),
        'edge_conditions_visualize': 
            lambda step, automaton_idx, graph, pos, ax: self.__visualize_edge_conditions(step, automaton_idx, graph, pos, ax),
        'edge':
            lambda step, automaton_idx, graph, pos, ax: self.__visualize_current_edge(step, automaton_idx, graph, pos, ax),
        'states':
            lambda step, automaton_idx, graph, pos, ax: self.__visualize_current_states(step, automaton_idx, graph, pos, ax),
        'target':
            lambda step, automaton_idx, graph, pos, ax: self.__visualize_target(step, automaton_idx, graph, pos, ax)
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
        # current_goal = current_step['goal']
        # target_state = current_goal['target_state']
        # target_automaton = current_goal['target_automaton']
        current_goal = 0
        target_automaton = current_step['automata_id']
        target_state = current_step['next_state']

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
            
            # Visualize edge conditions
            for edge in G.edges(data=True):
                    if 'enabling' in edge[2]:
                        for m, condition in enumerate(edge[2]['enabling']):
                            edge_pos = pos[edge[0]] + (pos[edge[1]] - pos[edge[0]]) * 0.5

                            plt.text(edge_pos[0]+0.05*m,
                                     edge_pos[1]+0.05,
                                     f'{condition["node_id"]}',
                                     color=self.colors[condition['automata_id']],
                                     fontsize=14)

            # Visualize current step
            for item in current_step:
                if item in self.visualize_dict:
                    self.visualize_dict[item](current_step, i, G, pos, ax)

            #ax.set_title(f'Random Graph {i+1}')

        # Create 1 subplot spanning all rows in second column
        ax = self.fig.add_subplot(self.gs[:1, 1])

        ax.clear()
        ax.axis('off')
        # Create vector of 5 random numbers
        top_row = [str(target_state)]
        top_colors = [self.colors[target_automaton]]
        # Create sample data for tables
        table1 = ax.table(cellText=[self.steps[frame]['states']],
                        cellColours=[self.colors],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 0.5])

        table2 = ax.table(cellText=[top_row],
                        cellColours=[top_colors],
                        cellLoc='center',
                        loc='center',
                        bbox=[0.2*target_automaton, 0.6, 0.2, 0.5]) # prvni souradnice ovlivnuje umisteni vrchni bunky

        # add text with current step number
        ax.text(0.5, -0.5, f'Step: {frame+1}', transform=ax.transAxes,
                fontsize=40, ha='center', va='center')

        # Style the tables
        table1.set_fontsize(14)
        table2.set_fontsize(14)

    def animate(self, filename):

        # Create animation
        anim = animation.FuncAnimation(self.fig, self.__update, frames=len(self.steps),
                                    
                                    interval=1000, repeat=True)
        anim.save(f'{filename}.gif', writer='pillow')

    def __visualize_target(self, step, automaton_idx, graph, pos, ax):
        if automaton_idx == step['target']['automata_id']:
            nx.draw_networkx_nodes(graph, pos, ax=ax,
                                nodelist=[step['target']['state']],
                                node_color='red',
                                node_size=500)

    def __visualize_current_states(self, step, automaton_idx, graph, pos, ax):
        if automaton_idx == step['automata_id']:
            edgecolors = "black"
            node_size = 400
            linewidths = 4
        else:
            edgecolors = "yellow"
            node_size = 300
            linewidths = 4
            
        color = self.colors[automaton_idx]
        if automaton_idx == step['target']['automata_id'] and step['states'][automaton_idx] == step['target']['state']:
            color = 'red'

        nx.draw_networkx_nodes(graph, pos, ax=ax,
                                nodelist=[step['states'][automaton_idx]],
                                node_color=color,
                                edgecolors=edgecolors,
                                node_size=node_size,
                                linewidths=linewidths)
        

    def __visualize_current_edge(self, step, automaton_idx, graph, pos, ax):
        if automaton_idx == step['automata_id']:
            current_edge = step['edge']
            
            nx.draw_networkx_edges(graph, pos, ax=ax,
                                    edgelist=[(current_edge[0], current_edge[1])],
                                    edge_color='red',
                                    width=5)
        
    def __highlight_shortest_path(self, step, automaton_idx, graph, pos, ax):
        if step['visualize']['shortest_paths_visualize'][automaton_idx] and step['shortest_paths'][automaton_idx] != None:
            shortest_path = step['shortest_paths'][automaton_idx]['path']

            for i in range(len(shortest_path)-1):
                nx.draw_networkx_edges(graph, pos, ax=ax,
                                    edgelist=[(shortest_path[i], shortest_path[i+1])],
                                    edge_color='blue',
                                    width=3)
                
    def __visualize_edge_conditions(self, step, automaton_idx, graph, pos, ax):
        pass
        