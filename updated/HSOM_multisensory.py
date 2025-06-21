
import numpy as np
import random
import networkx as nx
import logging
import matplotlib.pyplot as plt
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GWRSOM:
    def __init__(self, a=0.1, h=0.1, en=0.05, es=0.2, an=1.05, ab=1.05, h0=0.5, tb=3.33, tn=14.3, S=1):
        self.a = a
        self.h = h
        self.es = es
        self.en = en
        self.an = an
        self.ab = ab
        self.h0 = h0
        self.tb = tb
        self.tn = tn
        self.S = S
        self.t = 1  # Timestep
        self.A = None  # Node matrix A
        self.connections = None
        self.ages = None
        self.errors = None
        self.firing_vector = None
        self.max_age = 50  # Added max_age parameter
        self.sigma = 0.3  # Neighbourhood width for topological preservation
   
    def Distance(self, x1, x2):
        """Calculate Euclidean distance between two vectors"""
        return np.linalg.norm(x1 - x2)
   
    def initialize(self, X):
        # Create weight vectors for initial nodes - initialization step 1
        X = X.astype(float)
        w1 = np.round(X[np.random.randint(X.shape[0])])
        w2 = np.round(X[np.random.randint(X.shape[0])])
        # Node matrix A
        self.A = np.array([w1, w2])
        # Only 2 nodes available at the beginning
        self.connections = np.zeros((2, 2))  # Matrix nxn (n=|nodes|) of 0,1 to indicate connection - initialization step 2
        self.ages = np.zeros((2, 2))
        self.errors = np.zeros(2)
        self.firing_vector = np.ones(2)

    def find_best_matching_units(self, x):
        x = x.astype(float)                                       
        distances = np.linalg.norm(self.A - x, axis=1)
        return np.argsort(distances)[:2]

    def _create_connection(self, b, s):
        if self.connections[b, s] and self.connections[s, b]:
            self.ages[b, s] = 0
            self.ages[s, b] = 0
        else:
            self.connections[b, s] = 1
            self.connections[s, b] = 1

    def _below_activity(self, x, b):
        w_b = self.A[b]
        activity = np.exp(-np.linalg.norm(x - w_b))
        return activity < self.a

    def _below_firing(self, b):
        return self.firing_vector[b] < self.h

    def _add_new_node(self, b1, b2, x):
        w_b1 = self.A[b1]
        weight_vector = np.round(w_b1 + x) / 2
        self.A = np.vstack((self.A, weight_vector))
        n = self.A.shape[0]
        self.connections = np.pad(self.connections, ((0, 1), (0, 1)))
        self.ages = np.pad(self.ages, ((0, 1), (0, 1)))
        self.firing_vector = np.append(self.firing_vector, 1)
        self.errors = np.append(self.errors, 0)

        self._create_connection(b1, n - 1)
        self._create_connection(b2, n - 1)
        self.connections[b1, b2] = 0
        self.connections[b2, b1] = 0

        self.remove_old_edges()

    def remove_old_edges(self):
        self.connections[self.ages > self.max_age] = 0
        self.ages[self.ages > self.max_age] = 0
        nNeighbour = np.sum(self.connections, axis=0)
        NodeIndisces = np.array(list(range(self.A.shape[0])))
        AloneNodes = NodeIndisces[np.where(nNeighbour == 0)]
        if AloneNodes.any() and self.A.shape[0] > 2:  # Ensure we keep at least 2 nodes
            self.connections = np.delete(self.connections, AloneNodes, axis=0)
            self.connections = np.delete(self.connections, AloneNodes, axis=1)
            self.ages = np.delete(self.ages, AloneNodes, axis=0)
            self.ages = np.delete(self.ages, AloneNodes, axis=1)
            self.A = np.delete(self.A, AloneNodes, axis=0)
            self.firing_vector = np.delete(self.firing_vector, AloneNodes)
            self.errors = np.delete(self.errors, AloneNodes)

    def _best(self, x):
        b1, b2 = self.find_best_matching_units(x)
        self._create_connection(b1, b2)
        return b1, b2

    def _get_neighbours(self, w):
        return self.connections[w, :].astype(bool)

    def _adapt(self, w, x):
        x = x.astype(float)
        weight_vector = self.A[w]
        hs = self.firing_vector[w]
        # Calculate update
        delta = self.es * hs * (x - weight_vector)
        new_position = weight_vector + delta
        self.A[w] = np.round(new_position)
        
        # Round to maintain discrete positions
        b_neighbours = self._get_neighbours(w)
        w_neighbours = self.A[b_neighbours]
        hi = self.firing_vector[b_neighbours]

        # Calculate neighborhood influence
        distances = np.array([self.Distance(self.A[w], neighbor) for neighbor in w_neighbours])
        influences = np.exp(-distances**2 / (2 * self.sigma**2))

        # update neighbors with topological preservation
        delta = self.en * np.multiply(hi.reshape(-1, 1)*influences.reshape(-1, 1), (x - w_neighbours))
        self.A[b_neighbours] = np.round(w_neighbours + delta)

    def _age(self, w):
        b_neighbours = self._get_neighbours(w)
        self.ages[w, b_neighbours] += 1
        self.ages[b_neighbours, w] += 1

    def _reduce_firing(self, w):
        t = self.t
        self.firing_vector[w] = self.h0 - self.S / self.ab * (1 - np.exp(-self.ab * t / self.tb))
        b_neighbours = self._get_neighbours(w)
        self.firing_vector[b_neighbours] = self.h0 - self.S / self.an * (1 - np.exp(-self.an * t / self.tn))

    def train(self, X, epochs=1):
        X = X.astype(float)
        if self.A is None:
            self.initialize(X)
        for _ in range(epochs):
            for x in X:
                b1, b2 = self._best(x)
                if self._below_activity(x, b1) and self._below_firing(b1):
                    self._add_new_node(b1, b2, x)
                else:
                    self._adapt(b1, x)
                    self._age(b1)
                    self._reduce_firing(b1)
                self.t += 1

    def get_weights(self):
        return self.A

    def get_connections(self):
        return self.connections


class Value:
    """Value computation class"""
    def __init__(self, num_nodes=0):
        self.V = np.zeros(num_nodes)  # Value function
        self.R = np.zeros(num_nodes)  # Reward function
        self.w_g = None  # Index of goal node

    def Distance(self, x1, x2):
        """Calculate Euclidean distance between two vectors"""
        return np.linalg.norm(x1 - x2)

    def ComputeReward(self, nodes, connections, goal):
        """Compute reward function based on distance to goal"""
        self.R = np.zeros(len(nodes))
        
        # Find node closest to goal
        D = []
        for i, (_, pos) in enumerate(nodes):
            D.append(self.Distance(goal, pos))
        
        self.w_g = np.argmin(D)
        
        # Set rewards based on distance to goal
        for i in range(len(nodes)):
            if i == self.w_g:
                self.R[i] = 10  # High reward for goal node
            else:
                # Exponential decay based on distance to goal
                _, pos_i = nodes[i]
                distance = self.Distance(goal, pos_i)
                self.R[i] = np.exp(-distance**2 / 200)  # Adjusted scale factor

    def ComputeValue(self, nodes, connections, goal, gamma=0.99):
        """Compute value function using value iteration"""
        num_nodes = len(nodes)
        self.V = np.zeros(num_nodes)
        
        # Compute rewards first
        self.ComputeReward(nodes, connections, goal)
        
        # Value iteration
        for _ in range(100):  # Fixed number of iterations
            for i in range(num_nodes):
                # Get neighboring nodes
                neighbors = np.where(connections[i, :] == 1)[0]
                
                if len(neighbors) > 0:
                    # Calculate maximum value from neighbors
                    neighbor_values = self.V[neighbors]
                    max_value = np.max(neighbor_values) if len(neighbor_values) > 0 else 0
                    
                    # Update value using Bellman equation
                    self.V[i] = self.R[i] + gamma * max_value
        
        return self.V


class Action:
    """Action selection class"""
    def __init__(self):
        self.indEX = None  # Expected next node index

    def Distance(self, x1, x2):
        """Calculate Euclidean distance between two vectors"""
        return np.linalg.norm(x1 - x2)

    def actionSelect(self, state, nodes, values, connections, action_mappings):
        """Select action based on current state and value function"""
        # Find closest node to current state
        min_dist = float('inf')
        current_node_idx = None
        
        for i, (_, pos) in enumerate(nodes):
            dist = self.Distance(state, pos)
            if dist < min_dist:
                min_dist = dist
                current_node_idx = i
        
        if current_node_idx is None:
            return random.randint(0, 3)  # Default to random action if no node found
        
        # Find connected nodes
        connected_nodes = np.where(connections[current_node_idx, :] == 1)[0]
        
        if len(connected_nodes) == 0:
            return random.randint(0, 3)  # Default to random action if no connections
        
        # Find node with highest value
        neighbor_values = values[connected_nodes]
        best_neighbor_idx = connected_nodes[np.argmax(neighbor_values)]
        
        # Set expected next node for explainability
        self.indEX = best_neighbor_idx
        
        # Return action that leads to best neighbor
        key = (current_node_idx, best_neighbor_idx)
        if key in action_mappings:
            return action_mappings[key]
        else:
            return random.randint(0, 3)  # Default to random if no action mapping found


class MultisensoryHGWRSOMAgent:
    def __init__(self, sensory_dimensions=None, higher_dim=2, 
                 epsilon_b=0.35, epsilon_n=0.15, 
                 beta=0.7, delta=0.79, T_max=20, N_max=300, 
                 eta=0.5, phi=0.9, sigma=0.5):
        """
        Initialize MultisensoryHGWRSOMAgent with multiple sensory inputs
        
        Parameters:
        -----------
        sensory_dimensions : dict
            Dictionary mapping sensory modality names to their dimensions
            e.g., {'position': 2, 'beacon_distances': 2, 'temperature': 1}
        """
        # Default sensory dimensions if none provided
        if sensory_dimensions is None:
            sensory_dimensions = {'position': 2, 'beacon_distances': 2}
        
        self.sensory_dimensions = sensory_dimensions
        
        # Initialize lower level networks for each sensory modality
        self.lower_networks = {}
        for modality, dim in sensory_dimensions.items():
            if modality == 'position':
                # Special case for position - split into x and y
                self.lower_networks['position_x'] = GWRSOM(a=0.4, h=0.1)
                self.lower_networks['position_y'] = GWRSOM(a=0.4, h=0.1)
            else:
                # For other modalities, create appropriate networks
                if dim == 1:
                    self.lower_networks[modality] = GWRSOM(a=0.4, h=0.1)
                else:
                    # For multi-dimensional inputs (e.g., distances to multiple beacons)
                    for i in range(dim):
                        self.lower_networks[f"{modality}_{i}"] = GWRSOM(a=0.4, h=0.1)
        
        # Higher level stores patterns AND their continuous positions
        self.nodes = []  # Will store tuples of (pattern, continuous_position)
        self.connections = np.zeros((0, 0))  # Connectivity between nodes
        self.action_mappings = {}  # Maps (node1, node2) pairs to actions
        
        # TMGWR-style value and action components
        self.ValueClass = Value()
        self.ActionClass = Action()
        
        # Agent parameters
        self.start_epsilon = 0.5
        self.epsilon = self.start_epsilon
        self.goal = None
        self.is_plan = None
        self.expected_next_node = None
        self.prev_node_idx = None
        
        # Learning parameters
        self.delta = delta  # Pattern difference threshold
        self.T_max = T_max  # Maximum age for connections
        self.N_max = N_max  # Maximum number of nodes
        self.pattern_ages = np.zeros((0, 0))  # Age of connections between patterns

    def train_lower_networks(self, training_data, epochs=100):
        """
        Train lower level networks with multimodal sensory data
        
        Parameters:
        -----------
        training_data : dict
            Dictionary with keys matching sensory modalities and values 
            containing numpy arrays of training data
        """
        if not training_data:
            raise Exception("No training data provided!")
        
        for modality, data in training_data.items():
            if modality == 'position':
                # Special case for position - split into x and y
                x_data = data[:, 0].reshape(-1, 1)
                y_data = data[:, 1].reshape(-1, 1)
                
                logger.info(f"Training position networks with {len(data)} samples...")
                self.lower_networks['position_x'].train(x_data, epochs=epochs)
                self.lower_networks['position_y'].train(y_data, epochs=epochs)
                
                logger.info(f"Position training complete. X-network: {len(self.lower_networks['position_x'].A)} nodes, "
                            f"Y-network: {len(self.lower_networks['position_y'].A)} nodes")
            else:
                # For other sensory modalities
                dim = self.sensory_dimensions[modality]
                if dim == 1:
                    logger.info(f"Training {modality} network with {len(data)} samples...")
                    self.lower_networks[modality].train(data.reshape(-1, 1), epochs=epochs)
                    logger.info(f"{modality} training complete. Network: {len(self.lower_networks[modality].A)} nodes")
                else:
                    # For multi-dimensional inputs
                    for i in range(dim):
                        network_name = f"{modality}_{i}"
                        input_data = data[:, i].reshape(-1, 1)
                        logger.info(f"Training {network_name} network with {len(input_data)} samples...")
                        self.lower_networks[network_name].train(input_data, epochs=epochs)
                        logger.info(f"{network_name} training complete. Network: {len(self.lower_networks[network_name].A)} nodes")

    def get_firing_pattern(self, sensory_input):
        """
        Convert multimodal sensory input to binary pattern
        
        Parameters:
        -----------
        sensory_input : dict
            Dictionary with keys matching sensory modalities and values 
            containing current sensory readings
        """
        patterns = []
        
        # Process each sensory modality
        for modality, value in sensory_input.items():
            if modality == 'position':
                # Handle position (x,y) specially
                x_data = np.array([value[0]]).reshape(1, -1)
                y_data = np.array([value[1]]).reshape(1, -1)
                
                x_bmus = self.lower_networks['position_x'].find_best_matching_units(x_data)
                y_bmus = self.lower_networks['position_y'].find_best_matching_units(y_data)
                
                x_binary = np.zeros(len(self.lower_networks['position_x'].A))
                y_binary = np.zeros(len(self.lower_networks['position_y'].A))
                
                if isinstance(x_bmus, tuple) or isinstance(x_bmus, list):
                    x_binary[x_bmus[0]] = 1
                else:
                    x_binary[x_bmus] = 1
                    
                if isinstance(y_bmus, tuple) or isinstance(y_bmus, list):
                    y_binary[y_bmus[0]] = 1
                else:
                    y_binary[y_bmus] = 1
                
                patterns.append(tuple(x_binary))
                patterns.append(tuple(y_binary))
            else:
                # Process other sensory modalities
                dim = self.sensory_dimensions[modality]
                if dim == 1:
                    data = np.array([value]).reshape(1, -1)
                    bmus = self.lower_networks[modality].find_best_matching_units(data)
                    
                    binary = np.zeros(len(self.lower_networks[modality].A))
                    if isinstance(bmus, tuple) or isinstance(bmus, list):
                        binary[bmus[0]] = 1
                    else:
                        binary[bmus] = 1
                    
                    patterns.append(tuple(binary))
                else:
                    # For multi-dimensional inputs
                    for i in range(dim):
                        network_name = f"{modality}_{i}"
                        data = np.array([value[i]]).reshape(1, -1)
                        bmus = self.lower_networks[network_name].find_best_matching_units(data)
                        
                        binary = np.zeros(len(self.lower_networks[network_name].A))
                        if isinstance(bmus, tuple) or isinstance(bmus, list):
                            binary[bmus[0]] = 1
                        else:
                            binary[bmus] = 1
                        
                        patterns.append(tuple(binary))
        
        # Combine all patterns into a single representation
        return tuple(patterns)

    def find_node_index(self, pattern):
        """Find the index of a node with the given pattern"""
        for i, node_data in enumerate(self.nodes):
            stored_pattern = node_data[0]  # Extract pattern from (pattern, position) tuple
            if stored_pattern == pattern:
                return i
        return None

    def update_model(self, next_sensory_input, action):
        """Update the model with a new sensory state and action"""
        # Get binary pattern for the next sensory input
        pattern = self.get_firing_pattern(next_sensory_input)
        
        # Extract position for storing with the pattern (for visualization and navigation)
        position = next_sensory_input.get('position', np.zeros(2))
        
        # Check if node exists for this pattern
        node_idx = self.find_node_index(pattern)
        
        if node_idx is None:
            # Pattern doesn't exist yet, create a new node
            self.nodes.append((pattern, position))  # Store both pattern and position
            
            # Grow connection matrices
            new_size = len(self.nodes)
            new_connections = np.zeros((new_size, new_size))
            new_ages = np.zeros((new_size, new_size))
            
            if new_size > 1:
                # Copy existing data
                new_connections[:-1, :-1] = self.connections
                new_ages[:-1, :-1] = self.pattern_ages
                
            self.connections = new_connections
            self.pattern_ages = new_ages
            
            # The index of the new node
            node_idx = len(self.nodes) - 1
        else:
            # Update position information for existing pattern
            # Using a weighted average (0.9 old, 0.1 new)
            old_pattern, old_position = self.nodes[node_idx]
            updated_position = 0.9 * np.array(old_position) + 0.1 * np.array(position)
            self.nodes[node_idx] = (old_pattern, updated_position)
        
        # Create connection from previous node if exists
        if self.prev_node_idx is not None:
            # Create a connection between the previous and current node
            self.connections[self.prev_node_idx, node_idx] = 1
            self.pattern_ages[self.prev_node_idx, node_idx] = 0
            
            # Store the action that led to this transition
            self.action_mappings[(self.prev_node_idx, node_idx)] = action
            
            # Age all other connections from the previous node
            connected = np.where(self.connections[self.prev_node_idx] == 1)[0]
            for c in connected:
                if c != node_idx:
                    self.pattern_ages[self.prev_node_idx, c] += 1
            
            # Remove old connections
            old_connections = self.pattern_ages > self.T_max
            self.connections[old_connections] = 0
            self.pattern_ages[old_connections] = 0
        
        # Set current node as previous for next update
        self.prev_node_idx = node_idx

    def select_action(self, current_sensory_input):
        """
        Select action based on multisensory input
        
        Parameters:
        -----------
        current_sensory_input : dict
            Dictionary containing all sensory inputs
        """
        if self.goal is None:
            raise Exception("No goal defined")
            
        # Decide between exploration and exploitation
        if np.random.uniform(0, 1) > self.epsilon:
            # Exploitation - using TMGWR-style value-based approach
            
            # First, compute the value function for all nodes
            # Ensure ValueClass has the right size
            if len(self.nodes) != len(self.ValueClass.V):
                self.ValueClass = Value(len(self.nodes))
                
            # Compute value function
            V = self.ValueClass.ComputeValue(self.nodes, self.connections, self.goal)
            
            # Extract position for navigation
            position = current_sensory_input.get('position', np.zeros(2))
            
            # Use action selection mechanism
            action = self.ActionClass.actionSelect(
                position,  # Just use position for action selection
                self.nodes, 
                V, 
                self.connections, 
                self.action_mappings
            )
            
            # Track expected next node for explainability
            self.expected_next_node = self.ActionClass.indEX
            self.is_plan = True
            
            return action
        else:
            # Exploration - choose random action
            self.is_plan = False
            return random.randint(0, 3)

    def explain_change(self):
        """Explain any discrepancies between expected and actual transitions"""
        if self.is_plan and self.expected_next_node is not None:
            current_pattern = self.get_firing_pattern(self.nodes[self.expected_next_node][1])
            current_idx = self.find_node_index(current_pattern)
            
            if current_idx != self.expected_next_node:
                print(f"World Changed! Expected node: {self.expected_next_node}; Actual node: {current_idx}")
                self.is_plan = None
                self.expected_next_node = None

    def set_goal(self, goal):
        self.goal = goal

    def decay_epsilon(self, min_epsilon=0.2):
        self.epsilon = max(round(self.epsilon-0.1, 5), min_epsilon)

    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def show_map(self, show_modality_info=False):
        """
        Visualize the map using the continuous positions stored with each node
        
        Parameters:
        -----------
        show_modality_info : bool
            If True, add information about different sensory modality activations
        """
        if len(self.nodes) == 0:
            print("No nodes to display")
            return
            
        graph = nx.DiGraph()
        
        # Add nodes with positions
        for i, node_data in enumerate(self.nodes):
            pattern, position = node_data  # Extract pattern and position
            
            # Add node attributes for visualization
            node_attrs = {'pos': position}
            
            if show_modality_info:
                # Add information about which sensory modalities are active
                pattern_idx = 0
                for modality, dim in self.sensory_dimensions.items():
                    if modality == 'position':
                        # Position has x and y networks
                        x_pattern = pattern[pattern_idx]
                        y_pattern = pattern[pattern_idx + 1]
                        x_active = np.where(np.array(x_pattern) > 0)[0]
                        y_active = np.where(np.array(y_pattern) > 0)[0]
                        node_attrs['pos_x'] = x_active[0] if len(x_active) > 0 else -1
                        node_attrs['pos_y'] = y_active[0] if len(y_active) > 0 else -1
                        pattern_idx += 2
                    elif dim == 1:
                        # Single dimension modality
                        mod_pattern = pattern[pattern_idx]
                        mod_active = np.where(np.array(mod_pattern) > 0)[0]
                        node_attrs[modality] = mod_active[0] if len(mod_active) > 0 else -1
                        pattern_idx += 1
                    else:
                        # Multi-dimensional modality
                        for j in range(dim):
                            mod_pattern = pattern[pattern_idx]
                            mod_active = np.where(np.array(mod_pattern) > 0)[0]
                            node_attrs[f"{modality}_{j}"] = mod_active[0] if len(mod_active) > 0 else -1
                            pattern_idx += 1
            
            graph.add_node(i, **node_attrs)
        
        # Add edges
        rows, cols = np.where(self.connections == 1)
        for r, c in zip(rows, cols):
            # Add action information if available
            edge_attrs = {}
            if (r, c) in self.action_mappings:
                action_map = {0: "Up", 1: "Down", 2: "Right", 3: "Left"}
                edge_attrs['action'] = action_map.get(self.action_mappings[(r, c)], "Unknown")
            
            graph.add_edge(r, c, **edge_attrs)
        
        # Draw the graph using the continuous positions
        pos = nx.get_node_attributes(graph, 'pos')
        plt.figure(figsize=(12, 12))
        
        # Draw edges with optional action labels
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, arrows=True, arrowsize=15)
        
        if show_modality_info:
            # Draw nodes with different colors based on temperature reading
            if 'temperature' in self.sensory_dimensions:
                temp_values = nx.get_node_attributes(graph, 'temperature')
                normalized_temps = {node: (val + 1) / 10 for node, val in temp_values.items()}
                node_colors = [plt.cm.plasma(normalized_temps.get(node, 0.5)) for node in graph.nodes()]
                nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500, alpha=0.8)
            else:
                nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=500)
        else:
            # Simple visualization without modality info
            nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=500)
        
        # Add node labels
        nx.draw_networkx_labels(graph, pos, font_size=8)
        
        # Add action labels to edges if available
        if any('action' in graph[u][v] for u, v in graph.edges()):
            edge_labels = {(u, v): graph[u][v]['action'] for u, v in graph.edges() if 'action' in graph[u][v]}
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)
        
        plt.title("MultisensoryMINERVA Map")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def run_multisensory_simulation(maze_map, player_pos_index, goal_pos_index, 
                               beacon_positions=None, noise_level=0, num_episodes=10):
    """
    Run simulation with multisensory inputs
    
    Parameters:
    -----------
    maze_map : list
        2D list representing the maze
    player_pos_index : tuple
        (row, col) index of player start position
    goal_pos_index : tuple
        (row, col) index of goal position
    beacon_positions : list
        List of beacon positions (index format)
    noise_level : float
        Level of noise to add to sensory inputs
    num_episodes : int
        Number of episodes to run
    
    Returns:
    --------
    agent : MultisensoryHGWRSOMAgent
        Trained agent
    training_stats : dict
        Statistics from training
    """
    from Maze.Maze_player import MazePlayer
    
    # Set default beacon positions if none provided
    if beacon_positions is None:
        # Place beacons at corners of maze
        beacon_positions = []
        beacon_positions.append((1, len(maze_map[0])-2))  # Top-right
        beacon_positions.append((len(maze_map)-2, 1))     # Bottom-left
    
    # Initialize maze with temporary instance to calculate screen positions
    temp_maze = MazePlayer(maze_map=maze_map, 
                          player_index_pos=player_pos_index, 
                          goal_index_pos=goal_pos_index,
                          display_maze=False)
    
    # Convert beacon index positions to screen coordinates
    beacon_screen_positions = [temp_maze._calc_screen_coordinates(*pos) for pos in beacon_positions]
    
    # Initialize maze with beacon positions
    Maze = MazePlayer(maze_map=maze_map, 
                     player_index_pos=player_pos_index, 
                     goal_index_pos=goal_pos_index,
                     beacon_positions=beacon_screen_positions)
    
    # Get goal coordinates
    goal = Maze.get_goal_pos()
    
    # Collect multisensory training data
    def collect_multisensory_training_data(steps=5000):
        """Collect training data for multisensory learning"""
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Storage for multisensory data
        positions = []
        beacon_distances = []
        temperatures = []
        
        # Reset player to initial position
        Maze.reset_player()
        current_state = Maze.get_player_pos()
        
        # Perform random exploration to collect positions
        for step in range(steps):
            # Store current position
            positions.append(current_state)
            
            # Store beacon distances
            beacon_dists = Maze.get_beacon_distances(current_state)
            beacon_distances.append(beacon_dists)
            
            # Store temperature
            temp = Maze.get_temperature_reading(current_state)
            temperatures.append([temp])
            
            # Take random action
            action = np.random.randint(0, 4)
            Maze.move_player(action)
            current_state = Maze.get_player_pos()
            
            # Print progress occasionally
            if (step + 1) % 1000 == 0:
                print(f"Pre-exploration: {step + 1}/{steps} steps completed")
            
            # If goal reached, reset player
            if current_state == Maze.get_goal_pos():
                Maze.reset_player()
                current_state = Maze.get_player_pos()
        
        # Convert to numpy arrays
        positions = np.array(positions)
        beacon_distances = np.array(beacon_distances)
        temperatures = np.array(temperatures)
        
        return {
            'position': positions,
            'beacon_distances': beacon_distances,
            'temperature': temperatures
        }
    
