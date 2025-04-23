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
    # GWRSOM implementation remains the same
    # ... (same as previous implementation)
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
    """Value computation class, similar to TMGWR's ValueClass"""
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
    """Action selection class, similar to TMGWR's ActionClass"""
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


class HierarchicalGWRSOMAgent:
    def __init__(self, lower_dim=1, higher_dim=2, epsilon_b=0.35, epsilon_n=0.15, 
                 beta=0.7, delta=0.79, T_max=20, N_max=300, eta=0.5, phi=0.9, sigma=0.5):
        # Initialize lower level networks
        self.lower_x = GWRSOM(a=0.4, h=0.1)
        self.lower_y = GWRSOM(a=0.4, h=0.1)
        
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
        """Pre-train lower level networks with actual maze positions"""
        if len(training_data) == 0:
            raise Exception("No training data provided!")
        
        # Train x and y networks separately
        x_data = training_data[:, 0].reshape(-1, 1)
        y_data = training_data[:, 1].reshape(-1, 1)
        
        # Train the networks
        logger.info(f"Training lower networks with {len(training_data)} samples...")
        self.lower_x.train(x_data, epochs=epochs)
        self.lower_y.train(y_data, epochs=epochs)
        logger.info(f"Training complete. X-network: {len(self.lower_x.A)} nodes, Y-network: {len(self.lower_y.A)} nodes")

    def get_firing_pattern(self, state):
        """Convert continuous position to binary pattern"""
        # Use lower networks to encode position
        x_data = np.array([state[0]]).reshape(1, -1)
        y_data = np.array([state[1]]).reshape(1, -1)
        
        # Get best matching units from lower networks
        x_bmus = self.lower_x.find_best_matching_units(x_data)
        y_bmus = self.lower_y.find_best_matching_units(y_data)
        
        # Create binary vectors
        x_binary = np.zeros(len(self.lower_x.A))
        y_binary = np.zeros(len(self.lower_y.A))
        
        # Set the active units
        if isinstance(x_bmus, tuple) or isinstance(x_bmus, list):
            x_binary[x_bmus[0]] = 1
        else:
            x_binary[x_bmus] = 1
            
        if isinstance(y_bmus, tuple) or isinstance(y_bmus, list):
            y_binary[y_bmus[0]] = 1
        else:
            y_binary[y_bmus] = 1
        
        return (tuple(x_binary), tuple(y_binary))

    def find_node_index(self, pattern):
        """Find the index of a node with the given pattern"""
        for i, node_data in enumerate(self.nodes):
            stored_pattern = node_data[0]  # Extract pattern from (pattern, position) tuple
            if stored_pattern == pattern:
                return i
        return None

    def update_model(self, next_state, action):
        """Update the model with a new state-action pair"""
        # Get binary pattern for the next state
        pattern = self.get_firing_pattern(next_state)
        
        # Check if node exists for this pattern
        node_idx = self.find_node_index(pattern)
        
        if node_idx is None:
            # Pattern doesn't exist yet, create a new node
            self.nodes.append((pattern, next_state))  # Store both pattern and position
            
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
            updated_position = 0.9 * np.array(old_position) + 0.1 * np.array(next_state)
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

    def select_action(self, current_state):
        """TMGWR-style action selection"""
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
            
            # Use action selection mechanism
            action = self.ActionClass.actionSelect(
                current_state, 
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

    def show_map(self):
        """Visualize the map using the continuous positions stored with each node"""
        if len(self.nodes) == 0:
            print("No nodes to display")
            return
            
        graph = nx.DiGraph()
        
        # Add nodes with positions
        for i, node_data in enumerate(self.nodes):
            _, position = node_data  # Extract position from (pattern, position) tuple
            graph.add_node(i, pos=position)
        
        # Add edges
        rows, cols = np.where(self.connections == 1)
        for r, c in zip(rows, cols):
            graph.add_edge(r, c)
        
        # Draw the graph using the continuous positions
        pos = nx.get_node_attributes(graph, 'pos')
        plt.figure(figsize=(12, 12))
        nx.draw(graph, pos=pos, with_labels=True,
                node_color='skyblue', node_size=500,
                arrowsize=20, arrows=True)
        plt.title("MINERVA Map (Using Continuous Positions)")
        plt.show()


def run_hierarchical_simulation(noise_level=0, num_episodes=10):
    #get the maze details 
    from Maze.Mazes import MazeMaps
    from Maze.Maze_player import MazePlayer
    
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map() 

    #create the maze player
    Maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos_index, goal_index_pos=goal_pos_index)

    #get the goal in screen coordinates
    goal = Maze.get_goal_pos()

    #get player initial position 
    initial_state = Maze.get_initial_player_pos()

    # First collect maze positions through exploration
    def collect_maze_positions(exploration_steps=5000, save_path="maze_positions.csv"):
        """Explore the maze to collect valid positions"""
        import pandas as pd
        import os
        
        print(f"Starting pre-exploration to collect positional data...")
        
        # Check if the file already exists
        if os.path.exists(save_path):
            print(f"Position data already exists at {save_path}, using existing data")
            df = pd.read_csv(save_path)
            return df.values
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Storage for visited positions
        positions = []
        
        # Reset player to initial position
        Maze.reset_player()
        current_state = Maze.get_player_pos()
        
        # Perform random exploration to collect positions
        for step in range(exploration_steps):
            # Store current position
            positions.append(current_state)
            
            # Take random action
            action = np.random.randint(0, 4)  # Assuming 4 possible actions
            Maze.move_player(action)
            current_state = Maze.get_player_pos()
            
            # Print progress occasionally
            if (step + 1) % 1000 == 0:
                print(f"Pre-exploration: {step + 1}/{exploration_steps} steps completed")
                print(f"Unique positions collected: {len(set(map(tuple, positions)))}")
            
            # If goal reached, reset player
            if current_state == Maze.get_goal_pos():
                Maze.reset_player()
                current_state = Maze.get_player_pos()
        
        # Convert positions to DataFrame
        df = pd.DataFrame(positions, columns=['x', 'y'])
        
        # Remove duplicates to get unique positions
        df = df.drop_duplicates()
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        
        print(f"Pre-exploration complete. Collected {len(df)} unique positions.")
        print(f"Position data saved to {save_path}")
        
        return df.values
    
    # Use valid maze positions as training data
    training_data = collect_maze_positions()

    #initialize the hierarchical agent 
    HGWRSOM_agent = HierarchicalGWRSOMAgent(
        lower_dim=1,  # Each coordinate handled separately at lower level
        higher_dim=2, # Full 2D position at higher level
        epsilon_b=0.35,
        epsilon_n=0.15,
        beta=0.7,
        delta=0.79,
        T_max=20,
        N_max=100,
        eta=0.5,
        phi=0.9,
        sigma=0.5
    )

    # Train lower networks with collected maze positions
    HGWRSOM_agent.train_lower_networks(training_data, epochs=20)

    #set a goal 
    HGWRSOM_agent.set_goal(goal=goal)
    HGWRSOM_agent.set_epsilon(1)

    #track training statistics
    training_stats = {
        'episodes': [],
        'steps': [],
        'epsilon': [],
        'success': []
    }

    #set the current state to the initial state
    current_state = initial_state

    #track the number of times the goal has been reached to decay epsilon 
    reached_goal_count = 0

    print(f"\nStarting training with noise level σ² = {noise_level}")

    #start the learning loop 
    for episode_num in range(num_episodes):     
        current_state = initial_state
        Maze.reset_player() 
        step_counter = 0
        episode_success = False

        while current_state != goal and step_counter < 20000:
            step_counter += 1

            # Add noise to state observation if noise_level > 0
            if noise_level > 0:
                noisy_state = current_state + np.random.normal(0, np.sqrt(noise_level), 2)
            else:
                noisy_state = current_state

            # Select action using noisy state
            action = HGWRSOM_agent.select_action(current_state=noisy_state)

            # Execute action
            Maze.move_player(action=action)
            next_state = Maze.get_player_pos() 

            # Update model with true next state
            HGWRSOM_agent.update_model(next_state=next_state, action=action)
            current_state = next_state
            
            if step_counter % 100 == 0: 
                print(f"Episode {episode_num + 1}, step {step_counter}, noise σ² = {noise_level}")

            if current_state == goal:
                episode_success = True
                break

        if episode_success:
            reached_goal_count += 1
            if reached_goal_count > 10: 
                HGWRSOM_agent.decay_epsilon(min_epsilon=0.2) 

        training_stats['episodes'].append(episode_num + 1)
        training_stats['steps'].append(step_counter)
        training_stats['epsilon'].append(HGWRSOM_agent.get_epsilon())
        training_stats['success'].append(episode_success)

        print(f"Episode: {episode_num + 1}, Steps: {step_counter}, "
              f"Epsilon: {HGWRSOM_agent.get_epsilon()}, Success: {episode_success}\n")

    return HGWRSOM_agent, training_stats

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run without noise
    print("\nRunning experiment without noise...")
    agent_no_noise, stats_no_noise = run_hierarchical_simulation(noise_level=0)
    print("\nShowing map without noise:")
    agent_no_noise.show_map()

    # Run with noise
    print("\nRunning experiment with noise σ² = 1/6...")
    agent_with_noise, stats_with_noise = run_hierarchical_simulation(noise_level=1/6)
    print("\nShowing map with noise:")
    agent_with_noise.show_map()

    # Plot training results for both conditions
    plt.figure(figsize=(15, 5))
    
    # Plot steps per episode
    plt.plot(stats_no_noise['episodes'], stats_no_noise['steps'], 'b-', label='No Noise')
    plt.plot(stats_with_noise['episodes'], stats_with_noise['steps'], 'r-', label='With Noise')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()