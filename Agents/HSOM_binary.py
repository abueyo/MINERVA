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
    """
    Growing When Required Self-Organizing Map (GWRSOM) implementation
    Based on the algorithm in 'A self-organising network that grows when required'
    with improved stability and topological preservation
    """
    def __init__(self, a=0.1, h=0.1, en=0.05, es=0.2, an=1.05, ab=1.05, h0=1.0, tb=3.33, tn=14.3, S=0.3):
        """
        Initialize GWRSOM with parameters
        
        Parameters:
        a: Activity threshold (lower values create fewer nodes)
        h: Firing threshold (higher values allow more nodes)
        en: Neighbor learning rate
        es: Winner learning rate
        an: Firing curve parameter for neighbors
        ab: Firing curve parameter for winner
        h0: Initial firing value
        tb: Time constant for winner
        tn: Time constant for neighbors
        S: Stimulus strength
        """
        self.a = a              # Activity threshold
        self.h = h              # Firing threshold
        self.es = es            # Winner learning rate
        self.en = en            # Neighbor learning rate
        self.an = an            # Firing curve parameter for neighbors
        self.ab = ab            # Firing curve parameter for winner
        self.h0 = h0            # Initial firing value
        self.tb = tb            # Time constant for winner
        self.tn = tn            # Time constant for neighbors
        self.S = S              # Stimulus strength
        self.t = 1              # Timestep
        
        # These will be initialized later when we see data
        self.A = None           # Node matrix A (weight vectors)
        self.connections = None # Connection matrix
        self.ages = None        # Age of connections
        self.errors = None      # Error per node
        self.firing_vector = None # Firing counter for each node
        
        # Parameters for network management
        self.max_age = 50       # Maximum age of connections
        self.sigma = 0.3        # Neighborhood width for topological preservation
        
        # Debug flag
        self.debug = False
   
    def Distance(self, x1, x2):
        """Calculate Euclidean distance between two vectors"""
        return np.linalg.norm(x1 - x2)
   
    def initialize(self, X):
        """
        Initialize the network with two random nodes from input data
        
        Parameters:
        X: Input data [samples, features]
        """
        # Ensure data is float type
        X = X.astype(float)
        unique_X = np.unique(X, axis=0)
        # pick two random points
        if len(unique_X) >= 2:
            indices = np.random.choice(len(unique_X), 2, replace=False)
            w1 = unique_X[indices[0]] 
            w2 = unique_X[indices[1]] 
        else:
            raise ValueError("Not enough unique data points to initialize network.") 
        # Initialize node matrix with the two weights
        self.A = np.array([w1, w2])
        
          
        
        
        # Initialize connection matrices
        self.connections = np.zeros((2, 2))  # No initial connections
        self.ages = np.zeros((2, 2))         # Ages of connections
        self.errors = np.zeros(2)            # Error tracking per node
        self.firing_vector = np.ones(2)      # Initial firing is 1 for all nodes
        
        if self.debug:
            logger.info(f"Initialized network with 2 nodes: {w1} and {w2}")

    def find_best_matching_units(self, x):
        x = x.astype(float)

        if self.A is None or len(self.A) == 0:
            raise ValueError("No nodes in the network.")

        distances = np.linalg.norm(self.A - x, axis=1)

        bmu_indices = np.argsort(distances)

        if len(bmu_indices) == 0:
            return [0, 0]  # fallback case — should rarely happen
        elif len(bmu_indices) == 1:
            return [int(bmu_indices[0]), int(bmu_indices[0])]
        else:
            return [int(bmu_indices[0]), int(bmu_indices[1])]



    def _create_connection(self, b, s):
        """
        Create or reset connection between two nodes
        
        Parameters:
        b: Index of first node
        s: Index of second node
        """
        if self.connections[b, s] and self.connections[s, b]:
            # Connection already exists, reset age
            self.ages[b, s] = 0
            self.ages[s, b] = 0
        else:
            # Create new connection
            self.connections[b, s] = 1
            self.connections[s, b] = 1
            self.ages[b, s] = 0
            self.ages[s, b] = 0

    def _below_activity(self, x, b):
        """
        Check if activity (similarity) between input and BMU is below threshold
        
        Parameters:
        x: Input vector
        b: Index of best matching unit
        
        Returns:
        True if activity is below threshold (node is far from input)
        """
        w_b = self.A[b]
        distance = np.linalg.norm(x - w_b)
        activity = np.exp(-distance)  # Higher = more similar
         
        # # Add debug print
        # print(f"Input: {x}, Node: {w_b}, Distance: {distance:.4f}, Activity: {activity:.6f}, Threshold: {self.a:.6f}")
        # print(f"Is activity below threshold? {activity < self.a}")
        return activity < self.a  # True if activity is low (node is far)

    def _below_firing(self, b):
        """
        Check if firing rate of BMU is below threshold
        
        Parameters:
        b: Index of best matching unit
        
        Returns:
        True if firing rate is below threshold
        """
        
        value = self.firing_vector[b] < self.h
        # print(f"Node {b} firing: {self.firing_vector[b]:.4f}, threshold: {self.h:.4f}, below? {value}")
        return value
    
    def _calculate_activity(self, x, b):
        w_b = self.A[b]
        distance = np.linalg.norm(x - w_b)
        activity = np.exp(-distance)
        return activity

    def _should_add_node(self, x, b1):
        activity = self._calculate_activity(x, b1)
        firing = self.firing_vector[b1]
        
        # Add node if:
        # 1. Activity is low (input is poorly represented) AND
        # 2. Firing is high (node hasn't been updated much)
        return (activity < self.a) and (firing > self.h)

    def _add_new_node(self, b1, b2, x):
        """
        Add a new node between best matching unit and input
        
        Parameters:
        b1: Index of best matching unit
        b2: Index of second best matching unit
        x: Input vector
        """
        if self.debug:
            logger.info(f"Adding new node between node {b1} and input {x}")
            
        # Calculate new node's weight vector
        w_b1 = self.A[b1]
        weight_vector = x.copy()
        
        # Add new node to weight matrix
        self.A = np.vstack((self.A, weight_vector))
        
        # Get new node's index
        n = self.A.shape[0]
        
        # Expand connection matrices using np.pad (more efficient than stacking)
        self.connections = np.pad(self.connections, ((0, 1), (0, 1)))
        self.ages = np.pad(self.ages, ((0, 1), (0, 1)))
        
        # Initialize firing and error for new node
        self.firing_vector = np.append(self.firing_vector, 1)
        self.errors = np.append(self.errors, 0)

        # Create connections from new node to BMUs
        self._create_connection(b1, n - 1)
        self._create_connection(b2, n - 1)
        
        # Remove connection between BMUs (they're now connected through the new node)
        self.connections[b1, b2] = 0
        self.connections[b2, b1] = 0

        # Remove old edges
        self.remove_old_edges()
        
        if self.debug:
            logger.info(f"New node created at index {n-1} with weight {weight_vector}")

    def remove_old_edges(self):
        """
        Remove connections older than max_age and any resulting isolated nodes
        """
        # Remove old connections
        self.connections[self.ages > self.max_age] = 0
        self.ages[self.ages > self.max_age] = 0
        
        # Find isolated nodes (nodes with no connections)
        nNeighbour = np.sum(self.connections, axis=0)
        NodeIndisces = np.array(list(range(self.A.shape[0])))
        AloneNodes = NodeIndisces[np.where(nNeighbour == 0)]
        
        # Remove isolated nodes if there are more than 2 nodes total
        if AloneNodes.any() and self.A.shape[0] > 2:
            self.connections = np.delete(self.connections, AloneNodes, axis=0)
            self.connections = np.delete(self.connections, AloneNodes, axis=1)
            self.ages = np.delete(self.ages, AloneNodes, axis=0)
            self.ages = np.delete(self.ages, AloneNodes, axis=1)
            self.A = np.delete(self.A, AloneNodes, axis=0)
            self.firing_vector = np.delete(self.firing_vector, AloneNodes)
            self.errors = np.delete(self.errors, AloneNodes)
            
            if self.debug:
                logger.info(f"Removed {len(AloneNodes)} isolated nodes")

    def _best(self, x):
        """
        Find best matching units and create connection between them
        
        Parameters:
        x: Input vector
        
        Returns:
        Indices of two best matching units
        """
        # Find two closest nodes
        b1, b2 = self.find_best_matching_units(x)
        
        # Create or reset connection between them
        self._create_connection(b1, b2)
        
        return b1, b2

    def _get_neighbours(self, w):
        """
        Get boolean mask of neighbors connected to node w
        
        Parameters:
        w: Index of node
        
        Returns:
        Boolean array with True at indices of neighbors
        """
        return self.connections[w, :].astype(bool)

    def _adapt(self, w, x):
        """
        Adapt winner node and its neighbors toward input
        
        Parameters:
        w: Index of winner node
        x: Input vector
        """
        # Ensure input is float
        x = x.astype(float)
        
        # Get current weight of winner
        weight_vector = self.A[w]
        
        # Calculate winner adaptation based on firing rate
        hs = self.firing_vector[w]
        delta = self.es * hs * (x - weight_vector)
        new_position = weight_vector + delta
        
        # Update winner node with rounding for discrete positions
        self.A[w] = (new_position)
        
        # Get neighbors
        b_neighbours = self._get_neighbours(w)
        
        # Only update neighbors if there are any
        if np.any(b_neighbours):
            # Get weights and firing rates of neighbors
            w_neighbours = self.A[b_neighbours]
            hi = self.firing_vector[b_neighbours]

            # Calculate topological neighborhood influence
            distances = np.array([self.Distance(self.A[w], neighbor) for neighbor in w_neighbours])
            influences = np.exp(-distances**2 / (2 * self.sigma**2))

            # Update neighbors with topological preservation
            delta = self.en * np.multiply(hi.reshape(-1, 1) * influences.reshape(-1, 1), (x - w_neighbours))
            self.A[b_neighbours] = w_neighbours + delta
            
            if self.debug:
                logger.debug(f"Updated node {w} and {np.sum(b_neighbours)} neighbors")

    def _age(self, w):
        """
        Increase age of all connections to/from node w
        
        Parameters:
        w: Index of winner node
        """
        # Get neighbor indices
        b_neighbours = self._get_neighbours(w)
        
        # Increment ages of all connections to neighbors
        self.ages[w, b_neighbours] += 1
        self.ages[b_neighbours, w] += 1

    def _reduce_firing(self, w):
        """
        Reduce firing counter for winner node and its neighbors
        
        Parameters:
        w: Winner node index
        """
        # Current timestep
        t = self.t
        old_firing = self.firing_vector[w]
        # Update winner's firing rate
        self.firing_vector[w] = self.h0 - self.S / self.ab * (1 - np.exp(-self.ab * t / self.tb))
        # Add debug print
        # print(f"Node {w} - Firing before: {old_firing:.4f}, after: {self.firing_vector[w]:.4f}, t={self.t}")

        # Update neighbors' firing rates
        b_neighbours = self._get_neighbours(w)
        if np.any(b_neighbours):
            self.firing_vector[b_neighbours] = self.h0 - self.S / self.an * (1 - np.exp(-self.an * t / self.tn))

    def train(self, X, epochs=1):
        """
        Train the network on input data
        
        Parameters:
        X: Input data [samples, features]
        epochs: Number of training epochs
        
        Returns:
        self: For method chaining
        """
        # Ensure data is float type
        X = X.astype(float)
        
        # Initialize network if not already done
        if self.A is None:
            self.initialize(X)
            
        # Training loop
        for epoch in range(epochs):
            for i, x in enumerate(X):
              
                # Find best matching units
                b1, b2 = self._best(x)
              
                # Check if we need to add a new node
                # Note: Logic matches the improved version, adding node when activity is low and firing is high
                activity_below = self._below_activity(x, b1)
                firing_above = not self._below_firing(b1)
                
                if activity_below and firing_above:
                    self._add_new_node(b1, b2, x)
                else:
                    # Adapt existing nodes
                    self._adapt(b1, x)
                    self._age(b1)
                    self._reduce_firing(b1)
                
                # Increment timestep
                self.t += 1
                
        return self

    def get_weights(self):
        """Get the weight vectors of all nodes"""
        return self.A

    def get_connections(self):
        """Get the connection matrix"""
        return self.connections


class Value:
    """Fixed Value computation class for HSOM"""
    def __init__(self, num_nodes=0):
        self.V = np.zeros(num_nodes)  # Value function
        self.R = np.zeros(num_nodes)  # Reward function
        self.w_g = None  # Index of goal node
        self.initialized = False

    def Distance(self, x1, x2):
        """Calculate Euclidean distance between two vectors"""
        x1_array = np.array(x1)
        x2_array = np.array(x2)
        return np.linalg.norm(x1_array - x2_array)

    def _resize_if_needed(self, num_nodes):
        """Resize value and reward arrays if network has grown, preserving existing values"""
        if len(self.V) < num_nodes:
            # Preserve existing values and extend with zeros
            old_V = self.V.copy()
            old_R = self.R.copy()
            
            self.V = np.zeros(num_nodes)
            self.R = np.zeros(num_nodes)
            
            # Copy old values back
            if len(old_V) > 0:
                self.V[:len(old_V)] = old_V
                self.R[:len(old_R)] = old_R
            
            self.initialized = False  # Need to recompute rewards for new nodes

    def ComputeReward(self, node_positions, connections, goal):
        """Compute reward function based on distance to goal"""
        num_nodes = len(node_positions)
        self._resize_if_needed(num_nodes)
        
        if num_nodes == 0:
            return
        
        # Find node closest to goal
        distances = []
        for i, pos in enumerate(node_positions):
            distances.append(self.Distance(goal, pos))
        
        self.w_g = np.argmin(distances)
        
        # Set rewards based on distance to goal
        for i in range(num_nodes):
            if i == self.w_g:
                self.R[i] = 10  # High reward for goal node
            else:
                # Exponential decay based on distance to goal
                pos_i = node_positions[i]
                distance = self.Distance(goal, pos_i)
                self.R[i] = np.exp(-distance**2 / 200)  # Adjusted scale factor

    def ComputeValue(self, node_positions, connections, goal, gamma=0.99, max_iterations=50):
        """
        Compute value function using value iteration with proper convergence
        
        Parameters:
        - node_positions: List of node positions
        - connections: Adjacency matrix
        - goal: Goal position
        - gamma: Discount factor
        - max_iterations: Maximum number of value iterations
        """
        num_nodes = len(node_positions)
        
        if num_nodes == 0:
            return np.array([])
        
        # Resize arrays if needed (preserving existing values)
        self._resize_if_needed(num_nodes)
        
        # Compute rewards (only if not initialized or if goal changed)
        self.ComputeReward(node_positions, connections, goal)
        
        # Value iteration with convergence check
        prev_V = self.V.copy()
        
        for iteration in range(max_iterations):
            new_V = self.V.copy()
            
            for i in range(num_nodes):
                # Get neighboring nodes
                if i < len(connections):
                    neighbors = np.where(connections[i, :] == 1)[0]
                    
                    if len(neighbors) > 0:
                        # Calculate maximum value from neighbors
                        neighbor_values = self.V[neighbors]
                        max_neighbor_value = np.max(neighbor_values) if len(neighbor_values) > 0 else 0
                        
                        # Update value using Bellman equation
                        new_V[i] = self.R[i] + gamma * max_neighbor_value
                    else:
                        # No neighbors, just use reward
                        new_V[i] = self.R[i]
                else:
                    # Node index beyond connection matrix, use reward only
                    new_V[i] = self.R[i]
            
            # Check for convergence
            if np.allclose(new_V, self.V, rtol=1e-6):
                break
                
            self.V = new_V
        
        self.initialized = True
        return self.V


class Action:
    """Fixed Action selection class for HSOM"""
    def __init__(self):
        self.indEX = None  # Expected next node index

    def Distance(self, x1, x2):
        """Calculate Euclidean distance between two vectors"""
        x1_array = np.array(x1)
        x2_array = np.array(x2)
        return np.linalg.norm(x1_array - x2_array)

    def actionSelect(self, state, node_positions, values, connections, action_mappings):
        """
        Select action based on current state and value function with improved fallback
        
        Parameters:
        - state: Current state position
        - node_positions: List of all node positions  
        - values: Value function for all nodes
        - connections: Adjacency matrix
        - action_mappings: Dictionary mapping (from_node, to_node) -> action
        """
        if len(node_positions) == 0:
            return random.randint(0, 3)
        
        # Find closest node to current state
        min_dist = float('inf')
        current_node_idx = None
        
        for i, pos in enumerate(node_positions):
            dist = self.Distance(state, pos)
            if dist < min_dist:
                min_dist = dist
                current_node_idx = i
        
        if current_node_idx is None:
            return random.randint(0, 3)
        
        # Find connected nodes
        if current_node_idx < len(connections):
            connected_nodes = np.where(connections[current_node_idx, :] == 1)[0]
        else:
            connected_nodes = np.array([])
        
        if len(connected_nodes) == 0:
            # No connections yet, explore randomly
            return random.randint(0, 3)
        
        # Find node with highest value among connected nodes
        best_neighbor_idx = None
        best_value = -float('inf')
        
        for neighbor_idx in connected_nodes:
            if neighbor_idx < len(values) and values[neighbor_idx] > best_value:
                best_value = values[neighbor_idx]
                best_neighbor_idx = neighbor_idx
        
        if best_neighbor_idx is None:
            return random.randint(0, 3)
        
        # Set expected next node for explainability
        self.indEX = best_neighbor_idx
        
        # Return action that leads to best neighbor
        key = (current_node_idx, best_neighbor_idx)
        if key in action_mappings:
            return action_mappings[key]
        else:
            # Fallback: try to move toward the best neighbor
            current_pos = np.array(node_positions[current_node_idx])
            target_pos = np.array(node_positions[best_neighbor_idx])
            
            # Calculate direction vector
            direction = target_pos - current_pos
            
            # Choose action based on dominant direction
            if abs(direction[1]) > abs(direction[0]):  # Y movement is larger
                if direction[1] > 0:
                    return 1  # Down
                else:
                    return 0  # Up
            else:  # X movement is larger  
                if direction[0] > 0:
                    return 2  # Right
                else:
                    return 3  # Left


class HierarchicalGWRSOMAgent:
    def __init__(self, lower_dim=1, higher_dim=2, epsilon_b=0.35, epsilon_n=0.15, 
                 beta=0.7, delta=0.5, T_max=20, N_max=300, eta=0.5, phi=0.9, sigma=0.5):
        # Initialize lower level networks
        self.lower_x = GWRSOM(a=0.0001, h=0.0001)
        self.lower_y = GWRSOM(a=0.0001, h=0.0001)
        
        self.seen_bmu_pairs = set()
        self.layer1_insertions = 0
        self.layer1_blocks = 0

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

        self.state_node_coverage = {}  # Maps state (rounded) to node index

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
        if not isinstance(x_bmus, (list, tuple)) or len(x_bmus) < 2:
            raise ValueError(f"Expected list of at least 2 BMU indices, got: {x_bmus}")
        x_bmus_id = x_bmus[0]
        y_bmus = self.lower_y.find_best_matching_units(y_data)
        if not isinstance(y_bmus, (list, tuple)) or len(y_bmus) < 2:
            raise ValueError(f"Expected list of at least 2 BMU indices, got: {x_bmus}")
        y_bmus_id = y_bmus[0]
        
        
        pair = tuple(sorted((x_bmus_id, y_bmus_id)))

        # if pair not in self.seen_bmu_pairs:
        #     self.seen_bmu_pairs.add(pair)
        #     print(f"New BMU pair added to Layer 1 input: {pair}")
        # else:
        #     print(f"Repeated BMU pair (not added again): {pair}")

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
        
        # print("[DEBUG] BMUs:", x_bmus[0], y_bmus[0])  # assuming one BMU per input
        # print("[DEBUG] Binary pattern key:", tuple(x_binary), tuple(y_binary))
        
        return np.array(x_binary), np.array(y_binary)

    def find_node_index(self, pattern):
        for i, node_data in enumerate(self.nodes):
            stored_pattern = node_data[0]
            if (np.array_equal(stored_pattern[0], pattern[0]) and
                np.array_equal(stored_pattern[1], pattern[1])):
                return i
        return None


    def update_model(self, next_state, action):
        """Update the model with a new state-action pair"""
        pattern = self.get_firing_pattern(next_state)
        node_idx = self.find_node_index(pattern)
    
        if node_idx is None:
            # print(f"[DEBUG] NEW NODE created for pattern: ({pattern[0].argmax()}, {pattern[1].argmax()})")
            # print(f"[DEBUG] Total high-level nodes before adding: {len(self.nodes)}")

            self.nodes.append((pattern, next_state))
            new_size = len(self.nodes)

            # Grow or initialize connections
            if self.connections.size == 0:
                self.connections = np.zeros((1, 1))
                self.pattern_ages = np.zeros((1, 1))
            else:
                new_connections = np.zeros((new_size, new_size))
                new_ages = np.zeros((new_size, new_size))
                new_connections[:-1, :-1] = self.connections
                new_ages[:-1, :-1] = self.pattern_ages
                self.connections = new_connections
                self.pattern_ages = new_ages

            node_idx = new_size - 1

        else:
            # Update position with weighted average
            old_pattern, old_position = self.nodes[node_idx]
            updated_position = 0.9 * np.array(old_position) + 0.1 * np.array(next_state)
            self.nodes[node_idx] = (old_pattern, updated_position)

        # Create connection from previous node
        if self.prev_node_idx is not None:
            self.connections[self.prev_node_idx, node_idx] = 1
            self.pattern_ages[self.prev_node_idx, node_idx] = 0
            self.action_mappings[(self.prev_node_idx, node_idx)] = action

            # Age other connections from previous node
            connected = np.where(self.connections[self.prev_node_idx] == 1)[0]
            for c in connected:
                if c != node_idx:
                    self.pattern_ages[self.prev_node_idx, c] += 1

            # Prune old connections
            old = self.pattern_ages > self.T_max
            self.connections[old] = 0
            self.pattern_ages[old] = 0
        # Track which valid position maps to which node
        rounded_pos = tuple(np.round(next_state[:2]).astype(int))
        self.state_node_coverage[rounded_pos] = node_idx
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
            
            # Create a list of node positions for value computation
            # This is the key modification - use only positions for distance calculations
            node_positions = [pos for _, pos in self.nodes]
                
            # Compute value function using positions, not patterns
            V = self.ValueClass.ComputeValue(node_positions, self.connections, self.goal)
            
            # Use action selection mechanism with positions
            action = self.ActionClass.actionSelect(
                current_state, 
                node_positions,  # Pass positions instead of the full nodes
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