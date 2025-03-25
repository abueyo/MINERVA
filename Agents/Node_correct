import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

class GWRSOM:
    def __init__(self, a=0.1, h=0.1, en=0.1, es=0.1, an=1.05, ab=1.05, h0=0.5, tb=3.33, tn=14.3, S=1, sigma=0.5):
        self.a = a  # Reduced activity threshold for more node creation
        self.h = h
        self.es = es
        self.en = en
        self.an = an
        self.ab = ab
        self.h0 = h0
        self.tb = tb
        self.tn = tn
        self.S = S
        self.t = 1
        self.A = None
        self.connections = None
        self.ages = None
        self.errors = None
        self.firing_vector = None
        self.max_age = 50
        self.sigma = sigma

    def initialize(self, X):
        """Initialize network with two random nodes"""
        X = X.astype(float)
        w1 = X[np.random.randint(X.shape[0])]
        w2 = X[np.random.randint(X.shape[0])]
        self.A = np.array([w1, w2])
        self.connections = np.zeros((2, 2))
        self.ages = np.zeros((2, 2))
        self.errors = np.zeros(2)
        self.firing_vector = np.ones(2)

    def find_best_matching_units(self, x):
        """Find two best matching units using noise-tolerant distance"""
        x = x.astype(float)
        distances = np.array([self._noisy_distance(x, w) for w in self.A])
        return np.argsort(distances)[:2]

    def _noisy_distance(self, x, w):
        """Compute noise-tolerant distance metric"""
        d = np.linalg.norm(x - w)
        return d * np.exp(-d**2 / (2 * self.sigma**2))

    def _create_connection(self, b, s):
        """Create or refresh connection between nodes"""
        if b < self.connections.shape[0] and s < self.connections.shape[0]:
            self.connections[b, s] = 1
            self.connections[s, b] = 1
            self.ages[b, s] = 0
            self.ages[s, b] = 0

    def _below_activity(self, x, b):
        """Check if activity is below threshold with noise tolerance"""
        w_b = self.A[b]
        activity = np.exp(-self._noisy_distance(x, w_b))
        return activity < self.a * (1 + 0.5 * self.sigma)  # More lenient threshold

    def _below_firing(self, b):
        """Check firing rate threshold"""
        return self.firing_vector[b] < self.h

    def _get_neighbours(self, w):
        """Get indices of connected nodes"""
        return np.where(self.connections[w] == 1)[0]

    def _adapt(self, w, x):
        """Adapt winner and neighbor weights with noise tolerance"""
        x = x.astype(float)
        weight_vector = self.A[w]
        hs = self.firing_vector[w]
        
        # Winner update
        delta = self.es * hs * (x - weight_vector)
        delta = self._clip_update(delta)
        self.A[w] = weight_vector + delta
        
        # Neighbor updates
        b_neighbours = self._get_neighbours(w)
        if len(b_neighbours) > 0:
            w_neighbours = self.A[b_neighbours]
            hi = self.firing_vector[b_neighbours]
            delta = self.en * np.multiply(hi.reshape(-1, 1), (x - w_neighbours))
            delta = self._clip_update(delta)
            self.A[b_neighbours] = w_neighbours + delta

    def _clip_update(self, delta):
        """Limit weight updates based on noise level"""
        max_update = 2.0 * self.sigma
        return np.clip(delta, -max_update, max_update)

    def _age(self, w):
        """Age connections to neighbors"""
        b_neighbours = self._get_neighbours(w)
        self.ages[w, b_neighbours] += 1
        self.ages[b_neighbours, w] += 1

    def _reduce_firing(self, w):
        """Update firing rates"""
        t = self.t
        self.firing_vector[w] = self.h0 - (self.S / self.ab) * (1 - np.exp(-self.ab * t / self.tb))
        b_neighbours = self._get_neighbours(w)
        if len(b_neighbours) > 0:
            self.firing_vector[b_neighbours] = self.h0 - (self.S / self.an) * (1 - np.exp(-self.an * t / self.tn))

    def remove_old_edges(self):
        """Remove old edges and isolated nodes"""
        if self.A is None or len(self.A) < 3:  # Keep at least 2 nodes
            return
            
        self.connections[self.ages > self.max_age] = 0
        self.ages[self.ages > self.max_age] = 0
        nNeighbour = np.sum(self.connections, axis=0)
        NodeIndices = np.array(list(range(self.A.shape[0])))
        AloneNodes = NodeIndices[np.where(nNeighbour == 0)]
        
        if len(AloneNodes) > 0 and (self.A.shape[0] - len(AloneNodes)) >= 2:
            self.connections = np.delete(self.connections, AloneNodes, axis=0)
            self.connections = np.delete(self.connections, AloneNodes, axis=1)
            self.ages = np.delete(self.ages, AloneNodes, axis=0)
            self.ages = np.delete(self.ages, AloneNodes, axis=1)
            self.A = np.delete(self.A, AloneNodes, axis=0)
            self.firing_vector = np.delete(self.firing_vector, AloneNodes)
            self.errors = np.delete(self.errors, AloneNodes)

    def _add_new_node(self, b1, b2, x):
        """Add new node and update network structure"""
        w_b1 = self.A[b1]
        weight_vector = (w_b1 + x) / 2
        self.A = np.vstack((self.A, weight_vector))
        n = self.A.shape[0]
        
        # Expand matrices
        self.connections = np.pad(self.connections, ((0, 1), (0, 1)))
        self.ages = np.pad(self.ages, ((0, 1), (0, 1)))
        self.firing_vector = np.append(self.firing_vector, 1)
        self.errors = np.append(self.errors, 0)

        # Create connections
        self._create_connection(b1, n - 1)
        self._create_connection(b2, n - 1)
        self.connections[b1, b2] = 0
        self.connections[b2, b1] = 0

    def train(self, X, epochs=100):
        """Train the network"""
        X = X.astype(float)
        if self.A is None:
            self.initialize(X)
            
        for _ in range(epochs):
            np.random.shuffle(X)  # Randomize training order
            for x in X:
                b1, b2 = self.find_best_matching_units(x)
                
                # More aggressive node creation
                if self._below_activity(x, b1):
                    self._add_new_node(b1, b2, x)
                else:
                    self._adapt(b1, x)
                    self._age(b1)
                    self._reduce_firing(b1)
                
                self.t += 1
                if self.t % 10 == 0:  # Periodic cleanup
                    self.remove_old_edges()

class HierarchicalGWRSOMAgent:
    def __init__(self, lower_dim=1, higher_dim=2, epsilon_b=0.35, epsilon_n=0.15,
                 beta=0.7, delta=0.79, T_max=20, N_max=100, eta=0.5, phi=0.9, sigma=0.5):
        # Initialize lower networks with balanced parameters
        self.lower_x = GWRSOM(a=0.8, h=0.1, en=0.1, es=0.1, sigma=sigma)
        self.lower_y = GWRSOM(a=0.8, h=0.1, en=0.1, es=0.1, sigma=sigma)
        
        # Initialize higher level storage
        self.higher_nodes = []
        self.higher_weights = []
        self.higher_connections = np.zeros((0, 0))
        
        # Parameters
        self.epsilon = 0.5
        self.start_epsilon = 0.5
        self.goal = None
        self.sigma = sigma
        self.max_nodes = 50  # Target maximum nodes
        self.merge_threshold = 0.3  # Distance threshold for merging
        self.min_connection_age = 5  # Minimum age before pruning
        
        # Node usage tracking
        self.node_ages = {}  # Track node ages
        self.node_usage = {}  # Track node usage frequency
        self.connection_ages = {}  # Track connection ages
        
        # State tracking
        self.prev_node_idx = None
        self.is_plan = None
        self.expected_next_state = None
        self.active_neurons = []
        self.firing_combinations = {}
        self.lower_networks_trained = False
        self.position_to_firing = {}
        self.step_counter = 0

    def train_lower_networks(self, training_data, epochs=100):
        """Train lower networks with balanced parameters"""
        x_data = training_data[:, 0].reshape(-1, 1)
        y_data = training_data[:, 1].reshape(-1, 1)
        
        # Train networks
        self.lower_x.train(x_data, epochs=epochs)
        self.lower_y.train(y_data, epochs=epochs)
        
        # Initialize firing patterns
        for position in training_data:
            x = position[0].reshape(1, -1)
            y = position[1].reshape(1, -1)
            x_node, _ = self.lower_x.find_best_matching_units(x)
            y_node, _ = self.lower_y.find_best_matching_units(y)
            pos_key = tuple(position)
            self.position_to_firing[pos_key] = (x_node, y_node)
        
        self.lower_networks_trained = True

    def _should_merge_nodes(self, pattern1, pattern2, pos1, pos2):
        """Determine if two nodes should be merged based on pattern and position similarity"""
        pattern_dist = np.sqrt((pattern1[0] - pattern2[0])**2 + (pattern1[1] - pattern2[1])**2)
        pos_dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
        return pattern_dist < self.merge_threshold and pos_dist < self.merge_threshold * 2

    def _merge_nodes(self, idx1, idx2):
        """Merge two nodes and their connections"""
        # Average positions
        pos1, pos2 = np.array(self.higher_weights[idx1]), np.array(self.higher_weights[idx2])
        merged_pos = (pos1 + pos2) / 2
        
        # Combine connections
        self.higher_connections[idx1, :] = np.maximum(
            self.higher_connections[idx1, :],
            self.higher_connections[idx2, :]
        )
        self.higher_connections[:, idx1] = np.maximum(
            self.higher_connections[:, idx1],
            self.higher_connections[:, idx2]
        )
        
        # Update node
        self.higher_weights[idx1] = merged_pos
        
        # Remove second node
        self.higher_nodes.pop(idx2)
        self.higher_weights.pop(idx2)
        self.higher_connections = np.delete(self.higher_connections, idx2, axis=0)
        self.higher_connections = np.delete(self.higher_connections, idx2, axis=1)
        
        # Update indices in firing combinations
        for pattern, idx in list(self.firing_combinations.items()):
            if idx == idx2:
                self.firing_combinations[pattern] = idx1
            elif idx > idx2:
                self.firing_combinations[pattern] = idx - 1

    def _prune_network(self):
        """Remove unused nodes and connections"""
        if len(self.higher_nodes) <= self.max_nodes:
            return
            
        # Calculate node usage scores
        usage_scores = np.sum(self.higher_connections, axis=1) + np.sum(self.higher_connections, axis=0)
        
        while len(self.higher_nodes) > self.max_nodes:
            # Find least used node
            idx_to_remove = np.argmin(usage_scores)
            
            # Remove node
            self.higher_nodes.pop(idx_to_remove)
            self.higher_weights.pop(idx_to_remove)
            self.higher_connections = np.delete(self.higher_connections, idx_to_remove, axis=0)
            self.higher_connections = np.delete(self.higher_connections, idx_to_remove, axis=1)
            
            # Update firing combinations
            for pattern, idx in list(self.firing_combinations.items()):
                if idx == idx_to_remove:
                    del self.firing_combinations[pattern]
                elif idx > idx_to_remove:
                    self.firing_combinations[pattern] = idx - 1
            
            # Update usage scores
            usage_scores = np.delete(usage_scores, idx_to_remove)

    def create_new_node(self, firing_combination, position):
        """Create new node with controlled growth"""
        # Check for similar existing nodes
        for idx, (pattern, pos) in enumerate(zip(self.higher_nodes, self.higher_weights)):
            if self._should_merge_nodes(firing_combination, pattern, position, pos):
                return idx
        
        # Create new node if below max
        if len(self.higher_nodes) < self.max_nodes:
            new_idx = len(self.higher_nodes)
            self.higher_nodes.append(firing_combination)
            self.higher_weights.append(position)
            
            # Expand connection matrix
            new_connections = np.zeros((new_idx + 1, new_idx + 1))
            if new_idx > 0:
                new_connections[:-1, :-1] = self.higher_connections
            self.higher_connections = new_connections
            
            return new_idx
        else:
            # Find most similar existing node
            best_idx = 0
            best_dist = float('inf')
            for idx, (pattern, pos) in enumerate(zip(self.higher_nodes, self.higher_weights)):
                pattern_dist = np.sqrt((firing_combination[0] - pattern[0])**2 + 
                                     (firing_combination[1] - pattern[1])**2)
                pos_dist = np.linalg.norm(np.array(position) - np.array(pos))
                total_dist = pattern_dist + pos_dist
                if total_dist < best_dist:
                    best_dist = total_dist
                    best_idx = idx
            return best_idx

    def update_model(self, next_state, action):
        """Update model with controlled growth"""
        if not self.lower_networks_trained:
            raise Exception("Lower networks must be trained first!")
        
        firing_combination = self.get_firing_combination(next_state)
        curr_node_idx = self.create_new_node(firing_combination, next_state)
        
        # Update connections
        if self.prev_node_idx is not None:
            self.higher_connections[self.prev_node_idx, curr_node_idx] = 1
            self.higher_connections[curr_node_idx, self.prev_node_idx] = 1
        
        self.prev_node_idx = curr_node_idx
        self.step_counter += 1
        
        # Periodic maintenance
        if self.step_counter % 100 == 0:
            self._prune_network()

    def select_action(self, current_state):
        """Select action with improved exploration"""
        if self.goal is None:
            raise Exception("No goal defined")
        
        if np.random.uniform(0, 1) > self.epsilon:
            firing_combination = self.get_firing_combination(current_state)
            curr_node_idx = self.create_new_node(firing_combination, current_state)
            
            if self.prev_node_idx is not None:
                self.higher_connections[self.prev_node_idx, curr_node_idx] = 1
                self.higher_connections[curr_node_idx, self.prev_node_idx] = 1
                possible_actions = np.where(self.higher_connections[curr_node_idx] == 1)[0]
                
                if len(possible_actions) > 0:
                    # Consider goal direction
                    goal_dir = np.array(self.goal) - np.array(current_state)
                    best_action = None
                    best_alignment = -float('inf')
                    
                    for action in possible_actions:
                        next_pos = self.get_expected_next_state(current_state, action % 4)
                        action_dir = next_pos - np.array(current_state)
                        alignment = np.dot(goal_dir, action_dir)
                        
                        if alignment > best_alignment:
                            best_alignment = alignment
                            best_action = action % 4
                    
                    if best_action is not None:
                        self.is_plan = True
                        self.expected_next_state = self.get_expected_next_state(current_state, best_action)
                        return best_action
            
            action = random.randint(0, 3)
            self.is_plan = False
            return action
        else:
            action = random.randint(0, 3)
            self.is_plan = False
            return action

    def get_firing_combination(self, current_state):
        """Get firing pattern with noise tolerance"""
        if not self.lower_networks_trained:
            raise Exception("Lower networks must be trained first!")
        
        x = np.array([current_state[0]], dtype=float).reshape(1, -1)
        y = np.array([current_state[1]], dtype=float).reshape(1, -1)
        
        x_node, _ = self.lower_x.find_best_matching_units(x)
        y_node, _ = self.lower_y.find_best_matching_units(y)
        
        firing_pattern = (x_node, y_node)
        self.active_neurons = firing_pattern
        return firing_pattern

    def get_expected_next_state(self, current_state, action):
        """Calculate expected next state"""
        current_state = np.array(current_state)
        action = int(action) % 4
        
        actions = {
            0: np.array([0, 1]),   # Up
            1: np.array([0, -1]),  # Down
            2: np.array([1, 0]),   # Right
            3: np.array([-1, 0])   # Left
        }
        
        return current_state + actions[action]

    def set_goal(self, goal):
        self.goal = goal

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_epsilon(self):
        return self.epsilon

    def reset_epsilon(self):
        self.epsilon = self.start_epsilon

    def decay_epsilon(self, min_epsilon=0.2):
        self.epsilon = max(round(self.epsilon - 0.1, 5), min_epsilon)