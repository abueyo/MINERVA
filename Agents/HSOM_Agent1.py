import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

class GWRSOM:
    def __init__(self, a=0.1, h=0.1, en=0.1, es=0.1, an=1.05, ab=1.05, h0=0.5, tb=3.33, tn=14.3, S=1):
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
        self.t = 1
        self.A = None
        self.connections = None
        self.ages = None
        self.errors = None
        self.firing_vector = None
        self.max_age = 50

    def initialize(self, X):
        X = X.astype(float)
        w1 = np.round(X[np.random.randint(X.shape[0])])
        w2 = np.round(X[np.random.randint(X.shape[0])])
        self.A = np.array([w1, w2])
        self.connections = np.zeros((2, 2))
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

    def _get_neighbours(self, w):
        return np.where(self.connections[w] == 1)[0]

    def _adapt(self, w, x):
        x = x.astype(float)
        weight_vector = self.A[w]
        hs = self.firing_vector[w]
        
        delta = self.es * hs * (x - weight_vector)
        self.A[w] = np.round(weight_vector + delta)
        
        b_neighbours = self._get_neighbours(w)
        if len(b_neighbours) > 0:
            w_neighbours = self.A[b_neighbours]
            hi = self.firing_vector[b_neighbours]
            delta = self.en * np.multiply(hi.reshape(-1, 1), (x - w_neighbours))
            self.A[b_neighbours] = np.round(w_neighbours + delta)

    def _age(self, w):
        b_neighbours = self._get_neighbours(w)
        self.ages[w, b_neighbours] += 1
        self.ages[b_neighbours, w] += 1

    def _reduce_firing(self, w):
        t = self.t
        self.firing_vector[w] = self.h0 - (self.S / self.ab) * (1 - np.exp(-self.ab * t / self.tb))
        b_neighbours = self._get_neighbours(w)
        if len(b_neighbours) > 0:
            self.firing_vector[b_neighbours] = self.h0 - (self.S / self.an) * (1 - np.exp(-self.an * t / self.tn))

    def remove_old_edges(self):
        if self.A is None or len(self.A) < 3:
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
        w_b1 = self.A[b1]
        weight_vector = np.round((w_b1 + x) / 2)
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

    def train(self, X, epochs=100):
        X = X.astype(float)
        if self.A is None:
            self.initialize(X)
            
        for _ in range(epochs):
            np.random.shuffle(X)
            for x in X:
                b1, b2 = self.find_best_matching_units(x)
                
                if self._below_activity(x, b1) and self._below_firing(b1):
                    self._add_new_node(b1, b2, x)
                else:
                    self._adapt(b1, x)
                    self._age(b1)
                    self._reduce_firing(b1)
                
                self.t += 1
                if self.t % 10 == 0:
                    self.remove_old_edges()

class HierarchicalGWRSOMAgent:
    def __init__(self, lower_dim=1, higher_dim=2, epsilon_b=0.35, epsilon_n=0.15,
                 beta=0.7, delta=0.79, T_max=20, N_max=100, eta=0.5, phi=0.9, sigma=0.5):
        self.lower_x = GWRSOM(a=0.4, h=0.1)
        self.lower_y = GWRSOM(a=0.4, h=0.1)
        
        self.higher_nodes = []
        self.higher_weights = []
        self.higher_connections = np.zeros((0, 0))
        
        self.epsilon = 0.5
        self.start_epsilon = 0.5
        self.goal = None
        self.is_plan = None
        self.expected_next_state = None
        self.active_neurons = []
        
        self.firing_combinations = {}
        self.prev_node_idx = None
        self.lower_networks_trained = False
        
        # Track unique positions
        self.node_positions = {}
        self.position_history = set()

    def train_lower_networks(self, training_data, epochs=100):
        x_data = training_data[:, 0].reshape(-1, 1)
        y_data = training_data[:, 1].reshape(-1, 1)
        
        self.lower_x.train(x_data, epochs=epochs)
        self.lower_y.train(y_data, epochs=epochs)
        
        for position in training_data:
            x = position[0].reshape(1, -1)
            y = position[1].reshape(1, -1)
            x_node, _ = self.lower_x.find_best_matching_units(x)
            y_node, _ = self.lower_y.find_best_matching_units(y)
            pos_key = tuple(position)
            self.firing_combinations[pos_key] = (x_node, y_node)
        
        self.lower_networks_trained = True

    def get_firing_combination(self, current_state):
        if not self.lower_networks_trained:
            raise Exception("Lower networks must be trained first!")
        
        x = np.array([current_state[0]], dtype=float).reshape(1, -1)
        y = np.array([current_state[1]], dtype=float).reshape(1, -1)
        
        x_node, _ = self.lower_x.find_best_matching_units(x)
        y_node, _ = self.lower_y.find_best_matching_units(y)
        
        pattern = (x_node, y_node)
        self.active_neurons = pattern
        return pattern

    def create_new_node(self, firing_combination, position):
        """Create new node or return existing one with tolerance for noise"""
        tolerance = 2.0  # Distance threshold for considering positions the same
        position = np.array(position)
        
        # Check for existing nodes within tolerance
        for idx, pos in self.node_positions.items():
            if np.linalg.norm(np.array(pos) - position) < tolerance:
                return idx
        
        # If no close enough node exists, create new one
        new_idx = len(self.higher_nodes)
        self.higher_nodes.append(firing_combination)
        self.higher_weights.append(position)
        self.node_positions[new_idx] = position
        
        # Expand connection matrix
        new_connections = np.zeros((new_idx + 1, new_idx + 1))
        if new_idx > 0:
            new_connections[:-1, :-1] = self.higher_connections
        self.higher_connections = new_connections
        
        return new_idx

    def update_model(self, next_state, action):
        if not self.lower_networks_trained:
            raise Exception("Lower networks must be trained first!")
        
        firing_combination = self.get_firing_combination(next_state)
        curr_node_idx = self.create_new_node(firing_combination, next_state)
        
        if self.prev_node_idx is not None:
            self.higher_connections[self.prev_node_idx, curr_node_idx] = 1
            self.higher_connections[curr_node_idx, self.prev_node_idx] = 1
        
        self.prev_node_idx = curr_node_idx
    
    def select_action(self, current_state):
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
                    action = random.choice(possible_actions) % 4
                    self.is_plan = True
                    self.expected_next_state = self.get_expected_next_state(current_state, action)
                    return action
            
            action = random.randint(0, 3)
            self.is_plan = False
            return action
        else:
            action = random.randint(0, 3)
            self.is_plan = False
            return action

    def get_expected_next_state(self, current_state, action):
        current_state = np.array(current_state)
        action = int(action) % 4
        
        actions = {
            0: np.array([0, 1]),   # Up
            1: np.array([0, -1]),  # Down
            2: np.array([1, 0]),   # Right
            3: np.array([-1, 0])   # Left
        }
        
        return current_state + actions[action]

    def show_map(self):
        graph = nx.DiGraph()
        
        # Add nodes with their actual positions
        for i, position in self.node_positions.items():
            graph.add_node(i, pos=position)
        
        # Add edges
        rows, cols = np.where(self.higher_connections == 1)
        edges = zip(rows.tolist(), cols.tolist())
        graph.add_edges_from(edges)
        
        # Use actual positions for drawing
        pos = nx.get_node_attributes(graph, 'pos')
        plt.figure(figsize=(12, 12))
        nx.draw(graph, pos=pos, with_labels=True, 
                node_color='skyblue', node_size=500, 
                arrowsize=20, arrows=True)
        plt.title("Pattern Connectivity Map")
        plt.show()

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