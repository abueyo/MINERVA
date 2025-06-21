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
        NodeIndices = np.array(list(range(self.A.shape[0])))
        AloneNodes = NodeIndices[np.where(nNeighbour == 0)]
        
        if AloneNodes.any():
            self.connections = np.delete(self.connections, AloneNodes, axis=0)
            self.connections = np.delete(self.connections, AloneNodes, axis=1)
            self.ages = np.delete(self.ages, AloneNodes, axis=0)
            self.ages = np.delete(self.ages, AloneNodes, axis=1)
            self.A = np.delete(self.A, AloneNodes, axis=0)

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
        delta = self.es * hs * (x - weight_vector)
        self.A[w] = np.round(weight_vector + delta)
        
        b_neighbours = self._get_neighbours(w)
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
        self.firing_vector[w] = self.h0 - self.S / self.ab * (1 - np.exp(-self.ab * t / self.tb))
        b_neighbours = self._get_neighbours(w)
        self.firing_vector[b_neighbours] = self.h0 - self.S / self.an * (1 - np.exp(-self.an * t / self.tn))

    def train(self, X, epochs=100):
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


class HierarchicalGWRSOMAgent:
    def __init__(self, lower_dim=1, higher_dim=2, epsilon_b=0.35, epsilon_n=0.15,
                 beta=0.7, delta=0.79, T_max=20, N_max=100, eta=0.5, phi=0.9, sigma=0.5):
        self.lower_x = GWRSOM(a=0.4, h=0.1)
        self.lower_y = GWRSOM(a=0.4, h=0.1)
        
        self.higher_nodes = []
        self.higher_weights = []
        self.higher_connections = np.zeros((0, 0))
        
        self.start_epsilon = 0.5
        self.epsilon = self.start_epsilon
        self.goal = None
        self.is_plan = None
        self.expected_next_state = None
        self.active_neurons = []
        
        self.firing_combinations = {}
        self.prev_node_idx = None
        self.lower_networks_trained = False 
        self.position_to_firing = {}

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
            self.position_to_firing[pos_key] = (x_node, y_node)
        
        self.lower_networks_trained = True

    def get_firing_combination(self, current_state):
        if not self.lower_networks_trained:
            raise Exception("Lower networks must be trained first!")
            
        pos_key = tuple(current_state)
        firing_pattern = self.position_to_firing.get(pos_key)
        
        if firing_pattern is None:
            x = np.array([current_state[0]], dtype=float).reshape(1, -1)
            y = np.array([current_state[1]], dtype=float).reshape(1, -1)
            x_node, _ = self.lower_x.find_best_matching_units(x)
            y_node, _ = self.lower_y.find_best_matching_units(y)
            firing_pattern = (x_node, y_node)
            self.position_to_firing[pos_key] = firing_pattern
            
        self.active_neurons = firing_pattern
        return firing_pattern

    def create_new_node(self, firing_combination, position):
        if firing_combination in self.firing_combinations:
            return self.firing_combinations[firing_combination]
        
        self.higher_nodes.append(firing_combination)
        self.higher_weights.append(position)
        
        new_size = len(self.higher_nodes)
        new_connections = np.zeros((new_size, new_size))
        if new_size > 1:
            new_connections[:-1, :-1] = self.higher_connections
        self.higher_connections = new_connections
        
        new_idx = len(self.higher_nodes) - 1
        return new_idx

    def select_action(self, current_state):
        if self.goal is None:
            raise Exception("No goal defined")

        if np.random.uniform(0, 1) > self.epsilon:
            firing_combination = self.get_firing_combination(current_state)
            
            if firing_combination not in self.firing_combinations:
                curr_node_idx = self.create_new_node(firing_combination, current_state)
                self.firing_combinations[firing_combination] = curr_node_idx
            else:
                curr_node_idx = self.firing_combinations[firing_combination]
            
            if self.prev_node_idx is not None:
                self.higher_connections[self.prev_node_idx, curr_node_idx] = 1
                self.higher_connections[curr_node_idx, self.prev_node_idx] = 1
                possible_actions = np.where(self.higher_connections[curr_node_idx] == 1)[0]
                if len(possible_actions) > 0:
                    action = random.choice(possible_actions)
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
        """
        Calculate the expected next state based on current state and action.
        
        Args:
            current_state: numpy array of current position
            action: integer representing the action (should be 0-3)
            
        Returns:
            numpy array of expected next position
        """
        current_state = np.array(current_state)
        
        # Ensure action is within valid range
        action = int(action) % 4  # Convert any action to range 0-3
        
        actions = {
            0: np.array([0, 1]),   # Up
            1: np.array([0, -1]),  # Down
            2: np.array([1, 0]),   # Right
            3: np.array([-1, 0])   # Left
        }
        
        return current_state + actions[action]

    def select_action(self, current_state):
        """
        Select an action based on current state.
        
        Args:
            current_state: current position in the environment
            
        Returns:
            integer representing selected action (0-3)
        """
        if self.goal is None:
            raise Exception("No goal defined")

        if np.random.uniform(0, 1) > self.epsilon:
            firing_combination = self.get_firing_combination(current_state)
            
            if firing_combination not in self.firing_combinations:
                curr_node_idx = self.create_new_node(firing_combination, current_state)
                self.firing_combinations[firing_combination] = curr_node_idx
            else:
                curr_node_idx = self.firing_combinations[firing_combination]
            
            if self.prev_node_idx is not None:
                self.higher_connections[self.prev_node_idx, curr_node_idx] = 1
                self.higher_connections[curr_node_idx, self.prev_node_idx] = 1
                possible_actions = np.where(self.higher_connections[curr_node_idx] == 1)[0]
                if len(possible_actions) > 0:
                    action = random.choice(possible_actions) % 4  # Ensure action is 0-3
                    self.is_plan = True
                    self.expected_next_state = self.get_expected_next_state(current_state, action)
                    return action
            
            action = random.randint(0, 3)  # Explicitly limit to 0-3
            self.is_plan = False
            return action
        else:
            action = random.randint(0, 3)  # Explicitly limit to 0-3
            self.is_plan = False
            return action
        
    def update_model(self, next_state, action):
        if not self.lower_networks_trained:
            raise Exception("Lower networks must be trained first!")
            
        firing_combination = self.get_firing_combination(next_state)
        
        if firing_combination not in self.firing_combinations:
            curr_node_idx = self.create_new_node(firing_combination, next_state)
            self.firing_combinations[firing_combination] = curr_node_idx
        else:
            curr_node_idx = self.firing_combinations[firing_combination]
        
        if self.prev_node_idx is not None:
            self.higher_connections[self.prev_node_idx, curr_node_idx] = 1
            self.higher_connections[curr_node_idx, self.prev_node_idx] = 1
        
        self.prev_node_idx = curr_node_idx

    def explain_change(self):
        if self.is_plan is not None and self.is_plan:
            if not np.array_equal(self.active_neurons, self.expected_next_state):
                self.is_plan = None
                self.expected_next_state = None
    def save_model(self, file_path):
        if not file_path.endswith(".npz"):
            raise Exception(f"file does not have .npz extension")

        # Get all the values that need to be saved
        model_parameter_dict = {
            "W": self.model.W,  # W: weight vectors
            "Ct": self.model.Ct,  # Ct: context
            "C": self.model.C,  # C: links between nodes
            "A": self.model.A,  # A: action map
            "T_a": self.model.T_a,  # T_a: action index
            "W_a": self.model.W_a,  # W_a: lateral connectivity matrix
            "t": self.model.t,  # t: number of times links are traversed
            "H": self.model.H,  # H: neighbourhood matrix
            "HAB": self.model.H,  # HAB: habituation matrix in sensor space
            "HAB_a": self.model.HAB_a,  # ?
            "BMU": np.array([self.model.BMU]),  # BMU: best matching unit
            "BMU2": np.array([self.model.BMU2]),  # BMU2: previous best matching unit
        }

        # Save dictionary of arrays to a .npz file (use **dict to unpack the dict of arrays)
        np.savez(file_path, **model_parameter_dict)

        print(f"Model parameters saved to: {file_path}")

    def save_model(self, file_path):
        """Save the model parameters"""
        if not file_path.endswith(".npz"):
            raise Exception("file does not have .npz extension")

        model_parameters = {
            "lower_x_weights": self.lower_x.A,
            "lower_y_weights": self.lower_y.A,
            "firing_patterns": self.higher_nodes,
            "connections": self.higher_connections,
            # "action_mappings": self.action_mappings,
            # "pattern_ages": self.pattern_ages,
            # "node_positions": self.node_positions
        }
        
        np.savez(file_path, **model_parameters)
        print(f"Model parameters saved to: {file_path}")

    def load_model(self, file_path):
        """Load the model parameters"""
        if not file_path.endswith(".npz"):
            raise Exception("file does not have .npz extension")

        model_parameters = np.load(file_path, allow_pickle=True)
        
        self.lower_x.A = model_parameters["lower_x_weights"]
        self.lower_y.A = model_parameters["lower_y_weights"]
        self.higher_nodes = model_parameters["firing_patterns"].tolist()
        self.higher_connections = model_parameters["connections"]
        # self.action_mappings = model_parameters["action_mappings"].item()
        # self.pattern_ages = model_parameters["pattern_ages"]
        self.node_positions = model_parameters["node_positions"].item()
        
        print(f"Model parameters loaded from: {file_path}")

    def show_map(self):
        graph = nx.DiGraph()
        
        for i, (firing_combo, position) in enumerate(zip(self.higher_nodes, self.higher_weights)):
            if not np.isnan(position[0]):
                graph.add_node(i, pos=position, firing=f"X:{firing_combo[0]},Y:{firing_combo[1]}")

        for i in range(self.higher_connections.shape[0]):
            for j in range(self.higher_connections.shape[1]):
                if self.higher_connections[i, j] == 1:
                    graph.add_edge(i, j)

        node_positions = {i: pos for i, pos in enumerate(self.higher_weights)}

        plt.figure(figsize=(12, 12))
        nx.draw(graph, pos=node_positions, with_labels=True,
                node_color='skyblue', node_size=500,
                arrowsize=20, arrows=True)
        
        plt.title("Hierarchical GWRSOM Map")
        plt.show()
        # Store the final number of nodes
        self.num_nodes = graph.number_of_nodes()

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

