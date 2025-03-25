import numpy as np
import random
import networkx as nx
import logging
import matplotlib.pyplot as plt
#from Agents.GWRSOM import GWRSOM  # Assuming GWRSOM is in separate file

#setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
       self.t = 1  # Timestep
       self.A = None  # Node matrix A
       self.connections = None
       self.ages = None
       self.errors = None
       self.firing_vector = None
       self.max_age = 50  # Added max_age parameter
       self.sigma = 0.3 #Neighbourhood width for topological preservation
   
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
       nNeighbour = np.sum(self.connections,axis = 0)
       NodeIndisces = np.array(list(range(self.A.shape[0])))
       AloneNodes = NodeIndisces[np.where(nNeighbour == 0)]
       if AloneNodes.any():
           self.connections = np.delete(self.connections,AloneNodes,axis =0)
           self.connections = np.delete(self.connections,AloneNodes,axis =1)
           self.t = np.delete(self.t,AloneNodes,axis =0)
           self.t = np.delete(self.t,AloneNodes,axis =1)
           self.A = np.delete(self.W,AloneNodes,axis =0)

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
       self.A[b_neighbours]  = np.round(w_neighbours + delta)

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
   

class MapBuilder:
   def __init__(self, nDim, Ni, epsilon_b, epsilon_n, beta, delta, T_max, N_max, eta, phi, sigma):
       # Dimensionality of the input space
       self.nDim_a = 4  # Dimensionality of the action space
       self.N_a = 4  # Number of actions
       self.nDim = nDim  # Dimensionality of the world model
       self.epsilon_b = epsilon_b  # Learning rate for weight vector update
       self.epsilon_n = epsilon_n  # Learning rate for context update
       self.T_max = T_max  # Maximum time of idleness before deleting an edge
       self.N_max = N_max  # Maximum number of nodes
       self.delta = delta  # Error threshold
       self.beta = beta  # Learning rate for the update of the global context
       self.eta = eta  # Parameter for weight similarity as a function of BMU and global context
       self.phi = phi  # Neighbourhood threshold
       self.Ni = Ni

       # Initialize weight vectors randomly
       self.W = np.random.random((self.Ni, self.nDim)).astype(float)
    #    logger.debug(f"Initialized self.W with shape {self.W.shape}")
       # Initialize context vectors randomly
       self.Ct = np.random.random((self.Ni, self.nDim)).astype(float)
       # Initialize connection matrix
       self.C = np.zeros((self.Ni, self.Ni))
       # Initialize action map
       self.A = np.zeros((self.N_a, self.nDim_a)).astype(float)
       # Initialize action index matrix
       self.T_a = np.zeros((self.Ni, self.Ni))
       # Initialize lateral connectivity matrix
       self.W_a = np.zeros((self.Ni, self.Ni))
       # Initialize best matching unit (BMU)
       self.BMU = None
       # Initialize previous BMU
       self.BMU2 = None
       # Initialize action BMU
       self.BMU_a = None
       # Initialize edge traversal time matrix
       self.t = np.zeros((self.Ni, self.Ni))
       # Initialize neighbourhood matrix
       self.H = np.zeros((self.Ni, self.Ni))
       # Set the neighbourhood weight parameter
       self.sigma = sigma
       # Initialize global context vector
       self.Cg = np.zeros(self.nDim).astype(float)
       # Initialize habituation matrix in sensor space
       self.HAB = np.zeros((self.Ni, self.Ni))
       # Initialize habituation matrix for actions
       self.HAB_a = np.ones((self.Ni, self.N_a))

       # Habituation counter parameters
       self.kappa = 1.05
       self.tauB = 0.3
       self.tauN = 0.1
       self.hT = 0.0001
    
   def Distance(self, x, w):
       # The similarity metric (Euclidean norm)
       print("x", x, "w", w)
       return np.linalg.norm(x - w)

   def SpatialNeighbourHood(self):
       # Update the neighbourhood matrix based on spatial proximity
       for k in range(self.W.shape[0]):
           if self.Distance(self.W[self.BMU, :], self.W[k, :]) <= self.phi:
               self.H[self.BMU, k] = np.exp(-self.Distance(self.W[self.BMU, :], self.W[k, :]) ** 2 / (2 * self.sigma ** 2))
           else:
               self.H[self.BMU, k] = 0

   def remove_old_links(self):
       self.C[self.t > self.T_max] = 0
       self.t[self.t > self.T_max] = 0
       nNeighbour = np.sum(self.C, axis=0)
       NodeIndices = np.array(list(range(self.W.shape[0])))
       AloneNodes = NodeIndices[np.where(nNeighbour == 0)]
       
       if AloneNodes.any() and len(AloneNodes) < self.W.shape[1]:  # Ensure we don't remove all nodes
           valid_AloneNodes = AloneNodes[AloneNodes < self.W.shape[1]]
           if valid_AloneNodes.size > 1:
               self.C = np.delete(self.C, valid_AloneNodes, axis=0)
               self.C = np.delete(self.C, valid_AloneNodes, axis=1)
               self.t = np.delete(self.t, valid_AloneNodes, axis=0)
               self.t = np.delete(self.t, valid_AloneNodes, axis=1)
               self.W = np.delete(self.W, valid_AloneNodes, axis=0)
               if hasattr(self, 'HAB'):
                   self.HAB = np.delete(self.HAB, valid_AloneNodes, axis=0)
                   self.HAB = np.delete(self.HAB, valid_AloneNodes, axis=1)
               if hasattr(self, 'HAB_a'):
                   valid_AloneNodes = valid_AloneNodes[valid_AloneNodes < self.HAB_a.shape[0]]
                   if valid_AloneNodes.size > 0:
                       self.HAB_a = np.delete(self.HAB_a, valid_AloneNodes, axis=0)
               if hasattr(self, 'Ct'):
                   self.Ct = np.delete(self.Ct, valid_AloneNodes, axis=0)

       # Ensure there's always at least one node
       if self.W.shape[0] == 0:
           self.W = np.random.random((1, self.W.shape[1]))
           self.C = np.zeros((1, 1))
           self.t = np.zeros((1, 1))
           if hasattr(self, 'HAB'):
               self.HAB = np.zeros((1, 1))
           if hasattr(self, 'HAB_a'):
               self.HAB_a = np.ones((1, self.HAB_a.shape[1]))
           if hasattr(self, 'Ct'):
               self.Ct = np.random.random((1, self.Ct.shape[1]))

   def add_new_nodes(self, x):
       logger.debug(f"Before add_new_nodes: self.W shape is {self.W.shape}")
       # Add new nodes to the model
       C_ = self.C.copy()
       t_ = self.t.copy()
       H_ = self.H.copy()
       T_a_ = self.T_a.copy()
       W_a_ = self.W_a.copy()
       HAB = self.HAB.copy()

       # Initialize new matrices with the new size
       new_size = self.W.shape[0] + 1
       self.W = np.vstack((self.W, np.zeros((1, self.W.shape[1]))))  # Add a zero row with the same number of columns
       self.Ct = np.vstack((self.Ct, np.zeros((1, self.Ct.shape[1]))))  # Add a zero row with the same number of columns
       self.C = np.zeros((new_size, new_size))
       self.t = np.zeros((new_size, new_size))
       self.H = np.zeros((new_size, new_size))
       self.T_a = np.zeros((new_size, new_size))
       self.W_a = np.zeros((new_size, new_size))
       self.HAB = np.zeros((new_size, new_size))

       # Restore the copied values
       self.C[:-1, :-1] = C_
       self.t[:-1, :-1] = t_
       self.H[:-1, :-1] = H_
       self.T_a[:-1, :-1] = T_a_
       self.W_a[:-1, :-1] = W_a_
       self.HAB[:-1, :-1] = HAB

       # Add the new node to the last row
       self.W[-1, :] = x
       self.Ct[-1, :] = self.Cg

       # Update the BMU to the new node's index
       self.BMU = self.W.shape[0] - 1
    #    logger.debug(f"After add_new_nodes: self.W shape is {self.W.shape}")

   def create_link(self, a):
       # Create a link between BMU2 and BMU
       self.C[self.BMU2, self.BMU] = 1
       self.t[self.BMU2, self.BMU] = 0
       self.T_a[self.BMU2, self.BMU] = self.BMU_a
       Sim_a = np.exp(-self.Distance(self.A[self.BMU_a, :], a) ** 2 / (2 * self.sigma ** 2))
       self.W_a[self.BMU2, self.BMU] = self.W_a[self.BMU2, self.BMU] + 0.5 * (Sim_a * self.change(self.W[self.BMU2, :], self.W[self.BMU, :]) - self.W_a[self.BMU2, self.BMU])
       if np.any(self.HAB[self.BMU2, self.BMU]) == 0:
           self.HAB[self.BMU2, self.BMU] = 1
       else:
           self.HAB[self.BMU2, self.BMU] = max(self.HAB[self.BMU2, self.BMU] + self.tauB * self.kappa * (1 - self.HAB[self.BMU2, self.BMU]) - self.tauB, self.hT)

   def train(self, x, a):
    
    x = x.astype(float)
    a = a.astype(float)
    x = np.array(x).reshape(-1)  # Ensure x is a 1D array
    a = np.array(a).reshape(-1)  # Ensure a is a 1D array
    
    # Calculate the distances between the current state x and all weight vectors in self.W
    Dis_xw = [self.Distance(x, w) for w in self.W]
    
    # Calculate the distances between the global context vector self.Cg and all context vectors in self.Ct
    Dis_Ctci = [self.Distance(self.Cg, c) for c in self.Ct]
    
    # Calculate the distances between the action vector a and all action vectors in self.A
    Dis_act = [self.Distance(a, w_a) for w_a in self.A]
    
    # Calculate the composite distance metric D as a weighted sum of Dis_xw and Dis_Ctci
    D = self.eta * np.power(np.array(Dis_xw), 2) + (1 - self.eta) * np.power(np.array(Dis_Ctci), 2)

    # Find the index of the best matching unit (BMU) based on the composite distance metric D
    self.BMU = np.argmin(D)
    
    # Find the index of the BMU for actions based on the distances between a and action vectors
    self.BMU_a = np.argmin(Dis_act)

    # Identify the set of nodes connected to the previous BMU (self.BMU2)
    if self.BMU2 is not None:
        P = np.where(self.C[self.BMU2, :] == 1)
        # Update the traversal time for the edges connected to the previous BMU
        self.t[self.BMU2, P] = self.t[self.BMU2, P] + 1

    # Calculate the activation of the BMU for the current state
    a_BMU = np.exp(-Dis_xw[self.BMU] ** 2 / self.W.shape[1])

    # Check if the activation is below a threshold and the number of nodes is less than the maximum allowed
    if a_BMU < self.delta and self.W.shape[0] < self.N_max:
        self.add_new_nodes(x)
    else:
        self.W[self.BMU, :] = self.W[self.BMU, :] + self.epsilon_n * (x - self.W[self.BMU, :])
        self.Ct[self.BMU, :] = self.Ct[self.BMU, :] + self.epsilon_b * (self.Cg - self.Ct[self.BMU, :])
        for k in range(self.W.shape[0]):
            if k != self.BMU:
                self.W[k, :] = self.W[k, :] + self.H[self.BMU, k] * self.epsilon_n * (x - self.W[k, :])
                self.Ct[k, :] = self.Ct[k, :] + self.H[self.BMU, k] * self.epsilon_n * (self.Cg - self.Ct[k, :])
        # logger.debug(f"After updating weights and contexts: self.W shape is {self.W.shape}")

    # Update the global context vector
    self.Cg = self.beta * self.W[self.BMU, :] + (1 - self.beta) * self.Ct[self.BMU, :]
    # logger.debug(f"Updated global context vector (Cg): {self.Cg}")

    # Update the action vector for the BMU
    if self.BMU_a is not None and self.BMU_a < self.A.shape[0]:
        # logger.debug(f"Updating action vector. Before: self.A shape is {self.A.shape}")
        self.A[self.BMU_a, :] = self.A[self.BMU_a, :] + self.epsilon_b * (a - self.A[self.BMU_a, :])
        logger.debug(f"After updating action vector: self.A shape is {self.A.shape}")

    # Update the habituation matrix for actions
    if self.BMU < self.HAB_a.shape[0] and self.BMU_a < self.HAB_a.shape[1]:
        self.HAB_a[self.BMU, self.BMU_a] = max(self.HAB_a[self.BMU, self.BMU_a] + self.tauB * self.kappa * (1 - self.HAB_a[self.BMU, self.BMU_a]) - self.tauB, self.hT)
    else:
        logger.warning(f"BMU ({self.BMU}) or BMU_a ({self.BMU_a}) out of bounds for HAB_a with shape {self.HAB_a.shape}")

    # Save the current weight vector of the BMU for later use
    Temp_Val_BMU = self.W[self.BMU, :]
    
    # Create a link between the BMU and the action vector
    if self.BMU2 is not None:
        logger.debug(f"Creating link between BMU and action vector")
        self.create_link(a)
    
    # Remove old links that have exceeded the maximum traversal time
    logger.debug(f"Removing old links. Before: self.W shape is {self.W.shape}")
    self.remove_old_links()
    logger.debug(f"After removing old links: self.W shape is {self.W.shape}")

    # Calculate the distances between the saved weight vector and all weight vectors in self.W
    # logger.debug(f"Calculating distances for next BMU")
    Dist_wBMU = [self.Distance(Temp_Val_BMU, w) for w in self.W]
    
    # Find the index of the BMU for the next iteration based on the distances
    self.BMU2 = np.argmin(Dist_wBMU)
    # logger.debug(f"Next BMU (BMU2): {self.BMU2}")

    # logger.debug(f"End of train: self.W shape is {self.W.shape}")
    # logger.debug(f"End of train: self.HAB_a shape is {self.HAB_a.shape}")


   def get_node_index(self, state):
       # Calculate the distances between the current state x and all weight vectors in self.W
       Dis_xw = [self.Distance(state, w) for w in self.W]
       # Calculate the distances between the global context vector self.Cg and all context vectors in self.Ct
       Dis_Ctci = [self.Distance(self.Cg, c) for c in self.Ct]

       # Calculate the composite distance metric D as a weighted sum of Dis_xw and Dis_Ctci
       D = self.eta * np.power(np.array(Dis_xw), 2) + (1 - self.eta) * np.power(np.array(Dis_Ctci), 2)
       return np.argmin(D)

   def get_weights(self):
       return self.W

   def get_connections(self):
       return self.C
 
   def change(self,s1,s2):
       # Check if 2 vectors are different
       if self.Distance(s1,s2)==0:
           return 0
       return 1



class HierarchicalGWRSOMAgent:
    def __init__(self, lower_dim=1, higher_dim=2, epsilon_b=0.35, epsilon_n=0.15, 
                 beta=0.7, delta=0.79, T_max=20, N_max=300, eta=0.5, phi=0.9, sigma=0.5):
        # Initialize lower level networks
        self.lower_x = GWRSOM(a=0.4, h=0.1)
        self.lower_y = GWRSOM(a=0.4, h=0.1)
        
        # Higher level stores patterns and their relationships
        self.nodes = []  # Stores firing patterns
        self.connections = np.zeros((0, 0))  # Connectivity between nodes
        self.action_mappings = {}  # Maps (node1, node2) pairs to actions
        
        # For visualization consistency
        self.node_positions = {}  # Store positions for each node
        self.node_bounds = {}
        self.initial_layout_done = False  # Flag to track if initial layout is done

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
        """Pre-train lower level networks to preserve topological relationships"""
        if len(training_data) == 0:
            raise Exception("Lower networks must be trained first!")
        
        # Get data bounds
        x_data = training_data[:, 0]
        y_data = training_data[:, 1]
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        x_data = np.sort(training_data)[:, 0]
        y_data = np.sort(training_data)[:, 1]

        for x in x_data:
            self.lower_x.train(np.array([[x]]), epochs=5)
        for y in y_data:
            self.lower_y.train(np.array([[y]]), epochs=5)

    def get_firing_pattern(self, state):
        """Fixed implementation that properly handles BMU return values"""
        x_data = np.array([state[0]]).reshape(1, -1)
        y_data = np.array([state[1]]).reshape(1, -1)
        
        # Ensure we get usable BMU indices
        x_bmus, _ = self.lower_x.find_best_matching_units(x_data)
        y_bmus, _ = self.lower_y.find_best_matching_units(y_data)
        
        # Create binary vectors
        x_binary = np.zeros(len(self.lower_x.A), dtype=np.float32)
        y_binary = np.zeros(len(self.lower_y.A), dtype=np.float32)
        
        # Handle both scalar and array return types
        if np.isscalar(x_bmus):
            x_binary[x_bmus] = 1.0
        else:
            x_binary[x_bmus[0]] = 1.0  # Use first BMU if multiple returned
        
        if np.isscalar(y_bmus):
            y_binary[y_bmus] = 1.0
        else:
            y_binary[y_bmus[0]] = 1.0  # Use first BMU if multiple returned
        
        return (tuple(x_binary.tolist()), tuple(y_binary.tolist()))
    def find_node_index(self, pattern):
        """Find index of node matching the pattern with robust comparison"""
        pattern_x, pattern_y = pattern
        
        for idx, node_pattern in enumerate(self.nodes):
            node_x, node_y = node_pattern
            if (np.array_equal(pattern_x, node_x) and 
                np.array_equal(pattern_y, node_y)):
                return idx
        return None

    def create_node(self, pattern, state):
        """Enhanced node creation with bounds tracking"""
        node_idx = len(self.nodes)
        self.nodes.append(pattern)
        
        # Initialize or update bounds for this pattern
        self.node_bounds[node_idx] = {
            'x_min': state[0], 'x_max': state[0],
            'y_min': state[1], 'y_max': state[1]
        }
        
        # Store position
        self.node_positions[node_idx] = state
        
        # Rest of your existing node creation code...
        new_size = len(self.nodes)
        new_connections = np.zeros((new_size, new_size))
        new_ages = np.zeros((new_size, new_size))
        
        if new_size > 1:
            new_connections[:-1, :-1] = self.connections
            new_ages[:-1, :-1] = self.pattern_ages
            
        self.connections = new_connections
        self.pattern_ages = new_ages
        
        return node_idx

    def _add_node_position(self, node_idx):
        """Calculate position for a new node based on its connections"""
        graph = nx.DiGraph()
        
        # Add all existing nodes and edges
        for i in range(len(self.nodes)):
            graph.add_node(i)
        
        rows, cols = np.where(self.connections == 1)
        edges = zip(rows.tolist(), cols.tolist())
        graph.add_edges_from(edges)
        
        # Calculate position for new node based on connected nodes
        connected_nodes = np.where(self.connections[node_idx] == 1)[0]
        if len(connected_nodes) > 0:
            # Average position of connected nodes
            x = np.mean([self.node_positions[n][0] for n in connected_nodes])
            y = np.mean([self.node_positions[n][1] for n in connected_nodes])
            # Add small random offset to avoid overlap
            x += np.random.uniform(-0.1, 0.1)
            y += np.random.uniform(-0.1, 0.1)
        else:
            # If no connections, place near existing nodes
            existing_x = [pos[0] for pos in self.node_positions.values()]
            existing_y = [pos[1] for pos in self.node_positions.values()]
            x = np.mean(existing_x) + np.random.uniform(-0.5, 0.5)
            y = np.mean(existing_y) + np.random.uniform(-0.5, 0.5)
        
        self.node_positions[node_idx] = (x, y)


    def update_model(self, next_state, action):
        
        """Enhanced model update with bounds tracking"""
        current_pattern = self.get_firing_pattern(next_state)
        current_idx = self.find_node_index(current_pattern)
        
        if current_idx is None:
            current_idx = self.create_node(current_pattern, next_state)
        else:
            # Update bounds for existing pattern
            bounds = self.node_bounds[current_idx]
            bounds['x_min'] = min(bounds['x_min'], next_state[0])
            bounds['x_max'] = max(bounds['x_max'], next_state[0])
            bounds['y_min'] = min(bounds['y_min'], next_state[1])
            bounds['y_max'] = max(bounds['y_max'], next_state[1])
            
        if self.prev_node_idx is not None:
            # Create/update connection
            self.connections[self.prev_node_idx, current_idx] = 1
            self.pattern_ages[self.prev_node_idx, current_idx] = 0
            # Store action that led to this transition
            self.action_mappings[(self.prev_node_idx, current_idx)] = action
            
            # Age other connections
            connected = np.where(self.connections[self.prev_node_idx] == 1)[0]
            self.pattern_ages[self.prev_node_idx, connected] += 1
            
            # Remove old connections
            old_connections = self.pattern_ages > self.T_max
            self.connections[old_connections] = 0
            self.pattern_ages[old_connections] = 0
        
        self.prev_node_idx = current_idx

    def select_action(self, current_state):
        """Select action based on learned topology"""
        if self.goal is None:
            raise Exception("No goal defined")

        if np.random.uniform(0, 1) > self.epsilon:
            current_pattern = self.get_firing_pattern(current_state)
            current_idx = self.find_node_index(current_pattern)
            
            if current_idx is not None:
                # Get connected nodes
                connected_nodes = np.where(self.connections[current_idx] == 1)[0]
                
                if len(connected_nodes) > 0:
                    # Choose next node randomly from connected nodes
                    next_idx = np.random.choice(connected_nodes)
                    key = (current_idx, next_idx)
                    
                    if key in self.action_mappings:
                        self.is_plan = True
                        self.expected_next_node = next_idx
                        return self.action_mappings[key]
        
        # Default to exploration
        self.is_plan = False
        return random.randint(0, 3)
    
    def save_model(self, file_path):
        """Save the model parameters"""
        if not file_path.endswith(".npz"):
            raise Exception("file does not have .npz extension")

        model_parameters = {
            "lower_x_weights": self.lower_x.A,
            "lower_y_weights": self.lower_y.A,
            "firing_patterns": self.nodes,
            "connections": self.connections,
            "action_mappings": self.action_mappings,
            "pattern_ages": self.pattern_ages,
            "node_positions": self.node_positions
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
        self.nodes = model_parameters["firing_patterns"].tolist()
        self.connections = model_parameters["connections"]
        self.action_mappings = model_parameters["action_mappings"].item()
        self.pattern_ages = model_parameters["pattern_ages"]
        self.node_positions = model_parameters["node_positions"].item()
        
        print(f"Model parameters loaded from: {file_path}")
    def set_goal(self, goal):
        self.goal = goal

    def decay_epsilon(self, min_epsilon=0.2):
        self.epsilon = max(round(self.epsilon-0.1, 5), min_epsilon)

    def reset_epsilon(self):
        self.epsilon = self.start_epsilon

    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def explain_change(self):
        if self.is_plan is not None:
            current_pattern = self.get_firing_pattern(self.expected_next_node)
            current_idx = self.find_node_index(current_pattern)
            
            if self.is_plan and current_idx != self.expected_next_node:
                print(f"World Changed! Expected node: {self.expected_next_node}; Actual node: {current_idx}")
                self.is_plan = None
                self.expected_next_node = None

    def show_map(self):
        graph = nx.DiGraph()
        
        # Add nodes with their actual positions
        for i, position in self.node_positions.items():
            graph.add_node(i, pos=position)
        
        # Add edges
        rows, cols = np.where(self.connections == 1)
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