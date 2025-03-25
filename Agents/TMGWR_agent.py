import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

class TMGWRAgent:
    def __init__(self, nDim, Ni, epsilon_b, epsilon_n, beta, delta, T_max, N_max, eta, phi, sigma):
        self.model = MapBuilder(nDim=nDim, Ni=Ni, epsilon_b=epsilon_b, epsilon_n=epsilon_n, 
                                beta=beta, delta=delta, T_max=T_max, N_max=N_max, eta=eta, phi=phi, sigma=sigma)
        self.ValueClass = Value(self.model.W)
        self.ActionClass = Action()
        self.start_epsilon = 0.5
        self.epsilon = self.start_epsilon
        self.goal = None
        self.is_plan = None
        self.expected_next_state = None
        self.active_neurons = []

    def train(self, x, a):
        x = np.array(x)  # Ensure x is a NumPy array
        print(f"Training with input x shape: {x.shape}")
        # Convert the action from integer to array
        encoded_action = self.get_onehot_encoded_action(a)
        # Train the model
        self.model.train(x=x, a=encoded_action)
        # Track the active neurons
        self.active_neurons = self.get_active_neurons(x)

    def get_active_neurons(self, x):
        # Determine the active neurons based on the input x
        distances = [self.model.Distance(x, w) for w in self.model.W]
        active_index = np.argmin(distances)
        return active_index

    def output(self):
        return self.model.W
      
    def get_node_index(self, state):
        # Delegate the call to the MapBuilder instance
        return self.model.get_node_index(state)

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

    def show_map(self):
        # Create directed graph
        graph = nx.DiGraph()
        
        # Dictionary to track unique positions
        unique_positions = {}
        
        # Add nodes with unique positions
        for i, state in enumerate(self.model.W):
            if not np.isnan(state[0]):
                # Round to prevent floating point issues
                pos = tuple(np.round(state, 2))
                if pos not in unique_positions:
                    unique_positions[pos] = i
                    graph.add_node(i, pos=state)
        
        # Add edges
        rows, cols = np.where(self.model.C == 1)
        for r, c in zip(rows, cols):
            pos_r = tuple(np.round(self.model.W[r], 2))
            pos_c = tuple(np.round(self.model.W[c], 2))
            if pos_r in unique_positions and pos_c in unique_positions:
                graph.add_edge(unique_positions[pos_r], unique_positions[pos_c])
        
        # Use node positions for drawing
        pos = nx.get_node_attributes(graph, 'pos')
        plt.figure(figsize=(12, 12))
        nx.draw(graph, pos=pos, with_labels=True,
                node_color='skyblue', node_size=500,
                arrowsize=20, arrows=True)
        plt.title("TMGWR Map")
        plt.show()

    def select_action(self, current_state):
        if self.goal is None:
            print("Goal needs to be set")
            raise Exception("No goal defined")

        if np.random.uniform(0, 1) > self.epsilon:
            # Exploitation, select action using internal model
            V = self.ValueClass.ComputeValue(self.model.W, self.model.C, self.model.W_a, self.goal)
            t_a = self.ActionClass.actionSelect(current_state, self.model.W, V, self.model.T_a, self.model.C)
            action_vec = self.model.A[t_a, :]
            action = np.argmax(action_vec)  # Action index is the action

            # Get the expected next state for explainability
            self.expected_next_state = self.ActionClass.indEX
            self.is_plan = True
        else:
            # Exploration, select random action
            action = random.randint(0, 3)

            # For explainability
            self.is_plan = False

        return action

    def explain_change(self):
        if self.is_plan is not None:
            if self.is_plan and self.model.BMU != self.expected_next_state:
                print(f"World Changed! Anticipated node = {self.expected_next_state}; Actual node= {self.model.BMU}")

                # Set the expectation to nothing
                self.is_plan = None
                self.expected_next_state = None

    def update_model(self, next_state, action):
        next_state = np.array(next_state)  # Ensure next_state is a NumPy array
        # Convert the action from integer to array
        encoded_action = self.get_onehot_encoded_action(action)
        # Train the model
        self.model.train(x=next_state, a=encoded_action)

    def get_onehot_encoded_action(self, action):
        if action == 0:
            # Up
            return [1, 0, 0, 0]
        elif action == 1:
            # Down
            return [0, 1, 0, 0]
        elif action == 2:
            # Right
            return [0, 0, 1, 0]
        else:
            # Left (action == 3)
            return [0, 0, 0, 1]

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

    def load_model(self, file_path):
        if not file_path.endswith(".npz"):
            raise Exception(f"file does not have .npz extension")

        # Load the model parameters from the file
        model_parameter = np.load(file_path)

        # Assign the model parameters to the model
        self.model.W = model_parameter["W"]
        self.model.Ct = model_parameter["Ct"]
        self.model.C = model_parameter["C"]
        self.model.A = model_parameter["A"]
        self.model.T_a = model_parameter["T_a"]
        self.model.W_a = model_parameter["W_a"]
        self.model.t = model_parameter["t"]
        self.model.H = model_parameter["H"]
        self.model.HAB = model_parameter["HAB"]
        self.model.HAB_a = model_parameter["HAB_a"]
        self.model.BMU = model_parameter["BMU"][0]
        self.model.BMU2 = model_parameter["BMU2"][0]

        print(f"Model parameters loaded from: {file_path}")

class Action:
    def Distance(self, x1, x2):
        # The similarity metric (Euclidean norm)
        return np.linalg.norm(x1 - x2)

    def actionSelect(self, x, W, V, T_a, C):
        D = []
        Val = []
        ind = []
        for w in W:
            D.append(self.Distance(x, w))
        self.w_x = np.argmin(D)
        self.neigh = np.multiply(C[self.w_x, :], V)
        self.indEX = np.argmax(self.neigh)
        return int(T_a[self.w_x, self.indEX])

class Value:
    def __init__(self, W):
        self.V = None  # np.zeros(W.shape[0])
        self.R = None  # np.zeros(W.shape[0])

    def Distance(self, x1, x2):
        # The similarity metric (Euclidean norm)
        return np.linalg.norm(x1 - x2)

    def ComputeReward(self, W, W_a, goal):
        for i in range(W_a.shape[0]):
            W_a[i, i] = 0
        self.R = np.zeros(W.shape[0])
        D = []
        for w in W:
            D.append(self.Distance(goal, w))
        self.w_g = np.argmin(D)
        for i in range(W.shape[0]):
            if i == self.w_g:
                self.R[i] = 10  # *np.exp(-self.Distance(W[self.w_g ,:],W[i,:])**2/2)
            else:
                self.R[i] = np.exp(-self.Distance(W[self.w_g, :], W[i, :]) ** 2 / 2)

    def ComputeValue(self, W, C, W_a, goal):
        self.V = np.zeros(W.shape[0])
        self.ComputeReward(W, W_a, goal)
        WV = []
        for _ in range(100):
            for i in range(W.shape[0]):
                neigh = np.where(C[i, :] == 1)
                if neigh[0].size != 0:
                    for k in neigh[0]:
                        WV.append(W_a[i, k] * self.V[k])
                    self.V[i] = self.V[i] + 0.8 * (self.R[i] + 0.99 * np.max(WV) - self.V[i])
                    WV = []
        return self.V

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
        self.W = np.random.random((self.Ni, self.nDim))
        # Initialize context vectors randomly
        self.Ct = np.random.random((self.Ni, self.nDim))
        # Initialize connection matrix
        self.C = np.zeros((self.Ni, self.Ni))
        # Initialize action map
        self.A = np.zeros((self.N_a, self.nDim_a))
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
        self.Cg = np.zeros(self.nDim)
        # Initialize habituation matrix in sensor space
        self.HAB = np.zeros((self.Ni, self.Ni))
        # Initialize habituation matrix for actions
        self.HAB_a = np.ones((self.Ni, self.nDim_a))

        # Habituation counter parameters
        self.kappa = 1.05
        self.tauB = 0.3
        self.tauN = 0.1
        self.hT = 0.0001

    def Distance(self, x, w):
        # The similarity metric (Euclidean norm)
        return np.linalg.norm(x - w)

    def SpatialNeighbourHood(self):
        # Update the neighbourhood matrix based on spatial proximity
        for k in range(self.W.shape[0]):
            if self.Distance(self.W[self.BMU, :], self.W[k, :]) <= self.phi:
                self.H[self.BMU, k] = np.exp(-self.Distance(self.W[self.BMU, :], self.W[k, :]) ** 2 / (2 * self.sigma ** 2))
            else:
                self.H[self.BMU, k] = 0

    def remove_old_links(self):
        # Remove edges that have exceeded the maximum traversal time
        self.C[self.t > self.T_max] = 0
        self.W_a[self.t > self.T_max] = 0
        nNeighbour = np.sum(self.C, axis=0)
        NodeIndisces = np.array(list(range(self.W.shape[0])))
        AloneNodes = NodeIndisces[np.where(nNeighbour == 0)]
        # Don't remove nodes if it would leave us with no nodes
        if AloneNodes.any() and (self.W.shape[0] - len(AloneNodes)) > 0:
            self.C = np.delete(self.C, AloneNodes, axis=0)
            self.C = np.delete(self.C, AloneNodes, axis=1)
            self.t = np.delete(self.t, AloneNodes, axis=0)
            self.t = np.delete(self.t, AloneNodes, axis=1)
            self.H = np.delete(self.H, AloneNodes, axis=0)
            self.H = np.delete(self.H, AloneNodes, axis=1)
            self.T_a = np.delete(self.T_a, AloneNodes, axis=0)
            self.T_a = np.delete(self.T_a, AloneNodes, axis=1)
            self.W_a = np.delete(self.W_a, AloneNodes, axis=0)
            self.W_a = np.delete(self.W_a, AloneNodes, axis=1)
            self.W = np.delete(self.W, AloneNodes, axis=0)
            self.HAB = np.delete(self.HAB, AloneNodes, axis=0)
            self.HAB = np.delete(self.HAB, AloneNodes, axis=1)
            self.HAB_a = np.delete(self.HAB_a, AloneNodes, axis=0)
            self.Ct = np.delete(self.Ct, AloneNodes, axis=0)

    def add_new_nodes(self, x):
        # Add new nodes to the model
        C_ = self.C.copy()
        t_ = self.t.copy()
        H_ = self.H.copy()
        T_a_ = self.T_a.copy()
        W_a_ = self.W_a.copy()
        HAB = self.HAB.copy()
        HAB_a_ = self.HAB_a.copy()  # Add this line

        # Initialize new matrices with the new size
        new_size = self.W.shape[0] + 1
        self.W = np.vstack((self.W, np.zeros((1, self.W.shape[1]))))
        self.Ct = np.vstack((self.Ct, np.zeros((1, self.Ct.shape[1]))))
        self.C = np.zeros((new_size, new_size))
        self.t = np.zeros((new_size, new_size))
        self.H = np.zeros((new_size, new_size))
        self.T_a = np.zeros((new_size, new_size))
        self.W_a = np.zeros((new_size, new_size))
        self.HAB = np.zeros((new_size, new_size))
        # Resize HAB_a properly
        new_HAB_a = np.ones((new_size, self.nDim_a))  # Changed this line
        
        # Restore the copied values
        self.C[:-1, :-1] = C_
        self.t[:-1, :-1] = t_
        self.H[:-1, :-1] = H_
        self.T_a[:-1, :-1] = T_a_
        self.W_a[:-1, :-1] = W_a_
        self.HAB[:-1, :-1] = HAB
        new_HAB_a[:-1, :] = HAB_a_  # Add this line
        self.HAB_a = new_HAB_a      # Add this line

        # Add the new node to the last row
        self.W[-1, :] = x
        self.Ct[-1, :] = self.Cg

        # Update the BMU to the new node's index
        self.BMU = self.W.shape[0] - 1
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
        x = np.array(x)  # Ensure x is a NumPy array
        print(f"MapBuilder Training with input x shape: {x.shape}")
        print(f"MapBuilder Current W shape: {self.W.shape}")
        
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
            # If conditions are not met, update weights and contexts for the BMU and its neighbors
            self.W[self.BMU, :] = self.W[self.BMU, :] + self.epsilon_n * (x - self.W[self.BMU, :])
            self.Ct[self.BMU, :] = self.Ct[self.BMU, :] + self.epsilon_b * (self.Cg - self.Ct[self.BMU, :])
            for k in range(self.W.shape[0]):
                if k != self.BMU:
                    self.W[k, :] = self.W[k, :] + self.H[self.BMU, k] * self.epsilon_n * (x - self.W[k, :])
                    self.Ct[k, :] = self.Ct[k, :] + self.H[self.BMU, k] * self.epsilon_n * (self.Cg - self.Ct[k, :])

        # Update the global context vector
        self.Cg = self.beta * self.W[self.BMU, :] + (1 - self.beta) * self.Ct[self.BMU, :]
        # Update the action vector for the BMU
        self.A[self.BMU_a, :] = self.A[self.BMU_a, :] + self.epsilon_b * (a - self.A[self.BMU_a, :])
        # Update the habituation matrix for actions
        self.HAB_a[self.BMU, self.BMU_a] = max(self.HAB_a[self.BMU, self.BMU_a] + self.tauB * self.kappa * (1 - self.HAB_a[self.BMU, self.BMU_a]) - self.tauB, self.hT)
        # Save the current weight vector of the BMU for later use
        Temp_Val_BMU = self.W[self.BMU, :]
        # Create a link between the BMU and the action vector
        if self.BMU2 is not None:
            self.create_link(a)
        # Remove old links that have exceeded the maximum traversal time
        self.remove_old_links()

         # Before calculating Dist_wBMU, check if we have any nodes
        if self.W.shape[0] == 0:
            # If we have no nodes, add the current state as the first node
            self.add_new_nodes(x)
            self.BMU2 = self.BMU  # Set BMU2 to the same as BMU since we only have one node
            return
            
        # Calculate distances only if we have nodes
        Dist_wBMU = [self.Distance(Temp_Val_BMU, w) for w in self.W]
        if len(Dist_wBMU) > 0:
            self.BMU2 = np.argmin(Dist_wBMU)
        else:
            self.BMU2 = self.BMU  # Default to current BMU if no distances calculated
            # Find the index of the BMU for the next iteration based on the distances
            self.BMU2 = np.argmin(Dist_wBMU)

    def get_node_index(self, state):
        # Calculate the distances between the current state x and all weight vectors in self.W
        Dis_xw = [self.Distance(state, w) for w in self.W]
        # Calculate the distances between the global context vector self.Cg and all context vectors in self.Ct
        Dis_Ctci = [self.Distance(self.Cg, c) for c in self.Ct]

        # Calculate the composite distance metric D as a weighted sum of Dis_xw and Dis_Ctci
        D = self.eta * np.power(np.array(Dis_xw), 2) + (1 - self.eta) * np.power(np.array(Dis_Ctci), 2)

        # Find the index of the best matching unit (BMU) based on the composite distance metric D
        return np.argmin(D)

    def show_map(self):
        g = nx.DiGraph()
        P = []
        Labels = {}
        for i in range(self.C.shape[0]):
            for j in range(self.C[i, :].size):
                if self.C[i, j] == 1:
                    g.add_edge(tuple(np.round(self.W[i, :], 2)), tuple(np.round(self.W[j, :], 2)))
        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=False, node_color='skyblue', node_size=500, arrowsize=20, arrows=T)
        plt.show()

    def change(self, s1, s2):
        # Check if 2 vectors are different
        if self.Distance(s1, s2) == 0:
            return 0
        return 1

