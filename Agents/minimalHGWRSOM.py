import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

class MinimalHGWRSOM:
    def __init__(self, **kwargs):
        """
        Accept any keyword arguments to maintain compatibility with original class
        """
        # Core attributes
        self.nodes = []  # Will store patterns
        self.connections = np.zeros((0, 0))  # Connectivity matrix
        self.action_mappings = {}  # Maps (node1, node2) to actions
        
        # For visualization
        self.node_positions = {}  # Maps node index to position
        
        # For tracking nodes
        self.prev_node_idx = None
        
        # For agent interface
        self.goal = None
        self.epsilon = 0.5
        self.start_epsilon = 0.5
        
        # Compatibility with comparison script
        self.higher_nodes = self.nodes  # Alternative reference to nodes
        
        # Create dummy objects for attributes that might be accessed
        class DummyGWRSOM:
            def __init__(self):
                self.A = np.array([[0], [0]])
                
            def find_best_matching_units(self, data):
                return [0], [0]
                
        self.lower_x = DummyGWRSOM()
        self.lower_y = DummyGWRSOM()
    
    def train_lower_networks(self, training_data, epochs=100):
        """Dummy method that does nothing but exists for compatibility"""
        pass
    
    def get_firing_pattern(self, state):
        """Trivial pattern - just use the state coordinates directly"""
        return tuple(state)
    
    def find_node_index(self, pattern):
        """Simple exact matching"""
        for idx, node_pattern in enumerate(self.nodes):
            if node_pattern == pattern:
                return idx
        return None
    
    def create_node(self, pattern, state):
        """Create a new node with the given pattern and state"""
        node_idx = len(self.nodes)
        self.nodes.append(pattern)
        self.higher_nodes = self.nodes  # Keep these in sync
        
        # Store position
        self.node_positions[node_idx] = state
        
        # Expand connection matrix
        new_size = len(self.nodes)
        new_connections = np.zeros((new_size, new_size))
        
        if new_size > 1:
            new_connections[:-1, :-1] = self.connections
        
        self.connections = new_connections
        
        return node_idx
    
    def update_model(self, next_state, action):
        """Minimal model update"""
        # Get pattern for current state
        current_pattern = self.get_firing_pattern(next_state)
        
        # Find or create node for current state
        current_idx = self.find_node_index(current_pattern)
        if current_idx is None:
            current_idx = self.create_node(current_pattern, next_state)
        
        # Connect to previous node if exists
        if self.prev_node_idx is not None:
            # Make sure both connections arrays are the same size
            if self.prev_node_idx < self.connections.shape[0] and current_idx < self.connections.shape[0]:
                self.connections[self.prev_node_idx, current_idx] = 1
                self.action_mappings[(self.prev_node_idx, current_idx)] = action
        
        # Update previous node index
        self.prev_node_idx = current_idx
    
    def select_action(self, current_state):
        """Basic action selection"""
        if self.goal is None:
            return random.randint(0, 3)
        
        if np.random.uniform(0, 1) > self.epsilon:
            # Get current node
            current_pattern = self.get_firing_pattern(current_state)
            current_idx = self.find_node_index(current_pattern)
            
            if current_idx is not None and current_idx < self.connections.shape[0]:
                # Get connected nodes
                connected = np.where(self.connections[current_idx] == 1)[0]
                
                if len(connected) > 0:
                    # Choose a random connected node
                    next_idx = np.random.choice(connected)
                    key = (current_idx, next_idx)
                    
                    if key in self.action_mappings:
                        return self.action_mappings[key]
        
        # Default to random action
        return random.randint(0, 3)
    
    def set_goal(self, goal):
        """Set the goal state"""
        self.goal = goal
    
    def set_epsilon(self, epsilon):
        """Set exploration rate"""
        self.epsilon = epsilon
    
    def decay_epsilon(self, min_epsilon=0.2):
        """Reduce exploration rate"""
        self.epsilon = max(self.epsilon - 0.1, min_epsilon)
    
    def reset_epsilon(self):
        """Reset exploration rate"""
        self.epsilon = self.start_epsilon
    
    def get_epsilon(self):
        """Get current exploration rate"""
        return self.epsilon
    
    def show_map(self):
        """Visualization method for compatibility"""
        graph = nx.DiGraph()
        
        # Add nodes with their positions
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
        plt.title("MinimalHGWRSOM Map")
        plt.show()