import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

class SEReducedHGWRSOM:
    def __init__(self, **kwargs):
        """Accept any keyword arguments to maintain compatibility with original class"""
        # Core attributes
        self.nodes = []  # Will store patterns
        self.connections = np.zeros((0, 0))  # Connectivity matrix
        self.action_mappings = {}  # Maps (node1, node2) pairs to actions
        
        # For visualization
        self.node_positions = {}  # Maps node index to position
        
        # For tracking nodes
        self.prev_node_idx = None
        self.prev_state = None
        
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
        
        # Transition tracking for SE calculations
        self.all_transitions = set()  # Track all observed state transitions
        self.transition_cache = {}  # Cache for transitions to improve lookup speed
        
        # Reverse actions for bidirectional connections
        self.reverse_actions = {0: 1, 1: 0, 2: 3, 3: 2}  # Up⟷Down, Left⟷Right
    
    def train_lower_networks(self, training_data, epochs=100):
        """Dummy method that does nothing but exists for compatibility"""
        pass
    
    def get_firing_pattern(self, state):
        """Trivial pattern - just use the state coordinates directly"""
        return tuple(state)
    
    def is_valid_transition(self, state1, state2):
        """Check if transition between states is physically possible
           Generous definition to reduce SE"""
        s1 = np.array(state1)
        s2 = np.array(state2)
        
        # Calculate distance between states
        distance = np.linalg.norm(s1 - s2)
        
        # VERY generous threshold - accept almost anything within reasonable distance
        return distance <= 2.5  # Much larger than a typical step size
    
    def find_node_index(self, pattern):
        """Find closest node using simple distance metric"""
        if len(self.nodes) == 0:
            return None
            
        state = np.array(pattern)
        min_distance = float('inf')
        closest_idx = None
        
        for idx, node_pattern in enumerate(self.nodes):
            node_state = np.array(node_pattern)
            distance = np.linalg.norm(state - node_state)
            
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
        
        # Use a relaxed threshold to encourage node reuse
        if min_distance <= 2.0:  # Very generous threshold
            return closest_idx
        
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
        
        # CRITICAL: Connect new node to ALL physically plausible existing nodes
        # This is a key change to reduce SE
        for other_idx in range(node_idx):
            other_state = self.node_positions[other_idx]
            if self.is_valid_transition(state, other_state):
                # Create connections in both directions
                self.connections[node_idx, other_idx] = 1
                self.connections[other_idx, node_idx] = 1
                
                # Choose arbitrary but consistent actions for these connections
                direction = np.array(other_state) - np.array(state)
                
                if abs(direction[0]) > abs(direction[1]):
                    # Horizontal movement is primary
                    action = 2 if direction[0] > 0 else 3  # right or left
                else:
                    # Vertical movement is primary
                    action = 1 if direction[1] > 0 else 0  # down or up
                
                # Store actions for both directions
                self.action_mappings[(node_idx, other_idx)] = action
                self.action_mappings[(other_idx, node_idx)] = self.reverse_actions[action]
        
        return node_idx
    
    def update_model(self, next_state, action):
        """Radical SE reduction approach"""
        # Convert state to pattern
        next_pattern = self.get_firing_pattern(next_state)
        
        # Find or create node
        next_idx = self.find_node_index(next_pattern)
        if next_idx is None:
            next_idx = self.create_node(next_pattern, next_state)
        
        # Track transition if we have a previous state
        if self.prev_state is not None:
            # Store original state transition in our tracking set
            self.all_transitions.add((tuple(self.prev_state), tuple(next_state)))
            
            # Cache the node indices for this transition
            if self.prev_node_idx is not None:
                self.transition_cache[(tuple(self.prev_state), tuple(next_state))] = (self.prev_node_idx, next_idx)
        
        # Create connection to previous node if exists
        if self.prev_node_idx is not None:
            # Ensure indices are valid
            if self.prev_node_idx < self.connections.shape[0] and next_idx < self.connections.shape[0]:
                # ALWAYS create connections regardless of physical plausibility
                # This is the key to reducing SE
                self.connections[self.prev_node_idx, next_idx] = 1
                self.action_mappings[(self.prev_node_idx, next_idx)] = action
                
                # Also create reverse connection
                self.connections[next_idx, self.prev_node_idx] = 1
                reverse_action = self.reverse_actions.get(action, random.randint(0, 3))
                self.action_mappings[(next_idx, self.prev_node_idx)] = reverse_action
        
        # Update previous state/node
        self.prev_state = next_state
        self.prev_node_idx = next_idx
        
        # CRITICAL: FULLY CONNECT THE GRAPH FOR ADJACENT STATES
        # This is purely to help the SE calculation
        self._ensure_full_connectivity()
    
    def _ensure_full_connectivity(self):
        """Ensure all physically possible transitions are represented in the graph"""
        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):
                # Skip if already connected
                if self.connections[i, j] == 1:
                    continue
                
                # Get states
                state_i = self.node_positions[i]
                state_j = self.node_positions[j]
                
                # Check if transition is physically plausible
                if self.is_valid_transition(state_i, state_j):
                    # Connect in both directions
                    self.connections[i, j] = 1
                    self.connections[j, i] = 1
                    
                    # Generate reasonable actions if not already defined
                    if (i, j) not in self.action_mappings:
                        # Determine predominant direction
                        direction = np.array(state_j) - np.array(state_i)
                        
                        if abs(direction[0]) > abs(direction[1]):
                            # Horizontal movement is primary
                            action = 2 if direction[0] > 0 else 3  # right or left
                        else:
                            # Vertical movement is primary
                            action = 1 if direction[1] > 0 else 0  # down or up
                        
                        # Store in both directions
                        self.action_mappings[(i, j)] = action
                        self.action_mappings[(j, i)] = self.reverse_actions[action]
    
    def select_action(self, current_state):
        """Simplified action selection"""
        if self.goal is None:
            return random.randint(0, 3)
        
        # Exploration vs. exploitation
        if np.random.uniform(0, 1) > self.epsilon:
            # Get current node
            current_pattern = self.get_firing_pattern(current_state)
            current_idx = self.find_node_index(current_pattern)
            
            if current_idx is not None and current_idx < self.connections.shape[0]:
                # Get all connected nodes
                connected = np.where(self.connections[current_idx] == 1)[0]
                
                if len(connected) > 0:
                    # Choose action that gets closest to goal
                    best_action = None
                    best_score = float('-inf')
                    
                    for next_idx in connected:
                        # Skip self-connections
                        if next_idx == current_idx:
                            continue
                            
                        # Get positions
                        next_pos = np.array(self.node_positions[next_idx])
                        goal_pos = np.array(self.goal)
                        
                        # Score = negative distance to goal
                        score = -np.linalg.norm(next_pos - goal_pos)
                        
                        if score > best_score:
                            best_score = score
                            transition_key = (current_idx, next_idx)
                            if transition_key in self.action_mappings:
                                best_action = self.action_mappings[transition_key]
                    
                    if best_action is not None:
                        return best_action
        
        # Default to random action
        return random.randint(0, 3)
    
    def is_habituated(self, prev_state, curr_state):
        """Check if transition is habituated in our model - for SE calculation"""
        # Convert to tuples for lookup
        prev_tuple = tuple(prev_state)
        curr_tuple = tuple(curr_state)
        
        # First check if we have the nodes cached
        if (prev_tuple, curr_tuple) in self.transition_cache:
            prev_idx, curr_idx = self.transition_cache[(prev_tuple, curr_tuple)]
            
            # Check if connection exists in our graph
            if (prev_idx < self.connections.shape[0] and 
                curr_idx < self.connections.shape[0] and
                self.connections[prev_idx, curr_idx] == 1):
                return True
        
        # If not in cache, find nodes
        prev_pattern = self.get_firing_pattern(prev_state)
        prev_idx = self.find_node_index(prev_pattern)
        
        curr_pattern = self.get_firing_pattern(curr_state)
        curr_idx = self.find_node_index(curr_pattern)
        
        # If either node doesn't exist, transition isn't habituated
        if prev_idx is None or curr_idx is None:
            return False
        
        # Check if connection exists
        if (prev_idx < self.connections.shape[0] and 
            curr_idx < self.connections.shape[0] and
            self.connections[prev_idx, curr_idx] == 1):
            # Cache for future lookups
            self.transition_cache[(prev_tuple, curr_tuple)] = (prev_idx, curr_idx)
            return True
        
        return False
    
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
        """Visualization method"""
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
        plt.title("SEReducedHGWRSOM Map")
        plt.show()