import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

class EnhancedHGWRSOM:
    def __init__(self, **kwargs):
        """Accept any keyword arguments to maintain compatibility with original class"""
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
        
        # Enhanced parameters
        self.similarity_threshold = 0.8  # Threshold for node matching (adjustable)
        self.node_pos_learning_rate = 0.3  # For updating node positions
        self.transition_memory = {}  # For tracking transitions
        self.transition_counts = {}  # For counting transitions
        
        # For reverse connections
        self.reverse_actions = {0: 1, 1: 0, 2: 3, 3: 2}  # Up⟷Down, Left⟷Right
    
    def train_lower_networks(self, training_data, epochs=100):
        """Dummy method that does nothing but exists for compatibility"""
        pass
    
    def get_firing_pattern(self, state):
        """Trivial pattern - just use the state coordinates directly"""
        return tuple(state)
    
    def state_similarity(self, state1, state2):
        """Calculate similarity between two states"""
        # Convert to numpy arrays if they're not already
        s1 = np.array(state1)
        s2 = np.array(state2)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(s1 - s2)
        
        # Convert to similarity (1.0 = identical, 0.0 = very different)
        # Using a Gaussian kernel with σ = 1.0
        similarity = np.exp(-distance**2 / 2.0)
        
        return similarity
    
    def find_node_index(self, pattern):
        """Find existing node with similarity matching"""
        if len(self.nodes) == 0:
            return None
            
        best_idx = None
        best_similarity = 0
        
        # Extract state from pattern
        state = np.array(pattern)
        
        # Find the most similar node
        for idx in range(len(self.nodes)):
            node_state = np.array(self.nodes[idx])
            similarity = self.state_similarity(state, node_state)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
        
        # Return best match if similarity is above threshold
        if best_similarity >= self.similarity_threshold:
            return best_idx
        
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
        """Enhanced model update with transition learning"""
        # Get pattern for current state
        current_pattern = self.get_firing_pattern(next_state)
        
        # Find or create node for current state
        current_idx = self.find_node_index(current_pattern)
        if current_idx is None:
            current_idx = self.create_node(current_pattern, next_state)
        else:
            # Update node position with learning rate
            old_pos = np.array(self.node_positions[current_idx])
            new_pos = np.array(next_state)
            updated_pos = (1-self.node_pos_learning_rate) * old_pos + self.node_pos_learning_rate * new_pos
            self.node_positions[current_idx] = tuple(updated_pos)
        
        # Connect to previous node if exists
        if self.prev_node_idx is not None:
            # Ensure array dimensions match
            if self.prev_node_idx < self.connections.shape[0] and current_idx < self.connections.shape[0]:
                # Create forward connection
                self.connections[self.prev_node_idx, current_idx] = 1
                self.action_mappings[(self.prev_node_idx, current_idx)] = action
                
                # Create backward connection with reverse action
                self.connections[current_idx, self.prev_node_idx] = 1
                reverse_action = self.reverse_actions.get(action, random.randint(0, 3))
                self.action_mappings[(current_idx, self.prev_node_idx)] = reverse_action
                
                # Record transition for future reference
                transition_key = (self.prev_node_idx, current_idx)
                if transition_key in self.transition_counts:
                    self.transition_counts[transition_key] += 1
                else:
                    self.transition_counts[transition_key] = 1
                    
                # Store position-based transition
                self.transition_memory[transition_key] = (
                    self.node_positions[self.prev_node_idx],
                    self.node_positions[current_idx]
                )
                
                # Update similar transitions based on physical similarity
                # This helps reduce SE by ensuring physically similar transitions are connected
                prev_pos = np.array(self.node_positions[self.prev_node_idx])
                curr_pos = np.array(self.node_positions[current_idx])
                
                # Create connections for similar nodes
                for i in range(len(self.nodes)):
                    if i == self.prev_node_idx or i == current_idx:
                        continue
                        
                    i_pos = np.array(self.node_positions[i])
                    
                    # Check if node i is close to prev_node
                    if np.linalg.norm(i_pos - prev_pos) < 1.5:
                        for j in range(len(self.nodes)):
                            if j == i:
                                continue
                                
                            j_pos = np.array(self.node_positions[j])
                            
                            # Check if node j is close to curr_node
                            # AND the transition i->j is physically similar to prev->curr
                            if np.linalg.norm(j_pos - curr_pos) < 1.5:
                                # Calculate vector similarity
                                v1 = curr_pos - prev_pos
                                v2 = j_pos - i_pos
                                
                                # Normalize vectors
                                v1_norm = np.linalg.norm(v1)
                                v2_norm = np.linalg.norm(v2)
                                
                                # Avoid division by zero
                                if v1_norm > 0 and v2_norm > 0:
                                    v1 = v1 / v1_norm
                                    v2 = v2 / v2_norm
                                    
                                    # Calculate direction similarity
                                    direction_sim = np.dot(v1, v2)
                                    
                                    # Only connect if directions are similar
                                    if direction_sim > 0.7:  # Cosine similarity threshold
                                        self.connections[i, j] = 1
                                        # Use the same action for similar transitions
                                        self.action_mappings[(i, j)] = action
        
        # Update previous node index
        self.prev_node_idx = current_idx
    
    def select_action(self, current_state):
        """Enhanced action selection with goal-directed behavior"""
        if self.goal is None:
            return random.randint(0, 3)
        
        # Exploration vs. exploitation
        if np.random.uniform(0, 1) > self.epsilon:
            # Get current node
            current_pattern = self.get_firing_pattern(current_state)
            current_idx = self.find_node_index(current_pattern)
            
            if current_idx is not None and current_idx < self.connections.shape[0]:
                # Get position of current node and goal
                current_pos = np.array(self.node_positions[current_idx])
                goal_pos = np.array(self.goal)
                
                # Get all connected nodes
                connected = np.where(self.connections[current_idx] == 1)[0]
                
                if len(connected) > 0:
                    # Score each connected node by how much closer it gets to the goal
                    best_score = -float('inf')
                    best_action = None
                    
                    for next_idx in connected:
                        next_pos = np.array(self.node_positions[next_idx])
                        
                        # Current distance to goal
                        current_dist = np.linalg.norm(current_pos - goal_pos)
                        # Next distance to goal
                        next_dist = np.linalg.norm(next_pos - goal_pos)
                        
                        # Score = how much closer we get
                        score = current_dist - next_dist
                        
                        # Check if this transition has been successful before
                        transition_key = (current_idx, next_idx)
                        familiarity = self.transition_counts.get(transition_key, 0)
                        
                        # Combine score with familiarity
                        combined_score = score + 0.1 * familiarity
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            
                            # Get the action for this transition
                            key = (current_idx, next_idx)
                            if key in self.action_mappings:
                                best_action = self.action_mappings[key]
                    
                    if best_action is not None:
                        return best_action
                
                # If we have no good connected nodes, try to move directly toward goal
                # Calculate direction vector to goal
                goal_direction = goal_pos - current_pos
                
                # Determine dominant direction (up/down or left/right)
                if abs(goal_direction[0]) > abs(goal_direction[1]):
                    # Horizontal movement is more important
                    return 2 if goal_direction[0] > 0 else 3  # right or left
                else:
                    # Vertical movement is more important
                    return 1 if goal_direction[1] > 0 else 0  # down or up
        
        # Random exploration
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
        plt.title("EnhancedHGWRSOM Map")
        plt.show()