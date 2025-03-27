import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import pickle
import os
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.TMGWR_agent import TMGWRAgent
from Agents.HSOM_binary import HierarchicalGWRSOMAgent

# Add new function to collect and analyze MINERVA binary pattern mappings
def analyze_minerva_binary_patterns(noise_level=0, grid_size=20, range_min=-120, range_max=120):
    """
    Generate a comprehensive mapping of states to binary patterns for MINERVA.
    
    Parameters:
    - noise_level: The noise level to add to states (default: 0)
    - grid_size: Number of points in each dimension to sample (default: 20)
    - range_min/max: Min/max coordinate values to sample (default: -120 to 120)
    
    Returns:
    - DataFrame containing the mapping information
    """
    print(f"Analyzing MINERVA binary patterns with noise level σ² = {noise_level}...")
    
    # Initialize MINERVA agent
    hgwrsom_agent = HierarchicalGWRSOMAgent(
        lower_dim=1, higher_dim=2, epsilon_b=0.35,
        epsilon_n=0.15, beta=0.7, delta=0.79,
        T_max=20, N_max=100, eta=0.5,
        phi=0.9, sigma=0.5
    )
    
    # Training data
    x_train = np.linspace(range_min, range_max, 10).reshape(-1, 1)
    y_train = np.linspace(range_min, range_max, 10).reshape(-1, 1)
    training_data = np.hstack((x_train, y_train))
    
    # Train lower networks
    hgwrsom_agent.train_lower_networks(training_data, epochs=100)
    
    # Generate grid of states to analyze
    x_coords = np.linspace(range_min, range_max, grid_size)
    y_coords = np.linspace(range_min, range_max, grid_size)
    states = []
    
    # Create a list of all states in the grid
    for x in x_coords:
        for y in y_coords:
            states.append([x, y])
    
    # Set seed for reproducible noise
    np.random.seed(42)
    
    # Store mappings
    mapping_data = []
    
    # Process each state
    for state in states:
        # Add noise if specified
        if noise_level > 0:
            noisy_state = np.array(state) + np.random.normal(0, np.sqrt(noise_level), 2)
        else:
            noisy_state = np.array(state)
        
        # Get binary pattern
        pattern = hgwrsom_agent.get_firing_pattern(noisy_state)
        
        # Find active neurons
        x_active = np.where(np.array(pattern[0]) > 0)[0]
        y_active = np.where(np.array(pattern[1]) > 0)[0]
        
        # Convert pattern to a hashable string for grouping
        pattern_str = f"x{x_active[0]}_y{y_active[0]}"
        
        # Add to mapping data
        mapping_data.append({
            'x': state[0],
            'y': state[1],
            'noisy_x': noisy_state[0],
            'noisy_y': noisy_state[1],
            'x_neuron': int(x_active[0]),
            'y_neuron': int(y_active[0]),
            'pattern_str': pattern_str
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(mapping_data)
    
    # Count how many states map to each pattern
    pattern_counts = df.groupby('pattern_str').size().reset_index(name='state_count')
    
    # Merge count information back to the main dataframe
    df = df.merge(pattern_counts, on='pattern_str')
    
    # Identify patterns with multiple states
    df['is_collision'] = df['state_count'] > 1
    
    # Print summary statistics
    total_patterns = df['pattern_str'].nunique()
    total_states = len(df)
    collision_patterns = df[df['is_collision']]['pattern_str'].nunique()
    collision_states = df[df['is_collision']].shape[0]
    
    print(f"Total unique patterns: {total_patterns}")
    print(f"Total states: {total_states}")
    print(f"Patterns with collisions: {collision_patterns} ({collision_patterns/total_patterns*100:.2f}%)")
    print(f"States in collisions: {collision_states} ({collision_states/total_states*100:.2f}%)")
    
    # Calculate theoretical purity
    states_per_pattern = defaultdict(int)
    for pattern, states in df.groupby('pattern_str').size().items():
        states_per_pattern[pattern] = states
        
    max_states_per_pattern = {pattern: 1 for pattern in df['pattern_str'].unique()}
    total_purity = sum(max_states_per_pattern.values()) / total_states * 100
    print(f"Theoretical purity: {total_purity:.2f}%")
    
    return df

def visualize_binary_mappings(df, save_path="minerva_binary_analysis"):
    """
    Create visualizations of the MINERVA binary pattern mappings.
    
    Parameters:
    - df: DataFrame containing mapping information
    - save_path: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Plot pattern distribution across space
    plt.figure(figsize=(12, 10))
    # Create a color map with enough distinct colors
    unique_patterns = df['pattern_str'].unique()
    n_patterns = len(unique_patterns)
    pattern_to_idx = {pattern: i for i, pattern in enumerate(unique_patterns)}
    df['pattern_idx'] = df['pattern_str'].map(pattern_to_idx)
    
    # Plot points colored by pattern
    scatter = plt.scatter(df['x'], df['y'], c=df['pattern_idx'], cmap='viridis', 
                          alpha=0.7, s=50)
    plt.colorbar(scatter, label='Binary Pattern Index')
    plt.title('MINERVA Binary Pattern Distribution', fontsize=14)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/pattern_distribution.png")
    
    # 2. Plot active neuron heatmap for x dimension
    plt.figure(figsize=(12, 10))
    heatmap_x = df.pivot_table(index='y', columns='x', values='x_neuron', aggfunc='mean')
    sns.heatmap(heatmap_x, cmap='viridis')
    plt.title('X Dimension Active Neurons', fontsize=14)
    plt.savefig(f"{save_path}/x_neurons_heatmap.png")
    
    # 3. Plot active neuron heatmap for y dimension
    plt.figure(figsize=(12, 10))
    heatmap_y = df.pivot_table(index='y', columns='x', values='y_neuron', aggfunc='mean')
    sns.heatmap(heatmap_y, cmap='viridis')
    plt.title('Y Dimension Active Neurons', fontsize=14)
    plt.savefig(f"{save_path}/y_neurons_heatmap.png")
    
    # 4. Highlight pattern collisions
    plt.figure(figsize=(12, 10))
    # Plot all points in gray
    plt.scatter(df['x'], df['y'], color='lightgray', s=30, alpha=0.3)
    # Highlight collision points in red
    collision_df = df[df['is_collision']]
    plt.scatter(collision_df['x'], collision_df['y'], color='red', s=50, alpha=0.7)
    plt.title('States with Pattern Collisions', fontsize=14)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/pattern_collisions.png")
    
    # 5. Create a visualization of specific collision examples
    # Find patterns with the most collisions
    top_collision_patterns = df[df['is_collision']].groupby('pattern_str').size().nlargest(5)
    
    plt.figure(figsize=(15, 12))
    for i, (pattern, count) in enumerate(top_collision_patterns.items()):
        plt.subplot(2, 3, i+1)
        pattern_states = df[df['pattern_str'] == pattern]
        plt.scatter(pattern_states['x'], pattern_states['y'], color='red', s=80)
        plt.title(f"Pattern {pattern}: {count} states", fontsize=12)
        plt.xlabel('X Coordinate', fontsize=10)
        plt.ylabel('Y Coordinate', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/collision_examples.png")
    
    # Save complete analysis as a CSV file
    df.to_csv(f"{save_path}/binary_mapping_data.csv", index=False)
    
    print(f"Visualizations and data saved to {save_path}/")
    return True

def generate_pattern_clash_examples(df, save_path="minerva_binary_analysis"):
    """
    Find specific examples of pattern clashes and save them in a readable format.
    
    Parameters:
    - df: DataFrame containing mapping information
    - save_path: Directory to save examples
    """
    # Find patterns with multiple states
    collision_patterns = df[df['is_collision']]['pattern_str'].unique()
    
    # Prepare a list to store clash examples
    clash_examples = []
    
    # For each collision pattern, find examples
    for pattern in collision_patterns:
        # Get states that map to this pattern
        states = df[df['pattern_str'] == pattern][['x', 'y', 'x_neuron', 'y_neuron']].values
        
        # Find the two states that are farthest apart
        max_distance = 0
        max_pair = None
        
        for i in range(len(states)):
            for j in range(i+1, len(states)):
                state1 = states[i, :2]  # x, y coordinates of first state
                state2 = states[j, :2]  # x, y coordinates of second state
                distance = np.linalg.norm(state1 - state2)
                
                if distance > max_distance:
                    max_distance = distance
                    max_pair = (state1, state2, states[i, 2:], pattern)
        
        if max_pair is not None:
            clash_examples.append({
                'pattern': pattern,
                'state1': max_pair[0],
                'state2': max_pair[1],
                'neurons': max_pair[2],  # x_neuron, y_neuron
                'distance': max_distance
            })
    
    # Sort examples by distance (most distant first)
    clash_examples.sort(key=lambda x: x['distance'], reverse=True)
    
    # Save clash examples to a text file
    with open(f"{save_path}/pattern_clash_examples.txt", 'w') as f:
        f.write("Top Pattern Clash Examples (States mapping to the same binary pattern):\n")
        f.write("=" * 80 + "\n\n")
        
        for i, example in enumerate(clash_examples[:10]):  # Top 10 examples
            f.write(f"Example {i+1}:\n")
            f.write(f"  Pattern: {example['pattern']}\n")
            f.write(f"  Active neurons: X-neuron {int(example['neurons'][0])}, Y-neuron {int(example['neurons'][1])}\n")
            f.write(f"  State 1: ({example['state1'][0]:.2f}, {example['state1'][1]:.2f})\n")
            f.write(f"  State 2: ({example['state2'][0]:.2f}, {example['state2'][1]:.2f})\n")
            f.write(f"  Distance between states: {example['distance']:.2f}\n\n")
    
    print(f"Pattern clash examples saved to {save_path}/pattern_clash_examples.txt")
    
    # Return top examples for immediate viewing
    return clash_examples[:5]

def calculate_purity(state_node_mappings, total_visits):
    """Calculate purity metric based on the mathematical formula with improved reliability"""
    if total_visits == 0:
        return 0
    
    total_purity = 0
    
    # Print debug info on state-node mapping distribution
    node_counts = {node: sum(state_counts.values()) for node, state_counts in state_node_mappings.items()}
    
    for node, state_counts in state_node_mappings.items():
        # Find the maximum count for this node (maximum overlap with any state)
        if state_counts:  # Check if the dictionary is not empty
            max_state_count = max(state_counts.values())
            total_purity += max_state_count
    
    return (total_purity / total_visits) * 100

def calculate_se(transitions, agent, tau=0.0001):
    """
    Calculate sensorimotor representation error (SE) with consistent criteria and improved reliability
    SE = (∑|E|i=1 I{Ei[w⃗{1}t−1,w⃗{1}t]∉H} + τ) / (|E| + τ)
    """
    if not transitions:
        return 0
    
    error_count = 0
    E = len(transitions)
    
    # MINERVA specific handling
    if hasattr(agent, 'nodes'):  # This is a MINERVA agent
        for prev_state, curr_state in transitions:
            is_habituated = False
            
            prev_pattern = agent.get_firing_pattern(prev_state)
            curr_pattern = agent.get_firing_pattern(curr_state)
            
            prev_node_idx = agent.find_node_index(prev_pattern)
            curr_node_idx = agent.find_node_index(curr_pattern)
            
            if prev_node_idx is not None and curr_node_idx is not None:
                # Check connection only if both nodes exist
                if prev_node_idx < agent.connections.shape[0] and curr_node_idx < agent.connections.shape[1]:
                    is_habituated = agent.connections[prev_node_idx, curr_node_idx] == 1

            if not is_habituated:
                error_count += 1
    
    # TMGWR specific handling
    elif hasattr(agent, 'model'):  # This is a TMGWR agent
        for prev_state, curr_state in transitions:
            is_habituated = False
            
            prev_node = agent.model.get_node_index(prev_state)
            curr_node = agent.model.get_node_index(curr_state)
            
            # Check if nodes exist and are connected
            if prev_node < agent.model.C.shape[0] and curr_node < agent.model.C.shape[1]:
                is_habituated = agent.model.C[prev_node, curr_node] == 1

            if not is_habituated:
                error_count += 1

    SE = (error_count + tau) / (E + tau)
    return SE
def demonstrate_node_creation(agent, current_state, noise_level, agent_name):
    """
    Demonstrate how nodes are created or adapted under noise for MINERVA and TMGWR.
    """
    np.random.seed(42)  # For reproducibility
    
    # Add noise to the current state
    noisy_state = np.array(current_state) + np.random.normal(0, np.sqrt(noise_level), 2)
    
    result = {
        'Original State': current_state,
        'Noisy State': noisy_state.tolist(),
        'Agent': agent_name,
        'Interpretation': None
    }
    
    if agent_name == 'MINERVA':
        # Use lower networks to get activity
        x_data = np.array([noisy_state[0]]).reshape(1, -1)
        y_data = np.array([noisy_state[1]]).reshape(1, -1)
        
        x_bmu, _ = agent.lower_x.find_best_matching_units(x_data)
        y_bmu, _ = agent.lower_y.find_best_matching_units(y_data)
        
        x_binary = np.zeros(len(agent.lower_x.A))
        y_binary = np.zeros(len(agent.lower_y.A))
        x_binary[x_bmu] = 1
        y_binary[y_bmu] = 1
        
        pattern = (tuple(x_binary), tuple(y_binary))
        
        # Get or create node index
        found_idx = agent.find_node_index(pattern)
        
        if found_idx is None:
            # Node doesn't exist yet, would be created
            node_idx = len(agent.nodes)  # This would be the new node's index
        else:
            node_idx = found_idx
        
        result['Interpretation'] = {
            'Binary Pattern': pattern,
            'Node Index': node_idx
        }
    
    elif agent_name == 'TMGWR':
        # Calculate distance between noisy state and node position
        node_idx = agent.model.get_node_index(noisy_state)
        node_position = agent.model.W[node_idx]
        
        distance = np.linalg.norm(noisy_state - node_position)
        
        # Define a threshold for node adaptation
        threshold = 15
        
        if distance < threshold:
            # Adapt the existing node
            learning_rate = 0.2
            new_node_position = node_position + learning_rate * (noisy_state - node_position)
            result['Interpretation'] = {
                'Original Node Position': node_position.tolist(),
                'Distance': distance,
                'New Node Position': new_node_position.tolist()
            }
        else:
            # Create a new node
            result['Interpretation'] = {
                'Original Node Position': node_position.tolist(),
                'Distance': distance,
                'New Node Created at': noisy_state.tolist()
            }
    
    return result
def run_enhanced_noise_comparison(noise_levels=[0, 1/6, 1/3, 1/2, 2/3, 5/6, 1], episodes_per_noise=3):
    """
    Enhanced version of run_noise_comparison that also collects state-to-pattern mappings.
    All required functions are defined within this script.
    """
    # Same initialization as original function
    results = {
        'TMGWR': {level: {'nodes': [], 'purity': [], 'se': []} for level in noise_levels},
        'MINERVA': {level: {'nodes': [], 'purity': [], 'se': []} for level in noise_levels}
    }
    
    # Enhanced to store pattern mappings
    pattern_mappings = {level: [] for level in noise_levels}
    
    # Initialize table data storage
    table_data = []

    # Get maze details
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()

    # Smaller, simpler training set for HGWRSOM
    x_train = np.linspace(-120, 120, 50).reshape(-1, 1)
    y_train = np.linspace(-120, 120, 50).reshape(-1, 1)
    training_data = np.hstack((x_train, y_train))

    # Set consistent random seed for reproducibility
    np.random.seed(42)
    
    for noise_level in noise_levels:
        print(f"\nRunning simulations with noise level σ² = {noise_level}")
        
        for episode in range(episodes_per_noise):
            print(f"\nEpisode {episode + 1}/{episodes_per_noise}")
            
            # Set episode-specific seed for controlled variation
            episode_seed = 42 + episode
            np.random.seed(episode_seed)
            
            Maze = MazePlayer(maze_map=maze_map, 
                            player_index_pos=player_pos_index, 
                            goal_index_pos=goal_pos_index)
            
            goal = Maze.get_goal_pos()
            initial_state = Maze.get_initial_player_pos()

            # Initialize TMGWR agent
            tmgwr_agent = TMGWRAgent(
                nDim=2, Ni=2, epsilon_b=0.55194, epsilon_n=0.90,
                beta=0.8, delta=0.6235, T_max=17, N_max=300,
                eta=0.95, phi=0.6, sigma=1
            )
            tmgwr_agent.set_goal(goal)
            tmgwr_agent.set_epsilon(1)  # Pure exploration

            # Initialize HGWRSOM agent
            hgwrsom_agent = HierarchicalGWRSOMAgent(
                lower_dim=1, higher_dim=2, epsilon_b=0.35,
                epsilon_n=0.15, beta=0.7, delta=0.79,
                T_max=20, N_max=100, eta=0.5,
                phi=0.9, sigma=0.5
            )
            
            # NEW: Create storage for state-pattern mappings
            state_to_pattern = {}
            
            # Run agents in sequence to ensure fair comparison
            for agent_idx, (agent, agent_name) in enumerate([(tmgwr_agent, 'TMGWR'), (hgwrsom_agent, 'MINERVA')]):
                try:
                    # Reset random seed for consistent noise patterns between agents
                    np.random.seed(episode_seed + agent_idx)
                    
                    if agent_name == 'MINERVA':
                        # Train lower networks with smaller epochs
                        agent.train_lower_networks(training_data, epochs=100)
                        
                        # IMPORTANT: Ensure nodes and connections are properly initialized
                        agent.nodes = []  # Reset nodes list
                        agent.connections = np.zeros((0, 0))  # Reset connections
                        agent.pattern_ages = np.zeros((0, 0))  # Reset pattern ages
                    
                    agent.set_goal(goal)
                    agent.set_epsilon(1)  # Pure exploration
                    
                    # Run episode
                    Maze.reset_player()  # Reset for each agent
                    current_state = initial_state
                    step_counter = 0
                    max_steps = 5000  # Limit to prevent infinite loops
                    
                    state_node_mappings = defaultdict(lambda: defaultdict(int))
                    transitions = []
                    total_visits = 0
                    
                    # Main episode loop
                    while current_state != goal and step_counter < max_steps:
                        step_counter += 1
                        
                        # Print progress occasionally
                        if step_counter % 1000 == 0:
                            print(f"{agent_name}: Step {step_counter}, Nodes: {len(agent.nodes) if agent_name == 'MINERVA' else len(agent.model.W)}")
                        
                        prev_state = np.array(current_state)
                        
                        # Add noise to current state observation
                        noisy_state = np.array(current_state) + np.random.normal(0, np.sqrt(noise_level), 2)
                        
                        # Get node assignment for current state
                        if agent_name == 'TMGWR':
                            node_idx = agent.model.get_node_index(noisy_state)
                        else:
                            # For MINERVA, use its native methods
                            pattern = agent.get_firing_pattern(noisy_state)
                            
                            # NEW: Record the mapping of state to pattern
                            state_key = tuple(current_state)
                            if state_key not in state_to_pattern:
                                # Extract active neurons for easier analysis
                                x_active = np.where(np.array(pattern[0]) > 0)[0]
                                y_active = np.where(np.array(pattern[1]) > 0)[0]
                                
                                # Store the full mapping
                                state_to_pattern[state_key] = {
                                    'pattern': pattern,
                                    'x_neuron': int(x_active[0]) if len(x_active) > 0 else -1,
                                    'y_neuron': int(y_active[0]) if len(y_active) > 0 else -1,
                                    'noisy_state': noisy_state.tolist()
                                }
                            
                            found_idx = agent.find_node_index(pattern)
                            
                            if found_idx is None:
                                # Pattern doesn't exist yet
                                node_idx = len(agent.nodes)
                            else:
                                node_idx = found_idx
                        
                        # Update state-node mapping counts
                        state_tuple = tuple(current_state)
                        state_node_mappings[node_idx][state_tuple] += 1
                        total_visits += 1
                        
                        # Select and execute action
                        action = agent.select_action(noisy_state)
                        Maze.move_player(action)
                        next_state = Maze.get_player_pos()
                        
                        # Ensure next_state is not None or invalid
                        if next_state is None:
                            print(f"Warning: Invalid next_state after action {action}")
                            break
                            
                        # Update model with actual next state (no noise)
                        agent.update_model(next_state, action)
                        
                        # Record transition
                        transitions.append((prev_state, next_state))
                        current_state = next_state
                    
                    # Check if episode terminated normally
                    if current_state == goal:
                        print(f"{agent_name} reached goal in {step_counter} steps")
                    else:
                        print(f"{agent_name} did not reach goal, stopped after {step_counter} steps")
                    
                    # Calculate metrics using the calculate_purity and calculate_se functions defined above
                    if total_visits > 0:
                        purity = calculate_purity(state_node_mappings, total_visits)
                    else:
                        purity = 0.0  # Default if no visits
                        
                    if len(transitions) > 0:
                        se = calculate_se(transitions, agent)
                    else:
                        se = 0.0  # Default if no transitions
                    
                    # Record number of nodes
                    if agent_name == 'TMGWR':
                        num_nodes = len(agent.model.W)
                    else:
                        num_nodes = len(agent.nodes)
                    
                    # Print detailed debug info
                    print(f"{agent_name} episode summary:")
                    print(f"  Nodes: {num_nodes}")
                    print(f"  Purity: {purity:.2f}%")
                    print(f"  SE: {se:.4f}")
                    print(f"  Total states visited: {total_visits}")
                    print(f"  Transitions recorded: {len(transitions)}")
                    
                    # Store results
                    results[agent_name][noise_level]['nodes'].append(num_nodes)
                    results[agent_name][noise_level]['purity'].append(purity)
                    results[agent_name][noise_level]['se'].append(se)
                    
                    # Demonstrate node creation for the current state - use the demonstrate_node_creation function defined above
                    node_creation_info = demonstrate_node_creation(agent, current_state, noise_level, agent_name)
                    table_data.append(node_creation_info)
                
                except Exception as e:
                    # More detailed error reporting
                    print(f"Error in {agent_name} episode {episode} with noise level {noise_level}: {str(e)}")
                    import traceback
                    traceback.print_exc()  # Print full stack trace for debugging
                    print(f"Agent state: {agent_name} has {len(agent.nodes) if agent_name == 'MINERVA' else len(agent.model.W)} nodes")
                    
                    # Record zeros for failed episodes, but clearly mark them
                    results[agent_name][noise_level]['nodes'].append(-1)  # Use -1 to indicate failure
                    results[agent_name][noise_level]['purity'].append(0)
                    results[agent_name][noise_level]['se'].append(0)
            
            # Store the pattern mappings for this episode
            if state_to_pattern:
                pattern_mappings[noise_level].append(state_to_pattern)
    
    # Save pattern mappings to a file for later analysis
    with open("minerva_pattern_mappings.pkl", "wb") as f:
        pickle.dump(pattern_mappings, f)
    
    # Post-process results to exclude failed episodes
    for agent_name in results:
        for noise_level in results[agent_name]:
            # Filter out failed episodes (marked with -1)
            valid_nodes = [n for n in results[agent_name][noise_level]['nodes'] if n >= 0]
            valid_indices = [i for i, n in enumerate(results[agent_name][noise_level]['nodes']) if n >= 0]
            
            if valid_nodes:
                results[agent_name][noise_level]['nodes'] = valid_nodes
                results[agent_name][noise_level]['purity'] = [results[agent_name][noise_level]['purity'][i] for i in valid_indices]
                results[agent_name][noise_level]['se'] = [results[agent_name][noise_level]['se'][i] for i in valid_indices]
            else:
                # If all episodes failed, keep one zero entry for reporting
                results[agent_name][noise_level]['nodes'] = [0]
                results[agent_name][noise_level]['purity'] = [0]
                results[agent_name][noise_level]['se'] = [0]

    return results, table_data, pattern_mappings

    
    # Save pattern mappings to a file for later analysis
    with open("minerva_pattern_mappings.pkl", "wb") as f:
        pickle.dump(pattern_mappings, f)
    
    # Post-process results to exclude failed episodes
    for agent_name in results:
        for noise_level in results[agent_name]:
            # Filter out failed episodes (marked with -1)
            valid_nodes = [n for n in results[agent_name][noise_level]['nodes'] if n >= 0]
            valid_indices = [i for i, n in enumerate(results[agent_name][noise_level]['nodes']) if n >= 0]
            
            if valid_nodes:
                results[agent_name][noise_level]['nodes'] = valid_nodes
                results[agent_name][noise_level]['purity'] = [results[agent_name][noise_level]['purity'][i] for i in valid_indices]
                results[agent_name][noise_level]['se'] = [results[agent_name][noise_level]['se'][i] for i in valid_indices]
            else:
                # If all episodes failed, keep one zero entry for reporting
                results[agent_name][noise_level]['nodes'] = [0]
                results[agent_name][noise_level]['purity'] = [0]
                results[agent_name][noise_level]['se'] = [0]

    return results, table_data, pattern_mappings
def analyze_saved_pattern_mappings(file_path="minerva_pattern_mappings.pkl"):
    """
    Analyze previously saved pattern mappings to find concrete examples of the same pattern
    mapping to different states.
    """
    # Load the saved pattern mappings
    with open(file_path, "rb") as f:
        pattern_mappings = pickle.load(f)
    
    # For each noise level
    for noise_level, episodes in pattern_mappings.items():
        print(f"\nAnalyzing noise level σ² = {noise_level}")
        
        for episode_idx, mapping in enumerate(episodes):
            print(f"  Episode {episode_idx + 1}:")
            
            # Count patterns and find collisions
            patterns_to_states = defaultdict(list)
            
            # Group states by their patterns
            for state, data in mapping.items():
                pattern_key = (data['x_neuron'], data['y_neuron'])
                patterns_to_states[pattern_key].append(state)
            
            # Count number of unique patterns
            unique_patterns = len(patterns_to_states)
            total_states = len(mapping)
            
            # Find patterns that map to multiple states
            collision_patterns = {p: states for p, states in patterns_to_states.items() if len(states) > 1}
            collision_count = len(collision_patterns)
            collision_state_count = sum(len(states) for states in collision_patterns.values())
            
            print(f"    Unique patterns: {unique_patterns}")
            print(f"    Total states: {total_states}")
            print(f"    Patterns with collisions: {collision_count} ({collision_count/unique_patterns*100:.2f}%)")
            print(f"    States in collisions: {collision_state_count} ({collision_state_count/total_states*100:.2f}%)")
            
            # Find the pattern with the most states and show as an example
            if collision_patterns:
                most_collisions = max(collision_patterns.items(), key=lambda x: len(x[1]))
                pattern, states = most_collisions
                
                print(f"\n    Example collision - Pattern x{pattern[0]}_y{pattern[1]} maps to {len(states)} states:")
                for i, state in enumerate(states[:5]):  # Show at most 5 states
                    state_data = mapping[state]
                    print(f"      State {i+1}: {state} (noisy: {state_data['noisy_state']})")
                
                # Find the two states with the largest distance between them
                max_distance = 0
                max_pair = None
                
                for i in range(len(states)):
                    for j in range(i+1, len(states)):
                        state1 = np.array(states[i])
                        state2 = np.array(states[j])
                        distance = np.linalg.norm(state1 - state2)
                        
                        if distance > max_distance:
                            max_distance = distance
                            max_pair = (states[i], states[j])
                
                if max_pair:
                    print(f"\n    Most distant states with same pattern:")
                    print(f"      State A: {max_pair[0]}")
                    print(f"      State B: {max_pair[1]}")
                    print(f"      Distance: {max_distance:.2f}")
                    
                    # Calculate purity for this pattern
                    states_per_pattern = sum(1 for states in patterns_to_states.values())
                    theoretical_purity = states_per_pattern / total_states * 100
                    print(f"\n    Theoretical purity: {theoretical_purity:.2f}%")

if __name__ == "__main__":
    print("MINERVA Binary Pattern Analysis")
    print("=" * 40)
    
    # First, run a grid-based analysis to see how states map to binary patterns
    print("\n1. Analyzing MINERVA binary patterns on a grid...")
    mapping_df = analyze_minerva_binary_patterns(noise_level=0, grid_size=40)
    visualize_binary_mappings(mapping_df)
    top_clashes = generate_pattern_clash_examples(mapping_df)
    
    print("\n2. Top pattern clash examples:")
    for i, example in enumerate(top_clashes):
        print(f"  Example {i+1}:")
        print(f"    Pattern: {example['pattern']}")
        print(f"    Active neurons: X-neuron {int(example['neurons'][0])}, Y-neuron {int(example['neurons'][1])}")
        print(f"    State 1: ({example['state1'][0]:.2f}, {example['state1'][1]:.2f})")
        print(f"    State 2: ({example['state2'][0]:.2f}, {example['state2'][1]:.2f})")
        print(f"    Distance between states: {example['distance']:.2f}")
    
    # Option 1: Run a new experiment and collect pattern data
    print("\n3. Running enhanced noise comparison to collect pattern data...")
    results, table_data, pattern_mappings = run_enhanced_noise_comparison(noise_levels=[0, 1/3, 2/3, 1])
    
    # Print a subset of the table data
    print("\nNode Creation Table (sample):")
    for entry in table_data[:10]:  # First 10 entries
        print(f"State: {entry['Original State']}, Agent: {entry['Agent']}, Node: {entry['Interpretation']['Node Index'] if 'Node Index' in entry['Interpretation'] else 'N/A'}")
    
    # Save full results
    print("\nSummary Statistics:")
    for agent in results:
        print(f"\n{agent}:")
        for noise_level in results[agent]:
            nodes = results[agent][noise_level]['nodes']
            purity = results[agent][noise_level]['purity']
            se = results[agent][noise_level]['se']
            print(f"Noise Level σ² = {noise_level}:")
            print(f"  Mean nodes: {np.mean(nodes):.2f}")
            print(f"  Std nodes: {np.std(nodes):.2f}")
            print(f"  Mean purity: {np.mean(purity):.2f}%")
            print(f"  Std purity: {np.std(purity):.2f}%")
            print(f"  Mean SE: {np.mean(se):.2f}")
            print(f"  Std SE: {np.std(se):.2f}")
    
    # Option 2: Analyze pattern mappings from the saved data
    print("\n4. Analyzing pattern mappings...")
    analyze_saved_pattern_mappings("minerva_pattern_mappings.pkl")
    
    print("\nAnalysis complete. Check the 'minerva_binary_analysis' directory for detailed results.")