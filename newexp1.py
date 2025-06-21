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

def collect_maze_positions(maze_map, player_pos_index, goal_pos_index, 
                          exploration_steps=15000, save_path="maze_positions.csv"):
    """
    Perform pre-exploration to collect positional data from the maze environment.
    
    Parameters:
    - maze_map: The maze map to explore
    - player_pos_index: Initial player position index
    - goal_pos_index: Goal position index
    - exploration_steps: Number of steps to explore (default: 5000)
    - save_path: Path to save the collected positions (default: "maze_positions.csv")
    
    Returns:
    - Path to the saved CSV file
    """
    import numpy as np
    import pandas as pd
    import os
    from Maze.Maze_player import MazePlayer
    
    print(f"Starting pre-exploration to collect positional data...")
    
    # Check if the file already exists for this maze configuration
    if os.path.exists(save_path):
        print(f"Position data already exists at {save_path}, using existing data")
        return save_path
    
    # Initialize maze
    Maze = MazePlayer(maze_map=maze_map, 
                     player_index_pos=player_pos_index, 
                     goal_index_pos=goal_pos_index)
    
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
    
    return save_path

def load_and_prepare_training_data(csv_path):
    """
    Load position data from CSV and prepare for HGWRSOM training.
    
    Parameters:
    - csv_path: Path to the CSV file containing position data
    
    Returns:
    - training_data: NumPy array ready for training
    """
    import pandas as pd
    import numpy as np
    
    # Load data from CSV
    df = pd.read_csv(csv_path)
    
    # Convert to NumPy array
    training_data = df.values
    
    print(f"Loaded {len(training_data)} training samples from {csv_path}")
    print(f"Data range: X[{training_data[:, 0].min():.2f}, {training_data[:, 0].max():.2f}], "
          f"Y[{training_data[:, 1].min():.2f}, {training_data[:, 1].max():.2f}]")
    
    return training_data

def get_network_stats(agent, agent_name):
    """
    Get network statistics for both TMGWR and MINERVA agents
    Uses the proper counting method from newexp.py for MINERVA
    """
    if agent_name == 'TMGWR':
        return {
            'lower_x_nodes': 0,  # TMGWR doesn't have separate lower networks
            'lower_y_nodes': 0,
            'higher_nodes': len(agent.model.W),
            'higher_connections': int(np.sum(agent.model.C)) if hasattr(agent.model, 'C') else 0
        }
    else:  # MINERVA
        return {
            'lower_x_nodes': len(agent.lower_x.A) if hasattr(agent.lower_x, 'A') else 0,
            'lower_y_nodes': len(agent.lower_y.A) if hasattr(agent.lower_y, 'A') else 0,
            'higher_nodes': len(agent.nodes),  # This is the key fix from newexp.py
            'higher_connections': int(np.sum(agent.connections)) if hasattr(agent, 'connections') else 0
        }

# Modified run_enhanced_noise_comparison function with proper node counting
def run_enhanced_noise_comparison_with_maze_data(noise_levels=[0, 1/6, 1/3, 1/2, 2/3, 5/6, 1, 7/6, 4/3], 
                                               episodes_per_noise=2,
                                               maze_positions_path=None):
    """
    Enhanced version of run_noise_comparison that uses actual maze position data for training
    and proper node counting from newexp.py.
    
    Parameters:
    - noise_levels: List of noise levels to test
    - episodes_per_noise: Number of episodes to run per noise level
    - maze_positions_path: Path to pre-collected maze positions CSV. If None, pre-exploration will be performed.
    
    Returns:
    - results: Dictionary containing metrics for each agent and noise level
    - table_data: Data for node creation visualization
    - pattern_mappings: Information about state to pattern mappings
    """
        
    results = {
        'TMGWR': {level: {'nodes': [], 'purity': [], 'se': [], 'lower_x_nodes': [], 'lower_y_nodes': [], 'higher_nodes': [], 'higher_connections': []} for level in noise_levels},
        'MINERVA': {level: {'nodes': [], 'purity': [], 'se': [], 'lower_x_nodes': [], 'lower_y_nodes': [], 'higher_nodes': [], 'higher_connections': []} for level in noise_levels}
    }
    
    # Enhanced to store pattern mappings
    pattern_mappings = {level: [] for level in noise_levels}
    
    # Initialize table data storage
    table_data = []

    # Get maze details
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()
    
    # Perform pre-exploration if needed
    if maze_positions_path is None or not os.path.exists(maze_positions_path):
        maze_positions_path = collect_maze_positions(
            maze_map, player_pos_index, goal_pos_index,
            exploration_steps=6000, save_path="maze_positions.csv"
        )
    
    # Load training data from pre-exploration
    training_data = load_and_prepare_training_data(maze_positions_path)
    
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

            # Initialize HGWRSOM agent with maze data
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
                        # Train lower networks with actual maze position data
                        print(f"Training MINERVA with {len(training_data)} maze positions...")
                        
                        # Train lower networks using the method from newexp.py
                        agent.train_lower_networks(training_data, epochs=20)  # Reduced epochs for efficiency
                        
                        # Ensure higher-level network is properly initialized
                        # This is crucial for proper node counting
                        if not hasattr(agent, 'nodes'):
                            agent.nodes = []
                        if not hasattr(agent, 'connections'):
                            agent.connections = np.zeros((0, 0))
                        if not hasattr(agent, 'pattern_ages'):
                            agent.pattern_ages = np.zeros((0, 0))
                    
                    agent.set_goal(goal)
                    agent.set_epsilon(1)  # Pure exploration
                    
                    # Run episode
                    Maze.reset_player()  # Reset for each agent
                    current_state = initial_state
                    step_counter = 0
                    max_steps = 2000  # Limit to prevent infinite loops
                    
                    state_node_mappings = defaultdict(lambda: defaultdict(int))
                    transitions = []
                    total_visits = 0
                    
                    # Main episode loop
                    while current_state != goal and step_counter < max_steps:
                        step_counter += 1
                        
                        # Print progress occasionally with proper node counting
                        if step_counter % 1000 == 0:
                            network_stats = get_network_stats(agent, agent_name)
                            print(f"{agent_name}: Step {step_counter}, Higher Nodes: {network_stats['higher_nodes']}")
                        
                        prev_state = np.array(current_state)
                        
                        # Add noise to state observation if noise_level > 0
                        noise_probability = 0.5 # Probability of adding noise
                        if np.random.uniform(0, 1) < noise_probability:  # Add noise with a certain probability
                            noisy_state = np.array(current_state) + np.random.normal(0, np.sqrt(noise_level), 2)
                        else:  # Use the true state
                            noisy_state = np.array(current_state)
                        
                        # Get node assignment for current state
                        if agent_name == 'TMGWR':
                            node_idx = agent.model.get_node_index(noisy_state)
                        else:
                            # For MINERVA, use its native methods with proper node tracking
                            pattern = agent.get_firing_pattern(noisy_state)
                            
                            # Record the mapping of state to pattern
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
                                # Pattern doesn't exist yet, will be assigned next node index
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
                    
                    # # Check if episode terminated normally
                    # if current_state == goal:
                    #     print(f"{agent_name} reached goal in {step_counter} steps")
                    # else:
                    #     print(f"{agent_name} did not reach goal, stopped after {step_counter} steps")
                    
                    # Calculate metrics
                    if total_visits > 0:
                        purity = calculate_purity(state_node_mappings, total_visits)
                    else:
                        purity = 0.0  # Default if no visits
                        
                    if len(transitions) > 0:
                        se = calculate_se(transitions, agent)
                    else:
                        se = 0.0  # Default if no transitions
                    
                    # Get proper network statistics using the method from newexp.py
                    network_stats = get_network_stats(agent, agent_name)
                    
                    # For backward compatibility, also record total nodes
                    total_nodes = network_stats['higher_nodes']
                    
                    # # Print detailed debug info
                    # print(f"{agent_name} episode summary:")
                    # print(f"  Lower X Nodes: {network_stats['lower_x_nodes']}")
                    # print(f"  Lower Y Nodes: {network_stats['lower_y_nodes']}")
                    # print(f"  Higher Nodes: {network_stats['higher_nodes']}")
                    # print(f"  Higher Connections: {network_stats['higher_connections']}")
                    # print(f"  Purity: {purity:.2f}%")
                    # print(f"  SE: {se:.4f}")
                    # print(f"  Total states visited: {total_visits}")
                    # print(f"  Transitions recorded: {len(transitions)}")
                    
                    # Store results with detailed network statistics
                    results[agent_name][noise_level]['nodes'].append(total_nodes)
                    results[agent_name][noise_level]['purity'].append(purity)
                    results[agent_name][noise_level]['se'].append(se)
                    results[agent_name][noise_level]['lower_x_nodes'].append(network_stats['lower_x_nodes'])
                    results[agent_name][noise_level]['lower_y_nodes'].append(network_stats['lower_y_nodes'])
                    results[agent_name][noise_level]['higher_nodes'].append(network_stats['higher_nodes'])
                    results[agent_name][noise_level]['higher_connections'].append(network_stats['higher_connections'])
                
                except Exception as e:
                    # More detailed error reporting
                    print(f"Error in {agent_name} episode {episode} with noise level {noise_level}: {str(e)}")
                    import traceback
                    traceback.print_exc()  # Print full stack trace for debugging
                    
                    # Get network stats for error reporting
                    try:
                        network_stats = get_network_stats(agent, agent_name)
                        print(f"Agent state: {agent_name} has {network_stats['higher_nodes']} higher nodes")
                    except:
                        print(f"Could not get network stats for {agent_name}")
                    
                    # Record zeros for failed episodes, but clearly mark them
                    results[agent_name][noise_level]['nodes'].append(-1)  # Use -1 to indicate failure
                    results[agent_name][noise_level]['purity'].append(0)
                    results[agent_name][noise_level]['se'].append(0)
                    results[agent_name][noise_level]['lower_x_nodes'].append(0)
                    results[agent_name][noise_level]['lower_y_nodes'].append(0)
                    results[agent_name][noise_level]['higher_nodes'].append(-1)
                    results[agent_name][noise_level]['higher_connections'].append(0)
            
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
                # Keep only valid episodes for all metrics
                for metric in results[agent_name][noise_level]:
                    results[agent_name][noise_level][metric] = [results[agent_name][noise_level][metric][i] for i in valid_indices]
            else:
                # If all episodes failed, keep one zero entry for reporting
                for metric in results[agent_name][noise_level]:
                    if metric in ['nodes', 'higher_nodes']:
                        results[agent_name][noise_level][metric] = [0]
                    else:
                        results[agent_name][noise_level][metric] = [0]

    return results, table_data, pattern_mappings

def plot_comparison_results(results):
    """Plot comparison results with enhanced network statistics and colored boxplot elements."""
    metrics = {
        'nodes': {'title': 'Total Nodes vs Noise Level', 'ylabel': 'Total Nodes', 'ylim': (38, 41)},
        'purity': {'title': 'Purity vs Noise Level', 'ylabel': 'Purity (%)', 'ylim': (99, 100.2)},
        'se': {'title': 'Sensorimotor Error vs Noise Level', 'ylabel': 'SE', 'ylim': (0, 0.08)}
    }
    
    # Manual mapping for our specific noise levels
    fraction_mapping = {
        0: "0",
        1/6: "1/6",
        1/3: "1/3",
        1/2: "1/2",
        2/3: "2/3",           
        5/6: "5/6",
        1: "1",
        7/6: "7/6",
        4/3: "4/3"
    }
    
    for metric in metrics:
        if metric not in results['TMGWR'][list(results['TMGWR'].keys())[0]]:
            continue  # Skip metrics that don't exist
            
        data = []
        for agent in results:
            for noise_level in results[agent]:
                values = results[agent][noise_level][metric]
                for value in values:
                    data.append({
                        'Agent': agent,
                        'Noise Level': fraction_mapping[noise_level],
                        'Noise_sort': noise_level,
                        'Value': value
                    })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Noise_sort')
        
        plt.figure(figsize=(15, 8))
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        # Custom colors
        colors = {'TMGWR': 'green', 'MINERVA': 'orange'}
        
        # Get unique noise levels and agents
        noise_levels = sorted(df['Noise_sort'].unique())
        agents = df['Agent'].unique()
        
        # Prepare data for boxplot
        all_data = []
        labels = []
        box_colors = []
        
        for i, noise_level in enumerate(noise_levels):
            noise_label = fraction_mapping[noise_level]
            for j, agent in enumerate(agents):
                agent_data = df[(df['Agent'] == agent) & (df['Noise_sort'] == noise_level)]['Value'].values
                if len(agent_data) > 0:
                    all_data.append(agent_data)
                    labels.append(f"{noise_label}")
                    box_colors.append(colors[agent])
        
        # Create positions for the boxes with reduced spacing between both algorithms and noise levels
        n_noise = len(noise_levels)
        n_agents = len(agents)
        positions = []
        group_spacing = 2.8  # Reduced spacing between noise level groups
        for i in range(n_noise):
            for j in range(n_agents):
                positions.append(i * group_spacing + j * 0.8)  # 0.8 spacing between algorithms
        
        # Create the boxplot with matplotlib for full color control
        bp = plt.boxplot(all_data, positions=positions, patch_artist=True, 
                        widths=0.6, zorder=1)  # Slightly wider boxes to compensate
        
        # Color all elements consistently
        for i, (patch, color) in enumerate(zip(bp['boxes'], box_colors)):
            # Box face and edge
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            
            # Whiskers
            bp['whiskers'][i*2].set_color(color)
            bp['whiskers'][i*2+1].set_color(color)
            
            # Caps
            bp['caps'][i*2].set_color(color)
            bp['caps'][i*2+1].set_color(color)
            
            # Median
            bp['medians'][i].set_color(color)
            
            # Outliers (fliers)
            bp['fliers'][i].set_markerfacecolor(color)
            bp['fliers'][i].set_markeredgecolor(color)
            bp['fliers'][i].set_alpha(0.7)
        
        # Set x-axis labels with reduced spacing
        noise_labels = [fraction_mapping[level] for level in noise_levels]
        plt.xticks([i * (n_agents + 0.2) + (n_agents - 1) / 2 for i in range(n_noise)], 
                  noise_labels, rotation=45)
        
        # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[agent], label=agent) 
                          for agent in agents]
        plt.legend(handles=legend_elements, title='Agent Type', title_fontsize=12, 
                  fontsize=10, loc='upper right')
        
        plt.title(metrics[metric]['title'], fontsize=14, pad=20)
        plt.xlabel('Noise Level (σ²)', fontsize=12)
        plt.ylabel(metrics[metric]['ylabel'], fontsize=12)
        
        if metrics[metric]['ylim']:
            plt.ylim(metrics[metric]['ylim'])
        
        plt.tight_layout()
        plt.show()

def print_detailed_summary(results):
    """Print detailed summary with network statistics"""
    print("\nDetailed Summary Statistics:")
    for agent in results:
        print(f"\n{agent}:")
        for noise_level in results[agent]:
            print(f"  Noise Level σ² = {noise_level}:")
            
            # Print statistics for each metric
            for metric in ['nodes', 'higher_nodes', 'lower_x_nodes', 'lower_y_nodes', 'higher_connections', 'purity', 'se']:
                if metric in results[agent][noise_level]:
                    values = results[agent][noise_level][metric]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        if metric in ['purity']:
                            print(f"    Mean {metric}: {mean_val:.2f}% (±{std_val:.2f}%)")
                        elif metric in ['se']:
                            print(f"    Mean {metric}: {mean_val:.4f} (±{std_val:.4f})")
                        else:
                            print(f"    Mean {metric}: {mean_val:.2f} (±{std_val:.2f})")

if __name__ == "__main__":
    print("MINERVA Binary Pattern Analysis - Corrected Version")
    print("=" * 50)
    
    # Define maze positions file path
    maze_positions_path = "maze_positions.csv"
       
    # Now run the enhanced comparison using maze positions
    print("\nRunning enhanced noise comparison with proper node counting...")
    
    # Get maze details for pre-exploration
    from Maze.Mazes import MazeMaps
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()
    
    # First, collect maze positions through pre-exploration (will be skipped if file exists)
    collect_maze_positions(
        maze_map, player_pos_index, goal_pos_index,
        exploration_steps=10000, save_path=maze_positions_path
    )
    
    # Run the enhanced comparison with collected maze positions
    results, table_data, pattern_mappings = run_enhanced_noise_comparison_with_maze_data(
        noise_levels=[0, 1/3, 2/3, 1, 4/3],
        episodes_per_noise=2,
        maze_positions_path=maze_positions_path
    )
    
    # Plot the results
    plot_comparison_results(results)
    
    # Print detailed summary with network statistics
    print_detailed_summary(results)
    
    # Save enhanced results
    with open("enhanced_results_with_network_stats.pkl", "wb") as f:
        pickle.dump({
            'results': results,
            'pattern_mappings': pattern_mappings
        }, f)
    
    print("\nEnhanced results with proper node counting saved!")