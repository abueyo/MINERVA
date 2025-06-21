import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import pickle
import os
import time
from mazes import MazeMaps
from mazePlayer import MazePlayer
from TMGWR_agent import TMGWRAgent
from HSOM_multisensory import MultisensoryHGWRSOMAgent

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
            
            # Check if it's a MultisensoryHGWRSOMAgent
            if hasattr(agent, 'sensory_dimensions'):  # This is a multisensory agent
                # Convert position array to multisensory dictionary
                prev_sensory = {'position': prev_state}
                curr_sensory = {'position': curr_state}
                
                # For multisensory agent, we need to create dummy sensory input
                # since we only recorded position transitions
                if 'beacon_distances' in agent.sensory_dimensions:
                    # Create dummy beacon distances (zeros)
                    beacon_dim = agent.sensory_dimensions['beacon_distances']
                    prev_sensory['beacon_distances'] = np.zeros(beacon_dim)
                    curr_sensory['beacon_distances'] = np.zeros(beacon_dim)
                
                if 'temperature' in agent.sensory_dimensions:
                    # Create dummy temperature (zero)
                    prev_sensory['temperature'] = 0.0
                    curr_sensory['temperature'] = 0.0
                
                prev_pattern = agent.get_firing_pattern(prev_sensory)
                curr_pattern = agent.get_firing_pattern(curr_sensory)
            else:
                # Original MINERVA agent (position only)
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
def collect_maze_multisensory_data(maze_map, player_pos_index, goal_pos_index, beacon_positions=None,
                          exploration_steps=10000, save_path="maze_multisensory_data.pkl"):
    """
    Perform pre-exploration to collect multisensory data from the maze environment.
    
    Parameters:
    - maze_map: The maze map to explore
    - player_pos_index: Initial player position index
    - goal_pos_index: Goal position index
    - beacon_positions: List of beacon positions (index format)
    - exploration_steps: Number of steps to explore (default: 10000)
    - save_path: Path to save the collected data (default: "maze_multisensory_data.pkl")
    
    Returns:
    - Path to the saved data file
    """
    import numpy as np
    import pandas as pd
    import os
    import pickle
    from mazePlayer import MazePlayer
    
    print(f"Starting pre-exploration to collect multisensory data...")
    
    # Check if the file already exists for this maze configuration
    if os.path.exists(save_path):
        print(f"Multisensory data already exists at {save_path}, using existing data")
        with open(save_path, 'rb') as f:
            multisensory_data = pickle.load(f)
        return save_path, multisensory_data
    
    # Initialize maze with beacons
    if beacon_positions is None:
        # Default beacon positions at corners if none provided
        beacon_positions = []
        # Place beacons at corners of maze
        beacon_positions.append((1, len(maze_map[0])-2))  # Top-right
        beacon_positions.append((len(maze_map)-2, 1))      # Bottom-left
    
    # Calculate screen coordinates for beacons
    Maze = MazePlayer(maze_map=maze_map, 
                     player_index_pos=player_pos_index, 
                     goal_index_pos=goal_pos_index)
    
    # Convert beacon index positions to screen coordinates
    beacon_screen_positions = [Maze._calc_screen_coordinates(*pos) for pos in beacon_positions]
    
    # Reinitialize maze with beacon positions
    Maze = MazePlayer(maze_map=maze_map, 
                     player_index_pos=player_pos_index, 
                     goal_index_pos=goal_pos_index,
                     beacon_positions=beacon_screen_positions)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Storage for multisensory data
    positions = []
    beacon_distances = []
    temperatures = []
    
    # Reset player to initial position
    Maze.reset_player()
    current_state = Maze.get_player_pos()
    
    # Perform random exploration to collect data
    for step in range(exploration_steps):
        # Store current position
        positions.append(current_state)
        
        # Store beacon distances
        beacon_dists = Maze.get_beacon_distances(current_state)
        beacon_distances.append(beacon_dists)
        
        # Store temperature
        temp = Maze.get_temperature_reading(current_state)
        temperatures.append([temp])
        
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
    
    # Convert to numpy arrays
    positions = np.array(positions)
    beacon_distances = np.array(beacon_distances)
    temperatures = np.array(temperatures)
    
    # Organize collected data
    multisensory_data = {
        'position': positions,
        'beacon_distances': beacon_distances,
        'temperature': temperatures
    }
    
    # Save to pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(multisensory_data, f)
    
    print(f"Pre-exploration complete.")
    print(f"Collected {len(positions)} samples with dimensions:")
    print(f"  Positions: {positions.shape}")
    print(f"  Beacon distances: {beacon_distances.shape}")
    print(f"  Temperatures: {temperatures.shape}")
    print(f"Data saved to {save_path}")
    
    return save_path, multisensory_data

def plot_multisensory_comparison_results(results):
    """Plot comparison results for multisensory experiment"""
    metrics = {
        'nodes': {'title': 'Number of Nodes vs Noise Level', 'ylabel': 'Number of Nodes', 'ylim': (0, 120)},
        'purity': {'title': 'Purity vs Noise Level', 'ylabel': 'Purity (%)', 'ylim': (70, 102)},
        'se': {'title': 'Sensorimotor Error vs Noise Level', 'ylabel': 'SE', 'ylim': (0, 0.1)}
    }
    
    # Manual mapping for noise levels
    fraction_mapping = {
        0: "0",
        1/6: "1/6",
        1/3: "1/3",
        1/2: "1/2",
        2/3: "2/3",           
        5/6: "5/6",
        1: "1"
    }
    
    # Custom color palette for three agent types
    custom_palette = {
        'TMGWR': 'green', 
        'MINERVA-Classic': 'orange',
        'MINERVA-Multisensory': 'blue'
    }
    
    for metric in metrics:
        data = []
        for agent in results:
            for noise_level in results[agent]:
                values = results[agent][noise_level][metric]
                for value in values:
                    data.append({
                        'Agent': agent,
                        'Noise Level': fraction_mapping.get(noise_level, str(noise_level)),
                        'Value': value,
                        'Metric': metrics[metric]['ylabel'],
                        'Noise_sort': noise_level
                    })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Noise_sort')
        
        plt.figure(figsize=(15, 8))
        
        # Add grid before the plot
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Use the custom palette for three agent types
        sns.boxplot(data=df, x='Noise Level', y='Value', hue='Agent', palette=custom_palette)
        
        # Ensure grid is behind the plot
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        plt.title(metrics[metric]['title'], fontsize=14, pad=20)
        plt.xlabel('Noise Level (σ²)', fontsize=12)
        plt.ylabel(metrics[metric]['ylabel'], fontsize=12)
        plt.legend(title='Agent Type', title_fontsize=12, fontsize=10)
        
        if metrics[metric]['ylim']:
            plt.ylim(metrics[metric]['ylim'])
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"multisensory_comparison_{metric}.png")
        plt.show()

def run_multisensory_comparison(noise_levels=[0, 1/6, 1/3, 1/2, 2/3, 5/6, 1], 
                              episodes_per_noise=5,
                              multisensory_data_path=None):
    """
    Run comparison between classic and multisensory MINERVA architectures
    
    Parameters:
    -----------
    noise_levels : list
        Noise levels to test
    episodes_per_noise : int
        Number of episodes to run per noise level
    multisensory_data_path : str
        Path to pre-collected multisensory data
    """
    results = {
        'TMGWR': {level: {'nodes': [], 'purity': [], 'se': []} for level in noise_levels},
        'MINERVA-Classic': {level: {'nodes': [], 'purity': [], 'se': []} for level in noise_levels},
        'MINERVA-Multisensory': {level: {'nodes': [], 'purity': [], 'se': []} for level in noise_levels}
    }
    
    # Get maze details
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()
    
    # Setup beacon positions
    beacon_positions = []
    beacon_positions.append((1, len(maze_map[0])-2))  # Top-right
    beacon_positions.append((len(maze_map)-2, 1))      # Bottom-left
    
    # Calculate screen coordinates
    temp_maze = MazePlayer(maze_map=maze_map, 
                         player_index_pos=player_pos_index, 
                         goal_index_pos=goal_pos_index,
                         display_maze=False)
    
    beacon_screen_positions = [temp_maze._calc_screen_coordinates(*pos) for pos in beacon_positions]
    
    # Collect multisensory data if needed
    if multisensory_data_path is None or not os.path.exists(multisensory_data_path):
        multisensory_data_path, multisensory_data = collect_maze_multisensory_data(
            maze_map, player_pos_index, goal_pos_index,
            beacon_positions=beacon_positions,
            exploration_steps=5000,
            save_path="maze_multisensory_data.pkl"
        )
    else:
        # Load existing data
        with open(multisensory_data_path, 'rb') as f:
            multisensory_data = pickle.load(f)
            
    # Set consistent random seed for reproducibility
    np.random.seed(42)
    
    for noise_level in noise_levels:
        print(f"\nRunning simulations with noise level σ² = {noise_level}")
        
        for episode in range(episodes_per_noise):
            print(f"\nEpisode {episode + 1}/{episodes_per_noise}")
            
            # Set episode-specific seed for controlled variation
            episode_seed = 42 + episode
            np.random.seed(episode_seed)
            
            # Initialize maze with beacon positions for this episode
            Maze = MazePlayer(maze_map=maze_map, 
                             player_index_pos=player_pos_index, 
                             goal_index_pos=goal_pos_index,
                             beacon_positions=beacon_screen_positions)
            
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

            # Initialize classic HGWRSOM agent (x-y position only)
            from HSOM_binary import HierarchicalGWRSOMAgent
            classic_agent = HierarchicalGWRSOMAgent(
                lower_dim=1, higher_dim=2, epsilon_b=0.35,
                epsilon_n=0.15, beta=0.7, delta=0.79,
                T_max=20, N_max=100, eta=0.5,
                phi=0.9, sigma=0.5
            )
            classic_agent.train_lower_networks(multisensory_data['position'], epochs=50)
            classic_agent.set_goal(goal)
            classic_agent.set_epsilon(1)  # Pure exploration
            
            # Initialize multisensory HGWRSOM agent
            multisensory_agent = MultisensoryHGWRSOMAgent(
                sensory_dimensions={'position': 2, 'beacon_distances': 2, 'temperature': 1},
                epsilon_b=0.35, epsilon_n=0.15, beta=0.7, delta=0.79,
                T_max=20, N_max=100, eta=0.5, phi=0.9, sigma=0.5
            )
            # Train the multisensory agent with all collected data
            multisensory_agent.train_lower_networks(multisensory_data, epochs=50)
            multisensory_agent.set_goal(goal)
            multisensory_agent.set_epsilon(1)  # Pure exploration
            
            # Run each agent in sequence for fair comparison
            for agent_idx, (agent, agent_name) in enumerate([
                (tmgwr_agent, 'TMGWR'), 
                (classic_agent, 'MINERVA-Classic'),
                (multisensory_agent, 'MINERVA-Multisensory')
            ]):
                try:
                    # Reset random seed for consistent noise patterns between agents
                    np.random.seed(episode_seed + agent_idx)
                    
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
                            print(f"{agent_name}: Step {step_counter}, Nodes: {len(agent.nodes) if hasattr(agent, 'nodes') else len(agent.model.W)}")
                        
                        prev_state = np.array(current_state)
                        
                        # Add noise to current state observation
                        noisy_position = np.array(current_state) + np.random.normal(0, np.sqrt(noise_level), 2)
                        
                        # Get multisensory input for the current state
                        multisensory_input = Maze.get_multisensory_input(current_state)
                        
                        # Add noise to all sensory inputs
                        noisy_multisensory = {
                            'position': noisy_position,
                            'beacon_distances': multisensory_input['beacon_distances'] + np.random.normal(0, np.sqrt(noise_level), len(multisensory_input['beacon_distances'])),
                            'temperature': multisensory_input['temperature'] + np.random.normal(0, np.sqrt(noise_level), 1)[0]
                        }
                        
                        # Get node assignment based on agent type
                        if agent_name == 'TMGWR':
                            # TMGWR only uses position
                            node_idx = agent.model.get_node_index(noisy_position)
                            action = agent.select_action(noisy_position)
                        elif agent_name == 'MINERVA-Classic':
                            # Classic MINERVA uses only position
                            pattern = agent.get_firing_pattern(noisy_position)
                            found_idx = agent.find_node_index(pattern)
                            node_idx = found_idx if found_idx is not None else len(agent.nodes)
                            action = agent.select_action(noisy_position)
                        else:  # MINERVA-Multisensory
                            # Multisensory MINERVA uses all sensory inputs
                            pattern = agent.get_firing_pattern(noisy_multisensory)
                            found_idx = agent.find_node_index(pattern)
                            node_idx = found_idx if found_idx is not None else len(agent.nodes)
                            action = agent.select_action(noisy_multisensory)
                        
                        # Update state-node mapping counts
                        state_tuple = tuple(current_state)
                        state_node_mappings[node_idx][state_tuple] += 1
                        total_visits += 1
                        
                        # Execute action
                        Maze.move_player(action)
                        next_state = Maze.get_player_pos()
                        
                        # Ensure next_state is not None or invalid
                        if next_state is None:
                            print(f"Warning: Invalid next_state after action {action}")
                            break
                            
                        # Update model based on agent type
                        if agent_name == 'TMGWR':
                            agent.update_model(next_state, action)
                        elif agent_name == 'MINERVA-Classic':
                            agent.update_model(next_state, action)
                        else:  # MINERVA-Multisensory
                            next_multisensory = Maze.get_multisensory_input(next_state)
                            agent.update_model(next_multisensory, action)
                        
                        # Record transition
                        transitions.append((prev_state, next_state))
                        current_state = next_state
                    
                    # Check if episode terminated normally
                    if current_state == goal:
                        print(f"{agent_name} reached goal in {step_counter} steps")
                    else:
                        print(f"{agent_name} did not reach goal, stopped after {step_counter} steps")
                    
                    # Calculate metrics
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
                    
                except Exception as e:
                    # More detailed error reporting
                    print(f"Error in {agent_name} episode {episode} with noise level {noise_level}: {str(e)}")
                    import traceback
                    traceback.print_exc()  # Print full stack trace for debugging
                    
                    # Record zeros for failed episodes, but clearly mark them
                    results[agent_name][noise_level]['nodes'].append(-1)  # Use -1 to indicate failure
                    results[agent_name][noise_level]['purity'].append(0)
                    results[agent_name][noise_level]['se'].append(0)
    
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
                
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("MINERVA Multisensory Analysis")
    print("=" * 60)
    
    # Define multisensory data file path
    multisensory_data_path = "maze_multisensory_data.pkl"
    
    # Get maze details
    print("\nGetting maze configuration...")
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()
    
    # For a quick test, use fewer noise levels and episodes
    # For full analysis, consider using all noise levels
    simplified_run = True
    
    if simplified_run:
        print("\nRunning simplified comparison (fewer noise levels and episodes)...")
        noise_levels = [0, 1/3, 2/3]  # Just 3 noise levels
        episodes = 2  # Only 2 episodes per noise level
    else:
        print("\nRunning full comparison (may take a long time)...")
        noise_levels = [0, 1/6, 1/3, 1/2, 2/3, 5/6, 1]  # All noise levels
        episodes = 5  # 5 episodes per noise level
    
    # Run the comparison with multisensory data
    results = run_multisensory_comparison(
        noise_levels=noise_levels,
        episodes_per_noise=episodes,
        multisensory_data_path=multisensory_data_path
    )
    
    # Plot the results
    plot_multisensory_comparison_results(results)
    
    # Save results
    with open("multisensory_comparison_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Display summary statistics
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
            print(f"  Mean SE: {np.mean(se):.4f}")
            print(f"  Std SE: {np.std(se):.4f}")