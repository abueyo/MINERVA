import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.TMGWR_agent import TMGWRAgent
from Agents.HSOM_binary import HierarchicalGWRSOMAgent
import time
import os
import random

def run_pattern_completion_experiment(mask_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], runs_per_level=5):
    """
    Pattern Completion Experiment: Test how MINERVA and TMGWR handle partial inputs
    
    Args:
        mask_levels: Percentage of state information to mask (0.0 = no masking, 0.5 = 50% masked)
        runs_per_level: Number of runs for each masking level
    
    Returns:
        results: Dictionary containing recognition accuracy for both algorithms
    """
    # Initialize results storage
    results = {
        'TMGWR': {level: {'recognition_acc': [], 'recovery_error': []} for level in mask_levels},
        'MINERVA': {level: {'recognition_acc': [], 'recovery_error': []} for level in mask_levels}
    }

    # Get maze details
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()

    # Generate training data for pre-training
    x_train = np.linspace(-72, 72, 10).reshape(-1, 1)
    y_train = np.linspace(-72, 72, 10).reshape(-1, 1)
    training_data = np.hstack((x_train, y_train))

    # First, train both agents on the maze to build a full model
    for mask_level in mask_levels:
        print(f"\nRunning pattern completion experiment with mask level = {mask_level}")
        
        for run in range(runs_per_level):
            print(f"\nRun {run + 1}/{runs_per_level}")
            
            # Initialize maze environment
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
            tmgwr_agent.set_epsilon(1)

            # Initialize MINERVA agent
            minerva_agent = HierarchicalGWRSOMAgent(
                lower_dim=1, higher_dim=2, epsilon_b=0.35,
                epsilon_n=0.15, beta=0.7, delta=0.79,
                T_max=20, N_max=100, eta=0.5,
                phi=0.9, sigma=0.5
            )
            
            # Initialize nodes list and connections matrix
            minerva_agent.nodes = []
            minerva_agent.connections = np.zeros((0, 0))
            
            # Train lower networks
            minerva_agent.train_lower_networks(training_data, epochs=5)
            minerva_agent.set_goal(goal)
            minerva_agent.set_epsilon(0)  # No exploration during testing

            # Train both agents on a fixed exploration path
            train_steps = 1000
            current_state = initial_state
            Maze.reset_player()
            
            # Store all visited states for testing
            visited_states = []
            
            print("Training agents on maze environment...")
            for step in range(train_steps):
                # Random exploration
                action = random.randint(0, 3)
                Maze.move_player(action)
                next_state = Maze.get_player_pos()
                
                # Store state as numpy array (not tuple) for easier manipulation
                # Ensure it's a float array to allow NaN values later
                next_state_array = np.array(next_state, dtype=float)
                
                # Train both agents
                tmgwr_agent.update_model(next_state, action)
                minerva_agent.update_model(next_state, action)
                
                # Store for testing (as numpy array)
                visited_states.append(next_state_array)
                current_state = next_state
                
                # Reset if goal reached
                if current_state == goal:
                    Maze.reset_player()
                    current_state = Maze.get_player_pos()
            
            print(f"Training complete. TMGWR nodes: {len(tmgwr_agent.model.W)}, MINERVA nodes: {len(minerva_agent.nodes)}")
            
            # Now test pattern completion with masked inputs
            tmgwr_recognition = 0
            minerva_recognition = 0
            tmgwr_recovery_error = 0
            minerva_recovery_error = 0
            
            # Select 100 random states from visited states for testing
            test_states = random.sample(visited_states, min(100, len(visited_states)))
            
            print(f"Testing pattern completion with mask level {mask_level}...")
            for state in test_states:
                # Create masked version of the state (state is now a numpy array)
                masked_state = state.copy()
                original_state = state.copy()  # Keep an unmasked copy for comparison
                
                # Apply masking (choose which dimension to mask)
                masked_dim = None
                if mask_level > 0:
                    if random.random() < 0.5:
                        # Mask X dimension
                        masked_dim = 0
                        masked_state[0] = np.nan
                    else:
                        # Mask Y dimension
                        masked_dim = 1
                        masked_state[1] = np.nan
                
                # Skip masking if mask_level is 0
                if masked_dim is None:
                    continue
                
                # Test TMGWR
                try:
                    # Get original state node index
                    original_node_idx = tmgwr_agent.model.get_node_index(original_state)
                    
                    # Try to recover the complete state from the masked state
                    recovered_state = recover_state_tmgwr(tmgwr_agent, masked_state, masked_dim)
                    
                    # Get recovered state node index
                    recovered_node_idx = tmgwr_agent.model.get_node_index(recovered_state)
                    
                    # Check if recognition is successful
                    if original_node_idx == recovered_node_idx:
                        tmgwr_recognition += 1
                    
                    # Calculate recovery error (Euclidean distance)
                    tmgwr_recovery_error += np.linalg.norm(original_state - recovered_state)
                    
                except Exception as e:
                    print(f"TMGWR error: {e}")
                    continue
                
                # Test MINERVA
                try:
                    # Get original pattern
                    original_pattern = minerva_agent.get_firing_pattern(original_state)
                    original_node_idx = minerva_agent.find_node_index(original_pattern)
                    
                    # Try to recover the complete state
                    recovered_state = recover_state_minerva(minerva_agent, masked_state, masked_dim)
                    
                    # Get recovered pattern
                    recovered_pattern = minerva_agent.get_firing_pattern(recovered_state)
                    recovered_node_idx = minerva_agent.find_node_index(recovered_pattern)
                    
                    # Check if recognition is successful
                    if original_node_idx == recovered_node_idx:
                        minerva_recognition += 1
                    
                    # Calculate recovery error (Euclidean distance)
                    minerva_recovery_error += np.linalg.norm(original_state - recovered_state)
                    
                except Exception as e:
                    print(f"MINERVA error: {e}")
                    continue
            
            # Calculate accuracy and average recovery error
            total_tests = len(test_states) if mask_level == 0 else len(test_states)
            if total_tests == 0:
                total_tests = 1  # Avoid division by zero
                
            tmgwr_accuracy = tmgwr_recognition / total_tests * 100
            minerva_accuracy = minerva_recognition / total_tests * 100
            
            tmgwr_avg_error = tmgwr_recovery_error / total_tests
            minerva_avg_error = minerva_recovery_error / total_tests
            
            # Store results
            results['TMGWR'][mask_level]['recognition_acc'].append(tmgwr_accuracy)
            results['MINERVA'][mask_level]['recognition_acc'].append(minerva_accuracy)
            results['TMGWR'][mask_level]['recovery_error'].append(tmgwr_avg_error)
            results['MINERVA'][mask_level]['recovery_error'].append(minerva_avg_error)
            
            print(f"TMGWR Recognition Accuracy: {tmgwr_accuracy:.2f}%, Avg Error: {tmgwr_avg_error:.2f}")
            print(f"MINERVA Recognition Accuracy: {minerva_accuracy:.2f}%, Avg Error: {minerva_avg_error:.2f}")
    
    return results

def recover_state_tmgwr(agent, masked_state, masked_dim):
    """
    Recover the complete state for TMGWR agent when one dimension is masked
    
    Args:
        agent: TMGWR agent
        masked_state: State with one dimension masked (np.nan)
        masked_dim: Which dimension is masked (0=X, 1=Y)
    
    Returns:
        recovered_state: Complete state with best guess for the masked dimension
    """
    # Create a query with the available dimension
    available_dim = 1 - masked_dim  # If X is masked (0), then Y (1) is available
    
    # Find nodes that have similar value for the available dimension
    distances = []
    for node_weights in agent.model.W:
        # Compare only the available dimension
        if available_dim == 0:
            # X is available, Y is masked
            dist = abs(node_weights[0] - masked_state[0])
        else:
            # Y is available, X is masked
            dist = abs(node_weights[1] - masked_state[1])
        distances.append(dist)
    
    # Get the closest node
    best_node_idx = np.argmin(distances)
    best_node = agent.model.W[best_node_idx]
    
    # Create recovered state by using the original value for the available dimension
    # and the best node's value for the masked dimension
    recovered_state = masked_state.copy()
    recovered_state[masked_dim] = best_node[masked_dim]
    
    return recovered_state

def recover_state_minerva(agent, masked_state, masked_dim):
    """
    Recover the complete state for MINERVA agent when one dimension is masked
    
    Args:
        agent: MINERVA agent
        masked_state: State with one dimension masked (np.nan)
        masked_dim: Which dimension is masked (0=X, 1=Y)
    
    Returns:
        recovered_state: Complete state with best guess for the masked dimension
    """
    # Create a temporary state for querying
    query_state = masked_state.copy()
    
    # Replace NaN with a mean value temporarily to get the unmasked dimension's pattern
    if masked_dim == 0:
        # X is masked, use Y to recover
        query_state[0] = 0  # Temporary value, will be ignored
        
        # Get Y pattern
        y_data = np.array([query_state[1]]).reshape(1, -1)
        y_bmu = agent.lower_y.find_best_matching_units(y_data)[0]
        
        # Find all patterns with this Y BMU
        matching_nodes = []
        for idx, node_pattern in enumerate(agent.nodes):
            _, y_pattern = node_pattern
            if y_pattern[y_bmu] == 1:
                matching_nodes.append(idx)
        
        # Find positions for all matching nodes
        positions = [agent.node_positions[idx] for idx in matching_nodes if idx in agent.node_positions]
        
        # Use average X value from matching positions
        if positions:
            avg_x = np.mean([pos[0] for pos in positions])
            recovered_state = masked_state.copy()
            recovered_state[0] = avg_x
            return recovered_state
        
        # If no matches, use nearest BMU position
        if y_bmu < len(agent.lower_y.A):
            x_values = [pos[0] for pos in agent.node_positions.values()]
            recovered_state = masked_state.copy()
            recovered_state[0] = np.mean(x_values) if x_values else 0
            return recovered_state
    
    else:
        # Y is masked, use X to recover
        query_state[1] = 0  # Temporary value, will be ignored
        
        # Get X pattern
        x_data = np.array([query_state[0]]).reshape(1, -1)
        x_bmu = agent.lower_x.find_best_matching_units(x_data)[0]
        
        # Find all patterns with this X BMU
        matching_nodes = []
        for idx, node_pattern in enumerate(agent.nodes):
            x_pattern, _ = node_pattern
            if x_pattern[x_bmu] == 1:
                matching_nodes.append(idx)
        
        # Find positions for all matching nodes
        positions = [agent.node_positions[idx] for idx in matching_nodes if idx in agent.node_positions]
        
        # Use average Y value from matching positions
        if positions:
            avg_y = np.mean([pos[1] for pos in positions])
            recovered_state = masked_state.copy()
            recovered_state[1] = avg_y
            return recovered_state
        
        # If no matches, use nearest BMU position
        if x_bmu < len(agent.lower_x.A):
            y_values = [pos[1] for pos in agent.node_positions.values()]
            recovered_state = masked_state.copy()
            recovered_state[1] = np.mean(y_values) if y_values else 0
            return recovered_state
    
    # Fallback
    recovered_state = masked_state.copy()
    recovered_state[masked_dim] = 0
    return recovered_state

def plot_pattern_completion_results(results):
    """Plot pattern completion experiment results"""
    # Convert mask level to percentage for more readable labels
    def format_mask_level(level):
        return f"{int(level * 100)}%"
    
    metrics = {
        'recognition_acc': {'title': 'State Recognition Accuracy vs Mask Level', 
                           'ylabel': 'Recognition Accuracy (%)', 'ylim': (0, 100)},
        'recovery_error': {'title': 'State Recovery Error vs Mask Level', 
                          'ylabel': 'Average Recovery Error', 'ylim': None}
    }
    
    for metric in metrics:
        data = []
        for agent in results:
            for mask_level in results[agent]:
                values = results[agent][mask_level][metric]
                for value in values:
                    data.append({
                        'Agent': agent,
                        'Mask Level': format_mask_level(mask_level),
                        'Value': value,
                        'Metric': metrics[metric]['ylabel'],
                        'Mask_sort': mask_level
                    })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Mask_sort')
        
        plt.figure(figsize=(12, 8))
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        custom_palette = {'TMGWR': 'green', 'MINERVA': 'orange'}
        sns.boxplot(data=df, x='Mask Level', y='Value', hue='Agent', palette=custom_palette)
        
        plt.title(metrics[metric]['title'], fontsize=14, pad=20)
        plt.xlabel('Mask Level (percentage of state masked)', fontsize=12)
        plt.ylabel(metrics[metric]['ylabel'], fontsize=12)
        plt.legend(title='Agent Type', title_fontsize=12, fontsize=10)
        
        if metrics[metric]['ylim']:
            plt.ylim(metrics[metric]['ylim'])
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    random.seed(42)
    
    # Run the experiment
    print("Starting pattern completion experiment...")
    results = run_pattern_completion_experiment(
        mask_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        runs_per_level=3
    )
    
    # Plot results
    plot_pattern_completion_results(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for agent in results:
        print(f"\n{agent}:")
        for mask_level in results[agent]:
            recognition = results[agent][mask_level]['recognition_acc']
            error = results[agent][mask_level]['recovery_error']
            print(f"Mask Level = {mask_level*100:.0f}%:")
            print(f"  Mean Recognition Accuracy: {np.mean(recognition):.2f}%")
            print(f"  Std Recognition Accuracy: {np.std(recognition):.2f}%")
            print(f"  Mean Recovery Error: {np.mean(error):.2f}")
            print(f"  Std Recovery Error: {np.std(error):.2f}")