# Import necessary libraries
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.HSOM_binary import GWRSOM, HierarchicalGWRSOMAgent
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from tqdm import tqdm
import seaborn as sns
import pickle
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)

def collect_maze_positions(noise_level=0, exploration_steps=10000, save_path="maze_positions.csv"):
    """
    Explore the maze to collect valid positions with optional noise
    
    Parameters:
    - noise_level: Standard deviation of Gaussian noise to add to observations
    - exploration_steps: Number of steps to explore
    - save_path: Path to save the positions
    
    Returns:
    - NumPy array of collected positions
    """
    print(f"Starting pre-exploration to collect positional data with noise level {noise_level}...")
    
    # Always create a new file for each noise level
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Removed existing file at {save_path}")
    
    # Get maze details
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map() 
    
    # Initialize maze
    Maze = MazePlayer(maze_map=maze_map, 
                     player_index_pos=player_pos_index, 
                     goal_index_pos=goal_pos_index)
    
    # Storage for visited positions
    positions = []
    
    # Reset player to initial position
    Maze.reset_player()
    current_state = Maze.get_player_pos()
    
    # Perform random exploration to collect positions
    for step in range(exploration_steps):
        # Store current position - with noise if noise_level > 0
        noise_probability = 0.5  # Probability of adding noise
        
        if noise_level > 0 and np.random.uniform(0, 1) < noise_probability:
            # Add noise to the observation
            noisy_state = np.array(current_state) + np.random.normal(0, np.sqrt(noise_level), 2)
            positions.append(noisy_state)
        else:
            # Use the true state
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
    
    print(f"Pre-exploration complete. Collected {len(df)} unique positions with noise level {noise_level}.")
    print(f"Position data saved to {save_path}")
    
    return df.values

class CustomHierarchicalGWRSOMAgent(HierarchicalGWRSOMAgent):
    """
    Extends HierarchicalGWRSOMAgent to allow customizing GWRSOM parameters
    with proper distinction between winner learning rate and other parameters
    """
    
    def __init__(self, winner_learning_rate=0.2, neighbor_learning_rate=0.05, 
                 activity_threshold=0.1, firing_threshold=0.1, **kwargs):
        """
        Initialize with custom parameters for lower networks
        
        Parameters:
        - winner_learning_rate: Learning rate for winner node (es)
        - neighbor_learning_rate: Learning rate for neighbor nodes (en)
        - activity_threshold: Threshold for determining when to add nodes (a)
        - firing_threshold: Threshold for node firing counter (h)
        - **kwargs: Other parameters passed to parent class
        """
        super().__init__(**kwargs)
        
        # Override lower networks with custom parameters
        self.lower_x = GWRSOM(es=winner_learning_rate, 
                              en=neighbor_learning_rate,
                              a=activity_threshold, 
                              h=firing_threshold)
        
        self.lower_y = GWRSOM(es=winner_learning_rate, 
                              en=neighbor_learning_rate,
                              a=activity_threshold, 
                              h=firing_threshold)
        
        # Store the parameters for analysis
        self.winner_learning_rate = winner_learning_rate
        self.neighbor_learning_rate = neighbor_learning_rate
        self.activity_threshold = activity_threshold
        self.firing_threshold = firing_threshold
        
    def get_network_stats(self):
        """Return statistics about the networks"""
        return {
            'winner_learning_rate': self.winner_learning_rate,
            'neighbor_learning_rate': self.neighbor_learning_rate,
            'activity_threshold': self.activity_threshold,
            'firing_threshold': self.firing_threshold,
            'lower_x_nodes': len(self.lower_x.A) if hasattr(self.lower_x, 'A') else 0,
            'lower_y_nodes': len(self.lower_y.A) if hasattr(self.lower_y, 'A') else 0,
            'higher_nodes': len(self.nodes),
            'higher_connections': int(np.sum(self.connections)) if hasattr(self, 'connections') else 0
        }

    def print_node_weights(self):
        """Print the weights of all nodes in the lower and higher networks."""
        print("\n--- Debug: Node Weights ---")
        
        # Print lower_x network weights
        if hasattr(self.lower_x, 'A') and len(self.lower_x.A) > 0:  # Check if the array is not empty
            print("Lower X Network Weights:")
            for i, weight in enumerate(self.lower_x.A):
                print(f"  Node {i}: {weight}")
        else:
            print("Lower X Network has no nodes.")
        
        # Print lower_y network weights
        if hasattr(self.lower_y, 'A') and len(self.lower_y.A) > 0:  # Check if the array is not empty
            print("\nLower Y Network Weights:")
            for i, weight in enumerate(self.lower_y.A):
                print(f"  Node {i}: {weight}")
        else:
            print("Lower Y Network has no nodes.")
        
        print("--- End of Debug ---\n")
        
    def get_firing_pattern_with_bmus(self, state):
        """
        Modified version of get_firing_pattern that also returns the BMU indices
        """
        # Use lower networks to encode position
        x_data = np.array([state[0]]).reshape(1, -1)
        y_data = np.array([state[1]]).reshape(1, -1)
        
        # Get best matching units from lower networks
        x_bmus = self.lower_x.find_best_matching_units(x_data)
        if not isinstance(x_bmus, (list, tuple)) or len(x_bmus) < 2:
            raise ValueError(f"Expected list of at least 2 BMU indices, got: {x_bmus}")
        x_bmus_id = x_bmus[0]
        y_bmus = self.lower_y.find_best_matching_units(y_data)
        if not isinstance(y_bmus, (list, tuple)) or len(y_bmus) < 2:
            raise ValueError(f"Expected list of at least 2 BMU indices, got: {x_bmus}")
        y_bmus_id = y_bmus[0]
        
        # Create binary vectors
        x_binary = np.zeros(len(self.lower_x.A))
        y_binary = np.zeros(len(self.lower_y.A))
        
        # Set the active units
        if isinstance(x_bmus, tuple) or isinstance(x_bmus, list):
            x_binary[x_bmus[0]] = 1
        else:
            x_binary[x_bmus] = 1
            
        if isinstance(y_bmus, tuple) or isinstance(y_bmus, list):
            y_binary[y_bmus[0]] = 1
        else:
            y_binary[y_bmus] = 1
        
        return np.array(x_binary), np.array(y_binary), (x_bmus_id, y_bmus_id)

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

def run_single_experiment(winner_learning_rate, noise_level, verbose=True):
    """
    Run a single experiment with specified winner learning rate and noise level
    
    Parameters:
    - winner_learning_rate: Learning rate for the winner node (es parameter)
    - noise_level: Level of noise to add to observations
    - verbose: Whether to print detailed progress
    
    Returns:
    - agent: Trained agent
    - training_stats: Dictionary with training statistics
    - final_stats: Final network statistics
    """
    if verbose:
        print(f"\nRunning experiment with winner_learning_rate = {winner_learning_rate}, noise_level = {noise_level}")
    
    # Get maze details
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map() 
    
    # Create maze
    Maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos_index, goal_index_pos=goal_pos_index)
    
    # Get goal and initial positions
    goal = Maze.get_goal_pos()
    initial_state = Maze.get_initial_player_pos()
    
    # Get training data with the current noise level
    # This ensures maze positions are regenerated for each noise level
    training_data = collect_maze_positions(
        noise_level=noise_level,
        exploration_steps=10000,
        save_path=f"maze_positions_noise_{noise_level}.csv"
    )
    
    # Initialize agent with specified winner learning rate
    # Keep other parameters constant
    agent = CustomHierarchicalGWRSOMAgent(
        winner_learning_rate=winner_learning_rate,  # Variable parameter (es)
        neighbor_learning_rate=0.05,                # Fixed (en)
        activity_threshold=0.0001,                  # Fixed low value based on previous findings (a)
        firing_threshold=0.1,                       # Fixed (h)
        lower_dim=1,
        higher_dim=2,
        epsilon_b=0.35,  # Higher-level parameters kept fixed
        epsilon_n=0.15,
        beta=0.7,
        delta=0.79,
        T_max=20,
        N_max=100,
        eta=0.5,
        phi=0.9,
        sigma=0.5
    )
    
    # Train lower networks
    print(f"Training lower networks with noise level {noise_level}...")
    agent.train_lower_networks(training_data, epochs=20)
    
    if verbose:
        agent.print_node_weights()
    
    # Get network stats
    network_stats = agent.get_network_stats()
    
    # Record training statistics
    training_stats = {
        'winner_learning_rate': winner_learning_rate,
        'noise_level': noise_level,
        'lower_x_nodes': network_stats['lower_x_nodes'],
        'lower_y_nodes': network_stats['lower_y_nodes'],
        'higher_nodes': len(agent.nodes),
    }
    
    if verbose:
        print(f"Winner Learning Rate: {winner_learning_rate}, Noise Level: {noise_level}")
        print(f"Lower X Nodes: {network_stats['lower_x_nodes']}")
        print(f"Lower Y Nodes: {network_stats['lower_y_nodes']}")
    
    # Run one episode to test higher level node formation
    if verbose:
        print("Running one episode to test higher level node formation...")
    
    agent.set_goal(goal=goal)
    agent.set_epsilon(1.0)  # Pure exploration
    
    # Reset player to initial position
    Maze.reset_player()
    current_state = initial_state
    step_counter = 0
    
    # Run a single episode
    while current_state != goal and step_counter < 5000:  # Limiting to 5000 steps
        step_counter += 1
        
        # Get the true state
        true_state = np.array(current_state)
        
        # Add noise to observation with some probability if noise_level > 0
        noise_probability = 0.5
        if noise_level > 0 and np.random.uniform(0, 1) < noise_probability:
            noisy_state = true_state + np.random.normal(0, np.sqrt(noise_level), 2)
        else:
            noisy_state = true_state.copy()
        
        # Select action using noisy state
        action = agent.select_action(current_state=noisy_state)
        
        # Execute action
        Maze.move_player(action=action)
        next_state = Maze.get_player_pos()
        
        # Update model with true next state
        agent.update_model(next_state=next_state, action=action)
        current_state = next_state
        
        # Print progress occasionally
        if verbose and step_counter % 1000 == 0:
            print(f"Step {step_counter}, Higher-level nodes: {len(agent.nodes)}")
        
        # Check if goal reached
        if current_state == goal:
            if verbose:
                print(f"Goal reached in {step_counter} steps")
            break
    
    # Update training stats with final higher level node count
    training_stats['higher_nodes'] = len(agent.nodes)
    
    if verbose:
        print(f"Final higher-level nodes: {len(agent.nodes)}")
    
    return agent, training_stats, network_stats

def run_learning_rate_noise_study(learning_rates, noise_levels):
    """
    Run a study to analyze the effect of winner learning rate and noise on
    node formation in GWRSOM networks
    
    Parameters:
    - learning_rates: List of winner learning rates to test
    - noise_levels: List of noise levels to test
    
    Returns:
    - results: DataFrame with experiment results
    """
    results = {
        'winner_learning_rate': [],
        'noise_level': [],
        'lower_x_nodes': [],
        'lower_y_nodes': [],
        'higher_nodes': []
    }
    
    # Run all combinations of learning rates and noise levels
    print(f"Running {len(learning_rates) * len(noise_levels)} experiments...")
    
    # Track progress
    total_experiments = len(learning_rates) * len(noise_levels)
    experiment_count = 0
    
    for lr in learning_rates:
        for noise in noise_levels:
            experiment_count += 1
            print(f"\nExperiment {experiment_count}/{total_experiments}: Winner Learning Rate = {lr}, Noise Level = {noise}")
            
            agent, stats, final_stats = run_single_experiment(
                winner_learning_rate=lr,
                noise_level=noise,
                verbose=True
            )
            
            # Record results
            results['winner_learning_rate'].append(lr)
            results['noise_level'].append(noise)
            results['lower_x_nodes'].append(stats['lower_x_nodes'])
            results['lower_y_nodes'].append(stats['lower_y_nodes'])
            results['higher_nodes'].append(stats['higher_nodes'])
    
    return pd.DataFrame(results)

def plot_node_counts_by_learning_rate(df):
    """
    Create separate bar charts for each learning rate showing
    the effect of noise on the number of nodes
    
    Parameters:
    - df: DataFrame with results from the study
    """
    # Get unique learning rates and noise levels
    learning_rates = sorted(df['winner_learning_rate'].unique())
    noise_levels = sorted(df['noise_level'].unique())
    
    # Create nice labels for noise levels (using fractions)
    noise_labels = []
    for noise in noise_levels:
        if noise == 0:
            noise_labels.append("0")
        elif noise == 1/6:
            noise_labels.append("1/6")
        elif noise == 1/3:
            noise_labels.append("1/3")
        elif noise == 1/2:
            noise_labels.append("1/2")
        else:
            noise_labels.append(str(noise))
    
    # Create one figure per learning rate
    for lr in learning_rates:
        # Filter data for this learning rate
        lr_data = df[df['winner_learning_rate'] == lr]
        
        # Create a figure with three subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))
        fig.suptitle(f'Effect of Noise on Node Counts (Winner Learning Rate = {lr})', fontsize=16)
        
        # Bar width
        width = 0.35
        
        # X positions
        x_pos = np.arange(len(noise_levels))
        
        # Plot X network nodes
        x_values = [lr_data[lr_data['noise_level'] == noise]['lower_x_nodes'].values[0] for noise in noise_levels]
        bars1 = ax1.bar(x_pos, x_values, width, label='X Network', color='green')
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Number of Nodes')
        ax1.set_title('X Network Nodes')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(noise_labels)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(x_values):
            ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        # Plot Y network nodes
        y_values = [lr_data[lr_data['noise_level'] == noise]['lower_y_nodes'].values[0] for noise in noise_levels]
        bars2 = ax2.bar(x_pos, y_values, width, label='Y Network', color='orange')
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Number of Nodes')
        ax2.set_title('Y Network Nodes')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(noise_labels)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(y_values):
            ax2.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        # Plot higher-level nodes
        h_values = [lr_data[lr_data['noise_level'] == noise]['higher_nodes'].values[0] for noise in noise_levels]
        bars3 = ax3.bar(x_pos, h_values, width, label='Higher Network', color='blue')
        ax3.set_xlabel('Noise Level')
        ax3.set_ylabel('Number of Nodes')
        ax3.set_title('Higher-Level Network Nodes')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(noise_labels)
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(h_values):
            ax3.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'node_counts_winner_lr_{lr}.png')
        plt.show()

def analyze_results(df):
    """
    Analyze the results of the winner learning rate and noise study
    
    Parameters:
    - df: DataFrame with results from the study
    """
    # 1. Create separate bar charts for each learning rate
    plot_node_counts_by_learning_rate(df)
    
    # 2. Create a summary data frame that's easy to read
    summary = df.pivot_table(
        index='winner_learning_rate',
        columns='noise_level',
        values=['lower_x_nodes', 'lower_y_nodes', 'higher_nodes']
    )
    
    print("\nSummary of Results:")
    print(summary)
    
    # 3. Save summary to CSV
    summary.to_csv('winner_lr_noise_summary.csv')
    print("Summary saved to winner_lr_noise_summary.csv")
    
    return summary

def main():
    """Main function to run the winner learning rate and noise study"""
    start_time = time.time()
    
    # Define winner learning rates to test (covering a wide range)
    winner_learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    # Define noise levels to test
    noise_levels = [0, 1/6, 1/3, 1/2]
    
    # Run experiments
    results_df = run_learning_rate_noise_study(
        learning_rates=winner_learning_rates,
        noise_levels=noise_levels
    )
    
    # Analyze and visualize results
    print("Analyzing results...")
    summary_df = analyze_results(results_df)
    
    # Save complete results DataFrame
    results_df.to_csv('winner_lr_noise_complete_results.csv', index=False)
    print("Complete results saved to winner_lr_noise_complete_results.csv")
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"Study completed in {elapsed_time:.2f} seconds")
    
    # Print overall findings
    print("\nKey Observations:")
    
    # 1. Effect of noise on node counts by winner learning rate
    for lr in winner_learning_rates:
        lr_data = results_df[results_df['winner_learning_rate'] == lr]
        min_noise = lr_data['noise_level'].min()
        max_noise = lr_data['noise_level'].max()
        
        # Get node counts at min and max noise
        x_nodes_min = lr_data[lr_data['noise_level'] == min_noise]['lower_x_nodes'].values[0]
        x_nodes_max = lr_data[lr_data['noise_level'] == max_noise]['lower_x_nodes'].values[0]
        
        y_nodes_min = lr_data[lr_data['noise_level'] == min_noise]['lower_y_nodes'].values[0]
        y_nodes_max = lr_data[lr_data['noise_level'] == max_noise]['lower_y_nodes'].values[0]
        
        h_nodes_min = lr_data[lr_data['noise_level'] == min_noise]['higher_nodes'].values[0]
        h_nodes_max = lr_data[lr_data['noise_level'] == max_noise]['higher_nodes'].values[0]
        
        # Calculate percentage change
        x_change_pct = ((x_nodes_max - x_nodes_min) / x_nodes_min) * 100
        y_change_pct = ((y_nodes_max - y_nodes_min) / y_nodes_min) * 100
        h_change_pct = ((h_nodes_max - h_nodes_min) / h_nodes_min) * 100 if h_nodes_min > 0 else float('inf')
        
        print(f"Winner Learning Rate {lr}:")
        print(f"  X Network: {x_nodes_min} nodes with no noise, {x_nodes_max} nodes with noise={max_noise}")
        print(f"    Change: {x_change_pct:.1f}%")
        print(f"  Y Network: {y_nodes_min} nodes with no noise, {y_nodes_max} nodes with noise={max_noise}")
        print(f"    Change: {y_change_pct:.1f}%")
        print(f"  Higher Network: {h_nodes_min} nodes with no noise, {h_nodes_max} nodes with noise={max_noise}")
        print(f"    Change: {h_change_pct:.1f}%")
    
    # 2. Find which winner learning rate is most robust to noise
    # (smallest percentage increase in nodes from min to max noise)
    robustness_data = []
    
    for lr in winner_learning_rates:
        lr_data = results_df[results_df['winner_learning_rate'] == lr]
        min_noise = lr_data['noise_level'].min()
        max_noise = lr_data['noise_level'].max()
        
        # Get node counts at min and max noise
        x_nodes_min = lr_data[lr_data['noise_level'] == min_noise]['lower_x_nodes'].values[0]
        x_nodes_max = lr_data[lr_data['noise_level'] == max_noise]['lower_x_nodes'].values[0]
        
        y_nodes_min = lr_data[lr_data['noise_level'] == min_noise]['lower_y_nodes'].values[0]
        y_nodes_max = lr_data[lr_data['noise_level'] == max_noise]['lower_y_nodes'].values[0]
        
        h_nodes_min = lr_data[lr_data['noise_level'] == min_noise]['higher_nodes'].values[0]
        h_nodes_max = lr_data[lr_data['noise_level'] == max_noise]['higher_nodes'].values[0]
        
        # Calculate percentage change
        x_change_pct = ((x_nodes_max - x_nodes_min) / x_nodes_min) * 100 if x_nodes_min > 0 else float('inf')
        y_change_pct = ((y_nodes_max - y_nodes_min) / y_nodes_min) * 100 if y_nodes_min > 0 else float('inf')
        h_change_pct = ((h_nodes_max - h_nodes_min) / h_nodes_min) * 100 if h_nodes_min > 0 else float('inf')
        
        # Average change across all network types
        avg_change = (x_change_pct + y_change_pct + h_change_pct) / 3
        
        robustness_data.append({
            'winner_learning_rate': lr,
            'avg_change_pct': avg_change
        })
    
    # Find most robust winner learning rate
    robustness_df = pd.DataFrame(robustness_data)
    most_robust_lr = robustness_df.loc[robustness_df['avg_change_pct'].idxmin()]['winner_learning_rate']
    
    print(f"\nMost robust winner learning rate: {most_robust_lr}")
    print("This winner learning rate shows the smallest percentage increase in node count as noise increases.")

if __name__ == "__main__":
    main()