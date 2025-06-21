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

# Set random seed for reproducibility
np.random.seed(42)

def collect_maze_positions(exploration_steps=10000, save_path="maze_positions.csv"):
    """Explore the maze to collect valid positions"""
    print(f"Starting pre-exploration to collect positional data...")
    
    # Check if the file already exists
    if os.path.exists(save_path):
        print(f"Position data already exists at {save_path}, using existing data")
        df = pd.read_csv(save_path)
        return df.values
    
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
    
    return df.values

class CustomHierarchicalGWRSOMAgent(HierarchicalGWRSOMAgent):
    """Extends HierarchicalGWRSOMAgent to allow customizing lower network learning rates"""
    
    def __init__(self, lower_a=0.01, lower_h=0.01, **kwargs):
        """
        Initialize with custom learning rates for lower networks
        
        Parameters:
        - lower_a: Activity threshold for lower networks (learning rate)
        - lower_h: Firing threshold for lower networks
        - **kwargs: Other parameters passed to parent class
        """
        super().__init__(**kwargs)
        
        # Override lower networks with custom learning rates
        self.lower_x = GWRSOM(a=lower_a, h=lower_h)
        self.lower_y = GWRSOM(a=lower_a, h=lower_h)
        
        # Store the learning rate for analysis
        self.lower_learning_rate = lower_a
        
    def get_network_stats(self):
        """Return statistics about the networks"""
        return {
            'lower_learning_rate': self.lower_learning_rate,
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
        
        # Print higher-level network weights
        # if hasattr(self, 'nodes') and len(self.nodes) > 0:  # Check if the list is not empty
        #     print("\nHigher-Level Network Weights:")
        #     for i, node in enumerate(self.nodes):
        #         print(f"  Node {i}: {node}")
        # else:
        #     print("Higher-Level Network has no nodes.")
        
        print("--- End of Debug ---\n")

def run_single_experiment(epsilon_b, epsilon_n, num_episodes=15, verbose=True):
    """
    Run a single experiment with the specified learning rates for BMU and neighbors.
    
    Parameters:
    - epsilon_b: Learning rate for the BMU
    - epsilon_n: Learning rate for the neighbors
    - num_episodes: Number of training episodes
    - verbose: Whether to print detailed progress
    
    Returns:
    - agent: Trained agent
    - training_stats: Dictionary with training statistics
    - final_stats: Final network statistics
    """
    if verbose:
        print(f"\nRunning experiment with epsilon_b = {epsilon_b}, epsilon_n = {epsilon_n}")
    
    # Get maze details
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map() 
    
    # Create maze
    Maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos_index, goal_index_pos=goal_pos_index)
    
    # Get goal and initial positions
    goal = Maze.get_goal_pos()
    initial_state = Maze.get_initial_player_pos()
    
    # Get training data
    training_data = collect_maze_positions(exploration_steps=10000)
    
    # Initialize agent with specified learning rates
    agent = CustomHierarchicalGWRSOMAgent(
        lower_a=0.1,  # Keep activity threshold constant
        lower_h=0.1,
        lower_dim=1,
        higher_dim=2,
        epsilon_b=epsilon_b,  # Vary BMU learning rate
        epsilon_n=epsilon_n,  # Vary neighbor learning rate
        beta=0.7,
        delta=0.79,
        T_max=20,
        N_max=100,
        eta=0.5,
        phi=0.9,
        sigma=0.5
    )
    
    # Train lower networks
    agent.train_lower_networks(training_data, epochs=20)
    agent.print_node_weights()
    
    # Set goal and exploration rate
    agent.set_goal(goal=goal)
    agent.set_epsilon(1)
    
    # Track training statistics
    training_stats = {
        'episodes': [],
        'steps': [],
        'epsilon': [],
        'success': [],
        'epsilon_b': epsilon_b,
        'epsilon_n': epsilon_n,
        'lower_x_nodes': [],
        'lower_y_nodes': [],
        'higher_nodes': [],
        'higher_connections': []
    }
    
    # Set current state to initial state
    current_state = initial_state
    
    # Track goal reaching to decay epsilon
    reached_goal_count = 0
    
    # Training loop
    for episode_num in range(num_episodes):
        current_state = initial_state
        Maze.reset_player()
        step_counter = 0
        episode_success = False
        
        # Get network stats at start of episode
        network_stats = agent.get_network_stats()
        
        # Store current network stats
        for key in ['lower_x_nodes', 'lower_y_nodes', 'higher_nodes', 'higher_connections']:
            training_stats[key].append(network_stats[key])
        
        # Episode loop
        while current_state != goal and step_counter < 10000:  # Limiting to 1000 steps for efficiency
            step_counter += 1
            
            # Use the true state directly (no noise added)
            noisy_state = current_state
            
            # Select action
            action = agent.select_action(current_state=noisy_state)
            
            # Execute action
            Maze.move_player(action=action)
            next_state = Maze.get_player_pos()
            
            # Update model
            agent.update_model(next_state=next_state, action=action)
            current_state = next_state
            
            # Print progress
            if verbose and step_counter % 1000 == 0:
                print(f"Episode {episode_num + 1}, step {step_counter}, BMU learning rate = {epsilon_b}, Neighbor learning rate = {epsilon_n}")            
            # Check if goal reached
            if current_state == goal:
                episode_success = True
                break
        
        # Update epsilon based on success
        if episode_success:
            reached_goal_count += 1
            if reached_goal_count > 5:  # Decay epsilon sooner for faster adaptation
                agent.decay_epsilon(min_epsilon=0.1)
        
        # Store episode stats
        training_stats['episodes'].append(episode_num + 1)
        training_stats['steps'].append(step_counter)
        training_stats['epsilon'].append(agent.get_epsilon())
        training_stats['success'].append(episode_success)
        
        if verbose:
            print(f"Episode: {episode_num + 1}, Steps: {step_counter}, "
                  f"Epsilon: {agent.get_epsilon():.2f}, Success: {episode_success}")
        
        print(f"End of Episode {episode_num + 1}")
        agent.print_node_weights()
    
    # Get final network stats
    final_stats = agent.get_network_stats()
    
    return agent, training_stats, final_stats

def run_learning_rate_study(learning_rates, num_episodes=15):
    """
    Run the study for multiple learning rates
    
    Parameters:
    - learning_rates: List of learning rates to test
    - num_episodes: Number of episodes per experiment
    
    Returns:
    - results: Dictionary mapping learning rates to experiment results
    """
    results = {}
    
    # Run sequentially
    print(f"Running {len(learning_rates)} experiments sequentially...")
    for lr in tqdm(learning_rates):
        print(f"\nRunning experiment with learning rate = {lr}")
        agent, stats, final_stats = run_single_experiment(
            learning_rate=lr,
            num_episodes=num_episodes,
            verbose=True
        )
        results[lr] = (agent, stats, final_stats)  # Use learning_rate as the key
    
    return results

def analyze_results(results, episode_to_display=10):
    """
    Analyze and visualize the results of the learning rate study with a single chart 
    showing side-by-side bar graphs for X and Y networks.

    Parameters:
    - results: Dictionary mapping learning_rate to experiment results
    - episode_to_display: Which episode number to display (default: last episode)
    """
    # Extract learning rates
    learning_rates = sorted(results.keys())

    # Prepare dataframes
    episode_data = []

    if not results:
        print("No results to analyze!")
        return pd.DataFrame()

    for lr, (_, stats, _) in results.items():
        # Extract episode data
        for i in range(len(stats['episodes'])):
            episode_data.append({
                'learning_rate': lr,
                'episode': stats['episodes'][i],
                'lower_x_nodes': stats['lower_x_nodes'][i] if 'lower_x_nodes' in stats else 0,
                'lower_y_nodes': stats['lower_y_nodes'][i] if 'lower_y_nodes' in stats else 0,
            })

    # Convert to dataframe
    episode_df = pd.DataFrame(episode_data)

    if episode_df.empty:
        print("Warning: No data to plot!")
        return episode_df
        
    # Filter data for the specified episode
    if episode_to_display > max(episode_df['episode']):
        episode_to_display = max(episode_df['episode'])
        print(f"Warning: Requested episode not available. Using episode {episode_to_display} instead.")
        
    episode_data = episode_df[episode_df['episode'] == episode_to_display]
    
    # Check if we have data for the specified episode
    if episode_data.empty:
        print(f"No data available for episode {episode_to_display}")
        return episode_df
    
    # Create a new figure with larger size for better visualization
    plt.figure(figsize=(14, 8))
    
    # Define colors for the bars - deep green and amber
    x_color = '#006400'  # Deep green
    y_color = '#FFA500'  # Amber
    
    # Set width and positions for the bars
    bar_width = 0.35
    x_positions = np.arange(len(learning_rates))
    
    # Get the values for each learning rate
    x_node_counts = []
    y_node_counts = []
    for lr in learning_rates:
        subset = episode_data[episode_data['learning_rate'] == lr]
        if not subset.empty:
            x_node_counts.append(subset['lower_x_nodes'].values[0])
            y_node_counts.append(subset['lower_y_nodes'].values[0])
        else:
            x_node_counts.append(0)
            y_node_counts.append(0)
    
    # Create the side-by-side bar chart
    bars_x = plt.bar(x_positions - bar_width/2, x_node_counts, bar_width, 
                     color=x_color, label='X Network Nodes')
    bars_y = plt.bar(x_positions + bar_width/2, y_node_counts, bar_width, 
                     color=y_color, label='Y Network Nodes')
    
    # Add value labels on top of each bar
    for i, v in enumerate(x_node_counts):
        plt.text(i - bar_width/2, v + 0.5, str(v), ha='center', fontweight='bold')
    
    for i, v in enumerate(y_node_counts):
        plt.text(i + bar_width/2, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # Set axis labels and title
    plt.xlabel('Learning Rate', fontsize=14)
    plt.ylabel('Number of Nodes', fontsize=14)
    plt.title(f'X and Y Network Nodes by Learning Rate', fontsize=16)
    
    # Set x-axis ticks
    plt.xticks(x_positions, [f"{lr}" for lr in learning_rates], rotation=45)
    plt.legend(fontsize=12)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    try:
        plt.savefig(f'network_nodes_comparison_episode_{episode_to_display}.png', dpi=300)
        print(f"Saved visualization to network_nodes_comparison_episode_{episode_to_display}.png")
    except Exception as e:
        print(f"Could not save figure: {e}")

    return episode_df

def main():
    """Main function to run the learning rate study"""
    start_time = time.time()
    
    # Define learning rates to test
    learning_rates_b = [0.1, 0.01, 0.001, 0.0001, 0.00001]  # BMU learning rates
    learning_rates_n = [0.1, 0.01, 0.001, 0.0001, 0.00001]  # Neighbor learning rates

    # Run experiments
    results = {}
    for epsilon_b in learning_rates_b:
        for epsilon_n in learning_rates_n:
            agent, stats, final_stats = run_single_experiment(
                epsilon_b=epsilon_b,
                epsilon_n=epsilon_n,
                num_episodes=10,
                verbose=True
            )
            results[(epsilon_b, epsilon_n)] = (agent, stats, final_stats)
    
    # Analyze results with bar charts for the last episode
    print("Analyzing results...")
    episode_df = analyze_results(results, episode_to_display=10)  # Display the last episode
    
    # Save data if we have results
    if results:
        try:
            with open('learning_rate_study_results.pkl', 'wb') as f:
                pickle.dump({
                    'results': results,
                    'episode_df': episode_df
                }, f)
            
            # Save dataframes to CSV for easier analysis
            if not episode_df.empty:
                episode_df.to_csv('learning_rate_episode_data.csv', index=False)
                
            print(f"Results saved to learning_rate_study_results.pkl")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"Study completed in {elapsed_time:.2f} seconds")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()