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

def run_single_noise_experiment(learning_rate, noise_level, num_episodes=15, verbose=True):
    """
    Run a single experiment with specified learning rate and noise level
    
    Parameters:
    - learning_rate: Learning rate for the lower networks
    - noise_level: Level of noise to add to observations
    - num_episodes: Number of training episodes
    - verbose: Whether to print detailed progress
    
    Returns:
    - agent: Trained agent
    - training_stats: Dictionary with training statistics
    - final_stats: Final network statistics
    """
    if verbose:
        print(f"\nRunning experiment with learning_rate = {learning_rate}, noise_level = {noise_level}")
    
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
        lower_a=learning_rate,  # Learning rate for lower networks
        lower_h=learning_rate,  # Using same value for firing threshold
        lower_dim=1,
        higher_dim=2,
        epsilon_b=0.35,  # Default value 
        epsilon_n=0.15,  # Default value
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
    
    if verbose:
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
        'learning_rate': learning_rate,
        'noise_level': noise_level,
        'lower_x_nodes': [],
        'lower_y_nodes': [],
        'higher_nodes': [],
        'higher_connections': [],
        'bmu_stability': [],  # New metric to track pattern stability under noise
        'purity': []  # New metric to track state-node mapping purity
    }
    
    # Track state-to-bmu mappings for stability analysis
    state_to_bmu_map = {}
    bmu_changes = 0
    bmu_observations = 0
    
    # Track state-node mappings for purity calculation
    state_node_mappings = defaultdict(lambda: defaultdict(int))
    total_visits = 0
    
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
        
        # Reset metrics for this episode
        bmu_changes = 0
        bmu_observations = 0
        state_node_mappings = defaultdict(lambda: defaultdict(int))
        total_visits = 0
        
        # Get network stats at start of episode
        network_stats = agent.get_network_stats()
        
        # Store current network stats
        for key in ['lower_x_nodes', 'lower_y_nodes', 'higher_nodes', 'higher_connections']:
            training_stats[key].append(network_stats[key])
        
        # Episode loop
        while current_state != goal and step_counter < 10000:  # Limiting to 10000 steps for efficiency
            step_counter += 1
            
            # Get the true state
            true_state = np.array(current_state)
            
            # Add noise to observation with some probability if noise_level > 0
            noise_probability = 0.5  # Probability of adding noise (as in example.py)
            if noise_level > 0 and np.random.uniform(0, 1) < noise_probability:
                noisy_state = true_state + np.random.normal(0, np.sqrt(noise_level), 2)
            else:
                noisy_state = true_state.copy()
            
            # Get firing pattern and BMUs for the true state (for stability analysis)
            _, _, true_bmus = agent.get_firing_pattern_with_bmus(true_state)
            
            # Get firing pattern and BMUs for the noisy state
            _, _, noisy_bmus = agent.get_firing_pattern_with_bmus(noisy_state)
            
            # Check stability: are BMUs different between true and noisy state?
            if true_bmus != noisy_bmus:
                bmu_changes += 1
            bmu_observations += 1
            
            # Track state-to-node mapping for purity calculation
            state_tuple = tuple(true_state)
            
            # Get higher-level node index
            pattern = agent.get_firing_pattern(noisy_state)
            node_idx = agent.find_node_index(pattern)
            
            if node_idx is None:
                # Pattern doesn't exist yet, will be assigned next node index
                node_idx = len(agent.nodes)
            
            # Update state-node mapping counts
            state_node_mappings[node_idx][state_tuple] += 1
            total_visits += 1
            
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
                print(f"Episode {episode_num + 1}, step {step_counter}, Learning rate = {learning_rate}, Noise level = {noise_level}")
            
            # Check if goal reached
            if current_state == goal:
                episode_success = True
                break
        
        # Update epsilon based on success
        if episode_success:
            reached_goal_count += 1
            if reached_goal_count > 5:  # Decay epsilon sooner for faster adaptation
                agent.decay_epsilon(min_epsilon=0.1)
        
        # Calculate stability as percentage of BMUs that remain the same
        bmu_stability = 100 * (1 - (bmu_changes / max(1, bmu_observations)))
        
        # Calculate state-node mapping purity
        if total_visits > 0:
            purity = calculate_purity(state_node_mappings, total_visits)
        else:
            purity = 0.0
        
        # Store episode stats
        training_stats['episodes'].append(episode_num + 1)
        training_stats['steps'].append(step_counter)
        training_stats['epsilon'].append(agent.get_epsilon())
        training_stats['success'].append(episode_success)
        training_stats['bmu_stability'].append(bmu_stability)
        training_stats['purity'].append(purity)
        
        if verbose:
            print(f"Episode: {episode_num + 1}, Steps: {step_counter}, "
                  f"Epsilon: {agent.get_epsilon():.2f}, Success: {episode_success}")
            print(f"BMU Stability: {bmu_stability:.2f}%, Purity: {purity:.2f}%")
        
        if verbose:
            print(f"End of Episode {episode_num + 1}")
            agent.print_node_weights()
    
    # Get final network stats
    final_stats = agent.get_network_stats()
    
    return agent, training_stats, final_stats

def run_noise_learning_rate_study(learning_rates, noise_levels, num_episodes=15):
    """
    Run a comprehensive study of the effect of both learning rates and noise on GWRSOM
    
    Parameters:
    - learning_rates: List of learning rates to test
    - noise_levels: List of noise levels to test
    - num_episodes: Number of episodes per experiment
    
    Returns:
    - results: Dictionary mapping (learning_rate, noise_level) tuples to experiment results
    """
    results = {}
    
    # Run all combinations of learning rates and noise levels
    print(f"Running {len(learning_rates) * len(noise_levels)} experiments...")
    
    # Use tqdm for a progress bar
    total_experiments = len(learning_rates) * len(noise_levels)
    experiment_count = 0
    
    for lr in learning_rates:
        for noise in noise_levels:
            experiment_count += 1
            print(f"\nExperiment {experiment_count}/{total_experiments}: Learning Rate = {lr}, Noise Level = {noise}")
            
            agent, stats, final_stats = run_single_noise_experiment(
                learning_rate=lr,
                noise_level=noise,
                num_episodes=num_episodes,
                verbose=True
            )
            results[(lr, noise)] = (agent, stats, final_stats)
    
    return results

def analyze_noise_learning_rate_results(results):
    """
    Analyze and visualize the results of the learning rate and noise study
    
    Parameters:
    - results: Dictionary mapping (learning_rate, noise_level) to experiment results
    """
    # Prepare data for visualization
    data = []
    for (lr, noise), (_, stats, _) in results.items():
        # Extract final episode values for key metrics
        final_idx = -1  # Last episode
        
        # Check if we have data for this experiment 
        if len(stats['episodes']) > 0:
            data.append({
                'learning_rate': lr,
                'noise_level': noise,
                'lower_x_nodes': stats['lower_x_nodes'][final_idx],
                'lower_y_nodes': stats['lower_y_nodes'][final_idx],
                'higher_nodes': stats['higher_nodes'][final_idx],
                'higher_connections': stats['higher_connections'][final_idx],
                'steps': stats['steps'][final_idx],
                'success': stats['success'][final_idx],
                'bmu_stability': stats['bmu_stability'][final_idx] if 'bmu_stability' in stats else 0,
                'purity': stats['purity'][final_idx] if 'purity' in stats else 0
            })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)

     # 5. Heatmap of Higher-Level Nodes by Learning Rate and Noise Level
    plt.figure(figsize=(12, 8))
    heatmap_data = df.pivot_table(index='learning_rate', columns='noise_level', values='higher_nodes')
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='g')
    plt.title('Number of Higher-Level Network Nodes by Learning Rate and Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Learning Rate')
    plt.tight_layout()
    plt.savefig('higher_nodes_heatmap.png')
    plt.show()
    return df
   
   
    
    

def main():
    """Main function to run the learning rate and noise study"""
    start_time = time.time()
    
    # Define learning rates to test (covering a wider range)
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    
    # Define noise levels to test (similar to example.py)
    noise_levels = [0, 1/6, 1/3, 1/2]
    
    # Run experiments
    results = run_noise_learning_rate_study(
        learning_rates=learning_rates,
        noise_levels=noise_levels,
        num_episodes=10
    )
    
    # Analyze and visualize results
    print("Analyzing results...")
    summary_df = analyze_noise_learning_rate_results(results)
    
    # Save data
    try:
        with open('noise_learning_rate_study_results.pkl', 'wb') as f:
            pickle.dump({
                'results': results,
                'summary_df': summary_df
            }, f)
        
        # Save dataframe to CSV for easier analysis
        if not summary_df.empty:
            summary_df.to_csv('noise_learning_rate_summary.csv', index=False)
            
        print(f"Results saved to noise_learning_rate_study_results.pkl")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"Study completed in {elapsed_time:.2f} seconds")
    
    # Print summary of findings
    print("\nSummary of Key Findings:")
    
    # Best and worst configurations
    if not summary_df.empty:
        # For stability
        if 'bmu_stability' in summary_df.columns:
            best_stability = summary_df.loc[summary_df['bmu_stability'].idxmax()]
            print(f"\nBest BMU Stability Configuration:")
            print(f"Learning Rate: {best_stability['learning_rate']}")
            print(f"Noise Level: {best_stability['noise_level']}")
            print(f"Stability: {best_stability['bmu_stability']:.2f}%")
        
        # For node efficiency (fewer nodes is better)
        min_x_nodes = summary_df.loc[summary_df['lower_x_nodes'].idxmin()]
        print(f"\nMost Efficient Lower X Network Configuration:")
        print(f"Learning Rate: {min_x_nodes['learning_rate']}")
        print(f"Noise Level: {min_x_nodes['noise_level']}")
        print(f"X Nodes: {min_x_nodes['lower_x_nodes']}")
        
        # For purity
        if 'purity' in summary_df.columns:
            best_purity = summary_df.loc[summary_df['purity'].idxmax()]
            print(f"\nBest Mapping Purity Configuration:")
            print(f"Learning Rate: {best_purity['learning_rate']}")
            print(f"Noise Level: {best_purity['noise_level']}")
            print(f"Purity: {best_purity['purity']:.2f}%")

if __name__ == "__main__":
    main()