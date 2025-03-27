import numpy as np
import matplotlib.pyplot as plt
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.TMGWR_agent import TMGWRAgent
from Agents.HSOM_binary import HierarchicalGWRSOMAgent
from tqdm import tqdm

def add_noise(state, noise_level):
    """Add Gaussian noise to state"""
    noise = np.random.normal(0, np.sqrt(noise_level), size=len(state))
    return state + noise

def run_tmgwr_experiment(maze, noise_levels, num_episodes=20, max_steps=1000):
    """Run experiments for TMGWR agent across different noise levels"""
    results = {}
    
    for noise in noise_levels:
        print(f"\nRunning TMGWR experiments with noise level σ² = {noise}")
        
        # Initialize TMGWR agent
        tmgwr_agent = TMGWRAgent(nDim=2, Ni=10, epsilon_b=0.35, epsilon_n=0.15, 
                                beta=0.7, delta=0.79, T_max=20, N_max=100, 
                                eta=0.5, phi=0.9, sigma=0.5)
        tmgwr_agent.set_goal(maze.get_goal_pos())
        
        # Add epsilon for exploration
        tmgwr_agent.epsilon = 1.0  # Start with full exploration
        
        # Track number of nodes and other statistics
        tmgwr_nodes = []
        reached_goal_count = 0
        
        # Run episodes
        for episode in tqdm(range(num_episodes), desc=f"TMGWR Episodes (σ² = {noise})"):
            current_state = maze.get_initial_player_pos()
            maze.reset_player()
            step_count = 0
            episode_success = False
            
            while step_count < max_steps and not episode_success:
                step_count += 1
                noisy_state = add_noise(current_state, noise)
                
                # Use epsilon for exploration
                if np.random.random() > tmgwr_agent.epsilon:
                    action = tmgwr_agent.select_action(noisy_state)
                else:
                    # Random action for exploration
                    action = np.random.randint(0, 4)  # Assuming 4 possible actions
                
                maze.move_player(action)
                next_state = maze.get_player_pos()
                tmgwr_agent.update_model(next_state, action)
                current_state = next_state
                
                if current_state == maze.get_goal_pos():
                    episode_success = True
                    reached_goal_count += 1
                    # Decay epsilon after some successes
                    if reached_goal_count > 10:
                        tmgwr_agent.epsilon = max(0.2, tmgwr_agent.epsilon * 0.95)  # Decay factor of 0.95
                    break
            
            # Get number of nodes from weight matrix rows
            num_nodes = tmgwr_agent.model.W.shape[0]
            tmgwr_nodes.append(num_nodes)
            
            if (episode + 1) % 5 == 0:  # Print progress every 5 episodes
                print(f"Episode: {episode + 1}, Steps: {step_count}, Nodes: {num_nodes}, "
                      f"Epsilon: {tmgwr_agent.epsilon:.3f}, Success: {episode_success}")
        
        results[noise] = {
            'nodes_per_episode': tmgwr_nodes,
            'final_nodes': tmgwr_agent.model.W.shape[0],
            'success_rate': reached_goal_count / num_episodes
        }
    
    return results

def run_minerva_experiment(maze, noise_levels, num_episodes=20, max_steps=1000):
    """Run experiments for MINERVA agent across different noise levels"""
    results = {}
    
    print("Starting MINERVA experiment setup...")
    
    # Generate training data
    print("\nGenerating training data for MINERVA...")
    training_data = []
    for _ in range(50):
        state = [np.random.randint(-72, 72), np.random.randint(-72, 72)]
        training_data.append(state)
    training_data = np.array(training_data)
    
    for noise in noise_levels:
        print(f"\nInitializing experiment for noise level σ² = {noise}")
        
        try:
            # Initialize MINERVA agent
            print("Creating MINERVA agent...")
            minerva_agent = HierarchicalGWRSOMAgent(
                lower_dim=1,
                higher_dim=2,
                epsilon_b=0.35,
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
            print("Training MINERVA's lower networks...")
            minerva_agent.train_lower_networks(training_data, epochs=100)
            
            # Set goal and initial epsilon
            print("Setting goal and epsilon...")
            minerva_agent.set_goal(maze.get_goal_pos())
            minerva_agent.set_epsilon(1)
            
            # Track node counts and other statistics
            minerva_nodes = []
            reached_goal_count = 0
            
            print(f"Starting episodes for noise level {noise}...")
            # Run episodes
            for episode in tqdm(range(num_episodes), desc=f"MINERVA Episodes (σ² = {noise})"):
                current_state = maze.get_initial_player_pos()
                maze.reset_player()
                step_count = 0
                episode_success = False
                
                while step_count < max_steps and not episode_success:
                    step_count += 1
                    noisy_state = add_noise(current_state, noise)
                    
                    action = minerva_agent.select_action(noisy_state)
                    maze.move_player(action)
                    next_state = maze.get_player_pos()
                    minerva_agent.update_model(next_state, action)
                    current_state = next_state
                    
                    if current_state == maze.get_goal_pos():
                        episode_success = True
                        reached_goal_count += 1
                        if reached_goal_count > 10:
                            minerva_agent.decay_epsilon(min_epsilon=0.2)
                        break
                
                # Get both node counts for comparison
                nodes_list_count = len(minerva_agent.nodes)
                positions_count = len(minerva_agent.node_positions)
                
                # Use nodes list count as it represents actual nodes
                minerva_nodes.append(nodes_list_count)
                
                if (episode + 1) % 5 == 0:  # Print progress every 5 episodes
                    print(f"Episode: {episode + 1}, Steps: {step_count}, "
                          f"Nodes List Count: {nodes_list_count}, "
                          f"Positions Count: {positions_count}, "
                          f"Epsilon: {minerva_agent.get_epsilon()}, Success: {episode_success}")
            
            print(f"Completed all episodes for noise level {noise}")
            results[noise] = {
                'nodes_per_episode': minerva_nodes,
                'final_nodes': len(minerva_agent.nodes),  # Use nodes list count
                'success_rate': reached_goal_count / num_episodes
            }
            
        except Exception as e:
            print(f"Error during execution for noise level {noise}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise
    
    print("MINERVA experiment completed successfully")
    return results

def plot_comparison_results(tmgwr_results, minerva_results, noise_levels):
    """Plot comparison of node counts between TMGWR and MINERVA"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for box plots
    tmgwr_data = []
    minerva_data = []
    
    # Collect node counts for each noise level
    for noise in noise_levels:
        tmgwr_data.append(tmgwr_results[noise]['nodes_per_episode'])
        minerva_data.append(minerva_results[noise]['nodes_per_episode'])
    
    # Position for box plots
    positions = np.arange(len(noise_levels)) * 3
    width = 0.8
    
    # Create box plots
    bp1 = ax.boxplot(tmgwr_data, positions=positions - width/2, widths=width,
                     patch_artist=True, medianprops=dict(color="black"),
                     boxprops=dict(facecolor="lightblue"))
    bp2 = ax.boxplot(minerva_data, positions=positions + width/2, widths=width,
                     patch_artist=True, medianprops=dict(color="black"),
                     boxprops=dict(facecolor="lightcoral"))
    
    # Customize plot
    ax.set_xlabel('Noise Level (σ²)')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Distribution of Node Counts Across Episodes for Different Noise Levels')
    
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{noise:.2f}' for noise in noise_levels])
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['TMGWR', 'MINERVA'],
              loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define reduced noise levels for faster testing
    noise_levels = [0, 1/3, 2/3, 1]  # Reduced from 9 to 4 levels
    
    # Initialize maze environment
    maze_map, player_pos, goal_pos = MazeMaps.get_default_map()
    maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos, goal_index_pos=goal_pos)
    
    # Run experiments separately
    print("\n=== Starting TMGWR experiments ===")
    tmgwr_results = run_tmgwr_experiment(maze, noise_levels)
    
    print("\n=== TMGWR Results Summary ===")
    for noise in noise_levels:
        print(f"\nNoise Level σ² = {noise}:")
        print(f"Final number of nodes: {tmgwr_results[noise]['final_nodes']}")
        print(f"Success rate: {tmgwr_results[noise]['success_rate']:.2%}")
        print(f"Average nodes per episode: {np.mean(tmgwr_results[noise]['nodes_per_episode']):.1f}")
    
    print("\n=== Starting MINERVA experiments ===")
    minerva_results = run_minerva_experiment(maze, noise_levels)
    
    print("\n=== MINERVA Results Summary ===")
    for noise in noise_levels:
        print(f"\nNoise Level σ² = {noise}:")
        print(f"Final number of nodes: {minerva_results[noise]['final_nodes']}")
        print(f"Success rate: {minerva_results[noise]['success_rate']:.2%}")
        print(f"Average nodes per episode: {np.mean(minerva_results[noise]['nodes_per_episode']):.1f}")
    
    # Plot comparison
    print("\n=== Creating Comparison Plot ===")
    plot_comparison_results(tmgwr_results, minerva_results, noise_levels)
    
    # Print comparative analysis
    print("\n=== Comparative Analysis ===")
    for noise in noise_levels:
        print(f"\nNoise Level σ² = {noise}:")
        tmgwr_nodes = tmgwr_results[noise]['final_nodes']
        minerva_nodes = minerva_results[noise]['final_nodes']
        tmgwr_success = tmgwr_results[noise]['success_rate']
        minerva_success = minerva_results[noise]['success_rate']
        
        print(f"TMGWR:   {tmgwr_nodes} nodes, {tmgwr_success:.2%} success rate")
        print(f"MINERVA: {minerva_nodes} nodes, {minerva_success:.2%} success rate")