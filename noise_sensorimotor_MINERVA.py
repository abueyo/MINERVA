#import the maze
from Maze.Mazes import MazeMaps
#import the maze player 
from Maze.Maze_player import MazePlayer
#import HierarchicalGWRSOM agent 
from Agents.HSOM_binary import HierarchicalGWRSOMAgent
import time 
import os                                                                              
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def collect_maze_positions(maze_map, player_pos_index, goal_pos_index, 
                           exploration_steps=10000, save_path="maze_positions.csv"):
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
    # Load data from CSV
    df = pd.read_csv(csv_path)
    
    # Convert to NumPy array
    training_data = df.values
    
    print(f"Loaded {len(training_data)} training samples from {csv_path}")
    print(f"Data range: X[{training_data[:, 0].min():.2f}, {training_data[:, 0].max():.2f}], "
          f"Y[{training_data[:, 1].min():.2f}, {training_data[:, 1].max():.2f}]")
    
    return training_data

def run_hierarchical_simulation(noise_level=0, num_episodes=15):
    #get the maze details 
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map() 

    #create the maze player
    Maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos_index, goal_index_pos=goal_pos_index)

    #get the goal in screen coordinates
    goal = Maze.get_goal_pos()

    #get player initial position 
    initial_state = Maze.get_initial_player_pos()

    # Generate training data by exploring the maze
    maze_positions_path = "maze_positions.csv"
    
    # Collect maze positions through pre-exploration (will be skipped if file exists)
    collect_maze_positions(
        maze_map, player_pos_index, goal_pos_index,
        exploration_steps=50000, save_path=maze_positions_path
    )
    
    # Load the collected maze positions for training
    training_data = load_and_prepare_training_data(maze_positions_path)

    #initialize the hierarchical agent 
    HGWRSOM_agent = HierarchicalGWRSOMAgent(
        lower_dim=1,  # Each coordinate handled separately at lower level
        higher_dim=2, # Full 2D position at higher level
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

    # Train lower networks first
    HGWRSOM_agent.train_lower_networks(training_data, epochs=20)

    #set a goal 
    HGWRSOM_agent.set_goal(goal=goal)
    HGWRSOM_agent.set_epsilon(1)

    #track training statistics
    training_stats = {
        'episodes': [],
        'steps': [],
        'epsilon': [],
        'success': []
    }

    #set the current state to the initial state
    current_state = initial_state

    #track the number of times the goal has been reached to decay epsilon 
    reached_goal_count = 0

    print(f"\nStarting training with noise level σ² = {noise_level}")

    #start the learning loop 
    for episode_num in range(num_episodes):     
        current_state = initial_state
        Maze.reset_player() 
        step_counter = 0
        episode_success = False

        while current_state != goal and step_counter < 10000:
            step_counter += 1

            # Add noise to state observation if noise_level > 0
            if noise_level > 0:
                noisy_state = current_state + np.random.normal(0, np.sqrt(noise_level), 2)
            else:
                noisy_state = current_state

            # Select action using noisy state
            action = HGWRSOM_agent.select_action(current_state=noisy_state)

            # Execute action
            Maze.move_player(action=action)
            next_state = Maze.get_player_pos() 

            # Update model with true next state
            HGWRSOM_agent.update_model(next_state=next_state, action=action)
            current_state = next_state
            
            if step_counter % 100 == 0: 
                print(f"Episode {episode_num + 1}, step {step_counter}, noise σ² = {noise_level}")

            if current_state == goal:
                episode_success = True
                break

        if episode_success:
            reached_goal_count += 1
            if reached_goal_count > 10: 
                HGWRSOM_agent.decay_epsilon(min_epsilon=0.1) 

        training_stats['episodes'].append(episode_num + 1)
        training_stats['steps'].append(step_counter)
        training_stats['epsilon'].append(HGWRSOM_agent.get_epsilon())
        training_stats['success'].append(episode_success)

        print(f"Episode: {episode_num + 1}, Steps: {step_counter}, "
              f"Epsilon: {HGWRSOM_agent.get_epsilon()}, Success: {episode_success}\n")

    return HGWRSOM_agent, training_stats

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run without noise
    print("\nRunning experiment without noise...")
    agent_no_noise, stats_no_noise = run_hierarchical_simulation(noise_level=0)
    print("\nShowing map without noise:")
    agent_no_noise.show_map()

    # Run with noise
    print("\nRunning experiment with noise σ² = 1/6...")
    agent_with_noise, stats_with_noise = run_hierarchical_simulation(noise_level=1/6)
    print("\nShowing map with noise:")
    agent_with_noise.show_map()

    # Plot training results for both conditions
    plt.figure(figsize=(15, 5))
    
    # Plot steps per episode
    plt.plot(stats_no_noise['episodes'], stats_no_noise['steps'], 'b-', label='No Noise')
    plt.plot(stats_with_noise['episodes'], stats_with_noise['steps'], 'r-', label='With Noise')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True)

    # # Plot success rate
    # plt.subplot(1, 2, 2)
    # window_size = 5
    # success_rate_no_noise = [
    #     sum(stats_no_noise['success'][max(0, i-window_size):i])/min(i, window_size)
    #     for i in range(1, len(stats_no_noise['success']) + 1)
    # ]
    # success_rate_with_noise = [
    #     sum(stats_with_noise['success'][max(0, i-window_size):i])/min(i, window_size)
    #     for i in range(1, len(stats_with_noise['success']) + 1)
    # ]
    # plt.plot(stats_no_noise['episodes'], success_rate_no_noise, 'b-', label='No Noise')
    # plt.plot(stats_with_noise['episodes'], success_rate_with_noise, 'r-', label='With Noise')
    # plt.xlabel('Episode')
    # plt.ylabel('Success Rate')
    # plt.title(f'MINERVA Success Rate (Moving Average, Window={window_size})')
    # plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.show()