import numpy as np
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.TMGWR_agent import TMGWRAgent
import time 
import os
import matplotlib.pyplot as plt

# Change the current working directory to the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

def run_experiment(noise_level=0, num_episodes=5):
    """Run experiment with specified noise level"""
    # Get maze details 
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()
    Maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos_index, goal_index_pos=goal_pos_index)
    goal = Maze.get_goal_pos()
    initial_state = Maze.get_initial_player_pos()

    # Initialize agent 
    agent = TMGWRAgent(nDim=2, Ni=2, epsilon_b=0.55194, epsilon_n=0.90, beta=0.8, 
                      delta=0.6235, T_max=17, N_max=300, eta=0.95, phi=0.6, sigma=1)
    agent.set_goal(goal=goal)
    agent.set_epsilon(1)

    # Training statistics
    training_stats = {
        'episodes': [],
        'steps': [],
        'epsilon': [],
        'success': []
    }

    # Parameters
    reached_goal_count = 0

    print(f"\nStarting training with noise level σ² = {noise_level}")

    # Training loop
    for episode_num in range(num_episodes):     
        current_state = initial_state
        Maze.reset_player() 
        step_counter = 0
        episode_success = False

        while current_state != goal and step_counter < 700: 
            step_counter += 1

            # Add noise to state observation if noise_level > 0
            if noise_level > 0:
                noisy_state = current_state + np.random.normal(0, np.sqrt(noise_level), 2)
            else:
                noisy_state = current_state

            # Select and execute action
            action = agent.select_action(current_state=noisy_state)
            Maze.move_player(action=action)
            next_state = Maze.get_player_pos() 

            # Update model with true next state
            agent.update_model(next_state=next_state, action=action)
            current_state = next_state
            
            if step_counter % 100 == 0: 
                print(f"Episode {episode_num + 1}, step {step_counter}, noise σ² = {noise_level}")

            # Check if goal reached
            if current_state == goal:
                episode_success = True
                break

        # Update epsilon
        if episode_success:
            reached_goal_count += 1
            if reached_goal_count > 10: 
                agent.decay_epsilon(min_epsilon=0.2) 

        # Record statistics
        training_stats['episodes'].append(episode_num + 1)
        training_stats['steps'].append(step_counter)
        training_stats['epsilon'].append(agent.get_epsilon())
        training_stats['success'].append(episode_success)

        print(f"Episode: {episode_num + 1}, Steps: {step_counter}, "
              f"Epsilon: {agent.get_epsilon()}, Success: {episode_success}\n")

    return agent, training_stats

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Set number of episodes
    num_episodes = 5

    # Run without noise
    print("\nRunning experiment without noise...")
    agent_no_noise, stats_no_noise = run_experiment(noise_level=0, num_episodes=num_episodes)
    print("\nShowing map without noise:")
    agent_no_noise.show_map()

    # Run with noise
    noise_level = 1/6
    print(f"\nRunning experiment with noise σ² = {noise_level}...")
    agent_with_noise, stats_with_noise = run_experiment(noise_level=noise_level, num_episodes=num_episodes)
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

    plt.tight_layout()
    plt.show()