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

def run_hierarchical_simulation(noise_level=0, num_episodes=20):
    #get the maze details 
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map() 

    #create the maze player
    Maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos_index, goal_index_pos=goal_pos_index)

    #get the goal in screen coordinates
    goal = Maze.get_goal_pos()

    #get player initial position 
    initial_state = Maze.get_initial_player_pos()

    # Generate training data by exploring the maze
    training_data = []
    for _ in range(50):  # Generate some exploration data
        state = [np.random.randint(-72, 72), np.random.randint(-72, 72)]
        training_data.append(state)
    training_data = np.array(training_data)

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

        while current_state != goal and step_counter < 20000:
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
                HGWRSOM_agent.decay_epsilon(min_epsilon=0.2) 

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