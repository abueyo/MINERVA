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

# Change the current working directory to the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

def run_hierarchical_simulation(num_episodes=5, slow_episode=1, load_model=False, 
                              save_model=True, show_map=True, model_path="Data/HGWRSOM/model4.npz"):
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
    HGWRSOM_agent.train_lower_networks(training_data, epochs=100)

    #set a goal 
    HGWRSOM_agent.set_goal(goal=goal)
    HGWRSOM_agent.set_epsilon(1)


    #load model if requested
    if load_model: 
        HGWRSOM_agent.load_model(model_path)

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

    #start the learning loop 
    for episode_num in range(num_episodes):     
        #set the current state to the initial state
        current_state = initial_state
        #move the player to the initial position 
        Maze.reset_player() 

        #step counter
        step_counter = 0
        episode_success = False

        #while not in terminal state
        while current_state != goal and step_counter < 20000: # Added step limit
            step_counter += 1

            #take an action 
            action = HGWRSOM_agent.select_action(current_state=current_state)

            #move the player 
            Maze.move_player(action=action)

            #get the next state
            next_state = Maze.get_player_pos() 

            #update the model 
            HGWRSOM_agent.update_model(next_state=next_state, action=action)

            #update current state 
            current_state = next_state

            #update the maze 
            Maze.update_screen() 
            
            #printing progress every 100 steps
            if step_counter % 100 == 0: 
                print(f"Episode number: {episode_num + 1} step number: {step_counter}")

            #slow down action for visualization
            if episode_num == slow_episode: 
                time.sleep(0.25)

            #check if goal reached
            if current_state == goal:
                episode_success = True
                break

        #reached goal
        if episode_success:
            reached_goal_count += 1
            #decay epsilon after some successes
            if reached_goal_count > 10: 
                HGWRSOM_agent.decay_epsilon(min_epsilon=0.2) 

        #record statistics
        training_stats['episodes'].append(episode_num + 1)
        training_stats['steps'].append(step_counter)
        training_stats['epsilon'].append(HGWRSOM_agent.get_epsilon())
        training_stats['success'].append(episode_success)

        print(f"Episode: {episode_num + 1}, Steps: {step_counter}, "
              f"Epsilon: {HGWRSOM_agent.get_epsilon()}, Success: {episode_success}\n")

        if show_map: 
            #show the learned map
            HGWRSOM_agent.show_map() 

    #save the model if requested
    if save_model: 
        HGWRSOM_agent.save_model("Data/MINERVA/modell.npz")

    return HGWRSOM_agent, training_stats

def plot_training_results(training_stats):
    """Plot training statistics."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot steps per episode
    ax1.plot(training_stats['episodes'], training_stats['steps'], 'b-')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps')
    ax1.set_title('Steps per Episode')
    ax1.grid(True)

    # Plot epsilon decay
    ax2.plot(training_stats['episodes'], training_stats['epsilon'], 'r-')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Epsilon Decay')
    ax2.grid(True)

    # Plot success rate (moving average)
    window_size = 10
    success_rate = [
        sum(training_stats['success'][max(0, i-window_size):i])/min(i, window_size)
        for i in range(1, len(training_stats['success']) + 1)
    ]
    ax3.plot(training_stats['episodes'], success_rate, 'g-')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.set_title(f'Success Rate (Moving Average, Window={window_size})')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run simulation
    print("Starting Hierarchical GWRSOM simulation...")
    agent, stats = run_hierarchical_simulation(
        num_episodes=20,
        slow_episode=1,
        load_model=False,  # Start with fresh model
        save_model=True,   # Save final model
        show_map=True
    )

    # Plot training results
    plot_training_results(stats)

    # Show final learned map
    print("\nFinal learned map:")
    agent.show_map()