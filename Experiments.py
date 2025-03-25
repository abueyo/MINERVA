import numpy as np
import matplotlib.pyplot as plt
from time import time
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
import random
from Agents.TMGWR_agent import TMGWRAgent
from Agents.HSOM_Agent import HierarchicalGWRSOMAgent


def run_comparison_experiment(num_episodes=10, num_trials=5):
    # Store results for both algorithms
    results = {
        'TMGWR': {
            'steps': [],
            'success_rates': [],
            'nodes': [],
            'time': []
        },
        'MINERVA': {
            'steps': [],
            'success_rates': [],
            'nodes': [],
            'time': []
        }
    }
     
    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials}")
        
        # Get maze environment
        maze_map, player_pos, goal_pos = MazeMaps.get_default_map()
        
        # Run TMGWR
        tmgwr_maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos, goal_index_pos=goal_pos)
        tmgwr_agent = TMGWRAgent(nDim=2, Ni=10, epsilon_b=0.35, epsilon_n=0.15, 
                                beta=0.7, delta=0.79, T_max=20, N_max=100, 
                                eta=0.5, phi=0.9, sigma=0.5)
        tmgwr_stats = run_single_experiment(tmgwr_agent, tmgwr_maze, num_episodes)
        
        # Run MINERVA with pre-training
        minerva_maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos, goal_index_pos=goal_pos)
        minerva_agent = HierarchicalGWRSOMAgent(lower_dim=1, higher_dim=2, epsilon_b=0.35, 
                                               epsilon_n=0.15, beta=0.7, delta=0.79, T_max=20, 
                                               N_max=100, eta=0.5, phi=0.9, sigma=0.5)
        
        # Generate training data using actual exploration
        training_data = []
        minerva_maze.reset_player()
        current_state = minerva_maze.get_initial_player_pos()
        
        # Reshape training data to match expected format
        for _ in range(50):
            action = random.randint(0, 3)
            minerva_maze.move_player(action)
            state = minerva_maze.get_player_pos()
            # Convert state to tuple so it's hashable
            training_data.append(tuple(state))
            # Split x and y coordinates
           
            
            if len(training_data) % 10 == 0:
                minerva_maze.reset_player()
                current_state = minerva_maze.get_initial_player_pos()
        # Convert to array only for training
        training_array = np.array(training_data)
        minerva_agent.train_lower_networks(training_array, epochs=100)
        minerva_stats = run_single_experiment(minerva_agent, minerva_maze, num_episodes)
        # Train each network separately
        x_data = np.vstack([d[0] for d in training_data])
        y_data = np.vstack([d[1] for d in training_data])
        
        minerva_agent.lower_x.initialize(x_data)  # Initialize first
        minerva_agent.lower_y.initialize(y_data)
        
        minerva_agent.train_lower_networks(np.array(training_data), epochs=100)
        minerva_stats = run_single_experiment(minerva_agent, minerva_maze, num_episodes)
        
        # Store results
        for algo, stats in [('TMGWR', tmgwr_stats), ('MINERVA', minerva_stats)]:
            results[algo]['steps'].append(stats['steps'])
            results[algo]['success_rates'].append(stats['success'])
            results[algo]['nodes'].append(stats['nodes'])
            results[algo]['time'].append(stats['time'])
    
    plot_comparison_results(results, num_episodes)
    return results

def run_single_experiment(agent, maze, num_episodes):
    # Set goal for the agent
    goal = maze.get_goal_pos()
    agent.set_goal(goal)
    agent.set_epsilon(1)  # Start with exploration
    
    stats = {
        'steps': [],
        'success': [],
        'nodes': [],
        'time': []
    }
    
    reached_goal_count = 0  # Track successful episodes
    
    for episode in range(num_episodes):
        start_time = time()
        current_state = maze.get_initial_player_pos()
        maze.reset_player()
        step_count = 0
        success = False
        
        print(f"Episode {episode + 1}, Epsilon: {agent.get_epsilon()}")
        
        while step_count < 1000 and not success:  # Add success check
            action = agent.select_action(current_state)
            maze.move_player(action)
            next_state = maze.get_player_pos()
            
            # Train the model
            agent.update_model(next_state=next_state, action=action)
            
            # Update state
            current_state = next_state
            step_count += 1
            
            # Check for goal
            if current_state == maze.get_goal_pos():
                success = True
                reached_goal_count += 1
                # Decay epsilon after some successes
                if reached_goal_count > 10:
                    agent.decay_epsilon(min_epsilon=0.2)
                break
        
        episode_time = time() - start_time
        stats['steps'].append(step_count)
        stats['success'].append(success)
        stats['nodes'].append(len(agent.model.W) if hasattr(agent.model, 'W') else 
                            sum(len(n.A) for n in [agent.lower_x, agent.lower_y]))
        stats['time'].append(episode_time)
        
        print(f"Episode completed: Steps = {step_count}, Success = {success}")
        
    return stats
def plot_comparison_results(results, num_episodes):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot steps per episode
    ax = axes[0,0]
    for algo in ['TMGWR', 'MINERVA']:
        mean_steps = np.mean(results[algo]['steps'], axis=0)
        std_steps = np.std(results[algo]['steps'], axis=0)
        ax.plot(range(num_episodes), mean_steps, label=algo)
        ax.fill_between(range(num_episodes), mean_steps-std_steps, mean_steps+std_steps, alpha=0.2)
    ax.set_title('Steps per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend()
    
    # Plot success rate
    ax = axes[0,1]
    window = 10
    for algo in ['TMGWR', 'MINERVA']:
        success_rate = [np.mean(results[algo]['success'][i], axis=0) for i in range(len(results[algo]['success']))]
        mean_rate = np.mean(success_rate, axis=0)
        std_rate = np.std(success_rate, axis=0)
        ax.plot(range(num_episodes), mean_rate, label=algo)
        ax.fill_between(range(num_episodes), mean_rate-std_rate, mean_rate+std_rate, alpha=0.2)
    ax.set_title('Success Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.legend()
    
    # Plot node growth
    ax = axes[1,0]
    for algo in ['TMGWR', 'MINERVA']:
        mean_nodes = np.mean(results[algo]['nodes'], axis=0)
        std_nodes = np.std(results[algo]['nodes'], axis=0)
        ax.plot(range(num_episodes), mean_nodes, label=algo)
        ax.fill_between(range(num_episodes), mean_nodes-std_nodes, mean_nodes+std_nodes, alpha=0.2)
    ax.set_title('Network Growth')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Number of Nodes')
    ax.legend()
    
    # Plot computation time
    ax = axes[1,1]
    for algo in ['TMGWR', 'MINERVA']:
        mean_time = np.mean(results[algo]['time'], axis=0)
        std_time = np.std(results[algo]['time'], axis=0)
        ax.plot(range(num_episodes), mean_time, label=algo)
        ax.fill_between(range(num_episodes), mean_time-std_time, mean_time+std_time, alpha=0.2)
    ax.set_title('Computation Time per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Time (seconds)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = run_comparison_experiment()