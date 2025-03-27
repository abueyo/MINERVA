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

# Import our custom KeyDoorMazePlayer extension
from key_door_maze_extension import create_key_door_maze

def run_key_door_experiment(runs=3, training_steps=2000, testing_steps=1000):
    """
    Key and Door Experiment: Test how MINERVA and TMGWR handle context-dependent input
    
    Args:
        runs: Number of experiment runs
        training_steps: Number of steps for training phase
        testing_steps: Number of steps for testing phase
    
    Returns:
        results: Dictionary containing success metrics for both algorithms
    """
    # Initialize results storage
    results = {
        'TMGWR': {'success_rate': [], 'steps_to_goal': [], 'key_door_failed': []},
        'MINERVA': {'success_rate': [], 'steps_to_goal': [], 'key_door_failed': []}
    }

    # Get maze details
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()

    # Place door between player and goal - this is crucial for the task
    # Determine suitable door position (adjacent to goal)
    goal_row, goal_col = goal_pos_index
    possible_door_positions = [
        (goal_row - 1, goal_col),  # above goal
        (goal_row + 1, goal_col),  # below goal
        (goal_row, goal_col - 1),  # left of goal
        (goal_row, goal_col + 1)   # right of goal
    ]
    
    # Filter out positions that are walls
    valid_door_positions = [pos for pos in possible_door_positions if pos not in maze_map]
    
    if valid_door_positions:
        door_position = valid_door_positions[0]  # Use first valid position
    else:
        # If all adjacent positions are walls, we need to modify the maze
        # This is a simplification - you might need to adapt this to your specific maze
        door_position = possible_door_positions[0]
        # Remove this position from walls if it's a wall
        if door_position in maze_map:
            maze_map.remove(door_position)
    
    # Define key position far from both player and goal
    key_position = (3, 6)  # Customize this based on your maze
    
    # Make sure key isn't at same position as door or in a wall
    while key_position == door_position or key_position in maze_map:
        key_position = (key_position[0] + 1, key_position[1] + 1)

    # Generate training data for pre-training
    x_train = np.linspace(-72, 72, 10).reshape(-1, 1)
    y_train = np.linspace(-72, 72, 10).reshape(-1, 1)
    training_data = np.hstack((x_train, y_train))

    for run in range(runs):
        print(f"\nRun {run + 1}/{runs}")
        
        # Initialize modified maze environment with key and door
        maze = create_key_door_maze(
            maze_map=maze_map, 
            player_pos_index=player_pos_index, 
            goal_pos_index=goal_pos_index,
            key_position=key_position,
            door_position=door_position,
            display_maze=False  # Set to True to visualize
        )
        
        goal_pos = maze.get_goal_pos()

        # Run experiment for both agents
        for agent_type in ['TMGWR', 'MINERVA']:
            print(f"\nTesting {agent_type}...")
            
            # Initialize proper agent
            if agent_type == 'TMGWR':
                # For TMGWR, we'll use a 2D agent but process the key status separately
                agent = TMGWRAgent(
                    nDim=2, Ni=2, epsilon_b=0.55194, epsilon_n=0.90,
                    beta=0.8, delta=0.6235, T_max=17, N_max=300,
                    eta=0.95, phi=0.6, sigma=1
                )
                # Set the 2D goal for TMGWR
                agent.set_goal(goal_pos)
            else:  # MINERVA
                agent = initialize_minerva_agent(training_data)
                # Set the 3D goal for MINERVA (goal with key=1 since it's optimal)
                enhanced_goal = np.array([goal_pos[0], goal_pos[1], 1.0], dtype=float)
                agent.set_goal(enhanced_goal)
            
            # Training phase with random exploration
            maze.reset_player()
            maze.reset_key_collected()
            
            # Track metrics
            training_goal_reached = 0
            training_key_door_failed = 0
            
            print("Training phase...")
            current_pos = maze.get_player_pos()
            
            for step in range(training_steps):
                # Get current state including key status
                has_key = maze.is_key_collected()
                
                # Random action for exploration
                action = random.randint(0, 3)
                
                # Execute action
                result = maze.move_player_with_key_door(action)
                next_pos = maze.get_player_pos()
                
                # Get enhanced next state for MINERVA
                has_key_next = maze.is_key_collected()
                
                # Update agent's model
                if agent_type == 'TMGWR':
                    # For TMGWR, just use the position part
                    agent.update_model(next_pos, action)
                else:  # MINERVA
                    # For MINERVA, use the enhanced state
                    enhanced_next_state = create_enhanced_state(next_pos, has_key_next)
                    update_minerva_model(agent, enhanced_next_state, action)
                
                # Track door failure
                if result == 'door_locked':
                    training_key_door_failed += 1
                
                # Check if goal reached
                if next_pos == goal_pos:
                    training_goal_reached += 1
                    maze.reset_player()
                    maze.reset_key_collected()
                    next_pos = maze.get_player_pos()
                
                current_pos = next_pos
            
            print(f"Training complete. Goal reached {training_goal_reached} times. Door failures: {training_key_door_failed}")
            
            # Testing phase
            maze.reset_player()
            maze.reset_key_collected()
            
            # Reset exploration for testing
            agent.set_epsilon(0.1)  # Small exploration during testing
            
            # Track metrics
            testing_goal_reached = 0
            testing_steps_to_goal = []
            testing_key_door_failed = 0
            step_count = 0
            
            print("Testing phase...")
            current_pos = maze.get_player_pos()
            
            for step in range(testing_steps):
                # Get current state including key status
                has_key = maze.is_key_collected()
                
                # Select action using agent's policy
                if agent_type == 'TMGWR':
                    # For TMGWR, just use the position part
                    action = agent.select_action(current_pos)
                else:  # MINERVA
                    # For MINERVA, use the enhanced state
                    enhanced_state = create_enhanced_state(current_pos, has_key)
                    action = select_minerva_action(agent, enhanced_state)
                
                # Execute action
                result = maze.move_player_with_key_door(action)
                next_pos = maze.get_player_pos()
                
                # Count steps
                step_count += 1
                
                # Track door failure
                if result == 'door_locked':
                    testing_key_door_failed += 1
                
                # Check if goal reached
                if next_pos == goal_pos:
                    testing_goal_reached += 1
                    testing_steps_to_goal.append(step_count)
                    
                    # Reset for next attempt
                    maze.reset_player()
                    maze.reset_key_collected()
                    next_pos = maze.get_player_pos()
                    step_count = 0
                
                current_pos = next_pos
                
                # Stop if step_count gets too large (prevent infinite loop)
                if step_count > 500:
                    print("Exceeded max steps for a single attempt")
                    maze.reset_player()
                    maze.reset_key_collected()
                    next_pos = maze.get_player_pos()
                    step_count = 0
            
            # Calculate success rate
            attempts = max(1, testing_steps // 500)
            success_rate = (testing_goal_reached / attempts) * 100
            avg_steps_to_goal = sum(testing_steps_to_goal) / max(1, len(testing_steps_to_goal))
            
            # Store results
            results[agent_type]['success_rate'].append(success_rate)
            results[agent_type]['steps_to_goal'].append(avg_steps_to_goal)
            results[agent_type]['key_door_failed'].append(testing_key_door_failed)
            
            print(f"Testing complete. Success rate: {success_rate:.2f}%. Avg steps to goal: {avg_steps_to_goal:.2f}. Door failures: {testing_key_door_failed}")
    
    return results

def create_enhanced_state(position, has_key):
    """
    Create a state vector that includes key status
    
    Args:
        position: [x, y] position
        has_key: Boolean indicating whether key has been collected
    
    Returns:
        enhanced_state: [x, y, key_status] where key_status is 0 or 1
    """
    # Create state with key information
    return np.array([position[0], position[1], 1.0 if has_key else 0.0], dtype=float)

def initialize_minerva_agent(training_data):
    """Initialize MINERVA agent with a key status network"""
    agent = HierarchicalGWRSOMAgent(
        lower_dim=1, higher_dim=3, epsilon_b=0.35,
        epsilon_n=0.15, beta=0.7, delta=0.79,
        T_max=20, N_max=100, eta=0.5,
        phi=0.9, sigma=0.5
    )
    
    # Initialize nodes list and connections matrix
    agent.nodes = []
    agent.connections = np.zeros((0, 0))
    
    # Initialize special key status network
    agent.lower_key = agent.lower_x.__class__(a=0.2, h=0.1)
    
    # Train lower networks
    x_data = training_data[:, 0]
    y_data = training_data[:, 1]
    key_data = np.array([0, 1]).reshape(-1, 1)  # Binary key status
    
    # Regular training for x and y
    for x in x_data:
        agent.lower_x.train(np.array([[x]]), epochs=5)
    for y in y_data:
        agent.lower_y.train(np.array([[y]]), epochs=5)
    
    # Train key network explicitly with binary values
    agent.lower_key.train(key_data, epochs=5)
    
    # Override get_firing_pattern to include key status
    def enhanced_get_pattern(state):
        """Enhanced pattern function that includes key status"""
        x_data = np.array([state[0]]).reshape(1, -1)
        y_data = np.array([state[1]]).reshape(1, -1)
        key_data = np.array([state[2]]).reshape(1, -1) if len(state) > 2 else np.array([0]).reshape(1, -1)
        
        x_bmu = agent.lower_x.find_best_matching_units(x_data)[0]
        y_bmu = agent.lower_y.find_best_matching_units(y_data)[0]
        key_bmu = agent.lower_key.find_best_matching_units(key_data)[0]
        
        x_binary = np.zeros(len(agent.lower_x.A))
        y_binary = np.zeros(len(agent.lower_y.A))
        key_binary = np.zeros(len(agent.lower_key.A))
        
        x_binary[x_bmu] = 1
        y_binary[y_bmu] = 1
        key_binary[key_bmu] = 1
        
        return (tuple(x_binary), tuple(y_binary), tuple(key_binary))
    
    # Replace the method
    agent.get_firing_pattern = enhanced_get_pattern
    
    return agent

def update_minerva_model(agent, next_state, action):
    """Update MINERVA agent with enhanced state"""
    # Get pattern for next state
    current_pattern = agent.get_firing_pattern(next_state)
    current_idx = agent.find_node_index(current_pattern)
    
    if current_idx is None:
        # Create new node
        current_idx = len(agent.nodes)
        agent.nodes.append(current_pattern)
        
        # Store position with key status
        agent.node_positions[current_idx] = next_state
        
        # Expand connection matrices
        new_size = len(agent.nodes)
        new_connections = np.zeros((new_size, new_size))
        new_ages = np.zeros((new_size, new_size))
        
        if new_size > 1:
            new_connections[:-1, :-1] = agent.connections
            new_ages[:-1, :-1] = agent.pattern_ages
            
        agent.connections = new_connections
        agent.pattern_ages = new_ages
    
    # Update connections
    if agent.prev_node_idx is not None:
        agent.connections[agent.prev_node_idx, current_idx] = 1
        agent.pattern_ages[agent.prev_node_idx, current_idx] = 0
        agent.action_mappings[(agent.prev_node_idx, current_idx)] = action
    
    agent.prev_node_idx = current_idx

def select_minerva_action(agent, current_state):
    """Select action for MINERVA agent with enhanced state"""
    if agent.goal is None:
        raise Exception("No goal defined")

    if np.random.uniform(0, 1) > agent.epsilon:
        current_pattern = agent.get_firing_pattern(current_state)
        current_idx = agent.find_node_index(current_pattern)
        
        if current_idx is not None:
            # Get connected nodes
            connected_nodes = np.where(agent.connections[current_idx] == 1)[0]
            
            if len(connected_nodes) > 0:
                # Choose next node based on key status priority
                key_status = current_state[2]
                preferred_nodes = []
                
                for node_idx in connected_nodes:
                    if node_idx in agent.node_positions:
                        node_key_status = agent.node_positions[node_idx][2]
                        
                        # Prioritize nodes that maintain or increase key status
                        if node_key_status >= key_status:
                            preferred_nodes.append(node_idx)
                
                # If no preferred nodes, use all connected nodes
                target_nodes = preferred_nodes if preferred_nodes else connected_nodes
                
                # Choose a node
                next_idx = np.random.choice(target_nodes)
                key = (current_idx, next_idx)
                
                if key in agent.action_mappings:
                    agent.expected_next_node = next_idx
                    agent.is_plan = True
                    return agent.action_mappings[key]
    
    # Default to random exploration
    agent.is_plan = False
    return random.randint(0, 3)

def plot_key_door_results(results):
    """Plot key door experiment results"""
    metrics = {
        'success_rate': {'title': 'Goal Success Rate', 'ylabel': 'Success Rate (%)', 'ylim': (0, 100)},
        'steps_to_goal': {'title': 'Average Steps to Goal', 'ylabel': 'Steps', 'ylim': None},
        'key_door_failed': {'title': 'Number of Door Failures', 'ylabel': 'Failures', 'ylim': None}
    }
    
    plt.figure(figsize=(15, 5))
    
    for i, (metric, config) in enumerate(metrics.items(), 1):
        plt.subplot(1, 3, i)
        
        # Extract data
        tmgwr_data = results['TMGWR'][metric]
        minerva_data = results['MINERVA'][metric]
        
        # Bar positions
        x = np.arange(2)
        width = 0.35
        
        # Plot bars
        plt.bar(x[0], np.mean(tmgwr_data), width, label='TMGWR', color='green', 
                yerr=np.std(tmgwr_data), capsize=10)
        plt.bar(x[1], np.mean(minerva_data), width, label='MINERVA', color='orange', 
                yerr=np.std(minerva_data), capsize=10)
        
        # Add labels and title
        plt.title(config['title'])
        plt.ylabel(config['ylabel'])
        plt.xticks(x, ['TMGWR', 'MINERVA'])
        
        if config['ylim']:
            plt.ylim(config['ylim'])
        
        if i == 1:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    random.seed(42)
    
    # Run the experiment
    print("Starting key and door experiment...")
    results = run_key_door_experiment(runs=3, training_steps=2000, testing_steps=1000)
    
    # Plot results
    plot_key_door_results(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for agent in results:
        print(f"\n{agent}:")
        print(f"  Success Rate: {np.mean(results[agent]['success_rate']):.2f}% ± {np.std(results[agent]['success_rate']):.2f}%")
        print(f"  Steps to Goal: {np.mean(results[agent]['steps_to_goal']):.2f} ± {np.std(results[agent]['steps_to_goal']):.2f}")
        print(f"  Door Failures: {np.mean(results[agent]['key_door_failed']):.2f} ± {np.std(results[agent]['key_door_failed']):.2f}")

        