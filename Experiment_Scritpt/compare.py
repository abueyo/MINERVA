import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.TMGWR_agent import TMGWRAgent
from Agents.se_reduced_grwrsom import SEReducedHGWRSOM
import time
import os

def calculate_purity(state_node_mappings, total_visits):
    """Calculate purity metric for both single and dual firing patterns"""
    if total_visits == 0:
        return 0
    
    total_purity = 0
    for node, state_counts in state_node_mappings.items():
        # For MINERVA, need to consider both active units
        if isinstance(node, tuple) and isinstance(node[0], tuple):
            # MINERVA pattern
            x_pattern, y_pattern = node
            # Find states that share either x or y BMUs
            related_states = sum(max(state_counts.values()) for state in state_counts)
            total_purity += related_states
        else:
            # TMGWR pattern
            max_state_count = max(state_counts.values()) if state_counts else 0
            total_purity += max_state_count
    
    return (total_purity / total_visits) * 100

def calculate_se(transitions, agent, tau=0.1):
    if not transitions:
        return 0
    
    error_count = 0
    E = len(transitions)
    
    for prev_state, curr_state in transitions:
        # For SEReducedHGWRSOM, use the dedicated method
        if hasattr(agent, 'is_habituated'):
            is_habituated = agent.is_habituated(prev_state, curr_state)
        elif isinstance(agent, TMGWRAgent):
            # Original TMGWR calculation 
            prev_node = agent.model.get_node_index(prev_state)
            curr_node = agent.model.get_node_index(curr_state)
            is_habituated = agent.model.C[prev_node, curr_node] == 1
        else:
            # Fallback for other agent types
            is_habituated = False
            
        if not is_habituated:
            error_count += 1

    SE = (error_count + tau) / (E + tau)
    return SE

def run_noise_comparison(noise_levels=[0, 1/6, 1/3, 1/2, 2/3, 5/6, 1, 7/6, 4/3], episodes_per_noise=5):
    # Initialize results storage
    results = {
        'TMGWR': {level: {'nodes': [], 'purity': [], 'se': []} for level in noise_levels},
        'MINERVA': {level: {'nodes': [], 'purity': [], 'se': []} for level in noise_levels}
    }

    # Get maze details
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()

    # Smaller, simpler training set for HGWRSOM
    x_train = np.linspace(-72, 72, 10).reshape(-1, 1)
    y_train = np.linspace(-72, 72, 10).reshape(-1, 1)
    training_data = np.hstack((x_train, y_train))

    for noise_level in noise_levels:
        print(f"\nRunning simulations with noise level σ² = {noise_level}")
        
        for episode in range(episodes_per_noise):
            print(f"\nEpisode {episode + 1}/{episodes_per_noise}")
            
            # Initialize maze
            Maze = MazePlayer(maze_map=maze_map, 
                            player_index_pos=player_pos_index, 
                            goal_index_pos=goal_pos_index)
            
            goal = Maze.get_goal_pos()
            initial_state = Maze.get_initial_player_pos()

            # Initialize TMGWR agent
            tmgwr_agent = TMGWRAgent(
                nDim=2, Ni=2, epsilon_b=0.55194, epsilon_n=0.90,
                beta=0.8, delta=0.6235, T_max=17, N_max=300,
                eta=0.95, phi=0.6, sigma=1
            )
            tmgwr_agent.set_goal(goal)
            tmgwr_agent.set_epsilon(1)

            # Initialize HGWRSOM agent
            hgwrsom_agent = SEReducedHGWRSOM(
                lower_dim=1, higher_dim=2, epsilon_b=0.35,
                epsilon_n=0.15, beta=0.7, delta=0.79,
                T_max=20, N_max=100, eta=0.5,
                phi=0.9, sigma=0.5
            )
            
            # ** KEY CHANGE: Run each agent in a separate try-except block **
            # Run TMGWR agent
            agent_name = 'TMGWR'
            try:
                current_state = initial_state
                Maze.reset_player()
                step_counter = 0
                
                state_node_mappings = defaultdict(lambda: defaultdict(int))
                transitions = []
                total_visits = 0
                
                while current_state != goal and step_counter < 20000:
                    step_counter += 1
                    prev_state = np.array(current_state)
                    
                    # Add noise to current state observation
                    noisy_state = np.array(current_state) + np.random.normal(0, np.sqrt(noise_level), 2)
                    
                    # Get node assignment for current state
                    node_idx = tmgwr_agent.model.get_node_index(noisy_state)
                    
                    # Update state-node mapping counts
                    state_tuple = tuple(current_state)
                    state_node_mappings[node_idx][state_tuple] += 1
                    total_visits += 1
                    
                    # Select and execute action
                    action = tmgwr_agent.select_action(noisy_state)
                    Maze.move_player(action)
                    next_state = Maze.get_player_pos()
                    
                    # Update model with actual next state (no noise)
                    tmgwr_agent.update_model(next_state, action)
                    
                    transitions.append((prev_state, next_state))
                    current_state = next_state
                
                # Calculate metrics
                purity = calculate_purity(state_node_mappings, total_visits)
                se = calculate_se(transitions, tmgwr_agent)
                
                # Record number of nodes
                num_nodes = len(tmgwr_agent.model.W)
                
                # Store results
                results[agent_name][noise_level]['nodes'].append(num_nodes)
                results[agent_name][noise_level]['purity'].append(purity)
                results[agent_name][noise_level]['se'].append(se)
            
            except Exception as e:
                print(f"Error in episode {episode} with noise level {noise_level} for {agent_name}: {str(e)}")
                results[agent_name][noise_level]['nodes'].append(0)
                results[agent_name][noise_level]['purity'].append(0)
                results[agent_name][noise_level]['se'].append(0)
            
            # Run MINERVA agent
            agent_name = 'MINERVA'
            try:
                # Train lower networks first
                # hgwrsom_agent.train_lower_networks(training_data, epochs=10)
                hgwrsom_agent.set_goal(goal)
                hgwrsom_agent.set_epsilon(1)
                
                current_state = initial_state
                Maze.reset_player()
                step_counter = 0
                
                state_node_mappings = defaultdict(lambda: defaultdict(int))
                transitions = []
                total_visits = 0
                
                while current_state != goal and step_counter < 20000:
                    step_counter += 1
                    prev_state = np.array(current_state)
                    
                    # Add noise to current state observation
                    noisy_state = np.array(current_state) + np.random.normal(0, np.sqrt(noise_level), 2)
                    
                    # Get pattern for current state
                    current_pattern = hgwrsom_agent.get_firing_pattern(noisy_state)
                    
                    # Find or create node
                    node_idx = hgwrsom_agent.find_node_index(current_pattern)
                    if node_idx is None:
                        node_idx = hgwrsom_agent.create_node(current_pattern, noisy_state)
                    
                    # Update state-node mapping counts
                    state_tuple = tuple(current_state)
                    state_node_mappings[node_idx][state_tuple] += 1
                    total_visits += 1
                    
                    # Select and execute action
                    action = hgwrsom_agent.select_action(noisy_state)
                    Maze.move_player(action)
                    next_state = Maze.get_player_pos()
                    
                    # Update model with actual next state (no noise)
                    hgwrsom_agent.update_model(next_state, action)
                    
                    transitions.append((prev_state, next_state))
                    current_state = next_state
                
                # Calculate metrics
                purity = calculate_purity(state_node_mappings, total_visits)
                se = calculate_se(transitions, hgwrsom_agent)
                
                # Record number of nodes
                num_nodes = len(hgwrsom_agent.nodes)
                
                # Store results
                results[agent_name][noise_level]['nodes'].append(num_nodes)
                results[agent_name][noise_level]['purity'].append(purity)
                results[agent_name][noise_level]['se'].append(se)
            
            except Exception as e:
                print(f"Error in episode {episode} with noise level {noise_level} for {agent_name}: {str(e)}")
                results[agent_name][noise_level]['nodes'].append(0)
                results[agent_name][noise_level]['purity'].append(0)
                results[agent_name][noise_level]['se'].append(0)

    return results

def plot_comparison_results(results):
    metrics = {
        'nodes': {'title': 'Number of Nodes vs Noise Level', 'ylabel': 'Number of Nodes', 'ylim': (0, 200)},
        'purity': {'title': 'Purity vs Noise Level', 'ylabel': 'Purity (%)', 'ylim': (0, 120)},
        'se': {'title': 'Sensorimotor Error vs Noise Level', 'ylabel': 'SE', 'ylim': (0, 1)}
    }
    
    # Manual mapping for our specific noise levels
    fraction_mapping = {
        0: "0",
        1/6: "1/6",
        1/3: "1/3",
        1/2: "1/2",
        2/3: "2/3",
        5/6: "5/6",
        1: "1",
        7/6: "7/6",
        4/3: "4/3"
    }
    
    for metric in metrics:
        data = []
        for agent in results:
            for noise_level in results[agent]:
                values = results[agent][noise_level][metric]
                for value in values:
                    data.append({
                        'Agent': agent,
                        'Noise Level': fraction_mapping[noise_level],
                        'Value': value,
                        'Metric': metrics[metric]['ylabel'],
                        'Noise_sort': noise_level
                    })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Noise_sort')
        
        plt.figure(figsize=(15, 8))
        
        # Add grid before the plot
        plt.grid(True, linestyle='--', alpha=0.7)
        
        custom_palette = {'TMGWR': 'green', 'MINERVA': 'orange'}
        sns.boxplot(data=df, x='Noise Level', y='Value', hue='Agent', palette=custom_palette)
        
        # Ensure grid is behind the plot
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        plt.title(metrics[metric]['title'], fontsize=14, pad=20)
        plt.xlabel('Noise Level (σ²)', fontsize=12)
        plt.ylabel(metrics[metric]['ylabel'], fontsize=12)
        plt.legend(title='Agent Type', title_fontsize=12, fontsize=10)
        
        if metrics[metric]['ylim']:
            plt.ylim(metrics[metric]['ylim'])
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    print("Starting comparative simulation...")
    results = run_noise_comparison()
    plot_comparison_results(results)

    print("\nSummary Statistics:")
    for agent in results:
        print(f"\n{agent}:")
        for noise_level in results[agent]:
            nodes = results[agent][noise_level]['nodes']
            purity = results[agent][noise_level]['purity']
            se = results[agent][noise_level]['se']
            print(f"Noise Level σ² = {noise_level}:")
            print(f"  Mean nodes: {np.mean(nodes):.2f}")
            print(f"  Std nodes: {np.std(nodes):.2f}")
            print(f"  Mean purity: {np.mean(purity):.2f}%")
            print(f"  Std purity: {np.std(purity):.2f}%")
            print(f"  Mean SE: {np.mean(se):.2f}")
            print(f"  Std SE: {np.std(se):.2f}")