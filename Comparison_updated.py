import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.TMGWR_agent import TMGWRAgent
from Agents.HSOM_Agent import HierarchicalGWRSOMAgent
import time
import os

def calculate_purity(state_node_mappings, total_visits):
    """Calculate purity metric according to formula: purity=∑n∈Nmaxs∈S|n∩s|/M×100%"""
    if total_visits == 0:
        return 0
    
    total_purity = 0
    for node, state_counts in state_node_mappings.items():
        max_state_count = max(state_counts.values()) if state_counts else 0
        total_purity += max_state_count
    
    return (total_purity / total_visits) * 100

def calculate_se(transitions, agent, tau=0.1):
    """
    Calculate sensorimotor representation error (SE) normalized between 0 and 1
    SE = (∑|E|i=1 I{Ei[w⃗{1}t−1,w⃗{1}t]∉H} + τ) / (|E| + τ)
    """
    if not transitions:
        return 0
    
    error_count = 0
    E = len(transitions)
    
    for prev_state, curr_state in transitions:
        is_habituated = False
        
        if isinstance(agent, TMGWRAgent):
            # Get node indices for TMGWR
            prev_node = agent.model.get_node_index(prev_state)
            curr_node = agent.model.get_node_index(curr_state)
            # Check connection in model.C
            is_habituated = agent.model.C[prev_node, curr_node] == 1
        else:
            # For HGWRSOM
            # Process previous state
            prev_x_data = np.array([prev_state[0]]).reshape(1, -1)
            prev_y_data = np.array([prev_state[1]]).reshape(1, -1)
            
            prev_x_bmu, _ = agent.lower_x.find_best_matching_units(prev_x_data)
            prev_y_bmu, _ = agent.lower_y.find_best_matching_units(prev_y_data)
            
            prev_x_binary = np.zeros(len(agent.lower_x.A))
            prev_y_binary = np.zeros(len(agent.lower_y.A))
            prev_x_binary[prev_x_bmu] = 1
            prev_y_binary[prev_y_bmu] = 1
            
            prev_pattern = (tuple(prev_x_binary), tuple(prev_y_binary))
            
            # Process current state
            curr_x_data = np.array([curr_state[0]]).reshape(1, -1)
            curr_y_data = np.array([curr_state[1]]).reshape(1, -1)
            
            curr_x_bmu, _ = agent.lower_x.find_best_matching_units(curr_x_data)
            curr_y_bmu, _ = agent.lower_y.find_best_matching_units(curr_y_data)
            
            curr_x_binary = np.zeros(len(agent.lower_x.A))
            curr_y_binary = np.zeros(len(agent.lower_y.A))
            curr_x_binary[curr_x_bmu] = 1
            curr_y_binary[curr_y_bmu] = 1
            
            curr_pattern = (tuple(curr_x_binary), tuple(curr_y_binary))
            
            # Check if patterns exist and get indices
            prev_found = False
            curr_found = False
            prev_node = -1
            curr_node = -1

            # Search for previous pattern
            for idx, existing_pattern in enumerate(agent.nodes):
                if all(np.array_equal(p1, p2) for p1, p2 in zip(prev_pattern, existing_pattern)):
                    prev_found = True
                    prev_node = idx
                    break

            # Search for current pattern
            for idx, existing_pattern in enumerate(agent.nodes):
                if all(np.array_equal(p1, p2) for p1, p2 in zip(curr_pattern, existing_pattern)):
                    curr_found = True
                    curr_node = idx
                    break

            # Check if both patterns were found and check connection
            if prev_found and curr_found:
                if prev_node < len(agent.connections) and curr_node < len(agent.connections):
                    is_habituated = agent.connections[prev_node, curr_node] == 1

        # Count non-habituated transitions
        if not is_habituated:
            error_count += 1

    # Calculate SE using the corrected formula
    SE = (error_count + tau) / (E + tau)
    
    return SE

def run_noise_comparison(noise_levels=[0, 1/6, 1/3, 1/2, 2/3, 5/6, 1, 7/6, 4/3], episodes_per_noise=20):
    # Initialize results storage
    results = {
        'TMGWR': {level: {'nodes': [], 'purity': [], 'se': []} for level in noise_levels},
        'HGWRSOM': {level: {'nodes': [], 'purity': [], 'se': []} for level in noise_levels}
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
            hgwrsom_agent = HierarchicalGWRSOMAgent(
                lower_dim=1, higher_dim=2, epsilon_b=0.35,
                epsilon_n=0.15, beta=0.7, delta=0.79,
                T_max=20, N_max=100, eta=0.5,
                phi=0.9, sigma=0.5
            )
            
            try:
                # Initialize nodes list and connections matrix
                hgwrsom_agent.nodes = []
                hgwrsom_agent.connections = np.zeros((0, 0))
                
                # Train lower networks with smaller epochs
                hgwrsom_agent.train_lower_networks(training_data, epochs=10)
                hgwrsom_agent.set_goal(goal)
                hgwrsom_agent.set_epsilon(1)

                # Run episode for each agent
                for agent, agent_name in [(tmgwr_agent, 'TMGWR'), (hgwrsom_agent, 'HGWRSOM')]:
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
                        if agent_name == 'TMGWR':
                            node_idx = agent.model.get_node_index(noisy_state)
                        else:
                            # For HGWRSOM, use lower networks to get activity
                            x_data = np.array([noisy_state[0]]).reshape(1, -1)
                            y_data = np.array([noisy_state[1]]).reshape(1, -1)
                            
                            x_bmu, _ = agent.lower_x.find_best_matching_units(x_data)
                            y_bmu, _ = agent.lower_y.find_best_matching_units(y_data)
                            
                            x_binary = np.zeros(len(agent.lower_x.A))
                            y_binary = np.zeros(len(agent.lower_y.A))
                            x_binary[x_bmu] = 1
                            y_binary[y_bmu] = 1
                            
                            pattern = (tuple(x_binary), tuple(y_binary))
                            
                            # Get or create node index
                            found = False
                            node_idx = 0
                            for idx, existing_pattern in enumerate(agent.nodes):
                                if all(np.array_equal(p1, p2) for p1, p2 in zip(pattern, existing_pattern)):
                                    found = True
                                    node_idx = idx
                                    break
                            
                            if not found:
                                # Create new node and expand connections matrix
                                node_idx = len(agent.nodes)
                                agent.nodes.append(pattern)
                                
                                # Expand connections matrix
                                old_size = len(agent.connections)
                                new_connections = np.zeros((old_size + 1, old_size + 1))
                                if old_size > 0:  # Only copy if there were existing connections
                                    new_connections[:-1, :-1] = agent.connections
                                agent.connections = new_connections
                        
                        # Update state-node mapping counts
                        state_tuple = tuple(current_state)
                        state_node_mappings[node_idx][state_tuple] += 1
                        total_visits += 1
                        
                        # Select and execute action
                        action = agent.select_action(noisy_state)
                        Maze.move_player(action)
                        next_state = Maze.get_player_pos()
                        
                        # Update model with actual next state (no noise)
                        agent.update_model(next_state, action)
                        
                        transitions.append((prev_state, next_state))
                        current_state = next_state
                    
                    # Calculate metrics
                    purity = calculate_purity(state_node_mappings, total_visits)
                    se = calculate_se(transitions, agent)
                    
                    # Record number of nodes
                    if agent_name == 'TMGWR':
                        num_nodes = len(agent.model.W)
                    else:
                        num_nodes = len(agent.nodes)
                    
                    # Store results
                    results[agent_name][noise_level]['nodes'].append(num_nodes)
                    results[agent_name][noise_level]['purity'].append(purity)
                    results[agent_name][noise_level]['se'].append(se)
            
            except Exception as e:
                print(f"Error in episode {episode} with noise level {noise_level}: {str(e)}")
                results[agent_name][noise_level]['nodes'].append(0)
                results[agent_name][noise_level]['purity'].append(0)
                results[agent_name][noise_level]['se'].append(0)
                continue

    return results

def plot_comparison_results(results):
    metrics = {
        'nodes': {'title': 'Number of Nodes vs Noise Level', 'ylabel': 'Number of Nodes', 'ylim': (0, 250)},
        'purity': {'title': 'Purity vs Noise Level', 'ylabel': 'Purity (%)', 'ylim': (0, 100)},
        'se': {'title': 'Sensorimotor Error vs Noise Level', 'ylabel': 'SE', 'ylim': None}
    }
    
    for metric in metrics:
        data = []
        for agent in results:
            for noise_level in results[agent]:
                values = results[agent][noise_level][metric]
                for value in values:
                    data.append({
                        'Agent': agent,
                        'Noise Level': f"{noise_level:.3f}",
                        'Value': value,
                        'Metric': metrics[metric]['ylabel']
                    })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(15, 8))
        custom_palette = {'TMGWR': 'green', 'HGWRSOM': 'orange'}
        sns.boxplot(data=df, x='Noise Level', y='Value', hue='Agent', palette=custom_palette)
        
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