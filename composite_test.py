import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import time
import os
import math
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from scipy.stats import pearsonr

# Import your existing classes
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.TMGWR_agent import TMGWRAgent
from Agents.HSOM_binary import HierarchicalGWRSOMAgent

class HybridEvaluationFramework:
    """
    A framework for comprehensive evaluation of spatial representation algorithms
    using multiple metrics and a weighted composite score.
    """
    
    def __init__(self, 
                 weights={
                     'efficiency': 0.3,  # Resource usage metrics
                     'performance': 0.4,  # Task completion metrics
                     'robustness': 0.3    # Noise handling metrics
                 }):
        """
        Initialize the framework with custom weights for different evaluation aspects.
        
        Parameters:
        weights (dict): Custom weights for different evaluation categories
        """
        self.weights = weights
        self.results = {}
        self.metrics = {
            'efficiency': ['node_count', 'memory_usage', 'information_density'],
            'performance': ['goal_reaching_efficiency', 'planning_stability', 'topological_preservation'],
            'robustness': ['critical_noise_threshold', 'partial_observation_robustness']
        }
        
    def run_evaluation(self, agents, maze_sizes, noise_levels, trials_per_config=5, 
                      partial_observation_rates=[0, 0.2, 0.4]):
        """
        Run a comprehensive evaluation of agents across different maze sizes,
        noise levels, and partial observation conditions.
        
        Parameters:
        agents (dict): Dictionary of agent instances {name: agent}
        maze_sizes (list): List of maze sizes to test [(width, height),...]
        noise_levels (list): List of noise levels to test
        trials_per_config (int): Number of trials per configuration
        partial_observation_rates (list): Probabilities of feature masking
        
        Returns:
        dict: Complete results data
        """
        self.results = {
            agent_name: {
                'efficiency': defaultdict(list),
                'performance': defaultdict(list),
                'robustness': defaultdict(list),
                'composite_scores': defaultdict(list)
            } for agent_name in agents.keys()
        }
        
        for maze_width, maze_height in maze_sizes:
            print(f"\nEvaluating on {maze_width}x{maze_height} maze")
            
            # Generate maze for this size
            maze_map, player_pos_index, goal_pos_index = self._generate_maze(maze_width, maze_height)
            
            for noise_level in noise_levels:
                print(f"  Testing with noise level σ² = {noise_level}")
                
                for trial in range(trials_per_config):
                    print(f"    Trial {trial+1}/{trials_per_config}")
                    
                    # Reset maze for new trial
                    maze = MazePlayer(maze_map=maze_map, 
                                     player_index_pos=player_pos_index, 
                                     goal_index_pos=goal_pos_index,
                                     display_maze=False)
                    
                    for agent_name, agent_cls in agents.items():
                        # Initialize fresh agent for this trial
                        if agent_name == 'TMGWR':
                            agent = TMGWRAgent(
                                nDim=2, Ni=2, epsilon_b=0.55194, epsilon_n=0.90,
                                beta=0.8, delta=0.6235, T_max=17, N_max=300,
                                eta=0.95, phi=0.6, sigma=1
                            )
                        else:  # MINERVA/HGWRSOM
                            agent = HierarchicalGWRSOMAgent(
                                lower_dim=1, higher_dim=2, epsilon_b=0.35,
                                epsilon_n=0.15, beta=0.7, delta=0.79,
                                T_max=20, N_max=100, eta=0.5,
                                phi=0.9, sigma=0.5
                            )
                            
                            # Pre-train lower networks with training data
                            x_train = np.linspace(-maze_width*10, maze_width*10, 10).reshape(-1, 1)
                            y_train = np.linspace(-maze_height*10, maze_height*10, 10).reshape(-1, 1)
                            training_data = np.hstack((x_train, y_train))
                            agent.train_lower_networks(training_data, epochs=5)
                            
                        # Set goal and exploration parameters
                        agent.set_goal(maze.get_goal_pos())
                        agent.set_epsilon(1.0)  # Start with full exploration
                        
                        # Evaluate metrics for this configuration
                        config_key = f"{maze_width}x{maze_height}_noise{noise_level}_trial{trial}"
                        
                        # Gather metrics
                        self._evaluate_efficiency_metrics(agent, agent_name, maze, config_key)
                        self._evaluate_performance_metrics(agent, agent_name, maze, config_key, noise_level)
                        self._evaluate_robustness_metrics(agent, agent_name, maze, config_key, noise_level, 
                                                         partial_observation_rates)
                        
                        # Calculate composite score for this configuration
                        self._calculate_composite_score(agent_name, config_key)
        
        return self.results
    
    def _generate_maze(self, width, height):
        """Generate a maze of the specified dimensions"""
        if width <= 9 and height <= 9:
            # Use built-in small maze
            return MazeMaps.get_default_map()
        else:
            # Use built-in large maze for larger requests
            return MazeMaps.default_maze_map_1, MazeMaps.default_player_pos_index, MazeMaps.default_goal_pos_index
    
    def _evaluate_efficiency_metrics(self, agent, agent_name, maze, config_key):
        """Evaluate resource usage metrics"""
        # Run a short training period
        current_state = maze.get_initial_player_pos()
        maze.reset_player()
        
        start_time = time.time()
        num_steps = 0
        
        # Train for a limited number of steps
        while num_steps < 500:  # Limit training duration
            # Add noise to current state
            num_steps += 1
            
            # Select and execute action
            action = agent.select_action(current_state)
            maze.move_player(action)
            next_state = maze.get_player_pos()
            
            # Update model
            agent.update_model(next_state, action)
            current_state = next_state
            
            # Check if goal reached
            if current_state == maze.get_goal_pos():
                break
        
        computation_time = time.time() - start_time
        
        # Measure node count
        if agent_name == 'TMGWR':
            node_count = len(agent.model.W)
            memory_usage = (agent.model.W.size + agent.model.C.size) * 8  # Approximation in bytes
        else:
            node_count = len(agent.nodes)
            memory_usage = (len(agent.lower_x.A) + len(agent.lower_y.A) + 
                           len(agent.nodes) + agent.connections.size) * 8  # Approximation in bytes
        
        # Calculate information density (bits per node)
        total_states_visited = num_steps
        if node_count > 0:
            information_density = total_states_visited / node_count
        else:
            information_density = 0
            
        # Store metrics
        self.results[agent_name]['efficiency']['node_count'].append((config_key, node_count))
        self.results[agent_name]['efficiency']['memory_usage'].append((config_key, memory_usage))
        self.results[agent_name]['efficiency']['information_density'].append((config_key, information_density))
        self.results[agent_name]['efficiency']['computation_time'].append((config_key, computation_time))
    
    def _evaluate_performance_metrics(self, agent, agent_name, maze, config_key, noise_level):
        """Evaluate task completion metrics"""
        # Calculate optimal path length
        optimal_path = self._calculate_optimal_path_length(maze)
        
        current_state = maze.get_initial_player_pos()
        maze.reset_player()
        
        # Storage for transitions to track topological preservation
        transitions = []
        plan_stability_scores = []
        
        # Training phase
        agent.set_epsilon(0.8)  # High exploration at first
        for training_step in range(500):
            # Decay exploration rate over time
            if training_step % 100 == 0 and agent.get_epsilon() > 0.2:
                agent.decay_epsilon()
                
            # Add noise to current state
            noisy_state = np.array(current_state) + np.random.normal(0, np.sqrt(noise_level), 2)
            
            prev_state = current_state
            
            # Record if this is a planning step
            is_planning = agent.get_epsilon() < 0.5
            
            # Select and execute action
            action = agent.select_action(noisy_state)
            maze.move_player(action)
            next_state = maze.get_player_pos()
            
            # Store transition for topological preservation analysis
            transitions.append((prev_state, next_state))
            
            # Check plan stability - if agent expected a different outcome
            if is_planning and hasattr(agent, 'is_plan') and agent.is_plan:
                if agent_name == 'TMGWR':
                    expected_idx = agent.expected_next_state if hasattr(agent, 'expected_next_state') else None
                    actual_idx = agent.model.get_node_index(next_state)
                    plan_stability = 1.0 if expected_idx == actual_idx else 0.0
                else:
                    expected_node = agent.expected_next_node if hasattr(agent, 'expected_next_node') else None
                    pattern = agent.get_firing_pattern(next_state)
                    actual_node = agent.find_node_index(pattern)
                    plan_stability = 1.0 if expected_node == actual_node else 0.0
                    
                plan_stability_scores.append(plan_stability)
            
            # Update model
            agent.update_model(next_state, action)
            current_state = next_state
            
            # Check if goal reached
            if current_state == maze.get_goal_pos():
                break
        
        # Testing phase - evaluate goal reaching with low exploration
        maze.reset_player()
        current_state = maze.get_initial_player_pos()
        agent.set_epsilon(0.1)  # Mostly exploitation
        
        steps_to_goal = 0
        max_steps = optimal_path * 3  # Limit the maximum steps
        
        goal_reached = False
        for step in range(max_steps):
            steps_to_goal += 1
            
            # Add noise to current state
            noisy_state = np.array(current_state) + np.random.normal(0, np.sqrt(noise_level), 2)
            
            # Select and execute action
            action = agent.select_action(noisy_state)
            maze.move_player(action)
            next_state = maze.get_player_pos()
            
            # Update model
            agent.update_model(next_state, action)
            current_state = next_state
            
            # Check if goal reached
            if current_state == maze.get_goal_pos():
                goal_reached = True
                break
        
        # Calculate goal reaching efficiency
        if goal_reached:
            goal_efficiency = optimal_path / steps_to_goal
        else:
            goal_efficiency = 0
            
        # Calculate planning stability (average)
        if plan_stability_scores:
            planning_stability = sum(plan_stability_scores) / len(plan_stability_scores)
        else:
            planning_stability = 0
            
        # Calculate topological preservation
        topo_preservation = self._calculate_topological_preservation(transitions, agent, agent_name)
        
        # Store metrics
        self.results[agent_name]['performance']['goal_reaching_efficiency'].append((config_key, goal_efficiency))
        self.results[agent_name]['performance']['planning_stability'].append((config_key, planning_stability))
        self.results[agent_name]['performance']['topological_preservation'].append((config_key, topo_preservation))
    
    def _evaluate_robustness_metrics(self, agent, agent_name, maze, config_key, base_noise, 
                                   partial_observation_rates):
        """Evaluate robustness metrics including noise and partial observation handling"""
        # Test critical noise threshold
        noise_tolerance = self._find_critical_noise_threshold(agent, agent_name, maze, base_noise)
        
        # Test partial observation robustness
        partial_obs_robustness = self._test_partial_observation_robustness(agent, agent_name, maze, 
                                                                          partial_observation_rates)
        
        # Store metrics
        self.results[agent_name]['robustness']['critical_noise_threshold'].append((config_key, noise_tolerance))
        self.results[agent_name]['robustness']['partial_observation_robustness'].append((config_key, partial_obs_robustness))
    
    def _calculate_composite_score(self, agent_name, config_key):
        """Calculate weighted composite score from all metrics"""
        category_scores = {}
        
        # Normalize and aggregate scores for each category
        for category, metrics in self.metrics.items():
            category_score = 0
            metrics_count = 0
            
            for metric in metrics:
                if metric in self.results[agent_name][category]:
                    # Find the value for this config
                    for k, v in self.results[agent_name][category][metric]:
                        if k == config_key:
                            # Normalize based on metric (different for each)
                            if metric == 'node_count' or metric == 'memory_usage':
                                # Lower is better, invert
                                normalized_value = 1.0 / (1.0 + v/100)
                            else:
                                # Higher is better
                                normalized_value = v
                            
                            category_score += normalized_value
                            metrics_count += 1
                            break
            
            if metrics_count > 0:
                category_scores[category] = category_score / metrics_count
            else:
                category_scores[category] = 0
        
        # Calculate weighted composite
        composite_score = 0
        for category, score in category_scores.items():
            composite_score += score * self.weights.get(category, 1.0/len(category_scores))
            
        self.results[agent_name]['composite_scores'][config_key] = composite_score
        return composite_score
    
    def _calculate_optimal_path_length(self, maze):
        """Calculate the optimal path length from start to goal"""
        # Ideally, you would use A* or BFS to find the shortest path
        # For simplicity, we'll use a Manhattan distance approximation
        
        # Access the indices directly from the maze object
        start_idx = maze.initial_player_index_pos 
        goal_idx = maze.goal_index_pos
        
        # Calculate Manhattan distance in grid coordinates
        return abs(start_idx[0] - goal_idx[0]) + abs(start_idx[1] - goal_idx[1])
    
    def _calculate_topological_preservation(self, transitions, agent, agent_name):
        """
        Calculate how well the agent's internal representation preserves
        topological relationships from the environment
        """
        if len(transitions) < 2:
            return 0
            
        # Calculate actual Euclidean distances between sequential states
        actual_distances = []
        for i in range(len(transitions)-1):
            state1 = transitions[i][0]
            state2 = transitions[i+1][0]
            actual_distances.append(np.linalg.norm(np.array(state1) - np.array(state2)))
            
        # Calculate corresponding distances in agent's representation
        model_distances = []
        
        if agent_name == 'TMGWR':
            for i in range(len(transitions)-1):
                state1 = transitions[i][0]
                state2 = transitions[i+1][0]
                
                # Get node indices
                idx1 = agent.model.get_node_index(state1)
                idx2 = agent.model.get_node_index(state2)
                
                # Calculate distance in node space - approximation using graph distance
                if not hasattr(agent, 'graph_distances'):
                    agent.graph_distances = {}
                
                if idx1 not in agent.graph_distances or idx2 not in agent.graph_distances.get(idx1, {}):
                    # Build graph if not already built
                    graph = nx.Graph()
                    
                    # Add nodes and edges from the connection matrix
                    for row in range(agent.model.C.shape[0]):
                        for col in range(agent.model.C.shape[1]):
                            if agent.model.C[row, col] > 0:
                                graph.add_edge(row, col, weight=1)
                    
                    # Try to find shortest path
                    try:
                        path_length = nx.shortest_path_length(graph, idx1, idx2)
                    except nx.NetworkXNoPath:
                        path_length = 100  # Large value if no path exists
                    
                    # Cache result
                    if idx1 not in agent.graph_distances:
                        agent.graph_distances[idx1] = {}
                    agent.graph_distances[idx1][idx2] = path_length
                else:
                    path_length = agent.graph_distances[idx1][idx2]
                
                model_distances.append(path_length)
        else:
            # HGWRSOM/MINERVA
            for i in range(len(transitions)-1):
                state1 = transitions[i][0]
                state2 = transitions[i+1][0]
                
                # Get patterns and indices
                pattern1 = agent.get_firing_pattern(state1)
                pattern2 = agent.get_firing_pattern(state2)
                
                idx1 = agent.find_node_index(pattern1)
                idx2 = agent.find_node_index(pattern2)
                
                if idx1 is None or idx2 is None:
                    model_distances.append(100)  # Large value for states not in the model
                    continue
                
                # Calculate distance in node space
                if idx1 < agent.connections.shape[0] and idx2 < agent.connections.shape[0]:
                    # Build graph if not already built
                    if not hasattr(agent, 'graph'):
                        agent.graph = nx.Graph()
                        
                        # Add nodes and edges from the connection matrix
                        for row in range(agent.connections.shape[0]):
                            for col in range(agent.connections.shape[1]):
                                if agent.connections[row, col] > 0:
                                    agent.graph.add_edge(row, col, weight=1)
                    
                    # Try to find shortest path
                    try:
                        path_length = nx.shortest_path_length(agent.graph, idx1, idx2)
                    except nx.NetworkXNoPath:
                        path_length = 100  # Large value if no path exists
                    
                    model_distances.append(path_length)
                else:
                    model_distances.append(100)  # Large value for nodes outside connection matrix
        
        # Calculate correlation between actual and model distances
        if len(actual_distances) > 1 and len(model_distances) > 1:
            try:
                correlation, _ = pearsonr(actual_distances, model_distances)
                # If negative correlation or NaN, return 0
                if np.isnan(correlation) or correlation < 0:
                    return 0
                return correlation
            except:
                return 0
        return 0
    
    def _find_critical_noise_threshold(self, agent, agent_name, maze, base_noise):
        """
        Find the noise level at which the agent's performance degrades significantly
        """
        # Train the agent with base noise level
        current_state = maze.get_initial_player_pos()
        maze.reset_player()
        
        # Initial training
        for _ in range(200):
            noisy_state = np.array(current_state) + np.random.normal(0, np.sqrt(base_noise), 2)
            action = agent.select_action(noisy_state)
            maze.move_player(action)
            next_state = maze.get_player_pos()
            agent.update_model(next_state, action)
            current_state = next_state
            
            if current_state == maze.get_goal_pos():
                maze.reset_player()
                current_state = maze.get_initial_player_pos()
        
        # Test with increasing noise levels
        noise_levels = [base_noise * factor for factor in [1, 1.5, 2, 2.5, 3, 4, 5]]
        success_rates = []
        
        for noise in noise_levels:
            successes = 0
            trials = 5
            
            for _ in range(trials):
                maze.reset_player()
                current_state = maze.get_initial_player_pos()
                goal_reached = False
                agent.set_epsilon(0.1)  # Mostly exploitation
                
                for step in range(100):  # Limit steps
                    noisy_state = np.array(current_state) + np.random.normal(0, np.sqrt(noise), 2)
                    action = agent.select_action(noisy_state)
                    maze.move_player(action)
                    next_state = maze.get_player_pos()
                    
                    if next_state == maze.get_goal_pos():
                        goal_reached = True
                        break
                        
                    current_state = next_state
                
                if goal_reached:
                    successes += 1
            
            success_rate = successes / trials
            success_rates.append(success_rate)
            
            # Stop if success rate drops below 50%
            if success_rate < 0.5:
                break
        
        # Find noise level where performance drops below 80% of baseline
        if success_rates[0] > 0:
            baseline = success_rates[0]
            threshold = 0.8 * baseline
            
            for i, rate in enumerate(success_rates):
                if rate < threshold:
                    return noise_levels[i]
            
            # If no clear threshold found, return the highest tested noise
            return noise_levels[-1]
        else:
            return base_noise  # Default to base noise if no successful baseline
    
    def _test_partial_observation_robustness(self, agent, agent_name, maze, partial_rates):
        """
        Test the agent's ability to handle partial observations by masking
        components of the state vector
        """
        # First get baseline performance with full observations
        baseline_success = self._run_partial_observation_trial(agent, agent_name, maze, 0.0)
        
        # Test with different partial observation rates
        rob_scores = []
        
        for rate in partial_rates:
            if rate == 0:
                continue  # Skip baseline
                
            success_rate = self._run_partial_observation_trial(agent, agent_name, maze, rate)
            
            # Calculate robustness score (normalized to baseline)
            if baseline_success > 0:
                rob_score = success_rate / baseline_success
            else:
                rob_score = 0
                
            rob_scores.append(rob_score)
        
        # Return average robustness score across all rates
        if rob_scores:
            return sum(rob_scores) / len(rob_scores)
        else:
            return 0
    
    def _run_partial_observation_trial(self, agent, agent_name, maze, mask_rate):
        """Run a trial with partial observations and return success rate"""
        trials = 5
        successes = 0
        
        for _ in range(trials):
            maze.reset_player()
            current_state = maze.get_initial_player_pos()
            goal_reached = False
            agent.set_epsilon(0.1)  # Mostly exploitation
            
            for step in range(100):  # Limit steps
                # Apply partial observation masking
                if mask_rate > 0:
                    masked_state = self._apply_state_masking(current_state, mask_rate)
                else:
                    masked_state = current_state
                
                action = agent.select_action(masked_state)
                maze.move_player(action)
                next_state = maze.get_player_pos()
                
                if next_state == maze.get_goal_pos():
                    goal_reached = True
                    break
                    
                # Update with full observation
                agent.update_model(next_state, action)
                current_state = next_state
            
            if goal_reached:
                successes += 1
        
        return successes / trials
    
    def _apply_state_masking(self, state, mask_rate):
        """
        Apply random masking to state vector.
        For coordinate masking, we don't fully mask but add noise instead.
        """
        state = np.array(state).copy()
        
        # Randomly decide if we'll mask x or y coordinate
        if np.random.random() < mask_rate:
            # X coordinate gets masked (add strong noise)
            state[0] += np.random.normal(0, 50)
            
        if np.random.random() < mask_rate:
            # Y coordinate gets masked (add strong noise)
            state[1] += np.random.normal(0, 50)
            
        return state
    
    def visualize_results(self):
        """Generate visualizations of the evaluation results"""
        self._visualize_composite_scores()
        self._visualize_category_performance()
        self._visualize_pareto_frontier()
        
    def _visualize_composite_scores(self):
        """Visualize composite scores across different configurations"""
        data = []
        
        for agent_name in self.results:
            for config_key, score in self.results[agent_name]['composite_scores'].items():
                # Parse config key
                maze_size, noise_info, trial_info = config_key.split('_')
                noise_level = float(noise_info.replace('noise', ''))
                
                data.append({
                    'Agent': agent_name,
                    'Maze Size': maze_size,
                    'Noise Level': noise_level,
                    'Composite Score': score
                })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x='Noise Level', y='Composite Score', hue='Agent', 
                    style='Maze Size', markers=True, dashes=False)
        
        plt.title('Composite Performance Score vs. Noise Level')
        plt.xlabel('Noise Level (σ²)')
        plt.ylabel('Composite Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def _visualize_category_performance(self):
        """Visualize performance across different metric categories"""
        categories = ['efficiency', 'performance', 'robustness']
        
        for category in categories:
            # Prepare data
            data = []
            
            for agent_name in self.results:
                # Aggregate metrics for this category
                for metric in self.metrics[category]:
                    if metric in self.results[agent_name][category]:
                        for config_key, value in self.results[agent_name][category][metric]:
                            # Parse config key
                            maze_size, noise_info, trial_info = config_key.split('_')
                            noise_level = float(noise_info.replace('noise', ''))
                            
                            # Normalize values
                            if metric in ['node_count', 'memory_usage'] and value > 0:
                                # Lower is better, invert
                                normalized_value = 1.0 / (1.0 + value/100)
                            else:
                                # Higher is better
                                normalized_value = value
                                
                            data.append({
                                'Agent': agent_name,
                                'Maze Size': maze_size,
                                'Noise Level': noise_level,
                                'Metric': metric,
                                'Value': normalized_value
                            })
            
            if not data:
                continue
                
            df = pd.DataFrame(data)
            
            plt.figure(figsize=(14, 8))
            
            g = sns.FacetGrid(df, col='Metric', hue='Agent', col_wrap=2, height=4, aspect=1.5)
            g.map(sns.lineplot, 'Noise Level', 'Value')
            g.add_legend()
            g.set_titles(col_template="{col_name}")
            
            plt.suptitle(f'{category.capitalize()} Metrics Performance', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()
    
    def _visualize_pareto_frontier(self):
        """Visualize the Pareto frontier of efficiency vs. performance"""
        data = []
        
        # Aggregate data for Pareto analysis
        for agent_name in self.results:
            # Get efficiency score (inverse of node count)
            for config_key, node_count in self.results[agent_name]['efficiency']['node_count']:
                efficiency_score = 1.0 / (1.0 + node_count/100) if node_count > 0 else 0
                
                # Find corresponding performance score
                perf_score = 0
                for ck, value in self.results[agent_name]['performance']['goal_reaching_efficiency']:
                    if ck == config_key:
                        perf_score = value
                        break
                
                # Parse config key
                maze_size, noise_info, trial_info = config_key.split('_')
                noise_level = float(noise_info.replace('noise', ''))
                
                data.append({
                    'Agent': agent_name,
                    'Maze Size': maze_size,
                    'Noise Level': noise_level,
                    'Efficiency': efficiency_score,
                    'Performance': perf_score
                })
        
        if not data:
            return
            
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(10, 8))
        
        sns.scatterplot(data=df, x='Efficiency', y='Performance', hue='Agent', 
                       style='Maze Size', s=100, alpha=0.7)
        
        # Highlight Pareto-optimal points
        for agent_name in df['Agent'].unique():
            agent_data = df[df['Agent'] == agent_name]
            
            # Find Pareto-optimal points
            pareto_points = []
            for i, row1 in agent_data.iterrows():
                is_pareto = True
                for j, row2 in agent_data.iterrows():
                    if i != j:
                        if (row2['Efficiency'] >= row1['Efficiency'] and 
                            row2['Performance'] > row1['Performance']) or (
                            row2['Efficiency'] > row1['Efficiency'] and 
                            row2['Performance'] >= row1['Performance']):
                            is_pareto = False
                            break
                if is_pareto:
                    pareto_points.append((row1['Efficiency'], row1['Performance']))
            
            # Draw lines connecting Pareto-optimal points
            if pareto_points:
                pareto_points.sort()
                x_vals, y_vals = zip(*pareto_points)
                plt.plot(x_vals, y_vals, 'k--', alpha=0.5)
        
        plt.title('Pareto Frontier: Efficiency vs. Performance')
        plt.xlabel('Efficiency Score (higher is better)')
        plt.ylabel('Performance Score (higher is better)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        """Generate a summary report of the evaluation"""
        summary = {
            'agent_comparison': {},
            'scaling_behavior': {},
            'noise_tolerance': {},
            'partial_observation_handling': {}
        }
        
        # Agent comparison based on composite scores
        for agent_name in self.results:
            if not self.results[agent_name]['composite_scores']:
                continue
                
            scores = list(self.results[agent_name]['composite_scores'].values())
            summary['agent_comparison'][agent_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            }
        
        # Scaling behavior across maze sizes
        maze_sizes = set()
        for agent_name in self.results:
            for config_key in self.results[agent_name]['composite_scores']:
                maze_size = config_key.split('_')[0]
                maze_sizes.add(maze_size)
        
        for maze_size in maze_sizes:
            summary['scaling_behavior'][maze_size] = {}
            
            for agent_name in self.results:
                scores = []
                
                for config_key, score in self.results[agent_name]['composite_scores'].items():
                    if config_key.startswith(maze_size):
                        scores.append(score)
                
                if scores:
                    summary['scaling_behavior'][maze_size][agent_name] = np.mean(scores)
        
        # Noise tolerance
        for agent_name in self.results:
            thresholds = []
            
            for config_key, value in self.results[agent_name]['robustness']['critical_noise_threshold']:
                thresholds.append(value)
            
            if thresholds:
                summary['noise_tolerance'][agent_name] = np.mean(thresholds)
        
        # Partial observation handling
        for agent_name in self.results:
            robustness_scores = []
            
            for config_key, value in self.results[agent_name]['robustness']['partial_observation_robustness']:
                robustness_scores.append(value)
            
            if robustness_scores:
                summary['partial_observation_handling'][agent_name] = np.mean(robustness_scores)
        
        return summary
    
    def print_summary_report(self):
        """Print a formatted summary report to the console"""
        summary = self.generate_summary_report()
        
        print("\n" + "="*80)
        print(" "*30 + "EVALUATION SUMMARY REPORT")
        print("="*80)
        
        print("\n1. OVERALL AGENT COMPARISON")
        print("-"*80)
        
        if summary['agent_comparison']:
            # Sort agents by mean score
            sorted_agents = sorted(
                summary['agent_comparison'].items(),
                key=lambda x: x[1]['mean_score'],
                reverse=True
            )
            
            for i, (agent_name, stats) in enumerate(sorted_agents):
                print(f"{i+1}. {agent_name}:")
                print(f"   - Mean Composite Score: {stats['mean_score']:.4f}")
                print(f"   - Score Range: [{stats['min_score']:.4f}, {stats['max_score']:.4f}]")
                print(f"   - Standard Deviation: {stats['std_score']:.4f}")
        else:
            print("No composite score data available.")
        
        print("\n2. SCALING BEHAVIOR ACROSS MAZE SIZES")
        print("-"*80)
        
        if summary['scaling_behavior']:
            # Sort maze sizes
            sorted_sizes = sorted(summary['scaling_behavior'].keys())
            
            for maze_size in sorted_sizes:
                print(f"\nMaze Size: {maze_size}")
                
                # Sort agents by score for this maze size
                sorted_agents = sorted(
                    summary['scaling_behavior'][maze_size].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for agent_name, score in sorted_agents:
                    print(f"   - {agent_name}: {score:.4f}")
        else:
            print("No scaling behavior data available.")
        
        print("\n3. NOISE TOLERANCE")
        print("-"*80)
        
        if summary['noise_tolerance']:
            # Sort agents by noise tolerance
            sorted_agents = sorted(
                summary['noise_tolerance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for agent_name, threshold in sorted_agents:
                print(f"   - {agent_name}: Critical Noise Threshold = {threshold:.4f}")
        else:
            print("No noise tolerance data available.")
        
        print("\n4. PARTIAL OBSERVATION HANDLING")
        print("-"*80)
        
        if summary['partial_observation_handling']:
            # Sort agents by partial observation robustness
            sorted_agents = sorted(
                summary['partial_observation_handling'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for agent_name, robustness in sorted_agents:
                print(f"   - {agent_name}: Partial Observation Robustness = {robustness:.4f}")
        else:
            print("No partial observation handling data available.")
        
        print("\n" + "="*80)


def run_experiment():
    """
    Main function to run the hybrid evaluation experiment
    """
    print("Starting Hybrid Evaluation Framework Experiment...")
    
    # Initialize the framework
    evaluator = HybridEvaluationFramework(
        weights={
            'efficiency': 0.3,
            'performance': 0.4,
            'robustness': 0.3
        }
    )
    
    # Define agents to test
    agents = {
        'TMGWR': TMGWRAgent,
        'MINERVA': HierarchicalGWRSOMAgent
    }
    
    # Define evaluation parameters
    maze_sizes = [(7, 7), (15, 15)]  # Small and large mazes
    noise_levels = [0, 1/6, 1/3, 2/3, 1]  # Varying noise levels
    partial_observation_rates = [0, 0.2, 0.4]  # Probability of masking features
    
    # Run evaluation
    results = evaluator.run_evaluation(
        agents=agents,
        maze_sizes=maze_sizes,
        noise_levels=noise_levels,
        trials_per_config=3,  # Using 3 trials for faster execution
        partial_observation_rates=partial_observation_rates
    )
    
    # Visualize results
    evaluator.visualize_results()
    
    # Print summary report
    evaluator.print_summary_report()
    
    print("\nExperiment completed!")
    return evaluator


if __name__ == "__main__":
    evaluator = run_experiment()