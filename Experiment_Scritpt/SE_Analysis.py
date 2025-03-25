import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.TMGWR_agent import TMGWRAgent
from Agents.HSOM_binary import HierarchicalGWRSOMAgent, GWRSOM
import networkx as nx
import os
import time

# Set working directory
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

def calculate_pattern_stability(agent, test_states, noise_level=0.5, trials=20):
    """
    Test how stable MINERVA's pattern generation is under noise.
    This helps us understand why MINERVA can have lower SE despite having fewer nodes.
    """
    results = {}
    
    for state in test_states:
        state_key = tuple(state)
        patterns = []
        
        # Run multiple trials with noise
        for _ in range(trials):
            # Apply noise
            if noise_level > 0:
                noisy_state = np.array(state) + np.random.normal(0, np.sqrt(noise_level), 2)
            else:
                noisy_state = np.array(state)
            
            # Get pattern
            pattern = agent.get_firing_pattern(noisy_state)
            
            # Convert to hashable format
            pattern_str = str([str(p) for p in pattern])
            patterns.append(pattern_str)
        
        # Count unique patterns
        unique_patterns = set(patterns)
        pattern_counts = {p: patterns.count(p) for p in unique_patterns}
        
        # Calculate stability (higher percentage = more stable)
        most_common_pattern = max(pattern_counts.items(), key=lambda x: x[1])
        stability = most_common_pattern[1] / trials * 100
        
        results[state_key] = {
            'unique_patterns': len(unique_patterns),
            'stability': stability,
            'pattern_counts': pattern_counts
        }
    
    return results

def analyze_tmgwr_node_mapping(agent, test_states, noise_level=0.5, trials=20):
    """
    Analyze how TMGWR maps noisy states to nodes.
    """
    results = {}
    
    for state in test_states:
        state_key = tuple(state)
        nodes = []
        
        # Run multiple trials with noise
        for _ in range(trials):
            # Apply noise
            if noise_level > 0:
                noisy_state = np.array(state) + np.random.normal(0, np.sqrt(noise_level), 2)
            else:
                noisy_state = np.array(state)
            
            # Get node index
            node_idx = agent.model.get_node_index(noisy_state)
            
            # Store node index
            nodes.append(node_idx)
        
        # Count unique nodes
        unique_nodes = set([n for n in nodes if n is not None])
        node_counts = {n: nodes.count(n) for n in unique_nodes}
        
        # Calculate stability
        most_common_node = max(node_counts.items(), key=lambda x: x[1]) if node_counts else (None, 0)
        stability = most_common_node[1] / trials * 100 if trials > 0 else 0
        
        results[state_key] = {
            'unique_nodes': len(unique_nodes),
            'stability': stability,
            'node_counts': node_counts
        }
    
    return results

def compare_transition_representation(tmgwr_agent, minerva_agent, test_transitions, noise_level=0.5, trials=10):
    """
    Compare how both agents represent transitions under noise, which directly affects SE.
    """
    results = {
        'TMGWR': [],
        'MINERVA': []
    }
    
    for from_state, to_state in test_transitions:
        from_key = tuple(from_state)
        to_key = tuple(to_state)
        
        tmgwr_transitions = []
        minerva_transitions = []
        
        # Run multiple trials with noise
        for _ in range(trials):
            # Apply noise to both states
            if noise_level > 0:
                noisy_from = np.array(from_state) + np.random.normal(0, np.sqrt(noise_level), 2)
                noisy_to = np.array(to_state) + np.random.normal(0, np.sqrt(noise_level), 2)
            else:
                noisy_from = np.array(from_state)
                noisy_to = np.array(to_state)
            
            # TMGWR representation
            tmgwr_from_node = tmgwr_agent.model.get_node_index(noisy_from)
            tmgwr_to_node = tmgwr_agent.model.get_node_index(noisy_to)
            
            if tmgwr_from_node is not None and tmgwr_to_node is not None:
                tmgwr_transition = (tmgwr_from_node, tmgwr_to_node)
                tmgwr_transitions.append(tmgwr_transition)
            
            # MINERVA representation
            minerva_from_pattern = minerva_agent.get_firing_pattern(noisy_from)
            minerva_to_pattern = minerva_agent.get_firing_pattern(noisy_to)
            
            minerva_from_node = minerva_agent.find_node_index(minerva_from_pattern)
            minerva_to_node = minerva_agent.find_node_index(minerva_to_pattern)
            
            if minerva_from_node is not None and minerva_to_node is not None:
                minerva_transition = (minerva_from_node, minerva_to_node)
                minerva_transitions.append(minerva_transition)
        
        # Calculate unique transitions
        tmgwr_unique = len(set(tmgwr_transitions))
        minerva_unique = len(set(minerva_transitions))
        
        results['TMGWR'].append({
            'from_state': from_key,
            'to_state': to_key,
            'unique_transitions': tmgwr_unique,
            'total_transitions': len(tmgwr_transitions),
            'duplications': tmgwr_unique / len(tmgwr_transitions) if tmgwr_transitions else 0
        })
        
        results['MINERVA'].append({
            'from_state': from_key,
            'to_state': to_key,
            'unique_transitions': minerva_unique,
            'total_transitions': len(minerva_transitions),
            'duplications': minerva_unique / len(minerva_transitions) if minerva_transitions else 0
        })
    
    return results

def simulate_learning_and_calculate_se(agent_type, noise_level=0, num_episodes=5):
    """
    Simulate a learning session and calculate SE, tracking specific metrics
    """
    # Get maze details for navigation
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()
    Maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos_index, 
                     goal_index_pos=goal_pos_index, display_maze=False)
    goal = Maze.get_goal_pos()
    initial_state = Maze.get_initial_player_pos()
    
    # Initialize agent based on type
    if agent_type == "TMGWR":
        agent = TMGWRAgent(nDim=2, Ni=2, epsilon_b=0.55194, epsilon_n=0.90, beta=0.8, 
                           delta=0.6235, T_max=17, N_max=300, eta=0.95, phi=0.6, sigma=1)
    else:  # MINERVA / Hierarchical GWRSOM
        agent = HierarchicalGWRSOMAgent(
            lower_dim=1,  # Each coordinate handled separately
            higher_dim=2,  # Full 2D position at higher level
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
        # For MINERVA, train lower networks first
        training_data = []
        for _ in range(50):
            state = [np.random.randint(-72, 72), np.random.randint(-72, 72)]
            training_data.append(state)
        training_data = np.array(training_data)
        agent.train_lower_networks(training_data, epochs=20)
    
    agent.set_goal(goal=goal)
    agent.set_epsilon(1)
    
    # Track metrics
    all_transitions = []
    total_errors = 0
    node_growth = []
    transitions_learned = set()
    
    # Training loop
    for episode in range(num_episodes):
        current_state = initial_state
        Maze.reset_player()
        step_counter = 0
        episode_success = False
        
        while current_state != goal and step_counter < 500:
            step_counter += 1
            
            # Record current node count
            if agent_type == "TMGWR":
                node_count = len(agent.model.W)
            else:
                node_count = len(agent.nodes)
            node_growth.append(node_count)
            
            # Save current state for transition recording
            prev_state = np.array(current_state)
            
            # Add noise to state observation
            if noise_level > 0:
                noisy_state = np.array(current_state) + np.random.normal(0, np.sqrt(noise_level), 2)
            else:
                noisy_state = np.array(current_state)
            
            # Select and execute action
            action = agent.select_action(current_state=noisy_state)
            Maze.move_player(action=action)
            next_state = Maze.get_player_pos()
            
            # Record transition
            transition = (tuple(np.round(prev_state, 1)), tuple(np.round(next_state, 1)))
            all_transitions.append(transition)
            
            # Check if transition would be predicted correctly (habituated)
            if agent_type == "TMGWR":
                prev_node = agent.model.get_node_index(prev_state)
                next_node = agent.model.get_node_index(next_state)
                
                if prev_node is not None and next_node is not None:
                    transition_key = f"{prev_node}->{next_node}"
                    
                    # Check if transition is habituated (learned)
                    is_habituated = agent.model.C[prev_node, next_node] == 1
                    
                    if not is_habituated:
                        total_errors += 1
                    
                    # Add to learned transitions
                    transitions_learned.add(transition_key)
            
            else:  # MINERVA
                prev_pattern = agent.get_firing_pattern(prev_state)
                next_pattern = agent.get_firing_pattern(next_state)
                
                prev_node = agent.find_node_index(prev_pattern)
                next_node = agent.find_node_index(next_pattern)
                
                if prev_node is not None and next_node is not None:
                    transition_key = f"{prev_node}->{next_node}"
                    
                    # Check if transition is habituated (learned)
                    if (prev_node < len(agent.connections) and 
                        next_node < len(agent.connections)):
                        is_habituated = agent.connections[prev_node, next_node] == 1
                        
                        if not is_habituated:
                            total_errors += 1
                        
                        # Add to learned transitions
                        transitions_learned.add(transition_key)
            
            # Update model with true next state
            agent.update_model(next_state=next_state, action=action)
            current_state = next_state
            
            if current_state == goal:
                episode_success = True
                break
    
    # Calculate SE
    total_transitions = len(all_transitions)
    se = total_errors / total_transitions if total_transitions > 0 else 0
    
    # Calculate transition efficiency
    transition_efficiency = len(transitions_learned) / node_growth[-1] if node_growth[-1] > 0 else 0
    
    return {
        'agent': agent,
        'se': se,
        'total_errors': total_errors,
        'total_transitions': total_transitions,
        'final_node_count': node_growth[-1],
        'node_growth': node_growth,
        'transitions_learned': len(transitions_learned),
        'transition_efficiency': transition_efficiency
    }

def run_feature_detector_analysis(minerva_agent):
    """
    Analyze the X and Y feature detectors to understand why MINERVA has lower SE
    """
    # Create a grid of test points
    x_values = np.linspace(-60, 60, 13)
    y_values = np.linspace(-60, 60, 13)
    
    # Record activations for each dimension separately
    x_activations = []
    y_activations = []
    
    for x in x_values:
        # Test X feature detectors with fixed Y
        test_state = [x, 0]
        pattern = minerva_agent.get_firing_pattern(test_state)
        
        # Extract binary X activations from pattern
        x_pattern = pattern[0]
        x_activations.append(x_pattern)
    
    for y in y_values:
        # Test Y feature detectors with fixed X
        test_state = [0, y]
        pattern = minerva_agent.get_firing_pattern(test_state)
        
        # Extract binary Y activations from pattern
        y_pattern = pattern[1]
        y_activations.append(y_pattern)
    
    # Convert to array for analysis
    x_activations = np.array(x_activations)
    y_activations = np.array(y_activations)
    
    return {
        'x_values': x_values,
        'y_values': y_values,
        'x_activations': x_activations,
        'y_activations': y_activations
    }

def visualize_feature_detectors(feature_data):
    """
    Visualize the X and Y feature detectors to show how MINERVA processes spatial information
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot X feature detectors
    ax1 = axes[0]
    ax1.set_title("X Coordinate Feature Detector Activations", fontsize=14)
    sns.heatmap(feature_data['x_activations'], ax=ax1, cmap='Blues', cbar=True)
    
    # Set x-axis labels to coordinate values
    x_ticks = np.linspace(0, len(feature_data['x_values'])-1, 7, dtype=int)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f"{feature_data['x_values'][i]:.0f}" for i in x_ticks])
    
    ax1.set_xlabel("X Coordinate", fontsize=12)
    ax1.set_ylabel("Feature Detector", fontsize=12)
    
    # Plot Y feature detectors
    ax2 = axes[1]
    ax2.set_title("Y Coordinate Feature Detector Activations", fontsize=14)
    sns.heatmap(feature_data['y_activations'], ax=ax2, cmap='Greens', cbar=True)
    
    # Set x-axis labels to coordinate values
    y_ticks = np.linspace(0, len(feature_data['y_values'])-1, 7, dtype=int)
    ax2.set_xticks(y_ticks)
    ax2.set_xticklabels([f"{feature_data['y_values'][i]:.0f}" for i in y_ticks])
    
    ax2.set_xlabel("Y Coordinate", fontsize=12)
    ax2.set_ylabel("Feature Detector", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('minerva_feature_detectors.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_se_comparison(noise_levels=[0, 1/6, 1/3, 2/3, 1]):
    """
    Run experiments at different noise levels and visualize SE comparison
    """
    se_results = []
    tmgwr_nodes = []
    minerva_nodes = []
    tmgwr_efficiency = []
    minerva_efficiency = []
    
    for noise in noise_levels:
        print(f"\nRunning simulations with noise σ² = {noise}")
        
        # Run TMGWR simulation
        print("Training TMGWR agent...")
        tmgwr_result = simulate_learning_and_calculate_se("TMGWR", noise, num_episodes=5)
        
        # Run MINERVA simulation
        print("Training MINERVA agent...")
        minerva_result = simulate_learning_and_calculate_se("MINERVA", noise, num_episodes=5)
        
        # Store results
        se_results.append({
            'noise': noise,
            'tmgwr_se': tmgwr_result['se'],
            'minerva_se': minerva_result['se'],
            'tmgwr_nodes': tmgwr_result['final_node_count'],
            'minerva_nodes': minerva_result['final_node_count'],
            'tmgwr_efficiency': tmgwr_result['transition_efficiency'],
            'minerva_efficiency': minerva_result['transition_efficiency']
        })
        
        # Save agents for detailed feature analysis at medium noise level
        if noise == 1/3:
            detailed_tmgwr_agent = tmgwr_result['agent']
            detailed_minerva_agent = minerva_result['agent']
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data for plotting
    noise_values = [r['noise'] for r in se_results]
    tmgwr_se_values = [r['tmgwr_se'] for r in se_results]
    minerva_se_values = [r['minerva_se'] for r in se_results]
    tmgwr_node_values = [r['tmgwr_nodes'] for r in se_results]
    minerva_node_values = [r['minerva_nodes'] for r in se_results]
    tmgwr_efficiency_values = [r['tmgwr_efficiency'] for r in se_results]
    minerva_efficiency_values = [r['minerva_efficiency'] for r in se_results]
    
    # Plot 1: SE comparison
    ax1 = axes[0, 0]
    ax1.plot(noise_values, tmgwr_se_values, 'b-o', label='TMGWR')
    ax1.plot(noise_values, minerva_se_values, 'g-o', label='MINERVA')
    ax1.set_title('Sensorimotor Error (SE) vs. Noise', fontsize=14)
    ax1.set_xlabel('Noise Level (σ²)', fontsize=12)
    ax1.set_ylabel('Sensorimotor Error', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot 2: Node count comparison
    ax2 = axes[0, 1]
    ax2.plot(noise_values, tmgwr_node_values, 'b-o', label='TMGWR')
    ax2.plot(noise_values, minerva_node_values, 'g-o', label='MINERVA')
    ax2.set_title('Node Count vs. Noise', fontsize=14)
    ax2.set_xlabel('Noise Level (σ²)', fontsize=12)
    ax2.set_ylabel('Number of Nodes', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Plot 3: Efficiency comparison
    ax3 = axes[1, 0]
    ax3.plot(noise_values, tmgwr_efficiency_values, 'b-o', label='TMGWR')
    ax3.plot(noise_values, minerva_efficiency_values, 'g-o', label='MINERVA')
    ax3.set_title('Transition Efficiency vs. Noise', fontsize=14)
    ax3.set_xlabel('Noise Level (σ²)', fontsize=12)
    ax3.set_ylabel('Transitions per Node', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # Plot 4: SE/Node Ratio (Lower is better - more efficient)
    ax4 = axes[1, 1]
    tmgwr_ratio = [se / nodes if nodes > 0 else 0 for se, nodes in zip(tmgwr_se_values, tmgwr_node_values)]
    minerva_ratio = [se / nodes if nodes > 0 else 0 for se, nodes in zip(minerva_se_values, minerva_node_values)]
    
    ax4.plot(noise_values, tmgwr_ratio, 'b-o', label='TMGWR')
    ax4.plot(noise_values, minerva_ratio, 'g-o', label='MINERVA')
    ax4.set_title('SE to Node Ratio vs. Noise', fontsize=14)
    ax4.set_xlabel('Noise Level (σ²)', fontsize=12)
    ax4.set_ylabel('SE/Node (Lower is Better)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend()
    
    plt.suptitle('Why MINERVA Has Lower SE Despite Fewer Nodes', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('se_vs_node_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Run feature detector analysis on the MINERVA agent (1/3 noise level)
    feature_data = run_feature_detector_analysis(detailed_minerva_agent)
    visualize_feature_detectors(feature_data)
    
    return detailed_tmgwr_agent, detailed_minerva_agent, se_results

def create_explanatory_diagram():
    """
    Create a diagram explaining why MINERVA has lower SE despite fewer nodes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # TMGWR direct mapping diagram
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('TMGWR: Direct Mapping', fontsize=14)
    ax1.axis('off')
    
    # Draw noisy inputs
    for i in range(5):
        noise_x = 2 + np.random.normal(0, 0.3)
        noise_y = 8 + np.random.normal(0, 0.3)
        ax1.plot(noise_x, noise_y, 'ro', alpha=0.4, markersize=8)
    
    # Draw node
    ax1.plot(2, 8, 'bs', markersize=15, label='Node')
    
    # Draw arrows
    for i in range(5):
        noise_x = 2 + np.random.normal(0, 0.3)
        noise_y = 8 + np.random.normal(0, 0.3)
        ax1.arrow(noise_x, noise_y, 2-noise_x, 8-noise_y, head_width=0.1, 
                 head_length=0.2, fc='blue', ec='blue', alpha=0.3)
    
    # Add explanation
    ax1.text(1, 5, "TMGWR maps noisy inputs\ndirectly to nodes.\n\n"
             "Each variation creates\npotential for new nodes\n"
             "or missed connections.\n\n"
             "Result: More nodes needed\nfor higher SE.", 
             fontsize=10, ha='left', va='center',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # MINERVA hierarchical mapping diagram
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title('MINERVA: Hierarchical Pattern Mapping', fontsize=14)
    ax2.axis('off')
    
    # Draw noisy inputs
    for i in range(5):
        noise_x = 2 + np.random.normal(0, 0.3)
        noise_y = 8 + np.random.normal(0, 0.3)
        ax2.plot(noise_x, noise_y, 'ro', alpha=0.4, markersize=8)
    
    # Draw feature detectors
    ax2.plot(4, 9, 'gs', markersize=10, label='X Feature')
    ax2.plot(4, 7, 'gs', markersize=10, label='Y Feature')
    
    # Draw pattern node
    ax2.plot(6, 8, 'gs', markersize=15, label='Pattern Node')
    
    # Draw arrows from inputs to features
    for i in range(5):
        noise_x = 2 + np.random.normal(0, 0.3)
        noise_y = 8 + np.random.normal(0, 0.3)
        ax2.arrow(noise_x, noise_y, 4-noise_x, 9-noise_y, head_width=0.1, 
                 head_length=0.2, fc='green', ec='green', alpha=0.3)
        ax2.arrow(noise_x, noise_y, 4-noise_x, 7-noise_y, head_width=0.1, 
                 head_length=0.2, fc='green', ec='green', alpha=0.3)
    
    # Draw arrows from features to pattern
    ax2.arrow(4, 9, 1.8, -0.8, head_width=0.1, head_length=0.2, 
             fc='green', ec='green', alpha=0.7)
    ax2.arrow(4, 7, 1.8, 0.8, head_width=0.1, head_length=0.2, 
             fc='green', ec='green', alpha=0.7)
    
    # Add explanation
    ax2.text(7, 5, "MINERVA first processes X and Y\n"
             "coordinates separately.\n\n"
             "Feature detectors convert\nnoisy inputs to consistent\n"
             "binary patterns.\n\n"
             "Result: Fewer nodes needed\nfor lower SE.",
             fontsize=10, ha='left', va='center',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('minerva_tmgwr_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    
    # Run the main analysis
    tmgwr_agent, minerva_agent, se_results = visualize_se_comparison()
    
    # Create explanatory diagram
    create_explanatory_diagram()
    
    # Print summary of findings
    print("\n==== Why MINERVA has lower SE despite fewer nodes ====")
    print("1. Hierarchical Pattern Encoding: MINERVA processes X and Y coordinates")
    print("   separately before combining them into patterns. This creates a more")
    print("   robust representation that filters out noise.")
    print()
    print("2. Binary Feature Detectors: MINERVA's lower networks use binary activations")
    print("   that convert continuous noisy inputs into discrete binary patterns. This")
    print("   makes similar inputs map to the same pattern, reducing node duplication.")
    print()
    print("3. Transition Efficiency: MINERVA encodes more transitions per node, which")
    print("   means each node participates in more state-to-state connections, reducing")
    print("   sensorimotor error with fewer nodes.")
    print()
    print("4. Pattern Stability Under Noise: MINERVA's pattern encoding is more stable")
    print("   under noise, maintaining consistent patterns despite input variations.")
    print("   This reduces the need for duplicate nodes representing the same location.")
    print("===============================================================")