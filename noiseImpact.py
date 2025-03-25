import numpy as np
import matplotlib.pyplot as plt
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.TMGWR_agent import TMGWRAgent
from Agents.HSOM_binary import HierarchicalGWRSOMAgent
import networkx as nx
import os
from collections import defaultdict, Counter

# Change the current working directory to the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

def run_experiment(agent_type, noise_level=0, num_episodes=5):
    """Run experiment with specified agent type and noise level"""
    # Get maze details 
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()
    Maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos_index, 
                     goal_index_pos=goal_pos_index, display_maze=False)
    goal = Maze.get_goal_pos()
    initial_state = Maze.get_initial_player_pos()
    
    print(f"Initial state: {initial_state}, Goal: {goal}")

    # Initialize agent based on type
    if agent_type == "TMGWR":
        agent = TMGWRAgent(nDim=2, Ni=2, epsilon_b=0.55194, epsilon_n=0.90, beta=0.8, 
                         delta=0.6235, T_max=17, N_max=300, eta=0.95, phi=0.6, sigma=1)
    else:  # MINERVA / Hierarchical GWRSOM
        agent = HierarchicalGWRSOMAgent(
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
        # For MINERVA, we need to train lower networks first
        training_data = []
        for _ in range(50):
            state = [np.random.randint(-72, 72), np.random.randint(-72, 72)]
            training_data.append(state)
        training_data = np.array(training_data)
        agent.train_lower_networks(training_data, epochs=20)

    agent.set_goal(goal=goal)
    agent.set_epsilon(1)

    # Track state to node mappings for analyzing node duplication
    state_to_node_map = defaultdict(list)
    node_visit_counts = Counter()
    
    # Training statistics
    training_stats = {
        'episodes': [],
        'steps': [],
        'epsilon': [],
        'success': [],
        'node_growth': []  # Track node growth over steps
    }

    # Parameters
    reached_goal_count = 0
    node_count_history = []

    print(f"\nStarting training {agent_type} with noise level σ² = {noise_level}")

    # Training loop
    for episode_num in range(num_episodes):     
        current_state = initial_state
        Maze.reset_player() 
        step_counter = 0
        episode_success = False
        episode_node_growth = []

        while current_state != goal and step_counter < 5000:  # Reduced max steps for quicker testing
            step_counter += 1

            # Add noise to state observation if noise_level > 0
            if noise_level > 0:
                noisy_state = current_state + np.random.normal(0, np.sqrt(noise_level), 2)
            else:
                noisy_state = current_state

            # Get current node index for the state
            if agent_type == "TMGWR":
                current_node = agent.model.get_node_index(noisy_state)
            else:  # MINERVA
                pattern = agent.get_firing_pattern(noisy_state)
                current_node = agent.find_node_index(pattern)
            
            # Record state-node relationship
            if current_node is not None:
                state_key = tuple(np.round(current_state, 2))  # Round to reduce slight variations
                state_to_node_map[state_key].append(current_node)
                node_visit_counts[current_node] += 1
            
            # Track node growth
            if agent_type == "TMGWR":
                current_node_count = len(agent.model.W)
            else:  # MINERVA
                current_node_count = len(agent.nodes)
            
            node_count_history.append(current_node_count)
            episode_node_growth.append(current_node_count)

            # Select and execute action
            action = agent.select_action(current_state=noisy_state)
            Maze.move_player(action=action)
            next_state = Maze.get_player_pos() 

            # Update model with true next state
            agent.update_model(next_state=next_state, action=action)
            current_state = next_state
            
            if step_counter % 1000 == 0: 
                print(f"Episode {episode_num + 1}, step {step_counter}, noise σ² = {noise_level}")

            # Check if goal reached
            if current_state == goal:
                episode_success = True
                break

        # Update epsilon
        if episode_success:
            reached_goal_count += 1
            if reached_goal_count > 5:  # Reduced for quicker convergence 
                agent.decay_epsilon(min_epsilon=0.2) 

        # Record statistics
        training_stats['episodes'].append(episode_num + 1)
        training_stats['steps'].append(step_counter)
        training_stats['epsilon'].append(agent.get_epsilon())
        training_stats['success'].append(episode_success)
        training_stats['node_growth'].append(episode_node_growth)

        print(f"Episode: {episode_num + 1}, Steps: {step_counter}, "
              f"Epsilon: {agent.get_epsilon()}, Success: {episode_success}\n")

    # Analyze node duplication
    duplicated_states = 0
    unique_states = 0
    multi_node_states = []
    
    for state, nodes in state_to_node_map.items():
        unique_nodes = set(nodes)
        if len(unique_nodes) > 1:
            duplicated_states += 1
            multi_node_states.append((state, list(unique_nodes)))
        else:
            unique_states += 1
    
    duplication_stats = {
        'unique_states': unique_states,
        'duplicated_states': duplicated_states,
        'duplication_rate': duplicated_states / (unique_states + duplicated_states) if (unique_states + duplicated_states) > 0 else 0,
        'multi_node_states': multi_node_states,
        'node_visit_counts': node_visit_counts,
        'node_count_history': node_count_history
    }

    return agent, training_stats, Maze, duplication_stats

def extract_node_data(agent, agent_type):
    """Extract node data from an agent for visualization"""
    if agent_type == "TMGWR":
        nodes = agent.model.W
        connections = agent.model.C
        
        # Filter out any NaN values that might exist
        valid_indices = ~np.isnan(nodes[:, 0])
        nodes = nodes[valid_indices]
        
        # Print node statistics for debugging
        print(f"TMGWR node count: {len(nodes)}")
        if len(nodes) > 0:
            # Round node coordinates to nearest integer for cleaner presentation
            nodes = np.round(nodes)
            print(f"TMGWR node range: X [{np.min(nodes[:, 0])}, {np.max(nodes[:, 0])}], Y [{np.min(nodes[:, 1])}, {np.max(nodes[:, 1])}]")
        
        # Adjust connections matrix as well
        connections = connections[valid_indices][:, valid_indices]
        
        # Create a graph
        graph = nx.DiGraph()
        
        # Add nodes with positions and frequency data
        for i, node_pos in enumerate(nodes):
            graph.add_node(i, pos=tuple(node_pos))
        
        # Add edges based on connection matrix
        rows, cols = np.where(connections == 1)
        for r, c in zip(rows, cols):
            if r < len(nodes) and c < len(nodes):  # Safety check
                graph.add_edge(r, c)
            
    else:  # MINERVA
        # For MINERVA, the node positions are stored differently
        graph = nx.DiGraph()
        
        # Extract node positions for statistics
        positions = np.array(list(agent.node_positions.values()))
        
        if len(positions) > 0:
            # Round node coordinates to nearest integer for cleaner presentation
            positions = np.round(positions)
            
            print(f"MINERVA node count: {len(positions)}")
            print(f"MINERVA node range: X [{np.min(positions[:, 0])}, {np.max(positions[:, 0])}], Y [{np.min(positions[:, 1])}, {np.max(positions[:, 1])}]")
            
            # Add nodes with rounded positions
            for i, position in agent.node_positions.items():
                graph.add_node(i, pos=tuple(np.round(position)))
        else:
            print("MINERVA has no nodes")
            
        # Add edges based on connections matrix if any nodes exist
        if agent.connections.shape[0] > 0:
            rows, cols = np.where(agent.connections == 1)
            edges = zip(rows.tolist(), cols.tolist())
            graph.add_edges_from(edges)
    
    return graph

def get_maze_cell_positions(maze_map, cell_size=24):
    """Calculate center positions of all available (non-wall) cells in the maze"""
    height = len(maze_map)
    width = len(maze_map[0])
    
    # Calculate screen offsets (from MazePlayer._calc_screen_coordinates)
    x_offset = -int((width * cell_size / 2) - cell_size/2)
    y_offset = int((height * cell_size/2) - cell_size/2)
    
    # Debug offsets
    print(f"Maze dimensions: {height}x{width}")
    print(f"Screen offsets: x={x_offset}, y={y_offset}")
    
    # Calculate cell centers for all non-wall cells
    available_cells = []
    for row_idx in range(height):
        for col_idx in range(width):
            if maze_map[row_idx][col_idx] != 'X':
                # Calculate cell center in screen coordinates
                cell_x = x_offset + (col_idx * cell_size) + (cell_size / 2)
                cell_y = y_offset - (row_idx * cell_size) - (cell_size / 2)
                available_cells.append((cell_x, cell_y))
                # Debug first few cells
                if len(available_cells) <= 5:
                    print(f"Cell ({row_idx},{col_idx}) center: ({cell_x},{cell_y})")
    
    return available_cells

def adjust_node_positions_to_cells(graph, available_cells):
    """Adjust node positions to match closest available cell centers"""
    pos = nx.get_node_attributes(graph, 'pos')
    adjusted_pos = {}
    
    # For each node, find closest available cell
    for node, coords in pos.items():
        node_pos = np.array(coords)
        
        # Find closest available cell
        distances = [np.linalg.norm(np.array(cell) - node_pos) for cell in available_cells]
        closest_idx = np.argmin(distances)
        closest_cell = available_cells[closest_idx]
        
        # Use the exact cell center
        adjusted_pos[node] = closest_cell
    
    # Set new positions
    nx.set_node_attributes(graph, adjusted_pos, 'pos')
    return graph

def visualize_node_overlap(maze_map, player_pos_index, goal_pos_index, tmgwr_graph, minerva_graph, tmgwr_stats, minerva_stats, noise_level):
    """Create a visualization showing node duplication"""
    # Get the maze dimensions and convert to pixel dimensions
    height = len(maze_map)
    width = len(maze_map[0])
    cell_size = 24  # MazePlayer.MAZE_BLOCK_PIXEL_WIDTH
    
    # Create a figure with 3 subplots: 2 for agents and 1 for node growth
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(2, 2, height_ratios=[3, 1])
    
    # Agent visualization subplots
    ax1 = fig.add_subplot(gs[0, 0])  # TMGWR
    ax2 = fig.add_subplot(gs[0, 1])  # MINERVA
    ax3 = fig.add_subplot(gs[1, :])  # Node growth over time
    
    fig.suptitle(f"Impact of Noise (σ² = {noise_level}) on Node Creation and Duplication", fontsize=16)
    
    # Calculate screen offsets
    x_offset = -int((width * cell_size / 2) - cell_size/2)
    y_offset = int((height * cell_size/2) - cell_size/2)
    
    # Calculate player and goal positions (center of cells)
    player_x = x_offset + (player_pos_index[1] * cell_size) + (cell_size / 2)
    player_y = y_offset - (player_pos_index[0] * cell_size) - (cell_size / 2)
    
    goal_x = x_offset + (goal_pos_index[1] * cell_size) + (cell_size / 2)
    goal_y = y_offset - (goal_pos_index[0] * cell_size) - (cell_size / 2)
    
    # Plot the maze on both agent subplots
    for ax_idx, ax in enumerate([ax1, ax2]):
        # Draw the maze walls
        for i, row in enumerate(maze_map):
            for j, cell in enumerate(row):
                if cell == 'X':
                    # Calculate screen coordinates for this cell
                    cell_x = x_offset + (j * cell_size)
                    cell_y = y_offset - (i * cell_size)
                    rect = plt.Rectangle((cell_x, cell_y), cell_size, -cell_size, 
                                        facecolor='gray', edgecolor='black')
                    ax.add_patch(rect)
                else:
                    # Draw a subtle grid for open spaces
                    cell_x = x_offset + (j * cell_size)
                    cell_y = y_offset - (i * cell_size)
                    rect = plt.Rectangle((cell_x, cell_y), cell_size, -cell_size, 
                                        facecolor='none', edgecolor='lightgray', linestyle=':')
                    ax.add_patch(rect)
        
        # Draw player position and goal
        ax.plot(player_x, player_y, 'bo', markersize=10, label='Start')
        ax.plot(goal_x, goal_y, 'go', markersize=10, label='Goal')
        
        # Set axis limits to show the entire maze
        ax.set_xlim(x_offset - cell_size, 
                   x_offset + width * cell_size + cell_size)
        ax.set_ylim(y_offset - height * cell_size - cell_size, 
                   y_offset + cell_size)
        
        ax.set_aspect('equal')
        ax.grid(False)  # Disable matplotlib's grid, we're drawing our own
    
    # Draw TMGWR nodes
    ax1.set_title(f'TMGWR Agent Nodes (Count: {len(nx.get_node_attributes(tmgwr_graph, "pos"))})', fontsize=14)
    pos = nx.get_node_attributes(tmgwr_graph, 'pos')
    
    # Draw nodes with numbers
    nx.draw_networkx_nodes(tmgwr_graph, pos, ax=ax1, node_color='skyblue', 
                          node_size=100, alpha=0.7)
    nx.draw_networkx_labels(tmgwr_graph, pos, ax=ax1, font_size=8)
    nx.draw_networkx_edges(tmgwr_graph, pos, ax=ax1, width=1.0, 
                          edge_color='blue', alpha=0.6, arrowsize=10, 
                          arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
    # Draw MINERVA nodes
    ax2.set_title(f'MINERVA Agent Nodes (Count: {len(nx.get_node_attributes(minerva_graph, "pos"))})', fontsize=14)
    pos = nx.get_node_attributes(minerva_graph, 'pos')
    
    # Draw nodes with numbers
    nx.draw_networkx_nodes(minerva_graph, pos, ax=ax2, node_color='lightgreen', 
                          node_size=100, alpha=0.7)
    nx.draw_networkx_labels(minerva_graph, pos, ax=ax2, font_size=8)
    nx.draw_networkx_edges(minerva_graph, pos, ax=ax2, width=1.0, 
                          edge_color='green', alpha=0.6, arrowsize=10, 
                          arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
    # Plot node growth over time
    ax3.set_title('Node Growth During Training', fontsize=14)
    
    # Extract node growth data for both agents
    tmgwr_node_growth = tmgwr_stats['node_count_history']
    minerva_node_growth = minerva_stats['node_count_history']
    
    # Plot lines
    steps = range(len(tmgwr_node_growth))
    ax3.plot(steps, tmgwr_node_growth, 'b-', label='TMGWR')
    steps = range(len(minerva_node_growth))
    ax3.plot(steps, minerva_node_growth, 'g-', label='MINERVA')
    
    ax3.set_xlabel('Steps', fontsize=12)
    ax3.set_ylabel('Number of Nodes', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # Add summary text about duplication
    tmgwr_dup = tmgwr_stats['duplication_rate'] * 100
    minerva_dup = minerva_stats['duplication_rate'] * 100
    
    ax1.text(0.05, 0.05, f"Duplication rate: {tmgwr_dup:.1f}%\nUnique states: {tmgwr_stats['unique_states']}\nDuplicated states: {tmgwr_stats['duplicated_states']}",
             transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    ax2.text(0.05, 0.05, f"Duplication rate: {minerva_dup:.1f}%\nUnique states: {minerva_stats['unique_states']}\nDuplicated states: {minerva_stats['duplicated_states']}",
             transform=ax2.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'noise_analysis_sigma_{noise_level:.2f}.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_node_duplication(duplication_stats, agent_type, noise_level):
    """Create detailed analysis of node duplication"""
    print(f"\n--- Node Duplication Analysis for {agent_type} at σ² = {noise_level} ---")
    print(f"Total unique states visited: {duplication_stats['unique_states'] + duplication_stats['duplicated_states']}")
    print(f"States with single node mapping: {duplication_stats['unique_states']}")
    print(f"States with multiple node mappings: {duplication_stats['duplicated_states']}")
    print(f"Duplication rate: {duplication_stats['duplication_rate'] * 100:.2f}%")
    
    # Histogram of node visit frequency
    node_visits = list(duplication_stats['node_visit_counts'].values())
    if node_visits:
        plt.figure(figsize=(10, 6))
        plt.hist(node_visits, bins=20, alpha=0.7)
        plt.title(f'{agent_type} Node Visit Frequency at σ² = {noise_level}')
        plt.xlabel('Number of Visits')
        plt.ylabel('Number of Nodes')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{agent_type}_node_visits_sigma_{noise_level:.2f}.png', dpi=300)
        plt.show()
    
    # List top 5 most duplicated states
    if duplication_stats['multi_node_states']:
        multi_node_states = sorted(duplication_stats['multi_node_states'], 
                                 key=lambda x: len(x[1]), reverse=True)
        print("\nTop 5 most duplicated states:")
        for i, (state, nodes) in enumerate(multi_node_states[:5]):
            print(f"{i+1}. State {state} maps to {len(nodes)} different nodes: {nodes}")
    else:
        print("\nNo state duplications found.")

def run_noise_comparison(noise_levels=[0, 1/6, 2/3, 4/3]):
    """Run comparison for multiple noise levels"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get maze information first to use in visualization
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()
    
    # Get available cell positions for node adjustment
    available_cells = get_maze_cell_positions(maze_map)
    print(f"Found {len(available_cells)} available cells in maze")
    
    for noise_level in noise_levels:
        print(f"\n=== Running comparison with noise level σ² = {noise_level} ===")
        
        # Train both agents
        print("Training TMGWR agent...")
        tmgwr_agent, tmgwr_stats, tmgwr_maze, tmgwr_duplication = run_experiment("TMGWR", noise_level, num_episodes=5)
        
        print("Training MINERVA agent...")
        minerva_agent, minerva_stats, minerva_maze, minerva_duplication = run_experiment("MINERVA", noise_level, num_episodes=5)
        
        # Analyze node duplication for both agents
        analyze_node_duplication(tmgwr_duplication, "TMGWR", noise_level)
        analyze_node_duplication(minerva_duplication, "MINERVA", noise_level)
        
        # Extract node data
        tmgwr_graph = extract_node_data(tmgwr_agent, "TMGWR")
        minerva_graph = extract_node_data(minerva_agent, "MINERVA")
        
        # Adjust node positions to match available cell centers
        tmgwr_graph = adjust_node_positions_to_cells(tmgwr_graph, available_cells)
        minerva_graph = adjust_node_positions_to_cells(minerva_graph, available_cells)
        
        # Store node count history in stats for visualization
        tmgwr_stats['node_count_history'] = tmgwr_duplication['node_count_history']
        minerva_stats['node_count_history'] = minerva_duplication['node_count_history']
        
        # Visualize node duplication
        visualize_node_overlap(maze_map, player_pos_index, goal_pos_index, 
                               tmgwr_graph, minerva_graph, 
                               tmgwr_duplication, minerva_duplication,
                               noise_level)

if __name__ == "__main__":
    # Run comparison for multiple noise levels
    run_noise_comparison()