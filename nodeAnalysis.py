import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.HSOM_binary import HierarchicalGWRSOMAgent
import random
import os

def detailed_minerva_analysis():
    """
    Generate a detailed analysis of all MINERVA nodes, showing:
    - Each point encountered in the maze environment
    - The exact binary pattern generated for each point
    - The node index assigned to each point
    - Visualization of all binary patterns and their distribution
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Get maze environment
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()
    
    # Initialize maze environment
    maze = MazePlayer(maze_map=maze_map, 
                    player_index_pos=player_pos_index, 
                    goal_index_pos=goal_pos_index,
                    display_maze=False)
    
    goal = maze.get_goal_pos()
    initial_state = maze.get_initial_player_pos()
    
    # Training data for MINERVA
    x_train = np.linspace(-72, 72, 10).reshape(-1, 1)
    y_train = np.linspace(-72, 72, 10).reshape(-1, 1)
    training_data = np.hstack((x_train, y_train))
    
    # Initialize MINERVA agent
    minerva_agent = HierarchicalGWRSOMAgent(
        lower_dim=1, higher_dim=2, epsilon_b=0.35,
        epsilon_n=0.15, beta=0.7, delta=0.79,
        T_max=20, N_max=100, eta=0.5,
        phi=0.9, sigma=0.5
    )
    
    # Initialize nodes list and connections matrix for MINERVA
    minerva_agent.nodes = []
    minerva_agent.connections = np.zeros((0, 0))
    
    # Train lower networks for MINERVA
    print("Training MINERVA lower networks...")
    minerva_agent.train_lower_networks(training_data, epochs=10)
    minerva_agent.set_goal(goal)
    minerva_agent.set_epsilon(1)
    
    # Storage for collected data
    points = []  # Store the points visited
    binary_patterns = []  # Store binary patterns
    node_indices = []  # Store node indices
    
    # Track unique patterns and their node indices
    unique_patterns = {}  # Maps pattern string to node index
    
    # To fix the empty array issue, we'll ensure x_active and y_active indices are always
    # assigned valid values, and handle exceptions more gracefully
    
    # Explore the maze more thoroughly to collect data
    noise_level = 0.0  # No noise for clearer analysis
    current_state = initial_state
    visited_states = set()
    max_steps = 1000
    step_counter = 0
    
    # Record active bits for both x and y coordinates
    x_active_bits = []
    y_active_bits = []
    
    print("\nCollecting data points and binary patterns...")
    
    while step_counter < max_steps:
        step_counter += 1
        
        # Add current state to points list
        current_state_tuple = tuple(current_state)
        if current_state_tuple not in visited_states:
            visited_states.add(current_state_tuple)
            points.append(np.array(current_state))
            
            # Get binary pattern
            x_data = np.array([current_state[0]]).reshape(1, -1)
            y_data = np.array([current_state[1]]).reshape(1, -1)
            
            try:
                # Get BMUs from lower networks
                x_bmu, _ = minerva_agent.lower_x.find_best_matching_units(x_data)
                y_bmu, _ = minerva_agent.lower_y.find_best_matching_units(y_data)
                
                # Create binary vectors
                x_binary = np.zeros(len(minerva_agent.lower_x.A))
                y_binary = np.zeros(len(minerva_agent.lower_y.A))
                
                # Handle both scalar and array returns
                if np.isscalar(x_bmu):
                    x_binary[x_bmu] = 1
                    x_active = x_bmu
                else:
                    x_binary[x_bmu[0]] = 1
                    x_active = x_bmu[0]
                    
                if np.isscalar(y_bmu):
                    y_binary[y_bmu] = 1
                    y_active = y_bmu
                else:
                    y_binary[y_bmu[0]] = 1
                    y_active = y_bmu[0]
                
                # Store active bits
                x_active_bits.append(x_active)
                y_active_bits.append(y_active)
                
                # Store the binary pattern
                pattern = (tuple(x_binary), tuple(y_binary))
                pattern_str = str(pattern)
                binary_patterns.append(pattern)
                
                # Find or create node index
                if pattern_str in unique_patterns:
                    node_idx = unique_patterns[pattern_str]
                else:
                    node_idx = len(unique_patterns)
                    unique_patterns[pattern_str] = node_idx
                
                node_indices.append(node_idx)
                
            except Exception as e:
                print(f"Error processing point {current_state}: {e}")
                # Skip this point
                visited_states.remove(current_state_tuple)
                continue
        
        # Take a random action to explore the maze
        action = np.random.randint(0, 4)
        maze.move_player(action)
        current_state = maze.get_player_pos()
        
        # Reset to a random position occasionally for better coverage
        if step_counter % 50 == 0:
            valid_positions = []
            for i in range(len(maze_map)):
                for j in range(len(maze_map[i])):
                    if maze_map[i][j] == ' ':
                        valid_positions.append((i, j))
            
            random_pos = valid_positions[np.random.randint(0, len(valid_positions))]
            maze.current_player_index_pos = random_pos
            current_state = maze.get_player_pos()
    
    if len(points) == 0:
        print("No valid points collected. Cannot continue analysis.")
        return None, None
    
    # Convert data to a DataFrame for easier analysis
    data = []
    for i in range(len(points)):
        data.append({
            'Point': tuple(points[i]),
            'X Coordinate': points[i][0],
            'Y Coordinate': points[i][1],
            'X Active Bit': x_active_bits[i],
            'Y Active Bit': y_active_bits[i],
            'Node Index': node_indices[i],
            'Binary Pattern': str(binary_patterns[i])
        })
    
    df = pd.DataFrame(data)
    
    # Sort by node index
    df_sorted = df.sort_values(by=['Node Index', 'X Coordinate', 'Y Coordinate'])
    
    # Print unique node count
    num_unique_nodes = df['Node Index'].nunique()
    print(f"\nFound {num_unique_nodes} unique MINERVA nodes from {len(df)} points")
    
    # Save full data to CSV
    df_sorted.to_csv('minerva_node_analysis.csv', index=False)
    print("Full data saved to 'minerva_node_analysis.csv'")
    
    # Create summary of nodes
    node_summary = df.groupby('Node Index').agg({
        'Point': 'count',
        'X Active Bit': lambda x: x.iloc[0],
        'Y Active Bit': lambda x: x.iloc[0],
        'Binary Pattern': lambda x: x.iloc[0]
    }).reset_index()
    node_summary.columns = ['Node Index', 'Point Count', 'X Active Bit', 'Y Active Bit', 'Binary Pattern']
    node_summary = node_summary.sort_values(by='Node Index')
    
    # Save node summary to CSV
    node_summary.to_csv('minerva_node_summary.csv', index=False)
    print("Node summary saved to 'minerva_node_summary.csv'")
    
    # Print node summary
    print("\nNode Summary (Top 20 nodes):")
    print(node_summary.head(20).to_string())
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Scatter plot of points colored by node index
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['X Coordinate'], df['Y Coordinate'], c=df['Node Index'], 
                          cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Node Index')
    plt.title(f'MINERVA Node Assignment ({num_unique_nodes} unique nodes)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    plt.savefig('minerva_node_map.png')
    print("Saved node map to 'minerva_node_map.png'")
    
    # 2. Visualization of X and Y active bits
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.hist(df['X Active Bit'], bins=range(int(min(df['X Active Bit'])), int(max(df['X Active Bit']))+2), 
             alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of X Active Bits')
    plt.xlabel('Active Bit Index')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(df['Y Active Bit'], bins=range(int(min(df['Y Active Bit'])), int(max(df['Y Active Bit']))+2), 
             alpha=0.7, color='orange', edgecolor='black')
    plt.title('Distribution of Y Active Bits')
    plt.xlabel('Active Bit Index')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('minerva_bit_distribution.png')
    print("Saved bit distribution to 'minerva_bit_distribution.png'")
    
    # 3. Visualization of binary pattern grid - Fix the issue with empty arrays
    try:
        # Extract unique X and Y active bits
        unique_x_bits = sorted(df['X Active Bit'].unique())
        unique_y_bits = sorted(df['Y Active Bit'].unique())
        
        # Create a safe version of the lookup function
        def safe_index_lookup(array, value):
            indices = np.where(array == value)[0]
            if len(indices) > 0:
                return indices[0]
            else:
                # If not found, add it to the array
                new_array = np.append(array, value)
                return len(array)  # Return the index of the new element
        
        # Create a grid showing which pattern corresponds to which node
        pattern_grid = np.full((len(unique_y_bits), len(unique_x_bits)), -1)
        
        # Safe version of the grid filling
        for _, row in df.drop_duplicates(['X Active Bit', 'Y Active Bit']).iterrows():
            try:
                x_idx = np.where(np.array(unique_x_bits) == row['X Active Bit'])[0]
                y_idx = np.where(np.array(unique_y_bits) == row['Y Active Bit'])[0]
                
                if len(x_idx) > 0 and len(y_idx) > 0:
                    pattern_grid[y_idx[0], x_idx[0]] = row['Node Index']
            except Exception as e:
                print(f"Warning: Could not place node in grid: {e}")
                continue
        
        plt.figure(figsize=(14, 10))
        im = plt.imshow(pattern_grid, cmap='viridis')
        plt.colorbar(im, label='Node Index')
        plt.title('MINERVA Binary Pattern Grid')
        plt.xlabel('X Active Bit Index')
        plt.ylabel('Y Active Bit Index')
        
        # Add x and y tick labels
        plt.xticks(range(len(unique_x_bits)), unique_x_bits)
        plt.yticks(range(len(unique_y_bits)), unique_y_bits)
        
        # Add grid lines to better show the cells
        plt.grid(which='both', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('minerva_pattern_grid.png')
        print("Saved pattern grid to 'minerva_pattern_grid.png'")
    except Exception as e:
        print(f"Could not create pattern grid visualization: {e}")
    
    # 4. Create a 2D heatmap showing node density
    try:
        # Create a 2D histogram of points
        plt.figure(figsize=(12, 8))
        plt.hist2d(df['X Coordinate'], df['Y Coordinate'], bins=20, cmap='viridis')
        plt.colorbar(label='Count')
        plt.title('Density of Points')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.tight_layout()
        plt.savefig('point_density_map.png')
        print("Saved point density map to 'point_density_map.png'")
    except Exception as e:
        print(f"Could not create point density map: {e}")
    
    # 5. Create a visualization showing unique binary patterns
    try:
        # Get a sample of unique patterns (up to 20)
        unique_patterns_df = df.drop_duplicates('Binary Pattern').head(20)
        
        plt.figure(figsize=(20, 15))
        
        for i, (_, row) in enumerate(unique_patterns_df.iterrows()):
            # Get the binary pattern and convert to numpy array
            pattern = eval(row['Binary Pattern'])
            x_pattern = np.array(pattern[0])
            y_pattern = np.array(pattern[1])
            
            # Create a subplot
            plt.subplot(4, 5, i+1)
            
            # Plot X pattern
            plt.bar(range(len(x_pattern)), x_pattern, width=0.4, color='blue', label='X Pattern')
            # Plot Y pattern
            plt.bar([j + 0.4 for j in range(len(y_pattern))], y_pattern, width=0.4, color='orange', label='Y Pattern')
            
            plt.title(f"Node {row['Node Index']}")
            if i % 5 == 0:  # Add label only to leftmost plots
                plt.ylabel('Value')
            if i >= 15:  # Add label only to bottom plots
                plt.xlabel('Bit Index')
            
            # Only add legend to first plot
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig('unique_binary_patterns.png')
        print("Saved unique binary patterns to 'unique_binary_patterns.png'")
    except Exception as e:
        print(f"Could not create unique binary patterns visualization: {e}")
    
    # Try to show plots
    try:
        plt.show()
    except:
        print("Note: plt.show() failed, but the images were saved.")
    
    # Return the dataframe for further analysis
    return df, node_summary

if __name__ == "__main__":
    try:
        df, node_summary = detailed_minerva_analysis()
        
        if df is not None:
            # Print detailed analysis of the first 20 points
            print("\nDetailed Analysis of First 20 Points:")
            for i, row in df.head(20).iterrows():
                print(f"Point {row['Point']}:")
                print(f"  X Active Bit: {row['X Active Bit']}, Y Active Bit: {row['Y Active Bit']}")
                print(f"  Node Index: {row['Node Index']}")
                print(f"  Binary Pattern: {row['Binary Pattern']}")
                print()
    except Exception as e:
        print(f"An error occurred during analysis: {e}")