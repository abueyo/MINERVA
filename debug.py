"""
Standalone MINERVA Agent Debugging Script
This script isolates the MINERVA agent to understand its node creation process.
"""

import numpy as np
import matplotlib.pyplot as plt
from Maze.Mazes import MazeMaps
from Maze.Maze_player import MazePlayer
from Agents.HSOM_binary import HierarchicalGWRSOMAgent
import sys

def debug_minerva_agent():
    """Thoroughly debug the MINERVA agent to understand node creation issues"""
    print("=== MINERVA Agent Debugging ===")
    
    # 1. Basic setup - identical to the main experiment
    maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()
    Maze = MazePlayer(maze_map=maze_map, 
                    player_index_pos=player_pos_index, 
                    goal_index_pos=goal_pos_index)
    
    goal = Maze.get_goal_pos()
    initial_state = Maze.get_initial_player_pos()
    
    print(f"Initial state: {initial_state}, Goal: {goal}")
    
    # Create training data
    x_train = np.linspace(-120, 120, 10).reshape(-1, 1)
    y_train = np.linspace(-120, 120, 10).reshape(-1, 1)
    training_data = np.hstack((x_train, y_train))
    
    # 2. Create and inspect the MINERVA agent
    agent = HierarchicalGWRSOMAgent(
        lower_dim=1, higher_dim=2, epsilon_b=0.35,
        epsilon_n=0.15, beta=0.7, delta=0.79,
        T_max=20, N_max=100, eta=0.5,
        phi=0.9, sigma=0.5
    )
    
    # Print the agent's attributes
    print("\nOriginal agent attributes:")
    attribute_names = dir(agent)
    for attr in attribute_names:
        if not attr.startswith('__'):
            try:
                val = getattr(agent, attr)
                print(f"  {attr}: {type(val)}")
                if hasattr(val, 'shape'):
                    print(f"    Shape: {val.shape}")
                elif isinstance(val, list):
                    print(f"    Length: {len(val)}")
            except Exception as e:
                print(f"  {attr}: [Error accessing: {e}]")
    
    # 3. Train lower networks
    print("\nTraining lower networks...")
    agent.train_lower_networks(training_data, epochs=10)
    agent.set_goal(goal)
    
    # 4. Print lower network details
    print("\nLower network details:")
    print(f"  lower_x.A shape: {agent.lower_x.A.shape}")
    print(f"  lower_y.A shape: {agent.lower_y.A.shape}")
    
    # 5. Manually create a firing pattern and check its structure
    print("\nCreating a test firing pattern...")
    test_state = initial_state
    pattern = agent.get_firing_pattern(test_state)
    print(f"  Pattern type: {type(pattern)}")
    print(f"  Pattern structure: {pattern}")
    print(f"  X vector length: {len(pattern[0])}")
    print(f"  Y vector length: {len(pattern[1])}")
    
    # 6. Try finding this node index
    print("\nSearching for node index...")
    node_idx = agent.find_node_index(pattern)
    print(f"  Node index: {node_idx}")
    
    # 7. Directly inspect the nodes list 
    print("\nNodes list details:")
    if hasattr(agent, 'nodes'):
        print(f"  agent.nodes exists: {len(agent.nodes)} items")
    else:
        print("  agent.nodes does not exist")
    
    if hasattr(agent, 'higher_nodes'):
        print(f"  agent.higher_nodes exists: {len(agent.higher_nodes)} items")
    else:
        print("  agent.higher_nodes does not exist")
    
    # 8. Manually update the model to create a node
    print("\nManually creating a node...")
    agent.update_model(test_state, 0)  # action 0
    
    # 9. Check if node was created
    print("\nChecking node creation results:")
    if hasattr(agent, 'nodes'):
        print(f"  agent.nodes now has: {len(agent.nodes)} items")
    else:
        print("  agent.nodes does not exist")
    
    if hasattr(agent, 'higher_nodes'):
        print(f"  agent.higher_nodes now has: {len(agent.higher_nodes)} items")
    else:
        print("  agent.higher_nodes does not exist")
    
    # 10. Try creating a node with a different state
    print("\nCreating another node with a different state...")
    second_state = [initial_state[0] + 10, initial_state[1] + 10]
    agent.update_model(second_state, 1)  # action 1
    
    # 11. Check final node count
    print("\nFinal node counts:")
    if hasattr(agent, 'nodes'):
        print(f"  agent.nodes: {len(agent.nodes)} items")
        if len(agent.nodes) > 0:
            print(f"  First node: {agent.nodes[0]}")
    else:
        print("  agent.nodes does not exist")
    
    if hasattr(agent, 'higher_nodes'):
        print(f"  agent.higher_nodes: {len(agent.higher_nodes)} items")
        if len(agent.higher_nodes) > 0:
            print(f"  First higher_node: {agent.higher_nodes[0]}")
    else:
        print("  agent.higher_nodes does not exist")
    
    # 12. Examine prev_node_idx
    print("\nChecking prev_node_idx:")
    print(f"  prev_node_idx: {agent.prev_node_idx}")
    
    # 13. Check connections
    print("\nChecking connections:")
    if hasattr(agent, 'connections'):
        print(f"  connections shape: {agent.connections.shape}")
    else:
        print("  connections attribute not found")
    
    # 14. Try to understand what's being used in demonstrate_node_creation
    print("\nReplicating node creation from demonstrate_node_creation:")
    try:
        # This replicates the logic in demonstrate_node_creation
        x_data = np.array([test_state[0]]).reshape(1, -1)
        y_data = np.array([test_state[1]]).reshape(1, -1)
        
        x_bmu, _ = agent.lower_x.find_best_matching_units(x_data)
        y_bmu, _ = agent.lower_y.find_best_matching_units(y_data)
        
        x_binary = np.zeros(len(agent.lower_x.A))
        y_binary = np.zeros(len(agent.lower_y.A))
        x_binary[x_bmu] = 1
        y_binary[y_bmu] = 1
        
        manual_pattern = (tuple(x_binary), tuple(y_binary))
        
        print(f"  Manual pattern: {manual_pattern}")
        
        # Check which attribute is used for node storage in the example.py
        if hasattr(agent, 'higher_nodes'):
            node_storage = agent.higher_nodes
            print("  Using higher_nodes for storage")
        else:
            node_storage = agent.nodes
            print("  Using nodes for storage")
        
        # Find if this pattern exists
        found = False
        for idx, existing_pattern in enumerate(node_storage):
            if all(np.array_equal(p1, p2) for p1, p2 in zip(manual_pattern, existing_pattern)):
                found = True
                manual_node_idx = idx
                break
        
        if found:
            print(f"  Pattern found at index: {manual_node_idx}")
        else:
            print("  Pattern not found in node storage")
            
    except Exception as e:
        print(f"  Error in manual node creation: {e}")
    
    return "Debug complete"

# Run the debugging function
if __name__ == "__main__":
    debug_minerva_agent()