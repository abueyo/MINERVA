#!/usr/bin/env python3
"""
Simple test script for GWRSOM implementation.
Tests the GWRSOM using x,y coordinates from maze_positions.csv file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import GWRSOM from your module
from Agents.HSOM_binary import GWRSOM, HierarchicalGWRSOMAgent

def test_gwrsom_with_coordinates():
    """
    Test GWRSOM implementation with x,y coordinates from maze_positions.csv
    """
    # Load maze positions from CSV
    print("Loading maze positions from maze_positions.csv...")
    try:
        data = pd.read_csv("maze_positions.csv")
        positions = data.values
        print(f"Loaded {len(positions)} positions")
    except Exception as e:
        print(f"Error loading maze_positions.csv: {e}")
        print("Generating sample data instead...")
        # Generate sample data if file not found
        positions = np.array([[x, y] for x in range(1, 6) for y in range(1, 6)])
        print(f"Generated {len(positions)} sample positions")
    
    # Test with different activity thresholds
    activity_thresholds = [0.8, 0.05, 0.003, 0.0001]
    
    results = []
    
    for a in activity_thresholds:
        # Create GWRSOM with specific activity threshold
        print(f"\nTesting with activity threshold = {a}...")
        gwrsom = GWRSOM(a=a, h=0.1, es=0.2, en=0.05)
        
        # Train on positions
        gwrsom.train(positions, epochs=1)
        
        # Get weights and connections
        weights = gwrsom.get_weights()
        connections = gwrsom.get_connections()
        
        # Store results
        results.append({
            "activity": a,
            "nodes": len(weights),
            "connections": np.sum(connections) // 2  # Divide by 2 because connections are bidirectional
        })
        
        print(f"  Created {len(weights)} nodes with {np.sum(connections) // 2} connections")
        
        # Plot the network
        plt.figure(figsize=(10, 8))
        
        # Plot positions as small gray dots
        plt.scatter(positions[:, 0], positions[:, 1], color='gray', alpha=0.3, s=5, label='Input Positions')
        
        # Plot nodes as larger blue dots
        plt.scatter(weights[:, 0], weights[:, 1], color='blue', s=50, label='GWRSOM Nodes')
        
        # Draw connections
        for i in range(len(connections)):
            for j in range(i+1, len(connections)):
                if connections[i, j] > 0:
                    plt.plot([weights[i, 0], weights[j, 0]], 
                             [weights[i, 1], weights[j, 1]], 
                             'k-', alpha=0.5)
        
        plt.title(f"GWRSOM Network (activity={a}, nodes={len(weights)})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.savefig(f"gwrsom_network_a{a}.png")
        plt.close()
    
    # Print summary table
    print("\nResults Summary:")
    print("----------------------------------------")
    print("Activity Threshold | Nodes | Connections")
    print("----------------------------------------")
    for r in results:
        print(f"{r['activity']:<18} | {r['nodes']:<5} | {r['connections']:<11}")
    print("----------------------------------------")
    
    # Test with sequence data
    print("\nTesting with simple sequence [1, 2, 3, 4, 5]...")
    sequence = np.array([[i] for i in range(1, 6)])
    
    gwrsom_seq = GWRSOM(a=0.5, h=0.1, es=0.2, en=0.05)
    gwrsom_seq.train(sequence, epochs=3)
    
    weights_seq = gwrsom_seq.get_weights()
    connections_seq = gwrsom_seq.get_connections()
    
    print(f"Created {len(weights_seq)} nodes (expected 5)")
    print("Node weights:")
    for i, w in enumerate(weights_seq):
        print(f"  Node {i}: {w[0]:.2f}")
    
    # Plot the sequence results
    plt.figure(figsize=(10, 4))
    
    # Plot input sequence
    plt.scatter(sequence[:, 0], np.zeros_like(sequence[:, 0]), 
               color='gray', s=100, label='Input Sequence')
    
    # Plot nodes
    plt.scatter(weights_seq[:, 0], np.zeros_like(weights_seq[:, 0]), 
               color='red', marker='x', s=200, label='GWRSOM Nodes')
    
    # Draw connections
    for i in range(len(connections_seq)):
        for j in range(i+1, len(connections_seq)):
            if connections_seq[i, j] > 0:
                plt.plot([weights_seq[i, 0], weights_seq[j, 0]], [0, 0], 
                        'k--', alpha=0.7)
    
    # Add node labels
    for i, w in enumerate(weights_seq):
        plt.text(w[0], 0.1, f"Node {i} ({w[0]:.2f})", ha='center')
    
    plt.yticks([])
    plt.title("GWRSOM with Sequence [1, 2, 3, 4, 5]")
    plt.grid(True, alpha=0.3)
    plt.savefig("gwrsom_sequence_test.png")
    plt.close()

if __name__ == "__main__":
    test_gwrsom_with_coordinates()