#import the maze
from Maze.Mazes import MazeMaps
#import the maze player 
from Maze.Maze_player import MazePlayer
#import Qlearning agent 
from Agents.TMGWR_agent import TMGWRAgent

import time 
import os
# Change the current working directory to the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)


#get the maze details 
maze_map, player_pos_index, goal_pos_index = MazeMaps.get_default_map()

#create the maze player 
Maze = MazePlayer(maze_map=maze_map, player_index_pos=player_pos_index, goal_index_pos=goal_pos_index)


#get the goal in screen coordinates
goal = Maze.get_goal_pos()

#get player intital posistion 
initial_state = Maze.get_initial_player_pos()


#define the parameters 
load_model = True 
save_model = True 
show_map = True


#initialize the agent 
TMGWR_agent = TMGWRAgent(nDim=2, Ni=2, epsilon_b=0.55194, epsilon_n=0.90, beta=0.8, 
                         delta=0.6235, T_max=17, N_max=300, eta=0.95, phi=0.6, sigma=1)

#set a goal 
TMGWR_agent.set_goal(goal=goal)

TMGWR_agent.set_epsilon(1)

#load model 
if load_model: 
    TMGWR_agent.load_model("Data/TMGWR/model4.npz")

TMGWR_agent.show_map()

#set parameters 
num_episodes = 200 #episode: one sequence of states,actions and rewards, which ends with terminal state (in this case goal state)
slow_episode = 1990 #episode at which the movement is slowed down

#set the current state to the intial state
current_state = initial_state

#track the number of times the goal has been reached to decay epsilon 
reached_goal_count = 0

#start the learning loop 
for episode_num in range(num_episodes):     
    #set the current state to the intial state
    current_state = initial_state
    #move the player to the initial position 
    Maze.reset_player() 

    #step counter
    step_counter = 0

    #while not in terminal state
    while current_state != goal: 
        step_counter += 1

        #take an action 
        action = TMGWR_agent.select_action(current_state=current_state)
                                                                                                                                                                                                                                    
        #move the player 
        Maze.move_player(action=action)

        #get the next state
        next_state = Maze.get_player_pos() 

        #update the model 
        TMGWR_agent.update_model(next_state=next_state, action=action)

        #update current state 
        current_state = next_state

        #update the maze 
        Maze.update_screen() 
        
        #printing progress every 10 steps
        if step_counter % 100 == 0: 
            print(f"Episode number: {episode_num +1} step number: {step_counter}")

        #slow down action 
        if episode_num == slow_episode: 
            time.sleep(0.25)

    
    #reached goal
    reached_goal_count += 1
    #decay epsilon 
    if reached_goal_count > 10: 
        TMGWR_agent.decay_epsilon(min_epsilon=0.2) 

    print(f"Episode number: {episode_num +1} final step number: {step_counter} epsilon: {TMGWR_agent.get_epsilon()}\n")

    if show_map: 
        #show the map learned by the agent 
        TMGWR_agent.show_map() 

#save the model
if save_model: 
    TMGWR_agent.save_model("Data/TMGWR/model4.npz")


if show_map: 
    #show the map learned by the agent 
    TMGWR_agent.show_map() 