from collections import deque 
import random
import turtle
import random
import tkinter as tk



class MazeMaps: 
    #standard default maze 
    '''
    X: wall
    blank: path
    '''
   
    
    default_maze_map = [
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'], 
        ['X', 'X', ' ', ' ', 'X', ' ', ' ', 'X', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'], 
        ['X', 'X', ' ', ' ', ' ', ' ', ' ', 'X', 'X'], 
        ['X', 'X', ' ', ' ', 'X', ' ', ' ', 'X', 'X'], 
        ['X', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'X'], 
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'], 
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'], 
    ]
    
    default_player_pos_index = (1, 1) #row, column
    default_goal_pos_index = (8, 1) #row, column

    # 5x5 Maze Configuration
    maze_5x5 = [
        ['X', 'X', 'X', 'X', 'X', 'X', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', 'X', ' ', ' ', 'X'],
        ['X', ' ', ' ', 'X', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', 'X', 'X', 'X', 'X', 'X', 'X']
    ]

    # default_player_pos_index = (1, 1) #row, column
    # default_goal_pos_index = (6, 1) #row, column

    # 10x10 Maze Configuration
    maze_10x10 = [
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'X'],
        ['X', 'X', 'X', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ' ', 'X'],
        ['X', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'X', ' ', 'X'],
        ['X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'X'],
        ['X', 'X', 'X', ' ', 'X', 'X', ' ', 'X', 'X', 'X', ' ', 'X'],
        ['X', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'X', ' ', 'X'],
        ['X', ' ', 'X', 'X', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'X'],
        ['X', 'X', 'X', ' ', 'X', 'X', 'X', ' ', 'X', 'X', 'X', 'X'],
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    ]

    # default_player_pos_index = (1, 1) #row, column
    # default_goal_pos_index = (10, 1) #row, column

    # 15x15 Maze Configuration
    maze_15x15 = [
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'], 
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'], 
        ['X', 'X', 'X', ' ', 'X', 'X', 'X', 'X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
        ["X", " ", "X", " ", " ", " ", " ", "X", " ", "X", " ", " ", " ", " ", "X", " ", "X"],
        ['X', ' ', 'X', ' ', 'X', 'X', 'X', 'X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', 'X'], 
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X', 'X', 'X', ' ', 'X', ' ', 'X', 'X', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'X'], 
        ['X', ' ', 'X', ' ', 'X', ' ', 'X', ' ', 'X', 'X', 'X', ' ', 'X', ' ', 'X', 'X', 'X'], 
        ['X', ' ', ' ', ' ', 'X', 'X', 'X', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ' ', ' ', 'X'], 
        ['X', 'X', ' ', 'X', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', 'X', 'X', 'X', 'X', 'X'],
        ['X', ' ', ' ', 'X', 'X', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', 'X', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', ' ', ' ', 'X', 'X', 'X', ' ', 'X'], 
        ['X', ' ', ' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X', ' ', 'X', ' ', 'X', ' ', 'X'], 
        ['X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X'], 
        ['X', ' ', 'X', ' ', 'X', 'X', 'X', 'X', ' ', 'X', 'X', 'X', 'X', ' ', 'X', 'X', 'X'], 
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'], 
    ]
    
    # default_player_pos_index = (1, 1) #row, column
    # default_goal_pos_index = (16, 1) #row, column 

    # 20x20 Maze Configuration
    maze_20x20 = [
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', 'X', 'X', 'X', 'X', ' ', 'X', 'X', 'X', 'X', 'X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X'],
        ['X', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X'],
        ['X', ' ', 'X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X', 'X', 'X', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
        ['X', ' ', 'X', ' ', 'X', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', ' ', 'X', ' ', 'X'],
        ['X', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', 'X', 'X', 'X', ' ', 'X', ' ', 'X', ' ', 'X', 'X', ' ', 'X'],
        ['X', ' ', ' ', ' ', 'X', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', ' ', ' ', ' ', 'X'],
        ['X', 'X', 'X', 'X', 'X', ' ', 'X', ' ', 'X', 'X', 'X', ' ', 'X', ' ', 'X', ' ', 'X', 'X', 'X', 'X', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', ' ', ' ', 'X', ' ', 'X', ' ', ' ', ' ', ' ', 'X', ' ', 'X'],
        ['X', ' ', 'X', 'X', 'X', 'X', 'X', ' ', 'X', ' ', 'X', 'X', 'X', ' ', 'X', 'X', 'X', 'X', ' ', 'X', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X'],
        ['X', ' ', 'X', 'X', 'X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X', ' ', 'X', ' ', 'X'],
        ['X', ' ', 'X', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', 'X', ' ', 'X'],
        ['X', ' ', 'X', ' ', 'X', 'X', 'X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X', ' ', 'X', ' ', 'X', ' ', 'X'],
        ['X', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', 'X', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X', 'X', 'X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X'],
        ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
        ['X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    ]

    # default_player_pos_index = (1, 1) #row, column
    # default_goal_pos_index = (20, 1) #row, column
    
    def get_default_map(): 
        return MazeMaps.default_maze_map, MazeMaps.default_player_pos_index, MazeMaps.default_goal_pos_index
    

    def get_default_map_multiple_goals(): 
        # goal_index_positions = [MazeMaps.default_goal_pos_index, (7, 15), (1, 15), (3, 1), (9, 9)]
        goal_index_positions = [MazeMaps.default_goal_pos_index, (7, 15), (1, 15), (9, 9), MazeMaps.default_goal_pos_index]
        return MazeMaps.default_maze_map, MazeMaps.default_player_pos_index, goal_index_positions


    def get_default_map_variable_wall(): 
        varriable_wall_index_positions = [(8, 3), (2, 3), (6, 4)]
        return MazeMaps.default_maze_map, MazeMaps.default_player_pos_index, MazeMaps.default_goal_pos_index, varriable_wall_index_positions


    def get_default_map_variable_goals_and_walls(): 
        #(goal_index, wall_index)
        varriable_wall_goal_index_pairs = [((MazeMaps.default_goal_pos_index), (0, 0)), #((MazeMaps.default_goal_pos_index), (6, 5)), 
                                           ((7, 15), (6, 13))]   
        return MazeMaps.default_maze_map, MazeMaps.default_player_pos_index, varriable_wall_goal_index_pairs

    

    '''
    get maze scalled by a factor
    '''
    def scale_map(maze_map, scale_factor): 
        #calculate dimensions 
        num_rows = len(maze_map)
        num_columns = len(maze_map[0])
        scalled_num_rows = num_rows*scale_factor
        scalled_num_columns = num_columns*scale_factor

        #create map matrix, easier to manipulate than string
        scalled_map_matrix = [[" "]*scalled_num_columns for i in range(scalled_num_rows)]
        
        #go over initialized map and fill in walls 
        for row_index, row in enumerate(maze_map): 
            #scale row index
            row_index = row_index*scale_factor
            for col_index, char in enumerate(row): 
                #scale colum index 
                col_index = col_index*scale_factor

                #fill block 
                for i in range(scale_factor): 
                    for j in range(scale_factor): 
                        scalled_map_matrix[row_index+i][col_index+j] = char

        #convert matrix into array of row strings 
        scalled_map = ["".join(row) for row in scalled_map_matrix]
        return scalled_map


    def get_scalled_default_map(scale_factor=10):
        #factor needs to be an integer
        scale_factor = int(scale_factor)
        #get the scalled map
        scalled_map = MazeMaps.scale_map(maze_map=MazeMaps.default_maze_map, scale_factor=scale_factor)

        #adjust player and goal positions
        scalled_player_pos_index = (MazeMaps.default_player_pos_index[0]*scale_factor, MazeMaps.default_player_pos_index[1]*scale_factor) 
        scalled_goal_pos_index = (MazeMaps.default_goal_pos_index[0]*scale_factor, MazeMaps.default_goal_pos_index[1]*scale_factor) 

        return scalled_map, scalled_player_pos_index, scalled_goal_pos_index
    

    def get_optimal_path_len(maze_map, start_index_pos, end_index_pos, additional_walls=None): 
        #convert maze list of strings into matrix 
        maze_matrix = []
        for string_row in maze_map: 
            char_row = [char for char in string_row]
            maze_matrix.append(char_row)

        #add the additional walls 
        if additional_walls is not None: 
            for additional_wal_pos in additional_walls: 
                maze_matrix[additional_wal_pos[0]][additional_wal_pos[1]] = "X"

        #perform BFS 
        num_rows = len(maze_matrix) 
        num_cols = len(maze_matrix[0])
        visited = set() 
        elems_in_queue = set() 
        queue = deque() 
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  #Right, Left, Down, Up
        
        #initialize queue with start posisiton 
        queue.append((start_index_pos, 0)) #start position, path_len
        elems_in_queue.add(start_index_pos)

        #while there are elements in the queue 
        while queue: 
            #get the first element in the queue 
            (row_pos, col_pos), path_len = queue.popleft()

            #add element to visited 
            visited.add((row_pos,col_pos))
            #remove element from elements in queue set
            elems_in_queue.remove((row_pos,col_pos))

            # print(visited)
            #go throgh all possible moves in the position
            for row_move, col_move in moves: 
                new_row_pos = row_pos + row_move
                new_col_pos = col_pos + col_move

                #make sure we stay inside the matrix and it is a free block
                if ((0 <= new_row_pos < num_rows) and (0 <= new_col_pos < num_cols) and (maze_matrix[new_row_pos][new_col_pos] == " ")): 
                    # print(f"here: {(new_row_pos,new_col_pos)}")
                    #check if element has already been explored 
                    if ((new_row_pos, new_col_pos) not in visited): 
                        new_row_pos
                        #not explored, check if goal state 
                        if (new_row_pos, new_col_pos) == end_index_pos:    
                            #return the shortest path 
                            return path_len + 1

                        #if not goal, check if in queue, if not, add 
                        if (new_row_pos, new_col_pos) not in elems_in_queue:
                            #add to queue  
                            elems_in_queue.add((new_row_pos, new_col_pos))
                            queue.append(((new_row_pos, new_col_pos), path_len+1))
                    
        #if there does not exist a path return -1
        return -1
    

    def print_maze(maze, player_pos=None, goal_pos=None): 
        if player_pos is not None: 
            maze[player_pos[0]][player_pos[1]] = "P"
        if goal_pos is not None: 
            maze[goal_pos[0]][goal_pos[1]] = "G"

        for row in maze: 
            print("".join(row))

    
    def get_real_world_maze_map(): 
        '''
        map is 3 by 6, but surrounded by walls (there is a wall in between each block that can also be empty, so the maze becomes 7 by 13)
        '''
        maze_map = [
            ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
            ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X'],
            ['X', 'X', 'X', ' ', 'X', 'X', 'X', 'X', 'X', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', 'X', ' ', ' ', ' ', ' ', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', 'X', ' ', 'X', 'X', 'X', ' ', 'X'],
            ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', 'X'],
            ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
        ] 
        
        player_pos_index = (1, 1) #row, col
        goal_pos_index = (5, 11)

        return maze_map, player_pos_index, goal_pos_index

valid_positions = [
        (x, y)
        for y, row in enumerate(MazeMaps.default_maze_map)
        for x, val in enumerate(row)
        if val == ' '
    ]

#generates a random maze
class MazeGenerator: 

    def __init__(self, num_rows, num_cols) -> None: 
        self.num_rows = num_rows
        self.num_cols = num_cols 

    
    def generate_random_maze(self): 
        #generate maze 
        maze = [["X" for _ in range(self.num_cols)] for _ in range(self.num_rows)]

        # Randomly choose player start location
        player_initial_pos = (random.randint(1, self.num_rows - 2), random.randint(1, self.num_cols - 2))
        
        #generate maze path 
        maze, path_blocks = self.generate_path(maze, start=player_initial_pos)
        
        #choose goal from maze path and ensure that it is not the same as the player position
        goal_pos = random.choice(path_blocks)
        while player_initial_pos == goal_pos: 
            goal_pos = random.choice(path_blocks)
        
        return maze, player_initial_pos, goal_pos


    def is_valid_index(self, maze, i, j): 
        return (0 < i < self.num_rows- 1) and (0 < j < self.num_cols-1) and maze[i][j] == "X"
    
    def get_neighbors(self, maze, i, j): 
        #generate all possible neighbors 
        possible_neighbors = [(i-2, j), (i+2, j), (i, j-2), (i, j+2)]
        #return neighbors that are valid
        return [(ni, nj) for ni, nj in possible_neighbors if self.is_valid_index(maze, ni, nj)]


    def generate_path(self, maze, start): 
        #save the path
        path_blocks = []

        #generate a path
        stack = [start]
        while stack:
            i, j = stack[-1]
            maze[i][j] = " "
            path_blocks.append((i,j))
            neighbors_list = self.get_neighbors(maze, i, j)
            if neighbors_list:
                ni, nj = random.choice(neighbors_list)
                maze[(i + ni) // 2][(j + nj) // 2] = " "
                stack.append((ni, nj))
            else:
                stack.pop()

        return maze, path_blocks
    
# def draw_maze_turtle(maze_map, cell_size=20, title="Maze"):
#     # Create a separate Tkinter window for each maze
#     root = tk.Tk()
#     root.title(title)
    
#     # Create a canvas within the Tkinter window
#     canvas = tk.Canvas(root, width=len(maze_map[0])*cell_size + 50, height=len(maze_map)*cell_size + 50)
#     canvas.pack()
    
#     # Create a turtle screen on the canvas
#     screen = turtle.TurtleScreen(canvas)
#     t = turtle.RawTurtle(screen)
    
#     # Setup drawing
#     t.speed(0)  # Fastest drawing speed
#     t.hideturtle()
#     t.penup()
    
#     # Start from top-left corner
#     start_x = -len(maze_map[0])*cell_size/2
#     start_y = len(maze_map)*cell_size/2

#     # Draw maze
#     for row_idx, row in enumerate(maze_map):
#         for col_idx, cell in enumerate(row):
#             # Calculate current position
#             x = start_x + col_idx * cell_size
#             y = start_y - row_idx * cell_size
            
#             # Draw walls
#             if cell == 'X':
#                 t.goto(x, y)
#                 t.pendown()
#                 t.color('black')
#                 t.fillcolor('black')
#                 t.begin_fill()
#                 for _ in range(4):
#                     t.forward(cell_size)
#                     t.right(90)
#                 t.end_fill()
#                 t.penup()
    
#     # Start the Tkinter event loop
#     root.mainloop()

# def visualize_mazes():
#     # Different maze environments
#     mazes = [
#         (MazeMaps.default_maze_map, "Default Small Maze"),
#         (MazeMaps.default_maze_map_1, "Default Large Maze"),
#         (MazeGenerator(7, 13).generate_random_maze()[0], "Real World Maze"),
#         (MazeGenerator(10, 15).generate_random_maze()[0], "Random Generated Maze")
#     ]
    
#     # Draw each maze
#     for maze, title in mazes:
#         # Create a new window for each maze
#         draw_maze_turtle(maze, title=title)

# # Uncomment to run
# visualize_mazes()