import turtle
import math 
import random

class MazePlayer:
    MAZE_BLOCK_PIXEL_WIDTH = 24 
    MAZE_BLOCK_PIXEL_HEIGHT = 24
    SCREEN_SUROUNDING_WHITE_SPACE_PIXEL = 50

    def __init__(self, maze_map, player_index_pos, goal_index_pos=None, display_maze=True):
        self.maze_map = maze_map
        self.initial_player_index_pos = player_index_pos
        self.goal_index_pos = goal_index_pos
        self.display_maze = display_maze
        self.current_player_index_pos = self.initial_player_index_pos
        
        self.initial_player_pos = None
        self.goal_pos = None 
        self.walls = []
        self.walls_index = []

        (self.screen_width_pixel, self.screen_height_pixel, 
         self.screen_x_pixel_offset, self.screen_y_pixel_offset) = self._get_screen_size()
 
        if display_maze:
            self.turtle_screen = turtle.Screen()
            self.turtle_screen.bgcolor("white")
            self.turtle_screen.title("Maze Simulation")
            self.turtle_screen.setup(self.screen_width_pixel, self.screen_height_pixel)
            self.turtle_screen.tracer(0)
            self._register_custom_turtle_shapes()
            self.pen = Pen()
            self.player = Player()
            self.treasures = []

        self.setup_maze()

    def _get_screen_size(self):
        screen_width_num_blocks = len(self.maze_map[0])
        screen_height_num_blocks = len(self.maze_map)
        screen_width_pixel = screen_width_num_blocks * self.MAZE_BLOCK_PIXEL_WIDTH
        screen_height_pixel = screen_height_num_blocks * self.MAZE_BLOCK_PIXEL_HEIGHT
        screen_x_pixel_offset = -int((screen_width_pixel / 2) - self.MAZE_BLOCK_PIXEL_WIDTH/2)
        screen_y_pixel_offset = int((screen_height_pixel/2) - self.MAZE_BLOCK_PIXEL_HEIGHT/2)
        
        screen_width_pixel += self.SCREEN_SUROUNDING_WHITE_SPACE_PIXEL
        screen_height_pixel += self.SCREEN_SUROUNDING_WHITE_SPACE_PIXEL

        return screen_width_pixel, screen_height_pixel, screen_x_pixel_offset, screen_y_pixel_offset

    def _register_custom_turtle_shapes(self):
        shapes = [
            'player_24_right.gif', 'player_24_left.gif',
            'player_24_down.gif', 'player_24_up.gif',
            'bricks_24.gif', 'treasure_24.gif'
        ]
        for shape in shapes:
            turtle.register_shape(f'./Maze/images/{shape}')

    def _calc_screen_coordinates(self, row_index, col_index):
        screen_x = self.screen_x_pixel_offset + (col_index * self.MAZE_BLOCK_PIXEL_WIDTH)
        screen_y = self.screen_y_pixel_offset - (row_index * self.MAZE_BLOCK_PIXEL_HEIGHT)
        return screen_x, screen_y

    def change_maze(self, new_maze_map, new_player_index_pos, new_goal_index_pos):
        self.maze_map = new_maze_map
        self.initial_player_index_pos = new_player_index_pos
        self.goal_index_pos = new_goal_index_pos
        self.current_player_index_pos = self.initial_player_index_pos

        (self.screen_width_pixel, self.screen_height_pixel, 
         self.screen_x_pixel_offset, self.screen_y_pixel_offset) = self._get_screen_size()
        
        self.walls = []
        self.walls_index = []

        if self.display_maze:
            for treasure in self.treasures:
                treasure.destroy()
            self.treasures.clear()
            self.pen.clear()
        
        self.setup_maze()

    def setup_maze(self):
        for row_index in range(len(self.maze_map)):
            for col_index in range(len(self.maze_map[row_index])):
                if self.maze_map[row_index][col_index] == "X":
                    screen_x, screen_y = self._calc_screen_coordinates(row_index, col_index)
                    if self.display_maze:
                        self.pen.goto(screen_x, screen_y)
                        self.pen.stamp()
                    self.walls.append((screen_x, screen_y))
                    self.walls_index.append((row_index, col_index))

        screen_x, screen_y = self._calc_screen_coordinates(
            self.initial_player_index_pos[0], self.initial_player_index_pos[1])
        
        if self.display_maze:
            self.player.goto(screen_x, screen_y)
        self.initial_player_pos = (screen_x, screen_y)

        if self.goal_index_pos is not None:
            screen_x, screen_y = self._calc_screen_coordinates(
                self.goal_index_pos[0], self.goal_index_pos[1])
            if self.display_maze:
                self.treasures.append(Treasure(screen_x, screen_y))
            self.goal_pos = (screen_x, screen_y)

        if self.display_maze:
            self.turtle_screen.update()

    def move_player(self, action):
        moves = {
            0: (-1, 0, 'up'),    # up
            1: (1, 0, 'down'),   # down
            2: (0, 1, 'right'),  # right
            3: (0, -1, 'left')   # left
        }
        
        if action in moves:
            row_change, col_change, direction = moves[action]
            next_pos = (
                self.current_player_index_pos[0] + row_change,
                self.current_player_index_pos[1] + col_change
            )
            
            if next_pos not in self.walls_index:
                self.current_player_index_pos = next_pos
                if self.display_maze:
                    getattr(self.player, f'go_{direction}')(self.walls)

    def get_sensor_reading(self, player_pos_index=None):
        player_pos_index = player_pos_index or self.get_player_pos_as_index()
        player_row_index, player_col_index = player_pos_index
        
        obstacles = [
            self.maze_map[player_row_index - 1][player_col_index] == "X",  # up
            self.maze_map[player_row_index + 1][player_col_index] == "X",  # down
            self.maze_map[player_row_index][player_col_index - 1] == "X",  # left
            self.maze_map[player_row_index][player_col_index + 1] == "X"   # right
        ]
        
        player_pos = (self._calc_screen_coordinates(player_row_index, player_col_index) 
                     if player_pos_index else self.get_player_pos())
        
        goal_positions = [
            player_pos[1] < self.goal_pos[1],  # up
            player_pos[1] > self.goal_pos[1],  # down
            player_pos[0] > self.goal_pos[0],  # left
            player_pos[0] < self.goal_pos[0]   # right
        ]
        
        return tuple(int(x) for x in obstacles + goal_positions)

    def get_state_matrix(self):
        state_matrix = []
        for i in range(len(self.maze_map)):
            row = []
            for j in range(len(self.maze_map[i])):
                if self.maze_map[i][j] == "X":
                    row.append(None)
                elif (i, j) == self.goal_index_pos:
                    row.append("goal")
                else:
                    row.append(self.get_sensor_reading(player_pos_index=(i, j)))
            state_matrix.append(row)
        return state_matrix

    # Getter methods
    def get_walls(self): return self.walls
    def get_goal_pos(self): return self.goal_pos
    def get_initial_player_pos(self): return self.initial_player_pos
    def get_player_pos(self): return self._calc_screen_coordinates(
        self.current_player_index_pos[0], self.current_player_index_pos[1])
    def get_player_pos_as_index(self): return self.current_player_index_pos

    def reset_player(self):
        self.current_player_index_pos = self.initial_player_index_pos
        if self.display_maze:
            self.player.goto(self.initial_player_pos)

    def update_screen(self):
        if self.display_maze:
            self.turtle_screen.update()

class Pen(turtle.Turtle):
    def __init__(self):
        super().__init__()
        self.shape("./Maze/images/bricks_24.gif")
        self.color("white")
        self.penup()
        self.speed(0)

class Player(turtle.Turtle):
    def __init__(self):
        super().__init__()
        self.shape("./Maze/images/player_24_right.gif")
        self.color("blue")
        self.penup()
        self.speed(0)

    def move_if_valid(self, move_to_x, move_to_y, walls, direction):
        self.shape(f"./Maze/images/player_24_{direction}.gif")
        if (move_to_x, move_to_y) not in walls:
            self.goto(move_to_x, move_to_y)

    def go_left(self, walls): 
        self.move_if_valid(self.xcor()-24, self.ycor(), walls, "left")
    def go_right(self, walls): 
        self.move_if_valid(self.xcor()+24, self.ycor(), walls, "right")
    def go_up(self, walls): 
        self.move_if_valid(self.xcor(), self.ycor()+24, walls, "up")
    def go_down(self, walls): 
        self.move_if_valid(self.xcor(), self.ycor()-24, walls, "down")
def get_random_valid_state(self):
    """Generate random valid state based on maze layout"""
    valid_positions = []
    for i in range(self.maze_map.shape[0]):
        for j in range(self.maze_map.shape[1]):
            if self.maze_map[i,j] == 0:  # If not a wall
                valid_positions.append([i,j])
    return random.choice(valid_positions)
class Treasure(turtle.Turtle):
    def __init__(self, x, y):
        super().__init__()
        self.shape("./Maze/images/treasure_24.gif")
        self.color("gold")
        self.penup()
        self.speed(0)
        self.goto(x, y)

    def destroy(self):
        self.hideturtle()