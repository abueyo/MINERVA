from Maze.Maze_player import MazePlayer
import numpy as np

class KeyDoorMazePlayer(MazePlayer):
    """Extended MazePlayer that includes key and door mechanics"""
    
    def __init__(self, maze_map, player_index_pos, goal_index_pos, key_position, door_position, display_maze=False):
        super().__init__(maze_map, player_index_pos, goal_index_pos, display_maze)
        
        # Initialize key and door properties
        self.key_position = key_position
        self.door_position = door_position
        self.key_collected = False
        
        # If display is enabled, create visual representations of key and door
        if display_maze:
            from turtle import Turtle
            
            # Create key visual
            key_x, key_y = self._calc_screen_coordinates(key_position[0], key_position[1])
            self.key_turtle = Turtle()
            self.key_turtle.shape("circle")
            self.key_turtle.color("yellow")
            self.key_turtle.penup()
            self.key_turtle.goto(key_x, key_y)
            
            # Create door visual
            door_x, door_y = self._calc_screen_coordinates(door_position[0], door_position[1])
            self.door_turtle = Turtle()
            self.door_turtle.shape("square")
            self.door_turtle.color("red")
            self.door_turtle.penup()
            self.door_turtle.goto(door_x, door_y)
            
            # Update the screen
            self.turtle_screen.update()
    
    def is_key_collected(self):
        """Check if the key has been collected"""
        return self.key_collected
    
    def reset_key_collected(self):
        """Reset key collection status"""
        self.key_collected = False
        
        # If display is enabled, make the key visible again
        if self.display_maze and hasattr(self, 'key_turtle'):
            self.key_turtle.showturtle()
            self.door_turtle.color("red")
            self.turtle_screen.update()
    
    def move_player_with_key_door(self, action, key_position=None, door_position=None):
        """
        Move player with key and door mechanics
        
        Args:
            action: Direction to move (0=up, 1=down, 2=right, 3=left)
            key_position: Position of the key (optional, uses self.key_position if None)
            door_position: Position of the door (optional, uses self.door_position if None)
            
        Returns:
            result: String indicating outcome ('normal', 'key_collected', 'door_locked', 'door_opened')
        """
        # Use object attributes if parameters not provided
        key_position = key_position or self.key_position
        door_position = door_position or self.door_position
        
        # Determine next position after movement
        moves = {
            0: (-1, 0, 'up'),    # up
            1: (1, 0, 'down'),   # down
            2: (0, 1, 'right'),  # right
            3: (0, -1, 'left')   # left
        }
        
        result = 'normal'
        
        if action in moves:
            row_change, col_change, direction = moves[action]
            next_pos = (
                self.current_player_index_pos[0] + row_change,
                self.current_player_index_pos[1] + col_change
            )
            
            # Check if next position is a wall
            if next_pos in self.walls_index:
                return result  # Can't move into walls
            
            # Check if next position is the door
            if next_pos == door_position:
                if self.key_collected:
                    # Door can be opened
                    result = 'door_opened'
                else:
                    # Door is locked
                    result = 'door_locked'
                    return result  # Can't move through locked door
            
            # Actually move the player
            self.current_player_index_pos = next_pos
            if self.display_maze:
                getattr(self.player, f'go_{direction}')(self.walls)
                self.turtle_screen.update()
            
            # Check if player has collected the key
            if next_pos == key_position and not self.key_collected:
                self.key_collected = True
                result = 'key_collected'
                
                # If display is enabled, hide the key and change door color
                if self.display_maze and hasattr(self, 'key_turtle'):
                    self.key_turtle.hideturtle()
                    self.door_turtle.color("green")
                    self.turtle_screen.update()
        
        return result

def create_key_door_maze(maze_map, player_pos_index, goal_pos_index, key_position, door_position, display_maze=True):
    """Create a KeyDoorMazePlayer instance with the given parameters"""
    return KeyDoorMazePlayer(
        maze_map=maze_map, 
        player_index_pos=player_pos_index, 
        goal_index_pos=goal_pos_index,
        key_position=key_position,
        door_position=door_position,
        display_maze=True
    )