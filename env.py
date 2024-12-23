import pygame
import numpy as np
import time
import random
from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGame:
    def __init__(self, w=640, h=480, speed=40):
        self.w = w
        self.h = h
        self.speed = speed  # Controls game speed (FPS)
        self.block_size = 20
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                     Point(self.head.x-self.block_size, self.head.y),
                     Point(self.head.x-(2*self.block_size), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self.get_state()

    def _place_food(self):
        x = random.randint(0, (self.w-self.block_size)//self.block_size)*self.block_size
        y = random.randint(0, (self.h-self.block_size)//self.block_size)*self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def get_state(self):
        head = self.snake[0]
        
        # Points around the head
        point_l = Point(head.x - self.block_size, head.y)
        point_r = Point(head.x + self.block_size, head.y)
        point_u = Point(head.x, head.y - self.block_size)
        point_d = Point(head.x, head.y + self.block_size)

        # Current direction
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT

        # Check dangers
        danger_front = (
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d))
        )
        
        danger_right = (
            (dir_r and self._is_collision(point_d)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l))
        )
        
        danger_left = (
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_d and self._is_collision(point_r))
        )

        # Food direction relative to head
        food_up = self.food.y < head.y
        food_down = self.food.y > head.y
        food_left = self.food.x < head.x
        food_right = self.food.x > head.x

        state = [
            food_up,    # Food direction
            food_down,
            food_left,
            food_right,
            danger_front,  # Dangers
            danger_left,
            danger_right,
            dir_u,    # Current direction
            dir_d,
            dir_l,
            dir_r
        ]
        
        return np.array(state, dtype=int)

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - self.block_size or pt.x < 0 or pt.y > self.h - self.block_size or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        # action = [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Keep direction
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Turn right
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Turn left

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)

    def is_trapped(self):
        # Check if the snake's head is trapped between its body parts with only one or no escape route.
        if len(self.snake) < 4:  # Too short to trap itself
            return False
            
        head = self.snake[0]
        # Check all adjacent positions around the head
        adjacent_points = [
            Point(head.x + self.block_size, head.y),  # Right
            Point(head.x - self.block_size, head.y),  # Left
            Point(head.x, head.y + self.block_size),  # Down
            Point(head.x, head.y - self.block_size)   # Up
        ]
        
        # Count how many escape routes are available
        escape_routes = 0
        for point in adjacent_points:
            # A point is an escape route if it's not a collision
            if not self._is_collision(point):
                escape_routes += 1
    
        # If there's only one or no escape route, the snake is trapped
        return escape_routes <= 1

    def step(self, action, update_ui=True):
        self.frame_iteration += 1
        
        if update_ui:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # Move
        self._move(action)
        self.snake.insert(0, self.head)

        # Check if game over
        reward = -0.5 # if nothing happend (default value) # swalnya pake -0.1 cmn skrng envourage efficiency
        game_over = 0
        if self._is_collision():
            game_over = 1
            reward -= 10
            return reward, game_over, self.score

        # Check if eating food
        if self.head == self.food:
            self.score += 1
            reward += 10
            self._place_food()
        else:
            self.snake.pop()

        if self.is_trapped():
            reward -= 5

        # Update UI only if requested
        if update_ui:
            self._update_ui()
            self.clock.tick(self.speed)

        return reward, game_over, self.score
    
    def _update_ui(self):
        self.display.fill((0,0,0))  # Black background

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, (0,255,0), pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, (0,200,0), pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, (255,0,0), pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        pygame.display.flip()

def main(): # for manual testing
    game = SnakeGame(speed=1)  # Lower speed for manual play
    
    # Game loop
    while True:
        state = game.get_state()
        
        # Manual control
        action = [1, 0, 0]  # Default: go straight
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = [0, 0, 1]  # Turn left
                elif event.key == pygame.K_RIGHT:
                    action = [0, 1, 0]  # Turn right
                elif event.key == pygame.K_q:
                    pygame.quit()
                    quit()
        
        reward, game_over, score = game.step(action)
        print(state, reward, game_over, score)
        
        if game_over:
            print(f'Final Score: {score}')
            game.reset()

if __name__ == '__main__':
    pygame.init()
    main()