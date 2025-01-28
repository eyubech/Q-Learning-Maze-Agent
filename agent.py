import pygame
import random
import numpy as np
import time

pygame.init()

WIDTH, HEIGHT = 900, 900
GRID_SIZE = 30
TILE_SIZE = WIDTH // GRID_SIZE

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.5
EPSILON_DECAY = 0.995

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        self.epsilon = EPSILON
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        
        self.wall_memory = np.zeros((GRID_SIZE, GRID_SIZE))
        self.success_paths = []

    def get_action(self, state, valid_actions):
        x, y = state
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.q_table[y][x][a] if a in valid_actions else -np.inf for a in range(4)]
            return np.argmax(q_values)

    def update_q_table(self, state, action, reward, new_state):
        x, y = state
        new_x, new_y = new_state
        old_value = self.q_table[y][x][action]
        max_future = np.max(self.q_table[new_y][new_x])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * max_future)
        self.q_table[y][x][action] = new_value

class Game:
    def __init__(self, agent):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.agent = agent
        self.player_pos = (0, 0)
        self.target_pos = (GRID_SIZE-1, GRID_SIZE-1)
        self.moves = 0
        self.start_time = time.time()
        self.path_history = []
        self.generate_maze()
        
    def generate_maze(self):
        self.grid.fill(0)
        wall_count = 0
        target_walls = int(GRID_SIZE**2 * 0.45)
        
        while wall_count < target_walls:
            x = random.randint(0, GRID_SIZE-1)
            y = random.randint(0, GRID_SIZE-1)
            if (x, y) not in [self.player_pos, self.target_pos]:
                if self.agent.wall_memory[y][x] > 0.5 or random.random() < 0.6:
                    self.grid[y][x] = 1
                    wall_count += 1
                    self.agent.wall_memory[y][x] = 1

        while not self.bfs():
            self.generate_maze()

    def bfs(self):
        queue = [(self.player_pos[0], self.player_pos[1])]
        visited = set(queue)
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        
        while queue:
            x, y = queue.pop(0)
            if (x, y) == self.target_pos:
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if self.grid[ny][nx] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def get_valid_actions(self, x, y):
        actions = []
        if y > 0 and self.grid[y-1][x] == 0:
            actions.append(0)  # Up
        if y < GRID_SIZE-1 and self.grid[y+1][x] == 0:
            actions.append(1)  # Down
        if x > 0 and self.grid[y][x-1] == 0:
            actions.append(2)  # Left
        if x < GRID_SIZE-1 and self.grid[y][x+1] == 0:
            actions.append(3)  # Right
        return actions

    def move_player(self):
        state = self.player_pos
        valid_actions = self.get_valid_actions(*state)
        
        if not valid_actions:
            return False

        action = self.agent.get_action(state, valid_actions)
        x, y = state
        dx, dy = [(0,-1),(0,1),(-1,0),(1,0)][action]
        new_x = x + dx
        new_y = y + dy

        if (new_x, new_y) == self.target_pos:
            reward = 100
        elif self.grid[new_y][new_x] == 1:
            reward = -50
        elif (new_x, new_y) in self.path_history:
            reward = -1
        else:
            reward = -0.1

        self.agent.update_q_table(state, action, reward, (new_x, new_y))
        
        if (new_x, new_y) == self.target_pos:
            self.player_pos = (new_x, new_y)
            self.agent.success_paths.append(self.path_history + [self.player_pos])
            self.agent.epsilon *= EPSILON_DECAY
            return True
        
        if self.grid[new_y][new_x] == 0:
            self.player_pos = (new_x, new_y)
            self.path_history.append(self.player_pos)
            self.moves += 1
            
        return False

    def draw(self, screen):
        screen.fill(WHITE)
        
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if self.grid[y][x] == 1:
                    pygame.draw.rect(screen, GRAY, rect)
                else:
                    pygame.draw.rect(screen, WHITE, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)
        
        if len(self.path_history) >= 2:
            points = [(pos[0]*TILE_SIZE+TILE_SIZE//2, pos[1]*TILE_SIZE+TILE_SIZE//2) 
                      for pos in self.path_history]
            pygame.draw.lines(screen, BLUE, False, points, 3)
        
        # Draw player and target
        pygame.draw.circle(screen, RED, (self.player_pos[0]*TILE_SIZE+TILE_SIZE//2, 
                                       self.player_pos[1]*TILE_SIZE+TILE_SIZE//2), TILE_SIZE//3)
        pygame.draw.circle(screen, GREEN, (self.target_pos[0]*TILE_SIZE+TILE_SIZE//2,
                                         self.target_pos[1]*TILE_SIZE+TILE_SIZE//2), TILE_SIZE//3)

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Q-Learning Maze Solver")
    agent = QLearningAgent()
    font = pygame.font.Font(None, 36)
    
    total_games = 0
    total_moves = 0
    total_time = 0
    
    running = True
    while running:
        game = Game(agent)
        clock = pygame.time.Clock()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
            
            success = game.move_player()
            game.draw(screen)
            pygame.display.flip()
            clock.tick(200)  
            
            if success or not game.get_valid_actions(*game.player_pos):
                break
        
        if success:
            duration = time.time() - game.start_time
            total_games += 1
            total_moves += game.moves
            total_time += duration
  
            print(f"\nGame {total_games} Results:")
            print(f"Moves: {game.moves}")
            print(f"Time: {duration:.2f}s")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Average Moves: {total_moves/total_games:.1f}")
            print(f"Average Time: {total_time/total_games:.2f}s")

            text = [
                f"Game {total_games} Complete!",
                f"Moves: {game.moves}",
                f"Time: {duration:.2f}s",
                f"Average Moves: {total_moves/total_games:.1f}",
                f"Average Time: {total_time/total_games:.2f}s"
            ]
            
            y_pos = HEIGHT//2 - 80
            for line in text:
                text_surf = font.render(line, True, BLACK)
                screen.blit(text_surf, (WIDTH//2 - 200, y_pos))
                y_pos += 40
            
            pygame.display.update()
            pygame.time.wait(2000)

if __name__ == "__main__":
    main()