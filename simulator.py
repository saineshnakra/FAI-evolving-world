import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class Agent:
    def __init__(self, x, y, agent_type):
        self.x = x
        self.y = y
        self.agent_type = agent_type  # "forager" or "predator"
        self.radius = 10
        self.color = GREEN if agent_type == "forager" else RED
        self.speed = 2
        self.energy = 100

    def move(self):
        # Basic random movement
        self.x += random.uniform(-self.speed, self.speed)
        self.y += random.uniform(-self.speed, self.speed)
        
        # Keep agents within screen bounds
        self.x = max(self.radius, min(WINDOW_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(WINDOW_HEIGHT - self.radius, self.y))

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class Simulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Evolving World Simulation")
        self.clock = pygame.time.Clock()
        
        # Initialize populations
        self.foragers = [Agent(random.randint(0, WINDOW_WIDTH), 
                             random.randint(0, WINDOW_HEIGHT), 
                             "forager") for _ in range(20)]
        self.predators = [Agent(random.randint(0, WINDOW_WIDTH), 
                              random.randint(0, WINDOW_HEIGHT), 
                              "predator") for _ in range(10)]

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def update(self):
        # Update all agents
        for agent in self.foragers + self.predators:
            agent.move()

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw all agents
        for agent in self.foragers + self.predators:
            agent.draw(self.screen)
        
        pygame.display.flip()

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

if __name__ == "__main__":
    sim = Simulation()
    sim.run() 