"""
Evolving World Simulation
-------------------------
A simulation of agents that must find food to survive and reproduce.
Agents lose energy as they move and must find food to replenish their energy.
Every 60 seconds, surviving agents reproduce, creating the next generation.
Agents move like particles with collision physics.
"""

import pygame
import random
import sys
import time
import math

# Initialize Pygame
pygame.init()

# Simulation Constants
# -------------------
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Agent Constants
# --------------
ENERGY_LOSS_PER_MOVE = 0.1  # Energy units lost per movement
REPRODUCTION_INTERVAL = 60  # Seconds between reproduction cycles
FOOD_ENERGY = 50  # Energy units gained from consuming food
MAX_SPEED = 3.0  # Maximum speed of agents
INITIAL_SPEED = 2.0  # Initial speed of new agents

# Visual Constants
# ---------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Food:
    """
    Represents a food source in the simulation.
    
    Attributes:
        x (int): X-coordinate of the food
        y (int): Y-coordinate of the food
        radius (int): Visual radius of the food
        color (tuple): RGB color of the food
    """
    
    def __init__(self):
        """Initialize a new food source at a random position."""
        self.x = random.randint(0, WINDOW_WIDTH)
        self.y = random.randint(0, WINDOW_HEIGHT)
        self.radius = 5
        self.color = BLUE

    def draw(self, screen):
        """
        Draw the food source on the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class Agent:
    """
    Represents an agent in the simulation.
    
    Attributes:
        x (float): X-coordinate of the agent
        y (float): Y-coordinate of the agent
        vx (float): X-component of velocity
        vy (float): Y-component of velocity
        radius (int): Visual radius of the agent
        color (tuple): RGB color of the agent
        speed (float): Current speed of the agent
        energy (float): Current energy level
        generation (int): Current generation number
    """
    
    def __init__(self, x, y, generation=1):
        """
        Initialize a new agent.
        
        Args:
            x (float): Initial X-coordinate
            y (float): Initial Y-coordinate
            generation (int): Generation number (default: 1)
        """
        self.x = x
        self.y = y
        # Initialize with random direction
        angle = random.uniform(0, 2 * math.pi)
        self.vx = INITIAL_SPEED * math.cos(angle)
        self.vy = INITIAL_SPEED * math.sin(angle)
        self.radius = 10
        self.color = GREEN
        self.speed = INITIAL_SPEED
        self.energy = 100
        self.generation = generation

    def move(self, agents):
        """
        Move the agent and handle collisions with other agents and walls.
        
        Args:
            agents (list): List of all agents in the simulation
        """
        # Update position based on velocity
        self.x += self.vx
        self.y += self.vy
        
        # Handle wall collisions
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = abs(self.vx)
        elif self.x + self.radius > WINDOW_WIDTH:
            self.x = WINDOW_WIDTH - self.radius
            self.vx = -abs(self.vx)
            
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = abs(self.vy)
        elif self.y + self.radius > WINDOW_HEIGHT:
            self.y = WINDOW_HEIGHT - self.radius
            self.vy = -abs(self.vy)
        
        # Handle collisions with other agents
        for other in agents:
            if other is not self:
                dx = other.x - self.x
                dy = other.y - self.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < self.radius + other.radius:
                    # Collision detected - calculate collision response
                    # Normalize collision vector
                    nx = dx / distance
                    ny = dy / distance
                    
                    # Calculate relative velocity
                    dvx = self.vx - other.vx
                    dvy = self.vy - other.vy
                    
                    # Calculate relative velocity in terms of the normal direction
                    velocity_along_normal = dvx * nx + dvy * ny
                    
                    # Do not resolve if velocities are separating
                    if velocity_along_normal > 0:
                        continue
                    
                    # Calculate impulse scalar
                    j = -(1 + 0.9) * velocity_along_normal  # 0.9 is the coefficient of restitution
                    j /= 2  # Equal mass
                    
                    # Apply impulse
                    self.vx -= j * nx
                    self.vy -= j * ny
                    other.vx += j * nx
                    other.vy += j * ny
                    
                    # Move agents apart to prevent sticking
                    overlap = (self.radius + other.radius - distance) / 2
                    self.x -= overlap * nx
                    self.y -= overlap * ny
                    other.x += overlap * nx
                    other.y += overlap * ny
        
        # Update speed and limit to maximum
        self.speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if self.speed > MAX_SPEED:
            self.vx = (self.vx / self.speed) * MAX_SPEED
            self.vy = (self.vy / self.speed) * MAX_SPEED
            self.speed = MAX_SPEED
        
        # Decrease energy based on speed
        self.energy -= ENERGY_LOSS_PER_MOVE * (self.speed / MAX_SPEED)

    def eat(self, food):
        """
        Attempt to eat a food source.
        
        Args:
            food (Food): The food source to attempt to eat
            
        Returns:
            bool: True if food was eaten, False otherwise
        """
        distance = ((self.x - food.x) ** 2 + (self.y - food.y) ** 2) ** 0.5
        if distance < self.radius + food.radius:
            self.energy += FOOD_ENERGY
            return True
        return False

    def draw(self, screen):
        """
        Draw the agent and its UI elements on the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        # Draw agent body
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Draw direction indicator
        end_x = self.x + self.vx * 2
        end_y = self.y + self.vy * 2
        pygame.draw.line(screen, WHITE, (self.x, self.y), (end_x, end_y), 2)
        
        # Draw energy bar
        energy_bar_length = 20
        energy_bar_height = 3
        energy_percentage = self.energy / 100
        
        # Energy bar background
        pygame.draw.rect(screen, WHITE, 
                        (int(self.x - energy_bar_length/2), 
                         int(self.y - self.radius - 5), 
                         energy_bar_length, 
                         energy_bar_height))
        
        # Energy bar fill
        pygame.draw.rect(screen, GREEN, 
                        (int(self.x - energy_bar_length/2), 
                         int(self.y - self.radius - 5), 
                         int(energy_bar_length * energy_percentage), 
                         energy_bar_height))
        
        # Draw generation number
        font = pygame.font.Font(None, 20)
        gen_text = font.render(str(self.generation), True, WHITE)
        screen.blit(gen_text, (int(self.x - 5), int(self.y - self.radius - 20)))

class Simulation:
    """
    Main simulation class that manages the game loop and simulation state.
    
    Attributes:
        screen: Pygame display surface
        clock: Pygame clock for FPS control
        agents (list): List of active agents
        foods (list): List of available food sources
        last_reproduction_time (float): Timestamp of last reproduction cycle
    """
    
    def __init__(self):
        """Initialize the simulation with agents and food."""
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Evolving World Simulation")
        self.clock = pygame.time.Clock()
        
        # Initialize simulation state
        self.agents = [Agent(random.randint(0, WINDOW_WIDTH), 
                           random.randint(0, WINDOW_HEIGHT)) 
                      for _ in range(20)]
        self.foods = [Food() for _ in range(10)]
        self.last_reproduction_time = time.time()

    def handle_events(self):
        """Handle pygame events, including window close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def update(self):
        """
        Update simulation state:
        - Move agents
        - Handle food consumption
        - Remove dead agents
        - Trigger reproduction cycle
        """
        # Update agents
        for agent in self.agents[:]:
            agent.move(self.agents)
            
            # Check for food consumption
            for food in self.foods[:]:
                if agent.eat(food):
                    self.foods.remove(food)
                    self.foods.append(Food())
            
            # Remove dead agents
            if agent.energy <= 0:
                self.agents.remove(agent)
        
        # Reproduction cycle
        current_time = time.time()
        if current_time - self.last_reproduction_time >= REPRODUCTION_INTERVAL:
            self.reproduce_agents()
            self.last_reproduction_time = current_time

    def reproduce_agents(self):
        """
        Create new agents from surviving ones.
        Each surviving agent creates one offspring with an incremented generation number.
        """
        new_agents = []
        for agent in self.agents:
            new_agent = Agent(
                agent.x + random.uniform(-10, 10),
                agent.y + random.uniform(-10, 10),
                agent.generation + 1
            )
            new_agents.append(new_agent)
        self.agents.extend(new_agents)

    def draw(self):
        """
        Draw the current simulation state:
        - Food sources
        - Agents with their UI elements
        - Population statistics
        """
        self.screen.fill(BLACK)
        
        # Draw food sources
        for food in self.foods:
            food.draw(self.screen)
        
        # Draw agents
        for agent in self.agents:
            agent.draw(self.screen)
        
        # Draw population stats
        font = pygame.font.Font(None, 36)
        stats_text = font.render(f"Agents: {len(self.agents)}", True, WHITE)
        self.screen.blit(stats_text, (10, 10))
        
        pygame.display.flip()

    def run(self):
        """Main simulation loop."""
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

if __name__ == "__main__":
    sim = Simulation()
    sim.run() 