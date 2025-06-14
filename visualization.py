"""
Visualization module for the Artificial World Simulation.

Contains the real-time visualization system using Pygame for displaying
the simulation world, agents, and user interface.
"""

import pygame
from typing import List
import time

from config import config, AgentType


class Visualization:
    """
    Real-time visualization system using Pygame.
    
    This class handles:
    - Rendering the world, agents, and objects
    - Displaying real-time statistics and controls
    - Managing the graphical user interface
    - Performance optimization for smooth animation
    
    The visualization is crucial for understanding emergent behaviors
    and monitoring the evolutionary process in real-time.
    """
    
    def __init__(self):
        """Initialize Pygame and create the display window."""
        pygame.init()
        
        # Create main display window (world + UI panel)
        self.screen_width = config.WORLD_WIDTH + 300  # Extra space for UI
        self.screen_height = config.WORLD_HEIGHT
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Genetic Algorithm World Simulation")
        
        # Initialize fonts for text rendering
        self.font = pygame.font.Font(None, 24)          # Main text
        self.small_font = pygame.font.Font(None, 18)    # Details and stats
        self.title_font = pygame.font.Font(None, 28)    # Titles and headers
        
        # Performance management
        self.clock = pygame.time.Clock()
        
        # UI Layout constants
        self.world_rect = (0, 0, config.WORLD_WIDTH, config.WORLD_HEIGHT)
        self.ui_x = config.WORLD_WIDTH + 10  # Start of UI panel
        
    def draw(self, world, ga1, ga2, analytics):
        """
        Render the complete simulation display.
        
        This is the main drawing function called each frame. It coordinates
        all visual elements to create a comprehensive view of the simulation.
        
        Args:
            world: The environment containing food, hazards, and physics
            ga1: Cooperative population genetic algorithm
            ga2: Aggressive population genetic algorithm  
            analytics: Data collection and analysis system
        """
        # Clear screen with background color
        self.screen.fill(config.COLORS['BACKGROUND'])
        
        # Draw world boundary for visual clarity
        pygame.draw.rect(self.screen, (60, 60, 60), self.world_rect, 2)
        
        # Render world objects (food and hazards)
        self._draw_world_objects(world)
        
        # Render all agents with status indicators
        self._draw_agents(ga1.population + ga2.population)
        
        # Render user interface with statistics and controls
        self._draw_ui(ga1, ga2, analytics)
        
        # Update display (double buffering)
        pygame.display.flip()
    
    def _draw_world_objects(self, world):
        """
        Render food items and hazards in the world.
        
        Visual design choices:
        - Food: Green circles (representing energy/nutrition)
        - Hazards: Red squares (representing danger/obstacles)
        
        Args:
            world: World containing objects to render
        """
        # Draw food items as green circles
        for food in world.food:
            pygame.draw.circle(
                self.screen, 
                config.COLORS['FOOD'], 
                (food.x + 10, food.y + 10),  # Center the circle
                8  # Radius
            )
            
            # Optional: Add subtle glow effect for better visibility
            pygame.draw.circle(
                self.screen, 
                (0, 200, 0),  # Slightly darker green
                (food.x + 10, food.y + 10), 
                10, 
                1  # Outline only
            )
        
        # Draw hazards as red squares
        for hazard in world.hazards:
            # Draw larger, more visible hazard area
            pygame.draw.rect(
                self.screen, 
                config.COLORS['HAZARD'], 
                (hazard.x - 5, hazard.y - 5, 30, 30)  # Larger hazard zone
            )
            
            # Add pulsing warning border
            pulse = int(time.time() * 10) % 2  # Pulse every 0.1 seconds
            border_color = (255, 100, 100) if pulse else (150, 0, 0)
            pygame.draw.rect(
                self.screen, 
                border_color,
                (hazard.x - 8, hazard.y - 8, 36, 36), 
                3  # Thick warning border
            )
            
            # Add danger symbol (X)
            font = pygame.font.Font(None, 20)
            danger_text = font.render("X", True, (255, 255, 255))
            self.screen.blit(danger_text, (hazard.x + 5, hazard.y + 5))
    
    def _draw_agents(self, agents: List):
        """
        Render all agents with visual indicators for their status.
        
        Visual elements:
        - Agent body: Circle colored by population type
        - Energy bar: Shows current energy level above agent
        - Status indicators: Visual cues for behavior state
        
        Args:
            agents: List of all agents to render
        """
        for agent in agents:
            if not agent.alive:
                continue  # Don't render dead agents
            
            # Choose color based on agent population
            base_color = (config.COLORS['AGENT_GA1'] if agent.agent_type == AgentType.COOPERATIVE 
                         else config.COLORS['AGENT_GA2'])
            
            # Flash red if agent is taking hazard damage
            if hasattr(agent, 'in_hazard') and agent.in_hazard > 0:
                agent.in_hazard -= 1  # Countdown the flash effect
                color = (255, 100, 100)  # Flash red when in hazard
            else:
                color = base_color
            
            # Calculate agent size based on genome (5-15 pixel radius)
            agent_radius = 5 + int(agent.genome.size * 10)
            
            # Draw agent body as filled circle
            pygame.draw.circle(
                self.screen, 
                color, 
                (int(agent.x + 10), int(agent.y + 10)),  # Center position
                agent_radius  # Radius based on size trait
            )
            
            # Draw agent outline for better visibility
            pygame.draw.circle(
                self.screen, 
                (255, 255, 255),  # White outline
                (int(agent.x + 10), int(agent.y + 10)), 
                agent_radius, 
                1  # Outline thickness
            )
            
            # Draw energy bar above agent
            self._draw_energy_bar(agent)
            
            # Optional: Draw additional status indicators
            self._draw_agent_status_indicators(agent)
    
    def _draw_energy_bar(self, agent):
        """
        Draw energy bar above an agent showing current energy level.
        
        The energy bar uses color coding:
        - Green: High energy (healthy)
        - Yellow: Medium energy (caution)
        - Red: Low energy (danger)
        
        Args:
            agent: Agent to draw energy bar for
        """
        bar_width = 20
        bar_height = 4
        bar_x = int(agent.x)
        bar_y = int(agent.y - 8)
        
        # Calculate energy ratio (0.0 to 1.0)
        energy_ratio = max(0, min(1, agent.energy / config.AGENT_ENERGY))
        
        # Draw background (empty energy)
        pygame.draw.rect(
            self.screen, 
            (60, 60, 60),  # Dark gray background
            (bar_x, bar_y, bar_width, bar_height)
        )
        
        # Choose energy bar color based on energy level
        if energy_ratio > 0.6:
            energy_color = (0, 255, 0)      # Green (healthy)
        elif energy_ratio > 0.3:
            energy_color = (255, 255, 0)    # Yellow (caution)
        else:
            energy_color = (255, 0, 0)      # Red (danger)
        
        # Draw current energy level
        current_width = int(bar_width * energy_ratio)
        if current_width > 0:
            pygame.draw.rect(
                self.screen, 
                energy_color,
                (bar_x, bar_y, current_width, bar_height)
            )
    
    def _draw_agent_status_indicators(self, agent):
        """
        Draw additional visual indicators for agent status and behavior.
        
        This could include:
        - Fitness level indicators
        - Behavioral state icons
        - Age/generation markers
        
        Args:
            agent: Agent to draw indicators for
        """
        # Example: Draw small fitness indicator
        if agent.fitness > 50:  # High fitness agents get a small star
            star_x = int(agent.x + 15)
            star_y = int(agent.y + 5)
            pygame.draw.circle(self.screen, (255, 255, 0), (star_x, star_y), 2)
    
    def _draw_ui(self, ga1, ga2, analytics):
        """
        Render the user interface panel with statistics and controls.
        
        The UI provides:
        - Real-time population statistics
        - Generation information
        - Performance comparisons
        - Control instructions
        - Visual feedback on simulation state
        
        Args:
            ga1: Cooperative population
            ga2: Aggressive population
            analytics: Analysis system with collected data
        """
        y_offset = 10  # Starting Y position for UI elements
        
        # SECTION 1: Title and Generation Info
        title = self.title_font.render("Evolution Monitor", True, config.COLORS['TEXT'])
        self.screen.blit(title, (self.ui_x, y_offset))
        y_offset += 35
        
        # Current generation number
        current_gen = max(ga1.generation, ga2.generation)
        gen_text = f"Generation: {current_gen}"
        gen_surface = self.font.render(gen_text, True, config.COLORS['TEXT'])
        self.screen.blit(gen_surface, (self.ui_x, y_offset))
        y_offset += 30
        
        # SECTION 2: GA1 (Cooperative) Statistics
        y_offset = self._draw_population_stats(ga1, "GA1 (Cooperative)", 
                                             config.COLORS['AGENT_GA1'], y_offset)
        y_offset += 20
        
        # SECTION 3: GA2 (Aggressive) Statistics  
        y_offset = self._draw_population_stats(ga2, "GA2 (Aggressive)", 
                                             config.COLORS['AGENT_GA2'], y_offset)
        y_offset += 20
        
        # SECTION 4: Comparative Analysis
        y_offset = self._draw_comparative_analysis(ga1, ga2, y_offset)
        y_offset += 20
        
        # SECTION 5: Control Instructions
        self._draw_controls(y_offset)
    
    def _draw_population_stats(self, ga, title: str, color: tuple, y_start: int) -> int:
        """
        Draw statistics for one population.
        
        Args:
            ga: Genetic algorithm population to display
            title: Display name for the population
            color: Color to use for population identification
            y_start: Starting Y coordinate for this section
            
        Returns:
            int: Next available Y coordinate after this section
        """
        y_offset = y_start
        
        # Population title with colored indicator
        title_surface = self.font.render(title, True, color)
        self.screen.blit(title_surface, (self.ui_x, y_offset))
        y_offset += 25
        
        # Calculate current statistics
        alive_agents = [a for a in ga.population if a.alive]
        total_agents = len(alive_agents)
        
        if total_agents > 0:
            avg_fitness = sum(a.fitness for a in alive_agents) / total_agents
            avg_energy = sum(a.energy for a in alive_agents) / total_agents
            avg_food = sum(a.food_collected for a in alive_agents) / total_agents
            avg_age = sum(a.age for a in alive_agents) / total_agents
        else:
            avg_fitness = avg_energy = avg_food = avg_age = 0
        
        # Display statistics
        stats = [
            f"  Population: {total_agents}/{self._get_max_population_for_type(ga)}",
            f"  Avg Fitness: {avg_fitness:.1f}",
            f"  Avg Energy: {avg_energy:.1f}",
            f"  Avg Food: {avg_food:.1f}",
            f"  Avg Age: {avg_age:.0f}"
        ]
        
        for stat in stats:
            stat_surface = self.small_font.render(stat, True, config.COLORS['TEXT'])
            self.screen.blit(stat_surface, (self.ui_x, y_offset))
            y_offset += 18
        
        return y_offset
    
    def _draw_comparative_analysis(self, ga1, ga2, y_start: int) -> int:
        """
        Draw comparative analysis between the two populations.
        
        Args:
            ga1: Cooperative population
            ga2: Aggressive population  
            y_start: Starting Y coordinate
            
        Returns:
            int: Next available Y coordinate
        """
        y_offset = y_start
        
        # Section title
        comparison_title = self.font.render("Strategy Comparison", True, config.COLORS['TEXT'])
        self.screen.blit(comparison_title, (self.ui_x, y_offset))
        y_offset += 25
        
        # Calculate comparison metrics
        ga1_alive = len([a for a in ga1.population if a.alive])
        ga2_alive = len([a for a in ga2.population if a.alive])
        
        if ga1_alive > 0:
            ga1_avg_fitness = sum(a.fitness for a in ga1.population if a.alive) / ga1_alive
        else:
            ga1_avg_fitness = 0
            
        if ga2_alive > 0:
            ga2_avg_fitness = sum(a.fitness for a in ga2.population if a.alive) / ga2_alive
        else:
            ga2_avg_fitness = 0
        
        # Determine current leader
        if ga1_avg_fitness > ga2_avg_fitness:
            leader = "Cooperative (GA1)"
            leader_color = config.COLORS['AGENT_GA1']
        elif ga2_avg_fitness > ga1_avg_fitness:
            leader = "Aggressive (GA2)"  
            leader_color = config.COLORS['AGENT_GA2']
        else:
            leader = "Tied"
            leader_color = config.COLORS['TEXT']
        
        # Display comparison
        leader_text = f"  Current Leader: {leader}"
        leader_surface = self.small_font.render(leader_text, True, leader_color)
        self.screen.blit(leader_surface, (self.ui_x, y_offset))
        y_offset += 20
        
        # Survival rates
        ga1_survival = ga1_alive / config.POPULATION_SIZE_GA1
        ga2_survival = ga2_alive / config.POPULATION_SIZE_GA2
        
        survival_text = f"  Survival: GA1 {ga1_survival:.1%}, GA2 {ga2_survival:.1%}"
        survival_surface = self.small_font.render(survival_text, True, config.COLORS['TEXT'])
        self.screen.blit(survival_surface, (self.ui_x, y_offset))
        y_offset += 20
        
        return y_offset
    
    def _draw_controls(self, y_start: int):
        """
        Draw control instructions for user interaction.
        
        Args:
            y_start: Starting Y coordinate for controls section
        """
        y_offset = y_start
        
        # Controls title
        controls_title = self.font.render("Controls", True, config.COLORS['TEXT'])
        self.screen.blit(controls_title, (self.ui_x, y_offset))
        y_offset += 25
        
        # Control instructions
        controls = [
            "SPACE - Pause/Resume",
            "R - Reset Simulation", 
            "S - Save Data",
            "A - Analytics Graphs",
            "G - Full Report",
            "ESC - Exit",
            "",
            "⚠️  Extinct populations",
            "will NOT restart!",
            "",
            "🔥 Red pulsing = Hazards",
            "🍎 GA1 eats food",
            "🦁 GA2 hunts GA1",
            "",
            "Watch for emergent",
            "behaviors and strategy",
            "evolution over time!"
        ]
        
        for control in controls:
            if control == "":  # Empty line for spacing
                y_offset += 10
                continue
                
            color = config.COLORS['TEXT'] if not control.startswith("Watch") else (180, 180, 180)
            control_surface = self.small_font.render(control, True, color)
            self.screen.blit(control_surface, (self.ui_x, y_offset))
            y_offset += 16
    
    def _get_max_population_for_type(self, ga):
        """Get the maximum population size for the given GA type."""
        from config import AgentType
        if ga.agent_type == AgentType.COOPERATIVE:
            return config.POPULATION_SIZE_GA1
        else:
            return config.POPULATION_SIZE_GA2 