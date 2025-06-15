"""
Visualization module for the Artificial World Simulation.

Contains the real-time visualization system using Pygame for displaying
the simulation world, agents, and comprehensive analytics interface.
"""

import pygame
from typing import List
import time
import math

from config import config, AgentType


class Visualization:
    """
    Real-time visualization system using Pygame.
    
    This class handles:
    - Rendering the world, agents, and objects
    - Displaying comprehensive real-time statistics
    - Managing the graphical user interface
    - Performance optimization for smooth animation
    - Enhanced analytics visualization
    
    The visualization provides crucial insights into evolutionary dynamics
    and emergent behaviors in real-time.
    """
    
    def __init__(self):
        """Initialize Pygame and create the display window."""
        pygame.init()
        
        # Create main display window (world + expanded UI panel)
        self.screen_width = config.WORLD_WIDTH + 350  # Expanded UI space
        self.screen_height = config.WORLD_HEIGHT
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Genetic Algorithm World Simulation - Enhanced Analytics")
        
        # Initialize fonts for text rendering
        self.font = pygame.font.Font(None, 22)          # Main text
        self.small_font = pygame.font.Font(None, 16)    # Details and stats
        self.title_font = pygame.font.Font(None, 26)    # Titles and headers
        self.tiny_font = pygame.font.Font(None, 14)     # Very small details
        
        # Performance management
        self.clock = pygame.time.Clock()
        
        # UI Layout constants
        self.world_rect = (0, 0, config.WORLD_WIDTH, config.WORLD_HEIGHT)
        self.ui_x = config.WORLD_WIDTH + 10  # Start of UI panel
        
        # Color scheme for enhanced visibility
        self.colors = {
            'background': (20, 20, 30),
            'world_border': (60, 60, 80),
            'text': (220, 220, 220),
            'text_dim': (160, 160, 160),
            'success': (0, 200, 100),
            'warning': (255, 200, 0),
            'danger': (255, 100, 100),
            'neutral': (150, 150, 150)
        }
    
    def draw(self, world, ga1, ga2, analytics):
        """
        Render the complete simulation display with enhanced analytics.
        
        Args:
            world: The environment containing food, hazards, and physics
            ga1: Cooperative population genetic algorithm
            ga2: Aggressive population genetic algorithm  
            analytics: Enhanced data collection and analysis system
        """
        # Clear screen with dark background
        self.screen.fill(self.colors['background'])
        
        # Draw world boundary
        pygame.draw.rect(self.screen, self.colors['world_border'], self.world_rect, 2)
        
        # Render world objects (food and hazards)
        self._draw_world_objects(world)
        
        # Render all agents with enhanced status indicators
        self._draw_agents(ga1.population + ga2.population)
        
        # Render comprehensive user interface
        self._draw_enhanced_ui(ga1, ga2, analytics, world)
        
        # Update display (double buffering)
        pygame.display.flip()
    
    def _draw_world_objects(self, world):
        """
        Render food items and hazards with enhanced visual effects.
        
        Args:
            world: World containing objects to render
        """
        # Draw food items with pulsing effect
        pulse = math.sin(time.time() * 3) * 0.2 + 0.8  # Pulse between 0.6 and 1.0
        
        for food in world.food:
            # Main food circle
            food_color = (int(100 * pulse), int(200 * pulse), int(50 * pulse))
            pygame.draw.circle(
                self.screen, 
                food_color, 
                (food.x + 10, food.y + 10),
                8
            )
            
            # Glow effect
            glow_color = (0, int(150 * pulse), 0)
            pygame.draw.circle(
                self.screen, 
                glow_color,
                (food.x + 10, food.y + 10), 
                12, 
                2
            )
        
        # Draw hazards with warning animation
        warning_pulse = int(time.time() * 8) % 2  # Fast pulse
        
        for hazard in world.hazards:
            # Main hazard area
            hazard_color = (200, 50, 50) if warning_pulse else (150, 30, 30)
            pygame.draw.rect(
                self.screen, 
                hazard_color, 
                (hazard.x - 5, hazard.y - 5, 30, 30)
            )
            
            # Animated warning border
            border_color = (255, 100, 100) if warning_pulse else (200, 60, 60)
            pygame.draw.rect(
                self.screen, 
                border_color,
                (hazard.x - 8, hazard.y - 8, 36, 36), 
                3
            )
            
            # Warning symbol - using X instead of emoji
            symbol_color = (255, 255, 255) if warning_pulse else (200, 200, 200)
            danger_text = self.small_font.render("X", True, symbol_color)
            self.screen.blit(danger_text, (hazard.x + 8, hazard.y + 8))
    
    def _draw_agents(self, agents: List):
        """
        Render all agents with enhanced visual indicators and genetic traits display.
        
        Args:
            agents: List of all agents to render
        """
        for agent in agents:
            if not agent.alive:
                continue
            
            # Base colors
            base_color = (config.COLORS['AGENT_GA1'] if agent.agent_type == AgentType.COOPERATIVE 
                         else config.COLORS['AGENT_GA2'])
            
            # Flash effects for status
            color = base_color
            if hasattr(agent, 'in_hazard') and agent.in_hazard > 0:
                agent.in_hazard -= 1
                color = (255, 150, 150)  # Flash red in hazard
            
            # Calculate agent size based on genome (visual representation of size trait)
            base_radius = 6
            size_modifier = agent.genome.size * 8  # 0-8 additional pixels
            agent_radius = int(base_radius + size_modifier)
            
            # Draw agent body with fitness-based brightness
            fitness_modifier = min(1.0, agent.fitness / 100.0)  # Normalize fitness
            bright_color = tuple(int(c * (0.5 + 0.5 * fitness_modifier)) for c in color)
            
            pygame.draw.circle(
                self.screen, 
                bright_color, 
                (int(agent.x + 10), int(agent.y + 10)),
                agent_radius
            )
            
            # Draw genetic trait indicators
            self._draw_genetic_indicators(agent, agent_radius)
            
            # Draw energy bar
            self._draw_energy_bar(agent)
            
            # Draw vision range (for high-sense agents)
            if agent.genome.sense > 0.7:  # Only show for agents with good vision
                self._draw_vision_range(agent)
    
    def _draw_genetic_indicators(self, agent, radius):
        """
        Draw small indicators showing genetic traits.
        
        Args:
            agent: Agent to draw indicators for
            radius: Agent's visual radius
        """
        center_x = int(agent.x + 10)
        center_y = int(agent.y + 10)
        
        # Speed indicator (small line showing speed trait)
        if agent.genome.speed > 0.6:
            speed_length = int(3 + agent.genome.speed * 5)
            pygame.draw.line(
                self.screen,
                (255, 255, 100),  # Yellow for speed
                (center_x - speed_length, center_y - radius - 3),
                (center_x + speed_length, center_y - radius - 3),
                2
            )
        
        # Size indicator (border thickness)
        if agent.genome.size > 0.5:
            border_thickness = max(1, int(agent.genome.size * 3))
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (center_x, center_y),
                radius,
                border_thickness
            )
    
    def _draw_vision_range(self, agent):
        """
        Draw vision range for agents with high sense trait.
        
        Args:
            agent: Agent to draw vision range for
        """
        center_x = int(agent.x + 10)
        center_y = int(agent.y + 10)
        vision_range = int(agent.get_vision_range())
        
        # Draw subtle vision circle
        pygame.draw.circle(
            self.screen,
            (100, 100, 100, 50),  # Semi-transparent gray
            (center_x, center_y),
            vision_range,
            1
        )
    
    def _draw_energy_bar(self, agent):
        """
        Draw enhanced energy bar with genetic efficiency indicator.
        
        Args:
            agent: Agent to draw energy bar for
        """
        bar_width = 24
        bar_height = 5
        bar_x = int(agent.x - 2)
        bar_y = int(agent.y - 12)
        
        # Calculate energy ratio
        energy_ratio = max(0, min(1, agent.energy / config.AGENT_ENERGY))
        
        # Background
        pygame.draw.rect(
            self.screen, 
            (40, 40, 40),
            (bar_x, bar_y, bar_width, bar_height)
        )
        
        # Energy level with color coding
        if energy_ratio > 0.7:
            energy_color = (0, 200, 0)      # Green
        elif energy_ratio > 0.4:
            energy_color = (200, 200, 0)    # Yellow
        elif energy_ratio > 0.2:
            energy_color = (255, 150, 0)    # Orange
        else:
            energy_color = (255, 50, 50)    # Red
        
        current_width = int(bar_width * energy_ratio)
        if current_width > 0:
            pygame.draw.rect(
                self.screen, 
                energy_color,
                (bar_x, bar_y, current_width, bar_height)
            )
        
        # Efficiency indicator (small dot for genetic efficiency)
        efficiency = (agent.genome.speed + agent.genome.sense + (1-agent.genome.size)) / 3
        if efficiency > 0.6:
            pygame.draw.circle(
                self.screen,
                (0, 255, 255),  # Cyan for efficiency
                (bar_x + bar_width + 3, bar_y + 2),
                2
            )
    
    def _draw_enhanced_ui(self, ga1, ga2, analytics, world):
        """
        Render comprehensive user interface with enhanced analytics.
        
        Args:
            ga1: Cooperative population
            ga2: Aggressive population
            analytics: Analytics system
            world: World state
        """
        y_offset = 10
        
        # SECTION 1: Title and Status
        title = self.title_font.render("Evolution Analytics", True, self.colors['text'])
        self.screen.blit(title, (self.ui_x, y_offset))
        y_offset += 30
        
        # Current generation and frame info
        current_gen = max(ga1.generation, ga2.generation)
        frame_info = f"Gen: {current_gen} | Frame: {world.generation_time}"
        frame_surface = self.font.render(frame_info, True, self.colors['text'])
        self.screen.blit(frame_surface, (self.ui_x, y_offset))
        y_offset += 25
        
        # Real-time stats
        real_time_stats = analytics.get_real_time_stats()
        if real_time_stats:
            trend_ga1 = real_time_stats.get('population_trend_ga1', 'stable')
            trend_ga2 = real_time_stats.get('population_trend_ga2', 'stable')
            
            trend_color_ga1 = self.colors['success'] if trend_ga1 == 'increasing' else self.colors['danger'] if trend_ga1 == 'decreasing' else self.colors['neutral']
            trend_color_ga2 = self.colors['success'] if trend_ga2 == 'increasing' else self.colors['danger'] if trend_ga2 == 'decreasing' else self.colors['neutral']
            
            trend_text = f"Trends: GA1 {trend_ga1}, GA2 {trend_ga2}"
            trend_surface = self.small_font.render(trend_text, True, self.colors['text_dim'])
            self.screen.blit(trend_surface, (self.ui_x, y_offset))
            y_offset += 20
        
        y_offset += 10
        
        # SECTION 2: Enhanced Population Statistics
        y_offset = self._draw_enhanced_population_stats(ga1, "GA1 (Cooperative)", 
                                                       config.COLORS['AGENT_GA1'], y_offset, analytics)
        y_offset += 15
        
        y_offset = self._draw_enhanced_population_stats(ga2, "GA2 (Aggressive)", 
                                                       config.COLORS['AGENT_GA2'], y_offset, analytics)
        y_offset += 15
        
        # SECTION 3: Genetic Evolution Display
        y_offset = self._draw_genetic_evolution_display(ga1, ga2, analytics, y_offset)
        y_offset += 15
        
        # SECTION 4: Environmental Status
        y_offset = self._draw_environmental_status(world, y_offset)
        y_offset += 15
        
        # SECTION 5: Performance Comparison
        y_offset = self._draw_performance_comparison(ga1, ga2, analytics, y_offset)
        y_offset += 15
        
        # SECTION 6: Controls
        self._draw_enhanced_controls(y_offset)
    
    def _draw_enhanced_population_stats(self, ga, title: str, color: tuple, y_start: int, analytics) -> int:
        """
        Draw enhanced statistics for one population including genetic traits.
        
        Args:
            ga: Genetic algorithm population
            title: Display name
            color: Population color
            y_start: Starting Y coordinate
            analytics: Analytics system
            
        Returns:
            Next available Y coordinate
        """
        y_offset = y_start
        
        # Population title with colored indicator
        title_surface = self.font.render(title, True, color)
        self.screen.blit(title_surface, (self.ui_x, y_offset))
        y_offset += 22
        
        # Calculate statistics
        alive_agents = [a for a in ga.population if a.alive]
        total_agents = len(alive_agents)
        max_pop = config.POPULATION_SIZE_GA1 if ga.agent_type == AgentType.COOPERATIVE else config.POPULATION_SIZE_GA2
        
        if total_agents > 0:
            # Basic stats
            avg_fitness = sum(a.fitness for a in alive_agents) / total_agents
            avg_energy = sum(a.energy for a in alive_agents) / total_agents
            avg_age = sum(a.age for a in alive_agents) / total_agents
            
            # Genetic averages
            avg_speed = sum(a.genome.speed for a in alive_agents) / total_agents
            avg_sense = sum(a.genome.sense for a in alive_agents) / total_agents
            avg_size = sum(a.genome.size for a in alive_agents) / total_agents
            
            # Performance metrics
            if ga.agent_type == AgentType.COOPERATIVE:
                avg_food = sum(a.food_collected for a in alive_agents) / total_agents
                performance_metric = f"Food: {avg_food:.1f}"
            else:
                total_attacks = sum(a.attacks_made for a in alive_agents)
                successful_attacks = sum(getattr(a, 'successful_attacks', 0) for a in alive_agents)
                success_rate = successful_attacks / max(1, total_attacks)
                performance_metric = f"Hunt: {success_rate:.1%}"
        else:
            avg_fitness = avg_energy = avg_age = 0
            avg_speed = avg_sense = avg_size = 0
            performance_metric = "Extinct"
        
        # Display stats with color coding
        survival_rate = total_agents / max_pop
        survival_color = self.colors['success'] if survival_rate > 0.7 else self.colors['warning'] if survival_rate > 0.3 else self.colors['danger']
        
        stats = [
            (f"  Pop: {total_agents}/{max_pop} ({survival_rate:.1%})", survival_color),
            (f"  Fitness: {avg_fitness:.1f}", self.colors['text']),
            (f"  Energy: {avg_energy:.0f}", self.colors['text']),
            (f"  Age: {avg_age:.0f}", self.colors['text']),
            (f"  {performance_metric}", self.colors['text']),
            (f"  Genes: S{avg_speed:.2f} V{avg_sense:.2f} Z{avg_size:.2f}", self.colors['text_dim'])
        ]
        
        for stat_text, stat_color in stats:
            stat_surface = self.small_font.render(stat_text, True, stat_color)
            self.screen.blit(stat_surface, (self.ui_x, y_offset))
            y_offset += 16
        
        return y_offset
    
    def _draw_genetic_evolution_display(self, ga1, ga2, analytics, y_start: int) -> int:
        """
        Display genetic evolution trends and selection pressure.
        
        Args:
            ga1: GA1 population
            ga2: GA2 population  
            analytics: Analytics system
            y_start: Starting Y coordinate
            
        Returns:
            Next Y coordinate
        """
        y_offset = y_start
        
        # Section title
        evolution_title = self.font.render("Genetic Evolution", True, self.colors['text'])
        self.screen.blit(evolution_title, (self.ui_x, y_offset))
        y_offset += 22
        
        # Display trait evolution for both populations
        for pop_name, ga, color in [('GA1', ga1, config.COLORS['AGENT_GA1']), 
                                   ('GA2', ga2, config.COLORS['AGENT_GA2'])]:
            
            # Get trait-fitness correlations if available
            if hasattr(analytics.genome_analyzer, 'trait_fitness_correlations'):
                correlations = analytics.genome_analyzer.trait_fitness_correlations.get(pop_name, {})
                
                trait_text = f"  {pop_name}:"
                trait_surface = self.small_font.render(trait_text, True, color)
                self.screen.blit(trait_surface, (self.ui_x, y_offset))
                y_offset += 16
                
                # Show selection pressure for each trait
                for trait in ['speed', 'sense', 'size']:
                    if trait in correlations and correlations[trait]:
                        recent_corr = correlations[trait][-1] if correlations[trait] else 0
                        
                        # Determine selection strength
                        if abs(recent_corr) > 0.5:
                            strength = "STRONG"
                            pressure_color = self.colors['danger']
                        elif abs(recent_corr) > 0.2:
                            strength = "MOD"
                            pressure_color = self.colors['warning']
                        else:
                            strength = "WEAK"
                            pressure_color = self.colors['neutral']
                        
                        direction = "+" if recent_corr > 0 else "-"
                        pressure_text = f"    {trait[0].upper()}: {direction}{strength}"
                        pressure_surface = self.tiny_font.render(pressure_text, True, pressure_color)
                        self.screen.blit(pressure_surface, (self.ui_x, y_offset))
                        y_offset += 14
        
        return y_offset
    
    def _draw_environmental_status(self, world, y_start: int) -> int:
        """
        Display environmental status and resource availability.
        
        Args:
            world: World state
            y_start: Starting Y coordinate
            
        Returns:
            Next Y coordinate
        """
        y_offset = y_start
        
        # Section title
        env_title = self.font.render("Environment", True, self.colors['text'])
        self.screen.blit(env_title, (self.ui_x, y_offset))
        y_offset += 22
        
        # Environmental stats
        food_count = len(world.food)
        hazard_count = len(world.hazards)
        world_time = world.generation_time
        max_time = world.max_generation_time
        
        # Color code resource availability
        food_color = self.colors['success'] if food_count > 15 else self.colors['warning'] if food_count > 5 else self.colors['danger']
        time_color = self.colors['warning'] if world_time > max_time * 0.8 else self.colors['text']
        
        env_stats = [
            (f"  Food: {food_count}", food_color),
            (f"  Hazards: {hazard_count}", self.colors['text']),
            (f"  Time: {world_time}/{max_time}", time_color),
            (f"  Resources: {'Abundant' if food_count > 20 else 'Scarce' if food_count < 10 else 'Moderate'}", 
             self.colors['success'] if food_count > 20 else self.colors['danger'] if food_count < 10 else self.colors['warning'])
        ]
        
        for stat_text, stat_color in env_stats:
            stat_surface = self.small_font.render(stat_text, True, stat_color)
            self.screen.blit(stat_surface, (self.ui_x, y_offset))
            y_offset += 16
        
        return y_offset
    
    def _draw_performance_comparison(self, ga1, ga2, analytics, y_start: int) -> int:
        """
        Draw comparative performance analysis.
        
        Args:
            ga1: GA1 population
            ga2: GA2 population
            analytics: Analytics system
            y_start: Starting Y coordinate
            
        Returns:
            Next Y coordinate
        """
        y_offset = y_start
        
        # Section title
        comparison_title = self.font.render("Strategy Performance", True, self.colors['text'])
        self.screen.blit(comparison_title, (self.ui_x, y_offset))
        y_offset += 22
        
        # Calculate comparison metrics
        ga1_alive = len([a for a in ga1.population if a.alive])
        ga2_alive = len([a for a in ga2.population if a.alive])
        
        if ga1_alive > 0 and ga2_alive > 0:
            ga1_avg_fitness = sum(a.fitness for a in ga1.population if a.alive) / ga1_alive
            ga2_avg_fitness = sum(a.fitness for a in ga2.population if a.alive) / ga2_alive
            
            # Determine current leader
            if ga1_avg_fitness > ga2_avg_fitness * 1.1:
                leader = "GA1 Leading"
                leader_color = config.COLORS['AGENT_GA1']
            elif ga2_avg_fitness > ga1_avg_fitness * 1.1:
                leader = "GA2 Leading"
                leader_color = config.COLORS['AGENT_GA2']
            else:
                leader = "Balanced"
                leader_color = self.colors['neutral']
            
            # Survival comparison
            ga1_survival = ga1_alive / config.POPULATION_SIZE_GA1
            ga2_survival = ga2_alive / config.POPULATION_SIZE_GA2
            
            # Ecosystem stability
            total_pop = ga1_alive + ga2_alive
            max_total = config.POPULATION_SIZE_GA1 + config.POPULATION_SIZE_GA2
            stability = total_pop / max_total
            
            stability_color = self.colors['success'] if stability > 0.6 else self.colors['warning'] if stability > 0.3 else self.colors['danger']
            
            comparison_stats = [
                (f"  Leader: {leader}", leader_color),
                (f"  GA1 Survival: {ga1_survival:.1%}", config.COLORS['AGENT_GA1']),
                (f"  GA2 Survival: {ga2_survival:.1%}", config.COLORS['AGENT_GA2']),
                (f"  Ecosystem: {'Stable' if stability > 0.6 else 'Unstable'}", stability_color),
                (f"  Total Pop: {total_pop}/{max_total}", self.colors['text'])
            ]
            
        else:
            # Handle extinction scenarios
            if ga1_alive == 0 and ga2_alive == 0:
                comparison_stats = [
                    ("  Status: TOTAL EXTINCTION", self.colors['danger']),
                    ("  Both populations died", self.colors['danger']),
                    ("  Evolution failed", self.colors['danger'])
                ]
            elif ga1_alive == 0:
                comparison_stats = [
                    ("  Status: GA1 EXTINCT", self.colors['danger']),
                    ("  GA2 Dominance", config.COLORS['AGENT_GA2']),
                    (f"  GA2 Survivors: {ga2_alive}", config.COLORS['AGENT_GA2'])
                ]
            else:  # ga2_alive == 0
                comparison_stats = [
                    ("  Status: GA2 EXTINCT", self.colors['danger']),
                    ("  GA1 Dominance", config.COLORS['AGENT_GA1']),
                    (f"  GA1 Survivors: {ga1_alive}", config.COLORS['AGENT_GA1'])
                ]
        
        for stat_text, stat_color in comparison_stats:
            stat_surface = self.small_font.render(stat_text, True, stat_color)
            self.screen.blit(stat_surface, (self.ui_x, y_offset))
            y_offset += 16
        
        return y_offset
    
    def _draw_enhanced_controls(self, y_start: int):
        """
        Draw enhanced control instructions with current status.
        
        Args:
            y_start: Starting Y coordinate for controls section
        """
        y_offset = y_start
        
        # Controls title
        controls_title = self.font.render("Controls & Status", True, self.colors['text'])
        self.screen.blit(controls_title, (self.ui_x, y_offset))
        y_offset += 25
        
        # Enhanced control instructions with status indicators
        controls = [
            ("SPACE", "Pause/Resume", self.colors['text']),
            ("R", "Reset Simulation", self.colors['warning']),
            ("S", "Save Analytics", self.colors['success']),
            ("A", "Show Graphs", self.colors['success']),
            ("G", "Full Report", self.colors['success']),
            ("ESC", "Exit", self.colors['danger']),
            ("", "", self.colors['text']),  # Spacer
            ("📊", "Visual Indicators:", self.colors['text_dim']),
            ("🔵", "GA1 Cooperative", config.COLORS['AGENT_GA1']),
            ("🔴", "GA2 Aggressive", config.COLORS['AGENT_GA2']),
            ("🟢", "Food Resources", (0, 200, 0)),
            ("⚠️", "Hazard Zones", (255, 100, 100)),
            ("", "", self.colors['text']),  # Spacer
            ("💡", "Agent Traits:", self.colors['text_dim']),
            ("", "Size = Body Size", self.colors['text_dim']),
            ("", "Speed = Yellow Line", self.colors['text_dim']),
            ("", "Sense = Vision Circle", self.colors['text_dim']),
            ("", "Fitness = Brightness", self.colors['text_dim']),
            ("", "", self.colors['text']),  # Spacer
            ("🔬", "Evolution Indicators:", self.colors['text_dim']),
            ("", "+STRONG = Strong Selection", self.colors['danger']),
            ("", "±MOD = Moderate Selection", self.colors['warning']),
            ("", "±WEAK = Weak Selection", self.colors['neutral']),
            ("", "", self.colors['text']),  # Spacer
            ("⚡", "Watch for emergent", self.colors['text_dim']),
            ("", "behaviors, arms races,", self.colors['text_dim']),
            ("", "and trait evolution!", self.colors['text_dim'])
        ]
        
        for key, description, color in controls:
            if key == "" and description == "":  # Empty line for spacing
                y_offset += 8
                continue
            
            if key:
                # Draw key and description
                key_surface = self.small_font.render(f"{key}:", True, color)
                desc_surface = self.small_font.render(f" {description}", True, color)
                self.screen.blit(key_surface, (self.ui_x, y_offset))
                self.screen.blit(desc_surface, (self.ui_x + 35, y_offset))
            else:
                # Just description (indented)
                desc_surface = self.small_font.render(f"   {description}", True, color)
                self.screen.blit(desc_surface, (self.ui_x, y_offset))
            
            y_offset += 14