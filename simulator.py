"""
Main Simulation Coordinator for Artificial World Evolution
=========================================================

This module serves as the central coordinator for the genetic algorithm simulation,
orchestrating all components including world physics, competing populations,
real-time visualization, and comprehensive analytics.

The simulation demonstrates evolutionary competition between two distinct strategies:
- GA1 (Cooperative): Emphasizes efficiency, energy conservation, and peaceful coexistence
- GA2 (Aggressive): Focuses on resource competition, dominance, and territorial behavior

Usage: python simulator.py
"""

import pygame
import sys
from typing import List, Tuple

# Import all simulation components
from config import config, AgentType
from genetics import GeneticAlgorithm
from world import World
from visualization import Visualization
from analytics import Analytics


class Simulation:
    """
    Main simulation coordinator that manages all components.
    
    This class orchestrates:
    - World environment and physics
    - Two competing genetic algorithm populations
    - Real-time visualization and user interaction
    - Data collection and analysis
    - Generation evolution and timing
    
    The Simulation class serves as the central hub that coordinates
    all subsystems to create the complete artificial world experience.
    """
    
    def __init__(self):
        """Initialize all simulation components."""
        print("Initializing Artificial World Simulation...")
        
        # Core simulation components
        self.world = World()                                    # Physical environment
        self.ga1 = GeneticAlgorithm(AgentType.COOPERATIVE)     # Cooperative evolution
        self.ga2 = GeneticAlgorithm(AgentType.AGGRESSIVE)      # Aggressive evolution
        self.analytics = Analytics()                           # Data collection
        self.visualization = Visualization()                   # Real-time display
        
        # Simulation state management
        self.running = True         # Main loop control
        self.paused = False         # Pause/resume functionality
        
        # Initialize both populations at designated spawn points
        # Strategic spawn positioning to avoid initial clustering
        spawn_positions = [
            (50, 50),      # Top-left
            (750, 50),     # Top-right  
            (50, 550),     # Bottom-left
            (750, 550),    # Bottom-right
            (400, 300)     # Center
        ]
        
        self.ga1.initialize_population(spawn_positions)
        self.ga2.initialize_population(spawn_positions)
        
        print("Simulation initialized successfully!")
        print(f"GA1 (Cooperative): {len(self.ga1.population)} agents")
        print(f"GA2 (Aggressive): {len(self.ga2.population)} agents")
        print("Press SPACE to pause, R to reset, S to save data, ESC to exit")
    
    def run(self):
        """
        Execute the main simulation loop.
        
        This is the heart of the simulation that coordinates:
        1. Event handling (user input)
        2. Simulation updates (when not paused)
        3. Visualization rendering
        4. Performance management (frame rate)
        
        The loop continues until the user exits or maximum generations are reached.
        """
        print("Starting simulation main loop...")
        
        try:
            while self.running:
                # Handle user input and window events
                self._handle_events()
                
                # Update simulation state (only when not paused)
                if not self.paused:
                    self._update_simulation()
                
                # Render current state
                self._draw()
                
                # Maintain consistent frame rate
                self.visualization.clock.tick(config.FPS)
                
                # Check if maximum generations reached
                max_gen = max(self.ga1.generation, self.ga2.generation)
                if max_gen >= config.MAX_GENERATIONS:
                    print(f"\nMaximum generations ({config.MAX_GENERATIONS}) reached!")
                    print("Simulation complete. Final analytics:")
                    print(self.analytics.get_summary_report())
                    break
        
        except Exception as e:
            print(f"Simulation error: {e}")
            raise
        finally:
            # Cleanup when simulation ends
            print("Simulation ended. Cleaning up...")
            pygame.quit()
    
    def _handle_events(self):
        """
        Process user input and window events.
        
        Supported interactions:
        - SPACE: Pause/resume simulation
        - R: Reset entire simulation
        - S: Save current data to file
        - ESC: Exit simulation
        - Window close: Exit simulation
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # User closed window
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Toggle pause state
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    print(f"Simulation {status}")
                    
                elif event.key == pygame.K_r:
                    # Reset entire simulation
                    print("Resetting simulation...")
                    self._reset_simulation()
                    
                elif event.key == pygame.K_s:
                    # Save analytics data
                    print("Saving simulation data...")
                    self.analytics.save_data()
                    
                elif event.key == pygame.K_ESCAPE:
                    # Exit simulation
                    self.running = False
    
    def _update_simulation(self):
        """
        Update all simulation components for one time step.
        
        Update sequence:
        1. Update all agents (decision-making and movement)
        2. Apply world physics and interactions
        3. Check for generation completion
        4. Evolve to next generation if needed
        
        This function represents one "frame" of simulation time.
        """
        # Combine both populations for world interactions
        all_agents = self.ga1.population + self.ga2.population
        
        # AGENT UPDATES: Each agent makes decisions and acts
        for agent in all_agents:
            if not agent.alive:
                continue  # Skip dead agents
            
            # Provide agent with local environmental information
            world_state = self.world.get_world_state_for_agent(agent, all_agents)
            
            # Agent decides on action based on genetics and environment
            dx, dy = agent.update(world_state)
            
            # Apply movement with boundary checking
            # Movement speed scaling factor (5 pixels per direction unit)
            movement_speed = 5
            new_x = max(0, min(self.world.width - 20, agent.x + dx * movement_speed))
            new_y = max(0, min(self.world.height - 20, agent.y + dy * movement_speed))
            
            # Calculate distance traveled for analytics
            distance = ((new_x - agent.x)**2 + (new_y - agent.y)**2)**0.5
            agent.distance_traveled += distance
            
            # Update agent position
            agent.x, agent.y = new_x, new_y
        
        # WORLD UPDATES: Handle environmental interactions
        self.world.update(all_agents)
        
        # GENERATION MANAGEMENT: Check if current generation should end
        if self.world.is_generation_complete(all_agents):
            self._evolve_generation()
    
    def _evolve_generation(self):
        """
        Transition from current generation to the next.
        
        Evolution process:
        1. Record current generation statistics
        2. Display summary report
        3. Evolve both populations using genetic operators
        4. Reset world for new generation
        5. Prepare for next evolutionary cycle
        """
        current_gen = max(self.ga1.generation, self.ga2.generation)
        print(f"\n{'='*60}")
        print(f"EVOLVING TO GENERATION {current_gen + 1}")
        print(f"{'='*60}")
        
        # Record comprehensive analytics before evolution
        self.analytics.record_generation(current_gen, self.ga1, self.ga2)
        
        # Display detailed evolution report
        report = self.analytics.get_summary_report()
        print(report)
        
        # Execute genetic algorithm evolution for both populations
        spawn_positions = [
            (50, 50), (750, 50), (50, 550), (750, 550), (400, 300)
        ]
        
        print("Evolving GA1 (Cooperative)...")
        self.ga1.evolve_generation(spawn_positions)
        
        print("Evolving GA2 (Aggressive)...")
        self.ga2.evolve_generation(spawn_positions)
        
        # Reset environment for fresh start
        print("Resetting world environment...")
        self.world.reset_for_new_generation()
        
        print(f"Generation {current_gen + 1} ready!")
        print(f"GA1 Population: {len([a for a in self.ga1.population if a.alive])}")
        print(f"GA2 Population: {len([a for a in self.ga2.population if a.alive])}")
        print(f"{'='*60}\n")
    
    def _reset_simulation(self):
        """
        Completely reset the simulation to initial state.
        
        This creates a fresh simulation with:
        - New random populations
        - Reset world environment  
        - Cleared analytics data
        - Generation counter reset to 0
        
        Useful for comparing different evolutionary runs or
        recovering from problematic simulation states.
        """
        print("Performing complete simulation reset...")
        
        # Reinitialize all components
        self.world = World()
        self.ga1 = GeneticAlgorithm(AgentType.COOPERATIVE)
        self.ga2 = GeneticAlgorithm(AgentType.AGGRESSIVE)
        self.analytics = Analytics()
        
        # Create new populations
        spawn_positions = [(50, 50), (750, 50), (50, 550), (750, 550), (400, 300)]
        self.ga1.initialize_population(spawn_positions)
        self.ga2.initialize_population(spawn_positions)
        
        print("Simulation reset complete!")
        print("New random populations generated.")
    
    def _draw(self):
        """
        Render the current simulation state to the display.
        
        This function coordinates the visualization system to display:
        - The world environment with all objects
        - All agents with their status indicators
        - Real-time statistics and analytics
        - User interface and controls
        """
        self.visualization.draw(self.world, self.ga1, self.ga2, self.analytics)


def main():
    """
    Main entry point for the artificial world simulation.
    
    Provides user instructions and initializes the simulation.
    Handles errors gracefully and provides helpful feedback.
    """
    print("=" * 70)
    print("ARTIFICIAL WORLD SIMULATION - COMPETING GENETIC ALGORITHMS")
    print("=" * 70)
    print()
    print("🧬 EVOLUTIONARY STRATEGIES:")
    print("  • GA1 (Blue Agents): Cooperative - Focus on efficiency & survival")
    print("  • GA2 (Orange Agents): Aggressive - Focus on competition & dominance")
    print()
    print("🎮 CONTROLS:")
    print("  SPACE    - Pause/Resume simulation")
    print("  R        - Reset with new random populations")
    print("  S        - Save analytics data to JSON file")
    print("  ESC      - Exit simulation")
    print()
    print("👀 WHAT TO OBSERVE:")
    print("  • Population survival rates over generations")
    print("  • Emergent behavioral patterns")
    print("  • Resource acquisition strategies")
    print("  • Adaptation to environmental pressures")
    print()
    print("📊 Analytics are collected automatically and can be saved for analysis")
    print("=" * 70)
    
    try:
        # Create and run the simulation
        simulation = Simulation()
        simulation.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user.")
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install pygame numpy")
        
    except Exception as e:
        print(f"\n💥 Simulation error: {e}")
        print("Check that all module files are present and properly configured.")
        
    finally:
        print("\n🙏 Thank you for exploring artificial evolution!")
        print("💾 Data can be found in 'simulation_data.json' if you saved it.")


# Execute the simulation when this file is run directly
if __name__ == "__main__":
    main() 