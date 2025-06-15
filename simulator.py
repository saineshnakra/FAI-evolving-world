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
import random

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
    - Comprehensive data collection and analysis
    - Generation evolution and timing
    - Integration between all subsystems
    
    The Simulation class serves as the central hub that coordinates
    all subsystems to create the complete artificial world experience.
    """
    
    def __init__(self):
        """Initialize all simulation components with proper integration."""
        print("Initializing Artificial World Simulation...")
        
        # Core simulation components
        self.world = World()                                    # Physical environment
        self.ga1 = GeneticAlgorithm(AgentType.COOPERATIVE)     # Cooperative evolution
        self.ga2 = GeneticAlgorithm(AgentType.AGGRESSIVE)      # Aggressive evolution
        self.analytics = Analytics()                           # Enhanced data collection
        self.visualization = Visualization()                   # Real-time display with analytics
        
        # Simulation state management
        self.running = True         # Main loop control
        self.paused = False         # Pause/resume functionality
        
        # Performance tracking
        self.frame_count = 0
        self.last_analytics_update = 0
        
        # Initialize populations with safe spawn positions
        print("Generating safe spawn positions...")
        spawn_positions = self.world.get_spawn_positions(
            config.POPULATION_SIZE_GA1 + config.POPULATION_SIZE_GA2
        )
        
        print("Initializing populations...")
        self.ga1.initialize_population(spawn_positions)
        self.ga2.initialize_population(spawn_positions)
        
        # Record initial state
        self.analytics.record_generation(0, self.ga1, self.ga2, self.world)
        
        print("Simulation initialized successfully!")
        print(f"GA1 (Cooperative): {config.POPULATION_SIZE_GA1} agents")
        print(f"GA2 (Aggressive): {config.POPULATION_SIZE_GA2} agents")
        print("Enhanced analytics and visualization enabled!")
        print("Press SPACE to pause, R to reset, S to save data, A for graphs, ESC to exit")
    
    def run(self):
        """
        Execute the main simulation loop with enhanced integration.
        
        This is the heart of the simulation that coordinates:
        1. Event handling (user input)
        2. Simulation updates (when not paused)
        3. Real-time analytics collection
        4. Enhanced visualization rendering
        5. Performance management (frame rate)
        
        The loop continues until the user exits or maximum generations are reached.
        """
        print("Starting enhanced simulation main loop...")
        
        try:
            while self.running:
                # Handle user input and window events
                self._handle_events()
                
                # Update simulation state (only when not paused)
                if not self.paused:
                    self._update_simulation()
                    
                    # Collect frame-by-frame analytics
                    self.analytics.record_frame_data(self.world, self.ga1, self.ga2)
                    self.frame_count += 1
                
                # Render current state with enhanced analytics
                self._draw()
                
                # Maintain consistent frame rate
                self.visualization.clock.tick(config.FPS)
                
                # Check termination conditions
                if self._should_terminate():
                    break
        
        except Exception as e:
            print(f"Simulation error: {e}")
            self._emergency_save()
            raise
        finally:
            # Cleanup when simulation ends
            print("Simulation ended. Performing cleanup...")
            self._final_cleanup()
    
    def _should_terminate(self) -> bool:
        """
        Check if simulation should terminate.
        
        Returns:
            True if simulation should end
        """
        max_gen = max(self.ga1.generation, self.ga2.generation)
        
        # Check maximum generations
        if max_gen >= config.MAX_GENERATIONS:
            print(f"\nMaximum generations ({config.MAX_GENERATIONS}) reached!")
            print("Simulation complete. Generating final analytics...")
            final_report = self.analytics.generate_evolution_report()
            print(final_report)
            
            # Auto-save final data
            self.analytics.save_data(f"final_simulation_data_gen_{max_gen}.json")
            print("Final data automatically saved!")
            return True
        
        # Check for total extinction
        total_alive = len([a for a in self.ga1.population + self.ga2.population if a.alive])
        if total_alive == 0:
            print("\n💀 TOTAL POPULATION EXTINCTION!")
            print("All agents have died. Evolution failed.")
            print("This indicates the environment may be too harsh or genetic diversity was lost.")
            
            extinction_report = self.analytics.generate_evolution_report()
            print(extinction_report)
            
            # Save extinction data for analysis
            self.analytics.save_data(f"extinction_data_gen_{max_gen}.json")
            return True
        
        return False
    
    def _handle_events(self):
        """
        Process user input and window events with enhanced functionality.
        
        Supported interactions:
        - SPACE: Pause/resume simulation
        - R: Reset entire simulation
        - S: Save current data to file
        - A: Generate and display analytics graphs
        - G: Generate comprehensive evolution report
        - ESC: Exit simulation
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # User closed window
                print("Window closed by user. Saving data...")
                self.analytics.save_data("emergency_save.json")
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
                    current_gen = max(self.ga1.generation, self.ga2.generation)
                    filename = f"simulation_data_gen_{current_gen}.json"
                    print(f"Saving simulation data to {filename}...")
                    self.analytics.save_data(filename)
                    print("✅ Data saved successfully!")
                    
                elif event.key == pygame.K_a:
                    # Generate analytics graphs
                    print("Generating comprehensive evolution analytics graphs...")
                    try:
                        current_gen = max(self.ga1.generation, self.ga2.generation)
                        graph_filename = f"evolution_graphs_gen_{current_gen}.png"
                        self.analytics.create_evolution_graphs(graph_filename)
                        print("📊 Analytics graphs generated successfully!")
                    except Exception as e:
                        print(f"Error generating graphs: {e}")
                        print("Make sure matplotlib is installed: pip install matplotlib")
                    
                elif event.key == pygame.K_g:
                    # Generate comprehensive evolution report
                    print("=" * 60)
                    print("GENERATING COMPREHENSIVE EVOLUTION REPORT")
                    print("=" * 60)
                    report = self.analytics.generate_evolution_report()
                    print(report)
                    
                    # Also save report to file
                    current_gen = max(self.ga1.generation, self.ga2.generation)
                    report_filename = f"evolution_report_gen_{current_gen}.txt"
                    try:
                        with open(report_filename, 'w') as f:
                            f.write(report)
                        print(f"📄 Report saved to {report_filename}")
                    except Exception as e:
                        print(f"Could not save report to file: {e}")
                    
                elif event.key == pygame.K_ESCAPE:
                    # Exit simulation with final save
                    print("Exiting simulation. Performing final save...")
                    current_gen = max(self.ga1.generation, self.ga2.generation)
                    self.analytics.save_data(f"final_save_gen_{current_gen}.json")
                    print("Final save complete!")
                    self.running = False
                    
                elif event.key == pygame.K_d:
                    # DEBUG: Print detailed current state
                    self._debug_print_state()
    
    def _debug_print_state(self):
        """Print detailed debug information about current simulation state."""
        print("\n" + "="*50)
        print("DEBUG: CURRENT SIMULATION STATE")
        print("="*50)
        
        ga1_alive = [a for a in self.ga1.population if a.alive]
        ga2_alive = [a for a in self.ga2.population if a.alive]
        
        print(f"Frame: {self.frame_count}")
        print(f"World Time: {self.world.generation_time}")
        print(f"GA1 Population: {len(ga1_alive)}/{len(self.ga1.population)}")
        print(f"GA2 Population: {len(ga2_alive)}/{len(self.ga2.population)}")
        print(f"Food Items: {len(self.world.food)}")
        print(f"Hazards: {len(self.world.hazards)}")
        
        if ga1_alive:
            avg_fitness_ga1 = sum(a.fitness for a in ga1_alive) / len(ga1_alive)
            avg_energy_ga1 = sum(a.energy for a in ga1_alive) / len(ga1_alive)
            print(f"GA1 Avg Fitness: {avg_fitness_ga1:.2f}, Avg Energy: {avg_energy_ga1:.1f}")
        
        if ga2_alive:
            avg_fitness_ga2 = sum(a.fitness for a in ga2_alive) / len(ga2_alive)
            avg_energy_ga2 = sum(a.energy for a in ga2_alive) / len(ga2_alive)
            total_attacks = sum(a.attacks_made for a in ga2_alive)
            print(f"GA2 Avg Fitness: {avg_fitness_ga2:.2f}, Avg Energy: {avg_energy_ga2:.1f}")
            print(f"GA2 Total Attacks: {total_attacks}")
        
        print("="*50 + "\n")
    
    def _update_simulation(self):
        """
        Update all simulation components for one time step with enhanced integration.
        
        Update sequence:
        1. Update all agents (decision-making and actions)
        2. Apply agent movements with genetic speed scaling
        3. Handle world physics and interactions
        4. Process agent-agent combat
        5. Check for generation completion
        6. Evolve to next generation if needed
        
        This function represents one "frame" of simulation time.
        """
        # Combine both populations for world interactions
        all_agents = self.ga1.population + self.ga2.population
        
        # PHASE 1: AGENT DECISION MAKING AND MOVEMENT
        for agent in all_agents:
            if not agent.alive:
                continue
            
            # Provide agent with environmental information based on genetic vision
            world_state = self.world.get_world_state_for_agent(agent, all_agents)
            
            # Agent decides on action based on genetics and environment
            dx, dy = agent.update(world_state)
            
            # Apply movement using the world's movement system (respects genetic speed)
            self.world.apply_agent_movement(agent, dx, dy)
        
        # PHASE 2: WORLD PHYSICS AND INTERACTIONS
        self.world.update(all_agents)
        
        # PHASE 3: GENERATION MANAGEMENT
        if self.world.is_generation_complete(all_agents):
            self._evolve_generation()
    
    def _evolve_generation(self):
        """
        Transition from current generation to the next with comprehensive analytics.
        
        Evolution process:
        1. Record comprehensive generation statistics
        2. Display detailed evolution report
        3. Evolve both populations using genetic operators
        4. Reset world for new generation
        5. Update analytics and prepare for next cycle
        """
        current_gen = max(self.ga1.generation, self.ga2.generation)
        print(f"\n{'='*70}")
        print(f"🧬 EVOLVING TO GENERATION {current_gen + 1}")
        print(f"{'='*70}")
        
        # Record comprehensive analytics before evolution
        self.analytics.record_generation(current_gen, self.ga1, self.ga2, self.world)
        
        # Display current generation summary
        summary = self.analytics.get_summary_report()
        print(summary)
        
        # Check for concerning trends
        self._analyze_evolution_health()
        
        # Generate safe spawn positions for new generation
        spawn_positions = self.world.get_spawn_positions(
            config.POPULATION_SIZE_GA1 + config.POPULATION_SIZE_GA2
        )
        
        # Execute genetic algorithm evolution for both populations
        print("🔄 Evolving GA1 (Cooperative)...")
        self.ga1.evolve_generation(spawn_positions)
        
        print("🔄 Evolving GA2 (Aggressive)...")
        self.ga2.evolve_generation(spawn_positions)
        
        # Reset environment for fresh start
        print("🌍 Resetting world environment...")
        self.world.reset_for_new_generation()
        
        # Display post-evolution status
        ga1_alive = len([a for a in self.ga1.population if a.alive])
        ga2_alive = len([a for a in self.ga2.population if a.alive])
        
        print(f"✅ Generation {current_gen + 1} ready!")
        print(f"📊 GA1 Population: {ga1_alive}/{config.POPULATION_SIZE_GA1}")
        print(f"📊 GA2 Population: {ga2_alive}/{config.POPULATION_SIZE_GA2}")
        
        # Display genetic diversity status
        if ga1_alive > 0:
            ga1_avg_traits = {
                'speed': sum(a.genome.speed for a in self.ga1.population) / len(self.ga1.population),
                'sense': sum(a.genome.sense for a in self.ga1.population) / len(self.ga1.population),
                'size': sum(a.genome.size for a in self.ga1.population) / len(self.ga1.population)
            }
            print(f"🧬 GA1 Traits: Speed={ga1_avg_traits['speed']:.3f}, Sense={ga1_avg_traits['sense']:.3f}, Size={ga1_avg_traits['size']:.3f}")
        
        if ga2_alive > 0:
            ga2_avg_traits = {
                'speed': sum(a.genome.speed for a in self.ga2.population) / len(self.ga2.population),
                'sense': sum(a.genome.sense for a in self.ga2.population) / len(self.ga2.population),
                'size': sum(a.genome.size for a in self.ga2.population) / len(self.ga2.population)
            }
            print(f"🧬 GA2 Traits: Speed={ga2_avg_traits['speed']:.3f}, Sense={ga2_avg_traits['sense']:.3f}, Size={ga2_avg_traits['size']:.3f}")
        
        print(f"{'='*70}\n")
        
        # Reset frame counter
        self.frame_count = 0
    
    def _analyze_evolution_health(self):
        """
        Analyze the health of the evolutionary process and warn about issues.
        """
        ga1_alive = len([a for a in self.ga1.population if a.alive])
        ga2_alive = len([a for a in self.ga2.population if a.alive])
        
        # Check for population bottlenecks
        if ga1_alive < config.POPULATION_SIZE_GA1 * 0.1:
            print("⚠️ WARNING: GA1 population bottleneck detected!")
        if ga2_alive < config.POPULATION_SIZE_GA2 * 0.1:
            print("⚠️ WARNING: GA2 population bottleneck detected!")
        
        # Check for complete dominance
        if ga1_alive > 0 and ga2_alive == 0:
            print("🏆 GA1 has achieved complete dominance!")
        elif ga2_alive > 0 and ga1_alive == 0:
            print("🏆 GA2 has achieved complete dominance!")
        
        # Check for balanced coexistence
        if ga1_alive > 0 and ga2_alive > 0:
            ratio = ga2_alive / ga1_alive
            if 0.5 <= ratio <= 2.0:
                print("⚖️ Balanced coexistence detected!")
        
        # Analyze trait diversity
        if ga1_alive > 1:
            speed_diversity = len(set(round(a.genome.speed, 2) for a in self.ga1.population if a.alive))
            if speed_diversity < 3:
                print("⚠️ WARNING: Low genetic diversity in GA1!")
        
        if ga2_alive > 1:
            speed_diversity = len(set(round(a.genome.speed, 2) for a in self.ga2.population if a.alive))
            if speed_diversity < 3:
                print("⚠️ WARNING: Low genetic diversity in GA2!")
    
    def _reset_simulation(self):
        """
        Completely reset the simulation to initial state with enhanced cleanup.
        
        This creates a fresh simulation with:
        - New random populations with genetic diversity
        - Reset world environment  
        - Cleared analytics data but preserved for comparison
        - Generation counter reset to 0
        """
        print("🔄 Performing complete simulation reset...")
        
        # Save current run data before reset
        current_gen = max(self.ga1.generation, self.ga2.generation)
        if current_gen > 0:
            save_name = f"pre_reset_data_gen_{current_gen}.json"
            self.analytics.save_data(save_name)
            print(f"📁 Previous run data saved as {save_name}")
        
        # Reinitialize all components
        self.world = World()
        self.ga1 = GeneticAlgorithm(AgentType.COOPERATIVE)
        self.ga2 = GeneticAlgorithm(AgentType.AGGRESSIVE)
        self.analytics = Analytics()  # Fresh analytics
        
        # Reset counters
        self.frame_count = 0
        self.last_analytics_update = 0
        
        # Create new populations with safe spawning
        spawn_positions = self.world.get_spawn_positions(
            config.POPULATION_SIZE_GA1 + config.POPULATION_SIZE_GA2
        )
        
        self.ga1.initialize_population(spawn_positions)
        self.ga2.initialize_population(spawn_positions)
        
        # Record initial state
        self.analytics.record_generation(0, self.ga1, self.ga2, self.world)
        
        print("✅ Simulation reset complete!")
        print("🧬 New random populations generated with genetic diversity.")
        print("📊 Fresh analytics system initialized.")
        print("🌍 World environment restored to initial state.")
    
    def _emergency_save(self):
        """Save data in case of unexpected errors."""
        try:
            current_gen = max(self.ga1.generation, self.ga2.generation)
            emergency_filename = f"emergency_save_gen_{current_gen}.json"
            self.analytics.save_data(emergency_filename)
            print(f"🚨 Emergency save completed: {emergency_filename}")
        except Exception as e:
            print(f"❌ Emergency save failed: {e}")
    
    def _final_cleanup(self):
        """Perform final cleanup and summary."""
        try:
            # Final analytics save
            current_gen = max(self.ga1.generation, self.ga2.generation)
            if current_gen > 0:
                final_filename = f"final_session_data_gen_{current_gen}.json"
                self.analytics.save_data(final_filename)
                print(f"📁 Final session data saved: {final_filename}")
            
            # Display final summary
            if self.analytics.generation_data:
                print("\n📈 FINAL SESSION SUMMARY:")
                summary = self.analytics.get_summary_report()
                print(summary)
        except Exception as e:
            print(f"Warning: Cleanup error: {e}")
        finally:
            pygame.quit()
    
    def _draw(self):
        """
        Render the current simulation state with enhanced analytics visualization.
        
        This function coordinates the enhanced visualization system to display:
        - The world environment with all objects and agents
        - Real-time genetic trait visualization
        - Comprehensive statistics and analytics
        - Evolution progress indicators
        - User interface with enhanced controls
        """
        self.visualization.draw(self.world, self.ga1, self.ga2, self.analytics)


def main():
    """
    Main entry point for the artificial world simulation with enhanced features.
    
    Provides comprehensive user instructions and initializes the enhanced simulation.
    Handles errors gracefully and provides helpful feedback.
    """
    print("=" * 80)
    print("🧬 ARTIFICIAL WORLD SIMULATION - ENHANCED GENETIC ALGORITHMS")
    print("=" * 80)
    print()
    print("🎯 EVOLUTIONARY STRATEGIES:")
    print("  • GA1 (Blue Agents): Cooperative - Efficiency, foraging, energy conservation")
    print("  • GA2 (Red Agents): Aggressive - Hunting, competition, territorial dominance")
    print()
    print("🧬 GENETIC TRAITS:")
    print("  • Speed: Movement velocity (yellow indicators)")
    print("  • Sense: Vision range (gray circles for high-sense agents)")
    print("  • Size: Body size affects combat, energy cost, and movement")
    print()
    print("🎮 ENHANCED CONTROLS:")
    print("  SPACE    - Pause/Resume simulation")
    print("  R        - Reset with new random populations")
    print("  S        - Save comprehensive analytics data")
    print("  A        - Generate evolution graphs (requires matplotlib)")
    print("  G        - Generate detailed evolution report")
    print("  D        - Debug: Print current state")
    print("  ESC      - Exit with final save")
    print()
    print("👀 WHAT TO OBSERVE:")
    print("  • Real-time trait evolution and selection pressure")
    print("  • Population dynamics and survival strategies")
    print("  • Emergent behaviors and adaptation patterns")
    print("  • Arms races between competing strategies")
    print("  • Resource competition and ecosystem balance")
    print()
    print("📊 ENHANCED ANALYTICS:")
    print("  • Real-time trait-fitness correlations")
    print("  • Selection pressure indicators")
    print("  • Population dynamics analysis")
    print("  • Energy flow through ecosystem")
    print("  • Comprehensive evolution reports")
    print()
    print("💾 Data is automatically saved at key moments and can be exported for analysis")
    print("=" * 80)
    
    try:
        # Create and run the enhanced simulation
        simulation = Simulation()
        simulation.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user (Ctrl+C).")
        print("💾 Data may have been saved during the session.")
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("📦 Please install required packages:")
        print("   pip install pygame numpy matplotlib")
        print("   (matplotlib is optional for graphs)")
        
    except Exception as e:
        print(f"\n💥 Simulation error: {e}")
        print("🔧 Check that all module files are present and properly configured.")
        print("📁 Check if emergency save files were created.")
        
    finally:
        print("\n🙏 Thank you for exploring artificial evolution!")
        print("🔬 Analyze saved data files for deeper insights into evolution.")
        print("📊 Use saved graphs and reports for research and presentation.")


# Execute the enhanced simulation when this file is run directly
if __name__ == "__main__":
    main()