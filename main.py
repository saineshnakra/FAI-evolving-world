"""
Main entry point for the Artificial World Simulation.

This module provides the entry point and user instructions for running
the genetic algorithm simulation with competing populations.
"""

from simulator import Simulation


def main():
    """
    Main function to initialize and run the artificial world simulation.
    
    This function serves as the entry point for the entire simulation system.
    It provides user instructions and starts the main simulation loop.
    
    The simulation will continue running until the user exits or an error occurs.
    """
    print("=" * 60)
    print("ARTIFICIAL WORLD SIMULATION WITH COMPETING GENETIC ALGORITHMS")
    print("=" * 60)
    print()
    print("This simulation demonstrates evolution in action!")
    print("Two populations compete using different strategies:")
    print("  • GA1 (Blue): Cooperative strategy - rewards efficiency and cooperation")
    print("  • GA2 (Orange): Aggressive strategy - rewards competition and dominance")
    print()
    print("SIMULATION CONTROLS:")
    print("  SPACE BAR - Pause/Resume the simulation")
    print("  R KEY     - Reset simulation with new random populations")
    print("  S KEY     - Save current data and statistics to file")
    print("  ESC KEY   - Exit the simulation")
    print()
    print("WHAT TO WATCH FOR:")
    print("  • Which strategy evolves better survival rates?")
    print("  • Do agents develop emergent behaviors?")
    print("  • How do populations adapt to resource scarcity?")
    print("  • What behavioral patterns emerge over generations?")
    print()
    print("Starting simulation...")
    print("=" * 60)
    
    try:
        # Create and run the simulation
        simulation = Simulation()
        simulation.run()
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        
    except Exception as e:
        print(f"\nSimulation error: {e}")
        print("This might be due to missing dependencies.")
        print("Ensure you have installed: pip install pygame numpy")
        
    finally:
        print("Thank you for exploring artificial evolution!")
        print("Check 'simulation_data.json' for detailed analytics if you saved data.")


# Execute the simulation when this file is run directly
if __name__ == "__main__":
    main() 