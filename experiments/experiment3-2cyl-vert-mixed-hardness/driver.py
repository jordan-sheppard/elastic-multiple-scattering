import emults

import os
import logging
import cloudpickle
from matplotlib import pyplot as plt

# Standardized names for config files/folders and log files
OBSTACLE_CONFIG_FILENAME = 'obstacles.json'
MEDIUM_CONFIG_FILENAME = 'medium.json'
NUMERICAL_CONFIG_FILENAME = 'numerical.json'
REFERENCE_CONFIG_FILENAME = 'reference.json'
SIMULATION_CACHE_FOLDERNAME = 'cache/simulation'
REFERENCE_OBSTACLE_CACHE_FOLDERNAME = 'cache/obstacles/reference'
OTHER_OBSTACLE_CACHE_FOLDERNAME = 'cache/obstacles/other'
LOG_FILENAME = 'runtime_log.txt'

# Plot parameters
plt.rcParams['figure.figsize'] = [10.0/1.3, 7.0/1.3]
plt.rcParams['figure.dpi'] = 300

# Redirect logging to console (for use in notebook file)
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)

# Utility functions 
def get_cached_simulation():
    """Gets a cached simulation from the cache/simulation folder
    if it already exists to save runtime"""
    logging.info("Checking for cached simulation files . . .")
    obstacle_base_filename = emults.get_full_configuration_filename_base(
        OBSTACLE_CONFIG_FILENAME,
        MEDIUM_CONFIG_FILENAME,
        NUMERICAL_CONFIG_FILENAME,
        REFERENCE_CONFIG_FILENAME
    )
    potential_cached_file = os.path.join(
        "cache",
        "simulation",
        f"{obstacle_base_filename}.pickle"
    ) 
    
    if os.path.exists(potential_cached_file):
        logging.info("Cached simulation files found. Returning cached simulation.")
        with open(potential_cached_file, 'rb') as infile:
            cached_results = cloudpickle.load(infile)
        return cached_results
    else:
        logging.info("Cached simulation files not found. Running simulation . . .")
        return None
    
def run_experiment():
    """Runs and saves a multiple-scattering experiment."""
    # Try to retrieve simulation caches, if possible.
    cached_simulation = get_cached_simulation()
    if cached_simulation is not None:
        return cached_simulation

    # Load and run simulation (this may take a while
    # the first time since the reference solution 
    # is likely being computed then)
    simulation = emults.MKFE_FD_ScatteringProblem(
        obstacle_config_file=OBSTACLE_CONFIG_FILENAME,
        medium_config_file=MEDIUM_CONFIG_FILENAME,
        numerical_config_file=NUMERICAL_CONFIG_FILENAME,
        reference_config_file=REFERENCE_CONFIG_FILENAME,
        reference_cache_folder=REFERENCE_OBSTACLE_CACHE_FOLDERNAME,
        normal_cache_folder=OTHER_OBSTACLE_CACHE_FOLDERNAME
    )
    simulation.solve_PPWs(
        algorithm=emults.Algorithm.GAUSS_SEIDEL,
        pickle=True,
        reference=False,
        pickle_folder=SIMULATION_CACHE_FOLDERNAME, 
        cache_all=True
    )
    return simulation


# Driver code 
if __name__ == "__main__":
    # Run experiment 
    simulation = run_experiment()

    # Display/save convergence plots 
    simulation.convergence_analyzer.display_convergence(text=True, plots=True, text_filepath='convergence_tables.txt', plots_folderpath='figures')

    # Create plotter for highest PPW solution
    highest_PPW = max(simulation.obstacles.keys())
    plotter = emults.ScatteringProblemPlotter.from_scattering_problem_and_ppw(simulation, highest_PPW)

    # Plot various quantities of interest about the obstacles
    plotter.plot_scattered_phi(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_scattered_phi(emults.ComplexArrayQuantity.ABS, 'figures', exterior=True)
    plotter.plot_scattered_psi(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_scattered_psi(emults.ComplexArrayQuantity.ABS, 'figures', exterior=True)
    plotter.plot_total_phi(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_total_phi(emults.ComplexArrayQuantity.ABS, 'figures', exterior=True)
    plotter.plot_total_psi(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_total_psi(emults.ComplexArrayQuantity.ABS, 'figures', exterior=True)
    plotter.plot_scattered_x_displacement(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_scattered_y_displacement(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_total_x_displacement(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_total_y_displacement(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_scattered_stress_xx(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_scattered_stress_xy(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_scattered_stress_yy(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_total_stress_xx(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_total_stress_xy(emults.ComplexArrayQuantity.ABS, 'figures')
    plotter.plot_total_stress_yy(emults.ComplexArrayQuantity.ABS, 'figures')