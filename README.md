# Elastic Multiple Scattering (emults) Finite-Difference Solver

    Jordan Sheppard (MS Mathematics Student, Brigham Young University)
    July 2025 

## Overview 

This repository contains a Python implementation of a second-order finite-difference solver for multiple-obstacle wave scattering problems in elastodynamics. In particular, this solver is equipped to handle multiple circular obstacles/voids inside an unbounded elastic medium in plane strain or plane stress. 

This code stems from the method I proposed in my Master's thesis, "A Novel Set of Boundary Conditions for Multiple Obstacle Elastic Scattering Problems", which I am currently revising and defending. I will link a copy of the final product once the process of defending my thesis is finished.

In particular, the novel approach taken in my thesis (and in this code) is to take the vector-valued elastodynamic problem and breaking it up into coupled scalar-valued problems, and then implementing absorbing boundary conditions (ABC's) based on Karp's Farfield Expansions for the Helmholtz Equation. These absorbing boundary conditions, applied to this multiple-scattering problem, I have denoted as the MKFE-ABCs.

This solver has been made extendable wherever possible, so that other techniques (such as finite element) or other geometries (such as curves defined by parametric functions) can ultimately be integrated into this solver.

## Package Installation 
This package can be downloaded from PyPi using the following command:
```shell
python3 -m pip install emults
```
Note that this package was coded and tested in Python 3.11.6, so some bugs may occur using different versions of Python. But, in general, Python 3.10 or above should work fine.

Again, these steps will hopefully be replaced with a more professional installation process soon, but here you go for now! 

## Usage 
I hope to streamline this process in the future, and to get some professional-looking documentation, but for now, here are the basic details of how to use this software.

### Importing Functions and Classes 
This software has been assembled as a bunch of components that can be used individually, or together, depending on your use case.

The most useful high-level classes and functions for running experiments have been grouped together in the global namespace. These resources consist of the following: 

* `MKFE_FD_ScatteringProblem`: The main driver class for this software package; uses a second-order finite difference scheme in an iterative manner to solve an elastic multiple-scattering problem. Useful for setting up and running a simulation of an elastic multiple-scattering problem directly a set of basic configuration files.
* `ScatteringConvergenceAnalyzerPolar`: Useful for analyzing convergence of various approximations to a reference solution at various points-per-wavelength (PPW) values (or, lacking a reference solution, for approximating the order of convergence using Richardson extrapolation).
* `Algorithm`: An enum for stating which iterative algorithm the solver should use to approximate solutions (the only current accepted value is `Algorithm.GAUSS_SEIDEL`).
* `ComplexArrayQuantity`: An enum for stating which quantity you would like to observe given an array of complex numbers. Current options are the entrywise real parts (`ComplexArrayQuantity.REAL`), entrywise imaginary parts (`ComplexArrayQuantity.IMAG`), or entrywise complex moduli (`ComplexArrayQuantity.ABS`).
* `get_full_configuration_filename_base`: A utility function that is helpful for getting a unique (if lengthy) filename for a given scattering problem. This can help with organizing experimental results, such as for naming subfolders, log files, plots, etc. 

To use any of these high-level assets, the easiest way to use them would be to call them directly as `emults.<function-or-class-name>`. For example, you could create a multiple scattering problem finite-difference solver in two easy lines:
```python 
import emults 

solver = emults.MKFE_FD_ScatteringProblem(...)
```
In the previous example, the ellipses would obviously be replaced with the corresponding arguments to the class constructor; see below for more details on usage of the solver itself.

To import any other lower-level functions or classes from this package (such as software for creating and examining a single obstacle, etc.), you must use a fully qualified path to the asset you'd like to import. 

For example, to import the class for examining a single circular obstacle in a finite-difference scheme (denoted as `Circular_MKFE_FDObstacle`, located in the folder `src/emults/fd` in this repository, under the file `obstacles.py`), you would run this command at the top of your Python script or notebook:
```python 
from emults.fd.obstacles import Circular_MKFE_FDObstacle

obstacle = Circular_MKFE_FDObstacle(...)
```

### Running a Basic Multiple-Scattering Simulation
Below is a bare-bones script template you can use to get up and running quickly to run simulations:
```python 
import emults 

# Required configuration files
obstacle_config_file = 'obstacles.json'
medium_config_file = 'medium.json'
numerical_config_file = 'numerical.json'

# Optional configuration files for reference solution
# NOTE: If provided, the code will first compute a 
# reference solution and cache it (or, if this specific
# reference solution has been computed before, it will
# load it from the cache). Then, convergence analysis
# will be based on this reference solution.
# NOTE: If not provided, this code will not compute
# a reference solution, and will use Richardson extrapolation
# to approximate the error order (needs at least 3 PPW 
# values in the numerical config file for this method
# to run)
reference_numerical_config_file = 'reference.json'  

### Step 1: Create a finite-difference solver for this simulation 
solver = emults.MKFE_FD_ScatteringProblem(
   obstacle_config_file,
   medium_config_file,
   numerical_config_file,
   reference_numerical_config_file  
)

### Step 2: Run simulation at the PPW values provided by the algorithm 
solver.solve_PPWs(algorithm=emults.Algorithm.GAUSS_SEIDEL)     

### Step 3: Analyze convergence
solved_obstacles = solver.obstacles 
reference_obstacles = solver.reference_obstacles 
convergence_analyzer = emults.ScatteringConvergenceAnalyzerPolar(
   num_obstacles=len(reference_obstacles)
)

# If a reference solution is provided, use this code to approximate
# the convergence rate of the algorithm
convergence_analyzer.analyze_convergence_reference(
   solved_obstacles=solved_obstacles,
   reference_solution=reference_obstacles
)

# If not, use this code instead:
convergence_analyzer.analyze_convergence_richardson_extrapolation(
   solved_obstacles=solved_obstacles
)

# In either case, you can use the following function to display
# the convergence rate analysis (can display as a table, as 
# a log-log-plot, or both. Other formatting options for the table
# are also available)
convergence_analyzer.display_convergence(
   text=True,
   plots=True
)

### Step 4: Examine any desired plots of the quantities of interest
### in the 2D-plane with the obstacles inside it
PPW = 50    # NOTE: This should be a PPW value from numerical_config_file.
plotter = emults.ScatteringProblemPlotter.from_scattering_problem_and_ppw(
   solved_problem,
   PPW
)

# Here are some example plot commands:
plotter.plot_total_phi(emults.ComplexArrayQuantity.ABS)
plotter.plot_scattered_psi(emults.ComplexArrayQuantity.REAL)
plotter.plot_total_x_displacement(emults.ComplexArrayQuantity.ABS)
plotter.plot_scattered_stress_xx(emults.ComplexArrayQuantity.IMAG)
```

### JSON Configuration Files
Notice that there are 3 required JSON configuration files, and 1 optional one. These each contain vital information about the obstacles themselves, about the elastic medium of interest, and about the numerical method used to approximate solutions to this problem. 

The formats for these files are as follows:

1. Obstacle Configuration File (`'obstacles.json'` from above)
   ```json
   {
      "obstacles": {
         "circular": [
               {
                  "center": [0.0, 0.0],           # Coords. of center of circular obstacle
                  "r_obstacle": 1.0,              # Radius of obstacle from center
                  "r_artificial_boundary": 2.0,   # Radius of artificial boundary from center
                  "bc": "HARD"                    # Obstacle BC ("SOFT" or "HARD")
               },
               {
                  "center": [4.2, 0.0],
                  "r_obstacle": 1.0,
                  "r_artificial_boundary": 2.0,
                  "bc": "HARD"
               }
         ]
      }
   }
   ```

2. Elastic Medium Configuration File (`'medium.json'` from above)
   ```json
   {
      "medium": {     
         "nu": 0.3333333333333333,        # Must specify frequency (omega)
         "E": 1,                          # and 3 elastic constants.
         "lambda_s": 0.75,                # The code takes care of the rest
         "omega": 1                       # of the degrees of freedom
      }, 
      "incident_wave": {
         "angle_of_inc": 0,               # Angle of incidence (Only 0 currently supported)
         "wavenumber": 4.1887902047863905 # Should be k_p 
      }
   }
   ```

3. Numerical Method Configuration File (`'numerical.json'` from above)
   ```json
   {
      "PPWs": [20, 30, 40, 50],        # List of PPW values (int's)
      "num_farfield_terms": 15,        # Number of terms in truncated Karp expansion (int)
      "tol": 1e-5,                     # Error tolerance of iterative method
      "maxiter": 50                    # Maximum number of iterations of iterative method
   }
   ```

4. Reference Solution Configuration File (`'reference.json'` from above)
   ```json
   {
      "PPWs": [200],                   # List of SINGLE PPW VALUE (int)
      "num_farfield_terms": 30,        # Number of terms in truncated Karp expansion (int)
      "tol": 1e-4,                     # Error tolerance of iterative method 
      "maxiter": 50                    # Maximum number of iterations of iterative method
   }
   ```
   Notice that this is essentially identical to the numerical configuration file, but with only a single PPW value (much higher than other PPWs used in experiments/simulation).











