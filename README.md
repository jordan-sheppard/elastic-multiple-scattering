# Elastic Multiple Scattering (EMS) Finite-Difference Solver

    Jordan Sheppard (MS Mathematics Student, Brigham Young University)
    July 2025 

## Overview 

This repository contains a Python implementation of a second-order finite-difference solver for multiple-obstacle wave scattering problems in elastodynamics. In particular, this solver is equipped to handle multiple circular obstacles/voids inside an unbounded elastic medium in plane strain or plane stress. 

This code stems from the method I proposed in my Master's thesis, "A Novel Set of Boundary Conditions for Multiple Obstacle Elastic Scattering Problems", which I am currently revising and defending. I will link a copy of the final product once the process of defending my thesis is finished.

In particular, the novel approach taken in my thesis (and in this code) is to take the vector-valued elastodynamic problem and breaking it up into coupled scalar-valued problems, and then implementing absorbing boundary conditions (ABC's) based on Karp's Farfield Expansions for the Helmholtz Equation. These absorbing boundary conditions, applied to this multiple-scattering problem, I have denoted as the MKFE-ABCs.

This solver has been made extendable wherever possible, so that other techniques (such as finite element) or other geometries (such as curves defined by parametric functions) can ultimately be integrated into this solver.

## Package Installation 
This Python code is not currently on PyPi or any other package manager. I hope to publish it there soon. In the meantime, to install this package, follow these steps:


1. Clone this repository onto your machine in a convenient folder location
2. Create a new virtual environment named `venv` in whatever folder you which:
   ```shell
   python -m venv ./venv
   ```
3. In the same folder that you created it, activate that    virtual environment:
   ```shell
   source venv/bin/activate
   ```
4. In your terminal, navigate into the root folder of this repository (which you cloned in step 1).
5. Once there, run the following command:
   ```shell
   pip install .
   ```
   This will install this package into your virtual environment.


Again, these steps will hopefully be replaced with a more professional installation process soon, but here you go for now! 

## Usage 
I hope to streamline this process in the future, and to get some professional-looking documentation, but for now, here are the basic details of how to use this software.

### Importing Functions and Classes 
This software has been assembled as a bunch of components that can be used individually, or together, depending on your use case.

To import assets from this package (such as the finite-difference solver, Obstacle files, plotting software, etc.) into a Python file for scriptable use, you must use a fully qualified path to the asset you'd like to import. 

For example, to import the high-level solver (denoted as `MKFE_FD_ScatteringProblem`, located in the folder `ems/fd` in this repository, under the file `algorithm.py`), you would run this command at the top of your Python script or notebook:
```python 
from ems.fd.algorithm import MKFE_FD_Scattering problem
```

Again, 




