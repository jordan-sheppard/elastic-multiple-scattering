import numpy as np
from typing import Any
from scipy.sparse.linalg import SuperLU
from scipy.sparse import sparray
from enum import Enum

## ---------- Constants ----------
# Number of spatial dimensions
NUM_SPATIAL_DIMS = 2        

# Number of angular gridpoints on each local polar grid
NUM_ANGULAR_GRIDPOINTS = 50 

## ---------- Custom types ----------
Coordinates = tuple[float, float]
Grid = np.ndarray[(int, int), float]
Vector = np.ndarray[(int,), float]
Matrix = np.ndarray[(int, int), float]
SparseMatrix = sparray
SparseMatrixLUDecomp = Any

## ---------- Enums ----------
class BoundaryCondition(Enum):
    """Various boundary conditions that can be used at physical
    boundaries
    """
    HARD = 1 
    SOFT = 2

class Algorithm(Enum):
    """Various boundary conditions that can be used at physical
    boundaries
    """
    GAUSS_SEIDEL = 1

class QOI(Enum):
    """Quantities of interest"""
    DISPLACEMENT = 1 
    STRESS = 2
    POTENTIALS = 3

class CoordinateSystem(Enum):
    """Coordinate systems used for finding vector/tensor quantities."""
    LOCAL_POLAR = 1 
    LOCAL_CARTESIAN = 2
    GLOBAL_POLAR = 3
    GLOBAL_CARTESIAN = 4

class ErrorType(Enum):
    L2 = 1
    L2_RELATIVE = 2 
    LINFTY = 3

class ScalarQOI(Enum):
    PHI = 1
    PSI = 2 
    DISPLACEMENT_X = 3
    DISPLACEMENT_Y = 4
    STRESS_XX = 5
    STRESS_XY = 6
    STRESS_YY = 7
    


class ComplexArrayQuantity(Enum):
    ABS = 1 
    REAL = 2 
    IMAGINARY = 3

class PlotType(Enum):
    SCATTERED = 1 
    TOTAL = 2 

class StressType(Enum):
    RADIAL = 1 
    SHEAR = 2 
    HOOP = 3 
    XX = 4 
    XY = 5 
    YY = 6

