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

