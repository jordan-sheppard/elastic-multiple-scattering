from abc import abstractmethod
from scipy.sparse import linalg as spla 
import numpy as np
import logging

from ..base.consts import SparseMatrix


class FDSolver:
    @abstractmethod 
    def solve(self, F: np.ndarray) -> np.ndarray:
        pass 

class FD_SparseLUSolver(FDSolver):
    def __init__(self, fd_matrix: SparseMatrix) -> None:
        self.solver = spla.splu(fd_matrix)

    def solve(self, F: np.ndarray) -> np.ndarray:
        return self.solver.solve(F)
    
    def __del__(self):
        logging.info("Sparse solver deleted.")
    