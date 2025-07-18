from abc import ABC, abstractmethod
from itertools import count

from .consts import Coordinates, BoundaryCondition
from .grids import BaseGrid

class BaseObstacle(ABC):
    """A base class for an obstacle in an elastic multiple-
    scattering problem."""
    id_iter = count()

    def __init__(
        self, 
        center: Coordinates,
        boundary_condition: BoundaryCondition
    ):
        """Instantiate a bare-bones obstacle with no geometry."""
        self.center_global = center
        self.bc = boundary_condition
        self.id = next(self.id_iter)
    
    @classmethod 
    def reset_id_counter(cls):
        cls.id_iter = count()
        

