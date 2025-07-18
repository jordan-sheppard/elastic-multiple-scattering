import numpy as np
from typing import Any
from itertools import count

from .consts import Grid
from .grids import BaseGrid


class IncidentPlanePWave:
    """An incident plane P-wave in an elastic multiple-scattering
    problem.

    Attributes:
        phi (float): The global angle at which the incident wave 
            is propagating
        k (float): The wavenumber of the incident wave
        id (int): The id of the wave (for caching purposes)
    """
    id_iter = count()
    
    def __init__(
        self,
        angle_of_incidence: float,
        wavenumber: float
    ):
        """Constructs an incident plane P-wave, given an angle of
        incidence (phi) and a frequency (omega).
        
        Args:
            angle_of_incidence (float): The angle (phi) at which
                the incident plane P-wave is propagating.
            wavenumber (float): The wavenumber (k) of the incident
                wave
        """
        self.id = next(self.id_iter)
        self.phi = angle_of_incidence
        self.k = wavenumber
    
    def __call__(
        self, 
        grid: BaseGrid,
        indexes: Any = None
    ) -> Grid:
        """Evaluates the incident P-wave potential at given global
        Cartesian gridpoints.
        
        Args:
            grid (BaseGrid): A grid containing gridpoints at which
                to evaluate the incident P-wave potential.
            indexes (Any): An expression used to index into the 
                gridpoint arrays to select only certain gridpoints 
                to return (if None, defaults to all). Use
                np.s_[<slice>] to create a valid slicing index
                expression (booleans, integers, etc. are also
                allowed)
        
        Returns:
            Grid: An array containing the incident P-wave potential
                values at each given (x,y) gridpoint.
        """
        # This formula is from Eq. (1) in Villamizar et. al. 2024
        global_x, global_y = grid.local_coords_to_global_XY(indexes)
        dot_prod = (
            global_x * np.cos(self.phi) 
            + global_y * np.sin(self.phi)
        )
        return np.exp(1j * self.k * dot_prod)