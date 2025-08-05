from typing import Any, Self
from abc import abstractmethod
import numpy as np

from ..base.consts import Coordinates, Grid
from ..base.grids import BaseGrid


class FDPolarGrid_ArtBndry(BaseGrid):
    """A base class for a grid with a circular artificial boundary
    with imposed evenly-spaced (local) polar coordinates at that
    boundary.
    """
    @property
    def angular_gridpts_artificial_boundary(self) -> np.ndarray:
        """The values of the angular gridpoints at the artificial
        boundary"""
        return np.linspace(0, 2*np.pi, self.num_angular_gridpoints, endpoint=False)
    
    @property
    def num_gridpoints(self) -> int:
        """The total number of gridpoints """
        return self.shape[0] * self.shape[1]
    
    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the grid; that is, the number of gridpoints
        in the first coordinate direction ("angular"), and the number
        of gridpoints in the second coordinate direction ("radial")
        """
        return (self.num_angular_gridpoints, self.num_radial_gridpoints)
    
    @property 
    @abstractmethod
    def num_radial_gridpoints(self) -> int:
        """Number of gridpoints in the "radial" coordinate
        direction.
        """
        pass 

    @property 
    @abstractmethod
    def num_angular_gridpoints(self) -> int:
        """Number of gridpoints in the "angular" coordinate
        direction.
        """
        pass
    

class FDLocalPolarGrid(FDPolarGrid_ArtBndry):
    """An implementation of a Polar Grid for finite differences.
    
    Attributes:
        center (Coordinates): Global (x,y) coordinates of the center
            of the grid 
        r_local (np.ndarray): Local radial coordinates of
            gridpoints
        theta_local (np.ndarray): Local angular coordinates of
            gridpoints
        shape (tuple): The shape of the grid
    """
    def __init__(
        self,
        center: Coordinates,
        num_r_pts: int,
        num_theta_pts: int,
        r_min: float,
        r_max: float
    ):
        """Initialize a local polar grid at a specific point.

        Args:
            center (Coordinates): The global Cartesian X/Y
                coordinates of the center of this obstacle
            num_r_pts (int): The number of gridpoints to use in the 
                radial direction 
            num_theta_pts (int): The number of gridpoints to use 
                in the angular direction
            r_min (float): The smallest radius for the grid 
            r_max (float): The largest radius for the grid
        """
        # Store global centering of this grid
        self.center = center
        
        # Construct local polar grid
        self.r_vals = np.linspace(r_min, r_max, num_r_pts)
        self.theta_vals = np.linspace(0, 2*np.pi, num_theta_pts, endpoint=False)
        self.r_local, self.theta_local = np.meshgrid(self.r_vals, self.theta_vals)
        
        # Store constants used for coordinate conversions
        self.sin_theta_local = np.sin(self.theta_local)
        self.cos_theta_local = np.cos(self.theta_local)
        
        # Store other needed finite difference items 
        self.dr = self.r_vals[1] - self.r_vals[0]
        self.dtheta = self.theta_vals[1] - self.theta_vals[0]
    

    @property 
    def num_radial_gridpoints(self) -> int:
        """Number of gridpoints in the "radial" coordinate
        direction.
        """
        return self.r_local.shape[1] 


    @property 
    def num_angular_gridpoints(self) -> int:
        """Number of gridpoints in the "angular" coordinate
        direction.
        """
        return self.r_local.shape[0]


    def local_coords_to_global_XY(
        self,
        indexes: Any = None,
    ) -> tuple[Grid, Grid]:
        if indexes is None:
            global_x = (
                self.center[0] 
                + self.r_local * np.cos(self.theta_local) 
            )
            global_y = (
                self.center[1] 
                + self.r_local * np.sin(self.theta_local)
            )
        else:
            global_x = (
                self.center[0] 
                + (
                    self.r_local[indexes] 
                    * np.cos(self.theta_local[indexes]) 
                )
            )
            global_y = (
                self.center[1] 
                + (
                    self.r_local[indexes] 
                    * np.sin(self.theta_local[indexes])
                )
            )
        return global_x, global_y
    

    def global_XY_to_local_coords(
        self,
        global_X: Grid,
        global_Y: Grid
    ) -> tuple[Grid, Grid]:
        # Get local cartesian coordinates from global ones
        local_x = global_X - self.center[0]
        local_y = global_Y - self.center[1]

        # Convert these to local polar coordinates 
        local_r = np.sqrt(local_x**2 + local_y**2)
        local_theta = np.atan2(local_y, local_x)
        return local_r, local_theta
    

    
class CartesianGrid(BaseGrid):
    def __init__(self, X_global, Y_global):
        self.X_global = X_global
        self.Y_global = Y_global 

    @property
    def shape(self):
        return self.X_global.shape 
    
    @property
    def num_gridpoints(self):
        return self.X_global.shape[0] * self.X_global.shape[1]

    def local_coords_to_global_XY(self, indexes = None):
        if indexes is None:
            return self.X_global, self.Y_global 
        else:
            return self.X_global[indexes], self.Y_global[indexes]
        
    
    def global_XY_to_local_coords(self,
        global_X: Grid,
        global_Y: Grid
    ) -> tuple[Grid, Grid]:
        return global_X, global_Y   # Translation invariance!