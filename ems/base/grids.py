from typing import Any
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from .consts import Grid

class BaseGrid(ABC):
    """An abstract implementation of a grid class"""
    @property 
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """The number of gridpoints along each dimension of the grid.
        The first dimension is the "radial" dimension, the second is
        the "angular" dimension.
        """
        pass

    @property 
    @abstractmethod
    def num_gridpoints(self) -> int:
        """The number of gridpoints in the grid."""
        pass

    def plot(self, **kwargs):
        """Plot this grid on the current matplotlib axix as a
        mesh of piecewise-linear gridlines through each gridpoint.
        
        Args:
            **color (str): The color to draw these grids as 
                (default: black)
        """     
        # Parse out color, if provided
        if 'color' in kwargs:
            color = kwargs['color']
        else:
            color = 'black'

        # Get X and Y coordinates of the current grid 
        X, Y = self.local_coords_to_global_XY()

        # Get axis to plot this on 
        ax = plt.gca()     
        
        # Create gridlines for each of the gridlines to plot
        # (This code assumes we'll need to connect from the last
        # angular ray's gridpoints to the first angular ray's
        # gridpoints explicitlly)
        segs1 = np.stack(
            (np.vstack((X,X[0,:])), np.vstack((Y, Y[0,:]))), 
            axis=2
        )  
        gridlines1 = LineCollection(segs1, color=color, zorder=1)
        segs2 = segs1.transpose(1,0,2)      # Gets axis=1 gridlines
        gridlines2 = LineCollection(segs2, color=color, zorder=2)

        # Plot these intersecting gridlines 
        ax.add_collection(gridlines1)
        ax.add_collection(gridlines2)

        # Automatically scale with new grid attached 
        ax.autoscale()


    def local_coords_to_global_polar(
        self,
        indexes: Any = None
    ) -> tuple[Grid, Grid]:
        """Return the local gridpoints as global polar coordinates.

        Args:
            indexes (Any): An expression used to index into the 
                gridpoint arrays to select only certain gridpoints 
                to return (if None, defaults to all). Use
                np.s_[<slice>] to create a valid slicing index
                expression (booleans, integers, etc. are also
                allowed)
        
        Returns:
        
            Grid: The global r-coordinates of the local gridpoints 
            Grid: The global theta-coordinates of the local gridpoints
        """
        X_global, Y_global = self.local_coords_to_global_XY(indexes)
        r_global = np.sqrt(X_global**2 + Y_global**2)
        theta_global = np.atan2(Y_global, X_global)
        return r_global, theta_global


    @abstractmethod
    def local_coords_to_global_XY(
        self,
        indexes: Any = None
    )-> tuple[Grid, Grid]:
        """Return the local gridpoints as global X/Y coordinates.

        Args:
            indexes (Any): An expression used to index into the 
                gridpoint arrays to select only certain gridpoints 
                to return (if None, defaults to all). Use
                np.s_[<slice>] to create a valid slicing index
                expression (booleans, integers, etc. are also
                allowed)
        
        Returns:
            Grid: The global X-coordinates of the local gridpoints 
            Grid: The global Y-coordinates of the local gridpoints
        """
        pass

    @abstractmethod 
    def global_XY_to_local_coords(
        self,
        global_X: Grid,
        global_Y: Grid
    ) -> tuple[Grid, Grid]:
        """Represent global coordinates in this local coordinate system.
        
        Returns:
            Grid: The first local coordinate of the given gridpoints 
            Grid: The second local coordinate of the given gridpoints
        """
        pass
    
    






