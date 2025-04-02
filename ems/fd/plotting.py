from typing import Optional, Self
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Colormap

from ..base.waves import IncidentPlanePWave
from .grids import FDLocalPolarGrid, FDPolarGrid_ArtBndry
from .obstacles import MKFE_FDObstacle


class ScatteringProblemPlotter:
    """Plots the results of a scattering problem.
    
    Requires all obstacles to have each other obstacle and the 
    provided incident wave cached.
    """
    def __init__(
        self,
        obstacles: list[MKFE_FDObstacle],
        incident_wave: IncidentPlanePWave
    ) -> None:
        """Store the obstacles and incident wave to plot.
        
        Args:
            obstacles (list[MKFE_FDObstacle]): A list of 
                obstacles of interest in the scattering problem
            incident_wave (IncidentPlanePWave): The 
                incident wave of interest in the scattering problem
        """
        self.obstacles: dict[int, MKFE_FDObstacle] = {
            obstacle.id: obstacle for obstacle in obstacles
        }
        self.incident_wave = incident_wave

    def plot_grid(
        self,
        **kwargs
    ):
        """Plot the local grids of the obstacles in this problem.
        
        Args:
            **color (str): A desired color for the gridlines
                (default: black)
        """
        for obstacle in self.obstacles.values():
            obstacle.grid.plot(**kwargs)










