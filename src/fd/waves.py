import numpy as np

from ..base.waves import IncidentPlanePWave
from ..base.medium import LinearElasticMedium
from .grids import FDLocalPolarGrid


class IncidentPlanePWaveEvaluator:
    """An object allowing for easy evaluation of an IncidentPlanePWave
    (including displacement, stress, etc.) at each of the gridpoints
    on a given local grid.
    """
    def __init__(
        self,
        incident_wave: IncidentPlanePWave,
        local_grid: FDLocalPolarGrid,
        medium: LinearElasticMedium
    ) -> None:
        """Initialize both the incident wave itself, and grid that
        we would like to evaluate it on.
        
        Args:
            incident_wave (IncidentPlanePWave): The incident wave 
                that we would like to evaluate 
            local_grid (FDLocalPolarGrid): The grid on which we'd
                like to evaluate the incident wave
            medium (LinearElasticMedium): The elastic medium where 
                this incident wave is propagating
        """
        # Store grid attributes 
        self.incident_wave = incident_wave
        self.m_local_grid = local_grid
        self.Theta_global = local_grid.local_coords_to_global_polar()[1]
        self.Theta_m = local_grid.theta_local

        # Store medium attributes
        self.kp = medium.kp 

        # Store coordinate transformation attributes
        self._initialize_coordinate_transformation()


    def _initialize_coordinate_transformation(self) -> None:
        """Initializes the constants needed for the coordinate
        transformation from global to m-local coordinates
        """
        angle_diffs = self.Theta_m - 2 * self.Theta_global
        self.cosine_rotation = np.cos(angle_diffs)
        self.sine_rotation = np.sin(angle_diffs)
    

    def potentials(self, boundary_only: bool = True) -> np.ndarray:
        """Return the incident phi and psi potentials at each
        gridpoint of the m-local grid

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle m's grid. Defaults to False.

        Returns:
            np.ndarray: A shape (N_{theta}^m, N_r^m,  2) (or
                (N_{theta}^m, 2) array if boundary_only=True), where
                the [:,:,0] (or [:,0], respectively) slice carries
                the phi_{inc} potential values, and the 
                [:,:,1] ([:,1], respectively) slice carries the
                psi_{inc} potential values.
        """
        # Get phi_inc
        if boundary_only:
            phi_inc = self.incident_wave(
                self.m_local_grid,
                np.s_[:,0]
            )       # Shape (N_{theta}^m,)
        else:
            phi_inc = self.incident_wave(
                self.m_local_grid
            )       # Shape (N_{theta}^m, N_r^m)

        # Get psi_inc (will be zero for plane wave)
        psi_inc = np.zeros_like(phi_inc)
        return np.stack((phi_inc, psi_inc), axis=-1) 


    def displacement(self, boundary_only: bool = True) -> np.ndarray:
        """Return the (m)-local polar displacement caused by 
        the incident plane wave.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of the stored m-local grid. Defaults to False.

        Returns:
            np.ndarray: A shape (N_{theta}^m, N_{r}^m, 2) (or
                (N_{theta}^m, 2) array if boundary_only=True), where
                the [:,:,0] (or [:,0], respectively) slice carries
                the (m)-local radial displacement, and the 
                [:,:,1] ([:,1], respectively) slice carries the
                (m)-local angular displacement
        """
        # Get incident phi values based on global gridpoints
        if boundary_only:
            phi_inc = self.incident_wave(
                self.m_local_grid,
                np.s_[:,0]
            )                                               # Shape (N_{theta}^m,)
        else:
            phi_inc = self.incident_wave(
                self.m_local_grid
            )                                               # Shape (N_{theta}^m, N_r^m)
        
        # Get global-to-m rotation matrix values based on global gridpoints
        if boundary_only:
            cosine_rotation = self.cosine_rotation[:,0]     # Shape (N_{theta}^m,)
            sine_rotation = self.sine_rotation[:,0]         # Shape (N^{theta}^m,)
        else:
            cosine_rotation = self.cosine_rotation          # Shape (N_{theta}^m, N_r^m)
            sine_rotation = self.sine_rotation              # Shape (N_{theta}^m, N_r^m)

        # Return displacement in m-local coordinates
        u_inc_r_m = 1j * self.kp * phi_inc * cosine_rotation
        u_inc_theta_m = 1j * self.kp * phi_inc * sine_rotation
        return np.stack((u_inc_r_m, u_inc_theta_m), axis=-1)    # Shape (N_{theta}^m, N_r^m, 2)
    

        
        


