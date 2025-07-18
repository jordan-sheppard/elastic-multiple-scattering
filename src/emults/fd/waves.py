import numpy as np

from ..base.waves import IncidentPlanePWave
from ..base.medium import LinearElasticMedium
from ..base.consts import CoordinateSystem
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
        self.lam = medium.lam 
        self.mu = medium.mu

        # Store coordinate transformation attributes
        self._initialize_coordinate_transformation()


    def _initialize_coordinate_transformation(self) -> None:
        """Initializes the constants needed for the coordinate
        transformation from global to m-local coordinates
        """
        # For rotating for displacement representation in m-local polar coordinates
        self.cosine_rotation_displacement = np.cos(-self.Theta_m)
        self.sine_rotation_displacement = np.sin(-self.Theta_m)

        # For representation in global polar coordinates
        self.cosine_global = np.cos(self.Theta_global)
        self.sine_global = np.sin(self.Theta_global)
        self.cosine_global_doubled = np.cos(2 * self.Theta_global)
        self.sine_global_doubled = np.sin(2 * self.Theta_global)

        # For representation in global cartesian coordinates
        self.cosine_local = np.cos(self.Theta_m)
        self.sine_local = np.sin(self.Theta_m)
        self.cosine_local_doubled = np.cos(2 * self.Theta_m)
        self.sine_local_doubled = np.sin(2 * self.Theta_m)

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

        # Get psi_inc (will be zero for plane p-wave)
        psi_inc = np.zeros_like(phi_inc)
        return np.stack((phi_inc, psi_inc), axis=-1) 


    def displacement(
        self,
        boundary_only: bool = True,
        coordinate_system: CoordinateSystem = CoordinateSystem.LOCAL_POLAR
    ) -> np.ndarray:
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
            cosine_rotation = self.cosine_rotation_displacement[:,0]     # Shape (N_{theta}^m,)
            sine_rotation = self.sine_rotation_displacement[:,0]         # Shape (N^{theta}^m,)
            cosine_local = self.cosine_local[:,0]
            sine_local = self.sine_local[:,0]
            cosine_global = self.cosine_global[:,0]
            sine_global = self.sine_global[:,0]
        else:
            phi_inc = self.incident_wave(
                self.m_local_grid
            )                                               # Shape (N_{theta}^m, N_r^m)
            cosine_rotation = self.cosine_rotation_displacement          # Shape (N_{theta}^m, N_r^m)
            sine_rotation = self.sine_rotation_displacement              # Shape (N_{theta}^m, N_r^m)
            cosine_local = self.cosine_local 
            sine_local = self.sine_local
            cosine_global = self.cosine_global
            sine_global = self.sine_global 

        # Get displacement in m-local coordinates
        u_inc_r_m = 1j * self.kp * phi_inc * cosine_rotation
        u_inc_theta_m = 1j * self.kp * phi_inc * sine_rotation
        if coordinate_system is CoordinateSystem.LOCAL_POLAR:
            return np.stack((u_inc_r_m, u_inc_theta_m), axis=-1)    # Shape (N_{theta}^m, N_r^m, 2)
        
        # Get displacement in global polar coordinates
        u_inc_r_global = 1j * self.kp * phi_inc * cosine_global
        u_inc_theta_global = -1j * self.kp * phi_inc * sine_global
        if coordinate_system is CoordinateSystem.GLOBAL_POLAR:
            return np.stack((u_inc_r_global, u_inc_theta_global), axis=-1)    # Shape (N_{theta}^m, N_r^m, 2)

        # Get cartesian displacement (which is translation invariant, so same
        # in local and global cartesian coordinates)
        u_inc_x = u_inc_r_m * cosine_local - u_inc_theta_m * sine_local
        u_inc_y = u_inc_r_m * sine_local + u_inc_theta_m * cosine_local
        if coordinate_system is CoordinateSystem.GLOBAL_CARTESIAN or coordinate_system is CoordinateSystem.LOCAL_CARTESIAN:
            return np.stack((u_inc_x, u_inc_y), axis=-1)    # Shape (N_{theta}^m, N_r^m, 2)

    def stress(
        self,
        boundary_only: bool = True,
        coordinate_system: CoordinateSystem = CoordinateSystem.LOCAL_POLAR
    ) -> np.ndarray:
        """Return the (m)-local polar stresses caused by 
        the incident plane wave.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of the stored m-local grid. Defaults to False.

        Returns:
            np.ndarray: A shape (N_{theta}^m, N_{r}^m, 3) (or
                (N_{theta}^m, 3) array if boundary_only=True), where
                the [:,:,0] (or [:,0], respectively) slice carries
                the (m)-local compressional stress sigma_{rr} (sigma_{xx}), 
                the [:,:,1] ([:,1], respectively) slice carries the
                (m)-local compressional-shear stress sigma_{r theta} (sigma_{xy}),
                and the [:,:,2] ([:,2], respectively) slice carries the
                (m)-local shear-shear stress sigma_{theta theta} (sigma_{yy})
        """
        # Get incident phi values based on global gridpoints
        if boundary_only:
            phi_inc = self.incident_wave(
                self.m_local_grid,
                np.s_[:,0]
            )                                               # Shape (N_{theta}^m,)
            cosine_global = self.cosine_global[:,0]
            sine_global = self.sine_global[:,0]
            cosine_global_doubled = self.cosine_global_doubled[:,0]
            sine_global_doubled = self.sine_global_doubled[:,0]
            cosine_local_doubled = self.cosine_local_doubled[:,0]
            sine_local_doubled = self.sine_local_doubled[:,0]
        else:
            phi_inc = self.incident_wave(
                self.m_local_grid
            )                                               # Shape (N_{theta}^m, N_r^m)
            cosine_global = self.cosine_global
            sine_global = self.sine_global
            cosine_global_doubled = self.cosine_global_doubled
            sine_global_doubled = self.sine_global_doubled
            cosine_local_doubled = self.cosine_local_doubled
            sine_local_doubled = self.sine_local_doubled

        # Parse needed constants 
        k = self.kp 
        lam = self.lam
        mu = self.mu
      
        # Get local polar stress representation 
        sigma_rr_inc_local = -k**2 * (
            lam + mu + (mu * cosine_local_doubled)
        ) * phi_inc 
        sigma_rtheta_inc_local = k**2 * (
            mu * sine_local_doubled  
        ) * phi_inc 
        sigma_thetatheta_inc_local = -k**2 * (
            lam + mu - (mu * cosine_local_doubled)
        ) * phi_inc 
        if coordinate_system is CoordinateSystem.LOCAL_POLAR:
            return np.stack((sigma_rr_inc_local, sigma_rtheta_inc_local, sigma_thetatheta_inc_local), axis=-1)    # Shape (N_{theta}^m, N_r^m, 2)
        
        # Get global polar stress representation 
        sigma_rr_inc_global = k**2 * (
            -lam - mu - (mu * cosine_global_doubled)
        ) * phi_inc 
        sigma_rtheta_inc_global = k**2 * (
            mu * sine_global_doubled  
        ) * phi_inc 
        sigma_thetatheta_inc_global = k**2 * (
            -lam - mu + (mu * cosine_global_doubled)
        ) * phi_inc 
        if coordinate_system is CoordinateSystem.GLOBAL_POLAR:
            return np.stack((sigma_rr_inc_global, sigma_rtheta_inc_global, sigma_thetatheta_inc_global), axis=-1)
        
        # Get local/global cartesian stress representation 
        sigma_xx_inc = (
            sigma_rr_inc_global * cosine_global**2 
            + sigma_thetatheta_inc_global * sine_global**2 
            - 2 * sigma_rtheta_inc_global * sine_global * cosine_global
        )
        sigma_xy_inc = (
            (sigma_rr_inc_global - sigma_thetatheta_inc_global) * sine_global * cosine_global 
            + sigma_rtheta_inc_global * (cosine_global**2 - sine_global**2)
        )
        sigma_yy_inc = (
            sigma_rr_inc_global * sine_global**2 
            + sigma_thetatheta_inc_global * cosine_global**2 
            + 2 * sigma_rtheta_inc_global * sine_global * cosine_global
        )
        if coordinate_system is CoordinateSystem.GLOBAL_CARTESIAN or coordinate_system is CoordinateSystem.LOCAL_CARTESIAN:
            return np.stack((sigma_xx_inc, sigma_xy_inc, sigma_yy_inc), axis=-1)
        

