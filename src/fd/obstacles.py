from abc import abstractmethod
from typing import Optional, Self
import numpy as np
from math import floor, ceil
from scipy.sparse import linalg as spla
import scipy.sparse as sparse
from scipy.special import hankel1
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Colormap
import logging

from .utils import sparse_periodic_tridiag, sparse_block_row, sparse_block_antidiag
from ..base.consts import (
    Coordinates, BoundaryCondition, SparseMatrix,
    Vector, Grid
)
from ..base.obstacles import BaseObstacle
from ..base.waves import IncidentPlanePWave
from ..base.medium import LinearElasticMedium
from .grids import FDLocalPolarGrid, FDGrid_Polar_Art_Bndry
from ..base.interpolation import PeriodicInterpolator1D


class FDObstacle(BaseObstacle):
    """A base class for an obstacle in a finite-difference
    elastic multiple scattering problem.

    Contains gridpoints and compressional (phi) and shear (psi) 
    potential values at each gridpoint.

    Wraps obstacle geometry in a circular artificial boundary, which
    gives the finite computational domain as the area in between
    the obstacle boundary and the artificial bounndary.

    The obstacle is assumed to have either a hard or a soft boundary
    condition at the physical boundary of the geometry.
    NOTE: Currently, only a hard boundary condition is supported.
    
    Attributes:
        parent_medium (LinearElasticMedium): The medium this obstacle
            is placed in; the medium in which scattered waves from
            this obstacle must pass through
        center_global (Coordinates): The global X/Y coordinates
            of the center of the obstacle
        r_artificial_boundary (float): The radius of the circular
            artifical boundary from self.center_global
        bc (BoundaryCondition): The boundary 
            condition at the physical boundary (hard or soft)
            NOTE: Currently, only a hard boundary condition is 
            supported
        PPW (int): The points-per-wavelength for the grid to use 
            for this finite-difference scheme
        grid (BaseGrid): All finite-difference gridpoints at this
            obstacle
        fd_unknowns (Vector): A 1-dimensional vector of all
            finite-difference unknowns
        fd_matrix (SparseMatrix): A sparse matrix representing the 
            finite-difference scheme for this problem 
        fd_matrix_LU (SparseMatrixLUDecomp): A LU-decomposition
            object allowing quick solutions to finite-difference 
            updates
        phi_vals (Grid): The value of the compressional potential phi 
            at all gridpoints (initialized to all 0s)
        psi_vals (Grid): The value of the shear potential psi
            at all gridpoints (initialized to all 0s)
    """
    def __init__(
        self,
        center: Coordinates,
        r_artificial_boundary: float,
        boundary_condition: BoundaryCondition,
        parent_medium: LinearElasticMedium, 
        PPW: int
    ):
        """Initialize a FDObstacle with arbitrary geometry.
        
        Args:
            center (Coordinates): The global X/Y coordinates of the
                center of the obstacle
            r_artificial_boundary (float): The radius of the circular
                artifical boundary from the center of the obstacle
            boundary_condition (BoundaryCondition): The boundary 
                condition at the physical boundary (hard or soft)
                NOTE: Currently, only a hard boundary condition is 
                supported
            parent_medium (LinearElasticMedium): The medium this
                obstacle is placed in; the medium in which scattered
                waves from this obstacle must pass through
            PPW (int): The number of gridpoints per wavelength to use
                in the radial direction while constructing the grid
        """
        # Initialize BaseObstacle instance with obstacle center and 
        # the physical boundary condition
        super().__init__(center, boundary_condition)
        
        # Store other needed constants
        self.r_artificial_boundary = r_artificial_boundary
        self.parent_medium = parent_medium
        self.PPW = PPW

        # Choose wavelength to be the compressional wavelength
        # TODO: IS THIS BEST?
        self.wavelength = self.parent_medium.wavelength_p

        # Initialize other needed constants 
        self.setup()
    

    def setup(self):
        """Set up the obstacle by generating a finite-difference grid
        and creating the corresponding Finite-Difference matrix."""
        # Generate grid 
        self.grid = self.generate_grid()

        # Create unknown vector and unknown matrix
        self.fd_unknowns = np.zeros(self.num_unknowns, dtype='complex128') 
        self.fd_matrix = self.construct_fd_matrix()    # SHOULD BE OVERRIDDEN IN SUBCLASS
        self.fd_matrix_LU = spla.splu(self.fd_matrix)

        # Create solution vectors/matrices 
        self.phi_vals = np.zeros(self.grid.shape, dtype='complex128')
        self.psi_vals = np.zeros(self.grid.shape, dtype='complex128')
        
    

    def plot_grid(self, **kwargs):
        """Plot this obstacle's grid.
        
        Args:
            **color (str): A desired color for the gridlines
                (default: black)
        """
        self.grid.plot(**kwargs)


    def plot_fd_matrix(self, **kwargs):
        """Plot a sparsity pattern of this obstacle's finite difference
        matrix.
        
        Args:
            **markersize (float): The desired marker size for nonzero
                entries of the matrix (default=1)
        """
        plt.spy(self.fd_matrix, **kwargs)


    def solve(
        self,
        obstacles: list[Self],
        incident_wave: Optional[IncidentPlanePWave] = None
    ):
        """Solve the single-scattering finite-difference problem
        given incoming waves from other obstacles/incident wave.

        Constructs a forcing vector F from obstacle/incident wave
        data before solving the system Au = F (where A is the finite
        difference matrix stored in self.fd_matrix/self.fd_matrix_lu)
        
        Updates corresponding class attributes corresponding to
        solution quantities of interest. These include (but are 
        not limited to, depending on the subclass implementation
        of parse_FD_raw_result()):
        
        * phi_vals (Grid): The value of the compressional potential phi 
            at all gridpoints 
        * psi_vals (Grid): The value of the shear potential psi
            at all gridpoints

        Args:
            obstacles (list[FDObstacle]) : A list of other obstacles 
                whose scattered waves are incident upon this obstacle
            incident_wave (IncidentPlaneWave): If provided, an
                incident plane wave
        """
        F = self.construct_forcing_vector(obstacles, incident_wave)
        self.fd_unknowns = self.fd_matrix_LU.solve(F)
        self.parse_raw_FD_result(self.fd_unknowns)
    

    def parse_raw_FD_result(
        self,
        result: Vector
    ):
        """Updates corresponding class attributes corresponding to
        solution quantities of interest. These include (but are 
        not limited to, depending on the subclass implementation
        of parse_FD_raw_result()):
        
        * phi_vals (Grid): The value of the compressional potential phi 
            at all gridpoints 
        * psi_vals (Grid): The value of the shear potential psi
            at all gridpoints

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        """
        self.phi_vals = self.parse_phi_from_FD_result(result)
        self.psi_vals = self.parse_psi_from_FD_result(result)


    @property
    @abstractmethod
    def num_unknowns(self) -> int:
        """The total number of unknown field values."""
        pass

    @property
    @abstractmethod
    def num_ghost_points_physical_boundary(self) -> int:
        """The total number of ghost points at the physical
        boundary.
        """
        pass

    @property
    @abstractmethod
    def num_ghost_points_artificial_boundary(self) -> int:
        """The total number of ghost points at the artificial
        boundary.
        """
        pass

    @abstractmethod
    def generate_grid(
        self
    ) -> FDGrid_Polar_Art_Bndry:
        """Generate the grid given a specified wavelength in 
        self.wavelength, and a number of points per wavelength
        given by self.points_per_wavelength.

        Returns:
            BaseGrid: The corresponding grid.
        """
        pass
    

    @abstractmethod
    def construct_fd_matrix(self) -> SparseMatrix:
        """Construct the finite-difference matrix for this problem.
        
        Returns:
            SparseMatrix: A sparse finite-difference matrix for this
            problem
        """
        pass


    @abstractmethod
    def construct_forcing_vector(
        self,
        obstacles: list[Self],
        incident_wave: Optional[IncidentPlanePWave] = None
    ) -> Vector:
        """Construct a forcing vector F corresponding to the finite-
        difference matrix for this problem.
        
        Do this using data from other obstacles as well as incident 
        wave data (if provided).

        Args:
            obstacles (list[FDObstacle]) : A list of other obstacles 
                whose scattered waves are incident upon this obstacle
            incident_wave (IncidentPlaneWave): If provided, an
                incident plane wave
        
        Returns:
            Vector: The forcing vector F for the finite-difference
                method
        """
        pass


    @abstractmethod
    def parse_phi_from_FD_result(
        self,
        result: Vector
    ) -> Grid:
        """Parses the phi (compressional potential) values at each
        gridpoint from a given raw finite-difference result vector.

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        
        Returns:
            Grid: The value of the phi potential at each gridpoint
                in self.grid
        """
        pass


    @abstractmethod
    def parse_psi_from_FD_result(
        self,
        result: Vector
    ) -> Grid:
        """Parses the psi (shear potential) values at each
        angular gridpoint from a given raw finite-difference result vector.

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        
        Returns:
            Grid: The value of the psi potential at each gridpoint
                in self.grid
        """
        pass 


    @abstractmethod
    def get_scattered_wave(
        self,
        X: Optional[Grid]=None,
        Y: Optional[Grid]=None
    ) -> tuple[Grid, Grid]:
        """Retrive the value of the outgoing scattered wave from
        this obstacle at the given global gridpoints.

        If either X or Y is None, then these inputs are ignored,
        and the value of the outgoing scattered wave is returned
        at the local gridpoints.

        Otherwise, all points (X,Y) should be lying outside of the
        computational domain of this obstacle.
        
        Args:
            X (Grid): The global X-coordinates (if None, retrieve
                scattered wave values at each local gridpoint)
            Y (Grid): The global Y-coordinates (if None, retrieve
                scattered wave values at each local gridpoint)
        
        Returns:
            Grid: The value of the outgoing scattered phi-wave (the
                compressional potentional) at each gridpoint
            Grid: The value of the outgoing scattered psi-wave (the
                shear potential) at each gridpoint

        Raises:
            ValueError: If (X,Y) is within the computational 
                domain of this obstacle
        """
        pass


    @abstractmethod
    def get_scattered_wave_at_obstacle(
        self, 
        obstacle: Self,
    ) -> tuple[Grid, Grid]:
        """Gets the scattered wave at and around another obstacle's
        physical boundary."""
        pass


class MKFE_FDObstacle(FDObstacle):
    """A base class for an obstacle in a finite-difference
    elastic multiple scattering problem, using the MKFE ABC
    at the circular artificial boundary.

    Contains gridpoints, potential values, and KFE angular 
    coefficients that represent the value of the scattered
    fields at each gridpoint/everywhere in space.

    Wraps obstacle geometry in a circular artificial boundary, and 
    uses the MKFE ABC given by Villamizar et. al. for the scattered
    potentials at this artificial boundary.

    The obstacle is assumed to have either a hard or a soft boundary
    condition at the physical boundary of the geometry.
    NOTE: Currently, only a hard boundary condition is supported.
    
    Attributes:
        center_global (Coordinates): The global X/Y coordinates
            of the center of the obstacle
        r_artificial_boundary (float): The radius of the circular
            artifical boundary from self.center_global
        bc (BoundaryCondition): The boundary 
            condition at the physical boundary (hard or soft)
            NOTE: Currently, only a hard boundary condition is 
            supported
        parent_medium (LinearElasticMedium): The medium this obstacle
            is placed in; the medium in which scattered waves from
            this obstacle must pass through
        PPW (int): The points-per-wavelength for the grid to use 
            for this finite-difference scheme
        num_farfield_terms (int): The number of terms to use 
            in the farfield expansion ABC
        grid (BaseGrid): All finite-difference gridpoints at this
            obstacle (the 0th)
        fd_unknowns (Vector): A 1-dimensional vector of all
            finite-difference unknowns
        fd_matrix (SparseMatrix): A sparse matrix representing the 
            finite-difference scheme for this problem 
        fd_matrix_LU (SparseMatrixLUDecomp): A LU-decomposition
            object allowing quick solutions to finite-difference 
            updates
        phi_vals (Grid): The value of the compressional potential phi 
            at all gridpoints (initialized to all 0s)
        psi_vals (Grid): The value of the shear potential psi
            at all gridpoints (initialized to all 0s)
    """
    def __init__(
        self,
        center: Coordinates,
        r_artificial_boundary: float,
        boundary_condition: BoundaryCondition,
        num_farfield_terms: int,
        parent_medium: LinearElasticMedium,
        PPW: int,
    ):
        """Initializes an MKFE_FDObstacle instance.

        The obstacle is  centered at the given center point, with a
        circular artificial boundary with radius
        r_artifical_boundary.
        
        Args:
            center(tuple[float, float]): The center of the obstacle
                in global cartesian coordinates
            r_artificial_boundary (float): The radius of the circular
                artificial boundary from the given center point 
            boundary_condition (BoundaryCondition): The boundary
                condition (either soft or hard) imposed at the
                physical boundary of this obstacle.
            num_farfield_terms (int): The number of terms to use 
                in the farfield expansion for the outgoing wave
                radiating from this obstacle
            parent_medium (LinearElasticMedium): The medium this
                obstacle is placed in; the medium in which scattered
                waves from this obstacle must pass through
            PPW (int): The points-per-wavelength for the grid to use 
                for this finite-difference scheme
        """
        # Initialize MKFE attributes 
        self.num_farfield_terms = num_farfield_terms

        # Initialize lookup for other obstacle interactions 
        self.obstacle_boundary_info = dict()

        super().__init__(
            center,
            r_artificial_boundary,
            boundary_condition,
            parent_medium,
            PPW
        )
    
    def setup(self):
        """Set up the obstacle by generating a finite-difference grid
        and creating the corresponding Finite-Difference matrix."""
        # Initialize everything else but farfield coefficients
        super().setup()

        # Initialize farfield coefficients
        self.farfield_fp_coeffs = np.zeros((self.num_farfield_terms, self.grid.num_angular_gridpoints), dtype='complex128')
        self.farfield_fs_coeffs = np.zeros((self.num_farfield_terms, self.grid.num_angular_gridpoints), dtype='complex128')
        self.farfield_gp_coeffs = np.zeros((self.num_farfield_terms, self.grid.num_angular_gridpoints), dtype='complex128')
        self.farfield_gs_coeffs = np.zeros((self.num_farfield_terms, self.grid.num_angular_gridpoints), dtype='complex128')


    @property 
    def num_unknowns(self):
        # Constants and other needed values 
        NUM_POTENTIALS = 2 
        NUM_FFE_COEFFS_PER_POTENTIAL = 2
        num_layer_angular_gridpoints = self.grid.num_angular_gridpoints

        # Get number of gridpoints in physical domain
        num_physical_gridpoints = self.grid.num_gridpoints

        # Get number of "ghost" gridpoints outside physical domain
        num_ghost_points = (
            (
                self.num_ghost_points_artificial_boundary * 
                num_layer_angular_gridpoints 
            ) + 
            (
                self.num_ghost_points_physical_boundary * 
                num_layer_angular_gridpoints
            )
        )
        
        # Get total number of discretized Farfield angular
        # coefficients at each angular gridpoint
        num_farfield_coeffs = (
            num_layer_angular_gridpoints
            * self.num_farfield_terms
            * NUM_FFE_COEFFS_PER_POTENTIAL
            * NUM_POTENTIALS
        )
        return (
            NUM_POTENTIALS * (num_physical_gridpoints + num_ghost_points)
            + num_farfield_coeffs
        )
    

    def parse_raw_FD_result(
        self,
        result: Vector
    ):
        """Updates corresponding class attributes corresponding to
        solution quantities of interest. These include (but are 
        not limited to, depending on the subclass implementation
        of parse_FD_raw_result()):
        
        * phi_vals (Grid): The value of the compressional potential phi 
            at all gridpoints 
        * psi_vals (Grid): The value of the shear potential psi
            at all gridpoints

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        """
        # Parse phi_vals and psi_vals into self.phi_vals
        # and self.psi_vals
        super().parse_raw_FD_result(result)

        # Parse farfield coefficients into self.farfield_fp_coeffs,
        # self.farfield_gp_coeffs, self.farfield_fs_coeffs, and 
        # self.farfield_gs_coeffs
        self.farfield_fp_coeffs = self.parse_farfield_fp_coeffs_from_FD_result(result)
        self.farfield_gp_coeffs = self.parse_farfield_gp_coeffs_from_FD_result(result)
        self.farfield_fs_coeffs = self.parse_farfield_fs_coeffs_from_FD_result(result)
        self.farfield_gs_coeffs = self.parse_farfield_gs_coeffs_from_FD_result(result)


    def parse_phi_from_FD_result(self, result) -> Grid:
        # Get vector phi values at gridpoints from results vector
        begin_idx = (
            self.num_ghost_points_physical_boundary
            * self.grid.num_angular_gridpoints
        )
        step = self.grid.num_angular_gridpoints
        phi_grid = np.zeros(self.grid.shape, dtype='complex128')
        for r in range(self.grid.num_radial_gridpoints):
            # Parse current radial layer of phi gridpoints
            end_idx = begin_idx + self.grid.num_angular_gridpoints
            logging.debug(f"Phi value indexes for radial ring {r}: [{begin_idx}, {end_idx}]")
            phi_grid[:,r] = result[begin_idx:end_idx]

            # Increment to next radial layer (skip psi gridpoints on same layer)
            begin_idx = end_idx + step
        return phi_grid


    def parse_psi_from_FD_result(self, result) -> Grid:        
        psi_grid = np.zeros(self.grid.shape, dtype='complex128')

        step = self.grid.num_angular_gridpoints
        begin_idx = (
            self.num_ghost_points_physical_boundary
            * self.grid.num_angular_gridpoints
        ) + step        # Skips first round of phi gridpoints
        for r in range(self.grid.num_radial_gridpoints):
            # Parse current radial layer of phi gridpoints
            end_idx = begin_idx + self.grid.num_angular_gridpoints
            logging.debug(f"Psi value indexes for radial ring {r}: [{begin_idx}, {end_idx}]")
            psi_grid[:,r] = result[begin_idx:end_idx]

            # Increment to next radial layer (skip psi gridpoints on same layer)
            begin_idx = end_idx + step

        return psi_grid


    def parse_farfield_fp_coeffs_from_FD_result(
        self, 
        result:Vector
    ) -> np.ndarray:
        """Parse the farfield F_l^p (compressional F) coefficient
        values at each "angular" gridpoint at the artificial
        boundary, for each term l=0, ..., self.num_farfield_terms,
        all from a given raw finite-difference result vector.

        The coefficients would be the coefficients evaluated at the 
        angular gridpoints on the artificial boundary

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        
        Returns:
            np.ndarray: A (L, self.grid.shape[0]) array of 
                compressional F_l^p coefficient values at each
                angular gridpoint on the artificial boundary.
        """
        num_unknown_grid_vals = (
            2 * self.grid.num_gridpoints 
            + 2 * (
                self.num_ghost_points_artificial_boundary 
                + self.num_ghost_points_physical_boundary
            ) * self.grid.num_angular_gridpoints
        )
        
        fp_coeffs = np.zeros(
            (self.num_farfield_terms, self.grid.num_angular_gridpoints),
            dtype='complex128'
        )
        start = num_unknown_grid_vals
        end = num_unknown_grid_vals + self.grid.num_angular_gridpoints
        logging.debug(f"Farfield F_p coeffs start index: {start}")
        for l in range(self.num_farfield_terms):
            fp_coeffs[l] = result[start:end]
            start = end 
            end += self.grid.num_angular_gridpoints
        logging.debug(f"Farfield F_p coeffs end index: {start}")
        return fp_coeffs
    

    def parse_farfield_gp_coeffs_from_FD_result(
        self, 
        result:Vector
    ) -> np.ndarray:
        """Parse the farfield G_l^p (compressional G) coefficient
        values at each "angular" gridpoint at the artificial
        boundary, for each term l=0, ..., self.num_farfield_terms,
        all from a given raw finite-difference result vector.

        The coefficients would be the coefficients evaluated at the 
        angular gridpoints on the artificial boundary

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        
        Returns:
            np.ndarray: A (L, self.grid.shape[0]) array of 
                compressional G_l^p coefficient values at each
                angular gridpoint on the artificial boundary.
        """
        num_unknown_grid_vals = (
            2 * self.grid.num_gridpoints 
            + 2 * (
                self.num_ghost_points_artificial_boundary 
                + self.num_ghost_points_physical_boundary
            ) * self.grid.num_angular_gridpoints
        )
        num_fp_coeffs = (
            self.num_farfield_terms
            * self.grid.num_angular_gridpoints
        )
        
        gp_coeffs = np.zeros(
            (self.num_farfield_terms, self.grid.num_angular_gridpoints),
            dtype='complex128'
        )
        start = num_unknown_grid_vals + num_fp_coeffs
        end = start + self.grid.num_angular_gridpoints
        logging.debug(f"Farfield G_p coeffs start index: {start}")
        for l in range(self.num_farfield_terms):
            gp_coeffs[l] = result[start:end]
            start = end 
            end += self.grid.num_angular_gridpoints
        logging.debug(f"Farfield G_p coeffs end index: {start}")
        return gp_coeffs
    

    def parse_farfield_fs_coeffs_from_FD_result(
        self, 
        result:Vector
    ) -> np.ndarray:
        """Parse the farfield F_l^s (shear F) coefficient
        values at each "angular" gridpoint at the artificial
        boundary, for each term l=0, ..., self.num_farfield_terms,
        all from a given raw finite-difference result vector.

        The coefficients would be the coefficients evaluated at the 
        angular gridpoints on the artificial boundary

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        
        Returns:
            np.ndarray: A (L, self.grid.shape[0]) array of 
                compressional F_l^s coefficient values at each
                angular gridpoint on the artificial boundary.
        """
        num_unknown_grid_vals = (
            2 * self.grid.num_gridpoints 
            + 2 * (
                self.num_ghost_points_artificial_boundary 
                + self.num_ghost_points_physical_boundary
            ) * self.grid.num_angular_gridpoints
        )
        num_compressional_coeffs = 2 * (
            self.num_farfield_terms
            * self.grid.num_angular_gridpoints
        )
        
        fs_coeffs = np.zeros(
            (self.num_farfield_terms, self.grid.num_angular_gridpoints),
            dtype='complex128'
        )
        start = num_unknown_grid_vals + num_compressional_coeffs
        end = start + self.grid.num_angular_gridpoints
        logging.debug(f"Farfield F_s coeffs start index: {start}")
        for l in range(self.num_farfield_terms):
            fs_coeffs[l] = result[start:end]
            start = end 
            end += self.grid.num_angular_gridpoints
        logging.debug(f"Farfield F_s coeffs end index: {start}")
        return fs_coeffs
    

    def parse_farfield_gs_coeffs_from_FD_result(
        self, 
        result:Vector
    ) -> np.ndarray:
        """Parse the farfield G_l^s (shear G) coefficient
        values at each "angular" gridpoint at the artificial
        boundary, for each term l=0, ..., self.num_farfield_terms,
        all from a given raw finite-difference result vector.

        The coefficients would be the coefficients evaluated at the 
        angular gridpoints on the artificial boundary

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        
        Returns:
            np.ndarray: A (L, self.grid.shape[0]) array of 
                compressional G_l^s coefficient values at each
                angular gridpoint on the artificial boundary.
        """
        num_unknown_grid_vals = (
            2 * self.grid.num_gridpoints 
            + 2 * (
                self.num_ghost_points_artificial_boundary 
                + self.num_ghost_points_physical_boundary
            ) * self.grid.num_angular_gridpoints
        )
        num_compressional_coeffs = 2 * (
            self.num_farfield_terms
            * self.grid.num_angular_gridpoints
        )
        num_gs_coeffs = (
            self.num_farfield_terms 
            * self.grid.num_angular_gridpoints
        )
        
        gs_coeffs = np.zeros(
            (self.num_farfield_terms, self.grid.num_angular_gridpoints),
            dtype='complex128'
        )
        start = (
            num_unknown_grid_vals + num_compressional_coeffs
            + num_gs_coeffs
        )
        logging.debug(f"Farfield G_s coeffs start index: {start}")
        end = start + self.grid.num_angular_gridpoints
        for l in range(self.num_farfield_terms):
            gs_coeffs[l] = result[start:end]
            start = end 
            end += self.grid.num_angular_gridpoints
        logging.debug(f"Farfield G_s coeffs end index: {start}")
        return gs_coeffs


    def _cache_obstacle_boundary_data(self, obstacle: Self) -> None:
        """If an obstacle's boundary info is not in self.obstacle_boundary_info,
        caches the desired information about the obstacle boundary there.
        
        Args:
            obstacle (MKFE_FDObstacle): An obstacle we'd like to store
                boundary information about.
        """
        if obstacle.id not in self.obstacle_boundary_info:
            interpolator = PeriodicInterpolator1D((0, 2*np.pi), self.grid.angular_gridpts_artificial_boundary)

            # Get global gridpoints around given obstacle
            obstacle_boundary_global_coords = obstacle.grid.local_coords_to_global_XY()
            X, Y = obstacle_boundary_global_coords
            
            # Validate computational domains don't overlap
            if np.any(
                np.sqrt(
                    (X-self.center_global[0])**2 + (Y-self.center_global[1])**2
                ) < self.r_artificial_boundary
            ):
                raise ValueError("All points must be outside computatational domain of this obstacle")

            # Convert the global coordinates to this obstacle's local coordinate system
            obstacle_boundary_cur_local_coords = self.grid.global_XY_to_local_coords(X, Y)
            obstacle_boundary_R, obstacle_boundary_Theta = obstacle_boundary_cur_local_coords
            obstacle_boundary_R = obstacle_boundary_R.T             # Now, radius is first axis, angle is second
            obstacle_boundary_Theta = obstacle_boundary_Theta.T     # Same here

            # Evaluate all sorts of Hankel functions beforehand, since these are expensive
            obstacle_boundary_kp_hankel0 = hankel1(0, self.parent_medium.kp * obstacle_boundary_R)
            obstacle_boundary_kp_hankel1 = hankel1(1, self.parent_medium.kp * obstacle_boundary_R)
            obstacle_boundary_ks_hankel0 = hankel1(0, self.parent_medium.ks * obstacle_boundary_R) 
            obstacle_boundary_ks_hankel1 = hankel1(1, self.parent_medium.ks * obstacle_boundary_R)

            # Evaluate radii to all the powers needed with kp/ks multiplied 
            # The shapes of these arrays will be (num_farfield_terms, (obstacle_boundary_R shape))
            powers = np.arange(self.num_farfield_terms)
            r_powers_kp = (self.parent_medium.kp * np.expand_dims(obstacle_boundary_R, axis=-1)) ** powers
            r_powers_ks = (self.parent_medium.ks * np.expand_dims(obstacle_boundary_R, axis=-1)) ** powers

            # Store all above information in a lookup dictionary and add to 
            # the obstacle_boundary_info data structure for future use
            obstacle_info = {
                'interpolator': interpolator, 
                'R_local': obstacle_boundary_R,
                'Theta_local': obstacle_boundary_Theta,
                'Hankels': {
                    'kp_ord_0': obstacle_boundary_kp_hankel0,
                    'kp_ord_1': obstacle_boundary_kp_hankel1,
                    'ks_ord_0': obstacle_boundary_ks_hankel0,
                    'ks_ord_1': obstacle_boundary_ks_hankel1
                },
                'Radius_powers': {
                    'kp': r_powers_kp,
                    'ks': r_powers_ks
                }
            }
            self.obstacle_boundary_info[obstacle.id] = obstacle_info
    
    def _plot_periodic_contourf(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        **kwargs
    ) -> None:
        """Plot a filled contour plot with given X/Y/Z values.

        Assumes that we need to attach the last angular gridpoint
        to the first to fill in that gap; assumes that angular
        gridpoint changes happen along axis 0 of each grid.
        
        Args:
            X (np.ndarray): The X-values of interest 
            Y (np.ndarray): The Y-values of interest 
            Z (np.ndarray): The Z-values of interest
            *cmap (Colormap): The Matplotlib colormap object 
                to use for plotting (cm.coolwarm is used if none
                is provided)
        """
        # Get smooth gradient
        color_grid_vals = np.arange(Z.min(), Z.max(), .001)

        # Get rid of gaps in plotting 
        # (assumes angular gridpoints change along axis 0)
        X = np.vstack((X,X[0,:]))
        Y = np.vstack((Y,Y[0,:]))
        Z = np.vstack((Z,Z[0,:]))

        # Plot countour plot
        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
        else:
            cmap = cm.coolwarm
        plt.contourf(X, Y, Z, color_grid_vals, cmap=cmap)
        plt.colorbar()


    def plot_phi_contourf(
        self,
        u_inc: Optional[IncidentPlanePWave] = None,
        other_obstacles: Optional[list[Self]] = None,
        **kwargs
    ) -> None:
        """Plot the scattered phi values at this obstacle as a
        filled contour plot.

        If no arguments are provided, plot the scattered phi
        potential eminating from this obstacle.

        If incident wave and/or other obstacles provided, linearly
        add their phi potential values to this one before 
        plotting.

        Args:
            u_inc (IncidentPlanePWave): If provided, the incident
                plane p-wave on this obstacle.
            other_obstacles (list[Self]): If provided, a list
                of other obstacles whose scattered potentials 
                are incident upon this obstacle.
            *cmap (Colormap): The Matplotlib colormap object 
                to use for plotting (cm.coolwarm is used if none
                is provided)
        """
        phi_tot = self.phi_vals
        if u_inc is not None:
            phi_tot += u_inc(self.grid)
        
        if other_obstacles is not None:
            for obstacle in other_obstacles:
                phi_tot += obstacle.get_scattered_wave_at_obstacle(self, boundary_only=False)[0]


        # Plot contour with given color/colormap args
        X_global, Y_global = self.grid.local_coords_to_global_XY()
        # phi_abs = np.abs(phi_tot)
        # self._plot_periodic_contourf(X_global, Y_global, phi_abs, **kwargs)
        phi_real = np.real(phi_tot)
        self._plot_periodic_contourf(X_global, Y_global, phi_real, **kwargs)


    def plot_psi_contourf(
        self,
        other_obstacles: Optional[list[Self]] = None,
        **kwargs
    ) -> None:
        """Plot the scattered psi values at this obstacle as a
        filled contour plot.

        If no arguments are provided, plot the scattered phi
        potential eminating from this obstacle.

        If other obstacles provided, linearly
        add their psi potential values to this one before 
        plotting.

        Args:
            other_obstacles (list[Self]): If provided, a list
                of other obstacles whose scattered potentials 
                are incident upon this obstacle.
            *cmap (Colormap): The Matplotlib colormap object 
                to use for plotting (cm.coolwarm is used if none
                is provided)
        """
        psi_tot = self.psi_vals
        
        if other_obstacles is not None:
            for obstacle in other_obstacles:
                psi_tot += obstacle.get_scattered_wave_at_obstacle(self, boundary_only=False)[1]

        # Plot contour with given color/colormap args
        X_global, Y_global = self.grid.local_coords_to_global_XY()
        psi_abs = np.abs(psi_tot)
        self._plot_periodic_contourf(X_global, Y_global, psi_abs, **kwargs)


    def get_scattered_wave_at_obstacle(
        self, 
        obstacle: Self,
        boundary_only: bool = True
    ) -> tuple[Grid, Grid]:
        """Gets this obstacle's scattered wave potential (phi/psi)
        values at the gridpoints around another given obstacle.

        If desired, only gets scatterd wave values at 
        3 "radial" (axis=1) rows of gridpoints of the given
        obstacle's grid corresponding to the gridpoints closest
        to the physical boundary of this given obstacle.

        Uses Karp's Farfield expansion to approximate these values 
        from the angular coefficients at this obstacle (assumes 
        that all gridpoints are outside of the computational
        domain of this obstacle)

        Args:
            obstacle (MKFE_FDObstacle): The obstacle we'd like to
                get the scattered potentials at 
            boundary_only (bool): If true, only gets potentials at 3
                "radial" (axis=1) rings of gridpoints around the 
                obstacle's physical boundary. If False, gets 
                values at every gridpoint in the provided obstacle's
                grid (defaults to True).
        
        Returns:
            tuple[Grid, Grid]: The two scattered potentials at each 
                of the desired gridpoints (in a shape matching the 
                provided obstacle's grid). Returned in the order
                (phi, psi).
        """
        # Store obstacle boundary info if not already stored
        self._cache_obstacle_boundary_data(obstacle)

        ## Get the obstacle boundary info
        interpolator: PeriodicInterpolator1D = self.obstacle_boundary_info[obstacle.id]['interpolator']
        R_local: np.ndarray = self.obstacle_boundary_info[obstacle.id]['R_local']
        Theta_local: np.ndarray = self.obstacle_boundary_info[obstacle.id]['Theta_local']

        # Filter out only boundary rows if desired
        if boundary_only:
            R_local = R_local[:3,:]
            Theta_local = Theta_local[:3,:]
        
        ## Get farfield coefficients at each of the boundary gridpoints 
        # F_l^p
        interpolator.update_func_vals(self.farfield_fp_coeffs.T)  # Have to transpose so -1 axis becomes 0 axis
        fp_vals = interpolator.interpolate(Theta_local)

        # G_l^p
        interpolator.update_func_vals(self.farfield_gp_coeffs.T)
        gp_vals = interpolator.interpolate(Theta_local)

        # F_l^s 
        interpolator.update_func_vals(self.farfield_fs_coeffs.T)
        fs_vals = interpolator.interpolate(Theta_local)

        # G_l^s 
        interpolator.update_func_vals(self.farfield_gs_coeffs.T)
        gs_vals = interpolator.interpolate(Theta_local)

        ## Reconstruct phi and psi values at each gridpoint
        # Get Hankel function values
        hankels = self.obstacle_boundary_info[obstacle.id]['Hankels']
        phi_hankels_order_0 = hankels['kp_ord_0']
        phi_hankels_order_1 = hankels['kp_ord_1']
        psi_hankels_order_0 = hankels['ks_ord_0']
        psi_hankels_order_1 = hankels['ks_ord_1']

        # Filter out only boundary rows if desired
        if boundary_only:
            phi_hankels_order_0 = phi_hankels_order_0[:3,:]
            phi_hankels_order_1 = phi_hankels_order_1[:3,:]
            psi_hankels_order_0 = psi_hankels_order_0[:3,:]
            psi_hankels_order_1 = psi_hankels_order_1[:3,:]

        # Get (wavenumber * radius) powers
        radius_powers = self.obstacle_boundary_info[obstacle.id]['Radius_powers']
        phi_radius_powers = radius_powers['kp']
        psi_radius_powers = radius_powers['ks']

        # Filter out only boundary rows if desired
        if boundary_only:
            phi_radius_powers = phi_radius_powers[:3,:,:]
            psi_radius_powers = psi_radius_powers[:3,:,:]

        # Now, use Karp expansion to do what we want
        fp_sum = phi_hankels_order_0 * np.sum(fp_vals / phi_radius_powers, axis=-1)
        gp_sum = phi_hankels_order_1 * np.sum(gp_vals / phi_radius_powers, axis=-1)
        phi_approx = fp_sum + gp_sum 

        fs_sum = psi_hankels_order_0 * np.sum(fs_vals / psi_radius_powers, axis=-1)
        gs_sum = psi_hankels_order_1 * np.sum(gs_vals / psi_radius_powers, axis=-1)
        psi_approx = fs_sum + gs_sum
        
        # Return Karp expansion approximation for phi and psi 
        # at this obstacle's boundary (transposed now again to match
        # the original obstacle grid shape)
        return phi_approx.T, psi_approx.T 


    def get_scattered_wave(self, X = None, Y = None):
        # If argument missing, give scattered wave at obstacle gridpoints
        if X is None or Y is None:
            return self.phi_vals, self.psi_vals
        
        if np.any(
            np.sqrt(
                (X-self.center_global[0])**2 + (Y-self.center_global[1])**2
            ) < self.r_artificial_boundary
        ):
            raise ValueError("All points must be outside computatational domain of this obstacle")
        
        # Otherwise, get local-coordinate representation of X and Y
        R_local, Theta_local = self.grid.global_XY_to_local_coords(X, Y)

        # RECALL: With how polar grids are defined earlier,
        # axis 0 = theta points, axis 1 = r points
        R_local = R_local.T             # Now, radius is first axis, angle is second
        Theta_local = Theta_local.T     # Same here

        # Create way to interpolate farfield coefficients from this obstacle
        interpolator = PeriodicInterpolator1D((0, 2*np.pi), self.grid.angular_gridpts_artificial_boundary)

        # Evaluate all sorts of Hankel functions beforehand, since these are expensive
        kp_hankel0 = hankel1(0, self.parent_medium.kp * R_local)
        kp_hankel1 = hankel1(1, self.parent_medium.kp * R_local)
        ks_hankel0 = hankel1(0, self.parent_medium.ks * R_local) 
        ks_hankel1 = hankel1(1, self.parent_medium.ks * R_local)

        # Evaluate radii to all the powers needed with kp/ks multiplied 
        # The shapes of these arrays will be (num_farfield_terms, (obstacle_boundary_R shape))
        powers = np.arange(self.num_farfield_terms).reshape((-1,1,1))
        r_powers_kp = (self.parent_medium.kp * R_local) ** powers
        r_powers_ks = (self.parent_medium.ks * R_local) ** powers

        ## Get farfield coefficients at each of the boundary gridpoints 
        # F_l^p
        interpolator.update_func_vals(self.farfield_fp_coeffs.T)  # Have to transpose so -1 axis becomes 0 axis
        fp_vals = interpolator.interpolate(Theta_local)

        # G_l^p
        interpolator.update_func_vals(self.farfield_gp_coeffs.T)
        gp_vals = interpolator.interpolate(Theta_local)

        # F_l^s 
        interpolator.update_func_vals(self.farfield_fs_coeffs.T)
        fs_vals = interpolator.interpolate(Theta_local)

        # G_l^s 
        interpolator.update_func_vals(self.farfield_gs_coeffs.T)
        gs_vals = interpolator.interpolate(Theta_local)

        ## Now, use Karp expansion to do what we want
        fp_sum = kp_hankel0 * np.sum(fp_vals / r_powers_kp, axis=0)
        gp_sum = kp_hankel1 * np.sum(gp_vals / r_powers_kp, axis=0)
        phi_approx = fp_sum + gp_sum 

        fs_sum = ks_hankel0 * np.sum(fs_vals / r_powers_ks, axis=0)
        gs_sum = ks_hankel1 * np.sum(gs_vals / r_powers_ks, axis=0)
        psi_approx = fs_sum + gs_sum

        return phi_approx, psi_approx
    

class Circular_MKFE_FDObstacle(MKFE_FDObstacle):
    """A circular obstacle in an elastic multiple-scattering problem.

    Uses the Karp MKFE ABC at the artificial boundary to approximate
    behavior outside of the computational domain, and to model
    interactions between obstacles.

    Attributes:
        center_global (Coordinates): The global X/Y coordinates
            of the center of the obstacle
        r_obstacle (float): The radius of the (circular) obstacle 
            from self.center_global 
        r_artificial_boundary (float): The radius of the circular
            artifical boundary from self.center_global
        bc (BoundaryCondition): The boundary 
            condition at the physical boundary (hard or soft)
            NOTE: Currently, only a hard boundary condition is 
            supported
        num_farfield_terms (int): The number of terms to use 
            in the farfield expansion ABC
        wavelength (float): The wavelength of the incoming wave
        PPW (int): The points-per-wavelength for the grid to use 
            for this finite-difference scheme
        grid (BaseGrid): All finite-difference gridpoints at this
            obstacle
        fd_unknowns (Vector): A 1-dimensional vector of all
            finite-difference unknowns
        fd_matrix (SparseMatrix): A sparse matrix representing the 
            finite-difference scheme for this problem 
        fd_matrix_LU (SparseMatrixLUDecomp): A LU-decomposition
            object allowing quick solutions to finite-difference 
            updates
    """
    grid: FDLocalPolarGrid  # For type-hinting purposes

    def __init__(
        self,
        center: Coordinates,
        r_obstacle: float,
        r_artificial_boundary: float,
        boundary_condition: BoundaryCondition,
        num_farfield_terms: int,
        parent_medium: LinearElasticMedium,
        PPW: int,
    ):
        """Initializes an Circular_MKFE_FDObstacle instance.

        The obstacle is a circular obstacle with radius r_obstacle,
        centered at the given center point, with a circular
        artificial boundary with radius r_artifical_boundary.

        The constructor initializes a local polar grid around this 
        obstacle that is centered at the given center
        
        Args:
            center(tuple[float, float]): The center of the obstacle
                in global cartesian coordinates
            r_obstacle (float): The radius of the circular obstalce
                from the given center point
            r_artificial_boundary (float): The radius of the circular
                artificial boundary from the given center point 
            boundary_condition (BoundaryCondition): The boundary
                condition (either soft or hard) imposed at the
                physical boundary of this obstacle.
            num_farfield_terms (int): The number of terms to use 
                in the farfield expansion for the outgoing wave
                radiating from this obstacle
            parent_medium (LinearElasticMedium): The medium this
                obstacle is placed in; the medium in which scattered
                waves from this obstacle must pass through
            PPW (int): The points-per-wavelength for the grid to use 
                for this finite-difference scheme
            num_angular_gridpoints (int): The number of gridpoints to
                use in the angular (theta) grid direction
        """
        # Store circular geometry attributes
        self.r_obstacle = r_obstacle
        ks = parent_medium.ks
        self.num_angular_gridpoints = floor(PPW*(2*np.pi*r_obstacle)*ks/(2*np.pi))

        # Store lookup for other obstacles 
        self.obstacle_boundary_info = dict()

        # Store MKFE_FDObstacle attributes 
        super().__init__(
            center,
            r_artificial_boundary,
            boundary_condition,
            num_farfield_terms,
            parent_medium,
            PPW
        )

    @property
    def num_ghost_points_physical_boundary(self):
        return 1 
    
    @property
    def num_ghost_points_artificial_boundary(self):
        return 1


    def generate_grid(self) -> FDLocalPolarGrid:
        """Generate the local grid at this obstacle.

        Will use a polar grid due to circular geometry.
        
        Args:
            wavelength (float): The wavelength under consideration
            PPW (int): The number of points per wavelength
            num_theta_gridpoints (int): The number of gridpoints 
                to use in the angular direction.
        """
        # Parse number of radial gridpoints from PPW 
        length_radial_ray = self.r_artificial_boundary - self.r_obstacle 
        ks = self.parent_medium.ks
        num_r_gridpoints = ceil(self.PPW*(length_radial_ray)*ks/(2*np.pi))

        # Create grid with this number of radial gridpoints (and the 
        # provided number of angular gridpoints) and then 
        # return this to the user.
        return FDLocalPolarGrid(
            center=self.center_global,
            num_r_pts=num_r_gridpoints,
            num_theta_pts=self.num_angular_gridpoints,
            r_min=self.r_obstacle,
            r_max=self.r_artificial_boundary
        )
    

    def _get_incident_wave_displacement_data(
        self, 
        incident_wave: IncidentPlanePWave
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get incident wave displacement data in a local
        coordinate decomposition at each gridpoint of
        this obstacle's physical boundary.
        
        Args:
            incident_wave (IncidentPlanePWave): The incident wave
                whose influence we need to take into account at
                the physical boundary
        
        Returns:
            np.ndarray: An array representing the incident wave's
                local-r displacement at each gridpoint on the 
                physical boundary 
            np.ndarray: An array representing the incident wave's
                local-theta displacement at each gridpoint on 
                the physical boundary
        """
        # Parse needed angle/radius data
        bndry_r_global, bndry_theta_global = (
            self.grid.local_coords_to_global_polar(np.s_[:,0])
        )
        bndry_r_local = self.grid.r_local[:,0]
        bndry_theta_local = self.grid.theta_local[:,0]

        # Parse incident wave data
        angle_inc_global = incident_wave.phi
        k_inc_wave = incident_wave.k

        # Parse current obstacle data
        xc, yc = self.center_global

        # Get common constant for displacements (based on
        # incident wave values in global space)
        displacement_const = (
            1j * k_inc_wave * incident_wave(self.grid, np.s_[:,0])
        ) / bndry_r_global

        # Finally, get displacements in local radial and angular 
        # directions from the above data
        displacement_local_r = displacement_const * (
            bndry_r_local * np.cos(bndry_theta_global - angle_inc_global)
            + xc * np.cos(bndry_theta_local + bndry_theta_global - angle_inc_global)
            + yc * np.sin(bndry_theta_local + bndry_theta_global - angle_inc_global)
        )
        displacement_local_theta = displacement_const * (
            -bndry_r_local * np.sin(bndry_theta_global - angle_inc_global)
            - xc * np.sin(bndry_theta_local + bndry_theta_global - angle_inc_global)
            + yc * np.cos(bndry_theta_local + bndry_theta_global - angle_inc_global)
        )
        return displacement_local_r, displacement_local_theta
    

    def _get_other_obstacle_displacement_data(
        self,
        obstacle: MKFE_FDObstacle
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get scattered wave displacement data (from the 
        wave scattered from another obstacle) in a local
        coordinate decomposition at each gridpoint of
        this obstacle's physical boundary.
        
        Args:
            obstacle (MKFE_FDObstacle): The obstacle whose 
                scattered wave is incident upon this obstacle's
                physical boundary
        
        Returns:
            np.ndarray: An array representing the other obstacle's
                scattered wave's local-r displacement at each
                gridpoint on the physical boundary
            np.ndarray: An array representing the other obstacle's
                scattered wave's local-theta displacement at each
                gridpoint on the physical boundary
        """
        # Cache obstacle boundary info if not already seen
        self._cache_obstacle_boundary_data(obstacle)

        # Get the scattered wave from the provided obstacle at
        # this obstacle's boundary
        obstacle_boundary_vals = obstacle.get_scattered_wave_at_obstacle(self, boundary_only=True)
        phi_vals = obstacle_boundary_vals[0]
        psi_vals = obstacle_boundary_vals[1]
        boundary_r_vals = (
            self.obstacle_boundary_info[obstacle.id]['R_local'][0]
        )  # Only care about local radii of points on physical boundary (which is the [0,:] radial slice of the cached boundary data)
        
        # Take radial derivatives at physical boundary
        # using 2nd-order forward (one-sided) finite differences
        dphi_dr_local = (
            -3 * phi_vals[:,0] 
            + 4 * phi_vals[:,1] 
            - phi_vals[:,2]
        ) / (2 * self.grid.dr)
        dpsi_dr_local = (
            -3 * psi_vals[:,0] 
            + 4 * psi_vals[:,1] 
            - psi_vals[:,2]
        ) / (2 * self.grid.dr)

        # Take angular derivatives at physical boundary
        # using 2nd-order centered finite differences
        # NOTE: This code assumes varying theta coordinates are the 
        # coordinates that vary along axis=0 in these grids
        dphi_dtheta_local = (
            np.roll(phi_vals, -1, axis=0)
            - np.roll(phi_vals, 1, axis=0)
        ) / (2 * self.grid.dtheta)
        dphi_dtheta_local = dphi_dtheta_local[:,0] # Only care about points on physical boundary
        dpsi_dtheta_local = (
            np.roll(psi_vals, -1, axis=0)
            - np.roll(psi_vals, 1, axis=0)
        ) / (2 * self.grid.dtheta)
        dpsi_dtheta_local = dpsi_dtheta_local[:,0] # Only care about points on physical boundary

        # Finally, combine these as needed to get displacements 
        # using Helmholtz decomposition formula
        # TODO: DO WE NEED TO TAKE INTO ACCOUNT MORE COMPLICATED
        # GEOMETRY?
        displacement_local_r = (
            dphi_dr_local + dpsi_dtheta_local/boundary_r_vals
        )
        displacement_local_theta = (
            dphi_dtheta_local/boundary_r_vals - dpsi_dr_local
        )
        return displacement_local_r, displacement_local_theta
        

    def hard_bc_forcing_vector(
        self,
        obstacles: list[MKFE_FDObstacle],
        incident_wave: Optional[IncidentPlanePWave] = None
    ) -> np.ndarray:
        """Construct forcing vector entries for a hard physical
        boundary condition.
        
        Args:
            obstacles (list[MKFE_FDObstacle]): A list of obstacles
                whose influence we need to take into account 
            incident_wave (IncidentPlanePWave): The incident wave
                whose influence we need to take into account
        
        Returns:
            np.ndarray: The entries for the forcing vector
                corresponding to a hard BC at the physical boundary
        """
        displacement_local_r = np.zeros(self.grid.num_angular_gridpoints, dtype='complex128')
        displacement_local_theta = np.zeros(self.grid.num_angular_gridpoints, dtype='complex128')

        ## Process incident wave data
        if incident_wave is not None:
            inc_wave_displacement = self._get_incident_wave_displacement_data(incident_wave)
            displacement_local_r += inc_wave_displacement[0]
            displacement_local_theta += inc_wave_displacement[1]
        
        ## Process other obstacle incoming wave data
        # Get other obstacle influences on this obstacle using Karp expansions
        for obstacle in obstacles:
            obstacle_displacement = self._get_other_obstacle_displacement_data(obstacle)
            displacement_local_r += obstacle_displacement[0]
            displacement_local_theta += obstacle_displacement[1]

        # Recall: the focing vector is the negative of these
        # displacements (since we want total displacement = 0 at
        # the physical boundary)
        return np.hstack((-displacement_local_r, -displacement_local_theta))


    def soft_bc_forcing_vector(
        self,
        obstacles: list[MKFE_FDObstacle],
        incident_wave: Optional[IncidentPlanePWave] = None
    ) -> np.ndarray:
        """Construct forcing vector entries for a soft physical
        boundary condition.
        
        Args:
            obstacles (list[MKFE_FDObstacle]): A list of obstacles
                whose influence we need to take into account 
            incident_wave (IncidentPlanePWave): The incident wave
                whose influence we need to take into acount
        
        Returns:
            np.ndarray: The entries for the forcing vector
                corresponding to a soft BC at the physical boundary
        """
        raise NotImplementedError("Soft BC not yet implemented")
    

    def construct_forcing_vector(self, obstacles: list[MKFE_FDObstacle], incident_wave = None):
        # Get physical BC data for first 2*N_{theta} entries
        if self.bc is BoundaryCondition.HARD:
            physical_bc_data = self.hard_bc_forcing_vector(obstacles, incident_wave)
        else:
            physical_bc_data = self.soft_bc_forcing_vector(obstacles, incident_wave)

        # Get rest of forcing vector data 
        other_forcing_data = np.zeros(self.num_unknowns - 2*self.num_angular_gridpoints)
        
        # Construct and return total forcing vector
        return np.hstack((physical_bc_data, other_forcing_data))


    def _get_physical_BC_rows(self) -> list[sparse.csc_array]:
        """Gets rows/equations in the finite-difference matrix
        corresponding to the physical boundary condition at the
        obstacle boundary.

        Returns:
            list[sparse.csc_array]: A list of block rows 
                of the finite-difference matrix corresponding 
                to these equations (each element is a sparse array 
                that is short and fat,
                spanning the entire length of the finite-difference
                matrix, but only with the 2*N_{theta} equations
                for this boundary condition as rows).
        """
        # Parse needed constants 
        r0 = self.r_obstacle
        dtheta = self.grid.dtheta

        # Create FD submatrices
        R_BC = sparse.eye_array(self.num_angular_gridpoints, format='csc') / (2 * self.grid.dr)
        T_BC = (
            sparse_periodic_tridiag(
                self.num_angular_gridpoints,
                0., -1., 1.
            ) / (2 * r0 * dtheta)
        )
        R_left = sparse.block_diag((-R_BC, R_BC))
        R_right = -R_left
        T_middle = sparse_block_antidiag([T_BC, T_BC])

        # Create block row for these 2*N_{theta} equations
        physical_bc_row_shape = (2 * self.num_angular_gridpoints, self.num_unknowns)
        num_zeros_right = self.num_unknowns - (6 * self.num_angular_gridpoints)
        blocks = [R_left, T_middle, R_right, num_zeros_right]
        physical_bc_rows = sparse_block_row(physical_bc_row_shape, blocks)
        return [physical_bc_rows]


    def _get_governing_system_rows(self) -> list[sparse.csc_array]:
        """Gets rows/equations in the finite-difference matrix
        corresponding to the governing system in the computational
        domain.

        Returns:
            list[sparse.csc_array]: A list of block rows 
                of the finite-difference matrix corresponding 
                to these equations (each element is a sparse array 
                that is short and fat,
                spanning the entire length of the finite-difference
                matrix, but only with the 2*N_{theta}*(N_{r})
                equations for this boundary condition as rows).
        """
        # Parse common constants 
        dr = self.grid.dr
        dtheta = self.grid.dtheta
        kp = self.parent_medium.kp
        ks = self.parent_medium.ks

        # Create rows
        fd_matrix_block_rows = []
        physical_system_row_shape = (2 * self.num_angular_gridpoints, self.num_unknowns)
        for i, r_i in enumerate(self.grid.r_vals):
            # Parse i-dependant constants
            alpha_i_pos = 1/(dr**2) + 1/(2 * dr * r_i)
            alpha_i_neg = 1/(dr**2) - 1/(2 * dr * r_i)
            alpha_i_p = kp**2 - (2/(dr**2)) - (2/(r_i**2 * dtheta**2))
            alpha_i_s = ks**2 - (2/(dr**2)) - (2/(r_i**2 * dtheta**2))
            beta_i = 1/(dtheta**2 * r_i**2)

            # Create FD submatrices 
            H_i_neg = alpha_i_neg * sparse.eye_array(2 * self.num_angular_gridpoints)
            H_i_pos = alpha_i_pos * sparse.eye_array(2 * self.num_angular_gridpoints)
            G_i_p = sparse_periodic_tridiag(self.num_angular_gridpoints, alpha_i_p, beta_i, beta_i)
            G_i_s = sparse_periodic_tridiag(self.num_angular_gridpoints, alpha_i_s, beta_i, beta_i)
            H_i_c = sparse.block_diag([G_i_p, G_i_s])

            # Create block row for these 2*N_{theta} equations
            num_zeros_left = i * (2 * self.num_angular_gridpoints)
            num_zeros_right = self.num_unknowns - num_zeros_left - (6 * self.num_angular_gridpoints)
            
            if num_zeros_left == 0:
                blocks = [H_i_neg, H_i_c, H_i_pos, num_zeros_right]
            else:
                blocks = [num_zeros_left, H_i_neg, H_i_c, H_i_pos, num_zeros_right]

            physical_system_rows = sparse_block_row(physical_system_row_shape, blocks)
            fd_matrix_block_rows.append(physical_system_rows)
        
        return fd_matrix_block_rows


    def _get_continuity_field_interface_rows(self) -> list[sparse.csc_array]:
        """Gets rows/equations in the finite-difference matrix
        corresponding to the field continuity interface condition
        at the artificial boundary.

        Returns:
            list[sparse.csc_array]: A list of block rows 
                of the finite-difference matrix corresponding 
                to these equations (each element is a sparse array 
                that is short and fat,
                spanning the entire length of the finite-difference
                matrix, but only with the 2*N_{theta}
                equations for this boundary condition as rows).
        """
        fd_matrix_block_rows = []    # For storing output equation rows

        ## Parse constants 
        kp = self.parent_medium.kp
        ks = self.parent_medium.ks
        hankel_ord0_kp = hankel1(0, kp * self.r_artificial_boundary)
        hankel_ord1_kp = hankel1(1, kp * self.r_artificial_boundary)
        hankel_ord0_ks = hankel1(0, ks * self.r_artificial_boundary)
        hankel_ord1_ks = hankel1(1, ks * self.r_artificial_boundary)
        interface_block_row_shape = (self.num_angular_gridpoints, self.num_unknowns)
        num_r_gridpts = self.grid.num_radial_gridpoints

        ## Create repetitive arays 
        I_N_theta = sparse.eye_array(self.num_angular_gridpoints, format='csc')
        powers = np.arange(self.num_farfield_terms) # l = 0, 1, ..., L-1
        
        ## C.1.A - Continuity of Phi (N_{theta} equations)
        # Create needed constants/arrays of constants specific to phi
        kp_powers = (kp * self.r_artificial_boundary)**powers
        J_p = hankel_ord0_kp/kp_powers  # TODO: SCALING VERY BAD HERE 
        K_p = hankel_ord1_kp/kp_powers
        # J_p = hankel_ord0_kp/(powers + 1)
        # K_p = hankel_ord1_kp/(powers + 1)

        # Create needed FD matrices
        M_J_p = []      # List of sparse arrays
        M_K_p = []      # List of sparse arrays
        for J_l_p, K_l_p in zip(J_p, K_p):
            M_J_l_p = J_l_p * I_N_theta
            M_K_l_p = K_l_p * I_N_theta
            M_J_p.append(M_J_l_p)
            M_K_p.append(M_K_l_p)

        # Create block rows 
        zeros_left = 2 * self.num_angular_gridpoints * num_r_gridpts
        zeros_between = 3 * self.num_angular_gridpoints
        first_part_of_row = [zeros_left, -I_N_theta, zeros_between]
        zeros_end = 2 * self.num_farfield_terms * self.num_angular_gridpoints

        blocks = first_part_of_row + M_J_p + M_K_p + [zeros_end]
        phi_continuity_block_rows = sparse_block_row(
            interface_block_row_shape,
            blocks
        )
        fd_matrix_block_rows.append(phi_continuity_block_rows)

        ## C.1.B - Continuity of Psi
        # Create needed constants/arrays of constants specific to psi
        ks_powers = (ks * self.r_artificial_boundary)**powers  # l=0,1,...,L-1
        J_s = hankel_ord0_ks/ks_powers
        K_s = hankel_ord1_ks/ks_powers
        # J_s = hankel_ord0_ks/(powers + 1)
        # K_s = hankel_ord1_ks/(powers + 1)

        # Create needed FD matrices
        M_J_s = []      # List of sparse arrays
        M_K_s = []      # List of sparse arrays
        for J_l_s, K_l_s in zip(J_s, K_s):
            M_J_l_s = J_l_s * I_N_theta
            M_K_l_s = K_l_s * I_N_theta
            M_J_s.append(M_J_l_s)
            M_K_s.append(M_K_l_s)

        # Create block rows
        zeros_left = (2 * self.num_angular_gridpoints * num_r_gridpts) + self.num_angular_gridpoints
        zeros_between = (2 * self.num_angular_gridpoints) + (2 * self.num_farfield_terms * self.num_angular_gridpoints)
        first_part_of_row = [zeros_left, -I_N_theta, zeros_between]

        blocks = first_part_of_row + M_J_s + M_K_s
        psi_continuity_block_rows = sparse_block_row(
            interface_block_row_shape,
            blocks
        )
        fd_matrix_block_rows.append(psi_continuity_block_rows)

        return fd_matrix_block_rows


    def _get_continuity_1st_radial_derivative_interface_rows(self) -> list[sparse.csc_array]:
        """Gets rows/equations in the finite-difference matrix
        corresponding to the interface condition of continuity of
        the first radial derivative at the artificial boundary.

        Returns:
            list[sparse.csc_array]: A list of block rows 
                of the finite-difference matrix corresponding 
                to these equations (each element is a sparse array 
                that is short and fat,
                spanning the entire length of the finite-difference
                matrix, but only with the 2*N_{theta}
                equations for this boundary condition as rows).
        """
        fd_matrix_block_rows = []    # For storing output equation rows

        # Parse constants 
        kp = self.parent_medium.kp
        ks = self.parent_medium.ks
        dr = self.grid.dr
        hankel_ord0_kp = hankel1(0, kp * self.r_artificial_boundary)
        hankel_ord1_kp = hankel1(1, kp * self.r_artificial_boundary)
        hankel_ord0_ks = hankel1(0, ks * self.r_artificial_boundary)
        hankel_ord1_ks = hankel1(1, ks * self.r_artificial_boundary)
        interface_block_row_shape = (self.num_angular_gridpoints, self.num_unknowns)
        num_r_gridpts = self.grid.num_radial_gridpoints

        z_plus = 1/(2 * dr)
        z_minus = -1/(2 * dr)

        # Create repetitive arrays 
        I_N_theta = sparse.eye_array(self.num_angular_gridpoints, format='csc')

        powers_1der = np.arange(self.num_farfield_terms)    # l=0,1,...,L-1
        powers_1der_plus1 = powers_1der + 1                 # l+1, where l=0,1,...,L-1

        Z_plus = z_plus * I_N_theta
        Z_minus = z_minus * I_N_theta

        ## C.2.A Continuity of First Radial Derivative of phi
        # Create needed constants/arrays of constants specific to phi
        kp_powers_1der = (kp * self.r_artificial_boundary)**powers_1der
        kp_powers_1der_plus1 = (kp * self.r_artificial_boundary)**powers_1der_plus1
        A_p = (
            (-kp * hankel_ord1_kp / kp_powers_1der) 
            - (kp * powers_1der * hankel_ord0_kp / kp_powers_1der_plus1)
        )
        B_p = (
            (-kp * powers_1der_plus1 * hankel_ord1_kp / kp_powers_1der_plus1)
            + (kp * hankel_ord0_kp / kp_powers_1der)
        )       # TODO: VERY BAD SCALING HERE 

        # Create FD matrices
        M_A_p = []
        M_B_p = []
        for A_l_p, B_l_p in zip(A_p, B_p):
            M_A_l_p = A_l_p * I_N_theta
            M_B_l_p = B_l_p * I_N_theta
            M_A_p.append(M_A_l_p)
            M_B_p.append(M_B_l_p)

        # Create block rows
        num_zeros_left = 2 * self.num_angular_gridpoints * (num_r_gridpts - 1)
        num_zeros_between_1 = 3 * self.num_angular_gridpoints
        num_zeros_between_2 = self.num_angular_gridpoints
        num_zeros_right = 2 * self.num_farfield_terms * self.num_angular_gridpoints
        
        blocks = (
            [num_zeros_left, Z_plus, num_zeros_between_1, Z_minus, num_zeros_between_2]
            + M_A_p 
            + M_B_p
            + [num_zeros_right]
        )
        phi_1der_continuity_block_rows = sparse_block_row(
            interface_block_row_shape,
            blocks
        )
        fd_matrix_block_rows.append(phi_1der_continuity_block_rows)

        ## C.2.B Continuity of First Radial Derivative of psi
        # Create needed constants/arrays of constants specific to psi
        ks_powers_1der = (ks * self.r_artificial_boundary)**powers_1der
        ks_powers_1der_plus1 = (ks * self.r_artificial_boundary)**powers_1der_plus1

        A_s = (
            (-ks * hankel_ord1_ks / ks_powers_1der) 
            - (ks * powers_1der * hankel_ord0_ks / ks_powers_1der_plus1)
        )
        B_s = (
            (-ks * powers_1der_plus1 * hankel_ord1_ks / ks_powers_1der_plus1)
            + (ks * hankel_ord0_ks / ks_powers_1der)
        )

        # Create FD matrices
        M_A_s = []
        M_B_s = []
        for A_l_s, B_l_s in zip(A_s, B_s):
            M_A_l_s = A_l_s * I_N_theta
            M_B_l_s = B_l_s * I_N_theta
            M_A_s.append(M_A_l_s)
            M_B_s.append(M_B_l_s)

        # Create block rows
        num_zeros_left = 2 * self.num_angular_gridpoints * (num_r_gridpts - 1) + self.num_angular_gridpoints
        num_zeros_between_1 = 3 * self.num_angular_gridpoints
        num_zeros_between_2 = 2 * self.num_farfield_terms * self.num_angular_gridpoints
        blocks = (
            [num_zeros_left, Z_plus, num_zeros_between_1, Z_minus, num_zeros_between_2]
            + M_A_s 
            + M_B_s
        )
        psi_1der_continuity_block_rows = sparse_block_row(
            interface_block_row_shape,
            blocks
        )
        fd_matrix_block_rows.append(psi_1der_continuity_block_rows)
        
        return fd_matrix_block_rows


    def _get_continuity_2nd_radial_derivative_interface_rows(self) -> list[sparse.csc_array]:
        """Gets rows/equations in the finite-difference matrix
        corresponding to the interface condition of continuity of
        the first radial derivative at the artificial boundary.

        Returns:
            list[sparse.csc_array]: A list of block rows 
                of the finite-difference matrix corresponding 
                to these equations (each element is a sparse array 
                that is short and fat,
                spanning the entire length of the finite-difference
                matrix, but only with the 2*N_{theta}
                equations for this boundary condition as rows).
        """
        fd_matrix_block_rows = []    # For storing output equation rows

        # Parse constants 
        kp = self.parent_medium.kp
        ks = self.parent_medium.ks
        dr = self.grid.dr
        hankel_ord0_kp = hankel1(0, kp * self.r_artificial_boundary)
        hankel_ord1_kp = hankel1(1, kp * self.r_artificial_boundary)
        hankel_ord0_ks = hankel1(0, ks * self.r_artificial_boundary)
        hankel_ord1_ks = hankel1(1, ks * self.r_artificial_boundary)
        interface_block_row_shape = (self.num_angular_gridpoints, self.num_unknowns)
        num_r_gridpts = self.grid.num_radial_gridpoints

        q_minus = -1/(dr**2)
        q_plus = 2/(dr**2)

        ## Create repetitive arays 
        I_N_theta = sparse.eye_array(self.num_angular_gridpoints, format='csc')

        powers_2der = np.arange(self.num_farfield_terms)    # l = 0, 1, ..., L-1
        powers_2der_plus1 = powers_2der + 1                 # l+1, where l = 0, 1, ..., L-1
        powers_2der_plus2 = powers_2der + 2                 # l+2, where l = 0, 1, ..., L-1

        Q_plus = q_plus * I_N_theta
        Q_minus = q_minus * I_N_theta

        ## C.3.A Continuity of Second Radial Derivative of phi
        # Parse needed constants/arrays of constants specific to phi
        kp_powers_2der = (kp * self.r_artificial_boundary)**powers_2der
        kp_powers_2der_plus1 = (kp * self.r_artificial_boundary)**powers_2der_plus1
        kp_powers_2der_plus2 = (kp * self.r_artificial_boundary)**powers_2der_plus2

        C_p = (
            -(kp**2 * hankel_ord0_kp)/kp_powers_2der
            + (kp**2 * (2 * powers_2der + 1) * hankel_ord1_kp)/kp_powers_2der_plus1
            + (kp**2 * powers_2der * powers_2der_plus1 * hankel_ord0_kp)/kp_powers_2der_plus2
        ) 
        D_p = (
            -(kp**2 * hankel_ord1_kp)/kp_powers_2der
            -(kp**2 * (2 * powers_2der + 1) * hankel_ord0_kp)/kp_powers_2der_plus1
            + (kp**2 * powers_2der_plus1 * powers_2der_plus2 * hankel_ord1_kp)/kp_powers_2der_plus2
        ) # TODO: SHOULD THE FIRST SUBTRACTION BE AN ADDITION??? ACCORDING TO DANE, YES. MY DERIVATION SAYS NO.
        
        # TODO: VERY BAD SCALING HERE 
        
        # C_p = (
        #     -(kp**2 * hankel_ord0_kp)/powers_2der
        #     + (kp**2 * (2 * powers_2der + 1) * hankel_ord1_kp)
        #     + (kp**2 * powers_2der * powers_2der_plus1 * hankel_ord0_kp)
        # ) 
        # D_p = (
        #     -(kp**2 * hankel_ord1_kp)/powers_2der
        #     - (kp**2 * (2 * powers_2der + 1) * hankel_ord0_kp)
        #     + (kp**2 * powers_2der_plus1 * powers_2der_plus2 * hankel_ord1_kp)
        # ) 

        # Create FD matrices 
        M_C_p = []
        M_D_p = []
        for C_l_p, D_l_p in zip(C_p, D_p):
            M_C_l_p = C_l_p * I_N_theta
            M_D_l_p = D_l_p * I_N_theta
            M_C_p.append(M_C_l_p)
            M_D_p.append(M_D_l_p)

        # Create block rows 
        num_zeros_left = 2 * self.num_angular_gridpoints * (num_r_gridpts - 1)
        num_zeros_between_stuff = self.num_angular_gridpoints
        num_zeros_right = 2 * self.num_farfield_terms * self.num_angular_gridpoints
        blocks = (
            [num_zeros_left, Q_minus, num_zeros_between_stuff, Q_plus]
            + [num_zeros_between_stuff, Q_minus, num_zeros_between_stuff]
            + M_C_p 
            + M_D_p
            + [num_zeros_right]
        )
        phi_2der_continuity_block_rows = sparse_block_row(
            interface_block_row_shape,
            blocks
        )
        fd_matrix_block_rows.append(phi_2der_continuity_block_rows)

        ## C.3.B Continuity of Second Radial Derivative of psi
        # Parse needed constants/arrays of constants specific to psi
        ks_powers_2der = (ks * self.r_artificial_boundary)**powers_2der
        ks_powers_2der_plus1 = (ks * self.r_artificial_boundary)**powers_2der_plus1
        ks_powers_2der_plus2 = (ks * self.r_artificial_boundary)**powers_2der_plus2

        C_s = (
            -(ks**2 * hankel_ord0_ks)/ks_powers_2der
            + (ks**2 * (2 * powers_2der + 1) * hankel_ord1_ks)/ks_powers_2der_plus1
            + (ks**2 * powers_2der * powers_2der_plus1 * hankel_ord0_ks)/ks_powers_2der_plus2
        ) 
        D_s = (
            -(ks**2 * hankel_ord1_ks)/ks_powers_2der
            -(ks**2 * (2 * powers_2der + 1) * hankel_ord0_ks)/ks_powers_2der_plus1
            + (ks**2 * powers_2der_plus1 * powers_2der_plus2 * hankel_ord1_ks)/ks_powers_2der_plus2
        ) # TODO: SHOULD THE FIRST SUBTRACTION BE AN ADDITION??? ACCORDING TO DANE, YES. MY DERIVATION SAYS NO.
        # C_s = (
        #     -(ks**2 * hankel_ord0_ks)/powers_2der
        #     + (ks**2 * (2 * powers_2der + 1) * hankel_ord1_ks)
        #     + (ks**2 * powers_2der * powers_2der_plus1 * hankel_ord0_ks)
        # ) 
        # D_s = (
        #     -(ks**2 * hankel_ord1_ks)/powers_2der
        #     + (ks**2 * (2 * powers_2der + 1) * hankel_ord0_ks)
        #     + (ks**2 * powers_2der_plus1 * powers_2der_plus2 * hankel_ord1_ks)
        # )

        M_C_s = []
        M_D_s = []
        for C_l_s, D_l_s in zip(C_s, D_s):
            M_C_l_s = C_l_s * I_N_theta
            M_D_l_s = D_l_s * I_N_theta
            M_C_s.append(M_C_l_s)
            M_D_s.append(M_D_l_s)

        # Create block rows 
        num_zeros_left = 2 * self.num_angular_gridpoints * (num_r_gridpts - 1) + self.num_angular_gridpoints
        num_zeros_between_stuff = self.num_angular_gridpoints
        num_zeros_between_bigger = 2 * self.num_farfield_terms * self.num_angular_gridpoints
        blocks = (
            [num_zeros_left, Q_minus, num_zeros_between_stuff, Q_plus]
            + [num_zeros_between_stuff, Q_minus, num_zeros_between_bigger]
            + M_C_s 
            + M_D_s
        )
        psi_2der_continuity_block_rows = sparse_block_row(
            interface_block_row_shape,
            blocks
        )
        fd_matrix_block_rows.append(psi_2der_continuity_block_rows)

        return fd_matrix_block_rows


    def _get_art_bndry_interface_rows(self) -> list[sparse.csc_array]:
        """Gets rows/equations in the finite-difference matrix
        corresponding to the interface conditions at the artificial
        boundary.

        Returns:
            list[sparse.csc_array]: A list of block rows 
                of the finite-difference matrix corresponding 
                to these equations (each element is a sparse array 
                that is short and fat,
                spanning the entire length of the finite-difference
                matrix, but only with the 2*N_{theta}*(3)
                equations for this boundary condition as rows).
        """
        fd_matrix_block_rows = []    # For storing output equation rows
        
        # Get rows/equations for continuity of field 
        continuity_of_field_rows = self._get_continuity_field_interface_rows()
        fd_matrix_block_rows.extend(continuity_of_field_rows)

        # Get rows/equations for continuity of 1st radial derivative 
        first_deriv_continuity_rows = self._get_continuity_1st_radial_derivative_interface_rows()
        fd_matrix_block_rows.extend(first_deriv_continuity_rows)

        # Get rows/equations for continuity of 2nd radial derivative
        second_deriv_continuity_rows = self._get_continuity_2nd_radial_derivative_interface_rows()
        fd_matrix_block_rows.extend(second_deriv_continuity_rows)

        return fd_matrix_block_rows


    def _get_recursion_relation_rows(self) -> list[sparse.csc_array]:
        """Gets rows/equations in the finite-difference matrix
        corresponding to the recursive relationships between 
        angular coefficients.

        Returns:
            list[sparse.csc_array]: A list of block rows 
                of the finite-difference matrix corresponding 
                to these equations (each element is a sparse array 
                that is short and fat,
                spanning the entire length of the finite-difference
                matrix, but only with the 4 * (L * N_{\theta})
                equations for this boundary condition as rows).
        """
        fd_matrix_block_rows = []    # For storing output equation rows

        # Parse needed constants and arrays of constants
        dtheta = self.grid.dtheta 
        l_idxs = np.arange(1, self.num_farfield_terms)
        t_plus = 1/(dtheta**2)
        t_F = -2/(dtheta**2) + (l_idxs-1)**2
        t_G = -2/(dtheta*82) + l_idxs**2 
        s_F = 2 * l_idxs
        s_G = -2 * l_idxs
        recursive_relation_block_shape = (2 * self.num_angular_gridpoints, self.num_unknowns)
        num_r_gridpts = self.grid.num_radial_gridpoints

        # Create repetitive arrays 
        I_N_Theta = sparse.eye_array(self.num_angular_gridpoints, format='csc')

        # Iterate through all the l index values to create appropriate rows
        for i, (t_l_F, t_l_G, s_l_F, s_l_G) in enumerate(zip(t_F, t_G, s_F, s_G)):
            # Create smaller FD subarrays
            T_l_F = sparse_periodic_tridiag(self.num_angular_gridpoints, t_l_F, t_plus, t_plus)
            T_l_G = sparse_periodic_tridiag(self.num_angular_gridpoints, t_l_G, t_plus, t_plus)
            S_l_F = s_l_F * I_N_Theta
            S_l_G = s_l_G * I_N_Theta 

            # Combine smaller subarrays to create bigger subarrays 
            A_rec_l = sparse.block_diag([T_l_F, S_l_G], format='csc')
            Z_rec_l = sparse_block_antidiag([T_l_G, S_l_F])

            ## D.1-D.2: p-recursive relations
            # Get block rows
            num_zeros_left = (
                2 * self.num_angular_gridpoints * (
                    num_r_gridpts
                    + self.num_ghost_points_artificial_boundary 
                    + self.num_ghost_points_physical_boundary
                ) + (i * self.num_angular_gridpoints)
            )
            num_zeros_middle = (self.num_farfield_terms - 2) * self.num_angular_gridpoints
            num_zeros_right = self.num_unknowns - (
                num_zeros_left + num_zeros_middle + 4 * self.num_angular_gridpoints
            )

            if num_zeros_right == 0:
                block_data = (
                    [num_zeros_left, A_rec_l, num_zeros_middle, Z_rec_l]
                )
            else:
                block_data = (
                    [num_zeros_left, A_rec_l, num_zeros_middle, Z_rec_l, num_zeros_right]
                )

            block_rows = sparse_block_row(
                recursive_relation_block_shape,
                block_data
            )
            fd_matrix_block_rows.append(block_rows)

            ## D.3-D.4: s-recursive relations
            # Get block rows
            num_zeros_left = (
                2 * self.num_angular_gridpoints * (
                    num_r_gridpts 
                    + self.num_ghost_points_artificial_boundary 
                    + self.num_ghost_points_physical_boundary
                ) 
                + (2 * self.num_farfield_terms * self.num_angular_gridpoints)
                + (i * self.num_angular_gridpoints)
            )
            num_zeros_middle = (self.num_farfield_terms - 2) * self.num_angular_gridpoints
            num_zeros_right = self.num_unknowns - (
                num_zeros_left + num_zeros_middle + 4 * self.num_angular_gridpoints
            )
            if num_zeros_right == 0:
                block_data = (
                    [num_zeros_left, A_rec_l, num_zeros_middle, Z_rec_l]
                )
            else:
                block_data = (
                    [num_zeros_left, A_rec_l, num_zeros_middle, Z_rec_l, num_zeros_right]
                )

            block_rows = sparse_block_row(
                recursive_relation_block_shape,
                block_data
            )
            fd_matrix_block_rows.append(block_rows)
        
        return fd_matrix_block_rows


    def construct_fd_matrix(self):
        fd_matrix_block_rows = []     # Keeps track of block rows
    
        # A. Physical BC rows 
        physical_bc_rows = self._get_physical_BC_rows()
        fd_matrix_block_rows.extend(physical_bc_rows)

        # B. Rows for Governing System in Computational Domain
        governing_system_rows = self._get_governing_system_rows()
        fd_matrix_block_rows.extend(governing_system_rows)

        # C. Rows for Interface Conditions at Artificial Boundary
        interface_condition_rows = self._get_art_bndry_interface_rows()
        fd_matrix_block_rows.extend(interface_condition_rows)
        
        # D. Rows for Recursive Relations between Angular Coefficients
        recursive_relation_rows = self._get_recursion_relation_rows()
        fd_matrix_block_rows.extend(recursive_relation_rows)

        # Get the full finite-difference matrix by vertically
        # stacking A-D.
        fd_matrix = sparse.vstack(fd_matrix_block_rows, format='csc')
        return fd_matrix
    
    def plot_fd_matrix(self, **kwargs):
        # Actually plot stuff
        super().plot_fd_matrix(**kwargs)
        
        ## NOW, add gridlines

        # Minor gridlines always revolve around numbers angular coefficients 
        minor_gridlines = np.arange(0, self.num_unknowns, self.num_angular_gridpoints)

        # Y major gridlines are always around which equations are which:
        # Boundary equations, physical system equations, interface equations, and recurrance equations
        num_boundary_equations = 2 * self.num_angular_gridpoints
        num_physical_equations = 2 * self.num_angular_gridpoints * self.grid.num_radial_gridpoints
        num_interface_equations = 6 * self.num_angular_gridpoints
        num_recurrance_equations = 4 * (self.num_farfield_terms - 1) * self.num_angular_gridpoints
        y_major_gridlines = np.array([
            0,
            num_boundary_equations,
            num_boundary_equations + num_physical_equations,
            num_boundary_equations + num_physical_equations + num_interface_equations,
            num_boundary_equations + num_physical_equations + num_interface_equations + num_recurrance_equations
        ])

                                
        # Get X major gridlines for phi/psi unknowns and ghost points (separating each radial level)
        num_physical_unknowns = 2*(self.num_angular_gridpoints * (self.grid.num_radial_gridpoints + 2))
        physical_major_xgridlines = np.arange(0, num_physical_unknowns, 2 * self.num_angular_gridpoints)

        # Get X major gridlines for farfield coefficients (separating each F/G and p/s)
        farfield_major_xgridlines = np.arange(num_physical_unknowns, self.num_unknowns, self.num_angular_gridpoints * self.num_farfield_terms)

        # Combine these to get major X gridlines over entire domain 
        x_major_gridlines = np.array(list(physical_major_xgridlines) + list(farfield_major_xgridlines))
        
        plt.xticks(x_major_gridlines, minor=False, rotation=90)
        plt.xticks(minor_gridlines, minor=True, rotation=90)
        plt.yticks(y_major_gridlines, minor=False)
        plt.yticks(minor_gridlines, minor=True)
        plt.tick_params(axis='both', which='major', width=1, labelsize=5)
        plt.tick_params(axis='both', which='minor', width=0.5)
        plt.grid(which='major', linewidth=1.5)
        plt.grid(which='minor', linewidth=0.5)
        plt.ylabel("Equation Number")
        plt.xlabel("Unknown Number")



class CircularObstacleGeometry:
    """A basic object to hold a circular obstacle's geometry
    and material attributes
    
    Attributes:
        center (Coordinates): The center of this obstacle (in global
            Cartesian coordinates)
        r_obstacle (float): The radius of the circular obstacle
            from its center point 
        r_artificial_bndry (float): The radius of the artificial
            boundary of this obstacle's computational domain from 
            its center point 
        bc (BoundaryCondition): The boundary condition to apply
            at the physical boundary
    """
    def __init__(
        self, 
        center: Coordinates,
        r_obstacle: float,
        r_artificial_bndry: float,
        bc: BoundaryCondition
    ):
        self.center = center 
        self.r_obstacle = r_obstacle 
        self.r_artificial_bndry = r_artificial_bndry
        self.bc = bc
    
    @classmethod 
    def from_config_entry(cls, entry: dict) -> Self:
        """Create and return a circular obstacle geometry 
        from a corresponding entry in an appropriately-formatted
        JSON config file.
        
        Args:
            entry (dict): The entry from the config file to
                create this obstacle from

        Returns:
            CircularObstacleGeometry: The corresponding geometry
                for this obstacle
        """
        # Parse numerical elements 
        center = tuple(entry["center"])
        r_obstacle = float(entry["r_obstacle"])
        r_artificial_boundary = float(entry["r_artificial_boundary"])
        
        # Parse boundary condition from string 
        bc_written = entry["bc"].upper().strip()
        if bc_written == "HARD":
            bc = BoundaryCondition.HARD 
        elif bc_written == "SOFT":
            bc = BoundaryCondition.SOFT
        else:
            raise ValueError(f"Cannot parse bc = \"{bc_written}\" from config file")
        
        # Return propertly formatted object 
        return cls(center, r_obstacle, r_artificial_boundary, bc)
    

    






if __name__ == "__main__":
    # CODE FOR TESTING SETTING UP AND PLOTTING OBSTACLES
    medium = LinearElasticMedium.from_lame_constants(0.5, 1.3, 1.5, 2.5)

    # Obstacle 1
    center = (1.5, 1.5)
    r_obstacle = 1
    r_artificial_boundary = 2
    bc = BoundaryCondition.HARD
    num_farfield_terms=15
    PPW=3
    num_angular_gridpoints=30
    obs1 = Circular_MKFE_FDObstacle(
        center,
        r_obstacle,
        r_artificial_boundary,
        bc,
        num_farfield_terms,
        medium,
        PPW,
        num_angular_gridpoints
    )

    # Obstacle 2
    center = (-3.5, -2.5)
    r_obstacle = 1.5
    r_artificial_boundary = 3
    bc = BoundaryCondition.HARD
    num_farfield_terms=15
    PPW=3
    num_angular_gridpoints=30
    obs2 = Circular_MKFE_FDObstacle(
        center,
        r_obstacle,
        r_artificial_boundary,
        bc,
        num_farfield_terms,
        medium,
        PPW,
        num_angular_gridpoints
    )

    # Obstacle 3 
    center = (3.5, -2.0)
    r_obstacle = 0.5
    r_artificial_boundary = 2
    bc = BoundaryCondition.HARD
    num_farfield_terms=15
    PPW=3
    num_angular_gridpoints=30
    obs3 = Circular_MKFE_FDObstacle(
        center,
        r_obstacle,
        r_artificial_boundary,
        bc,
        num_farfield_terms,
        medium,
        PPW,
        num_angular_gridpoints
    )

    # Plot and show both obstacles 
    obs1.plot_grid(color='red')
    obs2.plot_grid(color='blue')
    obs3.plot_grid(color='purple')
    plt.axhline(0, color='black') # x = 0 axis line
    plt.axvline(0, color='black') # y = 0 axis line
    plt.show()