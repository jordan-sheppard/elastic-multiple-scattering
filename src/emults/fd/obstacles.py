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
from matplotlib.contour import QuadContourSet
import logging

from .utils import sparse_periodic_tridiag, sparse_block_row, sparse_block_antidiag
from ..base.consts import (
    Coordinates, BoundaryCondition, SparseMatrix,
    Vector, Grid, QOI, CoordinateSystem
)
from ..base.obstacles import BaseObstacle
from ..base.waves import IncidentPlanePWave
from ..base.medium import LinearElasticMedium
from .grids import FDLocalPolarGrid, FDPolarGrid_ArtBndry
from .coefficients import (
    ElasticFarfieldEvaluator, ElasticPolarFarfieldEvaluator, 
    FarfieldAngularCoefficients
)
from .waves import IncidentPlanePWaveEvaluator
from .solvers import FDSolver


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
        solver: FDSolver,
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
            solver (FDSolver): An object which has a .solve() method and
                returns the solution to the system Ax=F, where F 
                is the forcing vector from the other obstacles/incident
                wave, and A is the finite-difference matrix at this
                obstacle.
            obstacles (list[FDObstacle]) : A list of other obstacles 
                whose scattered waves are incident upon this obstacle
            incident_wave (IncidentPlaneWave): If provided, an
                incident plane wave
        """
        F = self.construct_forcing_vector(obstacles, incident_wave)
        self.fd_unknowns = solver.solve(F)
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
            at all physical gridpoints 
        * phi_vals_padded (Grid): The value of the compressional potential phi 
            at all physical AND ghost gridpoints 
        * psi_vals (Grid): The value of the shear potential psi
            at all gridpoints
        * psi_vals_padded (Grid): The value of the shear potential psi 
            at all physical AND ghost gridpoints 

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        """
        self.phi_vals, self.phi_vals_padded = self.parse_phi_from_FD_result(result)
        self.psi_vals, self.psi_vals_padded = self.parse_psi_from_FD_result(result)


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
    ) -> FDPolarGrid_ArtBndry:
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


    # @abstractmethod
    # def get_scattered_wave(
    #     self,
    #     X: Optional[Grid]=None,
    #     Y: Optional[Grid]=None,
    #     qoi: QOI=QOI.POTENTIALS
    # ) -> np.ndarray:
    #     """Retrive the value of the outgoing scattered potentials from
    #     this obstacle at the given global gridpoints.

    #     If either X or Y is None, then these inputs are ignored,
    #     and the value of the outgoing scattered wave is returned
    #     at the local gridpoints.

    #     Otherwise, all points (X,Y) should be lying outside of the
    #     computational domain of this obstacle.
        
    #     Args:
    #         X (Grid): The global X-coordinates (if None, retrieve
    #             scattered wave values at each local gridpoint)
    #         Y (Grid): The global Y-coordinates (if None, retrieve
    #             scattered wave values at each local gridpoint)
    #         qoi (QOI): The quantity of interest (either potentials,
    #             displacements, or stresses) to return. Defaults 
    #             to potentials
        
    #     Returns:
    #         np.ndarray: The values of the quantity of interest 
    #             (the -1 axis will be what differentiates the 
    #             different quantities of interest. For potentials,
    #             index 0 is phi and index 1 is psi. For displacements,
    #             index 0 is u_{r} and index 1 is u_{theta} (global 
    #             polar coordinates). For stresses, index 0 is 
    #             sigma_{rr} and index 1 is sigma_{r theta} (global
    #             polar coordinates).

    #     Raises:
    #         ValueError: If (X,Y) is within the computational 
    #             domain of this obstacle
    #     """
    #     pass


    @abstractmethod
    def get_scattered_wave_at_obstacle(
        self, 
        obstacle: Self,
        desired_quantity: QOI,
        boundary_only: bool = True
    ) -> tuple[Grid, Grid]:
        """Gets this obstacle's scattered wave displacement/stress
        at the provided obstacle's gridpoints.

        If desired, only gets scatterd wave values at 
        1 "radial" (axis=1) row of gridpoints of the given
        obstacle's grid corresponding to the gridpoints closest
        to the physical boundary of this given obstacle.

        Args:
            obstacle (MKFE_FDObstacle): The obstacle we'd like to
                evaluate this obstacle's scattered displacement
                or stress at
            desired_quantity (QOI): The desired quantity of interest
                (either stress, displacement, or potentials)
            boundary_only (bool): If true, only gets potentials at 
                the innermost "radial" (axis=1) ring of gridpoints
                around the obstacle's physical boundary. If False, gets 
                values at every gridpoint in the provided obstacle's
                grid. Defaults to True.
        
        Returns:
            np.ndarray: The quantity of interest evaluated at each of
                the gridpoints. (The -1 axis will be what differentiates the 
                different quantities of interest. For potentials,
                index 0 is phi and index 1 is psi. For displacements,
                index 0 is u_{r_mbar} and index 1 is u_{theta_mbar} 
                (the other obstacle's local polar coordinates).
                For stresses, index 0 is sigma_{r_mbar r_mbar}
                and index 1 is sigma_{r_mbar theta_mbar}
                (the other obstacle's local polar coordinates).
        """
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

        # Initialize lookup for other obstacle/incident wave interactions
        self.obstacle_farfield_evaluators: dict[int, ElasticPolarFarfieldEvaluator] = dict()
        self.incident_evaluators: dict[int, IncidentPlanePWaveEvaluator] = dict()

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
        self.farfield_coeffs = FarfieldAngularCoefficients(
            num_farfield_terms=self.num_farfield_terms,
            num_angular_gridpoints=self.grid.num_angular_gridpoints
        )
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

        # Parse and update farfield coefficients
        self.farfield_coeffs.update_fp_coeffs(
            self.parse_farfield_fp_coeffs_from_FD_result(result)
        )
        self.farfield_coeffs.update_gp_coeffs(
            self.parse_farfield_gp_coeffs_from_FD_result(result)
        )
        self.farfield_coeffs.update_fs_coeffs(
            self.parse_farfield_fs_coeffs_from_FD_result(result)
        )
        self.farfield_coeffs.update_gs_coeffs(
            self.parse_farfield_gs_coeffs_from_FD_result(result)
        )


    def parse_phi_from_FD_result(self, result) -> Grid:
        # Initialize empty grids to store phi values in
        num_ghost_points = self.num_ghost_points_artificial_boundary + self.num_ghost_points_physical_boundary
        grid_shape_with_ghost_points = (self.grid.num_angular_gridpoints, self.grid.num_radial_gridpoints + num_ghost_points)
        phi_full_grid = np.zeros(grid_shape_with_ghost_points, dtype='complex128')  # Includes ghost points
        phi_grid = np.zeros(self.grid.shape, dtype='complex128')                    # Does not include ghost points
        
        # Get ghost points at physical boundary 
        begin_idx = 0
        step = self.grid.num_angular_gridpoints
        for r in range(self.num_ghost_points_physical_boundary):
            # Parse current radial layer of phi gridpoints
            end_idx = begin_idx + step
            phi_full_grid[:,r] = result[begin_idx:end_idx]

            # Increment to next radial layer (skipping the psi gridpoints on the current layer)
            begin_idx = end_idx + step
        
        # Get physical gridpoints
        for r in range(self.grid.num_radial_gridpoints):
            # Parse current radial layer of phi gridpoints
            end_idx = begin_idx + step
            phi_grid[:,r] = result[begin_idx:end_idx]
            phi_full_grid[:,r + self.num_ghost_points_physical_boundary] = result[begin_idx:end_idx]

            # Increment to next radial layer (skipping the psi gridpoints on the current layer)
            begin_idx = end_idx + step

        # Get ghost points at artificial boundary 
        for r in range(self.num_ghost_points_artificial_boundary):
            # Parse current radial layer of phi gridpoints
            end_idx = begin_idx + step
            phi_full_grid[:,r + self.grid.num_radial_gridpoints + self.num_ghost_points_physical_boundary] = result[begin_idx:end_idx]

            # Increment to next radial layer (skipping the psi gridpoints on the current layer)
            begin_idx = end_idx + step
        
        return phi_grid, phi_full_grid


    def parse_psi_from_FD_result(self, result) -> Grid:        
        # Initialize empty grids to store psi values in
        num_ghost_points = self.num_ghost_points_artificial_boundary + self.num_ghost_points_physical_boundary
        grid_shape_with_ghost_points = (self.grid.num_angular_gridpoints, self.grid.num_radial_gridpoints + num_ghost_points)
        psi_full_grid = np.zeros(grid_shape_with_ghost_points, dtype='complex128')  # Includes ghost points
        psi_grid = np.zeros(self.grid.shape, dtype='complex128')                    # Does not include ghost points

        # Get ghost points at physical boundary 
        step = self.grid.num_angular_gridpoints
        begin_idx = 0 + step
        for r in range(self.num_ghost_points_physical_boundary):
            # Parse current radial layer of psi gridpoints
            end_idx = begin_idx + step
            psi_full_grid[:,r] = result[begin_idx:end_idx]

            # Increment to next radial layer (skipping the psi gridpoints on the current layer)
            begin_idx = end_idx + step

        # Get physical gridpoints
        for r in range(self.grid.num_radial_gridpoints):
            # Parse current radial layer of phi gridpoints
            end_idx = begin_idx + step
            psi_grid[:,r] = result[begin_idx:end_idx]
            psi_full_grid[:,r + self.num_ghost_points_physical_boundary] = result[begin_idx:end_idx]

            # Increment to next radial layer (skipping the psi gridpoints on the current layer)
            begin_idx = end_idx + step

        # Get ghost points at artificial boundary 
        for r in range(self.num_ghost_points_artificial_boundary):
            # Parse current radial layer of phi gridpoints
            end_idx = begin_idx + step
            psi_full_grid[:,r + self.grid.num_radial_gridpoints + self.num_ghost_points_physical_boundary] = result[begin_idx:end_idx]

            # Increment to next radial layer (skipping the psi gridpoints on the current layer)
            begin_idx = end_idx + step

        return psi_grid, psi_full_grid


    def parse_farfield_fp_coeffs_from_FD_result(
        self, 
        result:Vector
    ) -> np.ndarray:
        """Parse the scaled farfield F_l^p (compressional F) coefficient
        values at each "angular" gridpoint at the artificial
        boundary, for each term l=0, ..., self.num_farfield_terms,
        all from a given raw finite-difference result vector.

        The coefficients would be the coefficients evaluated at the 
        angular gridpoints on the artificial boundary

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        
        Returns:
            np.ndarray: A (self.grid.shape[0], L) array of 
                compressional F_l^p coefficient values at each
                angular gridpoint on the artificial boundary.
        """
        num_unknown_grid_vals = (
            2 * self.grid.num_gridpoints 
            + 2 * self.grid.num_angular_gridpoints * (
                self.num_ghost_points_artificial_boundary 
                + self.num_ghost_points_physical_boundary
            ) 
        )
        
        # Get the Fp coefficient chunk of the unknown vector, and reshape
        # it to be of shape (N_{theta}^{m}, L) 
        start = num_unknown_grid_vals
        end = start + (self.num_farfield_terms * self.grid.num_angular_gridpoints) 
        fp_coeffs = np.reshape(
            result[start:end],
            shape=(self.grid.num_angular_gridpoints, self.num_farfield_terms),
            order='F'
        )   # This command makes it so fp_coeffs[:,l] gives [F_{m,l}^p(theta_1), ..., F_{m,l}^p(theta_{N})]
        
        return fp_coeffs
    

    def parse_farfield_gp_coeffs_from_FD_result(
        self, 
        result:Vector
    ) -> np.ndarray:
        """Parse the scaled farfield G_l^p (compressional G) coefficient
        values at each "angular" gridpoint at the artificial
        boundary, for each term l=0, ..., self.num_farfield_terms,
        all from a given raw finite-difference result vector.

        The coefficients would be the coefficients evaluated at the 
        angular gridpoints on the artificial boundary

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        
        Returns:
            np.ndarray: A (self.grid.shape[0], L) array of 
                compressional G_l^p coefficient values at each
                angular gridpoint on the artificial boundary.
        """
        num_unknown_grid_vals = (
            2 * self.grid.num_gridpoints 
            + 2 * self.grid.num_angular_gridpoints * (
                self.num_ghost_points_artificial_boundary 
                + self.num_ghost_points_physical_boundary
            )
        )
        num_fp_coeffs = (
            self.num_farfield_terms
            * self.grid.num_angular_gridpoints
        )
        
        # Get the Gp coefficient chunk of the unknown vector, and reshape
        # it to be of shape (N_{theta}^{m}, L) 
        start = num_unknown_grid_vals + num_fp_coeffs
        end = start + (self.num_farfield_terms * self.grid.num_angular_gridpoints)
        gp_coeffs = np.reshape(
            result[start:end],
            shape=(self.grid.num_angular_gridpoints, self.num_farfield_terms),
            order='F'
        )   # This command makes it so gp_coeffs[:,l] gives [G_{m,l}^p(theta_1), ..., G_{m,l}^p(theta_{N})]
        
        return gp_coeffs
    

    def parse_farfield_fs_coeffs_from_FD_result(
        self, 
        result:Vector
    ) -> np.ndarray:
        """Parse the scaled farfield F_l^s (shear F) coefficient
        values at each "angular" gridpoint at the artificial
        boundary, for each term l=0, ..., self.num_farfield_terms,
        all from a given raw finite-difference result vector.

        The coefficients would be the coefficients evaluated at the 
        angular gridpoints on the artificial boundary

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        
        Returns:
            np.ndarray: A (self.grid.shape[0], L) array of 
                compressional F_l^s coefficient values at each
                angular gridpoint on the artificial boundary.
        """
        num_unknown_grid_vals = (
            2 * self.grid.num_gridpoints 
            + 2 * self.grid.num_angular_gridpoints * (
                self.num_ghost_points_artificial_boundary 
                + self.num_ghost_points_physical_boundary
            )
        )
        num_compressional_coeffs = 2 * (
            self.num_farfield_terms
            * self.grid.num_angular_gridpoints
        )
        
        # Get the Fs coefficient chunk of the unknown vector, and reshape
        # it to be of shape (N_{theta}^{m}, L) 
        start = num_unknown_grid_vals + num_compressional_coeffs
        end = start + (self.num_farfield_terms * self.grid.num_angular_gridpoints)
        fs_coeffs = np.reshape(
            result[start:end],
            shape=(self.grid.num_angular_gridpoints, self.num_farfield_terms),
            order='F'
        )   # This command makes it so fs_coeffs[:,l] gives [F_{m,l}^s(theta_1), ..., F_{m,l}^s(theta_{N})]
        
        return fs_coeffs
    

    def parse_farfield_gs_coeffs_from_FD_result(
        self, 
        result:Vector
    ) -> np.ndarray:
        """Parse the scaled farfield G_l^s (shear G) coefficient
        values at each "angular" gridpoint at the artificial
        boundary, for each term l=0, ..., self.num_farfield_terms,
        all from a given raw finite-difference result vector.

        The coefficients would be the coefficients evaluated at the 
        angular gridpoints on the artificial boundary

        Args:
            result (Vector): The raw output/solution vector of the FD
                matrix/vector system
        
        Returns:
            np.ndarray: A (self.grid.shape[0], L) array of 
                compressional G_l^s coefficient values at each
                angular gridpoint on the artificial boundary.
        """
        num_unknown_grid_vals = (
            2 * self.grid.num_gridpoints 
            + 2 * self.grid.num_angular_gridpoints * (
                self.num_ghost_points_artificial_boundary 
                + self.num_ghost_points_physical_boundary
            )
        )
        num_compressional_coeffs = 2 * (
            self.num_farfield_terms
            * self.grid.num_angular_gridpoints
        )
        num_gs_coeffs = (
            self.num_farfield_terms 
            * self.grid.num_angular_gridpoints
        )

        # Get the Gs coefficient chunk of the unknown vector, and reshape
        # it to be of shape (N_{theta}^{m}, L) 
        start = (
            num_unknown_grid_vals + num_compressional_coeffs
            + num_gs_coeffs
        )
        end = start + (self.num_farfield_terms * self.grid.num_angular_gridpoints)
        gs_coeffs = np.reshape(
            result[start:end],
            shape=(self.grid.num_angular_gridpoints, self.num_farfield_terms),
            order='F'
        )   # This command makes it so gs_coeffs[:,l] gives [G_{m,l}^s(theta_1), ..., G_{m,l}^s(theta_{N})]

        return gs_coeffs


    def get_total_phi(
        self,
        u_inc: Optional[IncidentPlanePWave] = None,
        other_obstacles: Optional[list[Self]] = None
    ) -> np.ndarray:
        """Gets the total phi at each gridpoint at this obstacle."""
        # Get current obstacle scattered phi
        phi_tot = self.phi_vals.copy()
        
        # Get incident wave phi
        if u_inc is not None:
            if u_inc.id not in self.incident_evaluators:
                raise ValueError(f"ERROR: Cannot get phi from incident wave with id {u_inc.id}. Was not used to solve this obstacle")
            phi_tot += self.incident_evaluators[u_inc.id].potentials(boundary_only=False)[:,:,0]
        
        # Get other obstacle scattered phi
        if other_obstacles is not None:
            for obstacle in other_obstacles:
                phi_tot += obstacle.get_scattered_wave_at_obstacle(
                    obstacle=self,
                    boundary_only=False,
                    desired_quantity=QOI.POTENTIALS,
                    update=False
                )[:,:,0]
        
        return phi_tot
    
    def get_total_psi(
        self,
        u_inc: Optional[IncidentPlanePWave] = None,
        other_obstacles: Optional[list[Self]] = None
    ) -> np.ndarray:
        """Gets the total psi at each gridpoint at this obstacle."""
        # Get current obstacle scattered psi
        psi_tot = self.psi_vals.copy()

        # Get incident wave psi
        if u_inc is not None:
            if u_inc.id not in self.incident_evaluators:
                raise ValueError(f"ERROR: Cannot plot incident wave with id {u_inc.id}. Was not used to solve this obstacle")
            psi_tot += self.incident_evaluators[u_inc.id].potentials(boundary_only=False)[:,:,1]

        # Get other obstacle scattered psi
        if other_obstacles is not None:
            for obstacle in other_obstacles:
                psi_tot += obstacle.get_scattered_wave_at_obstacle(
                    obstacle=self,
                    boundary_only=False,
                    desired_quantity=QOI.POTENTIALS,
                    update=False
                )[:,:,1]

        return psi_tot


    def _plot_periodic_contourf(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        **kwargs
    ) -> QuadContourSet:
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

        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
        else:
            vmin = None
        
        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
        else:
            vmax = None

        if vmax is not None and vmin is not None:
            levels = np.linspace(vmin, vmax, 1500)
        else:
            levels=None
        return plt.contourf(X, Y, Z, color_grid_vals, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)

    def plot_contourf(
        self, 
        vals: np.ndarray,
        **kwargs
    ) -> QuadContourSet:
        """Plot the contour over the obstacle grid"""
        # Plot contour over obstacle grid
        X_global, Y_global = self.grid.local_coords_to_global_XY()
        return self._plot_periodic_contourf(X_global, Y_global, vals, **kwargs)
    


    def _cache_farfield_evaluator(self, obstacle: Self) -> None:
        """If an obstacle's boundary info is not in self.obstacle_boundary_info,
        caches the desired information about the obstacle boundary there.
        
        Args:
            obstacle (MKFE_FDObstacle): An obstacle we'd like to store
                boundary information about.
        """
        # Skip any obstacles we've already cached
        if obstacle.id in self.obstacle_farfield_evaluators:
            return 
        
        # Check obstacle uses polar grid.
        # If it doesn't, current implementation won't work.
        if not isinstance(obstacle.grid, FDLocalPolarGrid):
            raise ValueError("Error: For current implementation, obstacle grids must all be polar")

        # Create a farfield evaluator object for evaluating this
        # current obstacle's (self's) farfield expansion data
        # at the other provided obstacle's gridpoints.
        farfield_evaluator = ElasticPolarFarfieldEvaluator(
            source_local_grid=self.grid,
            dest_local_grid=obstacle.grid,
            medium=obstacle.parent_medium,
            num_farfield_terms=obstacle.num_farfield_terms
        )
        self.obstacle_farfield_evaluators[obstacle.id] = farfield_evaluator
    

    def phi(self) -> np.ndarray:
        """Returns the phi_m potential at the m-local gridpoints"""
        return self.phi_vals 
    
    def psi(self) -> np.ndarray:
        """Returns the psi_m potential at the m-local gridpoints"""
        return self.psi_vals

    def get_scattered_wave_at_obstacle(
        self, 
        obstacle: Self,
        desired_quantity: QOI,
        boundary_only: bool = True,
        update: bool = True, 
        coordinate_system: CoordinateSystem = CoordinateSystem.LOCAL_POLAR
    ) -> np.ndarray:
        """Gets this obstacle's scattered wave displacement/stress
        at the provided obstacle's gridpoints.

        If desired, only gets scatterd wave values at 
        1 "radial" (axis=1) row of gridpoints of the given
        obstacle's grid corresponding to the gridpoints closest
        to the physical boundary of this given obstacle.

        Args:
            obstacle (MKFE_FDObstacle): The obstacle we'd like to
                evaluate this obstacle's scattered displacement
                or stress at
            desired_quantity (QOI): The desired quantity of interest
                (either stress, displacement, or potentials)
            boundary_only (bool): If true, only gets potentials at 
                the innermost "radial" (axis=1) ring of gridpoints
                around the obstacle's physical boundary. If False, gets 
                values at every gridpoint in the provided obstacle's
                grid. Defaults to True.
            update (bool): Whether to update to most current iteration
                of farfield coefficients. Default is True, but should be
                False for plotting.
            coordinate_system (CoordinateSystem): Which coordinate system
                to return quantity of interest in (only used for 
                displacement or stress desired_quantity values).
            
        
        Returns:
            np.ndarray: The quantity of interest evaluated at each of
                the gridpoints. (The -1 axis will be what differentiates the 
                different quantities of interest. For potentials,
                index 0 is phi and index 1 is psi. For displacements,
                index 0 is u_{r_mbar} and index 1 is u_{theta_mbar} 
                (the other obstacle's local polar coordinates).
                For stresses, index 0 is sigma_{r_mbar r_mbar}
                and index 1 is sigma_{r_mbar theta_mbar}
                (the other obstacle's local polar coordinates).
        """
        # Create farfield evaluator for given obstacle if not already done
        self._cache_farfield_evaluator(obstacle)

        # Update the angular coefficients in that farfield evaluator
        if update:
            self.obstacle_farfield_evaluators[obstacle.id].update_angular_coeffs(self.farfield_coeffs)

        # Check what sort of data is desired on the provided grid, and return it
        farfield_evaluator = self.obstacle_farfield_evaluators[obstacle.id]
        if desired_quantity is QOI.POTENTIALS:
            return farfield_evaluator.potentials(boundary_only)
        elif desired_quantity is QOI.DISPLACEMENT:
            return farfield_evaluator.displacement(boundary_only, coordinate_system)
        elif desired_quantity is QOI.STRESS:
            return farfield_evaluator.stress(boundary_only, coordinate_system)
        else:
            raise ValueError("desired_quantity must be a QOI Enum value (POTENTIALS, DISPLACEMENT, or STRESS)")
    
    def get_scattered_potentials_on_exterior_grid(
        self,
        X_global: np.ndarray,
        Y_global: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the scattered potentials phi and psi on a given exterior grid"""
        evaluator = ElasticFarfieldEvaluator(
            self.grid, X_global, Y_global, self.parent_medium, self.num_farfield_terms
        )
        evaluator.update_angular_coeffs(self.farfield_coeffs)
        return evaluator.potentials()



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

        # Store MKFE_FDObstacle attributes 
        super().__init__(
            center,
            r_artificial_boundary,
            boundary_condition,
            num_farfield_terms,
            parent_medium,
            PPW
        )

    def __getstate__(self):
        """Used when preparing to save a pickled version of this object."""
        # Prepare all class attributes 
        state = self.__dict__.copy()

        return state
    
    def __setstate__(self, state):
        """Restore the object from a serialized state"""
        # Recreate serializable attributes
        self.__dict__.update(state)

    



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
        # Cache incident wave evaluator if that wave not already seen 
        if not incident_wave.id in self.incident_evaluators:
            self.incident_evaluators[incident_wave.id] = (
                IncidentPlanePWaveEvaluator(
                    incident_wave=incident_wave,
                    local_grid = self.grid,
                    medium=self.parent_medium
                )
            )
        
        # Evaluate incident wave displacmenet in this local coordinate system
        uinc_evaluator = self.incident_evaluators[incident_wave.id]
        incident_wave_displacement = uinc_evaluator.displacement(boundary_only=True)
        displacement_local_r = incident_wave_displacement[:,0]
        displacement_local_theta = incident_wave_displacement[:,1]
        return displacement_local_r, displacement_local_theta
        
    
    def _get_other_obstacle_displacement_data(
        self,
        obstacle: MKFE_FDObstacle,
        update: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get scattered wave displacement data (from the 
        wave scattered from another obstacle) in a local
        coordinate decomposition at each gridpoint of
        this obstacle's physical boundary.
        
        Args:
            obstacle (MKFE_FDObstacle): The obstacle whose 
                scattered wave is incident upon this obstacle's
                physical boundary
            update (bool): Whether or not to update 
        
        Returns:
            np.ndarray: An array representing the other obstacle's
                scattered wave's local-r displacement at each
                gridpoint on the physical boundary
            np.ndarray: An array representing the other obstacle's
                scattered wave's local-theta displacement at each
                gridpoint on the physical boundary
        """
        # Get the scattered wave from the provided obstacle at
        # this obstacle's boundary
        obstacle_displacement = obstacle.get_scattered_wave_at_obstacle(
            obstacle=self,
            desired_quantity=QOI.DISPLACEMENT,
            boundary_only=True
        )
        displacement_local_r = obstacle_displacement[:,0]
        displacement_local_theta = obstacle_displacement[:,1]
        return displacement_local_r, displacement_local_theta
    
    def _get_incident_wave_stress_data(
        self, 
        incident_wave: IncidentPlanePWave
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get incident wave stress data in a local
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
        # Cache incident wave evaluator if that wave not already seen 
        if not incident_wave.id in self.incident_evaluators:
            self.incident_evaluators[incident_wave.id] = (
                IncidentPlanePWaveEvaluator(
                    incident_wave=incident_wave,
                    local_grid = self.grid,
                    medium=self.parent_medium
                )
            )
        
        # Evaluate incident wave displacmenet in this local coordinate system
        uinc_evaluator = self.incident_evaluators[incident_wave.id]
        incident_wave_stress_local = uinc_evaluator.stress(boundary_only=True)
        stress_rr_local = incident_wave_stress_local[:,0]
        stress_rtheta_local = incident_wave_stress_local[:,1]
        return stress_rr_local, stress_rtheta_local

    def _get_other_obstacle_stress_data(
        self,
        obstacle: MKFE_FDObstacle,
        update: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get scattered wave stress data (from the 
        wave scattered from another obstacle) in a local
        coordinate decomposition at each gridpoint of
        this obstacle's physical boundary.
        
        Args:
            obstacle (MKFE_FDObstacle): The obstacle whose 
                scattered wave is incident upon this obstacle's
                physical boundary
            update (bool): Whether or not to update 
        
        Returns:
            np.ndarray: An array representing the other obstacle's
                scattered wave's radial stress at the physical boundary
            np.ndarray: An array representing the other obstacle's
                scattered wave's shear stress at the physical boundary
        """
        # Get the scattered wave from the provided obstacle at
        # this obstacle's boundary
        obstacle_stress = obstacle.get_scattered_wave_at_obstacle(
            obstacle=self,
            desired_quantity=QOI.STRESS,
            boundary_only=True
        )
        stress_local_rr = obstacle_stress[:,0]
        stress_local_rtheta = obstacle_stress[:,1]
        return stress_local_rr, stress_local_rtheta
        

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
        stress_local_rr = np.zeros(self.grid.num_angular_gridpoints, dtype='complex128')
        stress_local_rtheta = np.zeros(self.grid.num_angular_gridpoints, dtype='complex128')

        ## Process incident wave stress data
        if incident_wave is not None:
            incident_stress = self._get_incident_wave_stress_data(incident_wave)
            stress_local_rr += incident_stress[0]
            stress_local_rtheta += incident_stress[1]
        
        ## Process other obstacle incoming wave data
        # Get other obstacle influences on this obstacle using Karp expansions
        for obstacle in obstacles:
            obstacle_stress = self._get_other_obstacle_stress_data(obstacle)
            stress_local_rr += obstacle_stress[0]
            stress_local_rtheta += obstacle_stress[1]

        # Recall: the focing vector is the negative of these
        # displacements (since we want total displacement = 0 at
        # the physical boundary)
        return np.hstack((-stress_local_rr, -stress_local_rtheta))
    

    def construct_forcing_vector(self, obstacles: list[MKFE_FDObstacle], incident_wave = None):
        # Get physical BC data for first 2*N_{theta} entries
        if self.bc is BoundaryCondition.HARD:
            physical_bc_data = self.hard_bc_forcing_vector(obstacles, incident_wave)
        elif self.bc is BoundaryCondition.SOFT:
            physical_bc_data = self.soft_bc_forcing_vector(obstacles, incident_wave)
        else:
            raise ValueError("Error: Unknown boundary condition. Cannot construct forcing vector.")
    
        # Get rest of forcing vector data 
        other_forcing_data = np.zeros(self.num_unknowns - 2*self.num_angular_gridpoints)
        
        # Construct and return total forcing vector
        return np.hstack((physical_bc_data, other_forcing_data))

    def _get_hard_physical_BC_rows(self) -> list[sparse.csc_array]:
        """Gets rows/equations in the finite-difference matrix
        corresponding to a hard physical boundary condition at the
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


    def _get_soft_physical_BC_rows(self) -> list[sparse.csc_array]:
        """Gets rows/equations in the finite-difference matrix
        corresponding to a soft physical boundary condition at the
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
        dr = self.grid.dr
        lam = self.parent_medium.lam 
        mu = self.parent_medium.mu
        kp = self.parent_medium.kp  

        # Create FD submatrix entries
        a = 2 * mu / (dr**2)
        b = mu / (r0**2 * dtheta)
        c = mu / (2 * r0 * dr * dtheta)
        p = mu / (dr**2)
        q = mu / (r0**2 * dtheta**2)
        s = mu / (2 * r0 * dr)


        ## Create FD submatrices
        # sigma_rr submatrices 
        A_rr_p_0 = (
            a * sparse.eye_array(
                self.num_angular_gridpoints,
                format='csc'
            )
        )
        A_rr_s_0 = (
            sparse_periodic_tridiag(
                self.num_angular_gridpoints,
                0., c, -c
            )
        )
        A_rr_p_1 = (
            (-lam * kp**2 - 2*a) * sparse.eye_array(
                self.num_angular_gridpoints,
                format='csc'
            )
        )
        A_rr_s_1 = (
            sparse_periodic_tridiag(
                self.num_angular_gridpoints,
                0., b, -b
            )
        )
        A_rr_p_2 = (
            a * sparse.eye_array(
                self.num_angular_gridpoints,
                format='csc'
            )
        )
        A_rr_s_2 = (
            sparse_periodic_tridiag(
                self.num_angular_gridpoints,
                0., -c, c
            )
        )

        # sigma_rtheta submatrices
        A_rtheta_p_0 = (
            sparse_periodic_tridiag(
                self.num_angular_gridpoints,
                0., c, -c
            )
        ) 
        A_rtheta_s_0 = (
            (-p-s) * sparse.eye_array(
                self.num_angular_gridpoints,
                format='csc'
            )
        )
        A_rtheta_p_1 = (
            sparse_periodic_tridiag(
                self.num_angular_gridpoints,
                0., b, -b
            )
        ) 
        A_rtheta_s_1 = (
            sparse_periodic_tridiag(
                self.num_angular_gridpoints,
                2*(p-q), q, q
            )
        ) 
        A_rtheta_p_2 = (
            sparse_periodic_tridiag(
                self.num_angular_gridpoints,
                0., -c, c
            )
        )
        A_rtheta_s_2 = (
            (-p+s) * sparse.eye_array(
                self.num_angular_gridpoints,
                format='csc'
            )
        )

        # Create block row for these 2*N_{theta} equations
        sigma_rr_row_shape = (self.num_angular_gridpoints, self.num_unknowns)
        sigma_rtheta_row_shape = (self.num_angular_gridpoints, self.num_unknowns)
        
        num_zeros_right = self.num_unknowns - (6 * self.num_angular_gridpoints)
        sigma_rr_blocks = [A_rr_p_0, A_rr_s_0, A_rr_p_1, A_rr_s_1, A_rr_p_2, A_rr_s_2, num_zeros_right]
        sigma_rtheta_blocks = [A_rtheta_p_0, A_rtheta_s_0, A_rtheta_p_1, A_rtheta_s_1, A_rtheta_p_2, A_rtheta_s_2, num_zeros_right]
        
        sigma_rr_rows = sparse_block_row(sigma_rr_row_shape, sigma_rr_blocks)
        sigma_rtheta_rows = sparse_block_row(sigma_rtheta_row_shape, sigma_rtheta_blocks)
        return [sigma_rr_rows, sigma_rtheta_rows]


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
        if self.bc is BoundaryCondition.HARD:
            return self._get_hard_physical_BC_rows()
        elif self.bc is BoundaryCondition.SOFT:
            return self._get_soft_physical_BC_rows()
        else:
            raise ValueError(f"Cannot get physical BC rows for boundary condition {self.bc} - not of type SOFT or HARD")


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

        NOTE: These interface rows do not directly use the farfield
        expansions. Instead, they use a scaled form of these, where
        the division by powers of (kp R) and (ks R) is limited as 
        much as possible. Thus, results coming from this 
        farfield expansion should be parsed and scaled 
        appropriately.

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
        R = self.r_artificial_boundary
        kpR = kp * R
        ksR = ks * R
        Hp0 = hankel1(0, kpR)
        Hp1 = hankel1(1, kpR)
        Hs0 = hankel1(0, ksR)
        Hs1 = hankel1(1, ksR)
        interface_block_row_shape = (self.num_angular_gridpoints, self.num_unknowns)
        num_r_gridpts = self.grid.num_radial_gridpoints

        ## Create arrays of coefficients to insert into FD submatrices
        J_p = np.repeat(Hp0, self.num_farfield_terms)
        K_p = np.repeat(Hp1, self.num_farfield_terms)
        J_s = np.repeat(Hs0, self.num_farfield_terms)
        K_s = np.repeat(Hs1, self.num_farfield_terms)

        ## Create repetitive arays 
        I_N_theta = sparse.eye_array(self.num_angular_gridpoints, format='csc')
        
        ## C.1.A - Continuity of Phi
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

        NOTE: These interface rows do not directly use the farfield
        expansions. Instead, they use a scaled form of these, where
        the division by powers of (kp R) and (ks R) is limited as 
        much as possible. Thus, results coming from this 
        farfield expansion should be parsed and scaled 
        appropriately.

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
        R = self.r_artificial_boundary
        kpR = kp * R
        ksR = ks * R
        dr = self.grid.dr
        Hp0 = hankel1(0, kpR)
        Hp1 = hankel1(1, kpR)
        Hs0 = hankel1(0, ksR)
        Hs1 = hankel1(1, ksR)
        interface_block_row_shape = (self.num_angular_gridpoints, self.num_unknowns)
        num_r_gridpts = self.grid.num_radial_gridpoints
        l = np.arange(self.num_farfield_terms)          # l = 0, 1, ..., L-1

        # Create needed coefficients for entries in arrays
        z_plus = 1/(2 * dr)
        z_minus = -1/(2 * dr)
        A_p = -kp * (
            Hp1 + (l * Hp0/kpR)
        )
        B_p = -kp * (
            -Hp0 + ((l+1) * Hp1/kpR)
        )
        A_s = -ks * (
            Hs1 + (l * Hs0/ksR)
        )
        B_s = -ks * (
            -Hs0 + ((l+1) * Hs1/ksR)
        )

        # Create repetitive arrays 
        I_N_theta = sparse.eye_array(self.num_angular_gridpoints, format='csc')

        # FD submatrices for phi/psi at radial N-1 and N+1 central 1st radial derivative
        Z_plus = z_plus * I_N_theta
        Z_minus = z_minus * I_N_theta

        ## FD submatrices for farfield coefficients 1st radial derivative
        # phi continuity - p coefficients
        M_A_p = []
        M_B_p = []
        for A_l_p, B_l_p in zip(A_p, B_p):
            M_A_l_p = A_l_p * I_N_theta
            M_B_l_p = B_l_p * I_N_theta
            M_A_p.append(M_A_l_p)
            M_B_p.append(M_B_l_p)

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

        # psi continuity - s coefficients
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

        NOTE: These interface rows do not directly use the farfield
        expansions. Instead, they use a scaled form of these, where
        the division by powers of (kp R) and (ks R) is limited as 
        much as possible. Thus, results coming from this 
        farfield expansion should be parsed and scaled 
        appropriately.

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
        R = self.r_artificial_boundary
        kpR = kp * R
        ksR = ks * R
        dr = self.grid.dr
        Hp0 = hankel1(0, kpR)
        Hp1 = hankel1(1, kpR)
        Hs0 = hankel1(0, ksR)
        Hs1 = hankel1(1, ksR)
        interface_block_row_shape = (self.num_angular_gridpoints, self.num_unknowns)
        num_r_gridpts = self.grid.num_radial_gridpoints
        l = np.arange(self.num_farfield_terms)          # l = 0, 1, ..., L-1

        # Create coefficients for insertion into FD submatrices
        q_minus = -1/(dr**2)
        q_plus = 2/(dr**2)
        C_p = kp**2 * (
            -Hp0 + ((2*l + 1) * Hp1 / kpR) + (l * (l+1) * Hp0 / (kpR**2))
        )
        D_p = kp**2 * (
            -Hp1 - ((2*l + 1) * Hp0 / kpR) + ((l+1) * (l+2) * Hp1 / (kpR**2))
        )
        C_s = ks**2 * (
            -Hs0 + ((2*l + 1) * Hs1 / ksR) + (l * (l+1) * Hs0 / (ksR**2))
        )
        D_s = kp**2 * (
            -Hs1 - ((2*l + 1) * Hs0 / ksR) + ((l+1) * (l+2) * Hs1 / (ksR**2))
        )

        # Create repetitive arays 
        I_N_theta = sparse.eye_array(self.num_angular_gridpoints, format='csc')

        # FD submatrices for gridpoints at radial levels Nr-1, Nr, and Nr+1
        # for 2nd radial derivative
        Q_plus = q_plus * I_N_theta
        Q_minus = q_minus * I_N_theta

        ## FD submatrices for phi/p-coefficient farfield coefficients
        ## for 2nd radial derivative
        # Create FD submatrices 
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

        ## FD submatrices for psi/s-coefficient farfield coefficients
        ## for 2nd radial derivative
        # Create FD submatrices
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
        r"""Gets rows/equations in the finite-difference matrix
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
        kp = self.parent_medium.kp
        ks = self.parent_medium.ks
        R = self.r_artificial_boundary
        kpR = kp * R 
        ksR = ks * R
        l = np.arange(1, self.num_farfield_terms)
        t_plus = 1/(dtheta**2)
        t_F = -2/(dtheta**2) + (l-1)**2
        t_G = -2/(dtheta**2) + l**2 
        s_Fp = -2 * l * kpR                 # NOTE: SCALING by kpR or ksR HERE IS FROM MATLAB CODE
        s_Fs = -2 * l * ksR 
        s_Gp = 2 * l * kpR
        s_Gs = 2 * l * ksR
        recursive_relation_block_shape = (2 * self.num_angular_gridpoints, self.num_unknowns)
        num_r_gridpts = self.grid.num_radial_gridpoints

        # Create repetitive arrays 
        I_N_Theta = sparse.eye_array(self.num_angular_gridpoints, format='csc')

        # Iterate through all the l index values to create appropriate rows
        for i, (t_l_F, t_l_G, s_l_Fp, s_l_Gp, s_l_Fs, s_l_Gs) in enumerate(zip(t_F, t_G, s_Fp, s_Gp, s_Fs, s_Gs)):
            # Create smaller FD subarrays
            T_l_F = sparse_periodic_tridiag(self.num_angular_gridpoints, t_l_F, t_plus, t_plus)
            T_l_G = sparse_periodic_tridiag(self.num_angular_gridpoints, t_l_G, t_plus, t_plus)
            S_l_F_p = s_l_Fp * I_N_Theta
            S_l_G_p = s_l_Gp * I_N_Theta
            S_l_F_s = s_l_Fs * I_N_Theta
            S_l_G_s = s_l_Gs * I_N_Theta 

            # Combine smaller subarrays to create bigger subarrays 
            A_rec_l_p = sparse.block_diag([T_l_F, S_l_G_p], format='csc')
            Z_rec_l_p = sparse_block_antidiag([T_l_G, S_l_F_p])
            A_rec_l_s = sparse.block_diag([T_l_F, S_l_G_s], format='csc')
            Z_rec_l_s = sparse_block_antidiag([T_l_G, S_l_F_s])

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
                    [num_zeros_left, A_rec_l_p, num_zeros_middle, Z_rec_l_p]
                )
            else:
                block_data = (
                    [num_zeros_left, A_rec_l_p, num_zeros_middle, Z_rec_l_p, num_zeros_right]
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
                    [num_zeros_left, A_rec_l_s, num_zeros_middle, Z_rec_l_s]
                )
            else:
                block_data = (
                    [num_zeros_left, A_rec_l_s, num_zeros_middle, Z_rec_l_s, num_zeros_right]
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

    
    def dr_phi(self) -> np.ndarray:
        """Returns the 1st radial derivative of the phi_m potential
        at the m-local gridpoints.
        
        Uses 2nd order centered differences (with ghost points) 
        to compute this:
        dr phi_{i,j} = (phi_{i+1,j}-phi_{i-1,j})/(2 * dr)
        """
        return 1/(2 * self.grid.dr) * (
            self.phi_vals_padded[:,2:] - self.phi_vals_padded[:,:-2]
        )
    
    def dr_psi(self) -> np.ndarray:
        """Returns the 1st radial derivative of the psi_m potential
        at the m-local gridpoints.
        
        Uses 2nd order centered differences (with ghost points) 
        to compute this:
        dr psi_{i,j} = (psi_{i+1,j}-psi_{i-1,j})/(2 * dr)
        """
        return 1/(2 * self.grid.dr) * (
            self.psi_vals_padded[:,2:]
            - self.psi_vals_padded[:,:-2]
        )
    
    def d2r_phi(self) -> np.ndarray:
        """Returns the 2nd radial derivative of the phi_m potential
        at the m-local gridpoints.
        
        Uses 2nd order centered differences (with ghost points) 
        to compute this:
        d2r phi_{i,j} = (phi_{i+1,j}-2phi{i,j}+phi_{i-1,j}) / (dr^2)
        """
        return 1/(self.grid.dr**2) * (
            self.phi_vals_padded[:,2:] 
            - 2 * self.phi_vals_padded[:,1:-1]
            + self.phi_vals_padded[:,:-2]
        )
    
    def d2r_psi(self) -> np.ndarray:
        """Returns the 2nd radial derivative of the psi_m potential
        at the m-local gridpoints.
        
        Uses 2nd order centered differences (with ghost points) 
        to compute this:
        d2r psi_{i,j} = (psi_{i+1,j}-2psi{i,j}+psi_{i-1,j}) / (dr^2)
        """
        return 1/(self.grid.dr**2) * (
            self.psi_vals_padded[:,2:] 
            - 2 * self.psi_vals_padded[:,1:-1]
            + self.psi_vals_padded[:,:-2]
        )
    
    def dtheta_phi(self) -> np.ndarray:
        """Returns the 1st angular derivative of the phi_m potential
        at the m-local gridpoints.
        
        Uses 2nd order centered differences (with ghost points) 
        to compute this:
        dtheta phi_{i,j} = (phi_{i,j+1}-phi_{i,j-1}) / (2*dtheta)
        """
        return 1/(2*self.grid.dtheta) * (
            np.roll(self.phi_vals, -1, axis=0)      # Roll of -1 along angular axis brings i+1 gridpoints into line with i gridpoints
            - np.roll(self.phi_vals, 1, axis=0)     # Roll of +1 along angular axis brings i-1 gridpoints into line with i gridpoints
        )
    
    def dtheta_psi(self) -> np.ndarray:
        """Returns the 1st angular derivative of the psi_m potential
        at the m-local gridpoints.
        
        Uses 2nd order centered differences (with ghost points) 
        to compute this:
        dtheta psi_{i,j} = (psi_{i,j+1}-psi_{i,j-1}) / (2*dtheta)
        """
        return 1/(2*self.grid.dtheta) * (
            np.roll(self.psi_vals, -1, axis=0)      # Roll of -1 along angular axis brings j+1 gridpoints into line with j gridpoints
            - np.roll(self.psi_vals, 1, axis=0)     # Roll of +1 along angular axis brings j-1 gridpoints into line with j gridpoints
        )

    def d2theta_phi(self) -> np.ndarray:
        """Returns the 2nd angular derivative of the phi_m potential
        at the m-local gridpoints.
        
        Uses 2nd order centered differences (with ghost points) 
        to compute this:
        d2theta phi_{i,j} = (phi_{i,j+1}-phi_{i,j}+phi_{i,j-1}) / (dtheta^2)
        """
        return 1/(self.grid.dtheta**2) * (
            np.roll(self.phi_vals, -1, axis=0)      # Roll of -1 along angular axis brings j+1 gridpoints into line with j gridpoints
            - 2 * self.phi_vals                         # Original i,j gridpoints
            + np.roll(self.phi_vals, 1, axis=0)     # Roll of +1 along angular axis brings j-1 gridpoints into line with j gridpoints
        )
    
    def d2theta_psi(self) -> np.ndarray:
        """Returns the 2nd angular derivative of the psi_m potential
        at the m-local gridpoints.
        
        Uses 2nd order centered differences (with ghost points) 
        to compute this:
        d2theta psi_{i,j} = (psi_{i,j+1}-psi_{i,j}+psi_{i,j-1}) / (dtheta^2)
        """
        return 1/(self.grid.dtheta**2) * (
            np.roll(self.psi_vals, -1, axis=0)      # Roll of -1 along angular axis brings j+1 gridpoints into line with j gridpoints
            - 2 * self.psi_vals                         # Original i,j gridpoints
            + np.roll(self.psi_vals, 1, axis=0)     # Roll of +1 along angular axis brings j-1 gridpoints into line with j gridpoints
        )

    def dr_dtheta_phi(self) -> np.ndarray:
        """Returns the mixed 2nd order radial/angular derivative
        of the phi_m potential at the m-local gridpoints.
        
        Uses 2nd order centered differences (with ghost points) 
        to compute this, using a 4 point stencil
        d2theta phi_{i,j} = (phi_{i+1,j+1} - phi_{i+1,j-1} - phi_{i-1,j+1} + phi_{i-1,j-1}) / (4*dr*dtheta)
        """
        phi_ip1_jp1 = np.roll(self.phi_vals_padded[:,2:], -1, axis=0)
        phi_ip1_jm1 = np.roll(self.phi_vals_padded[:,2:], 1, axis=0)
        phi_im1_jp1 = np.roll(self.phi_vals_padded[:,:-2], -1, axis=0)
        phi_im1_jm1 = np.roll(self.phi_vals_padded[:,:-2], 1, axis=0)

        return 1/(4 * self.grid.dr * self.grid.dtheta) * (
            phi_ip1_jp1 - phi_ip1_jm1 - phi_im1_jp1 + phi_im1_jm1
        )

    def dr_theta_psi(self) -> np.ndarray:
        """Returns the mixed 2nd order radial/angular derivative
        of the psi_m potential at the m-local gridpoints.
        
        Uses 2nd order centered differences (with ghost points) 
        to compute this, using a 4 point stencil
        d2theta psi_{i,j} = (psi_{i+1,j+1} - psi_{i+1,j-1} - psi_{i-1,j+1} + psi_{i-1,j-1}) / (4*dr*dtheta)
        """
        psi_ip1_jp1 = np.roll(self.psi_vals_padded[:,2:], -1, axis=0)
        psi_ip1_jm1 = np.roll(self.psi_vals_padded[:,2:], 1, axis=0)
        psi_im1_jp1 = np.roll(self.psi_vals_padded[:,:-2], -1, axis=0)
        psi_im1_jm1 = np.roll(self.psi_vals_padded[:,:-2], 1, axis=0)

        return 1/(4 * self.grid.dr * self.grid.dtheta) * (
            psi_ip1_jp1 - psi_ip1_jm1 - psi_im1_jp1 + psi_im1_jm1
        )

    def displacement(
        self,
        coordinate_system: CoordinateSystem
    ) -> np.ndarray:
        """Calculates the displacement at each gridpoint.

        Args:
            coord_sys (CoordinateSystem): The desired coordinate
                system to plot these potentials in.
        """
        # Parse needed constants and derivative values at each gridpoint
        dr_phi = self.dr_phi()
        dr_psi = self.dr_psi()
        dtheta_phi = self.dtheta_phi()
        dtheta_psi = self.dtheta_psi()
        r_vals = self.grid.r_local

        # Get m-local polar displacements 
        u_r_m = dr_phi + dtheta_psi/r_vals      # Local radial displacement
        u_theta_m = dtheta_phi/r_vals - dr_psi  # Local angular displacement
        if coordinate_system is CoordinateSystem.LOCAL_POLAR:
            return np.stack((u_r_m, u_theta_m), axis=-1)
        
        # Get cartesian displacement using x=r*cos(theta), y=r*sin(theta)
        # NOTE: This is translation invariant, so should be same local or global cartesian coordinates
        u_x_m = u_r_m * self.grid.cos_theta_local - u_theta_m * self.grid.sin_theta_local
        u_y_m = u_r_m * self.grid.sin_theta_local + u_theta_m * self.grid.cos_theta_local
        if coordinate_system is CoordinateSystem.LOCAL_CARTESIAN or coordinate_system is CoordinateSystem.GLOBAL_CARTESIAN:
            return np.stack((u_x_m, u_y_m), axis=-1)
        
        # Get global polar displacements using R^{m -> g} rotation matrix
        theta_local = self.grid.theta_local
        theta_global = self.grid.local_coords_to_global_polar()[1]
        angle_diffs = theta_local - theta_global 
        rotation_sine = np.sin(angle_diffs)
        rotation_cos = np.cos(angle_diffs)
        u_r_global = rotation_cos * u_r_m - rotation_sine * u_theta_m 
        u_theta_global = rotation_sine * u_r_m + rotation_cos * u_theta_m
        if coordinate_system is CoordinateSystem.GLOBAL_POLAR:
            return np.stack((u_r_global, u_theta_global), axis=-1)
        
        raise ValueError(f"Invalid coordinate system {coordinate_system}.")
    
    def stress(
        self,
        coordinate_system: CoordinateSystem
    ) -> np.ndarray:
        """Calculates the stress at each gridpoint.

        Args:
            coordinate_system (CoordinateSystem): The desired coordinate
                system to get these stresses in.
        """
        # Parse needed constants and derivative values at each gridpoint
        phi = self.phi()
        psi = self.psi()
        dr_phi = self.dr_phi()
        dr_psi = self.dr_psi()
        dtheta_phi = self.dtheta_phi()
        dtheta_psi = self.dtheta_psi()
        d2r_phi = self.d2r_phi()
        d2r_psi = self.d2r_psi()
        d2theta_phi = self.d2theta_phi()
        d2theta_psi = self.d2theta_psi()
        dr_dtheta_phi = self.dr_dtheta_phi()
        dr_dtheta_psi = self.dr_theta_psi()
        r = self.grid.r_local
        lam = self.parent_medium.lam 
        mu = self.parent_medium.mu 
        kp = self.parent_medium.kp

        # Get m-local polar stresses 
        sigma_rr_local = (
            (-lam * kp**2) * phi 
            + (2 * mu) * d2r_phi 
            - (2 * mu) * (dtheta_psi / (r**2) - dr_dtheta_psi / r)
        )
        sigma_rtheta_local = (
            (2 * mu) * (dr_dtheta_phi / r - dtheta_phi / (r**2))
            + mu * (d2theta_psi / (r**2) + dr_psi / r - d2r_psi)
        )
        sigma_thetatheta_local = (
            (-lam * kp**2) * phi 
            + (2 * mu) * (dr_phi / r + d2theta_phi / (r**2))
            + (2 * mu) * (dtheta_psi / (r**2) - dr_dtheta_psi / r)
        )
        if coordinate_system is CoordinateSystem.LOCAL_POLAR:
            return np.stack((sigma_rr_local, sigma_rtheta_local, sigma_thetatheta_local), axis=-1)
        
        # Get global polar stresses
        theta_local = self.grid.theta_local
        theta_global = self.grid.local_coords_to_global_polar()[1]
        angle_diffs = theta_local - theta_global 
        rotation_sine = np.sin(angle_diffs)
        rotation_cosine = np.cos(angle_diffs)
        
        sigma_rr_global = (
            sigma_rr_local * (rotation_cosine**2) 
            + sigma_thetatheta_local * (rotation_sine**2)
            - 2 * sigma_rtheta_local * rotation_sine * rotation_cosine
        )
        sigma_rtheta_global = (
            sigma_rtheta_local * (rotation_cosine**2 - rotation_sine**2)
            + (sigma_rr_local - sigma_thetatheta_local) * rotation_sine * rotation_cosine
        )
        sigma_thetatheta_global = (
            sigma_rr_local * (rotation_sine**2) 
            + sigma_thetatheta_local * (rotation_cosine**2)
            + 2 * sigma_rtheta_local * rotation_sine * rotation_cosine
        )
        if coordinate_system is CoordinateSystem.GLOBAL_POLAR:
            return np.stack((sigma_rr_global, sigma_rtheta_global, sigma_thetatheta_global), axis=-1)
        
        # Get global cartesian stresses from global polar stresses
        sine_theta_global = np.sin(theta_global)
        cosine_theta_global = np.cos(theta_global)

        sigma_xx = (
            sigma_rr_global * cosine_theta_global**2 
            + sigma_thetatheta_global * sine_theta_global**2 
            - 2 * sigma_rtheta_global * sine_theta_global * cosine_theta_global
        )
        sigma_xy = (
            (sigma_rr_global - sigma_thetatheta_global) * sine_theta_global * cosine_theta_global 
            + sigma_rtheta_global * (cosine_theta_global**2 - sine_theta_global**2)
        )
        sigma_yy = (
            sigma_rr_global * sine_theta_global**2 
            + sigma_thetatheta_global * cosine_theta_global**2 
            + 2 * sigma_rtheta_global * sine_theta_global * cosine_theta_global
        )
        if coordinate_system is CoordinateSystem.LOCAL_CARTESIAN or coordinate_system is CoordinateSystem.GLOBAL_CARTESIAN:
            return np.stack((sigma_xx, sigma_xy, sigma_yy), axis=-1)
        
        raise ValueError(f"Invalid coordinate system {coordinate_system}.")


    def get_total_displacement(
        self,
        incident_wave: Optional[IncidentPlanePWave] = None,
        other_obstacles: Optional[list[Self]] = None,
        coordinate_system: CoordinateSystem = CoordinateSystem.GLOBAL_POLAR
    ) -> np.ndarray:
        """Gets the total displacement at each gridpoint at this obstacle.
        
        x (or r) displacement is the [:,:,0] value at each gridpoint, and
        y (or theta) displacement is the [:,:,1] value at each gridpoint.
        """
        # Get obstacle displacement as global x/y vectors
        displacement = self.displacement(coordinate_system)
        
        # Get other obstacle displacements as global x/y vectors
        if other_obstacles is not None:
            for obstacle in other_obstacles:
                displacement += obstacle.get_scattered_wave_at_obstacle(
                    obstacle=self,
                    boundary_only=False,
                    desired_quantity=QOI.DISPLACEMENT,
                    update=False,
                    coordinate_system=coordinate_system
                )
        
        # Get incident wave displacement as as global x/y vectors
        if incident_wave is not None:
            incident_wave_evaluator = self.incident_evaluators[incident_wave.id]
            displacement += incident_wave_evaluator.displacement(
                boundary_only=False,
                coordinate_system=coordinate_system
            )
        
        return displacement
    

    def get_total_stress(
        self,
        incident_wave: Optional[IncidentPlanePWave] = None,
        other_obstacles: Optional[list[Self]] = None,
        coordinate_system: CoordinateSystem = CoordinateSystem.GLOBAL_POLAR
    ) -> np.ndarray:
        """Gets the total stress at each gridpoint at this obstacle.
        
        xx/rr stress is the [:,:,0] value at each gridpoint,
        xy/r_theta stress is the [:,:,1] value at each gridpoint,
        and yy/theta_theta stress is the [:,:,2] value at each gridpoint
        """
        # Get obstacle stresses in given coordinate system
        stress = self.stress(coordinate_system)
        
        # Add in other obstacle stresses in given coordinate system
        if other_obstacles is not None:
            for obstacle in other_obstacles:
                stress += obstacle.get_scattered_wave_at_obstacle(
                    obstacle=self,
                    boundary_only=False,
                    desired_quantity=QOI.STRESS,
                    update=False,
                    coordinate_system=coordinate_system
                )
        
        # Add in incident wave stress in given coordinate system
        if incident_wave is not None:
            incident_wave_evaluator = self.incident_evaluators[incident_wave.id]
            stress += incident_wave_evaluator.stress(
                boundary_only=False,
                coordinate_system=coordinate_system
            )

        return stress


    def plot_displacement_vector_field(
        self,
        method:str = 'abs',
        incident_wave: Optional[IncidentPlanePWave] = None,
        other_obstacles: Optional[list[Self]] = None,
        step:int = 1,
        **kwargs
    ) -> None:
        """Plot the displacement vector field at this obstacle.

        If no arguments are provided, plot the scattered phi
        potential eminating from this obstacle.

        If other obstacles provided, linearly
        add their psi potential values to this one before 
        plotting.

        Args:
            method (str): Whether to plot the complex modulus ('abs'),
                the real part ('real'), or the imaginary part ('imag')
                of the psi scattered potential.
                Default is 'abs'.
            incident_wave (IncidentPlanePWave): If provided, the incident
                plane p-wave on this obstacle.
            other_obstacles (list[Self]): If provided, a list
                of other obstacles whose scattered potentials 
                are incident upon this obstacle.
            step (int): The stride length while choosing which 
                gridpoints to plot arrows at (defaults to 1, 
                meaning plotting every gridpoint)
            *cmap (Colormap): The Matplotlib colormap object 
                to use for plotting (cm.coolwarm is used if none
                is provided)
        """
        displacement = self.get_total_displacement(
            incident_wave=incident_wave,
            other_obstacles=other_obstacles,
            coordinate_system=CoordinateSystem.GLOBAL_CARTESIAN
        )
        
        # Plot the displacement vector field
        X,Y = self.grid.local_coords_to_global_XY()
        if method.lower().strip() == 'abs':
            displacement_x = np.abs(displacement[:,:,0])
            displacement_y = np.abs(displacement[:,:,1])
        elif method.lower().strip() == 'real':
            displacement_x = np.real(displacement[:,:,0])
            displacement_y = np.real(displacement[:,:,1])
        elif method.lower().strip() == 'imag':
            displacement_x = np.imag(displacement[:,:,0])
            displacement_y = np.imag(displacement[:,:,1])
        else:
            raise ValueError(f"Error: Method {method} not one of the valid plotting methods ['abs', 'real', or 'imag']")

        displacement_magnitude = np.sqrt(displacement_x**2 + displacement_y**2)
        displacement_x_direction_only = displacement_x / (displacement_magnitude + 1e-8)    # Avoid division by zero by adding +1e-8
        displacement_y_direction_only = displacement_y / (displacement_magnitude + 1e-8)
        plt.quiver(
            X[::step,::step],              # X-positions of arrow tails
            Y[::step,::step],              # Y-position of arrow tails
            displacement_x_direction_only[::step,::step],  # X-coordinate of arrow pointing directions
            displacement_y_direction_only[::step,::step],  # Y-coordinate of arrow pointing directions
            displacement_magnitude[::step,::step],         # Color of arrows (depends on original magnitude of displacement)
            angles='xy',                    # Arrow direction happens in data (X/Y) coordinates; 
            scale=50,
            width=0.002,
            headlength=3.4,
            headaxislength=3
        )

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