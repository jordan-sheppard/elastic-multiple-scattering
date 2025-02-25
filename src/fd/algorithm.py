import numpy as np 
import json
import logging

from ..base.medium import LinearElasticMedium
from ..base.waves import IncidentPlanePWave
from ..base.exceptions import MaxIterationsExceedException
from ..base.consts import Algorithm
from .grids import FDPolarGrid_ArtBndry
from .obstacles import (
    MKFE_FDObstacle, Circular_MKFE_FDObstacle,
    CircularObstacleGeometry
)

class MKFE_FD_ScatteringProblem:
    """A class for setting up a multiple-scattering problem using 
    Finite Differences to approximate solutions, where boundary
    conditions at artificial boundaries are approximated using 
    the Karp Farfield Expansion Absorbing Boundary Condition
    (or MKFE ABC).
    
    This gives the ability to add obstacles of various geometries,
    to a linearly elastic medium with given physical properties.
    Additionally, this represents an incident wave, along with 
    the time-harmonic frequency at which it propagates, inside 
    this medium.

    Attributes:
        medium (LinearElasticMedium): The elastic medium where
            the scattering problem is taking place 
        incident_wave (IncidentPlaneWave): The incident plane 
            wave for this scattering problem 
        num_farfield_terms (int): The number of terms in the
            farfield expansion to use at each obstacle
        circular_obstacle_geometries (list[CircularObstacleGeometry]): A
            list of circular obstacle geometry information (including
            center locations, radii of obstacle and artificial
            boundaries, and physical boundary conditions)
        obstacles (list[MKFE_FDObstacle]): A list of obstacles
            to be considered in this scattering problem.
    """
    obstacles: list[MKFE_FDObstacle]
    circular_obstacle_geometries: list[CircularObstacleGeometry]

    def __init__(
        self,
        medium: LinearElasticMedium,
        incident_wave: IncidentPlanePWave,
        num_farfield_terms: int,
        obstacle_config_file: str
    ):
        """Initialize a multiple-scattering problem.
        
        Args:
            medium (LinearElasticMedium): The elastic medium where
                the scattering problem is taking place 
            incident_wave (IncidentPlaneWave): The incident plane 
                wave for this scattering problem 
            num_farfield_terms (int): The number of terms in the
                farfield expansion to use at each obstacle
            obstacle_config_file (str): A path to the obstacle
                configuration JSON file that contains information
                about obstacle geometries and boundary conditions.
        """
        # Store medium and incident wave info
        self.medium = medium
        self.incident_wave = incident_wave

        # Store obstacle geometry and boundary condition info
        self.circular_obstacle_geometries = []
        self._add_obstacle_geometries_from_config(obstacle_config_file)

        # Store other relevant information for MKFE FD algorithm
        self.num_farfield_terms = num_farfield_terms

        # Create a place to store fully-fleshed-out
        # obstacles of interest that can be used 
        # to iteratively update unknowns
        self.obstacles = []
        
    
    def _add_obstacle_geometries_from_config(self, config_file:str) -> None:
        """Adds obstacles from a configuration JSON file.
        
        Args:
            config_file (str): A path to the configuration JSON file
                containing obstacle geometry/boundary condition
                information.

        Raises:
            IOException: If the provided filename is invalid or
                otherwise accessible
        """
        with open(config_file, 'r') as in_json_file:
            config = json.load(in_json_file)

        # Parse circular obstacle geometries
        obstacle_geometries_circ = config['obstacles']['circular'] 
        for obstacle_geometry_info in obstacle_geometries_circ:
            obstacle_geometry = CircularObstacleGeometry.from_config_entry(obstacle_geometry_info)
            self.circular_obstacle_geometries.append(obstacle_geometry)
        
        # TODO: HANDLE PARAMETRIC GEOMETRIES


    def _setup_scattering_obstacles_for_PPW(
        self,
        PPW: int
    ) -> None:
        """Set up all obstacles for a scattering problem.

        Accomplishes this by combining the stored obstacle geometries
        with the given PPW, Elastic Medium, and other grid resolution
        information.
        
        Args:
            PPW (int): The number of points per wavelength to use 
                in creating a specific grid resolution at each
                obstacle.
        """
        # Clear obstacle list for this PPW 
        self.obstacles.clear()

        # Set up circular obstacles with grids, FD matrices, etc.
        for obstacle_geom in self.circular_obstacle_geometries:
            new_obs = Circular_MKFE_FDObstacle(
                center=obstacle_geom.center,
                r_obstacle=obstacle_geom.r_obstacle,
                r_artificial_boundary=obstacle_geom.r_artificial_bndry,
                boundary_condition=obstacle_geom.bc,
                num_farfield_terms=self.num_farfield_terms,
                parent_medium=self.medium,
                PPW=PPW
            )
            self.obstacles.append(new_obs)
            
        # TODO: Handle parametric obstacles

    def _solve_PPW_Gauss_Seidel(
        self,
        PPW: int,
        tol: float = 1e-6,
        maxiter: int = 100
    ) -> tuple[list[MKFE_FDObstacle], int]:
        """Solve the multiple scattering problem at a given PPW using
        Gauss Seidel algorithm.

        Assumes that self.obstacles is populated with a fresh set 
        of obstacles with unknowns all set to zeros.
        """
        logging.debug(f"Using Gauss-Seidel iteration with PPW = {PPW}, tol = {tol:.4e}, and maxiter = {maxiter}")

        prev_unknowns = np.hstack([obstacle.fd_unknowns for obstacle in self.obstacles])   # Will be all zeros to start out with 
        for itr in range(maxiter):
            logging.debug(f"Beginning iteration {itr}")
            cur_unknowns = np.array([])
            
            # Solve single scattering problems (using most currently
            # updated set of unknowns/farfield coefficients at each obstacle
            # during iteration when filling in forcing vectors)
            for i, obstacle in enumerate(self.obstacles):
                logging.debug(f"Solving Single-Scattering Problem at Obstacle ID {obstacle.id}")
                other_obstacles = self.obstacles[:i] + self.obstacles[i+1:]
                obstacle.solve(other_obstacles, self.incident_wave)
                
                # Store the updated values of these unknowns for comparison
                cur_unknowns = np.hstack((cur_unknowns, obstacle.fd_unknowns))
            
            # Check for convergence (in max norm)
            max_err = np.max(np.abs(cur_unknowns - prev_unknowns))
            prev_unknowns = cur_unknowns
            
            logging.debug(f"Max Error at iteration {itr}: {max_err:.5e}")
            if max_err < tol:
                logging.info(f"Gauss-Seidel Iteration Converged after {itr} iterations with max-norm error {max_err:.5e}")
                return self.obstacles, itr 
        
        # Getting here means that the algorithm did not converge.
        logging.exception(
            f"solve_PPW() with PPW={PPW} did not converge in {maxiter} iterations."
        )
        raise MaxIterationsExceedException(
            f"solve_PPW() with PPW={PPW} did not converge in {maxiter} iterations."
        )
    
    def solve_PPW(
        self,
        PPW: int,
        algorithm: Algorithm,
        tol: float = 1e-6,
        maxiter: int = 100
    ) -> tuple[list[MKFE_FDObstacle], int]:
        """Solve the multiple-elastic-scattering problem for grids
        with a resolution corresponding to a given number of 
        points per wavelength (PPW) to set obstacle grid resolution.
        
        Args:
            PPW (int): The number of points-per-wavelength to
                determine obstacle grid resolution.
            algorithm (algorithm): The iterative algorithm to use
                to solve the single-scattering systems
            tol (float): The minimum error tolerance between 
                successive iterations to determine convergence
            maxiter (int): The maximum number of iterations to
                run the given iterative algorithm

        Returns:
            list[MKFE_FDObstacle]: A list of the obstacles
                with solved fields and farfield coefficients
                ready for further analysis
            int: The number of iterations it took until convergence

        Raises:
            MaxIterationsExceededException: If the algorithm does 
                not converge to the given tolerance within the given
                maximum number of iterations
        """
        logging.debug(f"Entering solve_PPW() with PPW = {PPW}")

        # Create obstacle objects and initialize finite-difference
        # matrices for use during iteration
        self._setup_scattering_obstacles_for_PPW(PPW)

        # Iterate using the desired algorithm
        # TODO: (Potentially) Add GMRES as an option
        if algorithm is Algorithm.GAUSS_SEIDEL:
            return self._solve_PPW_Gauss_Seidel(PPW, tol, maxiter)
        else:
            raise ValueError(
                "Only supported iterative algorithm is Algorithm.GAUSS_SEIDEL at the moment"
            )
        

        
    

    
    










