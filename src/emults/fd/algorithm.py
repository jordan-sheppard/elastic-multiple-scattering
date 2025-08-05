from typing import Optional, Self
import numpy as np 
import json
import logging
import cloudpickle
from matplotlib import pyplot as plt
from matplotlib.contour import QuadContourSet
from scipy.interpolate import RegularGridInterpolator
from tabulate import tabulate
import sys  
import os 
import re
import gc 

from ..base.medium import LinearElasticMedium
from ..base.waves import IncidentPlanePWave
from ..base.exceptions import (
    MaxIterationsExceedException, AlgorithmDivergedException
)
from ..base.consts import Algorithm, ErrorType, ScalarQOI, ComplexArrayQuantity, PlotType, CoordinateSystem
from ..base.obstacles import BaseObstacle
from ..base.text_parsing import get_full_configuration_filename_base, get_filename_base
from .grids import FDPolarGrid_ArtBndry, FDLocalPolarGrid
from .obstacles import (
    MKFE_FDObstacle, Circular_MKFE_FDObstacle,
    CircularObstacleGeometry
)
from .solvers import FD_SparseLUSolver


class ErrorConvergenceRatesPolar:
    """A class for analyzing convergence rates on 
    polar grids."""
    def __init__(self):
        self.h = []
        self.grid_resolutions: list[tuple[int, int]] = []
        self.L2_errs = []
        self.L2_rates = [np.nan]
        self.L2_relative_errs = []
        self.L2_relative_rates = [np.nan]
        self.Linfty_errs = []
        self.Linfty_rates = [np.nan]        # Padding NaN since we really need 2 gridpoints to analyze convergence
    
    def _interpolate_grid_vals(
        self,
        source_grid: FDLocalPolarGrid,
        source_vals: np.ndarray,
        dest_grid: FDLocalPolarGrid,
    ) -> np.ndarray:
        """Interpolates source grid values onto a destination grid.
        
        source_vals should have the same 2D shape as source_grid.

        Assumes that the grids have the same bounds on both axes.
        """
        # Set up interpolation over source grid.
        r_gridpoints_source, theta_gridpoints_source  = source_grid.r_vals, source_grid.theta_vals
        theta_gridpoints_source_extended = np.array(list(theta_gridpoints_source) + [2 * np.pi])     # Allows 2pi (idenfitied with 0, the first gridpoint) to actually be an upper bound
        source_vals_extended = np.vstack((source_vals, source_vals[0]))     # Rolls over theta=0 gridpoints to be theta=2pi gridpoints
        
        source_grid_interpolation_func = RegularGridInterpolator(
            (theta_gridpoints_source_extended, r_gridpoints_source), 
            source_vals_extended,
            method='cubic'
        )
        
        # Get interpolation on destination grid
        r_dest_1d = np.ravel(dest_grid.r_local)
        theta_dest_1d = np.ravel(dest_grid.theta_local)
        gridpoints_of_interest = np.column_stack((theta_dest_1d, r_dest_1d))
        interpolated_values_1d = source_grid_interpolation_func(gridpoints_of_interest)
        
        # Reshape to be same shape as fine grid
        return np.reshape(interpolated_values_1d, shape=dest_grid.shape)


    def _approximate_error_orders(self):
        """Approximates the error order using Richardson
        extrapolation on the two most current error data points
        in the error arrays/grid parameter array."""
        if len(self.h) < 2:
            return      # Can't approximate error with less than 2 data points
        
        grid_param_ratio = np.log(self.h[-2] / self.h[-1])
        
        L2_order = np.log(self.L2_errs[-2] / self.L2_errs[-1]) / grid_param_ratio
        L2_relative_order =  np.log(self.L2_relative_errs[-2] / self.L2_relative_errs[-1]) / grid_param_ratio
        Linfty_order = np.log(self.Linfty_errs[-2] / self.Linfty_errs[-1]) / grid_param_ratio

        self.L2_rates.append(L2_order)
        self.L2_relative_rates.append(L2_relative_order)
        self.Linfty_rates.append(Linfty_order)


    def add_new_iteration_richardson_extrapolation(
        self,
        coarse_grid: FDLocalPolarGrid,
        fine_grid: FDLocalPolarGrid,
        coarse_vals: np.ndarray,
        fine_vals: np.ndarray
    ):
        """Gets convergence information from new iteration using a Richardson
        extrapolation scheme.
        
        Compares coarse grid values to fine grid values by interpolating coarse values
        on fine gridpoints.
        Uses cubic interpolation of coarse solution onto fine grid.
        """
        # Get coarse values on fine grid
        coarse_vals_interpolated = self._interpolate_grid_vals(coarse_grid, coarse_vals, fine_grid)

        # Get fine local polar grid radii (for L2 convergence)
        r_fine = fine_grid.r_local
        dr_fine = fine_grid.dr 
        dtheta_fine = fine_grid.dtheta

        # Compute AMPLITUDE ERROR quantities:
        # 1) Absolute value of difference between coarse and fine solutions
        # 2) Absolute value of fine solution (for relative-L2 norm analysis)
        amplitude_fine = np.abs(fine_vals)
        amplitude_diff = np.abs(np.abs(coarse_vals_interpolated) - amplitude_fine)
        
        # Approximate L2 norm, Relative L2 norm, and L-infinity norm errors
        # of the differences in amplitude for coarse-to-fine comparison
        L2_err = np.sqrt(
            np.sum(
                amplitude_diff**2 * r_fine * dr_fine * dtheta_fine
            )
        )
        self.L2_errs.append(L2_err)

        L2_norm_fine = np.sqrt(
            np.sum(
                amplitude_fine**2 * r_fine * dr_fine * dtheta_fine
            )
        )
        L2_relative_err = L2_err / L2_norm_fine
        self.L2_relative_errs.append(L2_relative_err)

        Linfty_err = np.max(amplitude_diff)
        self.Linfty_errs.append(Linfty_err)

        # Store grid parameter and resolution (coarse grid's dtheta used) 
        self.h.append(coarse_grid.dtheta) 
        self.grid_resolutions.append((coarse_grid.num_radial_gridpoints, coarse_grid.num_angular_gridpoints))

        # If possible, approximate error order for latest two datapoints 
        self._approximate_error_orders()

    def add_new_iteration_reference(
        self,
        coarse_grid: FDLocalPolarGrid,
        reference_grid: FDLocalPolarGrid,
        coarse_vals: np.ndarray,
        reference_vals: np.ndarray
    ):
        """Gets convergence/error information from new iteration by
        comparing to a reference solution.
        
        Compares coarse grid values to reference grid values,
        all on coarse gridpoints only, by interpolating the reference solution onto 
        the coarse gridpoints
        (using cubic interpolation of fine solution onto coarse grid for comparison).
        """
        # Get fine values on coarse grid
        reference_vals = self._interpolate_grid_vals(reference_grid, reference_vals, coarse_grid)

        # Get coarse local polar grid radii (for L2 convergence)
        r_coarse = coarse_grid.r_local

        # Compute AMPLITUDE ERROR quantities:
        # 1) Absolute value of difference between coarse and fine solutions
        # 2) Absolute value of fine solution (for relative-L2 norm analysis)
        amplitude_reference = np.abs(reference_vals)
        amplitude_diff = np.abs(np.abs(coarse_vals) - amplitude_reference)
        
        # Approximate L2 norm, Relative L2 norm, and L-infinity norm errors
        # of the differences in amplitude for coarse-to-fine comparison
        L2_err = np.sqrt(
            np.sum(
                amplitude_diff**2 * r_coarse * coarse_grid.dr * coarse_grid.dtheta
            )
        )
        self.L2_errs.append(L2_err)

        L2_norm_reference = np.sqrt(
            np.sum(
                amplitude_reference**2 * r_coarse * coarse_grid.dr * coarse_grid.dtheta
            )
        )
        L2_relative_err = L2_err / L2_norm_reference
        self.L2_relative_errs.append(L2_relative_err)

        Linfty_err = np.max(amplitude_diff)
        self.Linfty_errs.append(Linfty_err)

        # Store grid parameter and resolution (coarse grid's dtheta used) 
        self.h.append(coarse_grid.dtheta) 
        self.grid_resolutions.append((coarse_grid.num_radial_gridpoints, coarse_grid.num_angular_gridpoints))

        # If possible, approximate error order for latest two datapoints 
        self._approximate_error_orders()
  

class ScatteringConvergenceAnalyzerPolar:
    """A class for analyzing convergence on a problem with no 
    analytical solution.
    """
    def __init__(self, num_obstacles: int):
        """Initialize needed parameters"""
        self.PPWs: list[int] = []
        self.phi_rates = {
            i: ErrorConvergenceRatesPolar()
            for i in range(num_obstacles)
        }
        self.psi_rates = {
            i: ErrorConvergenceRatesPolar()
            for i in range(num_obstacles)
        }

    def analyze_convergence_reference(
        self,
        solved_obstacles: dict[int, list[Circular_MKFE_FDObstacle]],
        reference_solution: list[Circular_MKFE_FDObstacle]
    ):
        """Analyzes convergence at various PPW resolutions for
        a given obstacle setup by comparing to a refined reference
        grid solution.
        
        Args:
            solved_obstacles: A dictionary whose keys are PPWs resolutions,
                and whose values are a list of solved obstacles at
                that resolution value (should be same obstacles with
                same IDs, but more refined grids as PPW increases)
        """
        # Reset PPW and phi/psi convergence rates lists
        self.PPWs = sorted(list(solved_obstacles.keys()))
        for i in self.phi_rates.keys():
            self.phi_rates[i] = ErrorConvergenceRatesPolar()
            self.psi_rates[i] = ErrorConvergenceRatesPolar()

        solved_obstacles_coarse_to_fine = sorted(solved_obstacles.items())
        for i in range(len(solved_obstacles_coarse_to_fine)):
            # Get coarse/fine obstacles at this resolution lavel
            coarse_obstacles = solved_obstacles_coarse_to_fine[i][1]

            # Sort obstacles by id 
            def sort_by_id(obstacle: Circular_MKFE_FDObstacle):
                return obstacle.id
            coarse_obstacles.sort(key=sort_by_id)
            reference_solution.sort(key=sort_by_id)

            for i, (coarse_obs, reference_obs) in enumerate(zip(coarse_obstacles, reference_solution)):
                # Validate we're talking about the same obstacle 
                if coarse_obs.id != reference_obs.id:
                    raise RuntimeError("ERROR: Sorted obstacle IDs don't match up!")
                obs_id = coarse_obs.id 
                
                # Get scattered phi/psi values on each obstacle grid
                other_coarse_obstacles = coarse_obstacles[:i] + coarse_obstacles[i+1:]
                other_fine_obstacles = reference_solution[:i] + reference_solution[i+1:]

                scattered_phi_coarse = coarse_obs.get_total_phi(other_obstacles=other_coarse_obstacles)
                scattered_psi_coarse = coarse_obs.get_total_psi(other_obstacles=other_coarse_obstacles)
                scattered_phi_reference = reference_obs.get_total_phi(other_obstacles=other_fine_obstacles)
                scattered_psi_reference = reference_obs.get_total_psi(other_obstacles=other_fine_obstacles)

                # Analyze convergence at this step 
                self.phi_rates[obs_id].add_new_iteration_reference(
                    coarse_grid=coarse_obs.grid,
                    reference_grid=reference_obs.grid,
                    coarse_vals=scattered_phi_coarse,
                    reference_vals=scattered_phi_reference
                )
                self.psi_rates[obs_id].add_new_iteration_reference(
                    coarse_grid=coarse_obs.grid,
                    reference_grid=reference_obs.grid,
                    coarse_vals=scattered_psi_coarse,
                    reference_vals=scattered_psi_reference
                )

    def analyze_convergence_richardson_extrapolation(
        self,
        solved_obstacles: dict[int, list[Circular_MKFE_FDObstacle]]
    ):
        """Analyzes convergence at various PPW resolutions for
        a given obstacle setup using a point-by-point coarse-to-fine
        comparison.
        
        Args:
            solved_obstacles: A dictionary whose keys are PPWs resolutions,
                and whose values are a list of solved obstacles at
                that resolution value (should be same obstacles with
                same IDs, but more refined grids as PPW increases)
        """
        # Reset PPW and phi/psi convergence rates lists
        self.PPWs = sorted(list(solved_obstacles.keys()))
        for i in self.phi_rates.keys():
            self.phi_rates[i] = ErrorConvergenceRatesPolar()
            self.psi_rates[i] = ErrorConvergenceRatesPolar()

        solved_obstacles_coarse_to_fine = sorted(solved_obstacles.items())
        for i in range(len(solved_obstacles_coarse_to_fine) - 1):
            # Get coarse/fine obstacles at this resolution lavel
            coarse_obstacles = solved_obstacles_coarse_to_fine[i][1]
            fine_obstacles = solved_obstacles_coarse_to_fine[i+1][1]

            # Sort obstacles by id 
            def sort_by_id(obstacle: Circular_MKFE_FDObstacle):
                return obstacle.id
            coarse_obstacles.sort(key=sort_by_id)
            fine_obstacles.sort(key=sort_by_id)

            for i, (coarse_obs, fine_obs) in enumerate(zip(coarse_obstacles, fine_obstacles)):
                # Validate we're talking about the same obstacle 
                if coarse_obs.id != fine_obs.id:
                    raise RuntimeError("ERROR: Sorted obstacle IDs don't match up!")
                obs_id = coarse_obs.id 
                
                # Get scattered phi/psi values on each obstacle grid
                other_coarse_obstacles = coarse_obstacles[:i] + coarse_obstacles[i+1:]
                other_fine_obstacles = fine_obstacles[:i] + fine_obstacles[i+1:]

                scattered_phi_coarse = coarse_obs.get_total_phi(other_obstacles=other_coarse_obstacles)
                scattered_psi_coarse = coarse_obs.get_total_psi(other_obstacles=other_coarse_obstacles)
                scattered_phi_fine = fine_obs.get_total_phi(other_obstacles=other_fine_obstacles)
                scattered_psi_fine = fine_obs.get_total_psi(other_obstacles=other_fine_obstacles)

                # Analyze convergence at this step 
                self.phi_rates[obs_id].add_new_iteration_richardson_extrapolation(
                    coarse_grid=coarse_obs.grid,
                    fine_grid=fine_obs.grid,
                    coarse_vals=scattered_phi_coarse,
                    fine_vals=scattered_phi_fine
                )
                self.psi_rates[obs_id].add_new_iteration_richardson_extrapolation(
                    coarse_grid=coarse_obs.grid,
                    fine_grid=fine_obs.grid,
                    coarse_vals=scattered_psi_coarse,
                    fine_vals=scattered_psi_fine
                )

    def _print_convergence_tables(
        self,
        convergence_rates: dict[int, ErrorConvergenceRatesPolar], 
        quantity: str,
        fmt: str = 'fancy_grid',
        file: Optional[str] = None,
        append: bool = False
    ) -> None:
        """Displays a given convergence rate table for a given quantity"""
        mode = "a" if append else "w"       # Whether to write or append to file
        if file is not None:
            with open(file, mode) as outfile:
                self._write_convergence_tables(outfile, convergence_rates, quantity, fmt)
        else:
            self._write_convergence_tables(sys.stdout, convergence_rates, quantity, fmt)

    def _write_convergence_tables(self, output, convergence_rates, quantity, fmt):
        sorted_obstacle_ids = sorted(convergence_rates.keys())
        obstacle_table_headers = ["PPW", "h", "Grid Resolution", "L2 Error", "L2 Observed Order", "Relative L2 Error",  "Relative L2 Observed Order", "L-Infinity Error", "L-Infinity Observed Order"]
        print("\n===========================================================", file=output)
        print(f"             CONVERGENCE ANALYSIS: {quantity}             ", file=output)
        print("===========================================================\n", file=output)
        
        for obs_id in sorted_obstacle_ids:
            convergence_rate = convergence_rates[obs_id]
            table = [
                [PPW, h, f"{resolution[0]} x {resolution[1]}", L2, L2_order, L2rel, L2rel_order, Linfty, Linfty_order]
                for PPW, h, resolution, L2, L2_order, L2rel, L2rel_order, Linfty, Linfty_order in zip(
                    self.PPWs, convergence_rate.h, convergence_rate.grid_resolutions,
                    convergence_rate.L2_errs, convergence_rate.L2_rates,
                    convergence_rate.L2_relative_errs, convergence_rate.L2_relative_rates,
                    convergence_rate.Linfty_errs, convergence_rate.Linfty_rates
                )
            ]
            print(f"\n---------------- OBSTACLE {obs_id} ----------------\n", file=output)
            print(
                tabulate(
                    table,
                    headers=obstacle_table_headers,
                    tablefmt=fmt
                ), file=output
            )

    def _draw_convergence_plots(
        self,
        convergence_rates: dict[int, ErrorConvergenceRatesPolar], 
        quantity: str,
        folder: Optional[str] = None
    ) -> None:
        # Define the plot function to save to a file if a folder is provided
        # and to plot it to a screen otherwise 
        def plotfig(figure_name: str) -> None:
            if folder is not None:
                figure_path = os.path.join(folder, f"{figure_name}.png")
                plt.savefig(figure_path)
                plt.clf()
            else:
                plt.show()
        
        quantity_no_latex_formatting = re.sub(r'[$\\]', '', quantity)   # For figure saving filename purposes

        # Get errors for each obstacle
        h = []
        L2_errs = []
        L2_relative_errs = []
        Linfty_errs = []
        sorted_obstacle_ids = sorted(convergence_rates.keys())
        for obs_id in sorted_obstacle_ids:
            h.append(convergence_rates[obs_id].h)
            L2_errs.append(convergence_rates[obs_id].L2_errs)
            L2_relative_errs.append(convergence_rates[obs_id].L2_relative_errs)
            Linfty_errs.append(convergence_rates[obs_id].Linfty_errs)

        # Plot L2 errors
        for i, h_vals, obstacle_errs in zip(sorted_obstacle_ids, h, L2_errs):
            plt.loglog(h_vals, obstacle_errs, '.-', label=f"Obstacle {i}")

        plt.legend()
        plt.title(f"{quantity} - $L_2$ Errors")
        plt.xlabel("Grid Parameter $h$")
        plt.ylabel("Error")
        plotfig(f'L2_errors_{quantity_no_latex_formatting}')

        # Plot Relative L2 errors
        for i, h_vals, obstacle_errs in zip(sorted_obstacle_ids, h, L2_relative_errs):
            plt.loglog(h_vals, obstacle_errs, '.-', label=f"Obstacle {i}")

        plt.legend()
        plt.title(f"{quantity} - Relative $L_2$ Errors")
        plt.xlabel("Grid Parameter $h$")
        plt.ylabel("Error")
        plotfig(f'Relative_L2_errors_{quantity_no_latex_formatting}')

        # Plot L-Infinity errors
        for i, h_vals, obstacle_errs in zip(sorted_obstacle_ids, h, Linfty_errs):
            plt.loglog(h_vals, obstacle_errs, '.-', label=f"Obstacle {i}")

        plt.legend()
        plt.title(rf"{quantity} - $L_{{\infty}}$ Errors")
        plt.xlabel("Grid Parameter $h$")
        plt.ylabel("Error")
        plotfig(f'L_Infinity_errors_{quantity_no_latex_formatting}')
        plt.show()

    
    def display_convergence(
        self,
        table_fmt: str = 'fancy_grid',
        text: bool = True,
        plots: bool = True,
        text_filepath: Optional[str] = None,
        plots_folderpath: Optional[str] = None
    ):
        if text:
            # Display phi/psi convergence tables
            self._print_convergence_tables(
                convergence_rates=self.phi_rates,
                quantity='phi',
                fmt=table_fmt,
                file=text_filepath,
                append=False
            )
            self._print_convergence_tables(
                convergence_rates=self.psi_rates,
                quantity='psi',
                fmt=table_fmt,
                file=text_filepath,
                append=True
            )

        if plots:
            # Display phi/psi convergence plots
            self._draw_convergence_plots(
                convergence_rates=self.phi_rates,
                quantity=r'$\phi$',
                folder=plots_folderpath
            )
            self._draw_convergence_plots(
                convergence_rates=self.psi_rates,
                quantity=r'$\psi$',
                folder=plots_folderpath
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
        obstacle_config_file: str,
        medium_config_file: str,
        numerical_config_file: str,
        reference_config_file: Optional[str] = None,
        reference_cache_folder: Optional[str] = None,
        normal_cache_folder: Optional[str] = None
    ):
        """Initialize a multiple-scattering problem.
        
        Args:
            medium (LinearElasticMedium): The elastic medium where
                the scattering problem is taking place 
            obstacle_config_file (str): A path to the obstacle
                configuration JSON file that contains information
                about obstacle geometries and boundary conditions.
            reference_solution (MKFE_FD_ScatteringProblem): An optional
                reference solution on a very refined grid. If multiple 
                grids are provided, the most refined is taken to be
                the reference solution.
        """
        # Get medium and incident wave info from config
        # (populates self.medium and self.incident_wave)
        self._add_elastic_medium_from_config(medium_config_file)
        self._add_incident_wave_from_config(medium_config_file)

        # Get obstacle geometry info from config 
        # (populates self.circular_obstacle_geometries)
        self._add_obstacle_geometries_from_config(obstacle_config_file)

        # Get numerical method info from config
        # (populates self.PPWs and self.num_farfield_terms)
        self._add_numerical_method_info_from_config(numerical_config_file)

        # Create a place to store fully-fleshed-out
        # obstacles of interest that can be used 
        # to iteratively update unknowns.
        # Keys are PPW values, values are lists of obstacles
        self.obstacles: dict[int, list[MKFE_FDObstacle]] = dict()

        # Create an object for performing convergence analysis 
        self.convergence_analyzer = ScatteringConvergenceAnalyzerPolar(
            num_obstacles=len(self.circular_obstacle_geometries)
        ) 

        # Create a name for this scattering problem (for recalling
        # from storage)
        self.obstacle_config_label = get_filename_base(obstacle_config_file)
        self.medium_config_label = get_filename_base(medium_config_file)
        self.numerical_config_label = get_filename_base(numerical_config_file)

        self.problem_label = get_full_configuration_filename_base(
            obstacle_config=obstacle_config_file,
            medium_config=medium_config_file,
            numerical_config=numerical_config_file,
            reference_config=reference_config_file
        )
        
        if reference_cache_folder is None:
            reference_cache_folder = 'cache/obstacles/reference'
        if not os.path.isdir(reference_cache_folder):
            os.makedirs(reference_cache_folder)
        self.reference_cache_folder = reference_cache_folder

        if normal_cache_folder is None:
            normal_cache_folder = 'cache/obstacles/other'
        if not os.path.isdir(normal_cache_folder):
            os.makedirs(normal_cache_folder)
        self.normal_cache_folder = normal_cache_folder

        # Get reference solution
        if reference_config_file is not None:
            self._get_reference_solution(reference_config_file)


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
        self.circular_obstacle_geometries = []
        obstacle_geometries_circ = config['obstacles']['circular'] 
        for obstacle_geometry_info in obstacle_geometries_circ:
            obstacle_geometry = CircularObstacleGeometry.from_config_entry(obstacle_geometry_info)
            self.circular_obstacle_geometries.append(obstacle_geometry)
        
        # TODO: HANDLE PARAMETRIC GEOMETRIES

    def _add_elastic_medium_from_config(self, config_file:str) -> None:
        with open(config_file, 'r') as in_json_file:
            config = json.load(in_json_file)
        
        # Parse elastic medium
        medium_info = config['medium']
        nu = medium_info['nu']
        E = medium_info['E']
        lambda_s = medium_info['lambda_s']
        omega = medium_info['omega']

        self.medium = LinearElasticMedium.from_young_poisson_lambda_s(
            E=E,
            nu=nu,
            lambda_s=lambda_s,
            omega=omega
        )

    def _add_incident_wave_from_config(self, config_file:str) -> None:
        with open(config_file, 'r') as in_json_file:
            config = json.load(in_json_file)
        
        # Parse incident wave
        incident_info = config['incident_wave']
        angle_of_inc = incident_info['angle_of_inc']
        wavenumber = incident_info['wavenumber']
        self.incident_wave = IncidentPlanePWave(
            angle_of_incidence=angle_of_inc,
            wavenumber=wavenumber
        )

    def _add_numerical_method_info_from_config(self, config_file:str) -> None:
        with open(config_file, 'r') as in_json_file:
            config = json.load(in_json_file)
        
        # Parse numerical method info
        self.PPWs = [int(PPW) for PPW in config['PPWs']]
        self.num_farfield_terms = int(config['num_farfield_terms'])
        self.experimental_tol = config['tol']
        self.experimental_maxiter = config['maxiter']

    def _get_reference_solution(self, reference_config_file:str) -> None:
        with open(reference_config_file, 'r') as in_json_file:
            config = json.load(in_json_file)

        # Parse reference solution numerical method info 
        self.reference_config_label = get_filename_base(reference_config_file)
        
        self.reference_PPW = [int(PPW) for PPW in config['PPWs']][0]
        self.reference_num_farfield_terms = int(config['num_farfield_terms'])
        self.reference_tol = config['tol']
        self.reference_maxiter = config['maxiter']

        # Get the reference solution
        logging.info(f"Loading Reference Solution (PPW={self.reference_PPW}, Num Farfield Terms={self.reference_num_farfield_terms}) . . .")
        self.reference_solution, _ = self.solve_PPW(self.reference_PPW, Algorithm.GAUSS_SEIDEL, reference=True, cache=True)
        logging.info("Done loading reference solution.")

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
        if PPW in self.obstacles:
            self.obstacles[PPW].clear()
        else:
            self.obstacles[PPW] = []

        # Reset obstacle IDs for standardized comparisons from PPW
        # to PPW (for convergence, etc.)
        BaseObstacle.reset_id_counter()

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
            self.obstacles[PPW].append(new_obs)
            
        # TODO: Handle parametric obstacles

    def _cache_PPW_solution(
        self,
        PPW: int,
        solution: list[MKFE_FDObstacle], 
        reference: bool = False
    ) -> None:
        """Cache a given solution at a given PPW resolution.
        
        Caches are stored in the filepath
        results/obstacles/{obstacle_config}/{medium_config}_{numerical_config}/PPW_{PPW}.pickle
        """
        # Create cache directory for this setup if it hasn't already been created
        if reference:
            cache_folder = self.reference_cache_folder
        else:
            cache_folder = self.normal_cache_folder

        # Now, cache this obstacle setup if it doesn't exist 
        cache_file = os.path.join(cache_folder, f"PPW_{PPW}.pickle")
        if not os.path.exists(cache_file):
            with open(cache_file, 'wb') as outfile:
                cloudpickle.dump(solution, outfile)

    def _check_cached_PPW_solution(
        self, 
        PPW: int,
        reference: bool = False
    ) -> Optional[list[MKFE_FDObstacle]]:
        """Checks for cached solution of this obstacle configuration
        at a given PPW for a given numerical solution set of parameters
        (useful for distinguishing between reference and trial solutions)"""
        if reference:
            cache_file = os.path.join(
                self.reference_cache_folder,
                f"PPW_{PPW}.pickle"
            )
        else:
            cache_file = os.path.join(
                self.normal_cache_folder,
                f"PPW_{PPW}.pickle"
            )
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as infile:
                return cloudpickle.load(infile)
        else:
            return None 

    def _solve_PPW_Gauss_Seidel(
        self,
        PPW: int,
        reference: bool = False,
        cache: bool = False
    ) -> tuple[list[MKFE_FDObstacle], int]:
        """Solve the multiple scattering problem at a given PPW using
        Gauss Seidel algorithm.

        Assumes that self.obstacles is populated with a fresh set 
        of obstacles with unknowns all set to zeros.
        """
        if reference:
            tol = self.reference_tol
            maxiter = self.reference_maxiter
        else:
            tol = self.experimental_tol 
            maxiter = self.experimental_maxiter
        logging.info(f"Setting up Sparse LU solvers for PPW = {PPW}")
        lu_solvers = [FD_SparseLUSolver(obstacle.fd_matrix) for obstacle in self.obstacles[PPW]]

        
        try:
            logging.info(f"Using Gauss-Seidel iteration with PPW = {PPW}, tol = {tol:.4e}, and maxiter = {maxiter}")
            prev_unknowns = np.hstack([obstacle.fd_unknowns for obstacle in self.obstacles[PPW]])   # Will be all zeros to start out with 
            for itr in range(maxiter):
                logging.debug(f"Beginning iteration {itr}")
                cur_unknowns = np.array([])
                
                # Solve single scattering problems (using most currently
                # updated set of unknowns/farfield coefficients at each obstacle
                # during iteration when filling in forcing vectors)
                for i, (obstacle, solver) in enumerate(zip(self.obstacles[PPW], lu_solvers)):
                    logging.debug(f"Solving Single-Scattering Problem at Obstacle ID {obstacle.id}")
                    other_obstacles = self.obstacles[PPW][:i] + self.obstacles[PPW][i+1:]
                    obstacle.solve(solver, other_obstacles, self.incident_wave)
                    
                    # Store the updated values of these unknowns for comparison
                    cur_unknowns = np.hstack((cur_unknowns, obstacle.fd_unknowns))
                
                # Check for convergence (in max norm)
                max_err = np.max(np.abs(cur_unknowns - prev_unknowns))
                prev_unknowns = cur_unknowns
                
                logging.info(f"Max Error at iteration {itr}: {max_err:.5e}")
                if max_err < tol:
                    logging.info(f"Gauss-Seidel Iteration Converged for PPW = {PPW} after {itr} iterations with max-norm error {max_err:.5e}")
                    if cache:
                        self._cache_PPW_solution(PPW, self.obstacles[PPW], reference)
                    return self.obstacles[PPW], itr     # Only cache if converged
                
                # Check that we didn't have something really bad happen
                if (not np.isfinite(max_err)) or (max_err > 1e8):
                    logging.exception(f"ERROR: Gauss-Seidel Iteration Diverged for PPW = {PPW} after {itr} iterations.")
                    if cache:
                        self._cache_PPW_solution(PPW, self.obstacles[PPW], reference)
                    raise AlgorithmDivergedException(f"Gauss-Seidel Iteration Diverged for PPW = {PPW} after {itr} iterations.")
                
            
            # Getting here means that the algorithm did not converge.
            logging.exception(
                f"solve_PPW() with PPW={PPW} did not converge in {self.experimental_maxiter} iterations."
            )
            if cache:
                self._cache_PPW_solution(PPW, self.obstacles[PPW], reference)
            raise MaxIterationsExceedException(
                f"solve_PPW() with PPW={PPW} did not converge in {self.experimental_maxiter} iterations."
            )
        finally:
            # Delete RAM-intensive solvers
            del lu_solvers
            gc.collect()    

    def solve_PPW(
        self,
        PPW: int,
        algorithm: Algorithm,
        reference: bool = False,
        cache: bool = False
    ) -> tuple[list[MKFE_FDObstacle], int]:
        """Solve the multiple-elastic-scattering problem for grids
        with a resolution corresponding to a given number of 
        points per wavelength (PPW) to set obstacle grid resolution.
        
        Args:
            PPW (int): The number of points-per-wavelength to
                determine obstacle grid resolution.
            algorithm (algorithm): The iterative algorithm to use
                to solve the single-scattering systems

        Returns:
            list[MKFE_FDObstacle]: A list of the obstacles
                with solved fields and farfield coefficients
                ready for further analysis
            int: The number of iterations it took until convergence

        Raises:
            MaxIterationsExceededException: If the algorithm does 
                not converge to the given tolerance within the given
                maximum number of iterations
            AlgorithmDivergedException: If the algorithm diverged to
                Nan/Infinity (or an untolerably large number) in error 
                before termination
            ValueError: If an invalid algorithm type is provided.
        """
        logging.debug(f"Entering solve_PPW() with PPW = {PPW}")

        ## Check for cached versions of these obstacles
        if reference:
            cached_solution = self._check_cached_PPW_solution(PPW, reference=True) 
        else:
            cached_solution = self._check_cached_PPW_solution(PPW, reference=False)
        
        if cached_solution is not None:
            if not reference:
                self.obstacles[PPW] = cached_solution
            return cached_solution, -1  # -1 denotes we got it from cache

        ## Otherwise, run an iterative algorithm to solve the multiple
        ## scattering problem at this resolution
        
        # Create obstacle objects and initialize finite-difference
        # matrices for use during iteration
        self._setup_scattering_obstacles_for_PPW(PPW)

        # Iterate using the desired algorithm
        if algorithm is Algorithm.GAUSS_SEIDEL:
            solution = self._solve_PPW_Gauss_Seidel(PPW, reference, cache)
            if reference:
                self.obstacles.pop(PPW)
            return solution
        else:
            raise ValueError(
                "Only supported iterative algorithm is Algorithm.GAUSS_SEIDEL at the moment"
            )
        
    def solve_PPWs(
        self,
        algorithm: Algorithm,
        reference: bool = False,
        pickle: bool = False, 
        pickle_folder: Optional[str] = None,
        cache_all: bool = False
    ) -> None:
        """Solves the algorithm at the various PPW values given at initialization.
        Stores results from each PPW value in PPWs in self.obstacles[PPW].
        
        If desired, pickles the results object for further analysis
        later on.
        """
        for PPW in self.PPWs:
            try:
                self.solve_PPW(PPW, algorithm, reference=False, cache=cache_all)
            except (AlgorithmDivergedException, MaxIterationsExceedException):
                logging.exception(f"Saved PPW={PPW} at corrupted state. Continuing on with remaining PPWs...")

        # Analyze convergence if desired (that is, if it's not a reference solution)
        if not reference:
            self.analyze_convergence()

        # If desired, pickle this object for further analysis
        if pickle:
            if pickle_folder is None:
                pickle_folder = 'cache/simulation'
            elif not os.path.isdir(pickle_folder):
                os.makedirs(pickle_folder)

            outfile_name = os.path.join(pickle_folder, f"{self.problem_label}.pickle")
            with open(outfile_name, 'wb') as outfile:
                cloudpickle.dump(self, outfile)
            

    def analyze_convergence(self) -> ScatteringConvergenceAnalyzerPolar:
        """Analyzes the approximate convergence rate.
        
        Uses Richardson extrapolation if no reference solution is provided.
        Otherwise, uses a standard convergence analysis, treating the reference
        solution as the exact solution on the provided grid.
        """
        if hasattr(self, 'reference_solution') and self.reference_solution is not None:
            self.convergence_analyzer.analyze_convergence_reference(
                solved_obstacles=self.obstacles,
                reference_solution=self.reference_solution
            )
        else:
            # Analyze by point-by-point Richardson extrapolation
            # from coarse to fine grids
            if len(self.obstacles.keys()) < 3:
                raise ValueError("Cannot analyze convergence using Richardson extrapolation unless there are 3 distinct PPW solutions.")

            self.convergence_analyzer.analyze_convergence_richardson_extrapolation(self.obstacles)
        
        return self.convergence_analyzer


    def plot_scalar_field_at_obstacles(
        self,
        PPW: int,
        field_vals: list[np.ndarray],
        vmin: float,
        vmax:float,
        title: str,
        plot_folder:Optional[str] = None,
        plot_filename: Optional[str] = None
    ) -> None:
        for obstacle, vals in zip(self.obstacles[PPW], field_vals):
            # Plot field values
            quad_contour_set = obstacle.plot_contourf(vals, vmin=vmin, vmax=vmax)

         # Title and show the plot
        plt.title(title)
        plt.colorbar(quad_contour_set)
        if plot_folder is None:
            plt.show()
        else:
            plot_img_path = os.path.join(plot_folder, plot_filename)
            plt.savefig(plot_img_path)
            plt.clf()
    

    def get_scalars_for_plotting(
        self, 
        PPW: int,
        scalar_qoi: ScalarQOI,
        complex_array_quantity: ComplexArrayQuantity,
        plot_type: PlotType
    ) -> tuple[list[np.ndarray], float, float]:
        """Gets the desired scalar QOI (in absolute value, real,
        or imaginary form), either scattered or total, at each
        obstacle. Factors in contributions from all participating
        obstacles.
        
        Returns:
            list[np.ndarray] - The i'th entry is an array of quantities
                as the same shape of the obstacle local grid 
                at self.obstacles[PPW][i]
            float - The absolute minimum value encountered 
            float - The absolute maximum value encountered
        """
        vmin = np.inf 
        vmax = -np.inf 
        values_at_obstacles = []
        for obstacle in self.obstacles[PPW]:
            # Get other obstacle information 
            other_obstacles = []
            for other_obstacle in self.obstacles[PPW]:
                if other_obstacle.id != obstacle.id:
                    other_obstacles.append(other_obstacle)

            # If total wave desired, get effect of incident wave.
            # Otherwise, ignore it
            if plot_type is PlotType.SCATTERED:
                u_inc = None 
            elif plot_type is PlotType.TOTAL:
                u_inc = self.incident_wave
            else:
                raise ValueError(f"Unrecognized Plot Type {plot_type}")

            # Get desired potential (phi or psi)
            if scalar_qoi is ScalarQOI.PHI:
                vals = obstacle.get_total_phi(
                    u_inc=u_inc,
                    other_obstacles=other_obstacles
                )
            elif scalar_qoi is ScalarQOI.PSI:
                vals = obstacle.get_total_psi(
                    u_inc=u_inc,
                    other_obstacles=other_obstacles
                )
            elif scalar_qoi is ScalarQOI.DISPLACEMENT_X:
                vals = obstacle.get_total_displacement(
                    incident_wave=u_inc,
                    other_obstacles=other_obstacles,
                    coordinate_system=CoordinateSystem.GLOBAL_CARTESIAN
                )[:,:,0]
            elif scalar_qoi is ScalarQOI.DISPLACEMENT_Y:
                vals = obstacle.get_total_displacement(
                    incident_wave=u_inc,
                    other_obstacles=other_obstacles,
                    coordinate_system=CoordinateSystem.GLOBAL_CARTESIAN
                )[:,:,1]
            elif scalar_qoi is ScalarQOI.STRESS_XX:
                obstacle:Circular_MKFE_FDObstacle
                vals = obstacle.get_total_stress(
                    incident_wave=u_inc,
                    other_obstacles=other_obstacles,
                    coordinate_system=CoordinateSystem.GLOBAL_CARTESIAN
                )[:,:,0]
            elif scalar_qoi is ScalarQOI.STRESS_XY:
                obstacle:Circular_MKFE_FDObstacle
                vals = obstacle.get_total_stress(
                    incident_wave=u_inc,
                    other_obstacles=other_obstacles,
                    coordinate_system=CoordinateSystem.GLOBAL_CARTESIAN
                )[:,:,1]
            elif scalar_qoi is ScalarQOI.STRESS_YY:
                obstacle:Circular_MKFE_FDObstacle
                vals = obstacle.get_total_stress(
                    incident_wave=u_inc,
                    other_obstacles=other_obstacles,
                    coordinate_system=CoordinateSystem.GLOBAL_CARTESIAN
                )[:,:,2]
            else:
                raise ValueError(f"Unrecognized Potential Type {scalar_qoi}")
            
            # Parse desired complex potential into real scalar 
            # according to given method 
            if complex_array_quantity is ComplexArrayQuantity.ABS:
                vals = np.abs(vals)
            elif complex_array_quantity is ComplexArrayQuantity.REAL:
                vals = np.real(vals)
            elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
                vals = np.imag(vals)
            else:
                raise ValueError(f"Unrecognized Complex Array Quantity {ComplexArrayQuantity}")
            
            # Update absolute max/min by inspecting max/min of these values 
            vmax = np.max([vmax, np.max(vals)])
            vmin = np.min([vmin, np.min(vals)])

            # Store values to return at this obstacle 
            values_at_obstacles.append(vals)
        return values_at_obstacles, vmin, vmax


    def plot_total_phi(
        self,
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None
    ):
        """Plot total phi for a given PPW solution."""
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get total phi at each obstacle
        total_potentials, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.PHI,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.TOTAL
        )

        # Plot the contourf plot of the total phi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Total $\phi$ (Amplitude)'
            plot_filename = 'phi_total_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Total $\phi$ (Real Part)'
            plot_filename = 'phi_total_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Total $\phi$ (Imaginary Part)'
            plot_filename = 'phi_total_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=total_potentials,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_phi(
        self,
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None
    ):
        """Plot scattered phi for a given PPW solution."""
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered phi at each obstacle
        total_potentials, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.PHI,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.SCATTERED
        )

        # Plot the contourf plot of the scattered phi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Scattered $\phi$ (Amplitude)'
            plot_filename = 'phi_scattered_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Scattered $\phi$ (Real Part)'
            plot_filename = 'phi_scattered_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Scattered $\phi$ (Imaginary Part)'
            plot_filename = 'phi_scattered_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=total_potentials,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_total_psi(
        self,
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None
    ):
        """Plot total phi for a given PPW solution."""
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get total psi at each obstacle
        total_potentials, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.PSI,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.TOTAL
        )

        # Plot the contourf plot of the total psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Total $\psi$ (Amplitude)'
            plot_filename = 'psi_total_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Total $\psi$ (Real Part)'
            plot_filename = 'psi_total_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Total $\psi$ (Imaginary Part)'
            plot_filename = 'psi_total_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=total_potentials,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_psi(
        self,
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None
    ):
        """Plot scattered psi for a given PPW solution."""
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered psi at each obstacle
        total_potentials, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.PSI,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.SCATTERED
        )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Scattered $\psi$ (Amplitude)'
            plot_filename = 'psi_scattered_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Scattered $\psi$ (Real Part)'
            plot_filename = 'psi_scattered_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Scattered $\psi$ (Imaginary Part)'
            plot_filename = 'psi_scattered_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=total_potentials,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_total_x_displacement(
        self, 
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        """Plot total x-direction displacement for a given PPW solution
        as a scalar heatmap/contourf plot.
        """
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered psi at each obstacle
        total_x_displacement, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.DISPLACEMENT_X,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.TOTAL
        )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Total x-Displacement $\mathbf{u}_x$ (Amplitude)'
            plot_filename = 'x_displacement_total_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Total x-Displacement $\mathbf{u}_x$ (Real Part)'
            plot_filename = 'x_displacement_total_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Total x-Displacement $\mathbf{u}_x$ (Imaginary Part)'
            plot_filename = 'x_displacement_total_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=total_x_displacement,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_x_displacement(
        self, 
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        """Plot scattered x-direction displacement for a given PPW solution
        as a scalar heatmap/contourf plot.
        """
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered psi at each obstacle
        scattered_x_displacement, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.DISPLACEMENT_X,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.SCATTERED
        )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Scattered x-Displacement $\mathbf{u}_x$ (Amplitude)'
            plot_filename = 'x_displacement_scattered_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Scattered x-Displacement $\mathbf{u}_x$ (Real Part)'
            plot_filename = 'x_displacement_scattered_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Scattered x-Displacement $\mathbf{u}_x$ (Imaginary Part)'
            plot_filename = 'x_displacement_scattered_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=scattered_x_displacement,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    
    def plot_total_y_displacement(
        self, 
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        """Plot total y-direction displacement for a given PPW solution
        as a scalar heatmap/contourf plot.
        """
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered psi at each obstacle
        total_y_displacement, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.DISPLACEMENT_Y,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.TOTAL
        )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Total y-Displacement $\mathbf{u}_y$ (Amplitude)'
            plot_filename = 'y_displacement_total_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Total y-Displacement $\mathbf{u}_y$ (Real Part)'
            plot_filename = 'y_displacement_total_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Total y-Displacement $\mathbf{u}_y$ (Imaginary Part)'
            plot_filename = 'y_displacement_total_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=total_y_displacement,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_y_displacement(
        self, 
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        """Plot scattered y-direction displacement for a given PPW solution
        as a scalar heatmap/contourf plot.
        """
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered psi at each obstacle
        scattered_y_displacement, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.DISPLACEMENT_Y,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.SCATTERED
        )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Scattered y-Displacement $\mathbf{u}_y$ (Amplitude)'
            plot_filename = 'y_displacement_scattered_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Scattered y-Displacement $\mathbf{u}_y$ (Real Part)'
            plot_filename = 'y_displacement_scattered_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Scattered y-Displacement $\mathbf{u}_y$ (Imaginary Part)'
            plot_filename = 'y_displacement_scattered_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=scattered_y_displacement,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_total_stress_xx(
        self, 
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot total xx-stress \sigma_{xx} for a given PPW solution."""
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered psi at each obstacle
        sigma_xx_total, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.STRESS_XX,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.TOTAL
        )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Total xx-Stress $\sigma_{xx}$ (Amplitude)'
            plot_filename = 'stress_xx_total_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Total xx-Stress $\sigma_{xx}$ (Real Part)'
            plot_filename = 'stress_xx_total_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Total xx-Stress $\sigma_{xx}$ (Imaginary Part)'
            plot_filename = 'stress_xx_total_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=sigma_xx_total,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_stress_xx(
        self, 
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot scattered xx-stress \sigma_{xx} for a given PPW solution."""
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered psi at each obstacle
        sigma_xx_scattered, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.STRESS_XX,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.SCATTERED
        )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Scattered xx-Stress $\sigma_{xx}$ (Amplitude)'
            plot_filename = 'stress_xx_scattered_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Scattered xx-Stress $\sigma_{xx}$ (Real Part)'
            plot_filename = 'stress_xx_scattered_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Scattered xx-Stress $\sigma_{xx}$ (Imaginary Part)'
            plot_filename = 'stress_xx_scattered_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=sigma_xx_scattered,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_total_stress_xy(
        self, 
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot total xy-stress \sigma_{xy} for a given PPW solution."""
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered psi at each obstacle
        sigma_xy_total, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.STRESS_XY,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.TOTAL
        )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Total xy-Stress $\sigma_{xy}$ (Amplitude)'
            plot_filename = 'stress_xy_total_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Total xy-Stress $\sigma_{xy}$ (Real Part)'
            plot_filename = 'stress_xy_total_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Total xy-Stress $\sigma_{xy}$ (Imaginary Part)'
            plot_filename = 'stress_xy_total_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=sigma_xy_total,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_stress_xy(
        self, 
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot scattered xy-stress \sigma_{xy} for a given PPW solution."""
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered psi at each obstacle
        sigma_xy_scattered, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.STRESS_XY,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.SCATTERED
        )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Scattered xy-Stress $\sigma_{xy}$ (Amplitude)'
            plot_filename = 'stress_xy_scattered_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Scattered xy-Stress $\sigma_{xy}$ (Real Part)'
            plot_filename = 'stress_xy_scattered_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Scattered xy-Stress $\sigma_{xy}$ (Imaginary Part)'
            plot_filename = 'stress_xy_scattered_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=sigma_xy_scattered,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )
    
    def plot_total_stress_yy(
        self, 
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot total yy-stress \sigma_{yy} for a given PPW solution."""
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered psi at each obstacle
        sigma_yy_total, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.STRESS_YY,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.TOTAL
        )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Total yy-Stress $\sigma_{yy}$ (Amplitude)'
            plot_filename = 'stress_yy_total_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Total yy-Stress $\sigma_{yy}$ (Real Part)'
            plot_filename = 'stress_yy_total_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Total yy-Stress $\sigma_{yy}$ (Imaginary Part)'
            plot_filename = 'stress_yy_total_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=sigma_yy_total,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_stress_yy(
        self, 
        PPW: int,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot scattered yy-stress \sigma_{yy} for a given PPW solution."""
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        # Get scattered psi at each obstacle
        sigma_yy_scattered, vmin, vmax = self.get_scalars_for_plotting(
            PPW, 
            scalar_qoi=ScalarQOI.STRESS_YY,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.SCATTERED
        )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Scattered yy-Stress $\sigma_{yy}$ (Amplitude)'
            plot_filename = 'stress_yy_scattered_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Scattered yy-Stress $\sigma_{yy}$ (Real Part)'
            plot_filename = 'stress_yy_scattered_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Scattered yy-Stress $\sigma_{yy}$ (Imaginary Part)'
            plot_filename = 'stress_yy_scattered_imaginary_contour.png'
        
        self.plot_scalar_field_at_obstacles(
            PPW=PPW,
            field_vals=sigma_yy_scattered,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

        

    def plot_total_displacement_vector_field(self, PPW:int, step:int=1, plot_folder: Optional[str] = None):
        """Plot total displacement for a given PPW solution."""
        if PPW not in self.obstacles:
            raise ValueError(f"Error: No solution exists for PPW={PPW}")
        
        for obstacle in self.obstacles[PPW]:
            # Get other obstacle information 
            other_obstacles = []
            for other_obstacle in self.obstacles[PPW]:
                if other_obstacle.id != obstacle.id:
                    other_obstacles.append(other_obstacle)

            obstacle.plot_displacement_vector_field(
                other_obstacles=other_obstacles,
                incident_wave=self.incident_wave,
                step=step
            )
        
        # Display plot
        plt.title(f"Total Displacement Field (PPW = {PPW})")
        plt.colorbar()
        if plot_folder is None:
            plt.show()
        else:
            plot_img_path = os.path.join(plot_folder, "total_displacement_vector_field.png")
            plt.savefig(plot_img_path)
            plt.clf()


    def __getstate__(self):
        """Used when preparing to save a pickled version of this object."""
        # Prepare all class attributes 
        state = self.__dict__.copy()
        state.pop('obstacles')      # Don't save the full obstacles list here. Recreate them from serialization

        return state
    
    def __setstate__(self, state):
        """Restore the object from a serialized state"""
    
        # Recreate serializable attributes
        self.__dict__.update(state)
        self.obstacles = dict()
        missing_PPWs = []
        for PPW in self.PPWs:
            cached_PPW_solution = self._check_cached_PPW_solution(PPW)
            if cached_PPW_solution is None:
                missing_PPWs.append(PPW)
            else:
                self.obstacles[PPW] = cached_PPW_solution
        if len(missing_PPWs) > 0:
            raise RuntimeError(f"Error: Missing cache files for PPWs: {missing_PPWs}.")


                

        
    

    
    










