from typing import Optional, Self
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Colormap
from matplotlib.patches import Circle
import os

from ..base.waves import IncidentPlanePWave
from ..base.consts import ScalarQOI, ComplexArrayQuantity, PlotType, CoordinateSystem
from .grids import FDLocalPolarGrid, FDPolarGrid_ArtBndry, CartesianGrid
from .coefficients import ElasticFarfieldEvaluator
from .obstacles import MKFE_FDObstacle, Circular_MKFE_FDObstacle
from .algorithm import MKFE_FD_ScatteringProblem
from .waves import IncidentPlanePWaveEvaluator


class ScatteringProblemPlotter:
    """Plots the results of a scattering problem.
    
    Requires all obstacles to have each other obstacle and the 
    provided incident wave cached.
    """
    def __init__(
        self,
        obstacles: list[Circular_MKFE_FDObstacle],
        incident_wave: IncidentPlanePWave
    ) -> None:
        """Store the obstacles and incident wave to plot.
        
        Args:
            obstacles (list[MKFE_FDObstacle]): A list of 
                obstacles of interest in the scattering problem
            incident_wave (IncidentPlanePWave): The 
                incident wave of interest in the scattering problem
        """
        self.obstacles: list[Circular_MKFE_FDObstacle] = obstacles
        self.incident_wave = incident_wave



        self._setup_exterior_grid()

    def _setup_exterior_grid(self) -> None:
        """Sets up exterior grid for plotting outside artificial
        boundaries; see Section 4.1 of thesis.
        """
        # Get bounding box of all obstacles (for exterior problem plotting)
        x_min_global = np.inf 
        x_max_global = -np.inf 
        y_min_global = np.inf 
        y_max_global = -np.inf
        for obstacle in self.obstacles:
            X_obstacle, Y_obstacle = obstacle.grid.local_coords_to_global_XY()
            x_min_global = np.min([x_min_global, X_obstacle.min()])
            x_max_global = np.max([x_max_global, X_obstacle.max()])
            y_min_global = np.min([y_min_global, Y_obstacle.min()])
            y_max_global = np.max([y_max_global, Y_obstacle.max()])

        # Compute equal aspect lengths (so that we can fill the plot with exterior data
        # appropriately)
        x_length = x_max_global - x_min_global
        y_length = y_max_global - y_min_global
        max_length = max(x_length, y_length)

        x_center = (x_min_global + x_max_global) / 2
        y_center = (y_min_global + y_max_global) / 2
        self.plot_x_min = x_center - (max_length / 2)
        self.plot_x_max = x_center + (max_length / 2)
        self.plot_y_min = y_center - (max_length / 2)
        self.plot_y_max = y_center + (max_length / 2)
        
        # Make semi-refined cartesian grid; ignore any points that are
        # inside any computational domains 
        x_pts = np.linspace(self.plot_x_min, self.plot_x_max, 250)
        y_pts = np.linspace(self.plot_y_min, self.plot_y_max, 250)
        self.X_exterior_grid, self.Y_exterior_grid = np.meshgrid(x_pts, y_pts)

        # Find exterior points outside all donuts (in omega plus)
        self.omega_plus_grid_mask = np.ones_like(self.X_exterior_grid, dtype=bool)
        for obstacle in self.obstacles:
            R_obstacle, _ = obstacle.grid.global_XY_to_local_coords(self.X_exterior_grid, self.Y_exterior_grid)
            self.omega_plus_grid_mask &= (R_obstacle > obstacle.r_artificial_boundary)


    @classmethod 
    def from_scattering_problem_and_ppw(
        cls,
        scattering_problem: MKFE_FD_ScatteringProblem,
        PPW: int
    ) -> "ScatteringProblemPlotter":
        """Creates a plotter from the results of a scattering problem for 
        a specific PPW"""
        plotter = cls(
            obstacles=scattering_problem.obstacles[PPW],
            incident_wave=scattering_problem.incident_wave
        )
        return plotter


    def plot_scalar_field_at_obstacles(
        self,
        field_vals: list[np.ndarray],
        vmin: float,
        vmax:float,
        title: str,
        plot_folder:Optional[str] = None,
        plot_filename: Optional[str] = None
    ) -> None:
        for obstacle, vals in zip(self.obstacles, field_vals):
            # Plot field values
            quad_contour_set = obstacle.plot_contourf(vals, vmin=vmin, vmax=vmax)

        # Title and show the plot
        plt.tight_layout()
        plt.title(title)
        plt.colorbar(quad_contour_set)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(self.plot_x_min, self.plot_x_max)
        plt.ylim(self.plot_y_min, self.plot_y_max)
        if plot_folder is None:
            plt.show()
        else:
            plot_img_path = os.path.join(plot_folder, plot_filename)
            plt.savefig(plot_img_path, bbox_inches='tight', pad_inches=0.1)
            plt.clf()

    def plot_scalar_field_on_exterior_region(
        self,
        field_vals: np.ndarray,
        vmin: float,
        vmax:float,
    ) -> None:
        # Plot scalar field outside obstacles first 
        field_outside_boundaries = np.ma.masked_array(
            field_vals,
            ~self.omega_plus_grid_mask
        )
        color_grid_vals = np.arange(field_outside_boundaries.min(), field_outside_boundaries.max(), .001)

        # Plot using contourf
        levels = np.linspace(vmin, vmax, 1500)
        plt.contourf(self.X_exterior_grid, self.Y_exterior_grid, field_outside_boundaries, color_grid_vals, cmap=cm.coolwarm, vmin=vmin, vmax=vmax, levels=levels)

        # Draw circles around artificial boundaries 
        for obstacle in self.obstacles:
            x_center, y_center = obstacle.grid.center
            R_artificial_boundary = obstacle.r_artificial_boundary
            obstacle_art_boundary = Circle(
                (x_center, y_center),
                R_artificial_boundary,
                color='black',
                fill=False,
                linewidth=3  # Adjust thickness as needed
            )
            plt.gca().add_patch(obstacle_art_boundary)

    def get_scalars_for_plotting(
        self,
        scalar_qoi: ScalarQOI,
        complex_array_quantity: ComplexArrayQuantity,
        plot_type: PlotType,
        exterior: bool = False
    ) -> tuple[list[np.ndarray], float, float]:
        """Gets the desired scalar QOI (in absolute value, real,
        or imaginary form), either scattered or total, at each
        obstacle. Factors in contributions from all participating
        obstacles.
        
        Returns:
            list[np.ndarray] - The i'th entry is an array of quantities
                as the same shape of the obstacle local grid 
                at self.obstacles[i]
            float - The absolute minimum value encountered 
            float - The absolute maximum value encountered
        """
        vmin = np.inf 
        vmax = -np.inf 
        values_at_obstacles = []
        values_on_exterior_grid = np.zeros_like(self.X_exterior_grid, dtype=np.complex128)    # Grid for exterior problem
        
        ## Get QOI from incident wave on exterior grid (if desired)
        if exterior and plot_type is PlotType.TOTAL:
            uinc = self.obstacles[0].incident_evaluators.items
            uinc = list(self.obstacles[0].incident_evaluators.values())[0].incident_wave
            global_grid = CartesianGrid(self.X_exterior_grid, self.Y_exterior_grid)
            phi_inc = uinc(global_grid)
            psi_inc = np.zeros_like(phi_inc)
            
            potentials = np.stack((phi_inc, psi_inc), axis=-1)
            if scalar_qoi is ScalarQOI.PHI:
                values_on_exterior_grid += potentials[:,:,0]
            elif scalar_qoi is ScalarQOI.PSI:
                values_on_exterior_grid += potentials[:,:,1]
            else:
                raise NotImplementedError("Exterior plotting only implemented for potentials currently.")
            

        ## Get QOI from other obstacles at obstacle grids and on exterior grid
        for obstacle in self.obstacles:
            ## Get values to plot on obstacle m grid 
            # Get other obstacle information 
            other_obstacles = []
            for other_obstacle in self.obstacles:
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

            ## Get desired quantity from this obstacle on exterior grid
            ## (only potential implemented right now)
            if exterior:
                potentials = obstacle.get_scattered_potentials_on_exterior_grid(self.X_exterior_grid, self.Y_exterior_grid)
                if scalar_qoi is ScalarQOI.PHI:
                    values_on_exterior_grid += potentials[:,:,0]
                elif scalar_qoi is ScalarQOI.PSI:
                    values_on_exterior_grid += potentials[:,:,1]
                else:
                    raise NotImplementedError("Exterior plotting only implemented for potentials currently.")
        
        if exterior:
            # Finally, parse the exterior problem into real/abs/imag
            if complex_array_quantity is ComplexArrayQuantity.ABS:
                values_on_exterior_grid = np.abs(values_on_exterior_grid)
            elif complex_array_quantity is ComplexArrayQuantity.REAL:
                values_on_exterior_grid = np.real(values_on_exterior_grid)
            elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
                values_on_exterior_grid = np.imag(values_on_exterior_grid)

            # Update vmin/vmax accordingly
            vmax = np.max([vmax, np.max(values_on_exterior_grid[self.omega_plus_grid_mask])])
            vmin = np.min([vmin, np.min(values_on_exterior_grid[self.omega_plus_grid_mask])])

            # Return desired quantities
            return values_at_obstacles, values_on_exterior_grid, vmin, vmax 
        else:
            return values_at_obstacles, vmin, vmax
    
    def plot_total_phi(
        self,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None, 
        exterior: bool = False
    ):
        """Plot total phi for a given PPW solution."""
        # Get total phi at each obstacle
        if exterior:
            total_potentials, exterior_potentials, vmin, vmax = self.get_scalars_for_plotting(
                scalar_qoi=ScalarQOI.PHI,
                complex_array_quantity=complex_array_quantity,
                plot_type=PlotType.TOTAL, 
                exterior=True
            )
        else:
            total_potentials, vmin, vmax = self.get_scalars_for_plotting(
                scalar_qoi=ScalarQOI.PHI,
                complex_array_quantity=complex_array_quantity,
                plot_type=PlotType.TOTAL, 
                exterior=False
            )

        # Plot the contourf plot of the total phi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Total $\phi$ (Amplitude)'
            if exterior:
                plot_filename = 'phi_total_amplitude_exterior_contour.png'
            else:
                plot_filename = 'phi_total_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Total $\phi$ (Real Part)'
            if exterior:
                plot_filename = 'phi_total_real_exterior_contour.png'
            else:
                plot_filename = 'phi_total_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Total $\phi$ (Imaginary Part)'
            if exterior:
                plot_filename = 'phi_total_imaginary_exterior_contour.png'
            else:
                plot_filename = 'phi_total_imaginary_contour.png'
        
        # Plot on exterior region 
        if exterior:
            self.plot_scalar_field_on_exterior_region(
                exterior_potentials, vmin, vmax
            )
        # Plot on interior regions 
        self.plot_scalar_field_at_obstacles(
            field_vals=total_potentials,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )
        
        
        

    def plot_scattered_phi(
        self,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None,
        exterior: bool = False
    ):
        """Plot scattered phi for a given PPW solution."""
        # Get scattered phi at each obstacle
        if exterior:
            total_potentials, exterior_potentials, vmin, vmax = self.get_scalars_for_plotting(
                scalar_qoi=ScalarQOI.PHI,
                complex_array_quantity=complex_array_quantity,
                plot_type=PlotType.SCATTERED, 
                exterior=True
            )
        else:
            total_potentials, vmin, vmax = self.get_scalars_for_plotting(
                scalar_qoi=ScalarQOI.PHI,
                complex_array_quantity=complex_array_quantity,
                plot_type=PlotType.SCATTERED, 
                exterior=False
            )


        # Plot the contourf plot of the scattered phi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Scattered $\phi$ (Amplitude)'
            if exterior:
                plot_filename = 'phi_scattered_amplitude_exterior_contour.png'
            else:
                plot_filename = 'phi_scattered_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Scattered $\phi$ (Real Part)'
            if exterior:
                plot_filename = 'phi_scattered_real_exterior_contour.png'
            else:
                plot_filename = 'phi_scattered_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Scattered $\phi$ (Imaginary Part)'
            if exterior:
                plot_filename = 'phi_scattered_imaginary_exterior_contour.png'
            else:
                plot_filename = 'phi_scattered_imaginary_contour.png'
        
        # Plot on exterior region 
        if exterior:
            self.plot_scalar_field_on_exterior_region(
                exterior_potentials, vmin, vmax
            )
        
        # Plot on interior regions 
        self.plot_scalar_field_at_obstacles(
            field_vals=total_potentials,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_total_psi(
        self,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None, 
        exterior: bool = False
    ):
        """Plot total phi for a given PPW solution."""  
        # Get total phi at each obstacle
        if exterior:
            total_potentials, exterior_potentials, vmin, vmax = self.get_scalars_for_plotting(
                scalar_qoi=ScalarQOI.PSI,
                complex_array_quantity=complex_array_quantity,
                plot_type=PlotType.TOTAL, 
                exterior=True
            )
        else:
            total_potentials, vmin, vmax = self.get_scalars_for_plotting(
                scalar_qoi=ScalarQOI.PSI,
                complex_array_quantity=complex_array_quantity,
                plot_type=PlotType.TOTAL, 
                exterior=False
            )

        # Plot the contourf plot of the total psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Total $\psi$ (Amplitude)'
            if exterior:
                plot_filename = 'psi_total_amplitude_exterior_contour.png'
            else:
                plot_filename = 'psi_total_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Total $\psi$ (Real Part)'
            if exterior:
                plot_filename = 'psi_total_real_exterior_contour.png'
            else:
                plot_filename = 'psi_total_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Total $\psi$ (Imaginary Part)'
            if exterior:
                plot_filename = 'psi_total_imaginary_exterior_contour.png'
            else:
                plot_filename = 'psi_total_imaginary_contour.png'
        
        # Plot on exterior region 
        if exterior:
            self.plot_scalar_field_on_exterior_region(
                exterior_potentials, vmin, vmax
            )
        
        # Plot on interior regions 
        self.plot_scalar_field_at_obstacles(
            field_vals=total_potentials,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_psi(
        self,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None, 
        exterior: bool = False
    ):
        """Plot scattered psi for a given PPW solution."""
        # Get total psi at each obstacle
        if exterior:
            total_potentials, exterior_potentials, vmin, vmax = self.get_scalars_for_plotting(
                scalar_qoi=ScalarQOI.PSI,
                complex_array_quantity=complex_array_quantity,
                plot_type=PlotType.SCATTERED, 
                exterior=True
            )
        else:
            total_potentials, vmin, vmax = self.get_scalars_for_plotting(
                scalar_qoi=ScalarQOI.PSI,
                complex_array_quantity=complex_array_quantity,
                plot_type=PlotType.SCATTERED, 
                exterior=False
            )

        # Plot the contourf plot of the scattered psi at each obstacle
        # using uniform color scaling
        if complex_array_quantity is ComplexArrayQuantity.ABS:
            title = r'Scattered $\psi$ (Amplitude)'
            if exterior:
                plot_filename = 'psi_scattered_amplitude_exterior_contour.png'
            else:
                plot_filename = 'psi_scattered_amplitude_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.REAL:
            title = r'Scattered $\psi$ (Real Part)'
            if exterior:
                plot_filename = 'psi_scattered_real_exterior_contour.png'
            else:
                plot_filename = 'psi_scattered_real_contour.png'
        elif complex_array_quantity is ComplexArrayQuantity.IMAGINARY:
            title = r'Scattered $\psi$ (Imaginary Part)'
            if exterior:
                plot_filename = 'psi_scattered_imaginary_exterior_contour.png'
            else:
                plot_filename = 'psi_scattered_imaginary_contour.png'
        
        # Plot on exterior region 
        if exterior:
            self.plot_scalar_field_on_exterior_region(
                exterior_potentials, vmin, vmax
            )
        
        # Plot on interior regions 
        self.plot_scalar_field_at_obstacles(
            field_vals=total_potentials,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_total_x_displacement(
        self, 
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        """Plot total x-direction displacement for a given PPW solution
        as a scalar heatmap/contourf plot.
        """
        # Get scattered psi at each obstacle
        total_x_displacement, vmin, vmax = self.get_scalars_for_plotting(
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
            field_vals=total_x_displacement,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_x_displacement(
        self, 
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        """Plot scattered x-direction displacement for a given PPW solution
        as a scalar heatmap/contourf plot.
        """
        # Get scattered psi at each obstacle
        scattered_x_displacement, vmin, vmax = self.get_scalars_for_plotting(
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
            field_vals=scattered_x_displacement,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    
    def plot_total_y_displacement(
        self, 
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        """Plot total y-direction displacement for a given PPW solution
        as a scalar heatmap/contourf plot.
        """
        # Get scattered psi at each obstacle
        total_y_displacement, vmin, vmax = self.get_scalars_for_plotting(
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
            field_vals=total_y_displacement,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_y_displacement(
        self,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        """Plot scattered y-direction displacement for a given PPW solution
        as a scalar heatmap/contourf plot.
        """
        # Get scattered psi at each obstacle
        scattered_y_displacement, vmin, vmax = self.get_scalars_for_plotting(
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
            field_vals=scattered_y_displacement,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_total_stress_xx(
        self, 
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot total xx-stress \sigma_{xx} for a given PPW solution."""
        # Get scattered psi at each obstacle
        sigma_xx_total, vmin, vmax = self.get_scalars_for_plotting(
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
            field_vals=sigma_xx_total,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_stress_xx(
        self,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot scattered xx-stress \sigma_{xx} for a given PPW solution."""
        # Get scattered psi at each obstacle
        sigma_xx_scattered, vmin, vmax = self.get_scalars_for_plotting(
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
            field_vals=sigma_xx_scattered,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_total_stress_xy(
        self,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot total xy-stress \sigma_{xy} for a given PPW solution."""
        # Get scattered psi at each obstacle
        sigma_xy_total, vmin, vmax = self.get_scalars_for_plotting(
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
            field_vals=sigma_xy_total,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_stress_xy(
        self, 
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot scattered xy-stress \sigma_{xy} for a given PPW solution."""
        # Get scattered psi at each obstacle
        sigma_xy_scattered, vmin, vmax = self.get_scalars_for_plotting(
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
            field_vals=sigma_xy_scattered,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )
    
    def plot_total_stress_yy(
        self, 
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot total yy-stress \sigma_{yy} for a given PPW solution."""
        # Get scattered stress at each obstacle
        sigma_yy_total, vmin, vmax = self.get_scalars_for_plotting(
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
            field_vals=sigma_yy_total,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )

    def plot_scattered_stress_yy(
        self,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None 
    ) -> None:
        r"""Plot scattered yy-stress \sigma_{yy} for a given PPW solution."""
        # Get scattered stress at each obstacle
        sigma_yy_scattered, vmin, vmax = self.get_scalars_for_plotting(
            scalar_qoi=ScalarQOI.STRESS_YY,
            complex_array_quantity=complex_array_quantity,
            plot_type=PlotType.TOTAL
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
            field_vals=sigma_yy_scattered,
            vmin=vmin,
            vmax=vmax,
            title=title,
            plot_folder=plot_folder,
            plot_filename=plot_filename
        )











