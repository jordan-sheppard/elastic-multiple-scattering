from typing import Optional, Self
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Colormap
import os

from ..base.waves import IncidentPlanePWave
from ..base.consts import ScalarQOI, ComplexArrayQuantity, PlotType, CoordinateSystem
from .grids import FDLocalPolarGrid, FDPolarGrid_ArtBndry
from .obstacles import MKFE_FDObstacle, Circular_MKFE_FDObstacle
from .algorithm import MKFE_FD_ScatteringProblem


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
                at self.obstacles[i]
            float - The absolute minimum value encountered 
            float - The absolute maximum value encountered
        """
        vmin = np.inf 
        vmax = -np.inf 
        values_at_obstacles = []
        for obstacle in self.obstacles:
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
        return values_at_obstacles, vmin, vmax
    
    def plot_total_phi(
        self,
        complex_array_quantity:ComplexArrayQuantity,
        plot_folder: Optional[str] = None
    ):
        """Plot total phi for a given PPW solution."""
        # Get total phi at each obstacle
        total_potentials, vmin, vmax = self.get_scalars_for_plotting(
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
        plot_folder: Optional[str] = None
    ):
        """Plot scattered phi for a given PPW solution."""
        # Get scattered phi at each obstacle
        total_potentials, vmin, vmax = self.get_scalars_for_plotting(
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
        plot_folder: Optional[str] = None
    ):
        """Plot total phi for a given PPW solution."""  
        # Get total psi at each obstacle
        total_potentials, vmin, vmax = self.get_scalars_for_plotting(
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
        plot_folder: Optional[str] = None
    ):
        """Plot scattered psi for a given PPW solution."""
        # Get scattered psi at each obstacle
        total_potentials, vmin, vmax = self.get_scalars_for_plotting(
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
        """Plot total xx-stress \sigma_{xx} for a given PPW solution."""
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
        """Plot scattered xx-stress \sigma_{xx} for a given PPW solution."""
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
        """Plot total xy-stress \sigma_{xy} for a given PPW solution."""
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
        """Plot scattered xy-stress \sigma_{xy} for a given PPW solution."""
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
        """Plot total yy-stress \sigma_{yy} for a given PPW solution."""
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
        """Plot scattered yy-stress \sigma_{yy} for a given PPW solution."""
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











