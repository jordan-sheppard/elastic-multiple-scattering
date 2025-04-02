from typing import Self
import numpy as np
from scipy.special import hankel1

from ..base.interpolation import PeriodicInterpolator1D
from ..base.medium import LinearElasticMedium
from .grids import FDLocalPolarGrid
from ..base.consts import CoordinateSystem


class FarfieldAngularCoefficients:
    """A data storage class for storing truncated Karp expansion
    angular coefficients.
    
    Attributes:
        fp_coeffs (np.ndarray): A shape 
            (# angular gridpoints, # farfield terms) array whose
            [j,l] element is F_{m,l}^p ( (theta_m)_j )
        gp_coeffs (np.ndarray): A shape 
            (# angular gridpoints, # farfield terms) array whose
            [j,l] element is G_{m,l}^p ( (theta_m)_j )
        fs_coeffs (np.ndarray): A shape 
            (# angular gridpoints, # farfield terms) array whose
            [j,l] element is F_{m,l}^s ( (theta_m)_j )
        gs_coeffs (np.ndarray): A shape 
            (# angular gridpoints, # farfield terms) array whose
            [j,l] element is G_{m,l}^s ( (theta_m)_j )
    """
    def __init__(
        self,
        num_farfield_terms: int,
        num_angular_gridpoints: int
    ) -> None:
        """Initialize an empty container for farfield coefficients.
        
        Args:
            num_farfield_terms (int): The number of terms (L) to use
                in the truncated farfield expansion
            num_angular_gridpoints (int): The number of angular
                gridpoints (N_{theta}^m) to use to discretize the
                angular coefficients.
        """
        self.fp_coeffs = np.zeros(
            (num_angular_gridpoints, num_farfield_terms),
            dtype=np.complex128
        )
        self.fs_coeffs = np.zeros(
            (num_angular_gridpoints, num_farfield_terms),
            dtype=np.complex128
        )
        self.gp_coeffs = np.zeros(
            (num_angular_gridpoints, num_farfield_terms),
            dtype=np.complex128
        )
        self.gs_coeffs = np.zeros(
            (num_angular_gridpoints, num_farfield_terms),
            dtype=np.complex128
        )

    def update(
        self,
        updated: Self
    ) -> None:
        """Updates all 4 sets of angular coefficients from
        values stored in another instance of this class.
        
        Args:
            updated (FarfieldAngularCoefficients): A class
                containing updated information for all 
                4 sets of farfield coefficients.
        """
        self.update_fp_coeffs(updated.fp_coeffs)
        self.update_fs_coeffs(updated.fs_coeffs)
        self.update_gp_coeffs(updated.gp_coeffs)
        self.update_gs_coeffs(updated.gs_coeffs)

    def update_fp_coeffs(
        self,
        new_fp_coeffs: np.ndarray
    ) -> None:
        """Updates the stored F_{m,l}^p (theta_j) coefficients
        with new values.
        
        Args:
            new_fp_coeffs (np.ndarray): A shape 
                (# angular gridpoints, # farfield terms) array whose
                [j,l] element is the updated value of 
                F_{m,l}^p ( (theta_m)_j )
        """
        # Check for compatible shapes
        if not new_fp_coeffs.shape == self.fp_coeffs.shape:
            raise ValueError(f"Error: Updated Fp coefficient array should have shape (N_{{theta}}^m, L) = {self.fp_coeffs.shape}. Has shape {new_fp_coeffs.shape} instead.")

        # Update Fp coefficients
        self.fp_coeffs = new_fp_coeffs

    def update_gp_coeffs(
        self,
        new_gp_coeffs: np.ndarray
    ) -> None:
        """Updates the stored G_{m,l}^p (theta_j) coefficients
        with new values.
        
        Args:
            new_gp_coeffs (np.ndarray): A shape 
                (# angular gridpoints, # farfield terms) array whose
                [j,l] element is the updated value of 
                G_{m,l}^p ( (theta_m)_j )
        """
        # Check for compatible shapes
        if not new_gp_coeffs.shape == self.gp_coeffs.shape:
            raise ValueError(f"Error: Updated Gp coefficient array should have shape (N_{{theta}}^m, L) = {self.gp_coeffs.shape}. Has shape {new_gp_coeffs.shape} instead.")

        # Update Fp coefficients
        self.gp_coeffs = new_gp_coeffs

    def update_fs_coeffs(
        self,
        new_fs_coeffs: np.ndarray
    ) -> None:
        """Updates the stored F_{m,l}^s (theta_j) coefficients
        with new values.
        
        Args:
            new_fs_coeffs (np.ndarray): A shape 
                (# angular gridpoints, # farfield terms) array whose
                [j,l] element is the updated value of 
                F_{m,l}^s ( (theta_m)_j )
        """
        # Check for compatible shapes
        if not new_fs_coeffs.shape == self.fs_coeffs.shape:
            raise ValueError(f"Error: Updated Fs coefficient array should have shape (N_{{theta}}^m, L) = {self.fs_coeffs.shape}. Has shape {new_fs_coeffs.shape} instead.")

        # Update Fp coefficients
        self.fs_coeffs = new_fs_coeffs

    def update_gs_coeffs(
        self,
        new_gs_coeffs: np.ndarray
    ) -> None:
        """Updates the stored G_{m,l}^s (theta_j) coefficients
        with new values.
        
        Args:
            new_gs_coeffs (np.ndarray): A shape 
                (# angular gridpoints, # farfield terms) array whose
                [j,l] element is the updated value of 
                G_{m,l}^s ( (theta_m)_j )
        """
        # Check for compatible shapes
        if not new_gs_coeffs.shape == self.gs_coeffs.shape:
            raise ValueError(f"Error: Updated Gs coefficient array should have shape (N_{{theta}}^m, L) = {self.gs_coeffs.shape}. Has shape {new_gs_coeffs.shape} instead.")

        # Update Fp coefficients
        self.gp_coeffs = new_gs_coeffs


class InterpolatedGridFunction:
    """Stores scalar function and derivative values on a given grid.

    No real functionality; just used to make moving lots of data
    around a bit more organized.
    
    Attributes:
        f (np.ndarray): Discretized function values (f(x))
        df (np.ndarray): Discretized 1st derivative values (f'(x))
        d2f (np.ndarray): Discretized 2nd derivative values (f''(x))
    """
    def __init__(
        self,
        f: np.ndarray,
        df: np.ndarray,
        d2f: np.ndarray
    ):
        self.f = f 
        self.df = df 
        self.d2f = d2f


class ElasticPolarFarfieldEvaluator:
    """A class for evaluating the two truncated farfield
    expansions for the scattered wave outgoing from an obstacle (m)
    on an obstacle (mbar)'s local grid, in terms of (mbar)-local
    polar coordinates.
    
    Evaluates both the scattered p- and s-wave potentials
    outgoing from obstacle (m) using Karp's farfield expansion
    at the given (mbar) obstacle's grid. This grid should lie 
    entirely outside the artificial boundary of obstacle m.

    Assumes the obstacles in question are using local polar
    coordinates for their grids.

    Evaluations of any (m)-local angular farfield coefficients 
    in the expansion formulas are found using a 6-point 
    (5th order) barycentric Lagrange interpolation scheme on
    the points (theta_j, C(theta_j)), where C denotes the 
    angular farfield coefficient of interest.


    Attributes:
        R_mbar (np.ndarray): A shape (N_{theta}^{mbar}, N_r^{mbar})
            array containing mbar-local radii of all gridpoints on
            obstacle mbar's local grid
        Theta_mbar (np.ndarray): A shape (N_{theta}^{mbar}, N_r^{mbar})
            array containing mbar-local angles of all gridpoints on
            obstacle mbar's local grid
        R_m (np.ndarray): A shape (N_{theta}^{mbar}, N_r^{mbar})
            array containing m-local radii of all gridpoints on
            obstacle mbar's local grid
        Theta_m (np.ndarray): A shape (N_{theta}^{mbar}, N_r^{mbar})
            array containing m-local angles of all gridpoints on
            obstacle mbar's local grid
        X_global (np.ndarray): A shape (N_{theta}^{mbar}, N_r^{mbar})
            array containing global X-coordinates of all gridpoints on
            obstacle mbar's local grid
        Y_global (np.ndarray): A shape (N_{theta}^{mbar}, N_r^{mbar})
            array containing global Y-coordinates of all gridpoints on
            obstacle mbar's local grid
        R_artificial_boundary_m (float): The radius of the artificial
            boundary of obstacle m from its local center
        kp (float): The compressional wavenumber of the elastic medium
        ks (float): The shear wavenumber of the elastic medium
        num_farfield_terms (int): The number of terms (L) to use in the 
            truncated farfield expansion sums
        m_angle_interpolator (PeriodicInterpolator1D): An interpolation
            object for the m-local angular coefficients defined on
            the m-local angular gridpoints. Uses 5th-order (6-point)
            barycentric Lagrange interpolation scheme, and wraps
            around the periodic domain [0, 2*pi)
        V_m_p (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the compressional radial
            coefficients V_{m,l}^p ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the original compressional farfield expansion
            K_{m,l}^p
        V_m_s (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the shear radial
            coefficients V_{m,l}^s ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the original shear farfield expansion
            K_{m,l}^s
        W_m_p (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the compressional radial
            coefficients W_{m,l}^p ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the original compressional farfield expansion
            K_{m,l}^p
        W_m_s (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the shear radial coefficients 
            W_{m,l}^s ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the original shear farfield expansion
            K_{m,l}^s
        A_m_p (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the compressional radial
            coefficients A_{m,l}^p ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the 1st radial derivative of the compressional
            farfield expansion K_{m,l}^p
        A_m_s (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the shear radial
            coefficients A_{m,l}^s ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the 1st radial derivative of the shear
            farfield expansion K_{m,l}^s
        B_m_p (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the compressional radial
            coefficients B_{m,l}^p ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the 1st radial derivative of the compressional
            farfield expansion K_{m,l}^p
        B_m_s (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the shear radial
            coefficients B_{m,l}^s ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the 1st radial derivative of the shear
            farfield expansion K_{m,l}^s
        C_m_p (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the compressional radial
            coefficients C_{m,l}^p ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the 2nd radial derivative of the compressional
            farfield expansion K_{m,l}^p
        C_m_s (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the shear radial
            coefficients C_{m,l}^s ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the 2nd radial derivative of the shear
            farfield expansion K_{m,l}^s
        D_m_p (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the compressional radial
            coefficients D_{m,l}^p ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the 2nd radial derivative of the compressional
            farfield expansion K_{m,l}^p
        D_m_s (np.ndarray): A shape (L, N_{theta}^{mbar}, N_r^{mbar})
            array whose [l,j,i] contains the shear radial
            coefficients D_{m,l}^s ( (r_{mbar})_i, (theta_{mbar})_j )
            used in the 2nd radial derivative of the shear
            farfield expansion K_{m,l}^s
        cosine_rotation (np.ndarray): A shape 
            (N_{theta}^{mbar}, N_r^{mbar})) array whose [j,i]
            entry is the value
            cos( (theta_{mbar})_{j} - (theta_{m})_{i,j} )
            used in the rotation matrix from m-local coordinates
            to mbar-local coordinates
        sine_rotation (np.ndarray): A shape 
            (N_{theta}^{mbar}, N_r^{mbar})) array whose [j,i]
            entry is the value
            sin( (theta_{mbar})_{j} - (theta_{m})_{i,j} )
            used in the rotation matrix from m-local coordinates
            to mbar-local coordinates
        fp_coeffs (InterpolatedGridFunction): A container 
            containing 3 arrays of shape
            (N_{theta}^{mbar}, N_r^{mbar}, L)
            representing the interpolated angular coefficients
            F_{m,l}^p( (r_{mbar})_i, (theta_{mbar})_j )
            and their angular derivatives at each gridpoint in 
            the stored mbar-local grid of interest
        gp_coeffs (InterpolatedGridFunction): A container 
            containing 3 arrays of shape
            (N_{theta}^{mbar}, N_r^{mbar}, L)
            representing the interpolated angular coefficients
            G_{m,l}^p( (r_{mbar})_i, (theta_{mbar})_j )
            and their angular derivatives at each gridpoint in 
            the stored mbar-local grid of interest
        fs_coeffs (InterpolatedGridFunction): A container 
            containing 3 arrays of shape
            (N_{theta}^{mbar}, N_r^{mbar}, L)
            representing the interpolated angular coefficients
            F_{m,l}^s( (r_{mbar})_i, (theta_{mbar})_j )
            and their angular derivatives at each gridpoint in 
            the stored mbar-local grid of interest
        gs_coeffs (InterpolatedGridFunction): A container 
            containing 3 arrays of shape
            (N_{theta}^{mbar}, N_r^{mbar}, L)
            representing the interpolated angular coefficients
            G_{m,l}^s( (r_{mbar})_i, (theta_{mbar})_j )
            and their angular derivatives at each gridpoint in 
            the stored mbar-local grid of interest
    """
    def __init__(
        self,
        source_local_grid: FDLocalPolarGrid,
        dest_local_grid: FDLocalPolarGrid,
        medium: LinearElasticMedium,
        num_farfield_terms: int
    ) -> None:
        """Initializes the grid of interest to evaluate
        coefficients on, as well as other needed constants.
        
        Args:
            source_local_grid (FDLocalPolarGrid): The local polar
                grid at obstacle (m) (where the scattered wave
                is eminating from)
            dest_local_grid (FDLocalPolarGrid): The local polar
                grid at obstacle (mbar) (where we would like to 
                evaluate the scatted wave)
            medium (LinearElasticMedium): The linear elastic medium
                where this wave is propagating
            num_farfield_terms (int): The number of terms to use
                in the truncated farfield expansion eminating 
                from this obstacle
        """
        # Store mbar-local grid for wave evaluation in various coordinate systems:
        self.R_mbar, self.Theta_mbar = dest_local_grid.r_local, dest_local_grid.theta_local     # mbar-local polar coordinates
        self.R_global, self.Theta_global = dest_local_grid.local_coords_to_global_polar()       # Global polar coordinates  
        self.X_global, self.Y_global = dest_local_grid.local_coords_to_global_XY()              # Global cartesian coordinates
        self.R_m, self.Theta_m = source_local_grid.global_XY_to_local_coords(self.X_global, self.Y_global)   # m-local polar coordinates

        # Parse m-local radius of obstale m artificial boundary C^m
        self.R_artificial_boundary_m = source_local_grid.r_vals[-1]      # Assumes artificial boundary is the final physical gridpoint

        # Validate all gridpoints at destination (mbar) grid 
        # are outside of computational domain for source (m) obstacle 
        if np.any(self.R_m < self.R_artificial_boundary_m):
            raise ValueError("Destination grid must lie entirely outside of source grid's artificial boundary") 

        # Store medium attributes 
        self.kp = medium.kp
        self.ks = medium.ks

        # Store truncation attributes 
        self.num_farfield_terms = num_farfield_terms

        # Create interpolator for all m-local farfield coefficients
        # defined at the provided m-local angular gridpoint values
        self.m_angle_interpolator = PeriodicInterpolator1D(
            periodic_domain=(0., 2*np.pi),
            xi=source_local_grid.theta_vals
        )

        # Initialize all radial coefficients 
        self._initialize_radial_coeffs()

        # Initialize all rotation matrix entries 
        self._initialize_rotation_matrices()


    def _initialize_radial_coeffs(self) -> None:
        """Initialize all farfield radial coefficients from obstacle
        (m) onto the (mbar)-local grid of interest.
        """
        ## Get repeated constants (use array broadcasting)
        # I) Gridpoint-based arrays (that is,evaluated at each [j,i]
        #    gridpoint of mbar-local grid. Should each have shape
        #    (N_{theta}^{mbar}, N_r^{mbar})
        radius_ratios = self.R_artificial_boundary_m / self.R_m      
        kpR = self.kp * self.R_m
        ksR = self.ks * self.R_m
        Hp0 = hankel1(0, kpR)
        Hp1 = hankel1(1, kpR)
        Hs0 = hankel1(0, ksR) 
        Hs1 = hankel1(1, ksR)

        # II) Sum index (l)-based arrays (that is, evaluated at each [l]
        #     index value of l=0,1,...,L-1)
        l = np.arange(self.num_farfield_terms)[:,np.newaxis,np.newaxis]      # l = 0, 1, ..., L-1; shape (L,1,1)
        ratio_pow = (radius_ratios)**l

        ## Get radial farfield coefficients
        # I) Radial cefficients at other obstacle's gridpoints
        # NOTE: All of these will have shape (L, N_{theta}^{mbar}, N_r^{mbar})
        self.V_m_p = ratio_pow * Hp0
        self.V_m_s = ratio_pow * Hs0
        self.W_m_p = ratio_pow * Hp1 
        self.W_m_s = ratio_pow * Hs1
        self.A_m_p = -self.kp * ratio_pow * (Hp1 + (l * Hp0)/kpR)
        self.A_m_s = -self.ks * ratio_pow * (Hs1 + (l * Hs0)/ksR)
        self.B_m_p = -self.kp * ratio_pow * (-Hp0 + ((l+1) * Hp1)/kpR)
        self.B_m_s = -self.ks * ratio_pow * (-Hs0 + ((l+1) * Hs1)/ksR)
        self.C_m_p = self.kp**2 * ratio_pow * (
                -Hp0 + ((2*l + 1) * Hp1)/kpR + (l*(l+1)*Hp0)/(kpR**2)
        )
        self.C_m_s = self.ks**2 * ratio_pow * (
            -Hs0 + ((2*l + 1) * Hs1)/ksR + (l*(l+1)*Hs0)/(ksR**2)
        )
        self.D_m_p =  self.kp**2 * ratio_pow * (
            -Hp1 - ((2*l + 1) * Hp0)/kpR + ((l+2)*(l+1)*Hp1)/(kpR**2)
        )
        self.D_m_s = self.ks**2 * ratio_pow * (
            -Hs1 - ((2*l + 1) * Hs0)/ksR + ((l+2)*(l+1)*Hs1)/(ksR**2)
        )

        

    def _initialize_rotation_matrices(self) -> None:
        """Initialize all entries of the rotation matrices for 
        necessary changes of coordinates"""
        # NOTE: All of these following arrays will have shape (N_{theta}^{mbar}, N_r^{mbar})
        # Rotation matrices from m-local to mbar-local
        angles_m_to_mbar = (self.Theta_m - self.Theta_mbar)
        self.cosine_rotation_m_to_mbar = np.cos(angles_m_to_mbar)
        self.sine_rotation_m_to_mbar = np.sin(angles_m_to_mbar)

        # Rotation matrices from m-local to global
        angles_m_to_global = (self.Theta_m - self.Theta_global)
        self.cosine_rotation_m_to_global = np.cos(angles_m_to_global)
        self.sine_rotation_m_to_global = np.sin(angles_m_to_global)

        # Polar-to-cartesian rotation matrices from mbar-local polar to mbar-local cartesian
        self.cos_mbar = np.cos(self.Theta_mbar)
        self.sin_mbar = np.sin(self.Theta_mbar)

        # Polar-to-cartesian rotation matrices from global polar to global cartesian
        self.cos_global = np.cos(self.Theta_global)
        self.sin_global = np.sin(self.Theta_global)


    def _interpolate_angular_coeffs(
        self,
        coeffs: np.ndarray
    ) -> InterpolatedGridFunction:
        """Interpolates a set of discretized angular coefficients 
        defined at each of the m-local angular gripoints with a 
        5th-degree polynomial, and evaluates that polynomial at 
        each of the m-bar gridpoints on this object's mbar-local
        grid of interest.
        
        Args:
            coeffs (np.ndarray) - A shape 
                (N_{theta}^{mbar}, L) array whose
                [j,l] entry is the value of the l'th Farfield coefficient
                of interest at the m-local angular gridpoint (theta_m)_j. 

        Returns:
            InterpolatedGridFunction - The interpolated coefficient (and 
                angular derivative of coefficient) values on each 
                of the gridpoints in this object's stored grid.
        """
        self.m_angle_interpolator.update_func_vals(coeffs)
        coeff_vals = self.m_angle_interpolator.interpolate(self.Theta_m)            # Shape (N_{theta}^{mbar}, N_r^{mbar}, L)
        d_coeff_vals = self.m_angle_interpolator.interpolate(self.Theta_m, der=1)   # Shape (N_{theta}^{mbar}, N_r^{mbar}, L)
        d2_coeff_vals = self.m_angle_interpolator.interpolate(self.Theta_m, der=2)  # Shape (N_{theta}^{mbar}, N_r^{mbar}, L)

        return InterpolatedGridFunction(
            f=coeff_vals,
            df=d_coeff_vals,
            d2f=d2_coeff_vals
        )


    def update_angular_coeffs(
        self,
        angular_coeffs: FarfieldAngularCoefficients
    ) -> None:
        """Get updated m-local angular coefficients at each of the
        gridpoints by 5th-degree Barycentric Lagrange interpolation.

        All InterpolatedGridFunction objects for self.fp_coeffs,
        self.gs_coeffs, etc. will have attributes, f, df, and d2f
        populated with shape (N_{theta}^{mbar}, N_{r}^{mbar}, L) arrays 
        of angular coefficients, where the [j,i,l] index is 
        the angular coefficient evaluated at the (i,j) mbar-local
        gridpoint, at farfield sum index l:
        C_{m,l} ( (r_{mbar})_i , (theta_{mbar})_j )
        
        Args:
            angular_coeffs (FarfieldAngularCoefficients): An object
                containing obstacle m's angular farfield
                coefficient values at each of obstacle m's local
                angular gridpoint values. Each of the four
                coefficient arrays should be have shape
                (N_{theta}^{mbar}, L).
        """
        # Get all 4 sets of farfield coefficients on the mbar grid of interest
        self.fp_coeffs = self._interpolate_angular_coeffs(angular_coeffs.fp_coeffs)
        self.gp_coeffs = self._interpolate_angular_coeffs(angular_coeffs.gp_coeffs)
        self.fs_coeffs = self._interpolate_angular_coeffs(angular_coeffs.fs_coeffs)
        self.gs_coeffs = self._interpolate_angular_coeffs(angular_coeffs.gs_coeffs)


    def _product_sum_coeffs(
        self,
        f_radial_coeffs: np.ndarray,
        g_radial_coeffs: np.ndarray,
        f_angular_coeffs: np.ndarray,
        g_angular_coeffs: np.ndarray,
        boundary_only: bool
    ) -> np.ndarray:
        """Evaluates the sum 
        \sum_{l=0}^{L-1} (Rf[l,i,j] Af[i,j,l] + Rg[l,i,j] Ag[i,j,l])
        at every gridpoint of the stored (mbar)-local grid, where:
        
        * (i,j) is the index of the (mbar)-local gridpoint of interest
        * (l) is the index of the farfield coefficient in the above sum
        * Rf is an array of radial coefficients multiplying the F_{m,l} 
            angular coefficients and their derivatives 
        * Af is an array of the F_{m,l} angular coefficients (or their 
            derivatives)
        * Rg is an array of radial coefficients multiplying  the G_{m,l} 
            angular coefficients and their derivatives 
        * Ag is an array of the G_{m,l} angular coefficients (or their 
            derivatives)

        (If only evaluating on the boundary, the index i will always
        be 0)

        Args:
            f_radial_coeffs (np.ndarray): A shape (l,i,j) array
                denoting the radial coefficients Rf in the above formula
            g_radial_coeffs (np.ndarray): A shape (l,i,j) array
                denoting the angular coefficients Rf in the above formula
            f_angular_coeffs (np.ndarray): A shape (i,j,l) array
                denoting the radial coefficients Rf in the above formula
            g_angular_coeffs (np.ndarray): A shape (i,j,l) array
                denoting the angular coefficients Rf in the above formula
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (i,j) (or (j,) if boundary_only=True)
                array of the above sum of products evaluated at each
                [i,j] (or [j], respectively) (mbar)-local gridpoint.
        """
        if boundary_only:
            f_terms = np.einsum('lj, jl -> j', f_radial_coeffs[:,:,0], f_angular_coeffs[:,0,:])
            g_terms = np.einsum('lj, jl -> j', g_radial_coeffs[:,:,0], g_angular_coeffs[:,0,:])
        else:
            f_terms = np.einsum('lij, ijl -> ij', f_radial_coeffs, f_angular_coeffs)
            g_terms = np.einsum('lij, ijl -> ij', g_radial_coeffs, g_angular_coeffs)
        return f_terms + g_terms


    def Kp(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the compressional K_{m,L}^p farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the compressional farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                K_{m,L}^p ( r_{mbar}[i], theta_{mbar}[j] )
                (or K_{m,L}^p ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.V_m_p,
            g_radial_coeffs=self.W_m_p,
            f_angular_coeffs=self.fp_coeffs.f,
            g_angular_coeffs=self.gp_coeffs.f,
            boundary_only=boundary_only
        )
    

    def Ks(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the shear K_{m,L}^s farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
           np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the shear farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                K_{m,L}^s ( r_{mbar}[i], theta_{mbar}[j] )
                (or K_{m,L}^s ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.V_m_s,
            g_radial_coeffs=self.W_m_s,
            f_angular_coeffs=self.fs_coeffs.f,
            g_angular_coeffs=self.gs_coeffs.f,
            boundary_only=boundary_only
        )
    

    def dr_Kp(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the 1st (m)-local radial derivative 
        of the compressional K_{m,L}^p farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the 1st (m)-local radial derivative of 
                the compressional farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                dr K_{m,L}^p ( r_{mbar}[i], theta_{mbar}[j] )
                (or dr K_{m,L}^p ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.A_m_p,
            g_radial_coeffs=self.B_m_p,
            f_angular_coeffs=self.fp_coeffs.f,
            g_angular_coeffs=self.gp_coeffs.f,
            boundary_only=boundary_only
        )


    def dr_Ks(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the 1st (m)-local radial derivative 
        of the shear K_{m,L}^s farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the 1st (m)-local radial derivative of 
                the shear farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                dr K_{m,L}^s ( r_{mbar}[i], theta_{mbar}[j] )
                (or dr K_{m,L}^s ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.A_m_s,
            g_radial_coeffs=self.B_m_s,
            f_angular_coeffs=self.fs_coeffs.f,
            g_angular_coeffs=self.gs_coeffs.f,
            boundary_only=boundary_only
        )
    

    def dtheta_Kp(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the 1st (m)-local angular derivative 
        of the compressional K_{m,L}^p farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the 1st (m)-local angular derivative of 
                the shear farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                dr K_{m,L}^s ( r_{mbar}[i], theta_{mbar}[j] )
                (or dr K_{m,L}^s ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.V_m_p,
            g_radial_coeffs=self.W_m_p,
            f_angular_coeffs=self.fp_coeffs.df,
            g_angular_coeffs=self.gp_coeffs.df,
            boundary_only=boundary_only
        )
    

    def dtheta_Ks(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the 1st (m)-local angular derivative 
        of the shear K_{m,L}^p farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the 1st (m)-local angular derivative of 
                the shear farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                dr K_{m,L}^s ( r_{mbar}[i], theta_{mbar}[j] )
                (or dr K_{m,L}^s ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.V_m_s,
            g_radial_coeffs=self.W_m_s,
            f_angular_coeffs=self.fs_coeffs.df,
            g_angular_coeffs=self.gs_coeffs.df,
            boundary_only=boundary_only
        )
    

    def d2r_Kp(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the 2nd (m)-local radial derivative 
        of the compressional K_{m,L}^p farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the 2nd (m)-local radial derivative of 
                the compressional farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                d_{rr}^2 K_{m,L}^p ( r_{mbar}[i], theta_{mbar}[j] )
                (or d_{rr}^2 K_{m,L}^p ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.C_m_p,
            g_radial_coeffs=self.D_m_p,
            f_angular_coeffs=self.fp_coeffs.f,
            g_angular_coeffs=self.gp_coeffs.f,
            boundary_only=boundary_only
        )


    def d2r_Ks(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the 2nd (m)-local radial derivative 
        of the shear K_{m,L}^s farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the 2nd (m)-local radial derivative of 
                the shear farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                d_{rr}^2 K_{m,L}^s ( r_{mbar}[i], theta_{mbar}[j] )
                (or d_{rr}^2 K_{m,L}^s ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.C_m_s,
            g_radial_coeffs=self.D_m_s,
            f_angular_coeffs=self.fs_coeffs.f,
            g_angular_coeffs=self.gs_coeffs.f,
            boundary_only=boundary_only
        )
    

    def d2theta_Kp(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the 2nd (m)-local angular derivative 
        of the compressional K_{m,L}^p farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the 2nd (m)-local angular derivative of 
                the compressional farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                d_{theta theta}^2 K_{m,L}^p ( r_{mbar}[i], theta_{mbar}[j] )
                (or d_{theta theta}^2 K_{m,L}^p ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.V_m_p,
            g_radial_coeffs=self.W_m_p,
            f_angular_coeffs=self.fp_coeffs.d2f,
            g_angular_coeffs=self.gp_coeffs.d2f,
            boundary_only=boundary_only
        )
    

    def d2theta_Ks(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the 2nd (m)-local angular derivative 
        of the shear K_{m,L}^s farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the 2nd (m)-local angular derivative of 
                the shear farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                d_{theta theta}^2 K_{m,L}^s ( r_{mbar}[i], theta_{mbar}[j] )
                (or d_{theta theta}^2 K_{m,L}^s ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.V_m_s,
            g_radial_coeffs=self.W_m_s,
            f_angular_coeffs=self.fs_coeffs.d2f,
            g_angular_coeffs=self.gs_coeffs.d2f,
            boundary_only=boundary_only
        )
    

    def dr_dtheta_Kp(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the 2nd mixed (m)-local radial/angular derivative 
        of the compressional K_{m,L}^p farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the 2nd mixed (m)-local radial/angular derivative of 
                the compressional farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                d_{r theta}^2 K_{m,L}^p ( r_{mbar}[i], theta_{mbar}[j] )
                (or d_{r theta}^2 K_{m,L}^p ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.A_m_p,
            g_radial_coeffs=self.B_m_p,
            f_angular_coeffs=self.fp_coeffs.df,
            g_angular_coeffs=self.gp_coeffs.df,
            boundary_only=boundary_only
        )
    

    def dr_dtheta_Ks(self, boundary_only: bool = False) -> np.ndarray:
        """Evaluates the 2nd mixed (m)-local radial/angular derivative 
        of the shear K_{m,L}^s farfield expansion
        at each gridpoint of the stored mbar-local grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.
        
        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_{r}^{mbar}) array (or
                shape (N_{theta}^{mbar},) array if on boundary) whose
                [j,i] (or [j], respectively) entry contains the value
                of the 2nd mixed (m)-local radial/angular derivative of 
                the shear farfield expansion at the (i,j)
                mbar-local gridpoint (or (j) mbar-local angular
                gridpoint):
                d_{r theta}^2 K_{m,L}^s ( r_{mbar}[i], theta_{mbar}[j] )
                (or d_{r theta}^2 K_{m,L}^s ( r_0^{mbar}, theta_{mbar}[j] ) )
        """
        return self._product_sum_coeffs(
            f_radial_coeffs=self.A_m_s,
            g_radial_coeffs=self.B_m_s,
            f_angular_coeffs=self.fs_coeffs.df,
            g_angular_coeffs=self.gs_coeffs.df,
            boundary_only=boundary_only
        )
    

    def potentials(self, boundary_only: bool = True) -> np.ndarray:
        """Return the interpolated K_{m,L}^p and K_{m,L}^s potentials
        at each gridpoint of the (mbar)-local grid of interest. 

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.

        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_r^{mbar},  2) (or
                (N_{theta}^{mbar}, 2) array if boundary_only=True), where
                the [:,:,0] (or [:,0], respectively) slice carries
                the K_{m,L}^p potential values, and the 
                [:,:,1] ([:,1], respectively) slice carries the
                K_{m,L}^s potential values.
        """
        Kp_vals = self.Kp(boundary_only)
        Ks_vals = self.Ks(boundary_only)
        return np.stack((Kp_vals, Ks_vals), axis=-1) 

    def displacement(
        self,
        boundary_only: bool = True,
        coordinate_system: CoordinateSystem = CoordinateSystem.LOCAL_POLAR
        ) -> np.ndarray:
        """Return the (mbar)-local polar displacement caused by 
        these outgoing potentials/farfield expansions from obstacle m,
        on the provided (mbar)-local polar grid.

        Args:
            boundary_only (bool): Whether or not to only return
                the gridpoints on the physical (i=0) boundary
                of obstacle mbar's grid. Defaults to False.

        Returns:
            np.ndarray: A shape (N_{theta}^{mbar}, N_r^{mbar},  2) (or
                (N_{theta}^{mbar}, 2) array if boundary_only=True), where
                the [:,:,0] (or [:,0], respectively) slice carries
                the (mbar)-local radial displacement, and the 
                [:,:,1] ([:,1], respectively) slice carries the
                (mbar)-local angular displacement
        """
        # Get needed quantities/arrays with correct shapes for grid 
        # (e.g., only at boundary vs. on entire grid of interest)
        if boundary_only:
            R_m = self.R_m[:,0]                                                 # Shape (N_{theta}^{mbar},)
            cosine_rotation_m_to_mbar = self.cosine_rotation_m_to_mbar[:,0]     # Shape (N_{theta}^{mbar},)
            sine_rotation_m_to_mbar = self.sine_rotation_m_to_mbar[:,0]         # Shape (N_{theta}^{mbar},)
            cosine_rotation_m_to_global = self.cosine_rotation_m_to_global[:,0] # Shape (N_{theta}^{mbar},)
            sine_rotation_m_to_global = self.sine_rotation_m_to_global[:,0]     # Shape (N_{theta}^{mbar},)
            cos_mbar = self.cos_mbar[:,0]                                       # "" "" ""
            sin_mbar = self.sin_mbar[:,0]                                       # "" "" ""
            cos_global = self.cos_global[:,0]                                   # "" "" ""
            sin_global = self.sin_global[:,0]                                   # "" "" ""
        else:
            R_m = self.R_m                                                      # Shape (N_{theta}^{mbar}, N_r^{mbar})
            cosine_rotation_m_to_mbar = self.cosine_rotation_m_to_mbar          # Shape (N_{theta}^{mbar}, N_r^{mbar})
            sine_rotation_m_to_mbar = self.sine_rotation_m_to_mbar              # Shape (N_{theta}^{mbar}, N_r^{mbar})
            cosine_rotation_m_to_global = self.cosine_rotation_m_to_global      # Shape (N_{theta}^{mbar}, N_r^{mbar})
            sine_rotation_m_to_global = self.sine_rotation_m_to_global          # Shape (N_{theta}^{mbar}, N_r^{mbar})
            cos_mbar = self.cos_mbar                                            # "" "" ""
            sin_mbar = self.sin_mbar                                            # "" "" ""
            cos_global = self.cos_global                                        # "" "" ""
            sin_global = self.sin_global                                        # "" "" ""
        
        # Evaluate farfield derivatives needed for m-local displacement
        # before rotation 
        dr_m_Kp = self.dr_Kp(boundary_only)
        dr_m_Ks = self.dr_Ks(boundary_only)
        dtheta_m_Kp = self.dtheta_Kp(boundary_only)
        dtheta_m_Ks = self.dtheta_Ks(boundary_only)

        # Evaluate m-local displacement before rotation
        u_r_m = dr_m_Kp + dtheta_m_Ks/R_m
        u_theta_m = dtheta_m_Kp/R_m - dr_m_Ks
        
        # Get (mbar)-local polar displacement from (m)-local polar displacement
        # using rotation matrix entries for m -> mbar
        u_r_mbar = cosine_rotation_m_to_mbar * u_r_m - sine_rotation_m_to_mbar * u_theta_m
        u_theta_mbar = sine_rotation_m_to_mbar * u_r_m + cosine_rotation_m_to_mbar * u_theta_m 
        if coordinate_system is CoordinateSystem.LOCAL_POLAR:
            return np.stack((u_r_mbar, u_theta_mbar), axis=-1)      # Shape (N_{theta}^{mbar}, N_r^{mbar}, 2)
        
        # Get global polar displacement from (m)-local polar displacement
        # using rotation matrix entries from m -> global
        u_r_global = cosine_rotation_m_to_global * u_r_m - sine_rotation_m_to_global * u_theta_m 
        u_theta_global = sine_rotation_m_to_global * u_r_m + cosine_rotation_m_to_global * u_theta_m
        if coordinate_system is CoordinateSystem.GLOBAL_POLAR:
            return np.stack((u_r_global, u_theta_global), axis=-1)  # Shape (N_{theta}^{mbar}, N_r^{mbar}, 2)
        
        # Get cartesian displacement from
        # (mbar)-local polar displacement using definition 
        # of mbar-local polar unit vectors (r_m)_hat and (theta_m)_hat
        u_x_mbar = u_r_mbar * cos_mbar - u_theta_mbar * sin_mbar
        u_y_mbar = u_r_mbar * sin_mbar + u_theta_mbar * cos_mbar
        if coordinate_system is CoordinateSystem.LOCAL_CARTESIAN or coordinate_system is CoordinateSystem.GLOBAL_CARTESIAN:       # Cartesian coordinate displacmeents are translation invariant
            return np.stack((u_x_mbar, u_y_mbar), axis=-1)          # Shape (N_{theta}^{mbar}, N_r^{mbar}, 2)


        
        

