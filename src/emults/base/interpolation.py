from typing import Optional
from scipy.interpolate import BarycentricInterpolator
import numpy as np


class PeriodicInterpolator1D:
    """A 5th-Order Lagrange Interpolation scheme for a periodic
    function.

    Uses 6 points on each potential subinterval for 5th-order
    polynomial interpolation for any point lying in that subinterval.
    
    Attributes:
        xL (float): The left endpoint of the periodic domain
        xR (float): The right endpoint of the periodic domain
        num_pts (int): The total number of interpolation x-values
        y_shape (Optional[int | tuple]): The shape of the output.
            If None, then no interpolation y-data has been provided.
            If -1, then output for each point is a scalar.
            If a tuple, then output for each point is an array.
        interpolation_schemes (list[BarycentericInterpolator]): A
            collection of Barycentric Lagrange interpolators, one
            correspnding to each of the potential subintervals
            on the periodic domain, which are configured to use 
            5th-order Barycentric Lagrange interpolation.
    """
    def __init__(
        self,
        periodic_domain: tuple[float, float],
        xi: np.ndarray
    ):
        """Set up interpolating points and 3rd order Barycentric
        Lagrange interpolation for each group of points.
        
        Parameters:
            periodic_domain (tuple[float, float]): The domain [xL, xR]
                on which the function is periodic.
            xi (np.ndarray): A 1D array of interpolation points
                (should be sorted)
        """
        # Parse periodic domain info
        self.xL, self.xR = periodic_domain

        # Check interpolation points are sorted
        if not np.all(xi[:-1] <= xi[1:]):
            raise ValueError("Error: xi must be a sorted list")
        
        # Parse interpolation point info
        self.xi = xi
        self.num_pts = len(self.xi)
        self.y_shape = None

        # Initialize interpolation schemes on each interval [xi[q], xi[q+1]].
        self._initialize_interpolation_schemes()

    def _initialize_interpolation_schemes(self):
        """Initializes Barycentric Lagrange interpolation schemes on each
        subinterval of the interpolaion points in self.xi, using 6 surrounding
        points for a 5rd-order scheme.
        
        Populates self.interpolation_schemes[q] with the interpolation scheme
        to use in the subinterval [xi[q], xi[q+1]].
        """
        self.interpolation_schemes: list[BarycentricInterpolator] = []
        for q in range(self.num_pts):
            # Handle endpoints carefully for interpolation purposes
            left_wraparound_pts = [
                self.xi[i] - self.xR for i in range(q-2, q+4) if i < 0
            ]
            normal_pts = [
                self.xi[i] for i in range(q-2, q+4) if i >= 0 and i < self.num_pts
            ]
            right_wraparound_pts = [
                self.xR + self.xi[i%self.num_pts] for i in range(q-2, q+4) if i >= self.num_pts
            ]
            interp_xpts = np.array(
                left_wraparound_pts + normal_pts + right_wraparound_pts
            )
            self.interpolation_schemes.append(
                BarycentricInterpolator(interp_xpts)
            )

    def update_func_vals(
        self,
        new_func_vals: np.ndarray,
    ) -> None:
        """Update function values for interpolation.
        
        Args:
            new_func_vals (np.ndarray): An array whose i'th entry
                is a point (or subarray of points) representing the
                output of the function at the i'th interpolation
                point in self.xi.
        """
        # Check and store shape of new, incoming data (scalar or array)
        if isinstance(new_func_vals[0], (float, np.floating, int, np.integer)):
            self.y_shape = -1
        else:
            self.y_shape = new_func_vals[0].shape

        # Update function values for all interpolation points.
        for q in range(self.num_pts):
            interp_ypts = np.array(
                [new_func_vals[i % self.num_pts] for i in range(q-2, q+4)]
            )
            self.interpolation_schemes[q].set_yi(interp_ypts, axis=0)

    def interpolate(
        self,
        x: np.ndarray,
        der: Optional[int] = None
    ) -> np.ndarray:
        """Interpolate the function values at the given x values
        using a degree-5 Barycentric Lagrange polynomial at the 
        nearest 6 interpolation points (taking into account
        periodicity).

        Args:
            x (np.ndarray): The inputs to the function which we'd like
                approximate/interpolated outputs for.
            der (int): The number of the derivative we would like to 
                interpolate. If no derivative provided, by default,
                the function values itself are interpolated

        Returns:
            np.ndarray: With shape (*x_shape, *y_shape), where x_shape
                is the shape of the input, and y_shape is the shape of
                the function output. 
        """
        # Make sure each entry of x is inside our periodic range.
        len_domain = (self.xR - self.xL)
        shifted_x = np.mod(x - self.xR, len_domain) + self.xL

        # Find which interpolation point in xi is CLOSEST to the left
        # of each point in x 
        diffs = self.xi - np.expand_dims(shifted_x, -1)  # Takes (signed) distance from point (negative if to left, positive if to right).
        idx_closest_left_pt = (
            np.where(diffs < 0, diffs, -np.inf)
            .argmax(axis=-1)
        )  # Finds xi pt with smallest negative distance (that is, the negative distance closest to 0)
        
        # Edge case: points to left of first interpolation point but to right of xL
        # Shift this point up by the length of the interval, and then assign it to 
        # the last point (as this is how we have implemented the wraparound scheme)
        idx_closest_left_pt[np.all(diffs > 0, axis=-1)] = self.num_pts - 1  
        shifted_x[np.all(diffs > 0, axis=-1)] += len_domain

        # Create place to store interpolated output
        # Indexing should be in order (x indexes, y_indexes)
        if self.y_shape == -1:
            output_shape = x.shape
        else:
            output_shape = (*x.shape, *self.y_shape)  # First indexes of x-pt, then each y-val.
        output_pts = np.zeros(output_shape, dtype='complex128')

        # Now, iterate over "closest" points and interpolate accordingly
        for q in range(self.num_pts):
            mask = (idx_closest_left_pt == q)
            if der is None:
                der = 0             # Interpolate the 0th derivative by default, AKA the original function
            
            output_pts[mask] = self.interpolation_schemes[q].derivative(shifted_x[mask], der)

        return output_pts 



if __name__ == "__main__":
    # Code for testing this
    xi = np.linspace(0.2, 2*np.pi, 8, endpoint=False)
    new_func_vals = np.array([[j + 0.5*i for i in range(7)] for j in range(len(xi))])
    # new_func_vals = np.array([1,1.5,2,2.5,2.5,2,1.5,1])
    periodic_domain = (0, 2*np.pi)
    interpolator = PeriodicInterpolator1D(periodic_domain, xi)
    interpolator.update_func_vals(new_func_vals)
    x = np.linspace(0, 2*np.pi, 50).reshape((25,2))
    print(x)
    # x = np.linspace(0, 2*np.pi, 50)
    res = interpolator.interpolate(x)
    print(res)





