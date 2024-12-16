from typing import Optional
import numpy as np
from scipy.interpolate import BarycentricInterpolator

from ..base.consts import Grid



def Lagrange1D_3rdOrder_Interpolate_Periodic(
    xp: np.ndarray,
    f_xp: np.ndarray,
    new_x: np.ndarray,
    domain: tuple[float, float],
    interpolators: Optional[list[BarycentricInterpolator]] = None
) -> np.ndarray:
    """Approximates the value of a given 1D source function f(x)
    at the given points new_x, given only the gridpoints (xp, f(xp)).

    f should be defined on some closed interval [xL, xR]. f is
    assumed to be periodic on this domain; that is, xL and xR are
    associated to the same point. 
    
    Uses Lagrange 3rd order interpolation on n ordered gridpoints
    on [xL, xR] (where n is the number of points in f(x_p)) to
    accomplish this task.

    Args:
        xp (np.ndarray): A shape (n,) array corresponding to the
            interpolation points. Should be ordered
        f_xp (np.ndarray): A shape (n,) array corresponding to
            the desired function evaluated at each interpolation
            point
        new_x (np.ndarray): A shape (m,l) array of x-values we'd
            like to determine the value of this function at
        domain(tuple[float, float]): The periodic domain [xL, xR]
            for this problem
            
    Returns:
        np.ndarray: A (m, l) array (the same shape as new_x) with
            the interpolated values of the function at each of 
            the points in new_x.
    """
    # Get all elements to be correct representative inside periodic domain 
    xL, xR = domain
    len_domain = (xR - xL)
    new_x = np.mod(new_x - xR, len_domain) + xL

    # Sort interpolation points from least to greatest 
    sorted_indexes = np.argsort(xp)
    xp = xp[sorted_indexes]
    f_xp = f_xp[sorted_indexes]

    # Now, iterate through all possible interpolation points
    # and perform barycentric lagrange interpolation on them 


    # Make our way through all gridpoints, 
    dx = (domain[1] - domain[0])
