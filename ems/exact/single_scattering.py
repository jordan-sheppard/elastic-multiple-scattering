"""Exact solutions for single-scattering problems"""
import numpy as np
from scipy.special import hankel1, j0, j1, jv

from ..base.consts import BoundaryCondition
from ..fd.obstacles import CircularObstacleGeometry
from ..base.waves import IncidentPlanePWave
from ..base.medium import LinearElasticMedium
from ..fd.obstacles import Circular_MKFE_FDObstacle


def exact_solution_hard(
    obstacle: Circular_MKFE_FDObstacle,
    incident_wave: IncidentPlanePWave,
    num_terms: int
):
    """Get the exact solution for a hard obstacle.
    
    Args:
        obstacle (Circular_MKFE_FDObstacle): The obstacle of interest.
            Should be circular.
        incident_wave (IncidentPlanePWave): The incident compressional
            wave on this obstacle.
        num_terms (int): The number of terms used to compute the
            exact/analytical Fourier series solution        
    """
    ## ERROR CHECKING 
    # Check for circular obstacle
    if not isinstance(obstacle, Circular_MKFE_FDObstacle):
        raise TypeError("ERROR: Exact solution only defined for circular obstacles")

    # Check for positive-x direction incident wave
    if not np.allclose(incident_wave.phi, 0):
        raise ValueError("ERROR: Exact solution only defined for incident waves propogating in the positive-x direction (phi=0)")
    
    # Check for center to be at origin
    if not (np.allclose(obstacle.center_global[0], 0) and np.allclose(obstacle.center_global[1], 0)):
        raise ValueError("ERROR: Exact solution only defined for obstacles centered at origin")

    # Check for positive number of terms 
    if num_terms <= 0:
        raise ValueError("ERROR: Must have at least 1 term used in exact solution Fourier expansion")
    
    ## COMPUTE EXACT SOLUTION
    # Parse constants from Pao and Mow pages 253-255
    a = obstacle.r_obstacle
    i = 1j              # Imaginary constant
    phi_0 = 1           # Amplitude of the incident wave
    eps_n = 2           # for n >= 1 (n=0 it is 1)
    alpha = obstacle.parent_medium.kp
    beta = obstacle.parent_medium.ks
    eta = 0.             # Ratio of medium density to obstacle density (e.g., obstacle density is infinite)

    # Get An/Bn coefficients
    A = np.zeros(num_terms + 1, dtype=np.complex128)
    B = np.zeros(num_terms + 1, dtype=np.complex128)

    # Get zero-indexed coefficients
    Hankel_alpha_deriv_0 = -hankel1(1,alpha*a)
    Hankel_beta_deriv_0 = -hankel1(1,beta*a)
    j_alpha_deriv_0 = -jv(1,alpha*a)

    Delta_0 = alpha * beta * a**2 * Hankel_alpha_deriv_0 * Hankel_beta_deriv_0

    A[0] = -1/Delta_0 * alpha * beta * a**2 * j_alpha_deriv_0 * Hankel_beta_deriv_0

    for n in range(1, num_terms + 1):
        Hankel_alpha_deriv = -n/(alpha*a)*hankel1(n,alpha*a) + hankel1(n-1,alpha*a)
        Hankel_beta_deriv = -n/(beta*a)*hankel1(n,beta*a) + hankel1(n-1,beta*a)
        j_alpha_deriv = -n/(alpha*a)*jv(n,alpha*a) + jv(n-1,alpha*a)

        Delta = (
            alpha * beta * a**2 * Hankel_alpha_deriv * Hankel_beta_deriv 
            - n**2 * hankel1(n,alpha*a) * hankel1(n,beta*a)
        )

        A[n] = -(phi_0 * eps_n) * i**n / Delta * (
            alpha * beta * a**2 * j_alpha_deriv * Hankel_beta_deriv
            - n**2 * jv(n,alpha*a) * hankel1(n,beta*a)
        )
        B[n] = -(phi_0 * eps_n) * i**n / Delta * (2*i*n/np.pi); 

    # Compute scattered field
    R_vals = obstacle.grid.r_local 
    Theta_vals = obstacle.grid.theta_local
    phis = np.zeros_like(R_vals, dtype='complex128')
    psis = np.zeros_like(R_vals, dtype='complex128')

    phis += A[0] * hankel1(0, alpha * R_vals)   # Zero-valued terms
    for n in range(1, num_terms + 1):
        phis += A[n] * hankel1(n, alpha * R_vals) * np.cos(n * Theta_vals)
        psis += B[n] * hankel1(n, beta * R_vals) * np.sin(n * Theta_vals)
    
    # Return these values 
    return phis, psis 



def exact_solution_soft(
    obstacle: Circular_MKFE_FDObstacle,
    incident_wave: IncidentPlanePWave,
    num_terms: int
):
    """Get the exact solution for a soft obstacle.
    
    Args:
        obstacle (Circular_MKFE_FDObstacle): The obstacle of interest.
            Should be circular.
        incident_wave (IncidentPlanePWave): The incident compressional
            wave on this obstacle.
        num_terms (int): The number of terms used to compute the
            exact/analytical Fourier series solution        
    """
    ## ERROR CHECKING 
    # Check for circular obstacle
    if not isinstance(obstacle, Circular_MKFE_FDObstacle):
        raise TypeError("ERROR: Exact solution only defined for circular obstacles")

    # Check for positive-x direction incident wave
    if not np.allclose(incident_wave.phi, 0):
        raise ValueError("ERROR: Exact solution only defined for incident waves propogating in the positive-x direction (phi=0)")
    
    # Check for center to be at origin
    if not (np.allclose(obstacle.center_global[0], 0) and np.allclose(obstacle.center_global[1], 0)):
        raise ValueError("ERROR: Exact solution only defined for obstacles centered at origin")

    # Check for positive number of terms 
    if num_terms <= 0:
        raise ValueError("ERROR: Must have at least 1 term used in exact solution Fourier expansion")
    

    ## COMPUTE EXACT SOLUTION
    # Parse constants from Pao and Mow pgs. 240-243 and 391-393
    alpha = obstacle.parent_medium.kp
    beta = obstacle.parent_medium.ks
    i = 1j              # Imaginary constant
    phi_0 = 1           # Amplitude of the incident wave
    r0 = obstacle.r_obstacle

    # Get An/Bn coefficients
    A = np.zeros(num_terms + 1, dtype=np.complex128)
    B = np.zeros(num_terms + 1, dtype=np.complex128)

    for n in range(num_terms + 1): 
        E1_11 = (
            (n**2 + n - beta**2 * (r0**2)/2) * jv(n, alpha*r0)
            - alpha*r0 * jv(n-1, alpha*r0)
        )
        E3_11 = (
            (n**2 + n - beta**2 * (r0**2)/2) * hankel1(n, alpha*r0)
            - alpha * r0 * hankel1(n-1, alpha*r0)
        )
        E3_12 = (
            -n * (n+1) * hankel1(n, beta*r0)
            + n * beta * r0 * hankel1(n-1, beta * r0)
        )
        E1_41 = (
            (n**2 + n) * jv(n, alpha*r0)
            - n * alpha * r0 * jv(n-1, alpha*r0)
        )
        E3_41 = (
            (n**2 + n) * hankel1(n, alpha*r0)
            - n * alpha * r0 * hankel1(n-1, alpha*r0)
        )
        E3_42 = (
            -(n**2 + n - beta**2 * r0**2 / 2) * hankel1(n, beta*r0)
            + beta * r0 * hankel1(n-1, beta * r0)
        )
        delta = np.linalg.det(
            np.array([
                [E3_11, E3_12],
                [E3_41, E3_42]
            ])
        )
        A[n] = -2 * i**n * phi_0 * np.linalg.det(
            np.array([
                [E1_11, E3_12],
                [E1_41, E3_42]
            ])
        )/delta
        B[n] = -2 * i**n * phi_0 * np.linalg.det([
            [E3_11, E1_11],
            [E3_41, E1_41]
        ])/delta
    
    A[0] /= 2 
    B[0] /= 2 

    # Compute scattered field
    R = obstacle.grid.r_local 
    Theta = obstacle.grid.theta_local
    phis = np.zeros_like(R, dtype='complex128')
    psis = np.zeros_like(R, dtype='complex128')

    for n in range(num_terms + 1):
        phis += A[n] * hankel1(n, alpha * R) * np.cos(n * Theta)
        psis += B[n] * hankel1(n, beta * R) * np.sin(n * Theta)
    
    # Return these values 
    return phis, psis 
    

    














