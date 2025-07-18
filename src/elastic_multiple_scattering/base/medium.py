from typing import Self
import numpy as np

from .waves import IncidentPlanePWave
from .obstacles import BaseObstacle


class LinearElasticMedium:
    """A class for representing a linearly elastic medium in a
    time-harmonic scattering problem.
    
    This medium is assumed to be isotropic and homogeneous (outside 
    the scattering domains of interest), with constant physical 
    properties. These properties depend on one another in various 
    ways; see https://www.brown.edu/Departments/Engineering/Courses/EN224/Image146.gif
    
    Attributes:
        rho (float): The density of the medium
        mu (float): The Lame constant/Shear modulus mu
        lam (float): The Lame constant lambda 
        E (float): The Young's modulus of the medium
        nu (float): The Poisson's ratio of the medium
        omega (float): The frequency of the time-harmonic behavior 
            in this scattering problem 
        kp (float): The compressional wavenumber of this medium
        ks (float): The shear wavenumber of this medium
    """
    def __init__(
        self, 
        rho: float,
        mu: float,
        lam: float,
        E: float,
        nu: float, 
        omega: float
    ):
        """Initialize a linearly elastic medium.
        
        WARNING: Do not use this constructor explicitly. Use one of
        the follwing class methods to implicitly get the correct
        dependencies between values:

        * LinearElasticMedium.from_lame_constants()
        * LinearElasticMedium.from_young_and_poisson()
        * LinearElasticMedium.from_young_and_mu()
        * LinearElasticMedium.from_lam_and_poisson()
        * LinearElasticMedium.from_young_as_poisson()
        * LinearElasticMedium.from_poisson_and_mu()

        Parameters:
            rho (float): The density of the medium
            mu (float): The Lame constant/Shear modulus mu
            lam (float): The Lame constant lambda 
            E (float): The Young's modulus of the medium
            nu (float): The Poisson's ratio of the medium
            omega (float): The frequency of the time-harmonic
                behavior in this scattering problem 
        """
        # Store material constants
        self.rho = rho
        self.mu = mu 
        self.lam = lam
        self.E = E 
        self.nu = nu
        self.omega = omega 
        
        # Get compressional/shear wavenumbers 
        self.kp = np.sqrt(rho * omega**2 / (lam + 2 * mu))
        self.ks = np.sqrt(rho * omega**2 / mu)

        # Get compressional/shear wavelengths 
        self.wavelength_p = 2 * np.pi / self.kp
        self.wavelength_s = 2 * np.pi / self.ks

    @classmethod 
    def from_young_poisson_lambda_s(
        cls,
        E: float,
        nu: float,
        lambda_s: float,
        omega: float
    ) -> Self:
        """Constructs a linearly elastic medium from young/poisson
        and the shear wavelength.
        
        Args:
            E (float): The Young's modulus for this problem
            nu (float): The Poisson's ratio of the problem
            lambda_s (float): The shear wavelength of the medium
            omega (float): The frequency of the time-harmonic
                behavior in this scattering problem 
        """
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        rho = mu / (lambda_s * omega / (2 * np.pi))**2
        return cls(rho, mu, lam, E, nu, omega)
    
    @classmethod
    def from_lame_constants(
        cls,
        lam: float,
        mu: float,
        rho: float,
        omega: float
    ) -> Self:
        """Constructs a linearly elastic medium from Lame constants
        and density.
        
        Args:
            lam (float): The Lame constant lambda of the medium
            mu (float): The Lame constant/Shear modulus mu of
                the medium
            rho (float): The density of the medium
            omega (float): The frequency of the time-harmonic
                behavior in this scattering problem 
        """
        E = mu * (3 * lam + 2 * mu) / (lam + mu)
        nu = lam / (2 * (lam + mu))
        return cls(rho, mu, lam, E, nu, omega)
    
    @classmethod
    def from_young_poisson_rho(
        cls,
        E: float,
        nu: float,
        rho: float, 
        omega: float
    ) -> Self:
        """Constructs a linearly elastic medium from Young's modulus,
        Poisson's ratio, and density.
        
        Args:
            E (float): The Young's modulus of the medium
            nu (float): The Poisson's ratio of the medium
            rho (float): The density of the medium
            omega (float): The frequency of the time-harmonic
                behavior in this scattering problem 
        """
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))  # TODO: OUR BOOK SEEMS INCORRECT HERE??
        return cls(rho, mu, lam, E, nu, omega)
    
    @classmethod
    def from_young_and_mu(
        cls,
        E: float,
        mu: float,
        rho: float, 
        omega: float
    ) -> Self:
        """Constructs a linearly elastic medium from Young's modulus,
        the Lame constant/shear modulus mu, and density.
        
        Args:
            E (float): The Young's modulus of the medium
            mu (float): The Lame constant/shear modulus mu of
                the medium
            rho (float): The density of the medium
            omega (float): The frequency of the time-harmonic
                behavior in this scattering problem 
        """
        lam = mu * (E - 2 * mu)/(3 * mu - E)
        nu = (E - 2 * mu) / (2 * mu)
        return cls(rho, mu, lam, E, nu, omega)
    
    @classmethod
    def from_lam_and_poisson(
        cls,
        lam: float,
        nu: float,
        rho: float, 
        omega: float
    ) -> Self:
        """Constructs a linearly elastic medium from Lame's constant
        lambda, Poisson's ration nu, and density.
        
        Args:
            lam (float): The Lame constant lambda of the medium
            nu (float): The Poisson's ratio of the medium
            rho (float): The density of the medium
            omega (float): The frequency of the time-harmonic
                behavior in this scattering problem  
        """
        mu = lam * (1 - 2 * nu) / (2 * nu)
        E = lam * (1 + nu) * (1 - 2 * nu) / nu
        return cls(rho, mu, lam, E, nu, omega)
    
    @classmethod
    def from_poisson_and_mu(
        cls,
        nu: float,
        mu: float,
        rho: float, 
        omega: float
    ) -> Self:
        """Constructs a linearly elastic medium from Poisson's ratio
        nu, the Lame constant/Shear modulus mu, and density.
        
        Args:
            nu (float): The Poisson's ratio of the medium
            mu (float): The Lame constant/Shear modulus mu
            rho (float): The density of the medium
            omega (float): The frequency of the time-harmonic
                behavior in this scattering problem 
        """
        lam = (2 * mu * nu) / (1 - 2 * nu)
        E = 2 * mu * (1 + nu)
        return cls(rho, mu, lam, E, nu, omega)
    

    

    

