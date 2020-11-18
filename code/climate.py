import json
import numpy as np
import scipy.integrate
import pandas as pd
from scipy.optimize import least_squares


class Ode(scipy.integrate.ode):
    """ An interface for ode integration.
    
    This is specialized from scipy.integrate.ode to handle the rk4 and euler
    integration methods developed in EOSC 453.
    
    Args:
        *args: Arguments to pass scipy.integrate.ode
        **kwargs: Keyword arguments to pass scipy.integrate.ode
        
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def set_integrator(self, name, **integrator_params):
        """Define the ode integrator.
        
        Use a special case for the ubc_rk4 and ubc_euler integrators
        developed in EOSC 453. Otherwise, use the standard scipy
        integrators.
        
        Args:
            name: The name of the integrator to use. This can be any
                integrator supported by scipy.integrate.ode or
                'ubc_rk4' or 'ubc_euler'.
            **integrator_params: Integrator params to pass to
            scipy.integrate.ode
            
        """
        if name == 'ubc_rk4':
            self._ubc_integrator = name
            self.integrate = self._ubc_rk4
        elif name == 'ubc_euler':
            self._ubc_integrator = name
            self.integrate = self._ubc_euler
        else:
            super().set_integrator(name, **integrator_params)
    
    def solve(self, n, tf):
        """Solve the ode system.
        
        Args:
            n (int): The number of time steps.
            tf (number): The stop time.
            
        Returns:
            t (array(number)): An (n+1)-dimensional array containing the time
            steps and initial time.
            Y (array(number)): An (m, n+1)-dimensional array containing the
            solution. m is the number of variables, n is the number of time 
            steps.
                
        """
        Y = np.zeros((n + 1, self.y.size))
        t = np.zeros(n + 1)
        # Initial conditions
        Y[0, :] = self.y
        t[0] = self.t
        h = (tf - self.t) / n
        for i in range(n):
            y = self.integrate(self.t + h)
            Y[i + 1, :] = y
            t[i + 1] = self.t
        return t, Y
            
    def _ubc_rk4(self, t1):
        # An implementation of RK4 based on Assignment 0 of EOSC 453
        t0 = self.t # Start time
        y0 = self.y # Start value
        h = t1 - t0 # Step size
        k1 = h * self.f(t0, y0)
        k2 = h * self.f(t0 + h / 2, y0 + k1 / 2)
        k3 = h * self.f(t0 + h / 2, y0 + k2 / 2)
        k4 = h * self.f(t0 + h, y0 + k3)
        dy = (k1 + 2 * k2 +  2 * k3 + k4) / 6
        y1 = y0 + dy
        self.set_initial_value(y1, t1)
        return y1
    
    def _ubc_euler(self, t1):
        # An implementation of the Euler method based on Assignment 0 of
        # EOSC 453
        t0 = self.t # Start time
        y0 = self.y # Start value
        h = t1 - t0 # Step size
        dy = h * self.f(t0, y0)
        y1 = y0 + dy
        self.set_initial_value(y1, t1)
        return y1


class EarthModel():
    """A box model to investigate the effects of volcanism of Earth's energy
    balance
    """
    
    def __init__(self, file):
        self.size = 6 # Number of zones

        with open(file) as f:
            params = json.load(f)

        # Model wide properties
        self.land_albedo = params['Land Albedo']
        self.land_density = params['Land Density']
        self.land_thermal_depth = params['Land Thermal Scale Depth']
        self.land_specific_heat = params['Land Specific Heat Capacity']
        self.ocean_albedo = params['Ocean Albedo']
        self.ocean_density = params['Ocean Density']
        self.ocean_thermal_depth = params['Ocean Thermal Scale Depth']
        self.ocean_specific_heat = params['Ocean Specific Heat Capacity']
        self.ice_albedo = params['Ice Albedo']
        self.ice_density = params['Ice Density']
        self.ice_thermal_depth = params['Ice Thermal Scale Depth']
        self.ice_specific_heat = params['Ice Specific Heat Capacity']
        self.sky_albedo = params['Atmospheric Albedo']
        self.sky_transmissivity = params['Atmospheric Transmissivity']
        self.earth_emissivity = params['Earth Total Emissivity']
        self.earth_radius = params['Earth Radius']
        self.stefan_boltzmann = params['Stefan-Boltzmann Constant']
        self.solar_constant = params['Solar Constant']
        
        # Zone properties
        self.zone_gamma = self._read_zone_param('Geometric Factor', params)
        self.zone_area_frac = self._read_zone_param("Area Fraction", params)
        self.zone_land = self._read_zone_param('Land Fraction', params)
        self.zone_ocean = self._read_zone_param('Ocean Fraction', params)
        self.zone_ice = self._read_zone_param('Ice Fraction', params)

        # Average zone area
        self.zone_area = self.zone_area_frac * np.pi * self.earth_radius**2
        # Average zone albedo
        self.zone_albedo = self._zone_average(
            self.land_albedo, self.ocean_albedo, self.ice_albedo
        )
        # Average zone density
        self.zone_density = self._zone_average(
            self.land_density, self.ocean_density, self.ice_density
        )
        # Average zone specific heat capacity
        self.zone_specific_heat = self._zone_average(
            self.land_specific_heat, self.ocean_specific_heat,
            self.ice_specific_heat
        )
        # Average zone thermal depth
        self.zone_thermal_depth = self._zone_average(
            self.land_thermal_depth, self.ocean_thermal_depth,
            self.ice_thermal_depth
        )
        # Average zone product of density, specific heat and thermal depth
        self.zone_beta = self._zone_average(
            self.land_density * self.land_specific_heat * self.land_thermal_depth,
            self.ocean_density * self.ocean_specific_heat * self.ocean_thermal_depth,
            self.ice_density * self.ice_specific_heat * self.ice_thermal_depth,
        )

        # Zone boundary properties
        self.L12 = params["L12"]
        self.L23 = params["L23"]
        self.L34 = params["L34"]
        self.L45 = params["L45"]
        self.L56 = params["L56"]
        self.k12 = params["k12"]
        self.k23 = params["k23"]
        self.k34 = params["k34"]
        self.k45 = params["k45"]
        self.k56 = params["k56"]

        # Integrator parameters
        self.t0 = None # Initial time
        self.tf = None # Final time
        self.tn = None # Number of time steps
        self.T0 = None # Initial value
        self.method = 'ubc_rk4' # Integrator method
        self.ode = None

        # No occlusion by default
        self.occlusion = lambda t: np.ones(self.size)

        # Lag times (in seconds) between zones for a volcanic eruption
        lag_0 = 0
        # Lag 1: 30 degrees traveled
        lag_1 = 3 # months
        # Lag 2: 30 additional degrees traveled (10 at rate 1, 20 at rate 2)
        lag_2 = lag_1 + 1 + 20 / 18
        # The remaining lag times follow the rate from the second line
        lag_3 = lag_2 + 30 / 18
        lag_4 = lag_3 + 30 / 18
        lag_5 = lag_4 + 30 / 18
        # Convert to seconds
        self.lags = np.array(
            [lag_0, lag_1, lag_2, lag_3, lag_4, lag_5]
        ) / 12 * 365.25 * 24 * 3600

        # The maximum time to consider the effect of an eruption
        self.max_eruption_time = 100 * 365.25 * 24 * 3600

        # Zone eruption times and positions
        self.lag_k = None
        self.lag_t = None

        # Temperature dependent albedo parameters
        self.temp_i = 260 # K
        self.temp_0 = 290 # K
        self.alpha_max = self.ice_albedo

    def set_occlusion(self, occlusion):
        self.occlusion = occlusion

    def lagged_phi(self, k, t):
        """Returns the lagged oclusion factor for zones away from an eruption.

        Args:
            k (int): The number of zones away from the eruption.
            t (float): The time since the eruption in seconds.

        Returns:
            float: The occlusion factor for the zone.
        """
        return self.occlusion(t) if t >= self.lags[k] else 1.0

    @property
    def boundary_length(self):
        return np.array([
            [0, self.L12, 0, 0, 0, 0],
            [self.L12, 0, self.L23, 0, 0, 0],
            [0, self.L23, 0, self.L34, 0, 0],
            [0, 0, self.L34, 0, self.L45, 0],
            [0, 0, 0, self.L45, 0, self.L56],
            [0, 0, 0, 0, self.L56, 0],
        ])

    @property
    def boundary_conductivity(self):
        return np.array([
            [0, self.k12, 0, 0, 0, 0],
            [self.k12, 0, self.k23, 0, 0, 0],
            [0, self.k23, 0, self.k34, 0, 0],
            [0, 0, self.k34, 0, self.k45, 0],
            [0, 0, 0, self.k45, 0, self.k56],
            [0, 0, 0, 0, self.k56, 0],
        ])

    @property
    def boundary_matrix(self):
        K = self.boundary_length * self.boundary_conductivity
        # Add diagonal term
        diag = -1 * K.sum(axis=0)
        np.fill_diagonal(K, diag)
        return K

    @property
    def flux_out(self):
        return (
            self.earth_emissivity * self.sky_transmissivity *
            self.stefan_boltzmann
        )

    @property
    def flux_in(self):
        return (self.zone_gamma * self.solar_constant *
            (1 - self.sky_albedo) * (1 - self.zone_albedo)
        )

    @property
    def temp_range(self):
        return (self.temp_i - self.temp_0)**2


    def flux_in_albedo(self, T):
        return (self.zone_gamma * self.solar_constant *
            (1 - self.sky_albedo) * (1 - self.alpha(T))
        )

    def build_eruptions(self, zones, times):
        n = len(zones) # number of eruptions
        lag_k = np.empty((6, n), dtype=int)
        for i in range(6):
            lag_k[i] = np.array([abs(i - zone) for zone in zones], dtype=int)

        self.lag_k = lag_k
        self.lag_t = np.array(times)
    
    def zone_phi(self, zone, t, idxs):
        n = len(idxs)
        lag_k = self.lag_k[zone, idxs]
        lag_t = self.lag_t[idxs]
        phi_sum = np.sum([
            self.lagged_phi(k, t - dt) for k, dt in zip(lag_k, lag_t)
        ])
        return np.max([0, 1 - n + phi_sum])

    def phi(self, t):
        # Only consider eruptions that have happend in the past
        # Don't consider eruptions that have happend too far in the past
        eruption_indices = np.where(
            (self.lag_t <= t) &
            (t - self.lag_t < self.max_eruption_time)
        )[0]
        return np.array([
            self.zone_phi(0, t, eruption_indices),
            self.zone_phi(1, t, eruption_indices),
            self.zone_phi(2, t, eruption_indices),
            self.zone_phi(3, t, eruption_indices),
            self.zone_phi(4, t, eruption_indices),
            self.zone_phi(5, t, eruption_indices)
        ])

    def quadratic_albedo(self, T):
        a0 = self.zone_albedo
        ai = self.alpha_max * np.ones_like(T)
        T0 = self.temp_0
        return a0 + (ai - a0) * (T - T0)**2 / self.temp_range

    def alpha(self, T):
        mask1 = T <= self.temp_i
        mask2 = T >= self.temp_0
        return np.where(
            mask1, self.alpha_max * np.ones_like(T),
            np.where(
                mask2, self.zone_albedo, self.quadratic_albedo(T)
            )
        )

    def flux_balance(self, T):
        flux_in = self.flux_in
        flux_out = self.flux_out * T**4
        flux_zone = np.matmul(self.boundary_matrix, T) / self.zone_area
        return (flux_in - flux_out + flux_zone) / self.zone_beta

    def flux_balance_eruptions(self, t, T):
        flux_in = self.flux_in * self.phi(t)
        flux_out = self.flux_out * T**4
        flux_zone = np.matmul(self.boundary_matrix, T) / self.zone_area
        return (flux_in - flux_out + flux_zone) / self.zone_beta

    def flux_balance_eruptions_albedo(self, t, T):
        flux_in = self.flux_in_albedo(T) * self.phi(t)
        flux_out = self.flux_out * T**4
        flux_zone = np.matmul(self.boundary_matrix, T) / self.zone_area
        return (flux_in - flux_out + flux_zone) / self.zone_beta

    def flux_balance_albedo(self, t, T):
        flux_in = self.flux_in_albedo(T)
        flux_out = self.flux_out * T**4
        flux_zone = np.matmul(self.boundary_matrix, T) / self.zone_area
        return (flux_in - flux_out + flux_zone) / self.zone_beta

    def build(self):
        self.build_ode(lambda t, T: self.flux_balance_eruptions(t, T))

    def build_ode(self, f):
        self.ode = Ode(f)
        self.ode.set_initial_value(self.T0, self.t0)

    def solve(self):
        t, T = self.ode.solve(self.tn, self.tf)
        self.t = t
        self.T = T

    def _read_zone_param(self, param, params):
        n = self.size
        return np.array(
            [params['Zone']['{}'.format(i)][param] for i in range(1, n + 1)]
        )

    def _zone_average(self, land, ocean, ice):
        return (
            self.zone_land * land +
            self.zone_ocean * ocean +
            self.zone_ice * ice
        )

# Fit curve to eruption data
def fit_eruption(year, value, weights=None):
    if weights is None:
        weights = np.ones_like(value)

    # Translate eruption time to a small number greater than 0
    eps = 0.01
    t = year - year[0] + eps

    # Transform values to look like ~ 1/t
    data = value / value[-1]
    data = 1 / data
    data = data - 1

    # Find a least squares fit to the eruption curve using a 1/t function
    sol = least_squares(lambda x: weights * (x[0]/(t - x[1])**2 - data), [1, 0])

    # Transform to an "occluding" function that reduces incoming radiation
    # for time in seconds since the eruption
    def phi(t):
        # Function was fit in years.
        t = t / 365.25 / 24 / 3600
        return 1 / (sol.x[0]/(t - sol.x[1])**2 + 1)

    return phi

def fit_eruption_data(files):
    file_1 = files[0]
    file_2 = files[1]

    # Eruption 1 fit
    df = pd.read_csv(file_1)
    phi_1= fit_eruption(df['date'].values, df['value'].values)
    
    # Eruption 2 fit
    df = pd.read_csv(file_2)
    phi_2 = fit_eruption(df['date'].values, df['value'].values)

    # Take phi as the average of the two fits
    return lambda t: 0.5 * (phi_1(t)  + phi_2(t))
