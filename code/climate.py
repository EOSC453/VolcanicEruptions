import json
import numpy as np

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
        L12 = params["L12"]
        L23 = params["L23"]
        L34 = params["L34"]
        L45 = params["L45"]
        L56 = params["L56"]
        k12 = params["k12"]
        k23 = params["k23"]
        k34 = params["k34"]
        k45 = params["k45"]
        k56 = params["k56"]
        # Build off diagonal components first
        self.boundary_length = np.array([
            [0, L12, 0, 0, 0, 0],
            [L12, 0, L23, 0, 0, 0],
            [0, L23, 0, L34, 0, 0],
            [0, 0, L34, 0, L45, 0],
            [0, 0, 0, L45, 0, L56],
            [0, 0, 0, 0, L56, 0],
        ])
        self.boundary_conductivity = np.array([
            [0, k12, 0, 0, 0, 0],
            [k12, 0, k23, 0, 0, 0],
            [0, k23, 0, k34, 0, 0],
            [0, 0, k34, 0, k45, 0],
            [0, 0, 0, k45, 0, k56],
            [0, 0, 0, 0, k56, 0],
        ])
        self.boundary_coefficients = (
            self.boundary_length * self.boundary_conductivity
        )
        # Add the diagonal
        diag = -1 * self.boundary_coefficients.sum(axis=0)
        np.fill_diagonal(self.boundary_coefficients, diag)

    def outbound_flux(self):
        return (
            self.earth_emissivity * self.sky_transmissivity *
            self.stefan_boltzmann
        )

    def inbound_flux(self):
        return (
            self.zone_gamma * self.solar_constant *
            (1 - self.sky_albedo) * (1 - self.zone_albedo)
        )

    def vector_field(self, T):
        flux_in = self.inbound_flux()
        flux_out = self.outbound_flux() * T**4
        flux_zone = np.matmul(self.boundary_coefficients, T) / self.zone_area
        return (flux_in - flux_out + flux_zone) / self.zone_beta

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
