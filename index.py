import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location

# Load data
df = pd.read_csv('powerdemand_5min_2021_to_2024_with weather.csv',
                parse_dates=['datetime'],
                index_col='datetime')

# --------------------------------------------------
# 1. LOAD CALIBRATION 
# --------------------------------------------------
# Data shows 2000-8000 kW values - these likely represent:
# Aggregate for about 10 homes (not 100)
homes_in_group = 10  # Adjusted from 100 to 10
df['residential_load_kW'] = df['Power demand'] / homes_in_group

# --------------------------------------------------
# 2. PV GENERATION (TUNED FOR DELHI)
# --------------------------------------------------
site = Location(latitude=28.7041, longitude=77.1025,
               tz='Asia/Kolkata', altitude=216)

# Realistic 10 kW system (larger system to match load scale)
pv_config = {
    'system_capacity': 10,  # kW
    'tilt': 28,
    'azimuth': 180,
    'module_efficiency': 0.20,  
    'temp_coeff': -0.0035,
    'losses': 0.12
}

# Solar position
solar_position = site.get_solarposition(df.index)

# Clear sky irradiance
clearsky = site.get_clearsky(df.index, model='ineichen')
df['ghi'] = clearsky['ghi']
df['dni'] = clearsky['dni']
df['dhi'] = clearsky['dhi']

# POA irradiance
poa_irrad = pvlib.irradiance.get_total_irradiance(
    surface_tilt=pv_config['tilt'],
    surface_azimuth=pv_config['azimuth'],
    dni=df['dni'],
    ghi=df['ghi'],
    dhi=df['dhi'],
    solar_zenith=solar_position['apparent_zenith'],
    solar_azimuth=solar_position['azimuth']
)

# PV output calculation
df['pv_power_kw'] = (
    poa_irrad['poa_global'] * 
    pv_config['system_capacity'] * 
    pv_config['module_efficiency'] *
    (1 + pv_config['temp_coeff'] * (df['temp'] - 25)) * 
    (1 - pv_config['losses'])
)

# Nighttime adjustment
df.loc[(df.index.hour < 6) | (df.index.hour >= 18), 'pv_power_kw'] = 0

# --------------------------------------------------
# 3. NET POWER CALCULATION
# --------------------------------------------------
df['net_power_kW'] = df['residential_load_kW'] - df['pv_power_kw']
df['power_status'] = np.where(df['net_power_kW'] > 0, 'Grid Import', 'Excess Solar')

# Save results
df[['residential_load_kW', 'pv_power_kw', 'net_power_kW']].to_csv('final_power_balance.csv')