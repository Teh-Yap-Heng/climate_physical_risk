# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:16:27 2023
@author: YAPHENGTEH

"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import xarray as xr
import rioxarray
from xarray import Dataset
from numpy import ndarray
import os.path
from xarray import DataArray
from xarray.core.dataset import Dataset
from path import Path
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

# Example: Geocoordinate for Seksyen 25, 40400 Shah Alam, Selangor
lon = 101.527
lat = 3.032

# mf_load_tif: load multiple .tif files
os.chdir(r'C:\Users\yaphengteh\OneDrive')
file_list1 = ['inunriver_historical_000000000WATCH_1980_rp00001.tif','inunriver_historical_000000000WATCH_1980_rp00002.tif',
              'inunriver_historical_000000000WATCH_1980_rp00005.tif','inunriver_historical_000000000WATCH_1980_rp00010.tif',
              'inunriver_historical_000000000WATCH_1980_rp00025.tif','inunriver_historical_000000000WATCH_1980_rp00050.tif',
              'inunriver_historical_000000000WATCH_1980_rp00100.tif','inunriver_historical_000000000WATCH_1980_rp00250.tif',
              'inunriver_historical_000000000WATCH_1980_rp00500.tif','inunriver_historical_000000000WATCH_1980_rp01000.tif']

tif_data1 = xr.open_mfdataset( # open multiple .tif files as single file
    file_list1,
    engine="rasterio",
    combine="nested",
    concat_dim="probabilities",
    parallel=True,
    chunks={"x": 100, "y": 100}
)

# extract ARI from the filename and reciprocal it to get probabilities
probs = ['0001','0002','0005','0010','0025','0050','0100','0250','0500','1000']
probs = np.reciprocal(np.array(probs).astype(float))

# assign probabilities to combined .tif files' concat_dim (for the concat_dim probabilities naming)
tif_data1["probabilities"] = probs

# rename "x" as longitude and "y" as latitude, "band_data" is flood risk data from .tif files
if "x" in tif_data1:
    tif_data1 = tif_data1.rename({"x": "lon"})
if "y" in tif_data1:
    tif_data1 = tif_data1.rename({"y": "lat"})
if "band_data" in tif_data1:
    tif_data1 = tif_data1.rename({"band_data": "flood_risk"})

# extract information from the combined .tif files and export it to netcdf format
# this will try to search the nearest coordinate available in the .tif files (point_data1.flood_risk)
point_data1 = tif_data1.sel(indexers={'lon': lon, 'lat': lat}, method='nearest') 
point_data1.to_netcdf('test.nc') 

# monte carlo simulation
damage_paths = 1000
first_year = 2020
last_year = 2050
simulation_years = last_year - first_year + 1
inundation = xr.open_dataset('test.nc').flood_risk
np.random.seed(1)
random_floods = np.random.uniform(0, 1, size=(damage_paths, simulation_years))

# interpolate inundation.data with random_floods.flatten() probabilities, then reshape it back to 31000, 1000 x 31 format array
hist_inundation_interp = inundation.interp(probabilities=random_floods.flatten(), 
                                           method="cubic").values.flatten().reshape((damage_paths, simulation_years))

inun_ds = inundation.data
probabilities = random_floods.flatten() # all columns become single column of rows, then single column of rows append to the other rows
hist_inundation_interp_ds = inundation.interp(probabilities=random_floods.flatten(), method="cubic").data

###############################################################################
# extract rcp8.5 from .tif files
os.chdir(r'C:\Users\yaphengteh\OneDrive')
file_list2 = ['inunriver_rcp8p5_00000NorESM1-M_2050_rp00001.tif','inunriver_rcp8p5_00000NorESM1-M_2050_rp00002.tif',
              'inunriver_rcp8p5_00000NorESM1-M_2050_rp00005.tif','inunriver_rcp8p5_00000NorESM1-M_2050_rp00010.tif',
              'inunriver_rcp8p5_00000NorESM1-M_2050_rp00025.tif','inunriver_rcp8p5_00000NorESM1-M_2050_rp00050.tif',
              'inunriver_rcp8p5_00000NorESM1-M_2050_rp00100.tif','inunriver_rcp8p5_00000NorESM1-M_2050_rp00250.tif',
              'inunriver_rcp8p5_00000NorESM1-M_2050_rp00500.tif','inunriver_rcp8p5_00000NorESM1-M_2050_rp01000.tif']

tif_data2 = xr.open_mfdataset(
    file_list2,
    engine="rasterio",
    combine="nested",
    concat_dim="probabilities",
    parallel=True,
    chunks={"x": 100, "y": 100}
)

# assign probabilities to combined .tif files
tif_data2["probabilities"] = probs

if "x" in tif_data2:
    tif_data2 = tif_data2.rename({"x": "lon"})
if "y" in tif_data2:
    tif_data2 = tif_data2.rename({"y": "lat"})
if "band_data" in tif_data2:
    tif_data2 = tif_data2.rename({"band_data": "flood_risk"})

point_data2 = tif_data2.sel(indexers={'lon': lon, 'lat': lat}, method='nearest')
point_data2.to_netcdf('test.nc')

damage_paths = 1000
first_year = 2020
last_year = 2050
simulation_years = last_year - first_year + 1
inundation_rcp85 = xr.open_dataset('test.nc').flood_risk
np.random.seed(1)
random_floods = np.random.uniform(0, 1, size=(damage_paths, simulation_years))

rcp85_inundation_interp = inundation_rcp85.interp(probabilities=random_floods.flatten(),
                                                  method="cubic").values.flatten().reshape((damage_paths, simulation_years))

inun_ds_85 = inundation_rcp85.data
hist_inundation_interp_ds_85 = inundation_rcp85.interp(probabilities=random_floods.flatten(), method="cubic").data

###############################################################################
year_list = np.arange(first_year, last_year + 1, 1) # create year list for 2030 to 2050

year_matrix = np.array([(year_list - first_year)] * damage_paths)

total_flood = hist_inundation_interp + ((rcp85_inundation_interp - 
                                         hist_inundation_interp) * year_matrix) / (last_year - first_year)

###############################################################################
# Define climate protection level (building elevation)
name_comp = 'Selangor'
geo_unit_comp = 'Malaysia'
protection_level = 27.29509926

# set protection_level to be at least 30
protection_level = max(30, int(protection_level))
 
climate_protection_increase = 1 # estimated increase factor for the protection level in the future period (1 = no increase)
protection_level_climate = protection_level / climate_protection_increase

protection_level_river = protection_level
protection_level_river_climate = protection_level_river

protection_levels = pd.DataFrame({
    "asset_id": [],
    "asset": [],
    "historical_coast": [],
    "rcp_85_2050_coast": [],
    "historical_river": [],
    "rcp_85_2050_river": []
})

protection_level_i = pd.DataFrame({
     "asset_id": ['1'],
     "asset": ['Seksyen 25, 40400 Shah Alam, Selangor'],
     "historical_coast": [30],
     "rcp_85_2050_coast": [30],
     "historical_river": [30],
     "rcp_85_2050_river": [30]
})

protection_levels = pd.concat([protection_levels, protection_level_i])

climate_protection = {
    "asset_id": [],
    "coast": [],
    "river": []
}

protection_historical_coast = inundation.interp(probabilities=1 / 30, method="cubic").values.flatten()
protection_projection_coast = inundation_rcp85.interp(probabilities=1 / 30, method="cubic").values.flatten()
climate_scale_protection_coast = (protection_projection_coast - protection_historical_coast) / (last_year - first_year)
climate_protection['coast'].append(protection_historical_coast + climate_scale_protection_coast * (year_list - first_year))

protection_historical_river = inundation.interp(probabilities=1 / 30, method="cubic").values.flatten()
protection_projection_river = inundation_rcp85.interp(probabilities=1 / 30, method="cubic").values.flatten()
climate_scale_protection_river = (protection_projection_river - protection_historical_river) / (last_year - first_year)
climate_protection['river'].append(protection_historical_river + climate_scale_protection_river * (year_list - first_year))

climate_protection['asset_id'].append(1)
climate_protection = pd.DataFrame(climate_protection)

total_flood_where = np.where(total_flood < climate_protection.iloc[0,2], 0, total_flood) # coast is [0,1] ; river is [0,2]

###############################################################################
# Train model based on building type
os.chdir(r'C:\Users\yaphengteh')
df = pd.read_excel('global_flood_damage_functions.xlsx', sheet_name='DamageAsia')

# train damage function
# Commercial 
commercial_data = df[(df['PropertyType'] == "Commercial") & (df['LnDamage'].isna() == False) & (df['FloodLevel'] != 0)]
y_data = commercial_data['LnDamage']
x_data = commercial_data['FloodLevel']
x_data = sm.add_constant(x_data)
damage_model = sm.OLS(y_data, x_data).fit()

max_damage_commercial_structure = 572
max_damage_commercial_content = 572
inflation = 0.02
max_damage = (max_damage_commercial_structure + max_damage_commercial_content) * (1 + inflation) ** 10

flood_level = total_flood_where
damage_residential_ln = damage_model.predict([1, flood_level])[0] # [1, flood_level] -> constant of 1, flood_level = total_flood
damage_residential = np.exp(damage_residential_ln) / (1 + np.exp(damage_residential_ln)) * max_damage
damage_residential = np.where(flood_level == 0, 0, damage_residential)

'''
PHYSICAL RISK
'''
first_year = 2020
year_list = np.arange(first_year, last_year + 1, 1) # create year list for 2030 to 2050

os.chdir(r'C:\Users\yaphengteh\OneDrive')
portfolio = pd.read_csv('show_data.csv', sep=';')
portfolio[['LTV', 'LGD Collateral (Market value)']] = 0
portfolio['LTV'] = portfolio['EAD'] / (portfolio['Collateral (Market value)'] * (1 - portfolio['Haircut']))

# LGD only given by orig_debt of Collateral (Market value) or security
adjusted_lgd = (1 - 1 / portfolio["LTV"]) * portfolio["LGD"]
adjusted_lgd = np.where(adjusted_lgd < 0, 0, adjusted_lgd)
portfolio['LGD Collateral (Market value)'] = np.where(portfolio['Secured'] == 1, portfolio['LGD'], adjusted_lgd)

# set up dataframe for financial evolution
risk_features: list = []
for year in year_list:
    risk_features = risk_features + [f"Expenses river EUR/sqm {year}",
                                    f"Expenses coast EUR/sqm {year}",
                                    f"Additional expected expenses river EUR/sqm {year}",
                                    f"Additional expected expenses coast EUR/sqm {year}",
                                    f"Expenses river {year}",
                                    f"Expenses coast {year}",
                                    f"Additional expected expenses river {year}",
                                    f"Additional expected expenses coast {year}",
                                    f"Property Index Riverrisk {year}",
                                    f"Property Index Climate {year}",
                                    f"Adjusted LTV Climate {year}",
                                    f"Adjusted LGD Climate {year}",
                                    f"Expected Loss Climate {year}"]

portfolio[risk_features] = 0.0

# load financials of Collateral (Market value) owner for PD model
orig_pd = portfolio['PD']
orig_debt = portfolio['Debt']
orig_equity = portfolio['Equity']
orig_profit = portfolio['Profit']
stress_debt = portfolio['Debt']

# convert to relative damage
usd_euro = 0.9
m_sf = 10.7639             # 1 square meter = 10.7639 square feet
house_price_sqf = 130      # avg house price in dollar per sqf USA statista 2019 - 120
house_price_sqm = m_sf * house_price_sqf * usd_euro

insurance_premium_factor = 1
damage_array_river = pd.DataFrame(damage_residential)
damage_array_river.columns = [y for y in range(2020, 2051)]
# Load damage paths per damage function per location
for key, asset in portfolio.iterrows():
    # damage_array = damage_projections[asset[portfolio_cfg.params.name]]["coast"]
    for year in year_list:
        additional_exp_coast_euro_sqm = f"Additional expected expenses coast EUR/sqm {year}"
        additional_exp_river_euro_sqm = f"Additional expected expenses river EUR/sqm  {year}"
        exp_coast_euro_sqm = f"Expenses coast EUR/sqm {year}"
        exp_river_euro_sqm = f"Expenses river EUR/sqm {year}"
        # asset[additional_exp_coast_euro_sqm] = (damage_array[year].mean() - damage_array[first_year].mean()) * insurance_premium_factor
        asset[additional_exp_river_euro_sqm] = (damage_array_river[year].mean() - damage_array_river[first_year].mean()) * insurance_premium_factor
        # asset[exp_coast_euro_sqm] = damage_array[year].mean() * insurance_premium_factor
        asset[exp_river_euro_sqm] = damage_array_river[year].mean() * insurance_premium_factor
        asset[f"Additional expected expenses coast {year}"] = asset[additional_exp_coast_euro_sqm] / house_price_sqm
        asset[f"Additional expected expenses river {year}"] = asset[additional_exp_river_euro_sqm] / house_price_sqm
        asset[f"Expenses coast {year}"] = asset[exp_coast_euro_sqm] / house_price_sqm
        asset[f"Expenses river {year}"] = asset[exp_river_euro_sqm] / house_price_sqm
    portfolio.loc[key] = asset

def simulate_pd_merton(orig_pd, orig_debt, orig_equity, orig_profit, stress_debt, stress_equity, stress_profit):
    # pd model given financials of the company
    # return on assets represented in continuous compounding
    orig_mu = np.log(1 + orig_profit / (orig_equity + orig_debt))
    orig_dt_d = norm.ppf(q=(1 - orig_pd), loc=0, scale=1)
    # calibrate asset volatility from inputs consistent to original PD (t=1 year)
    b = orig_dt_d
    c = 2 * (-orig_mu - np.log(1 + (orig_equity / orig_debt)))
    sigma = -b + np.sqrt(b ** 2 - c)

    stress_mu = np.log(1 + stress_profit / (stress_equity + stress_debt))
    stress_dt_d = orig_dt_d + (np.log(1 + stress_equity / stress_debt) -
                               np.log(1 + orig_equity / orig_debt) + stress_mu - orig_mu) / sigma
    # transform back dd
    pd_merton = 1 - norm.cdf(stress_dt_d, loc=0, scale=1)
    # If the logarithm gives nan, we predict a PD of 1
    pd_merton = np.nan_to_num(pd_merton, nan=1)
    return pd_merton

# Climate LGD per year - calculate evolution of financials and predict pd
for year in year_list:
    # scale the expected damage to a houseprice shook via rent/price multiplier
    # and keep track of property index development for each property
    # portfolio.to_csv("portfolio.csv")
    portfolio[f"Property Index Climate {year}"] = 1 - (portfolio["Additional expected expenses coast " + str(year)] + 
                                                       portfolio[f"Additional expected expenses river {year}"]) * portfolio["Rent_Price"]
    portfolio[f"Property Index Climate {year}"] = np.where(portfolio[f"Property Index Climate {year}"] < 0, 0, portfolio[f"Property Index Climate {year}"])
    portfolio[f"Property Index Fullrisk {year}"] = 1 - (portfolio["Expenses coast " + str(year)] + portfolio[f"Expenses river {year}"]) * portfolio["Rent_Price"]
    portfolio[f"Property Index Fullrisk {year}"] = np.where(portfolio[f"Property Index Fullrisk {year}"] < 0, 0, portfolio[f"Property Index Fullrisk {year}"])

    # calculate climate adjusted LTVs via the development of the houseprice index due to increased damage
    portfolio[f"Adjusted LTV Climate {year}"] = portfolio["EAD"] / ((portfolio["Collateral (Market value)"] * 
                                                                     portfolio[f"Property Index Climate {year}"]) * (1 - portfolio["Haircut"]))

    # Equity is stressed by decreased value of Collateral (Market value)
    # due to a change in expected climate related damages which are substracted
    # from the market value of the property, if then the market value falls
    # below book value it stresses equity
    portfolio[f'Stress Equity {year}'] = portfolio['Equity'] - np.where(portfolio['Collateral (Market value)'] * portfolio[f"Property Index Climate {year}"] < portfolio['Collateral (Book value)'],
                                                                        portfolio['Collateral (Book value)'] - portfolio['Collateral (Market value)'] * portfolio[f"Property Index Climate {year}"], 0)
    # profit is stressed due to increased repair costs on Collateral (Market value)
    portfolio[f'Stress Profit {year}'] = portfolio['Profit'] - (portfolio[f"Additional expected expenses river {year}"] + 
                                                                portfolio[f"Additional expected expenses coast {year}"]) * portfolio['Collateral (Market value)']
    # call Merton model for PD calculation
    stress_equity = portfolio[f'Stress Equity {year}']
    stress_profit = portfolio[f'Stress Profit {year}']
    portfolio[f'PD {year}'] = simulate_pd_merton(orig_pd, orig_debt,orig_equity, orig_profit, stress_debt, stress_equity,stress_profit)

    # calculate LGD from new LTV and derive ECL
    adjusted_lgd_climate = (1 - 1 / portfolio[f"Adjusted LTV Climate {year}"]) * portfolio["LGD"]
    adjusted_lgd_climate = np.where(adjusted_lgd_climate < 0, 0, adjusted_lgd_climate)
    portfolio[f"Adjusted LGD Climate {year}"] = np.where(portfolio["Secured"] == 1, portfolio["LGD"], adjusted_lgd_climate)
    portfolio[f"Expected Loss Climate {year}"] = portfolio[f'PD {year}'] * portfolio["Adjusted LGD Climate " + str(year)] * portfolio["EAD"] - portfolio['PD'] * portfolio["LGD Collateral (Market value)"] * portfolio["EAD"]
    portfolio[f"Expected Loss Climate {year}"] = np.where(portfolio[f"Expected Loss Climate {year}"] < 0, 0, portfolio[f"Expected Loss Climate {year}"])

os.chdir(r'C:\Users\yaphengteh')
pd.DataFrame(portfolio).to_excel('portfolio_test.xlsx', sheet_name='damage')
