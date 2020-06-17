import pandas as pd
import numpy as np

# convert NG flow rate into methane flow rate
def convert_to_CH4_rate(data,CH4_vol_fraction):
  data['CH4_release_mcfd'] = data['total_release_mcfd'] * CH4_vol_fraction
  data['CH4_release_meter_error_mcfd'] = data['total_release_meter_error_mcfd'] * CH4_vol_fraction
  return data

# convert from Imperial to metric unit
def convert_to_metric(data,CH4_density):
  
  metric_data = data.copy()
  kgh_per_mcfd = CH4_density/24   # (kg/mcf) / (hour/day)
  mps_per_mph = 0.44704       # (mile per hour) / (meter per sec)

  for col in data.columns:
    if col.endswith('mcfdmph'):
      metric_data = metric_data.drop(columns=col)
      # extract column name and append new metric unit
      new_col = col[0:-7] + 'kghmps' 
      metric_data[new_col] = data[col]*kgh_per_mcfd/mps_per_mph
    elif col.endswith('mph'):
      metric_data = metric_data.drop(columns=col)
      # extract column name and append new metric unit
      new_col = col[0:-3] + 'mps' 
      metric_data[new_col] = data[col]*mps_per_mph
    
    if col.endswith('mcfd'):
      metric_data = metric_data.drop(columns = col)
      new_col = col[0:-4] + 'kgh'
      metric_data[new_col] = data[col]*kgh_per_mcfd
  
  return metric_data


# select data based on sensitivity cases and wind speed to use
def data_selection(data,case,wind_to_use,height_adj_Factor=(2.5/10)**0.15):

  """ 
  INPUT
  case: select from ['Base','Loose','Strict','All valid zero']
  wind_to_use: select from the METRIC wind speed colmns in the data table. DO NOT CHOOSE DARKSKY DATA. 
  """


  plot_data = data.copy()
  mps_per_mph = 0.44704       # (mile per hour) / (meter per sec)

  # if use DarkSky or NOAA wind, apply a height adjustment factor
  if wind_to_use.startswith('darksky') or wind_to_use.startswith('KSCK') or wind_to_use.startswith('HRRR_windSpeed'):
    plot_data[wind_to_use] = plot_data[wind_to_use]*height_adj_Factor
  
  # find quantification results in kgd based on wind speed measurements
  plot_data['CH4_release_kghmps'] = plot_data.CH4_release_kgh / plot_data[wind_to_use]
  plot_data['closest_plume_quantification_kgh'] = plot_data.closest_plume_quantification_kghmps * plot_data[wind_to_use]
  
  # find quantification error bars (y-axis)
  if wind_to_use.startswith('WS'):    # weather station
      plot_data['quantification_upper_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use]+2*mps_per_mph)  # cup wind meter is ± 2 mph rated accuracy
      plot_data['quantification_lower_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use]-2*mps_per_mph)
  
  elif wind_to_use.startswith('sonic'):
    """
    according to the error table provided by Gill Instrument (KN1201V4)
    reference speed  [2,5,10,20,30,40,50,60] m/s
    for this analysis, assume that if wind speed measured <= 2 m/s, then the RMS speed deviation is 2.98%l if (2,5] m/s then 1.99%, etc.
    """  
    wind_error_mps = np.zeros(plot_data.shape[0])
    for i in range(plot_data.shape[0]):
      if plot_data[wind_to_use][i]<=2:
        wind_error_mps[i] = (plot_data[wind_to_use][i]) * 0.0298
      elif plot_data[wind_to_use][i]<=5:
        wind_error_mps[i] = (plot_data[wind_to_use][i]) * 0.0199
      elif plot_data[wind_to_use][i]<=10:
        wind_error_mps[i] = (plot_data[wind_to_use][i]) * 0.0126
      elif plot_data[wind_to_use][i]<=20:
        wind_error_mps[i] = (plot_data[wind_to_use][i]) * 0.0136
      elif plot_data[wind_to_use][i]<=30:
        wind_error_mps[i] = (plot_data[wind_to_use][i]) * 0.0051
      elif plot_data[wind_to_use][i]<=40:
        wind_error_mps[i] = (plot_data[wind_to_use][i]) * 0.0048
      elif plot_data[wind_to_use][i]<=50:
        wind_error_mps[i] = (plot_data[wind_to_use][i]) * 0.0077
      else:
        wind_error_mps[i] = (plot_data[wind_to_use][i]) * 0.0259 
    plot_data['quantification_upper_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use].values+wind_error_mps)
    plot_data['quantification_lower_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use].values-wind_error_mps)

  elif wind_to_use.startswith('HRRR_windGust_9avg'):
    wind_error_mps = plot_data.HRRR_windGust_9stddev_mps.values
    plot_data['quantification_upper_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use].values+wind_error_mps)
    plot_data['quantification_lower_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use].values-wind_error_mps)

  elif wind_to_use.startswith('HRRR_windSpeed_9avg'):
    wind_error_mps = plot_data.HRRR_windSpeed_9stddev_mps.values
    plot_data['quantification_upper_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use].values+wind_error_mps)
    plot_data['quantification_lower_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use].values-wind_error_mps)

  elif wind_to_use.startswith('HRRR_windGust_27avg'):
    wind_error_mps = plot_data.HRRR_windGust_27stddev_mps.values
    plot_data['quantification_upper_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use].values+wind_error_mps)
    plot_data['quantification_lower_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use].values-wind_error_mps)

  elif wind_to_use.startswith('HRRR_windSpeed_27avg'):
    wind_error_mps = plot_data.HRRR_windSpeed_27stddev_mps.values
    plot_data['quantification_upper_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use].values+wind_error_mps)
    plot_data['quantification_lower_kgh'] = plot_data.closest_plume_quantification_kghmps*(plot_data[wind_to_use].values-wind_error_mps)

  else:  # darksky, HRRR point estimates, and NOAA wind assume accurate
    plot_data['quantification_upper_kgh'] = np.zeros(plot_data.shape[0])    
    plot_data['quantification_lower_kgh'] = np.zeros(plot_data.shape[0])    
 
  # in case the wind speed go negative, set the quantification result to be zero
  for i in range(plot_data.shape[0]):
    if plot_data.quantification_lower_kgh[i]<0:
        plot_data.quantification_lower_kgh[i]=0

  # length of the error bars (y-axis)
  plot_data['quantification_upper_error_kgh'] = plot_data.quantification_upper_kgh - plot_data.closest_plume_quantification_kgh
  plot_data['quantification_lower_error_kgh'] = plot_data.closest_plume_quantification_kgh - plot_data.quantification_lower_kgh

  """ sensitivity cases: exclude data points based on case = ['Loose','Base','Strict'] """
  # Data invalide if the flight was at wrong altitude, or if there were multiple release points, or if the closest plume had major cut off issues
  plot_data = plot_data[data.wrong_flight_altitude==0]
  plot_data = plot_data[plot_data.multiple_release_points==0]
  plot_data = plot_data[plot_data.closest_plume_cut_off==0]
  
  # Exclude data points based on wind speed unless the case is 'all valid'
  if (case != 'All valid nonnegative') & (case != 'All valid zero'):
    if wind_to_use.startswith('WS'):
      plot_data = plot_data[plot_data[wind_to_use]>=2*mps_per_mph]    # exclude data points with wind speed < rated accuracy
    elif wind_to_use.startswith('sonic') or wind_to_use.startswith('KSCK'):
      plot_data = plot_data[plot_data[wind_to_use]>0]    # exclude data points without anemometer measurements

  if case=='Strict':
    plot_data = plot_data[plot_data.blowoff_noticed_right_after_flow_rate_turned_down==0]

  if (case=='Base') | (case=='Strict'):
    # base & strict case exclude cases when time for plume development > 3 minutes (defined as plume length / 1-min gust wind observed) after a change in release reate is made
    full_development = np.ones(plot_data.shape[0])   # assume all fully developed as default
    plot_data = plot_data.reset_index()
    for j in range(plot_data.shape[0]):
      if plot_data.change_from_last_release[j]!=0:   # rate changed
        plume_length = plot_data.closest_plume_length_m[j]
        wind_speed = plot_data[wind_to_use][j]
        travel_time_sec = plume_length / wind_speed
        if travel_time_sec > 180:    # more than 3 minutes
          full_development[j] = 0    # update the plumes not fully developed
          # print(plot_data['pass'][j])
    plot_data = plot_data[full_development==1]
    
    if wind_to_use.startswith('WS'):
      plot_data = plot_data[plot_data.closest_plume_not_fully_developed_WS==0]
    elif wind_to_use.startswith('sonic'):
      plot_data = plot_data[plot_data.closest_plume_not_fully_developed_sonic==0]
      
  # count the number of valid data points and valid non-negative-control data points
  if case == 'All valid zero':
    plot_data = plot_data[plot_data.CH4_release_kgh.astype('float') < 0.0001]
  else:
    plot_data = plot_data[plot_data.CH4_release_kgh.astype('float') > 0.0001]
  
  # replace the false negatives with quantification results = 0
  # plot_data = plot_data.fillna(0)

  return plot_data