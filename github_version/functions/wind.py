import numpy as np
import pandas as pd
from github_version.functions.parity import linreg_results
from github_version.functions.parity import linreg_results_no_intercept
import matplotlib.pyplot as plt
import time
from datetime import datetime
import matplotlib.dates as mdate
from scipy.stats import norm



# convert the cup meter direction reaidngs from the 16-rose to radian
def convert_16_wind_compass_to_radian(compass_value):
    
  wind_compass = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
  wind_rad = np.pi/8*np.arange(0,16)
  ind = wind_compass.index(compass_value)
  rad = wind_rad[ind]
  
  return rad

# convert from mph to m/s
def convert_to_wind_metric(data):

  metric_data = data.copy()
  mps_per_mph = 0.44704       # (mile per hour) / (meter per sec)

  for col in data.columns:
    if col.endswith('mph'):
      metric_data = metric_data.drop(columns=col)
      # extract column name and append new metric unit
      new_col = col[0:-3] + 'mps' 
      metric_data[new_col] = data[col]*mps_per_mph
  
  return metric_data

# apply an height adjustment factor to darksky wind:
def height_adjust(data,HA_factor=(2.5/10**0.15)):

  '''
HA_factor is the height adjustment factor applied to the Dark Sky wind speed and wind gust data
so as to convert the 10-meter Dark Sky wind to wind speeds at 2.5 meter 

By default, we assume a power law for the vertical wind profile
  '''
  HA_data = data.copy()

  for col in data.columns:
    # adjust if it's darksky wind speed
    if (col.startswith('darksky') or col.startswith('KSCK')) and (col.endswith('mps') or col.endswith('mph')):
      # extract column name and append new metric unit
      new_col = col[0:-3] + 'height_adjusted_' + col[-3]+col[-2]+col[-1]
      HA_data[new_col] = data[col]*HA_factor
  
  return HA_data



# compute hourly average data
def compute_hourly_avg_wind(wind_measurements,NOAA_data,n_threshold = 30):
  """
NOAA reports hourly average wind and reports one number at the 55th minute of each hour
This function compiles the subhourly wind measurements of each hour and computes the hourly average from each wind meter

INPUT
- wind_measurements is the wind dataset with subhourly measurements
- NOAA_data is the NOAA dataframe
- n_threshold is the minimum number of data points collected for the computation of hourly average to be valid.
  Because our measurements were taken on a minutely basis, the default n_threshold is set to be 30, meaning that 
  we need at least 30 minutely data for computing the hourly average. If data is insufficient, the function will return 
  an NaN for that hour

OUTPUT
- hourly_data concatenates the NOAA_data with the computed hourly wind measurements
  """

  # find the types of wind measurements
  new_col = []
  for col in wind_measurements.columns:
    if col.endswith('mps')  or col.endswith('mph') :
        new_col.append(col)

  # create columns of month, day, and hour for ease of querying
  n = wind_measurements.shape[0]    # number of measurements
  day = np.zeros(n).astype('int')
  month = np.zeros(n).astype('int')
  hour = np.zeros(n).astype('int')
  for i in range(n):
    t = wind_measurements.date[i]
    day[i] = t.day
    month[i] = t.month
    hour[i] = wind_measurements.minute[i].hour
  wind_measurements['day'] = day
  wind_measurements['month'] = month
  wind_measurements['hour'] = hour

  # for each hour of the NOAA wind, find the concurrent measurements and compute hourly avg
  n_hour = NOAA_data.shape[0]
  hourly_avg_measured = pd.DataFrame(columns = new_col)  # initiate the 
  for i in range(n_hour):
    day_NOAA = NOAA_data.date[i].day
    month_NOAA = NOAA_data.date[i].month
    hour_NOAA = NOAA_data.date[i].hour
  
    measurement_of_hour = wind_measurements.loc[wind_measurements.day == day_NOAA].loc[wind_measurements.month==month_NOAA].loc[wind_measurements.hour==hour_NOAA]
    hourly_avg_measured.loc[i] = [np.nan]*len(new_col)   # initiate the column

    for col in new_col:
      measurements = measurement_of_hour[col].dropna()  # leave out invalid ones
      if len(measurements) >= n_threshold:
        hourly_avg = np.mean(measurements)
        hourly_avg_measured.loc[i,col] = hourly_avg

    hourly_data = pd.concat([NOAA_data,hourly_avg_measured],axis=1)

  return hourly_data





# plot the wind parity chart with NOAA data
def wind_parity_with_NOAA_plot(ax,plot_data,
                        ylabel='NOAA KSCK height adjusted hourly average wind [mps]',
                        windy='KSCK_windSpeed_height_adjusted_mps',
                        windx=['WS_windSpeed_mps','sonic_windSpeed_mps'],
                        force_intercept_origin=0,plot_interval=['conficence'],
                        plot_lim=[0,5],legend_loc='lower_right'):
  """
plot parity chart: scatter plot with a parity line, a regression line, and a confidence interval

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed hourly data from compute_hourly_avg_wind
- windx is the list of the name of the wind to be treated as x
- windy is name of the wind to be treated as y
- force_intercept_origin decides which regression to use
- plot_interval can be ['confidence','prediction'] or either one of those in the list
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend

OUTPUT
- ax is the plotted parity chart
  """
  # set up plot
  ax.set_ylabel(ylabel,fontsize=13)
  ax.set_xlabel('measured hourly average wind [mps]',fontsize=13)
  ax.set_xlim(plot_lim)
  ax.set_ylim(plot_lim)

  # parity line
  x_lim = np.array([0,20])
  y_lim = np.array([0,20])
  ax.plot(x_lim,y_lim,color='black',label = 'Parity line')

  # plot for each of the y wind
  colors = ['#8c1515','#9d9573','#00505c']
  l=0
  # plot_data = plot_data.sort_values(by=windx)
  
  for w in windx:
    x = plot_data[w]
    y = plot_data[windy]
    df = pd.DataFrame({'x':x,'y':y}).dropna()   # drop nan
    x = np.array(df.x.values).astype('float')
    y = np.array(df.y.values).astype('float')

    # regression
    if force_intercept_origin == 0:
      n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
    elif force_intercept_origin == 1:
      n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)

    # scatter plots
    if 'WS' in w:
      ax.scatter(x,y,color=colors[l],alpha = 0.4)
      ax.scatter(x_lim[1],y_lim[1],color=colors[l],label='cup wind meter hourly average $n$ = %d' %(n))  # for labeling only
      if force_intercept_origin == 0:
        if intercept>0:
          ax.plot(x,y_pred,'-',color=colors[l],linewidth=2.5,
            label = 'cup wind meter best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f' % (r_value**2,slope,intercept))
        else:
          ax.plot(x,y_pred,'-',color=colors[l],linewidth=2.5,
            label = 'cup wind meter best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$%0.2f' % (r_value**2,slope,intercept))
      elif force_intercept_origin ==1:
        ax.plot(x,y_pred,'-',color=colors[l],linewidth=2.5,
              label = 'cup wind meter best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$' % (r_squared,slope))
    elif 'sonic' in w:
      ax.scatter(x,y,color=colors[l],alpha = 0.4)
      ax.scatter(x_lim[1],y_lim[1],color=colors[l],label='ultrasonic anemometer hourly average $n$ = %d' %(n))  # for labeling only
      if force_intercept_origin == 0:
        if intercept>0:
          ax.plot(x,y_pred,'-',color=colors[l],linewidth=2.5,
            label = 'ultrasonic anemometer best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f' % (r_value**2,slope,intercept))
        else:
          ax.plot(x,y_pred,'-',color=colors[l],linewidth=2.5,
            label = 'ultrasonic anemometer best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$%0.2f' % (r_value**2,slope,intercept))
      elif force_intercept_origin ==1:
        ax.plot(x,y_pred,'-',color=colors[l],linewidth=2.5,
              label = 'ultrasonic anemometer best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$' % (r_squared,slope))

    # plot intervals
    if 'confidence' in plot_interval:
      ax.plot(np.sort(x),upper_CI,':',color='black', label='95% CI')
      ax.plot(np.sort(x),lower_CI,':',color='black')
      ax.fill_between(np.sort(x), lower_CI, upper_CI, color='black', alpha=0.05)
    if 'prediction' in plot_interval:
      ax.plot(np.sort(x),upper_PI,':',color=colors[l], label='95% PI')
      ax.plot(np.sort(x),lower_PI,':',color=colors[l])
      ax.fill_between(np.sort(x), lower_PI, upper_PI, color='black', alpha=0.05)
    
    l+=1  # use the next color

  ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.2),fontsize=12)
  plt.close()
  return ax





# parity plot with darksky wind
def wind_parity_with_darksky_plot(ax, plot_data, 
                windx='WS_windGust_mps',windy='darksky_windGust_height_adjusted_mps',
                xlabel='cup wind meter 1-min gust [mps]',
                ylabel='Dark Sky height-adjusted 1-min gust [mps]',
                legend_loc='lower right', plot_lim = [0,10],
                force_intercept_origin=0, 
                plot_interval=['confidence']):

  """
plot parity chart: scatter plot with a parity line, a regression line, and a confidence interval

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed wind data
- windx and windy are the wind to be plotted on the x and y axis
- xlabel and ylabel are the axis titles 
- force_intercept_origin decides which regression to use
- plot_interval can be ['confidence','prediction'] or either one of those in the list
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend

OUTPUT
- ax is the plotted parity chart
  """

  # set up plot
  ax.set_xlabel(xlabel,fontsize=13)
  ax.set_ylabel(ylabel,fontsize=13)
  ax.set_xlim(plot_lim)
  ax.set_ylim(plot_lim)

  # parity line
  x_lim = np.array([0,3000])
  y_lim = np.array([0,3000])
  ax.plot(x_lim,y_lim,color='black',label = 'Parity line')

  # define x and y data points
  x = plot_data[windx]
  y = plot_data[windy]
  # drop nan
  df = pd.DataFrame({'date':plot_data.date,'x':x,'y':y})
  df = df.dropna().reset_index()
  x = df.x.astype('float')   
  y = df.y.astype('float')
  if windx.startswith('WS'):
    df['xerr'] = [2 * 0.44704]*df.shape[0]     # cup wind meter error is +/- 2 mph
  elif windx.startswith('sonic'):
    df['xerr'] = np.zeros(df.shape[0])       # ultrasonic anemometer error varies with the wind speed measured
    for i in range(df.shape[0]):
      if x[i]<=2:
        df.xerr[i] = x[i] * 0.0298
      elif x[i]<=5:
        df.xerr[i] = x[i] * 0.0199
      elif x[i]<=10:
        df.xerr[i] = x[i] * 0.0126
      elif x[i]<=20:
        df.xerr[i] = x[i] * 0.0136
      elif x[i]<=30:
        df.xerr[i] = x[i] * 0.0051
      elif x[i]<=40:
        df.xerr[i] = x[i] * 0.0048
      elif x[i]<=50:
        df.xerr[i] = x[i] * 0.0077
      else:
        df.xerr[i] = x[i] * 0.0259 

  # regression
  if force_intercept_origin == 0:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
  elif force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)

  # scatter plots
  days = ['10-08','10-10','10-11','10-15']   # four dates of the field trial
  colors = {days[0]:'#53284f',days[1]:'#9d9573',days[2]:'#007c92',days[3]:'#b26f16'}
  for d in days:
    xd = x[df.date=='2019-'+d]    # to match with the format in the data, add the year to the front
    yd = y[df.date=='2019-'+d]
    xerrd = df[df.date=='2019-'+d].xerr
    ax.scatter(xd,yd,color=colors[d],alpha = 0.1)
    # ax.errorbar(xd, yd, xerr=xerrd,
            #  fmt='o',color=colors[d],ecolor=colors[d], elinewidth=1.5, capsize=0,alpha=0.2);
    ax.scatter(x_lim[-1],y_lim[-1],color=colors[d],alpha=0.9,
                label=d[:2]+'/'+d[3:5]+' $n$ = %d' %(len(xd)))   # for the sake of turning off transparency in the legend

  # plot regression line
  if force_intercept_origin == 0:
    ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
        label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f' % (r_value**2,slope,intercept))
  elif force_intercept_origin ==1:
    ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
            label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$' % (r_squared,slope))

  # plot intervals
  if 'confidence' in plot_interval:
    ax.plot(np.sort(x),upper_CI,':',color='black', label='95% CI')
    ax.plot(np.sort(x),lower_CI,':',color='black')
    ax.fill_between(np.sort(x), lower_CI, upper_CI, color='black', alpha=0.05)
  if 'prediction' in plot_interval:
    ax.plot(np.sort(x),upper_PI,':',color='#8c1515', label='95% PI')
    ax.plot(np.sort(x),lower_PI,':',color='#8c1515')
    ax.fill_between(np.sort(x), lower_PI, upper_PI, color='black', alpha=0.05)

  # ax.legend(loc='upper center', bbox_to_anchor=(1.6, 1.02),fontsize=10)
  ax.legend(loc=legend_loc)
  plt.close()
  return ax







# plot the wind parity chart
def wind_parity_plot(ax, plot_data,
              windx = 'sonic_windGust_mps',windy = 'WS_windGust_mps',
              force_intercept_origin=0, plot_interval=['confidence'],
              plot_lim = [0,10], legend_loc='lower right'):

  """
plot parity chart: scatter plot with a parity line, a regression line, and a confidence interval

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed data
- windx is the name of the wind to be treated as x
- windy is the name of the wind to be treated as y
- force_intercept_origin decides which regression to use
- plot_interval can be ['confidence','prediction'] or either one of those in the list
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend

OUTPUT
- ax is the plotted parity chart
  """

  # set up plot
  if 'sonic' in windx:
    ax.set_xlabel('ultrasonic anemometer [mps]',fontsize=13)
    ax.set_ylabel('cup wind meter [mps]',fontsize=13)
  elif 'WS' in windx:
    ax.set_xlabel('cup wind meter [mps]',fontsize=13)
    ax.set_ylabel('ultrasonic anemometer [mps]',fontsize=13)
  ax.set_xlim(plot_lim)
  ax.set_ylim(plot_lim)

  # parity line
  x_lim = np.array([0,20])
  y_lim = np.array([0,20])
  ax.plot(x_lim,y_lim,color='black',label = 'Parity line')

  # define x and y data points
  plot_data = plot_data.dropna().sort_values(by=windx)
  x = plot_data[windx]
  y = plot_data[windy]

  # regression
  if force_intercept_origin == 0:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
  elif force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)

  # scatter plots
  ax.scatter(x,y,color='#8c1515',alpha = 0.2,label='$n$ = %d' %(n))

  # plot regression line
  if force_intercept_origin == 0:
    ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
        label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$%0.2f' % (r_value**2,slope,intercept))
  elif force_intercept_origin ==1:
    ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
            label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$' % (r_squared,slope))

  # plot intervals
  if 'confidence' in plot_interval:
    ax.plot(np.sort(x),upper_CI,':',color='black', label='95% CI')
    ax.plot(np.sort(x),lower_CI,':',color='black')
    ax.fill_between(np.sort(x), lower_CI, upper_CI, color='black', alpha=0.05)
  if 'prediction' in plot_interval:
    ax.plot(np.sort(x),upper_PI,':',color='#8c1515', label='95% PI')
    ax.plot(np.sort(x),lower_PI,':',color='#8c1515')
    ax.fill_between(np.sort(x), lower_PI, upper_PI, color='black', alpha=0.05)

  ax.legend(loc=legend_loc,fontsize=12)
  plt.close()

  return ax






# daily variability rose plot
def daily_wind_scatter(data,fig_notation,date,wind_speed_to_use,wind_dir_to_use):

  """
dailiy_wind scatter: rose plot of the wind of each day

INPUTS
- data is the processed wind_data in the ipynb
- fig_notation is the index of plot to be added e.g. 'a'
- date choose from ['10/08/2019','10/10/2019','10/11/2019','10/15/2019']
- wind_speed_to_use ['sonic_windSpeed_mps','sonic_windGust_mps','WS_windSpeed_mps','WS_windGust_mps']
- wind_dir_to_use ['sonic_direction','WS_direction']

OUTPUT
- ax is the plotted rose chart
  """

  plot_data = data[[wind_speed_to_use,wind_dir_to_use]].loc[data.date==date]
  max_wind = 9  # for scaling plot only

  ax = plt.subplot(111, projection='polar')
  ax.set_facecolor('whitesmoke')
  plt.xticks(np.radians(range(0, 360, 45)), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

  ax.set_rgrids(np.arange(0,max_wind,3))
  ax.set_rlim([0,max_wind])
  
  if wind_dir_to_use == 'sonic_direction':
      ax.scatter(np.radians(plot_data.sonic_direction),plot_data[wind_speed_to_use],alpha=0.5,color='#8c1515',s=1.5,
                  label='('+fig_notation+') '+date+' ultrasonic')
  elif wind_dir_to_use == 'WS_direction':
      ax.scatter(plot_data.WS_direction,plot_data[wind_speed_to_use],alpha=0.5,color='#8c1515',s=1.5,
                  label='('+fig_notation+') '+date+' cup wind meter')
  ax.set_theta_zero_location("N")
  ax.set_theta_direction(-1)

  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),fontsize=12)

  return ax






# trial variabiltiy 4-min time series plot
def wind_speed_variability_plot(release_data,wind_data,sonic_data,trial,fig1_notation):

  """
dailiy_wind scatter: wind variability of the trial

INPUTS
- release_data is the processed release data
- wind_data is the processed wind data
- trial the trial number
- fig1_notation is the index of the graph

OUTPUT
- fig is the plotted wind time series and histogram
"""

  tt = int(release_data.unix_time[trial])

  fig,[ax1,ax2] = plt.subplots(1,2,figsize = [15,5],gridspec_kw={'width_ratios': [2, 1]})

  plot_sonic_minutely = wind_data[['unix_time','sonic_windGust_mps']].loc[wind_data.unix_time<=tt].loc[wind_data.unix_time>tt-60*4]     # four minute intervals
  plot_WS_minutely = wind_data[['unix_time','WS_windGust_mps']].loc[wind_data.unix_time<=tt].loc[wind_data.unix_time>tt-60*4]
  plot_sonic_secondly = sonic_data[['unix_time','sonic_wind_mps','sonic_direction']].loc[sonic_data.unix_time<=tt+60].loc[sonic_data.unix_time>tt-60*3]    

  # rename the columns
  plot_sonic_minutely = pd.DataFrame({'unix':plot_sonic_minutely.unix_time+30,    # make the timestamp of the average time to be at the 30th second
                                      'sonic_windGust': plot_sonic_minutely.sonic_windGust_mps})
  plot_WS_minutely = pd.DataFrame({'unix':plot_WS_minutely.unix_time+30,
                                      'WS_windGust': plot_WS_minutely.WS_windGust_mps})
  plot_sonic_secondly = pd.DataFrame({'unix':plot_sonic_secondly.unix_time,
                                      'sonic_wind': plot_sonic_secondly.sonic_wind_mps})                                        


  mins = mdate.epoch2num(plot_sonic_minutely.unix.values)    # convert to UTC
  ax1.plot_date(mins, plot_WS_minutely.WS_windGust,label = 'Cup wind meter 1-min gust',color='dimgrey')
  ax1.plot_date(mins, plot_sonic_minutely.sonic_windGust,label = 'Ultrasonic 1-min gust',color = '#8c1515')
  date_fmt = '%I:%M:%S %p'
  date_formatter = mdate.DateFormatter(date_fmt)
  ax1.xaxis.set_major_formatter(date_formatter)

  secs = mdate.epoch2num(plot_sonic_secondly.unix.values)    # convert to UTC time
  ax1.plot_date(secs, plot_sonic_secondly.sonic_wind,'-',label='Ultrasonic secondly measurement',color='#8c1515',alpha=0.3)

  mu, std = norm.fit(plot_sonic_secondly.sonic_wind.values)
  ax2.hist(plot_sonic_secondly.sonic_wind.values, bins=20, density=True, color='#8c1515',alpha=0.3)
  xmin, xmax = plot_sonic_secondly.sonic_wind.min(),plot_sonic_secondly.sonic_wind.max()
  x = np.linspace(xmin, xmax, 100)
  p = norm.pdf(x, mu, std)
  ax2.plot(x, p, 'k', linewidth=2)

  ticks = mdate.epoch2num(np.append(plot_sonic_minutely.unix.values-30,plot_sonic_minutely.unix.values[-1]+30))
  ax1.xaxis.set_ticks(np.array(ticks)), ax1.yaxis.set_ticks(np.arange(0,15,2))

  ax1.set_ylabel('wind speed [mps]',fontsize=13)
  ax2.set_xlabel('wind speed [mps]',fontsize=13)
  ax2.set_ylabel('density',fontsize=13)
  max_wind = 6   # for plotting boundary
  ax1.set_ylim([0,max_wind]),ax2.set_xlim([0,max_wind]),ax2.set_ylim([0,1.5])
  
  trial_date = str(release_data.date[trial])[:11]
  trial_minute = str(release_data.minute[trial])[:5]
  trial_pass = release_data['pass'].iloc[trial]
  known_release = np.round(release_data.iloc[trial].CH4_release_kgh)
  quantified_release_wind_normalized = release_data.iloc[trial].closest_plume_quantification_kghmps
  quantified_release = release_data.iloc[trial].closest_plume_quantification_kghmps * release_data.iloc[trial].WS_windGust_logged_mps
  
  ax1.annotate('('+fig1_notation+') '+ trial_date +' Pass '+ str(trial_pass)
                +' \n     known release = %d kgh \n     reported release = %d kgh' %(known_release,quantified_release),
                xy = [0.02,0.82],xycoords='axes fraction',fontsize=13)
  ax2.annotate('Ultrasonic secondly measurement\nFit results: $\\mu$ = %.2f,  $\\sigma$ = %.2f' % (mu, std),
                xy = [0.03,0.87],xycoords='axes fraction',fontsize=13)
  
  WS_reading = release_data.WS_windGust_logged_mps[trial]
  sonic_reading = plot_sonic_minutely.sonic_windGust.values[-1]
  
  if WS_reading<3:
    if WS_reading<sonic_reading:
        ax1.annotate('cup meter reading\n        %.2f mps' %WS_reading,
                      xy = [0.72,0.49],xycoords='axes fraction',fontsize=13,color='dimgrey')
        ax1.annotate('ultrasonic reading\n       %.2f mps' %sonic_reading,
                      xy = [0.72,0.62],xycoords='axes fraction',fontsize=13,color='#8c1515')
    else:
        ax1.annotate('cup meter reading\n        %.2f mps' %WS_reading,
                      xy = [0.72,0.62],xycoords='axes fraction',fontsize=13,color='dimgrey')
        ax1.annotate('ultrasonic reading\n       %.2f mps' %sonic_reading,
                      xy = [0.72,0.49],xycoords='axes fraction',fontsize=13,color='#8c1515')
  else:
    if WS_reading<sonic_reading:
      ax1.annotate('cup meter reading\n        %.2f mps' %WS_reading,
                    xy = [0.72,0.07],xycoords='axes fraction',fontsize=13,color='dimgrey')
      ax1.annotate('ultrasonic reading\n       %.2f mps' %sonic_reading,
                    xy = [0.72,0.2],xycoords='axes fraction',fontsize=13,color='#8c1515')
    else:
      ax1.annotate('cup meter reading\n        %.2f mps' %WS_reading,
                    xy = [0.72,0.2],xycoords='axes fraction',fontsize=13,color='dimgrey')
      ax1.annotate('ultrasonic reading\n       %.2f mps' %sonic_reading,
                    xy = [0.72,0.07],xycoords='axes fraction',fontsize=13,color='#8c1515')

  if sonic_reading<6:
      ax1.legend(loc='upper right',fontsize=13)
  else:
      ax1.legend(loc='lower right',fontsize=13)
  plt.close()

  return fig





# trial variabiltiy 4-min rose plot
def wind_direction_variability_plot(release_data,sonic_data,trial):

  tt = release_data.unix_time[trial]
  plot_sonic_secondly = sonic_data[['unix_time','sonic_wind_mps','sonic_direction']].loc[sonic_data.unix_time<=tt+30].loc[sonic_data.unix_time>tt-60*4+30]
  plot_sonic_secondly['sonic_wind'] = plot_sonic_secondly.sonic_wind_mps
  max_wind = 6  # for scaling plot only

  ax = plt.subplot(111, projection='polar')
  ax.set_facecolor('whitesmoke')
  plt.xticks(np.radians(range(0, 360, 45)), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

  ax.set_rgrids(np.arange(0,max_wind,2))
  ax.set_rlim([0,max_wind])
  
  ax.scatter(np.radians(plot_sonic_secondly.sonic_direction),plot_sonic_secondly.sonic_wind,alpha=0.5,color='grey',s=1.5)

  sonic_minutely_windGust = release_data.sonic_windGust_mps[trial]
  WS_minutely_windGust = release_data.WS_windGust_logged_mps[trial]    
  sonic_winddir = np.radians(release_data.sonic_direction[trial])
  WS_winddir = convert_16_wind_compass_to_radian(release_data.WS_direction[trial])
  
  if release_data.sonic_direction[trial]<0:
      sonicdir = int(release_data.sonic_direction[trial]+360)
  else:
      sonicdir = int(release_data.sonic_direction[trial])    
  
  ax.plot([WS_winddir,WS_winddir],[0,WS_minutely_windGust],color='k',
          label='cup meter: %.2f mps gust from '%release_data.WS_windGust_logged_mps[trial]+str(release_data.WS_direction[trial]) )    
  ax.plot([sonic_winddir,sonic_winddir],[0,sonic_minutely_windGust],color='#8c1515',
          label = 'ultrasonic: %.2f mps gust from %d$^{\circ}$' %(release_data.sonic_windGust_mps[trial],sonicdir))
  ax.set_theta_zero_location("N")
  ax.set_theta_direction(-1)

  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),fontsize=12)
  # plt.close()

  return ax







