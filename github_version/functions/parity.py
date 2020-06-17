import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
from datetime import date

def linreg_results(x,y):

    """
Regression to output all values to be potentially used in plotting:
n = number of points in scatter plot;
pearson_corr = peason's correlation coefficient;
slope, intercept = regression parameters;
r_value = R (not R_squared);
x_lim = (min&max) of x;
y_pred = (min&max) of y_predction computed with slope, intercept, and x_lim;
lower_CI, upper_CI are bounds of 95% confidence interval for the fit line;
lower_PI, upper_CI are bounds of 95% prediction interval for predictions;
see reference: http://www2.stat.duke.edu/~tjl13/s101/slides/unit6lec3H.pdf
    """

    n = len(x)
    pearson_corr, _ = stats.pearsonr(x, y)    # Pearson's correlation coefficient
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    x_lim = np.array([0,max(x)])
    y_pred = intercept + slope*x
    residual = y - (intercept+slope*x)
    dof = n - 2                               # degree of freedom
    t_score = stats.t.ppf(1-0.025, df=dof)    # one-sided t-test
    
    # sort x from smallest to largest for ease of plotting
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df = df.sort_values('x')
    x = df.x.values
    y = df.y.values
    
    y_hat = intercept + slope*x             
    x_mean = np.mean(x)
    S_yy = np.sum(np.power(y-y_hat,2))      # total sum of error in y
    S_xx = np.sum(np.power(x-x_mean,2))     # total sum of variation in x

    # find lower and upper bounds of CI and PI
    lower_CI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    upper_CI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    lower_PI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))
    upper_PI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))
    
    return n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err



def linreg_results_no_intercept(x,y):

    """
Regression to output all values to be potentially used in plotting:
n = number of points in scatter plot;
pearson_corr = peason's correlation coefficient;
slope = regression parameters;
r_value = R (not R_squared);
x_lim = (min&max) of x;
y_pred = (min&max) of y_predction computed with slope, intercept, and x_lim;
lower_CI, upper_CI are bounds of 95% confidence interval for the fit line;
lower_PI, upper_CI are bounds of 95% prediction interval for predictions;
see reference: http://www2.stat.duke.edu/~tjl13/s101/slides/unit6lec3H.pdf
    """

    n = len(x)

    model = sm.OLS(y,x)
    result = model.fit()
    slope = result.params[0]
    r_squared = result.rsquared
    std_err = result.bse[0]

    x_lim = np.array([0,max(x)])
    y_pred = slope*x
    residual = y - y_pred
    dof = n - 1                               # degree of freedom
    t_score = stats.t.ppf(1-0.025, df=dof)    # one-sided t-test
    
    # sort x from smallest to largest for ease of plotting
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df = df.sort_values('x')
    x = df.x.values
    y = df.y.values
    
    y_hat = slope*x             
    x_mean = np.mean(x)
    S_yy = np.sum(np.power(y-y_hat,2))      # total sum of error in y
    S_xx = np.sum(np.power(x-x_mean,2))     # total sum of variation in x

    # find lower and upper bounds of CI and PI
    lower_CI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    upper_CI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    lower_PI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))
    upper_PI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))
    
    return n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err



# plot the parity chart
def parity_plot(ax, plot_data, force_intercept_origin=0, plot_interval=['confidence'],
                plot_lim = [0,2000], legend_loc='lower right'):

  """
plot parity chart: scatter plot with a parity line, a regression line, and a confidence interval

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed data
- force_intercept_origin decides which regression to use
- plot_interval can be ['confidence','prediction'] or either one of those in the list
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend

OUTPUT
- ax is the plotted parity chart
  """

  # set up plot
  ax.set_xlabel('Methane release rate [kgh]',fontsize=13)
  ax.set_ylabel('Reported release rate [kgh]',fontsize=13)
  ax.set_xlim(plot_lim)
  ax.set_ylim(plot_lim)

  # parity line
  x_lim = np.array([0,3000])
  y_lim = np.array([0,3000])
  ax.plot(x_lim,y_lim,color='black',label = 'Parity line')

  # define x and y data points
  x = plot_data.CH4_release_kgh.values
  y = plot_data.closest_plume_quantification_kgh.fillna(0).values

  # regression
  if force_intercept_origin == 0:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
  elif force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)

  # scatter plots
  ax.scatter(x,y,color='#8c1515',alpha = 0.2,label='$n$ = %d' %(n))
  ax.errorbar(x, y, xerr=plot_data.CH4_release_meter_error_kgh,
             yerr=[plot_data.quantification_lower_error_kgh,plot_data.quantification_upper_error_kgh],
             fmt='o',color='#8c1515',ecolor='#8c1515', elinewidth=1.5, capsize=0,alpha=0.2);
  
  # plot regression line
  if force_intercept_origin == 0:
    if intercept<0:   # label differently depending on intercept
      ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
              label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$%0.2f' % (r_value**2,slope,intercept))
    elif intercept>=0:
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

  # ax.legend(loc=legend_loc, bbox_to_anchor=(1.6, 0.62),fontsize=12)   # legend box on the right
  ax.legend(loc=legend_loc,fontsize=12)   # legend box within the plot
  plt.close()

  return ax


# plot the parity chart with different days colored differently
def parity_plot_days_colored(ax, plot_data, force_intercept_origin=0, plot_interval=['confidence'],
                plot_lim = [0,2000], legend_loc='lower right'):

  """
plot parity chart: scatter plot with a parity line, a regression line, and a confidence interval

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed data
- force_intercept_origin decides which regression to use
- plot_interval can be ['confidence','prediction'] or either one of those in the list
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend

OUTPUT
- ax is the plotted parity chart
  """

  # set up plot
  ax.set_xlabel('Methane release rate [kgh]',fontsize=13)
  ax.set_ylabel('Reported release rate [kgh]',fontsize=13)
  ax.set_xlim(plot_lim)
  ax.set_ylim(plot_lim)

  # parity line
  x_lim = np.array([0,3000])
  y_lim = np.array([0,3000])
  ax.plot(x_lim,y_lim,color='black',label = 'Parity line')

  # define x and y data points
  x = plot_data.CH4_release_kgh.values
  y = plot_data.closest_plume_quantification_kgh.fillna(0).values

  # regression
  if force_intercept_origin == 0:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
  elif force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)

  # scatter plots
  days = ['10-08','10-10','10-11','10-15']   # four dates of the field trial
  colors = {days[0]:'#53284f',days[1]:'#9d9573',days[2]:'#007c92',days[3]:'#b26f16'}
  for d in days:
    xd = x[plot_data.date=='2019-'+d]    # to match with the format in the data, add the year to the front
    yd = y[plot_data.date=='2019-'+d]
    xerrd = plot_data[plot_data.date=='2019-'+d].CH4_release_meter_error_kgh
    yerrd = [plot_data[plot_data.date=='2019-'+d].quantification_lower_error_kgh,
            plot_data[plot_data.date=='2019-'+d].quantification_upper_error_kgh]
    ax.scatter(xd,yd,color=colors[d],alpha = 0.2)
    ax.errorbar(xd, yd, xerr=xerrd, yerr= yerrd,
             fmt='o',color=colors[d],ecolor=colors[d], elinewidth=1.5, capsize=0,alpha=0.2);
    ax.scatter(x_lim[-1],y_lim[-1],color=colors[d],alpha=0.9,
                label=d[:2]+'/'+d[3:5]+' $n$ = %d' %(len(xd)))   # for the sake of turning off transparency in the legend

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



# plot the color of data points of different gas temperature differently
def parity_plot_sensitivity_gas_temperature(ax, plot_data, 
                gas_temp = ['cold','mixed','warm'],
                force_intercept_origin=0,
                plot_interval=['confidence'], plot_lim = [0,2000], legend_loc='lower right'):
  """
plot parity chart: scatter plot with a parity line, a regression line, and a confidence interval

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed data
- force_intercept_origin decides which regression to use
- plot_interval can be ['confidence','prediction'] or either one of those in the list
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend
- gas_temp defines which sets of data points of different gas temperature to include in the plot

OUTPUT
- ax is the plotted parity chart
  """

  # set up plot
  ax.set_xlabel('Methane release rate [kgh]',fontsize=13)
  ax.set_ylabel('Reported release rate [k/h]',fontsize=13)
  ax.set_xlim(plot_lim)
  ax.set_ylim(plot_lim)

  # parity line
  x_lim = np.array([0,3000])
  y_lim = np.array([0,3000])
  ax.plot(x_lim,y_lim,color='black',label = 'Parity line')

  # plot cold gas blue
  if 'cold' in gas_temp:
    ax.scatter(plot_data[plot_data.cold_R12==1].CH4_release_kgh,
            plot_data[plot_data.cold_R12==1].closest_plume_quantification_kgh.fillna(0),
            color='blue',alpha=0.2, label='cold gas  $n$ = %d' %plot_data[plot_data.cold_R12==1].shape[0])
  if 'mixed' in gas_temp:
    ax.scatter(plot_data[plot_data.cold_R12==0.5].CH4_release_kgh,
            plot_data[plot_data.cold_R12==0.5].closest_plume_quantification_kgh.fillna(0),
            color='black',alpha = 0.2,label='mixed gas $n$ = %d' %plot_data[plot_data.cold_R12==0.5].shape[0])
  if 'warm' in gas_temp:
    ax.scatter(plot_data[plot_data.cold_R12.fillna(0)==0].CH4_release_kgh,
            plot_data[plot_data.cold_R12.fillna(0)==0].closest_plume_quantification_kgh.fillna(0),
            color='#8c1515',alpha = 0.2,label='warm gas $n$ = %d' %plot_data[plot_data.cold_R12.fillna(0)==0].shape[0])

  
  # define x and y data points for regression
  if 'cold' not in gas_temp:
    plot_data = plot_data[plot_data.cold_R12!=1]
  if 'mixed' not in gas_temp:
    plot_data = plot_data[plot_data.cold_R12!=0.5]
  if 'warm' not in gas_temp:
    plot_data = plot_data[~np.isnan(plot_data.cold_R12)]
  x = plot_data.CH4_release_kgh.values
  y = plot_data.closest_plume_quantification_kgh.fillna(0).values
  print(len(y))

  # regression
  if force_intercept_origin == 0:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
  elif force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)

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



# plot the color of data points of different wind speed differently
def parity_plot_sensitivity_wind_speed(ax, plot_data, wind_to_use,
                lower_bound_wind = 0, 
                upper_bound_wind = np.inf,
                max_wind = 7, 
                cmap = plt.cm.get_cmap('plasma'),
                force_intercept_origin = 0,
                plot_interval=['confidence'], plot_lim = [0,2000], legend_loc='lower right'):
  """
plot parity chart: scatter plot with a parity line, a regression line, and a confidence interval

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed data
- wind_to_use is the wind used for finding closest plume quantification in plot_data
- force_intercept_origin decides which regression to use
- plot_interval can be ['confidence','prediction'] or either one of those in the list
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend
- lower and upper bound of wind are criteria for selecting data points that have wind speeds
  measured by the wind_to_use that fall in the range of [lower_bound_wind, upper_bound_wind) unit is m/s
- max_wind is the max of the color bar in m/s. This number is used to normalize the colors chosen for each data point
- cmap is the color map for scatter plot

OUTPUT
- ax is the plotted parity chart
  """

  # set up plot
  ax.set_xlabel('Methane release rate [kgh]',fontsize=13)
  ax.set_ylabel('Reported release rate [kgh]',fontsize=13)
  ax.set_xlim(plot_lim)
  ax.set_ylim(plot_lim)

  # parity line
  x_lim = np.array([0,3000])
  y_lim = np.array([0,3000])
  ax.plot(x_lim,y_lim,color='black',label = 'Parity line')

  # define x and y data points for regression
  plot_data = plot_data.loc[plot_data[wind_to_use]>=lower_bound_wind].loc[plot_data[wind_to_use]<upper_bound_wind]
  x = plot_data.CH4_release_kgh.values
  y = plot_data.closest_plume_quantification_kgh.fillna(0).values

  # regression
  if force_intercept_origin == 0:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
  elif force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)

  # plot regression line
  if force_intercept_origin == 0:
    ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
        label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$%0.2f' % (r_value**2,slope,intercept))
  elif force_intercept_origin ==1:
    ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
            label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$' % (r_squared,slope))

  # scatter plot
  data_color = plot_data[wind_to_use].values/max_wind   # normalize for data color
  colors = cmap(data_color)
  ax.scatter(x,y,c=colors,alpha = 0.2,label='$n$ = %d' %(n))
  ax.errorbar(x, y, xerr=plot_data.CH4_release_meter_error_kgh,
             yerr=[plot_data.quantification_lower_error_kgh,plot_data.quantification_upper_error_kgh],
             fmt='o',color='#8c1515',ecolor=colors, elinewidth=1.5, capsize=0,alpha=0.2);

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






# quantile regression plot
def quantile_regression_plot(ax, plot_data, quantiles=[0.05,0.25,0.5,0.75,0.95],
                force_intercept_origin=0,
                plot_lim = [0,2000], legend_loc='lower right'):

  """
plot parity chart: scatter plot with a parity line, a regression line, quantile regression lines

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed data
- quantiles is a list of quantiles lines to graph
- force_intercept_origin decides which regression to use
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend

OUTPUT
- ax is the plotted parity chart
  """

  # set up plot
  ax.set_xlabel('Methane release rate [kgh]',fontsize=13)
  ax.set_ylabel('Reported release rate [kgh]',fontsize=13)
  ax.set_xlim(plot_lim)
  ax.set_ylim(plot_lim)

  # parity line
  x_lim = np.array([0,3000])
  y_lim = np.array([0,3000])
  ax.plot(x_lim,y_lim,color='black',label = 'Parity line')

  # define x and y data points
  x = plot_data.CH4_release_kgh.values
  y = plot_data.closest_plume_quantification_kgh.fillna(0).values

  # OLS regression
  if force_intercept_origin == 0:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
  elif force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)

  # scatter plots
  ax.scatter(x,y,color='#8c1515',alpha = 0.2,label='$n$ = %d' %(n))
  ax.errorbar(x, y, xerr=plot_data.CH4_release_meter_error_kgh,
             yerr=[plot_data.quantification_lower_error_kgh,plot_data.quantification_upper_error_kgh],
             fmt='o',color='#8c1515',ecolor='#8c1515', elinewidth=1.5, capsize=0,alpha=0.2);
  
  # plot regression line
  if force_intercept_origin == 0:
    ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
        label = 'OLS Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$%0.2f' % (r_value**2,slope,intercept))
  elif force_intercept_origin ==1:
    ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
            label = 'OLS Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$' % (r_squared,slope))

  # quantile regression
  quantile_data = pd.DataFrame({'x':x,'y':y})
  mod = smf.quantreg('y ~ x', quantile_data)
  def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['x']] + \
            res.conf_int().loc['x'].tolist()
  quantile_models = [fit_model(x) for x in quantiles]
  quantile_models = pd.DataFrame(quantile_models, columns=['q', 'a', 'b', 'lb', 'ub'])

  # plot quantile lines
  x = np.arange(min(x), max(x), 50)
  get_y = lambda a, b: a + b * x
  for i in range(quantile_models.shape[0]):
    y = get_y(quantile_models.a[i], quantile_models.b[i])
    ax.plot(x, y, linestyle='dotted', color='grey')
    ax.annotate('q=%.2f'%quantiles[i],
              xy=[max(x)+20,quantile_models.a[i]+max(x)*quantile_models.b[i]-20],
              fontsize=13)

  ax.plot(x, y, linestyle='dotted', color='grey',label='quantile regressions')
  ax.legend(loc=legend_loc,fontsize=12)
  plt.close()

  return ax



# weighted least square WLS plot
def WLS_plot(ax, plot_data, w, power,
                force_intercept_origin=0,
                plot_interval = ['WLS prediction'],
                plot_lim = [0,2000], legend_loc='lower right'):

  """
plot parity chart: scatter plot with a parity line, a regression line, quantile regression lines

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed data
- w is the weight assigned to each data point
- force_intercept_origin decides which regression to use
- plot_interval is whether or not to include a confidence interval or prediction interval or both in the graph
    unlike the previous plots, this OLS plot can include ['OLS confidence', 'OLS prediction', 'WLS prediction']
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend

OUTPUT
- ax is the plotted parity chart
  """

  # set up plot
  ax.set_xlabel('Methane release rate [kgh]',fontsize=13)
  ax.set_ylabel('Reported release rate [kgh]',fontsize=13)
  ax.set_xlim(plot_lim)
  ax.set_ylim(plot_lim)

  # parity line
  x_lim = np.array([0,3000])
  y_lim = np.array([0,3000])
  ax.plot(x_lim,y_lim,color='black',label = 'Parity line')

  # define x and y data points
  # exclude the data points where Kairos did not detect a plume; so that we won't incur an error bar lenght of zero and thus a weight of zero
  plot_data = plot_data[~np.isnan(plot_data.closest_plume_quantification_kgh)]
  plot_data = plot_data[plot_data.closest_plume_quantification_kgh != 0]
  plot_data = plot_data.sort_values(by=['CH4_release_kgh'])
  x = plot_data.CH4_release_kgh.values
  y = plot_data.closest_plume_quantification_kgh.fillna(0).values

  # OLS regression
  if force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)
    t=x
  else:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
    t = sm.add_constant(x, prepend=False)   # add an intercept to the WLS model
 
  # WLS regression
  weights = np.power(plot_data[w],power)
  mod_wls = sm.WLS(y, t, weights=weights)
  res_wls = mod_wls.fit()
  slope_wls = res_wls.params[0]
  if force_intercept_origin == 0:
    intercept_wls = res_wls.params[1]

  # scatter plots
  ax.scatter(x,y,color='#8c1515',alpha = 0.2,label='$n$ = %d' %(n))
  ax.errorbar(x, y, xerr=plot_data.CH4_release_meter_error_kgh,
             yerr=[plot_data.quantification_lower_error_kgh,plot_data.quantification_upper_error_kgh],
             fmt='o',color='#8c1515',ecolor='#8c1515', elinewidth=1.5, capsize=0,alpha=0.2);
  
  # plot WLS regression line
  if force_intercept_origin == 0:
    ax.plot(x, res_wls.fittedvalues,'-',color='#8c1515',linewidth=2.5,
        label = 'WLS Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$%0.2f' % (res_wls.rsquared,slope_wls,intercept_wls))
  elif force_intercept_origin ==1:
    ax.plot(x, res_wls.fittedvalues,'-',color='#8c1515',linewidth=2.5,
            label = 'OLS Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$' % (res_wls.rsquared,slope_wls))

  # plot confidence and prediction intervals
  if 'OLS confidence' in plot_interval:
    ax.plot(x,upper_CI,':',color='black', label='OLS 95% CI')
    ax.plot(x,lower_CI,':',color='black')
    ax.fill_between(np.sort(x), lower_CI, upper_CI, color='black', alpha=0.05)
  if 'OLS prediction' in plot_interval:
    ax.plot(np.sort(x),upper_PI,':',color='black', label='OLS 95% PI')
    ax.plot(np.sort(x),lower_PI,':',color='black')
    ax.fill_between(np.sort(x), lower_PI, upper_PI, color='black', alpha=0.05)  
  if 'WLS prediction' in plot_interval:
    prstd, iv_l, iv_u = wls_prediction_std(res_wls)
    ax.plot(x, iv_u, '--',color='#8c1515', label="WLS 95% PI")
    ax.plot(x, iv_l, '--',color='#8c1515')
    ax.fill_between(np.sort(x), iv_l, iv_u, color='black', alpha=0.05)

  ax.legend(loc=legend_loc,fontsize=12)  
  plt.close()

  return ax
