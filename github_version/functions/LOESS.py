import numpy as np
import pandas as pd
import scipy
from github_version.functions.parity import *
from scipy.stats import norm

def loc_eval(x, b):
    """
    Evaluate `x` using locally-weighted regression parameters.
    Degree of polynomial used in loess is inferred from b. `x`
    is assumed to be a scalar.
    """
    loc_est = 0
    for i in enumerate(b): loc_est+=i[1]*(x**i[0])
    return(loc_est)


def loess(xvals, yvals, alpha, poly_degree=1):
    """
    Perform locally-weighted regression on xvals & yvals.
    Variables used inside `loess` function:

        n         => number of data points in xvals
        m         => nbr of LOESS evaluation points
        q         => number of data points used for each
                     locally-weighted regression
        v         => x-value locations for evaluating LOESS
        locsDF    => contains local regression details for each
                     location v
        evalDF    => contains actual LOESS output for each v
        X         => n-by-(poly_degree+1) design matrix
        W         => n-by-n diagonal weight matrix for each
                     local regression
        y         => yvals
        b         => local regression coefficient estimates.
                     b = `(X^T*W*X)^-1*X^T*W*y`. Note that `@`
                     replaces `np.dot` in recent numpy versions.
        local_est => response for local regression
    """
    # Sort dataset by xvals.
    all_data = sorted(zip(xvals, yvals), key=lambda x: x[0])
    xvals, yvals = zip(*all_data)

    locsDF = pd.DataFrame(
                columns=[
                  'loc','x','weights','v','y','raw_dists',
                  'scale_factor','scaled_dists'
                  ])
    evalDF = pd.DataFrame(
                columns=[
                  'loc','est','b','v','g'
                  ])

    n = len(xvals)
    m = n + 1
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = ((max(xvals)-min(xvals))/len(xvals))
    v_lb = max(0,min(xvals)-(.5*avg_interval))
    v_ub = (max(xvals)+(.5*avg_interval))
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)

    # Generate design matrix based on poly_degree.
    xcols = [np.ones_like(xvals)]
    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    X = np.vstack(xcols).T


    for i in v:

        iterpos = i[0]
        iterval = i[1]

        # Determine q-nearest xvals to iterval.
        iterdists = sorted([(j, np.abs(j-iterval)) \
                           for j in xvals], key=lambda x: x[1])

        _, raw_dists = zip(*iterdists)

        # Scale local observations by qth-nearest raw_dist.
        scale_fact = raw_dists[q-1]
        scaled_dists = [(j[0],(j[1]/scale_fact)) for j in iterdists]
        weights = [(j[0],((1-np.abs(j[1]**3))**3 \
                      if j[1]<=1 else 0)) for j in scaled_dists]

        # Remove xvals from each tuple:
        _, weights      = zip(*sorted(weights,     key=lambda x: x[0]))
        _, raw_dists    = zip(*sorted(iterdists,   key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists,key=lambda x: x[0]))

        iterDF1 = pd.DataFrame({
                    'loc'         :iterpos,
                    'x'           :xvals,
                    'v'           :iterval,
                    'weights'     :weights,
                    'y'           :yvals,
                    'raw_dists'   :raw_dists,
                    'scale_fact'  :scale_fact,
                    'scaled_dists':scaled_dists
                    })

        locsDF    = pd.concat([locsDF, iterDF1])
        W         = np.diag(weights)
        y         = yvals
        b         = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
        local_est = loc_eval(iterval, b)
        iterDF2   = pd.DataFrame({
                       'loc':[iterpos],
                       'b'  :[b],
                       'v'  :[iterval],
                       'g'  :[local_est]
                       })

        evalDF = pd.concat([evalDF, iterDF2])

    # Reset indicies for returned DataFrames.
    locsDF.reset_index(inplace=True)
    locsDF.drop('index', axis=1, inplace=True)
    locsDF['est'] = 0; evalDF['est'] = 0
    locsDF = locsDF[['loc','est','v','x','y','raw_dists',
                     'scale_fact','scaled_dists','weights']]

    # Reset index for evalDF.
    evalDF.reset_index(inplace=True)
    evalDF.drop('index', axis=1, inplace=True)
    evalDF = evalDF[['loc','est', 'v', 'b', 'g']]

    return(locsDF, evalDF)




# stationary percent residual
def stationary_percent_residual(ax,plot_data,wind_to_use,
                                full_detection_limit = 15,
                                force_intercept_origin = 0,
                                plot_lim=[0,2000],
                                legend_loc='lower right'):
  """
plot percent residual as a function of x and fit with a LOESS curve

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed data
- wind_to_use is the wind used for finding closest plume quantification in plot_data
- force_intercept_origin decides which regression to use
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend
- full_detection_limit is the wind-normalized rate above which Kairos can detect all plumes, unit is kgh/mps

OUTPUT
- ax is the plotted parity chart
  """

  # set up plot
  ax.set_xlabel('Reported release rate [kgh]',fontsize=13)
  ax.set_ylabel('Percent residual [%]',fontsize=13)
  ax.set_xlim(plot_lim)
  ax.set_ylim([-500,500])

  # add a horizontal line through the origin
  ax.plot(plot_lim,[0,0],color='k')

  # regression and the residual and percent residual
  x = plot_data.CH4_release_kgh.values
  y = plot_data.closest_plume_quantification_kgh.fillna(0).values
  if force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)
    y_pred = x*slope
  elif force_intercept_origin == 0:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
    y_pred = x*slope + intercept
  percent_residual = residual/y_pred*100

  # leave the points below full detection threshold out of LOESS line fitting
  x_leave_out = x[x<full_detection_limit*plot_data[wind_to_use].median()]  # for now assume the median wind speed 
  df = pd.DataFrame({
    'x': x[x>=full_detection_limit*plot_data[wind_to_use].median()],
    'y': percent_residual[x>=full_detection_limit*plot_data[wind_to_use].median()]
  })
  # LOESS first-degree line fit
  regsDF, evalDF = loess(df.x, df.y,alpha=0.6, poly_degree=1)
  l_x  = evalDF['v'].values
  l_y  = evalDF['g'].values
  ax.plot(l_x, l_y, color='#FF0000', label="1st-degree Polynomial LOESS")

  # scatter plot color-coded
  ax.scatter(df.x[df.x<df.x.median()],df.y[df.x<df.x.median()],c='#8c1515',alpha = 0.5,label='lower half $n$ = %d' %df.x[df.x<df.x.median()].shape[0])
  ax.scatter(df.x[df.x>=df.x.median()],df.y[df.x>=df.x.median()],c = '#D2C295',alpha = 0.5,label='upper half $n$ = %d' %df.x[df.x>=df.x.median()].shape[0])  
  ax.scatter(x_leave_out,percent_residual[x<full_detection_limit*plot_data[wind_to_use].median()],c='k',alpha = 0.5,label = "below full detection threshold \nat moderate wind speed\n$n$ = %d" %(len(x_leave_out)))

  # add vertical dashed lines to partition the colored scatter plot
  ax.plot([np.max(x_leave_out),np.max(x_leave_out)],[-500,400],linestyle='--',color='k',alpha=0.5)
  ax.plot([np.min(df.x[df.x>=df.x.median()]),np.min(df.x[df.x>=df.x.median()])],[-500,400],linestyle='--',color='k',alpha=0.5)

  ax.legend(loc=legend_loc)

  plt.close()

  return ax



# fit percent residual to a normal distribution
def histogram_percent_residual(ax,plot_data,lower_bound_x,upper_bound_x,
                              hist_color='#8c1515',
                              n_bins=10,
                              x_lim=[-100,300], y_lim=[0,0.02],
                              fit_norm=1,
                              force_intercept_origin = 0,
                              legend_loc='lower right',):

  """
plot histogram of percent residual

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed data
- lower_bound_x is the min release rate threshold for including data points in the histogram
- upper_bound_x is the max release rate threshold for including data points in the histogram
- hist_color is the color code for histogram
- n_bins is the specified number of bins
- x_lim and y_lim are the limits of the x and y ranges for the plot
- fit_norm is a binary variable of deciding whether or not to fit a normal distribution curve to the histogram
- force_intercept_origin decides which regression to use
- legend_loc is the location of the legend

OUTPUT
- ax is the plotted parity chart
- mu is the mean of the percent residual with the normal fit
- std is the standard deviation of the percent residual with the normal fit
  """                      

  # set up
  ax.set_xlabel('Percent residual [%]',fontsize=13)
  ax.set_ylabel('Density', fontsize=13)
  ax.set(xlim=x_lim, ylim=y_lim)

  # regression and the residual and percent residual
  x = plot_data.CH4_release_kgh.values
  y = plot_data.closest_plume_quantification_kgh.fillna(0).values
  if force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)
    y_pred = x*slope
  elif force_intercept_origin == 0:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
    y_pred = x*slope + intercept
  percent_residual = residual/y_pred*100
    
  # find the data within the bound
  df = pd.DataFrame({
    'x': x[(x>=lower_bound_x) & (x<upper_bound_x)],
    'pct_resid': percent_residual[(x>=lower_bound_x) & (x<upper_bound_x)]
  })

  # plot histogram and add a normal fit line
  ax.hist(df.pct_resid,bins=n_bins, density=True, color=hist_color,alpha=0.5,label='$n$ = %d'%df.shape[0])
  if fit_norm == 1:
    mu,std = norm.fit(df.pct_resid)
    xmin, xmax = np.min(df.pct_resid), np.max(df.pct_resid)
    xx = np.linspace(xmin, xmax, 100)
    p = norm.pdf(xx, mu, std)
    ax.plot(xx, p, 'k', linewidth=2,label='normal fit \n$\mu$=%.2f%% \n$\sigma$=%.2f%%'%(mu,std))
  else:
    mu = df.x.mean()
    std = df.x.std()

  ax.legend(loc=legend_loc)
  plt.close()

  return ax,mu,std






























