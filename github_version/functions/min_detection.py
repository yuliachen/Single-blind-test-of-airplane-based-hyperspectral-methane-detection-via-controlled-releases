import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from github_version.functions.data_processing import *

def detection_rate_by_bin(data,n_bins=5,threshold=25,
      wind_to_use='WS_windGust_logged_mps'):

  """
  INPUT
  - data is the processed data after data selection
  - n_bins is the number of bins 
  - threshold is the highest release rate in kgh/mps to show in the detection threshold graph
  OUTPUT
  - detection is a dataframe of wind-normalized release rate of each data point and whether each release was detected by Kairos
  - detection_prob is a dataframe that has n_bins number of rows recording the detection rate of each bin
  """

  # find whether each pass was detected
  detection = pd.DataFrame()
  detection['released'] = data.CH4_release_kgh!=0
  detection['detected'] = ~np.isnan(data.closest_plume_quantification_kghmps)
  detection['release_rate_wind_normalized'] = data.CH4_release_kghmps
  detection[wind_to_use] = data[wind_to_use]
  detection = detection.loc[detection.release_rate_wind_normalized <= threshold]

  # find the median wind of passes below min detection
  median_wind = detection[wind_to_use].median()

  # initiate the bins 
  bins = np.linspace(0,threshold,n_bins+1)
  detection_probability = np.zeros(n_bins)
  detection_probability_highwind = np.zeros(n_bins)
  detection_probability_lowwind = np.zeros(n_bins)
  bin_size, bin_num_detected = np.zeros(n_bins).astype('int'), np.zeros(n_bins).astype('int')
  bin_size_highwind, bin_num_detected_highwind = np.zeros(n_bins).astype('int'), np.zeros(n_bins).astype('int')
  bin_size_lowwind, bin_num_detected_lowwind = np.zeros(n_bins).astype('int'), np.zeros(n_bins).astype('int')
  bin_median = np.zeros(n_bins)
  bin_two_sigma, bin_two_sigma_highwind, bin_two_sigma_lowwind = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
  two_sigma_upper, two_sigma_lower = np.zeros(n_bins),np.zeros(n_bins)
  two_sigma_upper_highwind, two_sigma_lower_highwind = np.zeros(n_bins),np.zeros(n_bins)
  two_sigma_upper_lowwind, two_sigma_lower_lowwind = np.zeros(n_bins),np.zeros(n_bins)

  # for each bin, find number of data points and detection prob
  for i in range(n_bins):
      bin_min = bins[i]
      bin_max = bins[i+1]
      bin_median[i] = (bin_min+bin_max)/2
      binned_data = detection.loc[detection.release_rate_wind_normalized<bin_max].loc[detection.release_rate_wind_normalized>=bin_min]
      
      ################# all data within bin ######################
      bin_num_detected[i] = binned_data.detected.sum()
      n = len(binned_data)
      bin_size[i] = n
      p = binned_data.detected.sum()/binned_data.shape[0]
      detection_probability[i] = p
      
      # std of binomial distribution
      sigma = np.sqrt(p*(1-p)/n)
      bin_two_sigma[i] = 2*sigma

      # find the lower and uppder bound defined by two sigma
      two_sigma_lower[i] = 2*sigma
      two_sigma_upper[i] = 2*sigma
      if 2*sigma + p > 1:
        two_sigma_upper[i] = 1-p
      if p - 2*sigma < 0 :
        two_sigma_lower[i] = p
      
      ################# high wind ###########################
      binned_data_highwind = binned_data.loc[binned_data[wind_to_use]>median_wind]
      bin_num_detected_highwind[i] = binned_data_highwind.detected.sum()
      n = len(binned_data_highwind)
      bin_size_highwind[i] = n
      p = binned_data_highwind.detected.sum()/binned_data_highwind.shape[0]
      detection_probability_highwind[i] = p
      
      # std of binomial distribution
      sigma = np.sqrt(p*(1-p)/n)
      bin_two_sigma_highwind[i] = 2*sigma

      # find the lower and uppder bound defined by two sigma
      two_sigma_lower_highwind[i] = 2*sigma
      two_sigma_upper_highwind[i] = 2*sigma
      if 2*sigma + p > 1:
        two_sigma_upper_highwind[i] = 1-p
      if p - 2*sigma < 0 :
        two_sigma_lower_highwind[i] = p

      ################# low wind ###########################
      binned_data_lowwind = binned_data.loc[binned_data[wind_to_use]<=median_wind]
      bin_num_detected_lowwind[i] = binned_data_lowwind.detected.sum()
      n = len(binned_data_lowwind)
      bin_size_lowwind[i] = n
      p = binned_data_lowwind.detected.sum()/binned_data_lowwind.shape[0]
      detection_probability_lowwind[i] = p
      
      # std of binomial distribution
      sigma = np.sqrt(p*(1-p)/n)
      bin_two_sigma_lowwind[i] = 2*sigma

      # find the lower and uppder bound defined by two sigma
      two_sigma_lower_lowwind[i] = 2*sigma
      two_sigma_upper_lowwind[i] = 2*sigma
      if 2*sigma + p > 1:
        two_sigma_upper_lowwind[i] = 1-p
      if p - 2*sigma < 0 :
        two_sigma_lower_lowwind[i] = p

  ################## store data #########################
  detection_prob = pd.DataFrame({
    "bin_median": bin_median,
    "detection_prob_mean": detection_probability,
    "detection_prob_two_sigma_upper": two_sigma_upper,
    "detection_prob_two_sigma_lower": two_sigma_lower,
    "n_data_points": bin_size,
    "n_detected": bin_num_detected})
  
  detection_prob_highwind = pd.DataFrame({
    "bin_median": bin_median,
    "detection_prob_mean": detection_probability_highwind,
    "detection_prob_two_sigma_upper": two_sigma_upper_highwind,
    "detection_prob_two_sigma_lower": two_sigma_lower_highwind,
    "n_data_points": bin_size_highwind,
    "n_detected": bin_num_detected_highwind})
  
  detection_prob_lowwind = pd.DataFrame({
    "bin_median": bin_median,
    "detection_prob_mean": detection_probability_lowwind,
    "detection_prob_two_sigma_upper": two_sigma_upper_lowwind,
    "detection_prob_two_sigma_lower": two_sigma_lower_lowwind,
    "n_data_points": bin_size_lowwind,
    "n_detected": bin_num_detected_lowwind})

  return detection, detection_prob, detection_prob_highwind, detection_prob_lowwind




# plot the minimum detection limit
def detection_rate_bin_plot(ax, data, n_bins = 5, threshold = 25, by_wind_speed = 0,
              wind_to_use = "WS_windGust_logged_mps"):
  """ 
  INPUT
  - ax is the subplot to be plotted
  - data is the processed data after data selection
  - n_bins: number of bins
  - threshold: max wind-normalized release rate to include in the plot
  - by_wind_speed is a binary that decides whether to split the data into low/high wind when plotting
  - wind_to_use is the type of wind speed to use to normalize the release rate
  OUTPUT
  - ax is the subplot to show the minium detection
  """

  detection, detection_prob, detection_prob_highwind, detection_prob_lowwind = detection_rate_by_bin(data,
                      n_bins,threshold,wind_to_use = wind_to_use)
  w = threshold/n_bins/2.5    # bin width

  if by_wind_speed==0:
    for i in range(n_bins):
      ax.annotate('%d / %d' %(detection_prob.n_detected[i],detection_prob.n_data_points[i]),
                  [detection_prob.bin_median[i]-w/1.8,0.03],fontsize=10)

    # for plotting purpose, we don't want a small hypen indicating zero uncertainty interval
    detection_prob.detection_prob_two_sigma_lower[detection_prob.detection_prob_two_sigma_lower==0]=np.nan  
    detection_prob.detection_prob_two_sigma_upper[detection_prob.detection_prob_two_sigma_upper==0]=np.nan
    detection_prob.detection_prob_mean[detection_prob.detection_prob_mean==0]=np.nan

    # plot the bars and the detection points
    ax.bar(detection_prob.bin_median,detection_prob.detection_prob_mean,
          yerr=[detection_prob.detection_prob_two_sigma_lower,detection_prob.detection_prob_two_sigma_upper],
          error_kw=dict(lw=2, capsize=3, capthick=1,alpha=0.3),
          width=threshold/n_bins-0.5,alpha=0.6,color='#8c1515',ecolor='black', capsize=2)
    ax.scatter(detection.release_rate_wind_normalized,np.multiply(detection.detected,1),
              edgecolor="black",facecolors='none')
  
  elif by_wind_speed == 1:   # split data into two sets based on wind speed
  
    # find median wind
    median_wind = detection[wind_to_use].median()
    detection_lowwind = detection[detection[wind_to_use]<median_wind]
    detection_highwind = detection[detection[wind_to_use]>=median_wind]
    n_below_median_wind = detection_lowwind.shape[0]
    n_above_median_wind = detection_lowwind.shape[0]
  
    # width and position of the bars
    x = detection_prob.bin_median
    w = threshold/n_bins/2.5

    for i in range(n_bins):
      offset_annotation = -0.3
      dist_annotation = 1.7
      # low wind
      ax.annotate("%d" %detection_prob_lowwind.n_detected[i],[x[i]-w/dist_annotation+offset_annotation,0.11],fontsize=10)
      ax.annotate("—",[detection_prob_lowwind.bin_median[i]-w/dist_annotation-0.2+offset_annotation,0.081],fontsize=10)
      ax.annotate("%d"%detection_prob_lowwind.n_data_points[i],[x[i]-w/dist_annotation+offset_annotation,0.04],fontsize=10) 
      # high wind
      ax.annotate("%d" %detection_prob_highwind.n_detected[i],[x[i]+w/dist_annotation+offset_annotation,0.11],fontsize=10)
      ax.annotate("—",[detection_prob_highwind.bin_median[i]+w/dist_annotation-0.2+offset_annotation,0.081],fontsize=10)
      ax.annotate("%d"%detection_prob_highwind.n_data_points[i],[x[i]+w/dist_annotation+offset_annotation,0.04],fontsize=10) 

    # for plotting purpose, we don't want a small hypen indicating zero uncertainty interval
    detection_prob_lowwind.detection_prob_two_sigma_lower[detection_prob_lowwind.detection_prob_two_sigma_lower==0]=np.nan  
    detection_prob_lowwind.detection_prob_two_sigma_upper[detection_prob_lowwind.detection_prob_two_sigma_upper==0]=np.nan
    detection_prob_lowwind.detection_prob_mean[detection_prob_lowwind.detection_prob_mean==0]=np.nan
    detection_prob_highwind.detection_prob_two_sigma_lower[detection_prob_highwind.detection_prob_two_sigma_lower==0]=np.nan  
    detection_prob_highwind.detection_prob_two_sigma_upper[detection_prob_highwind.detection_prob_two_sigma_upper==0]=np.nan
    detection_prob_highwind.detection_prob_mean[detection_prob_highwind.detection_prob_mean==0]=np.nan

    # low wind: plot the bars and the detection points
    ax.bar(x - w/2, detection_prob_lowwind.detection_prob_mean,
          yerr=[detection_prob_lowwind.detection_prob_two_sigma_lower,
                detection_prob_lowwind.detection_prob_two_sigma_upper],
          error_kw=dict(lw=2, capsize=3, capthick=1,alpha=0.3),
          width=w,alpha=0.6,color='#00505c',ecolor='black', capsize=2,align='center',
          label='below %.2f mps n=%d'%(median_wind,n_below_median_wind))
    ax.scatter(detection_lowwind.release_rate_wind_normalized,np.multiply(detection_lowwind.detected,1),
              edgecolor="black",facecolors='#00505c',alpha=0.3)

    # high wind: plot the bars and the detection points
    ax.bar(x + w/2, detection_prob_highwind.detection_prob_mean,
          yerr=[detection_prob_highwind.detection_prob_two_sigma_lower,
                detection_prob_highwind.detection_prob_two_sigma_upper],
          error_kw=dict(lw=2, capsize=3, capthick=1,alpha=0.3),
          width=w,alpha=0.6,color='#5f574f',ecolor='black', capsize=2,align='center',
          label='above %.2f mps n=%d'%(median_wind,n_above_median_wind))
    ax.scatter(detection_highwind.release_rate_wind_normalized,np.multiply(detection_highwind.detected,1),
              edgecolor="black",facecolors='#5f574f',alpha=0.3)
    
    ax.legend(loc='upper right',fontsize=11)

  # set more room on top for annotation
  ax.set_ylim([-0.05,1.22])
  ax.set_xlim([0,threshold])
  ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
  ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=11)
  ax.set_xlabel('Wind-speed-normalized methane release rate [kgh/mps]',fontsize=12)
  ax.set_ylabel('Proportion detected',fontsize=11)
  
  plt.close()

  return ax


# plot three panel comparison of min detection based on wind speed
def detection_rate_bin_plot_by_wind_speed(ax1,ax2,ax3, data, n_bins = 5, threshold = 25,
              wind_to_use = "WS_windGust_logged_mps"):

  """ 
  INPUT
  - ax1,ax2,ax3 are the subplots to be plotted
  - data is the processed data after data selection
  - n_bins: number of bins
  - threshold: max wind-normalized release rate to include in the plot
  - wind_to_use is the type of wind speed to use to normalize the release rate
  OUTPUT
  - ax1,ax2,ax3 are the subplots to show the minium detection
  """

  detection, detection_prob, detection_prob_highwind, detection_prob_lowwind = detection_rate_by_bin(data,
                      n_bins,threshold,wind_to_use = wind_to_use)

  # x-position of bars
  x = detection_prob.bin_median

  ####################### find median wind #####################
  median_wind = detection[wind_to_use].median()
  detection_lowwind = detection[detection[wind_to_use]<=median_wind]
  detection_highwind = detection[detection[wind_to_use]>median_wind]
  n_below_median_wind = detection_lowwind.shape[0]
  n_above_median_wind = detection_lowwind.shape[0]

  ############### overall #######################
  for i in range(n_bins):
    ax1.annotate('%d / %d' %(detection_prob.n_detected[i],detection_prob.n_data_points[i]),
                [x[i]-0.8,0.03],fontsize=10)

  # for plotting purpose, we don't want a small hypen indicating zero uncertainty interval
  detection_prob.detection_prob_two_sigma_lower[detection_prob.detection_prob_two_sigma_lower==0]=np.nan  
  detection_prob.detection_prob_two_sigma_upper[detection_prob.detection_prob_two_sigma_upper==0]=np.nan
  detection_prob.detection_prob_mean[detection_prob.detection_prob_mean==0]=np.nan

  # plot the bars and the detection points
  ax1.bar(x,detection_prob.detection_prob_mean,
        yerr=[detection_prob.detection_prob_two_sigma_lower,detection_prob.detection_prob_two_sigma_upper],
        error_kw=dict(lw=2, capsize=3, capthick=1,alpha=0.3),
        width=threshold/n_bins-0.5,alpha=0.6,color='#8c1515',ecolor='black', capsize=2,
        label = 'all wind speeds\nn=%d'%detection.shape[0])
  ax1.scatter(detection.release_rate_wind_normalized,np.multiply(detection.detected,1),
            edgecolor="black",facecolors='none')
  ax1.legend(loc = 'upper right',fontsize=11)

  ######################### low wind ##########################3
  for i in range(n_bins):
    ax2.annotate('%d / %d' %(detection_prob_lowwind.n_detected[i],detection_prob_lowwind.n_data_points[i]),
                [x[i]-0.8,0.03],fontsize=10)

  # for plotting purpose, we don't want a small hypen indicating zero uncertainty interval
  detection_prob_lowwind.detection_prob_two_sigma_lower[detection_prob_lowwind.detection_prob_two_sigma_lower==0]=np.nan  
  detection_prob_lowwind.detection_prob_two_sigma_upper[detection_prob_lowwind.detection_prob_two_sigma_upper==0]=np.nan
  detection_prob_lowwind.detection_prob_mean[detection_prob_lowwind.detection_prob_mean==0]=np.nan

  # plot the bars and the detection points
  ax2.bar(x,detection_prob_lowwind.detection_prob_mean,
        yerr=[detection_prob.detection_prob_two_sigma_lower,detection_prob.detection_prob_two_sigma_upper],
        error_kw=dict(lw=2, capsize=3, capthick=1,alpha=0.3),
        width=threshold/n_bins-0.5,alpha=0.6,color='#00505c',ecolor='black', capsize=2,
        label = '<= %.2f mps\nn=%d'%(median_wind,detection_lowwind.shape[0]))
  ax2.scatter(detection_lowwind.release_rate_wind_normalized,np.multiply(detection_lowwind.detected,1),
            edgecolor="black",facecolors='none')
  ax2.legend(loc = 'upper right',fontsize=11)

  ######################### high wind ##########################3
  for i in range(n_bins):
    ax3.annotate('%d / %d' %(detection_prob_highwind.n_detected[i],detection_prob_highwind.n_data_points[i]),
                [x[i]-0.8,0.03],fontsize=10)

  # for plotting purpose, we don't want a small hypen indicating zero uncertainty interval
  detection_prob_highwind.detection_prob_two_sigma_lower[detection_prob_highwind.detection_prob_two_sigma_lower==0]=np.nan  
  detection_prob_highwind.detection_prob_two_sigma_upper[detection_prob_highwind.detection_prob_two_sigma_upper==0]=np.nan
  detection_prob_highwind.detection_prob_mean[detection_prob_highwind.detection_prob_mean==0]=np.nan

  # plot the bars and the detection points
  ax3.bar(x,detection_prob_highwind.detection_prob_mean,
        yerr=[detection_prob_highwind.detection_prob_two_sigma_lower,detection_prob_highwind.detection_prob_two_sigma_upper],
        error_kw=dict(lw=2, capsize=3, capthick=1,alpha=0.3),
        width=threshold/n_bins-0.5,alpha=0.6,color='#5f574f',ecolor='black', capsize=2,
        label = '> %.2f mps\nn=%d'%(median_wind,detection_highwind.shape[0]))
  ax3.scatter(detection_highwind.release_rate_wind_normalized,np.multiply(detection_highwind.detected,1),
            edgecolor="black",facecolors='none')
  ax3.legend(loc='upper right',fontsize=11)

  ########### axis and other plot settings ##############3
  ax1.set_ylim([-0.05,1.22]),ax2.set_ylim([-0.05,1.22]),ax3.set_ylim([-0.05,1.22])
  ax1.set_xlim([0,threshold]),ax2.set_xlim([0,threshold]),ax3.set_xlim([0,threshold])
  ax1.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
  ax2.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
  ax3.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
  ax1.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=11)
  ax2.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=11)
  ax3.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=11)
  ax1.set_xlabel('Wind-speed-normalized methane release rate [kgh/mps]',fontsize=12)
  ax1.set_ylabel('Proportion detected',fontsize=11)
  ax2.set_xlabel('Wind-speed-normalized methane release rate [kgh/mps]',fontsize=12)
  ax2.set_ylabel('Proportion detected',fontsize=11)
  ax3.set_xlabel('Wind-speed-normalized methane release rate [kgh/mps]',fontsize=12)
  ax3.set_ylabel('Proportion detected',fontsize=11)
  
  plt.close()

  return ax1,ax2,ax3







  
