import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import numpy as np
from scipy.stats import norm



def flow_variability_plot(extracted_data,fig_index,hist_xlim):

  """
plot the time series and histogram of the flow variability recorded by the flow meter.
The readings are extracted with Google Cloud Vision OCR

INPUT
- extracted_data is the data extracted from the video of the flow meter
- ax1, ax2 are the subplots to be plotted with the time seires and the histogram
- hist_xlim is the limit of the x-axis of the histogram in scfh

OUTPUT
  """
  
  fig,[ax2,ax1] = plt.subplots(1,2,figsize=[15,5],gridspec_kw={'width_ratios': [1, 2]})
  ax1.set_ylabel('flow rate [scfh]',fontsize=13)
  ax2.set_ylabel('density',fontsize=13)
  ax2.set_xlabel('flow rate [scfh]',fontsize=13)  

  mins = mdate.epoch2num(extracted_data.unix_time.values)
  ax1.plot_date(mins,extracted_data.mid_meter_flow_scfh,'-',color='#8c1515')

  mu, std = norm.fit(extracted_data[~np.isnan(extracted_data.mid_meter_flow_scfh)].mid_meter_flow_scfh)
  ax2.hist(extracted_data.mid_meter_flow_scfh, bins=20, density=True, color='#8c1515',alpha=0.3)
  xmin, xmax = extracted_data.mid_meter_flow_scfh.min(),extracted_data.mid_meter_flow_scfh.max()
  x = np.linspace(xmin, xmax, 100)
  p = norm.pdf(x, mu, std)
  ax2.plot(x, p, 'k', linewidth=2)
  ax2.annotate('('+fig_index+') Max flow rate n = %d \nFit results: $\\mu$ = %d,  $\\sigma$ = %d' % (extracted_data[~np.isnan(extracted_data.mid_meter_flow_scfh)].shape[0],mu, std),
                 xy = [0.02,0.86],xycoords='axes fraction',fontsize=13)

  date_formatter = mdate.DateFormatter('%I:%M:%S %p')
  ax2.set_xlim(hist_xlim)
  ax1.set_ylim(hist_xlim)
  ax1.xaxis.set_major_formatter(date_formatter)
  fig.autofmt_xdate()
  plt.close()

  return fig