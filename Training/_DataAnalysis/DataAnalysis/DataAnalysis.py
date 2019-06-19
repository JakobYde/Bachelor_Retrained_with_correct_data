from DataSheet import DataSheet
from DataPlotting import *
import numpy as np

ds = DataSheet('results_0.csv')
#removed = ds.remove_by_performance('last_perf')

#ds = ds.exclude(parameters=['simplernn'])

ds_in = ds.include(categories=['in'])
ds_out = ds.include(categories=['out'])

plot_sheet(ds_in, ds_out, False, True, False)

pass