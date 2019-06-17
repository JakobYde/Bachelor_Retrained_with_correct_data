from DataSheet import DataSheet
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np


def scatter_plot(x_data, y_data):
    assert (isinstance(x_data, list)), 'x_data should be a list'
    assert (isinstance(y_data, list)), 'y_data should be a list'
    #f = go.Figure()
    #f.add_scatter(  x=x_data, y=y_data, mode='markers' )
    f = go.Scatter(  x=x_data, y=y_data, mode='markers' )
    return f

def density_plot(x_data, y_data):
    assert (isinstance(x_data, list)), 'x_data should be a list'
    assert (isinstance(y_data, list)), 'y_data should be a list'
    
    colorscale = ['rgb(255,0,0)', (1,1,1)]
    
    f = ff.create_2d_density(
        x_data, y_data, colorscale=colorscale,
        hist_color='rgb(255, 237, 222)', point_size=3, #rgb(255, 237, 222)
        title='',
        height=1080,
        width=1920)
    return f

def parallel_coordinates_plot(x_dict, y_data, y_label, lower_is_better=True):
    assert (isinstance(x_dict, dict)),'x_dict should be a dictionary'
    assert (isinstance(y_data, list)),'y_data should be a list'

    if lower_is_better:
        cmin = np.min(y_data)
        cmax = np.mean(y_data) + np.std(y_data)
    else:
        cmin = np.mean(y_data) - np.std(y_data)
        cmax = np.max(y_data)


    dimensionList = []
    for key in x_dict:
        if True in [isinstance(element, str) for element in x_dict[key]]:
            x_dict[key] = [str(element) for element in x_dict[key]]
            categories = list(set(x_dict[key]))
            values = [categories.index(element) for element in x_dict[key]]
            dimensionList.append(dict(range=[0, len(categories) - 1], tickvals=list(range(0,len(categories))), ticktext=categories, label=key,values=values))
        else:
            dimensionList.append(dict(range=[np.min(x_dict[key]), np.max(x_dict[key])], label=key,values=x_dict[key]))
    dimensionList.append(dict(range=[np.min(y_data), np.max(y_data)],label=y_label,values=y_data))

    f = [go.Parcoords(
            line = dict(color = y_data,
                       colorscale = 'Bluered',
                       reversescale=lower_is_better,
                       showscale = True,
                       cmin = cmin,
                       cmax = cmax),
            dimensions = list(dimensionList))]
    return f

ds = DataSheet('results_0.csv')
removed = ds.remove_by_performance('last_perf')

parcor_data = {}
for i, key in enumerate(ds.data):
    if ds.categories[i] == 'in':
        parcor_data[key] = ds.data[key]

parcor = parallel_coordinates_plot(parcor_data, ds.data['last_perf'], 'last_perf')

#plot_data = [scatter]

py.plot(parcor, filename='parcor.html')

pass