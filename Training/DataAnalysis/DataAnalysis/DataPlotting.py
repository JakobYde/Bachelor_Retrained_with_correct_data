from DataSheet import DataSheet
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

def scatter_plot(x_data, y_data):
    assert (isinstance(x_data, list)), 'x_data should be a list'
    assert (isinstance(y_data, list)), 'y_data should be a list'

    if True in [isinstance(element, str) for element in x_data]:
        x_data = [str(element) for element in x_data]
        categories = list(set(x_data))
        values = [categories.index(element) + 1 for element in x_data]
        tickvals = list(range(0, len(categories) + 2))
        categories = [''] + categories + ['']
        f = go.Scatter(x=values, y=y_data, mode='markers')
        layout = go.Layout(xaxis = dict(range=[min(tickvals), max(tickvals)], tickvals=tickvals, ticktext=categories))
        f = go.Figure(data=[f], layout=layout)
    else:
        f = go.Scatter(x=x_data, y=y_data, mode='markers')
        f = go.Figure(data=[f])

    return f

def density_plot(x_data, y_data):
    assert (isinstance(x_data, list)), 'x_data should be a list'
    assert (isinstance(y_data, list)), 'y_data should be a list'

    colorscale = ['rgb(255,0,0)', (1,1,1)]
    
    

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
            tickvals = [-0.5] + list(range(0, len(categories))) + [len(categories) - 0.5]
            categories = [''] + categories + ['']
            dimensionList.append(dict(range=[min(tickvals), max(tickvals)], tickvals=tickvals, ticktext=categories, label=key,values=values))
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

def plot_sheet(datasheet_x, datasheet_y, parcor=False, scatter=False, density=False):
    assert (isinstance(datasheet_x, DataSheet) and isinstance(datasheet_y, DataSheet))
    data_x = datasheet_x.get_data()
    data_y = datasheet_y.get_data()

    for y_parameter in data_y:
        if parcor: 
            plot = parallel_coordinates_plot(data_x, data_y[y_parameter], y_parameter)
            filename = 'parcor_{}.html'.format(y_parameter)
            py.plot(plot, filename=filename, auto_open=False)

        if scatter:
            for x_parameter in data_x:
                plot = scatter_plot(data_x[x_parameter], data_y[y_parameter])
                filename = 'scatter_{}_{}.html'.format(x_parameter, y_parameter)
                py.plot(plot, filename=filename, auto_open=False)
                
        if density:
            for x_parameter in data_x:
                plot = density_plot(data_x[x_parameter], data_y[y_parameter])
                filename = 'scatter_{}_{}.html'.format(x_parameter, y_parameter)
                py.plot(plot, filename=filename, auto_open=False)

    pass
