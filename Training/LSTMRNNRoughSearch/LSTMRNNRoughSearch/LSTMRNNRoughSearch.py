import sys
sys.path.insert(0, r"../../..")
import os


import Utility.ParameterGenerator as Pg
import Utility.BachelorUtilities as Bu

seed = 0

pg = Pg.ParameterGenerator(seed=seed)
pg.add_value('rnn_type', default_value='none', change_chance=0.5, choices=['lstm', 'simplernn'])
pg.add_value('rnn_size', choices=range(1,1000))
pg.add_layer('dense_layers', choice_layer_amount=10,choice_layer_sizes=list(range(1,500)))
pg.add_value('learning_rate', choices=[0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005])
pg.add_value('optimizer', choices=['adam', 'rmsprop', 'adadelta'])
pg.add_value('activation', choices=['relu', 'leaky_relu', 'sigmoid'])
pg.add_value('dropout', choices=[0, 0.1, 0.15, 0.2])

parameters = pg.sample(200, unique=True)


x1, x2, y = Bu.load_data('FixedTraining')
cvs = Bu.get_cross_validation(x1, x2, y, 5)

for i_param, param in enumerate(parameters):
    results = []
    for i_cv, cv in enumerate(cvs):
        results.append(Bu.train_network(param, cv, seed=seed))
pass
