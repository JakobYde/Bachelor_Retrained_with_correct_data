import sys
sys.path.insert(0, r"../../..")
import Utility.parameter_generator as Pg
import Utility.bachelor_utilities as Bu
import Utility.network_training as Tr

from random import shuffle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
filename = r'../../../Logs/results.csv'

seed = None
n_cv = 5

pg = Pg.ParameterGenerator(seed=seed)
pg.add_value('dense_layers', default_value=[139, 486, 152, 79, 61, 0, 0, 0, 0, 0])
pg.add_value('learning_rate', default_value=0.005)
pg.add_value('optimizer', default_value='adam')
pg.add_value('activation', default_value='relu')
pg.add_value('dropout', default_value=0.1)
pg.add_value('rnn_activation', default_value='tanh')
pg.add_value('rnn_dropout', default_value=0.1)
pg.add_value('last_activation', default_value='linear')

pg.add_value('rnn_type', default_value='')
pg.add_value('rnn_size', default_value=0)

param = pg.sample(amount=1, unique=True)

rnn_types = ['none', 'lstm', 'simplernn']
rnn_sizes = list(range(50,500,15))

for rnn_type in rnn_types:
    for rnn_size in rnn_sizes:

x1, x2, y = Bu.load_data('FixedTraining')
cvs = Bu.get_cross_validation(x1, x2, y, n_cv)

cbs = Tr.get_callbacks(plat=True, es=True)

arr = pg.as_array(first_param)

head = ['iteration']
head += pg.get_head()
head += ['last_perf', 'min_perf', 'time']
print(head)

log = Bu.CSVWriter(filename, head=head)

shuffle(parameters)

for i_param, param in enumerate(parameters):
    last_perfs = 0
    min_perfs = 0
    time = 0
    for i_cv, cv in enumerate(cvs):
        last_perf, min_perf, dt = Tr.train_network(param, cv, seed=seed, callbacks=cbs, verbose=False)
        last_perfs += last_perf
        min_perfs += min_perf
        time += dt
    last_perfs /= n_cv
    min_perfs /= n_cv
    
    row = [i_param]
    row += pg.as_array(param)
    row += [str(c) for c in [last_perfs, min_perfs, time]]
    print(row)
    log.write_row(row)
    
pass
