import sys
import network_training as Tr
sys.path.insert(0, r"../../..")
import Utility.parameter_generator as Pg
import Utility.bachelor_utilities as Bu

from tensorflow import set_random_seed
from numpy.random import seed as set_numpy_seed

from random import shuffle, randint

from keras.models import save_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
filename = r'../../../Logs/results.csv'

n_cv = 5

pg = Pg.ParameterGenerator()
pg.add_value('dense_layers', default_value=[139, 486, 152, 79, 61, 0, 0, 0, 0, 0])
pg.add_value('learning_rate', default_value=0.005)
pg.add_value('optimizer', default_value='adam')
pg.add_value('activation', default_value='relu')
pg.add_value('dropout', default_value=0.025)
pg.add_value('rnn_type', default_value='lstm')
pg.add_value('rnn_size', default_value=230)
pg.add_value('rnn_activation', default_value='tanh')
pg.add_value('rnn_dropout', default_value=0.125)
pg.add_value('last_activation', default_value='relu')

parameters = pg.sample(1, unique=True)

x1, x2, y = Bu.load_data('FixedTraining')
cvs = Bu.get_cross_validation(x1, x2, y, n_cv)

cbs = Tr.get_callbacks(plat=True, es=True, tb=True, tb_path=r'D:\WindowsFolders\Documents\GitHub\BachelorRetraining\Logs\TB')

head = ['iteration']
head += pg.get_head()
head += ['last_perf', 'min_perf', 'time']
print(head)

log = Bu.CSVWriter(filename, head=head)


best_perf = 100
i = 0
i += 1
for i_param, param in enumerate(parameters):
    last_perfs = 0
    min_perfs = 0
    time = 0
    for i_cv, cv in enumerate(cvs):
        model_path = 'weights_{}.h5'.format(i)
        if i_param + i_cv == 0:
            model_storage = 'save'
        else:
            model_storage = 'load'
                
        if i_cv == 4:
            last_perf, min_perf, dt = Tr.train_network(param, cv, seed=None, callbacks=cbs, verbose=False, model_path=model_path, model_storage=model_storage, save_threshold=best_perf, current_perf=last_perfs)
        else:
            last_perf, min_perf, dt = Tr.train_network(param, cv, seed=None, callbacks=cbs, verbose=False, model_path=model_path, model_storage=model_storage)
        last_perfs += last_perf
        min_perfs += min_perf
        time += dt
    last_perfs /= n_cv
    min_perfs /= n_cv

    if last_perfs < best_perf:
        best_perf = last_perfs

    row = [i]
    row += pg.as_array(param)
    row += [str(c) for c in [last_perfs, min_perfs, time]]
    print(row)
    log.write_row(row)
    
pass
