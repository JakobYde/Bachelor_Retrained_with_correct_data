import sys
sys.path.insert(0, r"../../..")
import Utility.parameter_generator as Pg
import Utility.bachelor_utilities as Bu
import Utility.network_training as Tr

filename = r'../../../Logs/initial_training.csv'

seed = 0
n_cv = 5

pg = Pg.ParameterGenerator(seed=seed)
pg.add_layer('dense_layers', choice_layer_amount=10,choice_layer_sizes=list(range(1,500)))
pg.add_value('learning_rate', choices=[0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005])
pg.add_value('optimizer', choices=['adam', 'rmsprop', 'adadelta'])
pg.add_value('activation', choices=['relu', 'leaky_relu'])
pg.add_value('dropout', choices=[0, 0.1, 0.15, 0.2, 0.25])
pg.add_value('rnn_type', default_value='none', change_chance=0.5, choices=['lstm', 'simplernn'])
pg.add_value('rnn_size', choices=range(1,1000))
pg.add_value('rnn_activation', choices=['relu','tanh'])
pg.add_value('rnn_dropout', choices=[0, 0.1, 0.15, 0.2, 0.25])
pg.add_value('last_activation', choices=['relu', 'linear'])

parameters = pg.sample(200, unique=True)

first_param = parameters[0]
last_param = parameters[-1]

x1, x2, y = Bu.load_data('FixedTraining')
cvs = Bu.get_cross_validation(x1, x2, y, n_cv)

cbs = Tr.get_callbacks(plat=True)

arr = pg.as_array(first_param)

head = ['iteration']
head += pg.get_head()
head += ['last_perf', 'min_perf', 'time']
print(head)

log = Bu.CSVWriter(filename, head=head)

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
