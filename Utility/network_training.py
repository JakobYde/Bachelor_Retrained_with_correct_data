import numpy as np
import datetime
from keras.layers import Input, LSTM, Dense, SimpleRNN, concatenate, Masking, Dropout, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.utils import to_categorical
from keras.initializers import RandomNormal, glorot_uniform, orthogonal
from keras.activations import tanh, relu, linear
from keras.layers import LeakyReLU
from keras.backend import clear_session

max_layers = 10

def get_inputs(data, rnn_type='none'):
    [[x1t, x1v], [x2t, x2v], [yt, yv]] = data
    if rnn_type == 'none':
        x1t = to_categorical(x1t, 5)
        x1v = to_categorical(x1v, 5)

        x1t = [i.flatten() for i in x1t]
        x1v = [i.flatten() for i in x1v]

        input_eular = Input(shape=(len(x1t[0]),), dtype='float32', name='input_eular')
        input_crp = Input(shape=(1,), dtype='float32', name='input_crp')

    else:
        input_eular = Input(shape=(x1t.shape[1], 1), dtype='float32', name='input_eular')
        input_crp = Input(shape=(1,), dtype='float32', name='input_crp')
    return input_eular, input_crp, [[x1t, x1v], [x2t, x2v], [yt, yv]]

def get_callbacks(plat=False, plat_patience=10, plat_delta=0.001, plat_factor=0.1, es=False, es_restore=False, es_patience=40, es_delta=0, tb=False, tb_path=''):
    assert(((not tb) or (tb_path != '')) and isinstance(tb_path, str)),'If TensorBoard is used, the path should be set.'
    cbs = []
    if plat: cbs.append(ReduceLROnPlateau(patience=plat_patience, factor=plat_factor, min_delta=plat_delta))
    if es: cbs.append(EarlyStopping(patience=es_patience, min_delta=es_delta, restore_best_weights=es_restore))
    if tb: cbs.append(TensorBoard(log_dir=tb_path))
    return cbs

def get_optimizer(optimizer='adam', learning_rate=0.001):
    assert (optimizer in ['adam', 'rmsprop', 'adadelta']),'Invalid optimizer.'
    assert (isinstance(learning_rate, float)),'Learning rate should be a floating point number.'
    
    if optimizer == 'adam': optimizer = Adam(lr=learning_rate)
    if optimizer == 'rmsprop': optimizer = RMSprop(lr=learning_rate)
    if optimizer == 'adadelta': optimizer = Adadelta()
    return optimizer

def get_activation(x, activation='relu'):
    assert (activation in ['linear', 'relu', 'leaky_relu', 'tanh', 'sigmoid']),'Activation function not recognized.'
    if activation == 'linear': act = Activation('linear')
    if activation == 'relu': act = Activation('relu')
    if activation == 'leaky_relu': act = LeakyReLU()
    if activation == 'tanh': act = Activation('relu')
    if activation == 'sigmoid': act = Activation('sigmoid')
    return act

def build_model(input_eular, input_crp, parameters, seed=None):
    dropout = parameters['dropout']
    optimizer = parameters['optimizer']
    learning_rate = parameters['learning_rate']

    rnn_type = parameters['rnn_type']
    rnn_size = parameters['rnn_size']
    rnn_activation = parameters['rnn_activation']
    rnn_dropout = parameters['rnn_dropout']

    dense_initializer = RandomNormal(seed=seed)

    if rnn_type == 'none': x = concatenate([input_eular, input_crp], axis=1)
    else:
        rnn_kernel_initializer = glorot_uniform(seed=seed)
        rnn_recurrent_initializer = orthogonal(seed=seed)
        x = Masking(mask_value=-1)(input_eular)

        if rnn_type == 'lstm':
            x = LSTM(rnn_size, activation=rnn_activation, 
                     kernel_initializer=rnn_kernel_initializer, recurrent_initializer=rnn_recurrent_initializer, 
                     dropout=dropout, recurrent_dropout=rnn_dropout)(x)

        elif rnn_type == 'simplernn':
            x = SimpleRNN(rnn_size, activation=rnn_activation, 
                          kernel_initializer=rnn_kernel_initializer, recurrent_initializer=rnn_recurrent_initializer, 
                          dropout=dropout, recurrent_dropout=rnn_dropout)(x)
        
        x = concatenate([x, input_crp])

    for layer_size in [val for val in parameters["dense_layers"] if val != 0]:
        x = Dense(layer_size, activation='linear', kernel_initializer=dense_initializer)(x)
        x = get_activation(x, parameters['activation'])(x)
        if dropout != 0:
            x = Dropout(rate=dropout)(x)
    output = Dense(1, kernel_initializer=dense_initializer)(x)
    output = get_activation(output, parameters['last_activation'])(output)

    model = Model(input=[input_eular, input_crp], output=output)

    return model

def print_row(filename, parameters, min_performance, last_performance, log, time=0.):
    assert(filename == ''),'filename should be set if printing to file.'
    assert(filename[-4:] != '.csv'),'file should be of .csv format'

    #row = [parameters[key] for key in parameters if key != 'dense_layers']
    #row.append(parameters[['dense_layers'], [0 for i in range(max_layers - len(parameters[key]))]])
    
    row = []
    for key in parameters:
        if key != 'dense_layers':
            row.append(parameters[key])
        else:
            for i in range(max_layers):
                if i < len(parameters[key]):
                    row.append(parameters[key][i])
                else:
                    row.append(0)
    log.write_row(row)
    return row

def train_network(parameters, data, epochs=1000, batch_size=32, loss='mse', verbose=False, seed=None, use_min_perf=False, callbacks=[], model_path='', model_storage=''):
    assert(isinstance(parameters, dict)),'Parameters should be a dictionary.'
    assert(model_storage in ["", "save", "load"]),'model_storage command not recognized.'
    clear_session()

    t_start = datetime.datetime.now()

    input_eular, input_crp, data = get_inputs(data, rnn_type=parameters['rnn_type'])

    [[x1t, x1v], [x2t, x2v], [yt, yv]] = data

    model = build_model(input_eular=input_eular, input_crp=input_crp, parameters=parameters, seed=seed)
    
    optimizer = get_optimizer(parameters['optimizer'], parameters['learning_rate'])
    model.compile(optimizer=optimizer, loss=loss)

    if verbose: model.summary()

    hist = model.fit(
                    x=[x1t, x2t],
                    y=yt,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=callbacks,
                    validation_data=([x1v, x2v], yv)
                    )

    min_perf = min(hist.history['val_loss'])
    last_perf = hist.history['val_loss'][-1]

    time = datetime.datetime.now() - t_start
    time = time.seconds + time.microseconds / 1e6

    return last_perf, min_perf, time