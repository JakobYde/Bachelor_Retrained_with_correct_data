import numpy as np
import keras
from os import getenv

from keras.layers import Input, LSTM, Dense, SimpleRNN, concatenate, Masking, Dropout
from keras.models import Sequential, Model
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau

# Training - training set
# Testing - testing set
# New - new testing set
# FixedTraining - training set with no incorrect das28 values
# FixedTesting - testing set like above
def load_data(name, categorical=False):
    assert (name in ['Training', 'Testing', 'New', 'FixedTraining', 'FixedTesting']),'Name not recognised.'
    datapath = getenv('PATH_DATA')

    if name == 'Training':
        eul = np.load(datapath + r"\BachelorOriginalData\TrainingDataEUL.npy")
        eul = np.array(eul, dtype='float32')
        crp = np.load(datapath + r"\BachelorOriginalData\TrainingDataCRP.npy")
        das = np.load(datapath + r"\BachelorOriginalData\TrainingDataY.npy")

    if name == 'Testing':
        eul = np.load(datapath + r"\BachelorOriginalData\TestingDataEUL.npy")
        eul = np.array(eul, dtype='float32')
        crp = np.load(datapath + r"\BachelorOriginalData\TestingDataCRP.npy")
        das = np.load(datapath + r"\BachelorOriginalData\TestingDataY.npy")
    
    if name == 'New':
        eul = np.load(datapath + r"\BachelorOriginalData\NewDataEUL.npy")
        eul = np.array(eul, dtype='float32')
        crp = np.load(datapath + r"\BachelorOriginalData\NewDataCRP.npy")
        das = np.load(datapath + r"\BachelorOriginalData\NewDataDAS.npy")
    
    if name == 'FixedTraining':
        eul = np.load(datapath + r"\BachelorFixedData\TrainingDataEUL.npy")
        eul = np.array(eul, dtype='float32')
        crp = np.load(datapath + r"\BachelorFixedData\TrainingDataCRP.npy")
        das = np.load(datapath + r"\BachelorFixedData\TrainingDataY.npy")
    
    if name == 'FixedTesting':
        eul = np.load(r"D:\WindowsFolders\Code\Data\BachelorFixedData\TestingDataEUL.npy")
        eul = np.array(eul, dtype='float32')
        crp = np.load(r"D:\WindowsFolders\Code\Data\BachelorFixedData\TestingDataCRP.npy")
        das = np.load(r"D:\WindowsFolders\Code\Data\BachelorFixedData\TestingDataY.npy")

    if categorical:      
        eul = keras.utils.to_categorical(eul, 5)
        eul = np.array([i.flatten() for i in eul])
        eul = np.reshape(eul, (eul.shape[0], eul.shape[1]))

    return eul, crp, das

def get_cross_validation(eul, crp, das28, n):
    eul = np.array_split(np.reshape(eul, (-1, eul.shape[1], 1)), n)
    crp = np.array_split(crp, n)
    das28 = np.array_split(das28, n)

    result = []
    for i in range(0, n):
        res = []
        for d in [eul, crp, das28]:
            dT = np.array([])
            for x in range(0, len(d)):
                if x is not i:
                    if len(dT) is 0:
                        dT = d[x]
                    else:
                        dT = np.append(dT, d[x], 0)
                else:
                    dV = d[x]
            res.append([dT, dV])
        result.append(res)

    for i, dataset in enumerate(result):
        [[x1t, x1v], [x2t, x2v], [yt, yv]] = dataset
        m = np.mean(x2t)
        std = np.std(x2t)
        x2t = (x2t - m) / std
        x2v = (x2v - m) / std
        result[i] = [[x1t, x1v], [x2t, x2v], [yt, yv]]  
    return result

def build_model(parameters, seed=None, rnn_type="none", input_eular, input_crp):

    dense_initializer = keras.initializers.RandomNormal(seed=seed)

    if rnn_type == "none":
        x = concatenate([input_eular, input_crp], axis=1)

    else:
        rnn_kernel_initializer = keras.initializers.glorot_uniform(seed=seed)
        rnn_recurrent_initializer = keras.initializers.orthogonal(seed=seed)
        x = Masking(mask_value=-1)(input_eular)

        if rnn_type == "lstm":
            x = LSTM(parameters["rnn_size"], activation=parameters["lstm_activation"], 
                     kernel_initializer=rnn_kernel_initializer, recurrent_initializer=rnn_recurrent_initializer, 
                     dropout=parameters["dropout"], recurrent_dropout=0.0)(x)

        elif rnn_type == "simplernn":
            x = SimpleRNN(parameters["rnn_size"], activation=parameters["lstm_activation"], 
                          kernel_initializer=rnn_kernel_initializer, recurrent_initializer=rnn_recurrent_initializer, 
                          dropout=parameters["dropout"], recurrent_dropout=0.0)(x)
        
        x = concatenate([x, input_crp])

    for layer_size in parameters["dense_layers"]:
        x = Dense(layer_size, activation=parameters["activation"])(x)
        if parameters["dropout"] != 0:
            x = Dropout(parameters["dropout"])(x)
    output = Dense(1, activation=parameters["last_activation"])(x)

    model = Model(input=[input_eular, input_crp], output=output)

    if parameters["optimizer"] == "adam":
        optimizer = Adam(lr=parameters["learning_rate"])
    elif parameters["optimizer"] == "rms_prop":
        optimizer = RMSprop(lr=parameters["learning_rate"])
    else:
        optimizer = Adadelta()

    model.compile(optimizer=optimizer, loss='mse')

    return model


def train_network(parameters, data, seed=None, model_path="", model_storage=""):
    assert(isinstance(parameters, dict)),'Parameters should be a dictionary.'
    assert(model_storage in ["", "save", "load"]),'model_storage command not recognized.'

    [[x1t, x1v], [x2t, x2v], [yt, yv]] = data

    rnn_type = parameters['rnn_type']

    if rnn_type == 'none':
        x1t = keras.utils.to_categorical(x1t, 5)
        x1v = keras.utils.to_categorical(x1v, 5)

        x1t = [i.flatten() for i in x1t]
        x1v = [i.flatten() for i in x1v]

        input_eular = Input(shape=(x1t.shape[1],), dtype='float32', name='input_eular')
        input_crp = Input(shape=(1,), dtype='float32', name='input_crp')

    else:
        input_eular = Input(shape=(n_joints, 1), dtype='float32', name='input_eular')
        input_crp = Input(shape=(1,), dtype='float32', name='input_crp')

    cb_lrs = ReduceLROnPlateau()
    cbs = [cb_lrs]

    model = build_model(parameters, seed=seed,rnn_type=rnn_type, input_eular=input_eular, input_crp=input_crp)

    hist = model.fit(
                    x=[x1t, x2t],
                    y=yt,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=False,
                    callbacks=cbs,
                    validation_data=([x1v, x2v], yv)
                    )
    return hist