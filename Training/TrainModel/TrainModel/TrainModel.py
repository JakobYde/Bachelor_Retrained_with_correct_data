import numpy as np
from keras.models import load_model, save_model
import os
import csv
import keras
import sys

sys.path.insert(0, r"../../..")
import Utility.bachelor_utilities as Bu
import Utility.network_training as Tr

parameters = {}
parameters['learning_rate'] = 0.005
parameters['optimizer'] = 'adam'
parameters['activation'] = 'relu'
parameters['dropout'] = 0.015
parameters['rnn_type'] = 'lstm'
parameters['rnn_size'] = 230
parameters['rnn_activation'] = 'tanh'
parameters['rnn_dropout'] = 0.125
parameters['last_activation'] = 'relu'
parameters['dense_layers'] = [139, 486, 152, 79, 61, 0, 0, 0, 0, 0]
#parameters[''] = 

eul, crp, das28 = Bu.load_data('FixedTraining')
eul = eul.reshape(eul.shape[0], eul.shape[1], 1)

crp = (crp - np.mean(crp)) / np.std(crp)

cbs = Tr.get_callbacks()

epochs = 30
batch_size = 32

dir = r'D:\WindowsFolders\Documents\GitHub\BachelorRetraining\Training\TrainModel\TrainModel\models'

input_eular = keras.layers.Input(shape=(eul.shape[1], 1), dtype='float32', name='input_eular')
input_crp = keras.layers.Input(shape=(1,), dtype='float32', name='input_crp')

optimizer = Tr.get_optimizer(parameters['optimizer'], parameters['learning_rate'])

for i, file in enumerate(os.listdir(dir)):
    model = Tr.build_model(input_eular, input_crp, parameters)
    model.load_weights('weights/' + file)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(x=[eul, crp], y=das28, epochs=epochs, verbose=True, batch_size=batch_size)
    save_model(model, r'trained_models/' + file)
