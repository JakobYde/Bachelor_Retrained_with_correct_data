import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
import os
import csv
from BachelorUtilities import loadData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def testModel(model, x1, x2, y):
    preds = model.predict([x1, x2])
    preds = preds.reshape(int(preds.shape[0] / 2), 2, )
    preds = np.mean(preds, 1)
    diff = preds - y
    return preds, diff

_, tr_crp, _ = loadData('Training')
eul, crp, das = loadData('New')

eul = np.reshape(eul, (eul.shape[0], eul.shape[1], 1))

model_path = r'C:\Users\Jakobs PC\Documents\GitHub\BachelorRetraining\Training\TrainModel\TrainModel\models/best_model.h5'

crp = (crp - np.mean(tr_crp)) / np.std(tr_crp)

all_preds = []
mses = []
maes = []

model = load_model(model_path)
preds, diff = testModel(model, eul, crp, das)
all_preds.append(preds)

mae = np.mean(np.abs(diff))
maes.append(mae)
mse = np.mean(np.square(diff))
mses.append(mse)

preds = np.mean(all_preds, 0)
diff = preds - das

mae = np.mean(np.abs(diff))
mse = np.mean(np.square(diff))

print(mae, mse)

with open('results.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    header = ['MSE', 'MAE']
    header += ['pred_{}'.format(n) for n in range(0, len(all_preds[0]))]
    writer.writerow(header)
    for i in range(0, len(all_preds)):
        to_write = [mses[i], maes[i]]
        to_write += [p for p in all_preds[i]]
        writer.writerow(to_write)
    writer.writerow([])
    to_write = [mse, mae]
    to_write += [p for p in preds]
    writer.writerow(to_write)

    to_write = [0, 0]
    to_write += [p for p in das]
    writer.writerow(to_write)
pass