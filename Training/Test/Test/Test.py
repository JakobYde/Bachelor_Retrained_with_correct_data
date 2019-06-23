import numpy as np
from keras.models import load_model
import os
import csv
import keras
import sys

sys.path.insert(0, r"../../..")
import Utility.bachelor_utilities as Bu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def testModel(model, x1, x2, y):
        preds = model.predict([x1, x2])
        preds = preds.reshape(len(x1),)
        diff = preds - y
        return preds, diff

def getCatIndex(ar):
    cats = [(0, 2.6), (2.6, 3.2), (3.2, 5.1), (5.1, 10)]
    res = []
    for v in ar:
        for i in range(0, len(cats)):
            if v >= cats[i][0] and v < cats[i][1]:
                res.append(i)
    return res

eul, crp, das = Bu.load_data('FixedTesting')
eul = eul.reshape(eul.shape[0], eul.shape[1], 1)
_, train_crp, _ = Bu.load_data('FixedTraining')
crp = (crp - np.mean(train_crp)) / np.std(train_crp)

dir=r'D:\WindowsFolders\Documents\GitHub\BachelorRetraining\Training\Test\Test'

all_preds = []
mses = []
maes = []

for i, file in enumerate(os.listdir(dir + r'\\models')):
    model = load_model('models/' + file)
    preds, diff = testModel(model, eul, crp, das)
    all_preds.append(preds)

    mae = np.mean(np.abs(diff))
    maes.append(mae)
    mse = np.mean(np.square(diff))
    mses.append(mse)

preds = np.mean(all_preds, 0)
diff = preds - das

preds_cats = getCatIndex(preds)
das_cats = getCatIndex(das)

count = [0,0,0,0]
for c in das_cats:
    count[c] += 1
max = np.argmax(count)

correct_cats = [das_cats[i] == max for i in range(0, len(preds_cats))]
acc_avg = sum(correct_cats)/len(preds)

correct_cats = [preds_cats[i] == das_cats[i] for i in range(0, len(preds_cats))]
acc = sum(correct_cats)/len(preds)

mae = np.mean(np.abs(diff))
mse = np.mean(np.square(diff))

assert (len(all_preds) > 0),'Missing predictions.'
assert (len(mses) == len(all_preds) and len(maes) == len(all_preds)),'An error occured.'

with open('results.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    header = ['MSE','MAE']
    header += ['pred_{}'.format(n) for n in range(0, len(all_preds[0]))]
    writer.writerow(header)
    writer.writerow([0, 0] + list(das))
    for i in range(0, len(all_preds)):
        to_write = [mses[i], maes[i]]
        to_write += [p for p in all_preds[i]]
        writer.writerow(to_write)
    writer.writerow([])
    to_write = [mse, mae]
    to_write += [p for p in preds]
    writer.writerow(to_write)

pass
