import numpy as np
import keras
from os import getenv


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