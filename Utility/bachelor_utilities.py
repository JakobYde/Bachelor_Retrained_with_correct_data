import numpy as np
import keras
import csv
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
        eul = np.load(datapath + r"\BachelorFixedData\CorrectTrainEUL.npy")
        eul = np.array(eul, dtype='float32')
        crp = np.load(datapath + r"\BachelorFixedData\CorrectTrainCRP.npy")
        das = np.load(datapath + r"\BachelorFixedData\CorrectTrainDAS28.npy")
    
    if name == 'FixedTesting':
        eul = np.load(datapath + r"\BachelorFixedData\CorrectTestEUL.npy")
        eul = np.array(eul, dtype='float32')
        crp = np.load(datapath + r"\BachelorFixedData\CorrectTestCRP.npy")
        das = np.load(datapath + r"\BachelorFixedData\CorrectTestDAS28.npy")

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

def partition_array(arr, i_partition, n_partitions = 2):
    result = []

    for i, element in enumerate(arr):
        if (i % n_partitions) == i_partition:
            result.append(element)

    return result

def npy_to_csv(npy_folder_path, csv_folder_path):
    assert (isinstance(npy_folder_path, str)), "npy_folder_path should be a string containing a path"
    assert (isinstance(csv_folder_path, str)), "csv_folder_path should be a string containing a path"

    x_training_eul = np.load(npy_folder_path + "CorrectTrainEUL.npy")
    x_test_eul = np.load(npy_folder_path + "CorrectTestEUL.npy")
    x_training_crp = np.load(npy_folder_path + "CorrectTrainCRP.npy")
    x_test_crp = np.load(npy_folder_path + "CorrectTestCRP.npy")
    y_training = np.load(npy_folder_path + "CorrectTrainDAS28.npy")
    y_test = np.load(npy_folder_path + "CorrectTestDAS28.npy")

    np.savetxt(csv_folder_path + "TrainingDataEUL.csv", x_training_eul, delimiter=",")
    np.savetxt(csv_folder_path + "TestingDataEUL.csv", x_test_eul, delimiter=",")
    np.savetxt(csv_folder_path + "TrainingDataCRP.csv", x_training_crp, delimiter=",")
    np.savetxt(csv_folder_path + "TestingDataCRP.csv", x_test_crp, delimiter=",")
    np.savetxt(csv_folder_path + "TrainingDataY.csv", y_training, delimiter=",")
    np.savetxt(csv_folder_path + "TestingDataY.csv", y_test, delimiter=",")

class CSVWriter:
    def __init__(self, filename, head=[], dialect=None, clear_on_creation=True):
        assert isinstance(filename, str),'File name should be a string.'
        assert isinstance(clear_on_creation,bool),'clear_on_creation not bool.'

        if dialect != None:
            csv.register_dialect('dial',dialect)

        self.filename = filename
        self.dialect = (dialect != None)

        if clear_on_creation:

            with open(self.filename, 'w') as f:
                if self.dialect:
                    writer = csv.writer(f,'dial')
                else:
                    writer = csv.writer(f)
                if head != []:
                    assert (isinstance(head, list)),'head should be a list of printable elements.'
                    writer.writerow(head)

    def write_row(self, row):

        with open(self.filename, 'a') as f:
            if self.dialect:
                writer = csv.writer(f,'dial')
            else:
                writer = csv.writer(f)

            writer.writerow(row)
