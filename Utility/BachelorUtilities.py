import numpy as np
import keras

# Training - training set
# Testing - testing set
# New - new testing set
# FixedTraining - training set with no incorrect das28 values
# FixedTesting - testing set like above
def loadData(name, categorical=False):
    assert (name in ['Training', 'Testing', 'New', 'FixedTraining', 'FixedTesting']),'Name not recognised.'

    if name == 'Training':
        eul = np.load(r"D:\WindowsFolders\Code\Data\BachelorOriginalData\TrainingDataEUL.npy")
        eul = np.array(eul, dtype='float32')
        crp = np.load(r"D:\WindowsFolders\Code\Data\BachelorOriginalData\TrainingDataCRP.npy")
        das = np.load(r"D:\WindowsFolders\Code\Data\BachelorOriginalData\TrainingDataY.npy")

    if name == 'Testing':
        eul = np.load(r"D:\WindowsFolders\Code\Data\BachelorOriginalData\TestingDataEUL.npy")
        eul = np.array(eul, dtype='float32')
        crp = np.load(r"D:\WindowsFolders\Code\Data\BachelorOriginalData\TestingDataCRP.npy")
        das = np.load(r"D:\WindowsFolders\Code\Data\BachelorOriginalData\TestingDataY.npy")
    
    if name == 'New':
        eul = np.load(r"D:\WindowsFolders\Code\Data\BachelorOriginalData\NewDataEUL.npy")
        eul = np.array(eul, dtype='float32')
        crp = np.load(r"D:\WindowsFolders\Code\Data\BachelorOriginalData\NewDataCRP.npy")
        das = np.load(r"D:\WindowsFolders\Code\Data\BachelorOriginalData\NewDataDAS.npy")
    
    if name == 'FixedTraining':
        eul = np.load(r"D:\WindowsFolders\Code\Data\BachelorFixedData\TrainingDataEUL.npy")
        eul = np.array(eul, dtype='float32')
        crp = np.load(r"D:\WindowsFolders\Code\Data\BachelorFixedData\TrainingDataCRP.npy")
        das = np.load(r"D:\WindowsFolders\Code\Data\BachelorFixedData\TrainingDataY.npy")
    
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