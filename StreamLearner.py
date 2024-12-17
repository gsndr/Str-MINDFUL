import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from Preprocessing import Preprocessing
from CNN1D import createImage
from keras.utils import np_utils
from keras import callbacks

np.random.seed(10000)

"""
Convert a time delta in a more readable form, that will be saved in the output log file

Parameters
----------
:param before:
lower extreme of the time laps
:param after:
higher extreme of the time laps

Returns
----------
:returns:
a string that show the time laps between the parameters
"""
def convertTimeDelta(before, after):
    tot = after - before
    d = np.timedelta64(tot, 'D')
    tot -= d
    h = np.timedelta64(tot, 'h')
    tot -= h
    m = np.timedelta64(tot, 'm')
    tot -= m
    s = np.timedelta64(tot, 's')
    tot -= s
    ms = np.timedelta64(tot, 'ms')
    out = str(d) + " " + str(h) + " " + str(m) + " " + str(s) + " " + str(ms)
    return out

"""
Preprocess the dataset. Function selects examples according to classLabel: if the classLabel is 'Normal', function returns
normal examples' subset; if the classLabel is 'Attack', function returns attack examples' subset.

Parameters
----------
:param dataset:
batch input dataset
:param classLabel:
label that identifies the examples to be obtained. Values in ['Normal, 'Attack']

Returns
----------
:returns dataX:
examples' train subset
:returns dataY:
labels of the examples contained in dataX
"""
def preprocessAutoencoderDataset(dataset, classLabel):
    prp = Preprocessing(dataset, None)
    clsT = prp.getStreamCls()
    if classLabel == 'Normal':
        train_normal = dataset[(dataset[clsT] == 1)]
        dataX, dataY = prp.getXY(train_normal)
        print('Train data shape normal', dataX.shape)
        print('Train target shape normal', dataY.shape)
    else:
        train_anormal = dataset[(dataset[clsT] == 0)]
        dataX, dataY = prp.getXY(train_anormal)
        print('Train data shape anormal', dataX.shape)
        print('Train target shape anormal', dataY.shape)
    return dataX, dataY

"""
Create the multi-channel image to train the CNN1D

Parameters
----------
:param dataset:
batch input dataset
:param autoencoderN:
autoencoder trained on normal examples
:param autoencoderA:
autoencoder trained on attack examples
:param n_classes:
number of output classes

Returns
----------
:returns imageX:
multi channel image that will be used to train CNN1D
:returns dataY:
labels of the examples of imageX
:returns dataY2:
categorical labels of the examples of imageX that will be used to train CNN1D
"""
def preprocessClassifierDataset(dataset, autoencoderN, autoencoderA, n_classes):
    prp = Preprocessing(dataset, None)
    dataX, dataY = prp.getXY(dataset)
    dataXN = autoencoderN.predict(dataX)
    dataXA = autoencoderA.predict(dataX)
    imageX, inputshape = createImage(dataX, dataXN, dataXA)
    dataY2 = np_utils.to_categorical(dataY, n_classes)
    print('Train data shape for classifier', dataX.shape)
    print('Train target shape for classifier', dataY.shape)
    print('Train target shape for classifier after categorization', dataY2.shape)
    return imageX, dataY, dataY2


class StreamLearner:
    """
    Class that contains all the functions useful to retrain models

    Parameters
    ----------
    :param dsConf:
    input dataset configuration
    :param outFile:
    output log file reference
    """

    def __init__(self, dsConf):
        self.dsConf = dsConf
        self.batchSizeN = self.getBatchSize('Normal')
        self.batchSizeA = self.getBatchSize('Attacks')
        self.batchSizeC = self.getBatchSize('Classifier')
        print("Batch size model:\nNormal: ", self.batchSizeN, "\nAttack: ", self.batchSizeA, "\nClassifier: ",
              self.batchSizeC, "\n")

    """
    Loads the batch dataset from disk
    
    Returns
    ----------
    :returns:
    loaded batch dataset
    """
    def loadDataset(self):
        return pd.read_csv(self.dsConf.get('pathDatasetNumeric') + self.dsConf.get('path') + 'Numeric.csv',
                           encoding='cp1252')

    """
    Saves the updated dataset on disk
    """
    def saveDataset(self, dataset):
        dataset.to_csv(self.dsConf.get('pathDatasetNumeric') + self.dsConf.get('path') + 'Numeric.csv',
                       encoding='cp1252', index=False)
        print("Dataset saved!")

    """
    Updates batch dataset with the examples stored in the sliding window. According to classLabel, attack or normal examples
    are casually selected from related subset of original dataset to be replaced with window's examples.
    
    Parameters
    ----------
    :param dataset:
    original batch dataset
    :param window:
    sliding window in which the new examples are stored
    :param classLabel:
    label that select the subset of examples to update. Value in ['Attack', 'Normal']
    
    Returns
    :returns:
    updated dataset
    """
    def updateTrainingSet(self, dataset, window, classLabel):
        print("Updating dataset..")
        prp = Preprocessing(dataset, None)
        clsT = prp.getStreamCls()
        dataN = dataset.loc[dataset[clsT] == 1]
        dataA = dataset.loc[dataset[clsT] == 0]
        remove_n = window.length()
        if classLabel == 'Normal':
            if remove_n > len(dataN):
                remove_n = len(dataN)
            drop_indices = np.random.choice(dataN.index, remove_n, replace=False)
            dataN = dataN.drop(drop_indices)
        else:
            if remove_n > len(dataA):
                remove_n = len(dataA)
            drop_indices = np.random.choice(dataA.index, remove_n, replace=False)
            dataA = dataA.drop(drop_indices)
        newExamples = window.getWindow(dataN.columns).tail(remove_n)
        data = dataN.append(dataA)
        dataset = data.append(newExamples)
        return dataset

    """
    Retrains the selected autoencoder with the upgraded batch dataset
    
    Parameters
    ----------
    :param autoencoder:
    autoencoder that will be retrained
    :param dataX:
    subset of batch dataset examples' that will be used to train the autoencoder
    :param dataY:
    label of dataX's examples
    :param classLabel:
    class label to identify which model will be trained. Function need this param to select the right batch size
    value in function autoencoder.fit()
    
    Returns
    ----------
    :returns:
    upgraded autoencoder
    """
    def updateAutoencoder(self, autoencoder, dataX, dataY, classLabel):
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                    restore_best_weights=True),
        ]

        XTraining, XValidation, YTraining, YValidation = train_test_split(dataX, dataY, stratify=dataY, test_size=0.2)
        if classLabel == 'Normal':
            batch=int(self.batchSizeN)
            autoencoder.fit(XTraining, XTraining, validation_data=(XValidation, XValidation),
                            batch_size=batch, epochs=150,callbacks=callbacks_list)
        else:
            autoencoder.fit(XTraining, XTraining, validation_data=(XValidation, XValidation),
                            batch_size=self.batchSizeA, epochs=150,callbacks=callbacks_list)
        autoencoder.save(self.dsConf.get('pathModels') + 'autoencoder' + classLabel + '.h5')
        return autoencoder

    """
    Retrain CNN1D with the upgraded batch dataset
    
    Parameters
    ----------
    :param CNN1D:
    model to upgrade
    :param imageX:
    multichannel image made by the upgraded batch dataset
    :param dataY:
    categorical labels of the examples in imageX
    
    Returns
    ----------
    :returns:
    upgraded CNN1D
    """
    def updateCNN1D(self, CNN1D, imageX, dataY):
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                    restore_best_weights=True),
        ]
        XTraining, XValidation, YTraining, YValidation = train_test_split(imageX, dataY, stratify=dataY, test_size=0.2)
        CNN1D.fit(XTraining, YTraining, validation_data=(XValidation, YValidation), batch_size=self.batchSizeC, epochs=150,callbacks=callbacks_list)
        CNN1D.save(self.dsConf.get('pathModels') + 'MINDFUL.h5')
        return CNN1D

    """
    Computes the reconstruction errors made by an autoencoder
    
    Parameters
    ----------
    :param autoencoder:
    autoencoder with which to compute the errors
    :param dataX:
    subset of attack or normal examples that will be rebuilt by autoencoder
    :param classLabel:
    classLabel of the autoencoder
    
    Returns
    ----------
    :returns:
    Reconstruction errors made by autoencoder
    """
    def getAutoencoderErrors(self, autoencoder, dataX, classLabel):
        prediction = autoencoder.predict(dataX)
        mse = np.mean(np.power(dataX - prediction, 2), axis=1)
        print(classLabel, " Autoencoder Errors: ", mse)
        print(len(mse), classLabel, " examples computed")
        # np.save(self.dsConf.get('pathUtils') + label + 'Errors.npy', mse)
        return mse

    """
    Computes the prediction errors made by CNN1D
    
    Parameters
    ----------
    :param CNN1D:
    CNN1D with which to compute the errors
    :param dataX:
    multi-channel examples that will be predicted by CNN1D
    :param dataY:
    labels of the examples in dataX
    
    Returns
    ----------
    :returns:
    Prediction errors made by CNN1D
    """
    def getCNN1DErrors(self, CNN1D, dataX, dataY):
        predictionErrors = []
        prediction = CNN1D.predict(dataX)
        prediction = np.argmax(prediction, axis=1)
        dataY = dataY.to_numpy()
        for i in range(0, len(dataY)):
            predictionErrors.append(abs(dataY[i] - prediction[i])[0])
        print(len(predictionErrors), " Examples computed")
        # np.save(self.ds.get('pathUtils') + 'PredictionErrors.npy', np.asarray(predictionErrors))
        return np.asarray(predictionErrors)

    """
    This function is used before the stream start. It selects the batch size values by the results csv file that will be 
    used to retrain models. Batch size values is selected by found the greatest value of OA_VAL for the classifier and
    lowest value of loss function for the autoencoder
    
    Parameters
    ----------
    :param classLabel:
    label that specifies which model to extract the batch size value for
    """
    def getBatchSize(self, classLabel):
        if classLabel == 'Classifier':
            data = pd.read_csv('results/' + self.dsConf.get('testPath') + 'MINDFUL.csv', encoding='cp1252')
            return int(data.iloc[data[' OA_VAL'].idxmax()][6])
        else:
            data = pd.read_csv('results/' + self.dsConf.get('testPath') + classLabel + 'Autoencoder.csv',
                               encoding='cp1252')
            return int(data.iloc[data[' loss '].idxmin()][5])
