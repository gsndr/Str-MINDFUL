import numpy as np

my_seed = 12
np.random.seed(my_seed)
import random

random.seed(my_seed)

import tensorflow

tensorflow.random.set_seed(12)

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from DatasetsConfig import Datasets
from Preprocessing import Preprocessing as prep
from Plot import Plot
from keras import callbacks
from keras.utils import np_utils

from sklearn.metrics import confusion_matrix
from keras.models import Model

from tensorflow.keras.models import load_model
from keras import backend as K
from keras.utils import plot_model

np.set_printoptions(suppress=True)
import AutoencoderHypersearch as ah
import CNNHypersearch as ch


def createImage(train_X, trainN, trainA):
    rows = [train_X, trainN, trainA]
    rows = [list(i) for i in zip(*rows)]

    train_X = np.array(rows)

    if K.image_data_format() == 'channels_first':
        x_train = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2])
        input_shape = (train_X.shape[1], train_X.shape[2])
    else:
        x_train = train_X.reshape(train_X.shape[0], train_X.shape[2], train_X.shape[1])
        input_shape = (train_X.shape[2], train_X.shape[1])
    return x_train, input_shape

def getResult(cm, N_CLASSES):
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / N_CLASSES
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = (tp, fn, fp, tn, OA, AA, P, R, F1, FAR, TPR)
    return r


def getAutoencoders(dsConfig):
    autoencoderN = load_model(dsConfig.get('pathModels') + 'autoencoderNormal.h5')
    autoencoderA = load_model(dsConfig.get('pathModels') + 'autoencoderAttacks.h5')
    return autoencoderN, autoencoderA


def getCNN1D(dsConfig):
    CNN1D = load_model(dsConfig.get('pathModels') + 'MINDFUL.h5')
    return CNN1D


class RunCNN1D():
    def __init__(self, dsConfig, config):
        self.config = config
        self.ds = dsConfig
        fileOutput = self.ds.get('pathModels') + 'result' + self.ds.get('testPath') + '.txt'
        self.file = open(fileOutput, 'w')
        self.file.write('Result time for: \n')
        self.file.write('\n')

    def run(self):

        print('MINDFUL EXECUTION')

        dsConf = self.ds
        pathModels = dsConf.get('pathModels')
        pathPlot = dsConf.get('pathPlot')
        configuration = self.config

        VALIDATION_SPLIT = float(configuration.get('VALIDATION_SPLIT'))
        N_CLASSES = int(configuration.get('N_CLASSES'))
        pd.set_option('display.expand_frame_repr', False)

        # contains path of dataset and model and preprocessing phases
        ds = Datasets(dsConf, configuration)
        PREPROCESSING1 = int(configuration.get('PREPROCESSING1'))
        if PREPROCESSING1 == 1:
            ds.preprocessing1()
            train, test = ds.getTrain_Test()
        else:
            train, test = ds.getNumericDatasets()

        print(test)
        prp = prep(train, test)
        train = train.replace(np.nan, 0)

        clsT, clsTest = prp.getCls()
        train_normal = train[(train[clsT] == 1)]
        train_anormal = train[(train[clsT] == 0)]
        test_normal = test[(test[clsTest] == 1)]
        test_anormal = test[(test[clsTest] == 0)]

        train_XN, train_YN = prp.getXY(train_normal)
        test_XN, test_YN = prp.getXY(test_normal)

        train_XA, train_YA, = prp.getXY(train_anormal)
        test_XA, test_YA = prp.getXY(test_anormal)

        train_X, train_Y = prp.getXY(train)
        test_X, test_Y = prp.getXY(test)

        print("Nan" + str(np.any(np.isnan(train_XN))))

        print('Train data shape normal', train_XN.shape)
        print('Train target shape normal', train_YN.shape)
        print('Test data shape normal', test_XN.shape)
        print('Test target shape normal', test_YN.shape)

        print('Train data shape anormal', train_XA.shape)
        print('Train target shape anormal', train_YA.shape)
        print('Test data shape anormal', test_XA.shape)
        print('Test target shape anormal', test_YA.shape)

        # convert class vectors to binary class matrices fo softmax
        train_Y2 = np_utils.to_categorical(train_Y, int(configuration.get('N_CLASSES')))
        print("Target train shape after", train_Y2.shape)
        test_Y2 = np_utils.to_categorical(test_Y, int(configuration.get('N_CLASSES')))
        print("Target test shape after", test_Y2.shape)
        print("Train all", train_X.shape)
        print("Test all", test_X.shape)

        # create pandas for results
        columns = ['TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
        results = pd.DataFrame(columns=columns)

        if (int(configuration.get('LOAD_AUTOENCODER_NORMAL')) == 0):

            autoencoderN, best_time, encoder = ah.hypersearch(train_XN, train_YN, test_XN, test_YN,
                                                              pathModels + 'autoencoderNormal.h5',
                                                              'results/' + self.ds.get('testPath') + 'Normal')

            self.file.write("Time Training Autoencoder Normal: %s" % best_time)
            self.file.write('\n')

        else:
            print("Load autoencoder Normal from disk")
            autoencoderN = load_model(pathModels + 'autoencoderNormal.h5')
            autoencoderN.summary()

        train_RE = autoencoderN.predict(train_X)
        test_RE = autoencoderN.predict(test_X)

        if (int(configuration.get('LOAD_AUTOENCODER_ADV')) == 0):

            autoencoderA, best_time, encoder = ah.hypersearch(train_XA, train_YA, test_XA, test_YA,
                                                              pathModels + 'autoencoderAttacks.h5',
                                                              'results/' + self.ds.get('testPath') + 'Attacks')
            self.file.write("Time Training Autoencoder Attacks: %s" % best_time)
            self.file.write('\n')


        else:
            print("Load autoencoder Attacks from disk")
            autoencoderA = load_model(pathModels + 'autoencoderAttacks.h5')
            autoencoderA.summary()

        train_REA = autoencoderA.predict(train_X)
        test_REA = autoencoderA.predict(test_X)

        train_X_image, input_Shape = createImage(train_X, train_RE, train_REA)  # XS UNSW
        test_X_image, input_shape = createImage(test_X, test_RE, test_REA)

        if (int(configuration.get('LOAD_CNN')) == 0):
            model, best_time, = ch.hypersearch(train_X_image, train_Y, test_X_image, test_Y,
                                               pathModels + 'MINDFUL.h5', 'results/' + self.ds.get('testPath'))
            self.file.write("Time Training CNN: %s" % best_time)
            self.file.write('\n')

        else:
            print("Load softmax from disk")
            model = load_model(pathModels + 'MINDFUL.h5')
            model.summary()

        predictionsL = model.predict(train_X_image)
        y_pred = np.argmax(predictionsL, axis=1)
        cmC = confusion_matrix(train_Y, y_pred)
        print('Prediction Training')
        print(cmC)

        predictionsL = model.predict(test_X_image)
        y_pred_test = np.argmax(predictionsL, axis=1)
        cm = confusion_matrix(test_Y, y_pred_test)
        print('Prediction Test')
        print(cm)

        r = getResult(cm, N_CLASSES)

        dfResults = pd.DataFrame([r], columns=columns)
        print(dfResults)

        results = results.append(dfResults, ignore_index=True)

        results.to_csv('results/' + ds._testpath + '_results.csv', index=False)

        print("Saving prediction errors")
        prediction_normal = autoencoderN.predict(train_XN)
        mse = np.mean(np.power(train_XN - prediction_normal, 2), axis=1)
        print("Normal Autoencoder Errors: ", mse)
        print(len(mse), " Normal examples computed")
        np.save(self.ds.get('pathUtils') + 'NormalErrors.npy', mse)

        prediction_attack = autoencoderA.predict(train_XA)
        mse = np.mean(np.power(train_XA - prediction_attack, 2), axis=1)
        print("Attack Autoencoder Errors: ", mse)
        print(len(mse), " Attack examples computed")
        np.save(self.ds.get('pathUtils') + 'AttackErrors.npy', mse)

        predictionErrors = []
        arrayTrain_y = train_Y.to_numpy()
        for i in range(0, (len(train_Y))):
            predictionErrors.append(abs(arrayTrain_y[i] - y_pred[i])[0])
        print(len(predictionErrors), " Examples computed")
        np.save(self.ds.get('pathUtils') + 'ClassifierErrors.npy', np.asarray(predictionErrors))

        print("Saving normal and predicted Labels")
        np.save(self.ds.get('pathUtils') + 'Batch_Original_Label.npy', test_Y.to_numpy())
        np.save(self.ds.get('pathUtils') + 'Batch_Predicted_Label.npy', y_pred_test)
