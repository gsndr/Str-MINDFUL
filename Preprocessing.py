import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder


class Preprocessing():

    def __init__(self, train, test):
        # classification column
        # print(train.columns)

        self._clsTrain = train.columns.values[-1]
        #print(self._clsTrain)
        if test is not None:
            self._clsTest = test.columns.values[-1]
        else:
            self._clsTest = None
            # only categorical features
        '''
        obj_dfTrain = train.select_dtypes(include=['object']).copy()
        self._objectListTrain = obj_dfTrain.columns.values
        print(self._objectListTrain)
        #remove classification column
        self._objectListTrain = np.delete(self._objectListTrain, -1)
        #classification test column
        

        # ronly categorical features test set
        obj_dfTest = test.select_dtypes(include=['object']).copy()
        self._objectListTest= obj_dfTest.columns.values

       #remove classification column from test set
        self._objectListTest = np.delete(self._objectListTest, -1)
        '''

    def getCls(self):
        return self._clsTrain, self._clsTest

    def getStreamCls(self):
        return self._clsTrain

    def labelCategorical(self, train, test):
        le = LabelEncoder()
        train_df = train.copy()
        test_df = test.copy()
        AllT = train_df.append(test_df, ignore_index=True)
        for col in self._objectListTrain:
            le.fit(AllT[col])
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
        return train, test

    def mapLabel(self, df):
        # creating labelEncoder
        le = preprocessing.LabelEncoder()
        cls_encoded = le.fit_transform(df[self._clsTrain])
        df[self._clsTrain] = le.transform(df[self._clsTrain])
        return cls_encoded

    # map label classification to number
    def preprocessinLabel(self, train, test):
        self.mapLabel(train)
        self.mapLabel(test)
        return train, test

    def preprocessingOH(self, train, test):
        # print(self._objectListTrain)
        train = pd.get_dummies(train, columns=self._objectListTrain)
        # print(self._objectListTest)
        test = pd.get_dummies(test, columns=self._objectListTest)
        return train, test

    def getXY(self, data):
        clssList = data.columns.values
        target = [i for i in clssList if i.startswith(self._clsTrain)]

        # remove label from dataset to create Y ds
        data_Y = data[target]
        # remove label from dataset
        data_X = data.drop(target, axis=1)
        data_X = data_X.values

        return data_X, data_Y


