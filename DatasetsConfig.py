import pandas as pd
import numpy as np
from Preprocessing import Preprocessing


class Datasets():
    def __init__(self, dsConf, configuration):

        # classification column
        self.pathModels = dsConf.get('pathModels')
        self._pathDataset = dsConf.get('pathDataset')
        self._path = dsConf.get('path')
        self._streampath = dsConf.get('pathStream')
        self._testpath = dsConf.get('testpath')
        self._pathDatasetNumeric = dsConf.get('pathDatasetNumeric')
        self._pathUtils = dsConf.get('pathUtils')
        self._pathTest = dsConf.get('pathTest')
        self._newSize = int(dsConf.get('smallerSize'))

        if int(configuration.get('EXECUTION_TYPE')) == 0:
            self._train = pd.read_csv(self._pathDataset + self._path + ".csv", encoding='cp1252')

            self._pathOutputTrain = self._pathDatasetNumeric + self._path + 'Numeric.csv'

        else:
            self._stream = pd.read_csv(self._pathDataset + self._streampath + ".csv", encoding='cp1252')

            self._pathOutputTrain = self._pathDatasetNumeric + self._streampath + 'Numeric.csv'

    def getTrain_Test(self):
        return self._train, self._test

    def getStreamDataset(self):
        return self._stream

    def getNumericDatasets(self):
        train = pd.read_csv(self._pathDatasetNumeric + self._path + "Numeric.csv", encoding='cp1252')
        test = pd.read_csv(self._pathDatasetNumeric + self._pathTest + "Numeric.csv", encoding='cp1252')
        return train, test

    def getStreamNumericDataset(self):
        return pd.read_csv(self._pathOutputTrain, encoding='cp1252')

    def getTimestampValues(self):
        return np.load(self._pathDatasetNumeric + "Timestamp values.npy")

    """
    If asked in the configuration file, this function reduces the size of the batch dataset
    to the value specified in 'smallerSize' in MINDFUL.conf.
    System load the batch dataset, reduces it stratifying like the original example's stratification,
    and saves it in a new csv file, that will be used during the stream program.
    """
    def setSmallerBatchDataset(self):
        data = pd.read_csv(self._pathDatasetNumeric + self._path + "Numeric.csv", encoding='cp1252')
        prp = Preprocessing(data, None)
        clsT = prp.getStreamCls()
        dataN = data.loc[data[clsT] == 1]
        dataA = data.loc[data[clsT] == 0]
        newNormalSize = int((len(dataN) * self._newSize) / len(data))
        newAttackSize = int((len(dataA) * self._newSize) / len(data)) + 1
        normal_indices = np.random.choice(dataN.index, newNormalSize, replace=False)
        attack_indices = np.random.choice(dataA.index, newAttackSize, replace=False)
        dataN = dataN[dataN.index.isin(normal_indices)]
        dataA = dataA[dataA.index.isin(attack_indices)]
        data = pd.concat([dataN, dataA])
        data.to_csv(self._pathDatasetNumeric + self._path + "Numeric.csv", encoding='cp1252', index=False)

    def preprocessing1(self):

        if ((self._testpath == 'KDDCUP')):
            self._test = pd.read_csv(self._pathDataset + self._pathTest + ".csv")
            print('Using:' + self._testpath)
            self._listNumerical10 = self._train.columns.values
            index = np.argwhere(self._listNumerical10 == ' classification.')
            self._listNumerical10 = np.delete(self._listNumerical10, index)
            # print(self._listNumerical10)

        elif (self._testpath == 'UNSW_NB15'):
            print('Using:' + self._testpath)
            self._test = pd.read_csv(self._pathDataset + self._pathTest + ".csv")
            cls = ['classification']
            listCategorical = ['proto', 'service', 'state']
            listBinary = ['is_ftp_login', 'is_sm_ips_ports']
            listAllColumns = self._train.columns.values
            self._listNumerical10 = list(set(listAllColumns) - set(listCategorical) - set(listBinary) - set(cls))

        elif (self._testpath == 'AAGM'):
            print('Using:' + self._testpath)
            self._test = pd.read_csv(self._pathDataset + self._pathTest + ".csv")
            cls = ['classification']
            listCategorical = []
            listBinary = []
            listAllColumns = self._train.columns.values
            self._listNumerical10 = list(set(listAllColumns) - set(listCategorical) - set(listBinary) - set(cls))


        elif (self._testpath == 'CICIDS2017'):
            print('Using:' + self._testpath)
            self._train.rename(
                columns={' Flow Packets/s': 'Flow_Packets', 'Flow Bytes/s': 'Flow_Bytes', ' Label': 'Classification',
                         'Fwd Packets/s': 'Fwd_Packets', ' Bwd Packets/s': 'Bwd_Packets'},
                inplace=True)

            cls = ['Classification']
            label = ['BENIGN', 'ATTACK']
            # listCategorical = ['Flow ID, Source IP, Destination IP, Timestamp, External IP']
            self._train = self._train.drop(['Flow ID', ' Source IP', ' Destination IP', ' Source Port', ' Protocol'], axis=1)
            self._train = self._train[self._train['Classification'].notna()]
            self._train = self._train.replace([np.inf, -np.inf], np.nan)
            self._train = self._train[self._train['Flow_Bytes'].notna()]
            self._train = self._train[self._train['Fwd_Packets'].notna()]
            print(self._train.columns)

            allowed_val = ['BENIGN']
            self._train.loc[~self._train['Classification'].isin(allowed_val), 'Classification'] = 'ATTACK'
            self._train["Classification"].replace(to_replace=dict(BENIGN=1, ATTACK=0), inplace=True)

            self._train[' Timestamp'] = pd.to_datetime(self._train[' Timestamp'], dayfirst=True)
            self._train = self._train.sort_values(by=' Timestamp')
            self._train = self._train.drop(' Timestamp', axis=1)

            col_names = self._train.columns

            nominal_inx = []
            binary_inx = [
                'Fwd PSH Flags,  Bwd PSH Flags,  Fwd URG Flags,  Bwd URG Flags, FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count,'
                'ACK Flag Count	,URG Flag Count, CWE Flag Count,  ECE Flag Count,  Fwd Header Length.1,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk,'
                ' Fwd Avg Bulk Rate	 Bwd Avg Bytes/Bulk	 Bwd Avg Packets/Bulk	Bwd Avg Bulk Rate']

            binary_inx = [30, 31, 32, 33, 43, 44, 45, 46, 47, 48, 49, 50, 56, 57, 58, 59, 60, 61]
            numeric_inx = list(set(range(78)).difference(nominal_inx).difference(binary_inx))

            self._listNumerical10 = col_names[numeric_inx].tolist()

            testset = self._pathTest
            test = pd.read_csv(self._pathDataset + testset + ".csv", encoding='cp1252')
            test.rename(
                columns={' Flow Packets/s': 'Flow_Packets', 'Flow Bytes/s': 'Flow_Bytes', ' Label': 'Classification',
                         'Fwd Packets/s': 'Fwd_Packets', ' Bwd Packets/s': 'Bwd_Packets'},
                inplace=True)
            test = test.drop(['Flow ID', ' Source IP', ' Destination IP', ' Source Port', ' Protocol'], axis=1)
            test = test[test['Classification'].notna()]
            test = test.replace([np.inf, -np.inf], np.nan)
            test = test[test['Flow_Bytes'].notna()]
            test = test[test['Fwd_Packets'].notna()]

            allowed_val = ['BENIGN']
            test.loc[~test['Classification'].isin(allowed_val), 'Classification'] = 'ATTACK'
            test["Classification"].replace(to_replace=dict(BENIGN=1, ATTACK=0), inplace=True)

            test[' Timestamp'] = pd.to_datetime(test[' Timestamp'], dayfirst=True)
            test = test.sort_values(by=' Timestamp')
            test = test.drop(' Timestamp', axis=1)
            self._test = test

            col_names = col_names.delete(-1)
            col_names = col_names.values.tolist()
            print(col_names)
            self._train[col_names] = self._train[col_names].apply(pd.to_numeric, errors='coerce')
            self._test[col_names] = self._test[col_names].apply(pd.to_numeric, errors='coerce')

            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
            print('Scaling')
            listContent = self._listNumerical10
            print(self._train[listContent].values)
            scaler.fit(self._train[listContent])
            # scaler.fit(train[listContent].values)
            self._train[listContent] = scaler.transform(self._train[listContent])
            self._test[listContent] = scaler.transform(self._test[listContent])
            self._pathOutputTest = self._pathDatasetNumeric + self._pathTest + 'Numeric.csv'
            self._train.to_csv(self._pathOutputTrain, index=False)  # X is an array
            self._test.to_csv(self._pathOutputTest, index=False)
            self._clsTrain = self._train.columns[-1]
            self._clsTest = self._test.columns[-1]

    """
    Same preprocessing of the function preprocessing1, but made on a single train set that
    will be used as stream dataset.
    Function saves the TimeStamp column of the .npy file that will be used by stream program
    to know the time value of examples
    """
    def preprocessing2(self):
        print('Using:' + self._testpath)
        self._stream.rename(
            columns={' Flow Packets/s': 'Flow_Packets', 'Flow Bytes/s': 'Flow_Bytes', ' Label': 'Classification',
                     'Fwd Packets/s': 'Fwd_Packets', ' Bwd Packets/s': 'Bwd_Packets'},
            inplace=True)

        cls = ['Classification']
        label = ['BENIGN', 'ATTACK']
        # listCategorical = ['Flow ID, Source IP, Destination IP, Timestamp, External IP']
        self._stream = self._stream.drop(['Flow ID', ' Source IP', ' Destination IP', ' Source Port', ' Protocol'], axis=1)
        self._stream = self._stream[self._stream['Classification'].notna()]
        self._stream = self._stream.replace([np.inf, -np.inf], np.nan)
        self._stream = self._stream[self._stream['Flow_Bytes'].notna()]
        self._stream = self._stream[self._stream['Fwd_Packets'].notna()]
        print(self._stream.columns)

        allowed_val = ['BENIGN']
        self._stream.loc[~self._stream['Classification'].isin(allowed_val), 'Classification'] = 'ATTACK'
        self._stream["Classification"].replace(to_replace=dict(BENIGN=1, ATTACK=0), inplace=True)

        self._stream[' Timestamp'] = pd.to_datetime(self._stream[' Timestamp'], dayfirst=True)
        self._stream = self._stream.sort_values(by=' Timestamp')
        timestamp = np.asarray(self._stream[' Timestamp'])
        np.save((self._pathUtils + "Timestamp values.npy"), timestamp)
        self._stream = self._stream.drop([' Timestamp'], axis=1)

        col_names = self._stream.columns

        nominal_inx = []
        binary_inx = [
            'Fwd PSH Flags,  Bwd PSH Flags,  Fwd URG Flags,  Bwd URG Flags, FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count,'
            'ACK Flag Count	,URG Flag Count, CWE Flag Count,  ECE Flag Count,  Fwd Header Length.1,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk,'
            ' Fwd Avg Bulk Rate	 Bwd Avg Bytes/Bulk	 Bwd Avg Packets/Bulk	Bwd Avg Bulk Rate']

        binary_inx = [30, 31, 32, 33, 43, 44, 45, 46, 47, 48, 49, 50, 56, 57, 58, 59, 60, 61]
        numeric_inx = list(set(range(78)).difference(nominal_inx).difference(binary_inx))

        self._listNumerical10 = col_names[numeric_inx].tolist()

        col_names = col_names.delete(-1)
        col_names = col_names.values.tolist()
        print(col_names)
        self._stream[col_names] = self._stream[col_names].apply(pd.to_numeric, errors='coerce')

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        print('Scaling')
        listContent = self._listNumerical10
        print(self._stream[listContent].values)
        scaler.fit(self._stream[listContent])
        # scaler.fit(train[listContent].values)
        self._stream[listContent] = scaler.transform(self._stream[listContent])
        self._pathOutputTest = self._pathDatasetNumeric + self._pathTest + 'Numeric.csv'
        self._stream.to_csv(self._pathOutputTrain, index=False)  # X is an array
        self._clsTrain = self._stream.columns[-1]
