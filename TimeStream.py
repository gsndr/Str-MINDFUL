from skmultiflow.data import DataStream
from Preprocessing import Preprocessing as prep
from DatasetsConfig import Datasets
import numpy as np


class TimeStream:
    """
    This class simulates a stream. It uses skmultiflow.data.Datastream that take a pandas DataFrame in input and builds
    a dataStream. In the constructor, the input stream dataset performs a preprocessing phase and then is transformed in
    a stream order by the timesramp values of the examples. Furthermore, if asked into the configuration file, the batch
    dataset size is decreased by the function DatasetConfig.setSmallerBatchDataset()

    Parameters
    ----------
    :param dsConf:
    dataset configuration
    :param conf:
    execution settings configuration
    """

    def __init__(self, dsConf, conf):
        self.timeID = 0

        ds = Datasets(dsConf, conf)
        PREPROCESSING1 = int(conf.get('PREPROCESSING1'))
        if PREPROCESSING1 == 1:
            print("Load and Preprocess dataset...")
            ds.preprocessing2()
            data = ds.getStreamDataset()
        else:
            print("Load dataset...")
            data = ds.getStreamNumericDataset()

        print("Create DataStream...")
        prp = prep(data, None)
        data_X, data_Y = prp.getXY(data)
        self.timeStamp = np.load(dsConf.get('pathUtils') + "Timestamp values.npy")
        print(data_X)
        print(data_Y)
        self.stream = DataStream(data=data_X, y=data_Y)

        DECREASE_BATCH_DATASET = int(conf.get('DECREASE_BATCH_DATASET'))
        if DECREASE_BATCH_DATASET == 1:
            print("Decreasing batch dataset...")
            ds.setSmallerBatchDataset()

    """
    Takes the next example from the stream
    
    Returns
    ----------
    :returns:
    an example from the stream
    """
    def getNextExamples(self):
        time = self.timeStamp[0]
        self.timeStamp = np.delete(self.timeStamp, 0)
        example, label = self.stream.next_sample()
        return example, label, time

    """
    Check if the stream has more examples
    
    Returns
    ----------
    :returns:
    True if the stream has al least one more example, False otherwise
    """
    def hasMoreExample(self):
        return self.stream.has_more_samples()

