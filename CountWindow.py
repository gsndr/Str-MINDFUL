import numpy as np
import pandas as pd


class CountWindow:
    """
    Count Sliding Window

    Parameters
    ----------
    :param dsConf:
    Input dataset configuration
    :param classLabel:
    Class label of the examples that will be contained
    """

    def __init__(self, dsConf, classLabel):
        self.window = []
        self.label = classLabel
        if self.label == 'Normal':
            self.maxLength = int(dsConf.get('countWindowNormal'))
        else:
            self.maxLength = int(dsConf.get('countWindowAttack'))

    """
    Insert a new example in the window
    
    Parameters
    ----------
    :param example:
    Flow example from the stream
    :param prediction:
    Label predicted by CNN1D on input example
    :parma time:
    Timestamp value of the input example 
    """
    def insertExample(self, example, prediction, time):
        element = [example, prediction, time]
        self.window.append(element)
        if len(self.window) >= self.maxLength:
            self.window.pop(0)

    """
    Returns all contained examples and initializes the window
    
    Parameter
    ----------
    :param columns:
    List of the names of the examples' features
    
    Returns:
    ----------
    :return:
    Pandas DataFrame which contains all the example stored in the window
    """
    def getWindow(self, columns):
        out = []
        if self.label == 'Normal':
            for val in self.window:
                out.append(np.append(val[0], int(1)))
            out = pd.DataFrame(out, columns=columns)
        else:
            for val in self.window:
                out.append(np.append(val[0], int(0)))
            out = pd.DataFrame(out, columns=columns)
        self.window = []
        return out

    """
    Print all the examples in the window. Used just for test
    """
    def printWindow(self):
        print("Count Window:\n", self.window)

    """
    Returns the number of examples stored in the window
    
    Returns
    ----------
    :return:
    Window's size
    """
    def length(self):
        return len(self.window)
