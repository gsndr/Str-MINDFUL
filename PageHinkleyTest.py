from skmultiflow.drift_detection import PageHinkley, KSWIN
import numpy as np
from Plot import Plot


class PageHinkleyTest:
    """
    Class that compute Page Hinkley test for each example analyzed.
    This class uses two log file to save prediction/reconstruction errors and
    which examples makes concept drifts

    Parameters
    ----------
    :param dsConfig:
    Input dataset configuration
    :param classLabel:
    Label that identifies the monitored model
    """

    def __init__(self, dsConfig, classLabel):
        self.conf = dsConfig
        self.errorsFile = open(self.conf.get('pathUtils') + classLabel + 'StreamErrors.txt', "w")
        self.driftsFile = open(self.conf.get('pathUtils') + classLabel + 'StreamDrifts.txt', "w")
        self.classLabel = classLabel
        self.PHT = PageHinkley()
        #self.PHT=KSWIN(alpha=0.01)
        for val in np.load(self.conf.get('pathUtils') + classLabel + 'Errors.npy'):
            self.PHT.add_element(val)

    """
    Computes PH test on an example. If the monitored model is an Autoencoder, function computes the mean square error
    between original example and its prediction made by related model. If the monitored model is the CNN1D, function computes
    the abs of the difference between original and predicted label.
    
    Parameters
    ----------
    :param example:
    original example from the stream
    :param prediction:
    prediction on the example made by monitored model
    
    Returns
    ----------
    :return:
    True if the new example makes a concept drift, False otherwise
    """
    def test(self, example, prediction):
        if self.classLabel == 'Classifier':
            error = abs(example - prediction)
        else:
            error = np.mean(np.power(example - prediction, 2), axis=1)
        self.errorsFile.write(str(error[0]) + " ")
        self.PHT.add_element(error[0])
        if self.PHT.detected_change():
            self.driftsFile.write(str(error[0]) + " ")
            return True
        self.driftsFile.write(str(-1) + " ")
        return False

    """
    Initializes the PH test after a train of the monitored model
    
    Parameters
    ----------
    :param Errors:
    Prediction/Reconstruction errors made by the monitored model on the updated batch dataset
    """
    def updatePHT(self, Errors):
        self.PHT = PageHinkley()
        for val in Errors:
            self.PHT.add_element(val)

    """
    Closes log files at the end of stream program
    """
    def closeTest(self):
        self.errorsFile.close()
        self.driftsFile.close()

    """
    When a concept drift rises from an autoencoder, PH Test is not computed on CNN1D on the examples
    that causes the drift.
    This function allows not to lose information about classification error on this example, saving 
    it on lof files.
    """
    def addElement(self, example, prediction):
        error = abs(example - prediction)
        self.errorsFile.write(str(error[0]) + " ")
        self.driftsFile.write(str(-1) + " ")

    """
    At the end of the stream computation, this function loads information from log files and build
    a plot of the prediction/reconstruction errors over the entire stream 
    """
    def makePlot(self):
        with open(self.conf.get('pathUtils') + self.classLabel + 'StreamErrors.txt', "r") as file:
            string = file.read()
        errors = np.fromstring(string, dtype=float, sep=" ")
        with open(self.conf.get('pathUtils') + self.classLabel + 'StreamDrifts.txt', "r") as file:
            string = file.read()
        drifts = np.fromstring(string, dtype=float, sep=" ")
        plotter = Plot()
        plotter.plotPageHinkleyTest(errors, drifts, 'Evaluation loss and PH Test on ' + self.classLabel, self.classLabel)
