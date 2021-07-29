from datetime import datetime
import numpy as np
import pandas as pd
from TimeStream import TimeStream
from TimeWindow import TimeWindow
from CountWindow import CountWindow
from PageHinkleyTest import PageHinkleyTest
from StreamLearner import StreamLearner, preprocessAutoencoderDataset, preprocessClassifierDataset, convertTimeDelta
from sklearn.metrics import confusion_matrix
import time

np.random.seed(12)
import tensorflow

tensorflow.random.set_seed(12)

from CNN1D import getAutoencoders, getCNN1D, createImage, getResult

"""
Print information of every concept drift found. 
Information contains:
    in which model is found the drift
    how many examples were analyzed
    how many normal examples were analyzed
    how many attack examples were analyzed
    timeStamp values of the example
These information are saved on the output files too

Parameters:
----------
:param iterator:
example number identifier
:param attacks:
number of attack examples analyzed
:param time:
timeStamp value of the example
:parm outfile:
log file reference
:param classLabel:
identifier of the model that caused the drift. Value in ['Normal Autoencoder'. 'Attack autoencoder', 'CNN1D']
"""
def printPHTInformation(iterator, attacks, time, outFile, classLabel):
    print("\nPHT: Change detected in " + classLabel)
    print("Example computed: ", iterator)
    print("Normal Example computed: ", iterator - attacks)
    print("Attack Example computed: ", attacks)
    print("Drift Example's timestemp: ", time)
    outFile.write("PHT: Change detected in " + classLabel + ":"
                  + "\nExample computed: " + str(iterator)
                  + "\nNormal Example computed: " + str(iterator - attacks)
                  + "\nAttack Example computed: " + str(attacks)
                  + "\nDrift Example's timestemp: " + str(time) + "\n\n")


class RunStream:
    """
    This class contains the execution of STREAM MINDFUL

    Parameter
    ----------
    :param dsConfig:
    input dataset configuration
    :param config:
    execution settings configuration
    """

    def __init__(self, dsConfig, config):
        self.ds = dsConfig
        self.config = config





    """
    execution of STREAM MINDFUL
    """
    def run(self):

        print('STREAM MINDFUL EXECUTION')

        outputFile = open("Results Stream Algorithm.txt", "w")
        trueLabelFile = open(self.ds.get('pathUtils') + "TrueLabels.txt", "w")
        predLabelFile = open(self.ds.get('pathUtils') + "PredictedLabels.txt", "w")

        print("Build Stream..")
        outputFile.write("Build Stream.." + "\n")
        learner = StreamLearner(self.ds)
        stream = TimeStream(self.ds, self.config)
        pathModel=self.ds.get('pathModels')
        if int(self.config.get('WINDOW_TYPE')) == 0:
            print("Build Count Windows..")
            outputFile.write("Build Time Windows.." + "\n")
            windowNormal = CountWindow(self.ds, classLabel='Normal')
            windowAttack = CountWindow(self.ds, classLabel='Attack')
            print(windowNormal.maxLength)
        else:
            print("Build Time Windows..")
            outputFile.write("Build Time Windows.." + "\n")
            windowNormal = TimeWindow(self.ds, classLabel='Normal')
            windowAttack = TimeWindow(self.ds, classLabel='Attack')
        print(" autoencoders...")
        outputFile.write("Load autoencoders..." + "\n")
        autoencoderN, autoencoderA = getAutoencoders(self.ds)
        print("Load CNN1D...")
        outputFile.write("Load CNN1D..." + "\n")
        CNN1D = getCNN1D(self.ds)
        print("Build Page Hinkley base data...")
        outputFile.write("Build Page Hinkley base data..." + "\n")
        before = np.datetime64(datetime.now())
        PHN = PageHinkleyTest(self.ds, classLabel='Normal')
        PHA = PageHinkleyTest(self.ds, classLabel='Attack')
        PHC = PageHinkleyTest(self.ds, classLabel='Classifier')
        after = np.datetime64(datetime.now())
        print("Time for build PageHinkley Test: ", convertTimeDelta(before, after))
        outputFile.write("Time for build PageHinkley Test: " + convertTimeDelta(before, after) + "\n")

        N_changes = 0
        A_changes = 0
        P_changes = 0
        attacks_found = 0
        iterator = 0
        before_program = np.datetime64(datetime.now())
        outputFile.write("Start stream program: " + str(before) + "\n\n")
        delay=0
        time_list=[]
        t1=0
        tot_diff=0
        autoencoderN_New=autoencoderN
        autoencoderA_New=autoencoderA
        CNN1D_New=CNN1D

        autoencoderN_old=autoencoderN
        autoencoderA_old=autoencoderA
        CNN1D_old=CNN1D
        switch=0
        n_switch=0
        list_time_pred=[]


        try:
            while stream.hasMoreExample():
                iterator += 1
                example, label, timeItem = stream.getNextExamples()
                t2=timeItem
                time_on=time.time()

                if label[0] == 0:
                    attacks_found += 1
                #delay
                #print("Delay:"+ str(delay))
                if delay>0:
                    delay=delay-tot_diff
                else:
                    delay=0

                if delay>0:
                    if switch==0:
                     print("Switched old")
                     outputFile.write("Switched old...\n")
                     n_switch+=1
                    switch=1
                    #print("Old models used:")
                    autoencoderN=autoencoderN_old
                    autoencoderA=autoencoderA_old
                    CNN1D=CNN1D_old
                    print("Delay:"+ str(delay))
                else:
                    switch=0
                    #print("New models used: ")
                    autoencoderN=autoencoderN_New
                    autoencoderA=autoencoderA_New
                    CNN1D=CNN1D_New
                    #print("New model done")

                p=time.time()
                normPred = autoencoderN.predict(example)
                attPred = autoencoderA.predict(example)
                image, input_shape = createImage(example, normPred, attPred)
                labelPred = CNN1D.predict(image)
                labelPred = (np.argmax(labelPred, axis=1))
                pfinal = time.time()
                trueLabelFile.write(str(label[0]) + " ")
                predLabelFile.write(str(labelPred[0]) + " ")
                #print("prediction done")
                pdiff=pfinal-p

                PHTN = False
                PHTA = False
                if label[0] == 1:
                    windowNormal.insertExample(example[0], labelPred[0], timeItem)
                    PHTN = PHN.test(example, normPred)
                    if PHTN:
                        N_changes += 1
                        printPHTInformation(iterator, attacks_found, timeItem, outputFile, classLabel='Normal Autoencoder')

                        # upgrade dataset
                        before = np.datetime64(datetime.now())
                        print("Updating Normal autoencoder and CNN1D: ", before)
                        outputFile.write("Updating Normal autoencoder and CNN1D: " + str(before) + "\n")
                        dataset = learner.loadDataset()
                        dataset = learner.updateTrainingSet(dataset, windowNormal, classLabel='Normal')
                        learner.saveDataset(dataset)

                        # for delay
                        autoencoderN_old = autoencoderN
                        CNN1D_old = CNN1D

                        # train autoencoder and CNN
                        upd=time.time()


                        dataXN, dataYN = preprocessAutoencoderDataset(dataset, classLabel="Normal")

                        autoencoderN = learner.updateAutoencoder(autoencoderN, dataXN, dataYN, classLabel="Normal")
                        dataXC, dataYC, dataYC2 = preprocessClassifierDataset(dataset, autoencoderN, autoencoderA,
                                                                              n_classes=int(
                                                                                  self.config.get('N_CLASSES')))
                        CNN1D = learner.updateCNN1D(CNN1D, dataXC, dataYC2)


                        updF=time.time()
                        delay=updF-upd
                        print(delay)
                        # new models
                        autoencoderN_New = autoencoderN
                        CNN1D_New = CNN1D

                        after = np.datetime64(datetime.now())
                        print('here')
                        autoencoderN.save(pathModel + 'autoencoder' + "Normal" + str(after)+ '.h5')
                        CNN1D.save(pathModel + 'MINDFUL'+str(after)+'.h5')
                        print("Normal autoencoder and CNN1D saved: ", after)
                        outputFile.write("Normal autoencoder and CNN1D saved: " + str(after) + "\n"
                                         + "Training time: " + convertTimeDelta(before, after) + "\n\n")
                        # computing errors
                        normalErrors = learner.getAutoencoderErrors(autoencoderN, dataXN, classLabel='Normal')
                        CNN_Errors = learner.getCNN1DErrors(CNN1D, dataXC, dataYC)
                        PHN.updatePHT(normalErrors)
                        PHC.updatePHT(CNN_Errors)


                else:
                    windowAttack.insertExample(example[0], labelPred[0], timeItem)
                    PHTA = PHA.test(example, attPred)
                    if PHTA:
                        A_changes += 1
                        printPHTInformation(iterator, attacks_found, timeItem, outputFile, classLabel='Attack Autoencoder')

                        # upgrade dataset
                        before = np.datetime64(datetime.now())
                        print("Updating Attack autoencoder and CNN1D: ", before)
                        outputFile.write("Updating Attack autoencoder and CNN1D: " + str(before) + "\n")
                        dataset = learner.loadDataset()
                        dataset = learner.updateTrainingSet(dataset, windowAttack, classLabel='Attacks')
                        learner.saveDataset(dataset)

                        # for delay
                        autoencoderA_old = autoencoderA
                        CNN1D_old = CNN1D

                        # train autoencoder and CNN
                        upd = time.time()

                        dataXA, dataYA = preprocessAutoencoderDataset(dataset, classLabel='Attacks')
                        autoencoderA = learner.updateAutoencoder(autoencoderA, dataXA, dataYA, classLabel='Attacks')
                        dataXC, dataYC, dataYC2 = preprocessClassifierDataset(dataset, autoencoderN, autoencoderA,
                                                                              n_classes=int(
                                                                                  self.config.get('N_CLASSES')))
                        CNN1D = learner.updateCNN1D(CNN1D, dataXC, dataYC2)

                        updF=time.time()
                        delay=updF-upd
                        print(delay)
                        # new models
                        autoencoderA_New = autoencoderA
                        CNN1D_New = CNN1D

                        print("autoencoder and CNN saved")
                        after = np.datetime64(datetime.now())
                        autoencoderA.save(pathModel+ 'autoencoder' + "Attacks" + str(after) + '.h5')
                        CNN1D.save(pathModel + 'MINDFUL' + str(after) + '.h5')
                        print("Attack autoencoder and CNN1D saved: ", after)
                        outputFile.write("Attack autoencoder and CNN1D saved: " + str(after) + "\n"
                                         + "Training time: " + convertTimeDelta(before, after) + "\n\n")
                        # computing errors
                        attackErrors = learner.getAutoencoderErrors(autoencoderA, dataXA, classLabel='Attacks')
                        CNN_Errors = learner.getCNN1DErrors(CNN1D, dataXC, dataYC)
                        PHA.updatePHT(attackErrors)
                        PHC.updatePHT(CNN_Errors)

                if not PHTN and not PHTA:
                    if PHC.test(label, labelPred):
                        P_changes += 1
                        printPHTInformation(iterator, attacks_found, timeItem, outputFile, classLabel='CNN1D')

                        # upgrade dataset
                        before = np.datetime64(datetime.now())
                        print("Updating CNN1D: ", before)
                        outputFile.write("Updating CNN1D: " + str(before) + "\n")
                        dataset = learner.loadDataset()
                        dataset = learner.updateTrainingSet(dataset, windowNormal, classLabel='Normal')
                        dataset = learner.updateTrainingSet(dataset, windowAttack, classLabel='Attacks')
                        learner.saveDataset(dataset)

                        CNN1D_old = CNN1D
                        # train CNN
                        upd = time.time()
                        dataXC, dataYC, dataYC2 = preprocessClassifierDataset(dataset, autoencoderN, autoencoderA,
                                                                              n_classes=int(
                                                                                  self.config.get('N_CLASSES')))
                        CNN1D = learner.updateCNN1D(CNN1D, dataXC, dataYC2)

                        updF = time.time()
                        delay = updF - upd
                        print(delay)
                        # new models
                        CNN1D_New = CNN1D

                        after = np.datetime64(datetime.now())
                        CNN1D.save(pathModel+ 'MINDFUL' + str(after) + '.h5')
                        print("Model saved: ", after)
                        outputFile.write("Updated CNN1D saved: " + str(after) + "\n"
                                              + "Training time: " + convertTimeDelta(before, after) + "\n\n")
                        # computing errors
                        CNN_Errors = learner.getCNN1DErrors(CNN1D, dataXC, dataYC)
                        PHC.updatePHT(CNN_Errors)
                else:
                    PHC.addElement(label, labelPred)

                if (iterator % 1000) == 0:
                    print(iterator, " Examples computed")
                t1=timeItem
                item_diff = (t2 - t1).item().total_seconds()
                #print("Item diff:" + str(item_diff))
                tot_diff = pdiff + item_diff
                #print("Total diff:" + str(tot_diff))
                time_off=time.time()
                time_pred=time_off-time_on
                list_time_pred.append(time_pred)
                

            # STREAM ENDED
            after_program = np.datetime64(datetime.now())
            predLabelFile.close()
            trueLabelFile.close()
            PHN.closeTest()
            #PHN.makePlot()
            PHA.closeTest()
            #PHA.makePlot()
            PHC.closeTest()
            #PHC.makePlot()
            print("PH changes found in Benign Autoencoder evaluation: ", N_changes, "\n")
            print("PH changes found in Attack Autoencoder evaluation: ", A_changes, "\n")
            print("PH changes found in CNN1D evaluation: ", P_changes, "\n")
            # print("Mean time to compute an example: ", (after - before) / iterator, "\n")
            outputFile.write("\n\nPH changes found in Benign Autoencoder evaluation: " + str(N_changes)
                             + "\nPH changes found in Attack Autoencoder evaluation: " + str(A_changes)
                             + "\nPH changes found in CNN1D evaluation: " + str(P_changes)
                             + "\nTotal time for computation: " + convertTimeDelta(before_program, after_program)
                             + "\n\n")

            print("Prediction Batch")
            trueLabel = np.load(self.ds.get('pathUtils') + 'Batch_Original_Label.npy')
            predictedLabel = np.load(self.ds.get('pathUtils') + 'Batch_Predicted_Label.npy')
            bcm = confusion_matrix(trueLabel, predictedLabel)
            print(bcm)
            outputFile.write("Prediction Batch\n" + str(bcm) + "\n\n")

            print("Prediction Stream")
            with open(self.ds.get('pathUtils') + "TrueLabels.txt", "r") as file:
                labels = file.read()
            trueLabel = np.fromstring(labels, sep=" ", dtype=float)
            with open(self.ds.get('pathUtils') + "PredictedLabels.txt", "r") as file:
                labels = file.read()
            predictedLabel = np.fromstring(labels, sep=" ", dtype=float)
            scm = confusion_matrix(trueLabel, predictedLabel)
            print(scm)
            outputFile.write("Prediction Stream\n" + str(scm) + "\n")

            columns = ['TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
            r = getResult(scm, int(self.config.get('N_CLASSES')))
            results = pd.DataFrame([r], columns=columns)
            print(results)
            outputFile.write(str(results) + "\n\n")
            results.to_csv('results/' + self.ds.get('testpath') + "results_stream.csv", index=False)
            outputFile.write("Number of switch: " + str(n_switch)+"\n\n")

            outputFile.write("End stream program: " + str(after))
            outputFile.close()
            with open("timePrediction.txt", "w") as f:
               for s in list_time_pred:
                  f.write(str(s) +"\n")

        except Exception as e:
            print("ATTENTION!!   EXCEPTION, " + str(e.__class__) + " RISED!\n")
            predLabelFile.close()
            trueLabelFile.close()
            PHN.closeTest()
            PHN.makePlot()
            PHA.closeTest()
            PHA.makePlot()
            PHC.closeTest()
            PHC.makePlot()
            outputFile.close()
