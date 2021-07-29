import pandas as pd
import numpy as np
import os



def main():

    COMBINE=0
    dirName = 'Datasets/DS'
    PREPROCESSING1=0

    if(PREPROCESSING1==1):
        if (COMBINE==1):
            pathFolder = 'CICIDS2017/MachineLearningCVE/'
            fileList = os.listdir(pathFolder)
            print(len(fileList))
            i=0

            for f in fileList:
                if i==0:
                    tempdf = pd.read_csv(pathFolder + f)
                    columnsList =tempdf.columns.values
                    df = pd.DataFrame(columns=columnsList)
                tempdf=pd.read_csv(pathFolder+f)

                print("Reading: %s" %f)
                print(tempdf.head())
                columnsList = tempdf.columns.values
                classificationName = columnsList[-1]
                print(classificationName)
                print(tempdf[classificationName].unique())
                print(tempdf[classificationName].value_counts())

                df=df.append(tempdf, ignore_index=True)
                i+=1

            df.to_csv('CICIDS2017.csv', index=False)
        else:
            df=pd.read_csv('CICIDS2017.csv')
        print(df.head())
        print(df.shape)
        columnsList=df.columns.values
        print(columnsList)
        classificationName= columnsList[-1]
        print(classificationName)
        typeAttacks=df[classificationName].unique()
        typeAttacks = np.delete(typeAttacks, np.where(typeAttacks == ['BENIGN']), axis=0)
        print(typeAttacks)
        print(df[classificationName].value_counts())
        df.replace(typeAttacks,
                   'ATTACK', inplace=True)
        print(df.head())

        df.to_csv('CICIDS2017OneCls.csv', index=False)
    else:
        df = pd.read_csv('CICIDS2017OneCls.csv')
        print(df.shape)
        columnsList = df.columns.values
        classificationName = columnsList[-1]
        y = df[classificationName].unique()
        print(df[classificationName].value_counts())
        print(df.head())




    numbFile=10
    numFolder=11
    for i in range(1, numFolder):
        df_anormal = df[(df[classificationName] == 'ATTACK')]
        df_normal = df[(df[classificationName] == 'BENIGN')]
        print(df_normal.shape)
        print(df_anormal.shape)
        if not os.path.exists(dirName+'_'+str(i)):
            # Create target Directory
            os.mkdir(dirName+'_'+str(i))
            print("Directory ", dirName+'_'+str(i)+ '/', " Created ")
        else:
            print("Directory ", dirName+'_'+str(i)+'/', " already exists")
        train_normal=df_normal.sample(n=80000, replace=False)
        train_anormal=df_anormal.sample(n=20000, replace=False)
        train=train_normal.append(train_anormal)
        train.to_csv(dirName+'_'+str(i)+"/train_CICIDS2017OneCls.csv", index=False)
        df_normal.drop(train_normal.index, inplace=True)
        df_anormal.drop(train_anormal.index, inplace=True)
        print("New shape:")
        print(df_normal.shape)
        print(df_anormal.shape)
        for j in range(1,numbFile):
            test_normal = df_normal.sample(n=80000, replace=False)
            test_anormal = df_anormal.sample(n=20000, replace=False)
            test = test_normal.append(test_anormal)
            test.to_csv(dirName + '_' + str(i) + "/test"+str(j)+'_CICIDS2017OneCls.csv', index=False)
            df_normal.drop(test_normal.index, inplace=True)
            df_anormal.drop(test_anormal.index, inplace=True)
            print("New shape:")
            print(df_normal.shape)
            print(df_anormal.shape)






if __name__ == "__main__":
    main()