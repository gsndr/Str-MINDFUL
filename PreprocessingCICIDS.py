import pandas as pd
import numpy as np
def preprocessing2(stream,path):

    stream.rename(
        columns={' Flow Packets/s': 'Flow_Packets', 'Flow Bytes/s': 'Flow_Bytes', ' Label': 'Classification',
                 'Fwd Packets/s': 'Fwd_Packets', ' Bwd Packets/s': 'Bwd_Packets'},
        inplace=True)

    cls = ['Classification']
    label = ['BENIGN', 'ATTACK']
    # listCategorical = ['Flow ID, Source IP, Destination IP, Timestamp, External IP']
    stream = stream.drop(['Flow ID', ' Source IP', ' Destination IP', ' Source Port', ' Protocol'], axis=1)
    stream = stream[stream['Classification'].notna()]
    stream = stream.replace([np.inf, -np.inf], np.nan)
    stream = stream[stream['Flow_Bytes'].notna()]
    stream = stream[stream['Fwd_Packets'].notna()]
    print(stream.columns)

    allowed_val = ['BENIGN']
    stream.loc[~stream['Classification'].isin(allowed_val), 'Classification'] = 'ATTACK'
    stream["Classification"].replace(to_replace=dict(BENIGN=1, ATTACK=0), inplace=True)
    from sklearn import preprocessing
    '''
    le = preprocessing.LabelEncoder()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    stream["Classification"] = le.fit_transform(stream["Classification"].values)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    exit()
    '''

    stream[' Timestamp'] = pd.to_datetime(stream[' Timestamp'], dayfirst=True)
    stream = stream.sort_values(by=' Timestamp')
    timestamp = np.asarray(stream[' Timestamp'])
    np.save((path+"TimestampvaluesAll.npy"), timestamp)
    stream = stream.drop([' Timestamp'], axis=1)

    col_names = stream.columns

    nominal_inx = []
    binary_inx = [
        'Fwd PSH Flags,  Bwd PSH Flags,  Fwd URG Flags,  Bwd URG Flags, FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count,'
        'ACK Flag Count	,URG Flag Count, CWE Flag Count,  ECE Flag Count,  Fwd Header Length.1,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk,'
        ' Fwd Avg Bulk Rate	 Bwd Avg Bytes/Bulk	 Bwd Avg Packets/Bulk	Bwd Avg Bulk Rate']

    binary_inx = [30, 31, 32, 33, 43, 44, 45, 46, 47, 48, 49, 50, 56, 57, 58, 59, 60, 61]
    numeric_inx = list(set(range(78)).difference(nominal_inx).difference(binary_inx))

    listNumerical10 = col_names[numeric_inx].tolist()

    col_names = col_names.delete(-1)
    col_names = col_names.values.tolist()
    print(col_names)
    stream[col_names] = stream[col_names].apply(pd.to_numeric, errors='coerce')


    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    print('Scaling')
    listContent = listNumerical10
    #print(stream[listContent].values)

    scaler.fit(stream[listContent])
    # scaler.fit(train[listContent].values)
    stream[listContent] = scaler.transform(stream[listContent])
    pathOutputTest =path+ 'completeStreamNumeric.csv'
    stream.to_csv(pathOutputTest, index=False)  # X is an array

    return stream


path = '../../DS/CICIDS/stream/CORRECT/'
#path = '../../DS/CICIDS/original/STREAM/'

df=pd.read_csv(path+'All.csv')
df1=preprocessing2(df, path)
exit()
print("Load dataset 1...")
monday = pd.read_csv(path+'Monday-WorkingHours.pcap_ISCX.csv')
print(monday.shape[0])
print(monday[' Timestamp'].shape[0])
timeMonday=pd.to_datetime(monday[' Timestamp'],format='%d/%m/%Y %H:%M:%S')
print(timeMonday.shape[0])
monday[' Timestamp']=timeMonday

print("Load dataset 2...")
tuesday = pd.read_csv(path+'Tuesday-WorkingHours.pcap_ISCX.csv')
print(tuesday.shape[0])
timeTuesday=pd.to_datetime(tuesday[' Timestamp'], format='%d/%m/%Y %H:%M')
tuesday[' Timestamp']=timeTuesday

print("Load dataset 3...")
wednsday = pd.read_csv(path+'Wednesday-workingHours.pcap_ISCX.csv')
print(wednsday.shape[0])

# selecting rows based on condition
#pomeriggio
df=wednsday[445858:].copy()
print(df.shape[0])
print(df.head(10))
#df = wednsday[wednsday[' Label'] == 'Heartbleed']
time=pd.to_datetime(df[' Timestamp'], format='%d/%m/%Y %H:%M')
print(time.tail(8))
time2=time+pd.Timedelta(hours=12)
df[' Timestamp']=time2
print(time2.tail(8))
print(df.shape[0])
wednsday=wednsday.iloc[:445858]
#wednsday.drop(wednsday[445858:], inplace=True)

timeW=pd.to_datetime(wednsday[' Timestamp'], format='%d/%m/%Y %H:%M')
wednsday[' Timestamp']=timeW
wednsday=pd.concat([wednsday,df])
print(wednsday.shape[0])


print("Load dataset 4...")
thursday1 = pd.read_csv(path+'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
thursday1.dropna(inplace=True)
timeT1=pd.to_datetime(thursday1[' Timestamp'], format='%d/%m/%Y %H:%M')
thursday1[' Timestamp']=timeT1
print(thursday1.shape[0])

print("Load dataset 5...")
thursday2 = pd.read_csv(path+'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
print(thursday2.shape[0])
timeT2=pd.to_datetime(thursday2[' Timestamp'], format='%d/%m/%Y %H:%M')
time2=timeT2+pd.Timedelta(hours=12)
thursday2[' Timestamp']=time2

print("Load dataset 6...")
friday1 = pd.read_csv(path+'Friday-WorkingHours-Morning.pcap_ISCX.csv')
timeF1=pd.to_datetime(friday1[' Timestamp'], format='%d/%m/%Y %H:%M')
friday1[' Timestamp']=timeF1
print(friday1.shape[0])

print("Load dataset 7...")
friday2 = pd.read_csv(path+'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
print(friday2.shape[0])
time=pd.to_datetime(friday2[' Timestamp'], format='%d/%m/%Y %H:%M')
print(time.tail(8))
time2=time+pd.Timedelta(hours=12)
friday2[' Timestamp']=time2

print("Load dataset 8...")
friday3 = pd.read_csv(path+'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
print(friday3.shape[0])
time=pd.to_datetime(friday3[' Timestamp'], format='%d/%m/%Y %H:%M')
print(time.tail(8))
time2=time+pd.Timedelta(hours=12)
friday3[' Timestamp']=time2
print(friday3[' Timestamp'].tail(8))
print("Concatening datsets...")
final_train = pd.concat([monday, tuesday,wednsday, thursday1,thursday2,friday1,friday2,friday3])
final_train = final_train.sort_values(by=' Timestamp')
final_train.to_csv('All.csv',index=False)
final_train = pd.concat([monday, tuesday])
final_train = final_train.sort_values(by=' Timestamp')
final_train.to_csv('Training.csv',index=False)
final_train = pd.concat([wednsday, thursday1,thursday2,friday1,friday2,friday3])
final_train = final_train.sort_values(by=' Timestamp')
final_train.to_csv('Testing.csv',index=False)
