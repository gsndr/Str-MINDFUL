import pandas as pd
pd.set_option('display.max_columns', None)
import datetime
import numpy as np

'''
dataFrame=pd.read_csv('datasets/CICIDS2017/original/streamDataset.csv')
t = pd.to_timedelta(dataFrame[' Timestamp'].to_numpy())
t= t.astype('datetime64',copy=False)
np.save('datasets/CICIDS2017/utils/Timestamp values.npy', t, allow_pickle=True)



ts=np.load('datasets/CICIDS2017/utils/Timestamp values.npy', allow_pickle=True)
print(ts)
print(ts.shape)
ts=np.load('datasets/CICIDS2017/utils/Timestamp valuesOld.npy')
print(ts)
print(ts.shape)
exit()
'''
path='datasets/CICIDS2017/original/'

'''
df=pd.read_csv(path+'streamDataset.csv', encoding='cp1252')
index = df. index
number_of_rows = len(index)
print(number_of_rows)

df.dropna(inplace=True)
index = df. index
number_of_rows = len(index)
print(number_of_rows)

timeStamp=df[' Timestamp'].astype(str)
print(timeStamp.isnull().values.any())
index=timeStamp.index
no_rows=len(index)
print(timeStamp.head(2))
listDifference=[]
j=1
for i in range(0, no_rows-1):
    print(i)
    r1=timeStamp.iloc[i]
    r1_date = datetime.datetime.strptime(r1, '%Y-%m-%d %H:%M:%S')
    print(timeStamp.iloc[j])
    r2=timeStamp.iloc[j]
    r2_date = datetime.datetime.strptime(r2, '%Y-%m-%d %H:%M:%S')
    print(r2_date)
    diff=(r1_date-r2_date).total_seconds()
    print(diff)

    listDifference.append(diff)
    j+=1

from statistics import mean
print(listDifference)
print(mean(listDifference))
'''
from datetime import timedelta

'''

ts=np.load('datasets/CICIDS2017/utils/Timestamp values.npy', allow_pickle=True)
print(type(ts))
print(ts.shape)
# Calculating difference list
from functools import reduce
import operator

TIME_DELTA_ATTR_MAP = (
    ('year', 'Y'),
    ('month','M'),
        ('day', 'D'),
    ('hour','H'),
    ('minute', 'M'),
        ('second', 's')
        )

def to_timedelta64(value: datetime.timedelta) -> np.timedelta64:
    return reduce(operator.add,
        (np.timedelta64(getattr(value, attr), code)
        for attr, code in TIME_DELTA_ATTR_MAP if getattr(value, attr) > 0))

diff_list = []
print(ts)
for x, y in zip(ts[0::], ts[1::]):
    print(x)
    s=y.item().strftime('%Y-%m-%d %H:%M:%S')
    t = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    delta = datetime.timedelta(days=t.day, hours=t.hour, minutes=t.minute, seconds=t.second)
    s = x.item().strftime('%Y-%m-%d %H:%M:%S')
    t = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    delta1 = datetime.timedelta(days=t.day, hours=t.hour, minutes=t.minute, seconds=t.second)
    print(delta1+delta)
    print(delta)
    print(type(x))
    print(type(y-x))
    exit()
    diff=(y-x).item().total_seconds()
    print(type(diff))
    diff_list.append(diff)

# printing difference list
print("difference list: ", str(diff_list))
from statistics import mean
print(mean(diff_list))
#time test
score = []

path='../../DS/CICIDS/numeric/STREAM/'
df=pd.read_csv(path+'streamDatasetAllConc.csv')
print(df.shape[0])
exit()
'''
import numpy as np
score=[]
with open("timePrediction1500.txt", "r") as f:
  for line in f:
    score.append(float(line.strip()))
print(len(score))
print("Mean: " + str(np.mean(score)))
print("Max:" + str(np.max(score)))
print("Min: " + str(np.min(score)))
import matplotlib.pyplot as plt
final=2827876
final2=len(score)
print(final2)
initial=int(final)-int(final2)
print(initial)
rows=list(range(initial,final))
print(rows[0])
print(score[0])







#plt.show()
import re

pathResult='Results Stream Algorithm1500.txt'
s = 'Example computed: '
t='PHT: '
dict_change={}
type_change=[]
n = 975126

with open(pathResult) as f:
  lines = f.readlines()

  for line in lines:
    if line.startswith(t):
      if line == 'PHT: Change detected in Normal Autoencoder:\n':
        type_change.append(0)
      elif line == 'PHT: Change detected in CNN1D:\n':
        type_change.append(2)
      else:
        type_change.append(1)
    if line.startswith(s):
      p = "\w+ *(?!.*Example computed : )"
      l = re.findall(p, line)
      example=int(l[2])
      dict_change[example]=type_change
      type_change=[]

print(len(dict_change))
group=[]
scatter_xA=[]
scatter_yA=[]
scatter_xN=[]
scatter_yN=[]
scatter_x=[]
scatter_y=[]
print(score[280704])

print(dict_change)

i=0
for k in dict_change:
  if dict_change[k][0]==0:
    scatter_xN.append(k + n)
    scatter_yN.append(score[k - 1])
  elif dict_change[k][0]==1:
    scatter_xA.append(k + n)
    scatter_yA.append(score[k - 1])
  else:
    scatter_x.append(k + n)
    scatter_y.append(score[k - 1])
  ''' 
  print(k, dict_change[k])
  group.append(int(dict_change[k][0]))
  print(group)
  scatter_x.append(k+n)
  print(scatter_x)
  scatter_y.append(score[k-1])
  print('y')
  print(scatter_y)



if dict_change[k][0] == 0:
  scatter_y.append(0)
elif dict_change[k][0] == 1:
  scatter_y.append(1)
else:
  scatter_y.append(2)
'''

'''
with open(pathResult) as f:
  contents = f.readlines()
  #line = f.readline()

result =[x for x in contents if x.startswith(s)]
p = "\w+ *(?!.*Example computed : )"
l = re.findall(p, line)
result= [ re.findall(p, x) for x in result]
result=[int(x[2]) for x in result]
print(result)
change =[x for x in contents if x.startswith('PHT')]
t_c=[]
print(t_c)
for x in change:
  if x == 'PHT: Change detected in Normal Autoencoder:\n':
    t_c.append(0)
  elif x=='PHT: Change detected in CNN1D:\n':
    t_c.append(2)
  else:
    t_c.append(1)


print(change)
print(type(contents))


change_d=[]
for x in result:
  change_d.append(score[int(x)])
'''
#cdict = {0: 'b', 1: 'o', 2: 'seagreen'}

plt.figure(figsize=(10,5))
plt.ticklabel_format(style='plain', useOffset=False, axis='both')


from matplotlib.colors import ListedColormap

scatter=plt.plot(rows, score, color='gold', zorder=1)
colours = ListedColormap(['navy','red','seagreen'])
classes=['Normal Autoencoder', 'Attack Autoencoder', 'CNN1D']
scatter=plt.scatter(scatter_xN, scatter_yN,marker='s', c='navy',  s=6, zorder=2, label='$z_n$')
scatter=plt.scatter(scatter_xA, scatter_yA,marker="x", c='red',  s=6, zorder=2, label='$z_a$')
scatter=plt.scatter(scatter_x, scatter_y,marker='o', c='seagreen',  s=4, zorder=2, label='$classifier$')
#scatter=plt.scatter(scatter_x, scatter_y, c=group, marker=['.','o', '*'], cmap=colours,  s=3, zorder=2)
index_day=[692704,458949, 703246]
n=975829
st_day=[]
for i in index_day:
  n = n + i
  st_day.append(n)
plt.vlines(x=st_day,ymin=0, ymax=23, color='r', linestyle='dotted')

plt.legend(handles=scatter.legend_elements()[0],  loc="upper left", labels=classes, fontsize=15, markerscale=3., scatterpoints=1)
plt.ylabel('Time(s)', fontsize='18')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.xlabel('Samples')

plt.savefig('Stream.png', bbox_inches='tight')
