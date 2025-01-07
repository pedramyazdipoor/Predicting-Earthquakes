!pip install solarsystem
import numpy as np
import pandas as pd
import solarsystem
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numpy import save
from numpy import load
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from matplotlib.pyplot import figure
from mlxtend.evaluate import feature_importance_permutation


sheet0 = pd.read_excel('4_5764873750877047261.xlsx',sheet_name=0) #1900 to 2006
sheet1 = pd.read_excel('4_5764873750877047261.xlsx',sheet_name=1) #2006 to 2020

#================================================= sheet_1 cleaning ==========================================================
columns = ['Year','Month','Day','Hour','Lat','Long','Mag']
#=================== check if any value is NaN ===============================================================================
ind = []
for i in range(0,len(sheet1.index)):
  for j in range(0,len(columns)):
    if  math.isnan(sheet1[columns[j]][i]):
      ind.append(i)
      break
#=================== remove if any value is NaN
sheet1_new = sheet1.drop(ind)

#=================== check if hour value is not valid (<6)
ind = []
for i in range(0,len(sheet1_new.index)): 
  if len(str(int(sheet1_new['Hour'][sheet1_new.index[i]]))) < 6 :
    ind.append(sheet1_new.index[i])
    
sheet1_n = sheet1_new.drop(ind)
#=============================================================================
min = []
for x in sheet1_n.index:
  h = str(int(sheet1_n.loc[x, "Hour"]))
#=============================================================================
  if (int(h[0:2]) < 25):
    hour = int(h[0:2])

    if int(h[2:4]) < 60 :
      min.append(int(h[2:4]))

    elif int(h[2:4]) > 60 :
      min.append(int(h[2]))

    sheet1_n.loc[x, "Hour"] = hour
#===========================================================================
  elif (int(h[0:2]) > 24):
    hour = int(h[0])

    if int(h[1:3]) < 60 :
      min.append(int(h[1:3]))

    elif int(h[1:3]) > 60 :
      min.append(int(h[1]))

    sheet1_n.loc[x, "Hour"] = hour
  
sheet1_n['Minute'] = min

year_1 = sheet1_n['Year']
Month_1 = sheet1_n['Month']
Day_1 = sheet1_n['Day']
Hour_1 = sheet1_n['Hour']
Minute_1 = sheet1_n['Minute']
Lat_1 = sheet1_n['Lat']
Long_1 = sheet1_n['Long']
Mag_1 = sheet1_n['Mag']

sheet1_df = {'Year' : year_1 , 'Month' : Month_1 , 'Day' : Day_1 ,
        'Hour' : Hour_1 , 'Minute' : Minute_1, 
        'Lat' : Lat_1 , 'Long' : Long_1 , 'Mag': Mag_1}

sheet1_final = pd.DataFrame(sheet1_df)
df_1 = sheet1_final.reset_index()
#================================================= sheet_0 cleaning ==========================================================

date = sheet0['Date']
year = []
month = []
day = []
for i in date:
  year.append(int(i[0:4]))
  month.append(int(i[5:7]))
  day.append(int(i[8:10]))

time = sheet0['Time']
hour_list = []
minute_list = []
for j in time:
  hour_list.append(int(j[0:2]))
  minute_list.append(int(j[3:5]))

mag =  sheet0['Mag.']
lat =  sheet0['Lat.']
long = sheet0['Long.']

sheet0_df = {'Year' : year , 'Month' : month , 'Day' : day ,
        'Hour' : hour_list , 'Minute' : minute_list, 
        'Lat' : lat , 'Long' : long , 'Mag': mag}

df_0 = pd.DataFrame(sheet0_df)

'''
columns2 = ['Year','Month','Day','Hour','Minute','Lat','Long','Mag']
ind = []
for i in range(0,len(df_0.index)):
  for j in range(0,len(columns2)):
    if  math.isnan(df_0[columns2[j]][i]):
      ind.append(i)
      break

if len(ind) == 0 : print("No NaN")
else : print(ind)'''

Df = [df_1 , df_0]
df = pd.concat(Df,ignore_index=True, sort=False)
del df["index"]


planet_names = ['Sun', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Ceres', 'Chiron', 'Eris']

def angle_calculator(planet_pos,sun_pos):
  Norm_p = np.sqrt((p[0]**2) + (p[1]**2))
  Norm_s = np.sqrt((sun[0]**2) + (sun[1]**2))
  return np.arccos((p.dot(sun))/(Norm_p*Norm_s))


def angle_sun_p(year, month, day, hour, minute, planet):
  year = int(year)
  month = int(month)
  day = int(day)
  hour = int(hour)
  minute = int(minute)

  if planet == 'Moon' :                               
    moon = solarsystem.moon.Moon(year=year, month=month, day=day, hour=hour, minute = minute,UT=3.5).position()
    sun = solarsystem.geocentric.Geocentric(year=year, month=month, day=day, hour=hour, minute = minute,UT=3.5).position()['Sun']
    return angle_calculator(np.array(moon[0:2]) ,np.array(sun[0:2]))

  else :
    positions = solarsystem.geocentric.Geocentric(year=year, month=month, day=day, hour=hour, minute = minute ,UT=3.5).position() 
    sun = positions['Sun']  
    p = positions[planet]
    return angle_calculator(np.array(p[0:2]), np.array(sun[0:2]))           


def long_lat_teta_calculator(df, planet_names):

  for planet in planet_names:
    longit, latit, dis = [],[],[]
    for x in df.index:
      p = solarsystem.geocentric.Geocentric(year=int(df.loc[x, "Year"]), 
                                            month=int(df.loc[x, "Month"]), 
                                            day=int(df.loc[x, 'Day']), hour=int(df.loc[x, "Hour"]), 
                                            minute=int(df.loc[x, "Minute"]), UT=3.5,dst=1 ).position()[planet]
      
      longit.append(p[0])
      latit.append(p[1])
      dis.append(p[2])
    
    df['longit'+ planet] = longit
    df['lait'+ planet] = latit
    df['dic'+ planet] = dis

   for j in planet_names:
    if j != 'Sun' :
      temp = []
      for x in df.index:
        temp.append(angle_sun_p(df.loc[x, "Year"],df.loc[x, "Month"],df.loc[x, 'Day'],df.loc[x, "Hour"],df.loc[x, "Minute"],j))
      df['teta'+j] = temp

def moon_features(df):
  longit_m , latit_m, dis_m = [],[],[]

  for x in df.index:
    m = solarsystem.moon.Moon(year=int(df.loc[x, "Year"]), 
                            month=int(df.loc[x, "Month"]), 
                            day=int(df.loc[x, 'Day']), 
                            hour=int(df.loc[x, "Hour"]), 
                            minute=int(df.loc[x, "Minute"]) ,UT=3.5,dst=1 ).position()
    
    longit_m.append(m[0])
    latit_m.append(m[1])
    dis_m.append(m[2])

  df['longitM'] = longit_m
  df['latitM'] = latit_m
  df['disM'] = dis_m

  temp = []
  for x in df.index:
    temp.append(angle_sun_p(df.loc[x, 'Year'],df.loc[x, 'Month'],df.loc[x, 'Day'],df.loc[x, 'Hour'],df.loc[x, 'Minute'],'Moon'))
  df['tetaM'] = temp
  

long_lat_teta_calculator(df,planet_names)
moon_features(df)
del df["Year"]
del df["Month"]
del df["Day"]
del df["Hour"]
del df["Minute"]
magnitude = df['Mag']
del df["Mag"]
#print("Data Frame\n ",df)
#print("magnitude\n ",magnitude)

'''
labels = []
for i in range(0,len(magnitude)):
  if magnitude[i] >= 4.5 :
    labels.append(1)
  else : 
    labels.append(0)
save('labels.npy',labels)
'''

#print(df)
data = df.values.tolist()
data = np.array(data)
data = preprocessing.normalize(data,axis=0)
data = data.tolist()
save("Normalized_data.npy",data)

'''
pca = PCA()
pca.fit(data)
transformed = pca.transform(data)
save("transformed.npy",transformed)

#plt.figure(figsize=(12,10))
pd.DataFrame(pca.explained_variance_ratio_).plot.bar()
plt.legend('')
plt.xlabel('Principal Components')
plt.ylabel('Explained Varience')
plt.savefig("pca")'''

'''

#transformed = load("transformed.npy")
#labels = load("labels.npy")

'''
'''
reduced_data = []
for t in range(0,len(transformed)):
  temp = []
  for r in range(0,18):
    temp.append(transformed[t][r])
  reduced_data.append(temp)  '''

data_norm = load('Normalized_data.npy')
labels = load("labels.npy")
X_train, X_test, y_train, y_test  = train_test_split(data_norm,labels ,train_size=0.8) 

clf = svm.SVC(kernel='rbf',C = 1000, gamma = 10)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

print("accuracy :  ",accuracy_score(y_test, predicted))
print("precision macro: ",precision_score(y_test, predicted, average='macro',zero_division=1))
print("recall  macro  : ",recall_score(y_test, predicted, average='macro',zero_division=1))
print("f1  macro  : ",f1_score(y_test, predicted, average='macro',zero_division=1))

metrics.plot_roc_curve(clf, X_test, y_test) 

imp_vals, imp_all = feature_importance_permutation(
    predict_method=clf.predict, 
    X=X_test,
    y=y_test,
    metric='accuracy',
    num_rounds = 10,
    seed=1)

#print(imp_vals)

std = np.std(imp_all, axis=1)
indices = np.argsort(imp_vals)[::-1]

plt.figure(figsize=(15,10),dpi=100)
plt.title("SVM feature importance via permutation importance")
plt.bar(range(data_norm.shape[1]), imp_vals[indices],
        yerr=std[indices])
plt.xticks(range(data_norm.shape[1]), indices)
plt.xlim([-1, data_norm.shape[1]])
plt.savefig('feature_importance.png')

#https://colab.research.google.com/drive/14Oq08KtZ1zLg5JIpChufTe3Ud38Rn-Co#scrollTo=nLfu-yl4mt2c