import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/Data%20of%20teeth.csv")
df['Symptom 1'] = df['Symptom 1'].map({'gum disease':0.0,'a cracked tooth':1.0,'worn-down fillings or crowns':2.0,'Black, white, or brown tooth stains':3.0,'Holes or pits in your teeth':4.0,'Pain when you bite down':5.0,'Yellowish discoloration':6.0,'Cracked or chipped teeth':7.0,'Grooves on your teeth’s surface':8.0,'bleeding':9.0,'pain':10.0,'sore throat':11.0,'Ear Pain':12.0,"Dramatic weight loss":13.0,'Difficulty chewing or swallowing':14.0,"Bad breath":15.0,"Painful chewing":16.0,'Red and swollen gums':17.0,'Tender or bleeding gums':18.0})

df['Symptom 2'] = df['Symptom 2'].map({'gum disease':0.0,'a cracked tooth':1.0,'worn-down fillings or crowns':2.0,'Black, white, or brown tooth stains':3.0,'Holes or pits in your teeth':4.0,'Pain when you bite down':5.0,'Yellowish discoloration':6.0,'Cracked or chipped teeth':7.0,'Grooves on your teeth’s surface':8.0,'bleeding':9.0,'pain':10.0,'sore throat':11.0,'Ear Pain':12.0,"Dramatic weight loss":13.0,'Difficulty chewing or swallowing':14.0,"Bad breath":15.0,"Painful chewing":16.0,'Red and swollen gums':17.0,'Tender or bleeding gums':18.0})

df['Symptom 3'] = df['Symptom 3'].map({'gum disease':0.0,'a cracked tooth':1.0,'worn-down fillings or crowns':2.0,'Black, white, or brown tooth stains':3.0,'Holes or pits in your teeth':4.0,'Pain when you bite down':5.0,'Yellowish discoloration':6.0,'Cracked or chipped teeth':7.0,'Grooves on your teeth’s surface':8.0,'bleeding':9.0,'pain':10.0,'sore throat':11.0,'Ear Pain':12.0,"Dramatic weight loss":13.0,'Difficulty chewing or swallowing':14.0,"Bad breath":15.0,"Painful chewing":16.0,'Red and swollen gums':17.0,'Tender or bleeding gums':18.0})

X = df.drop(['Disease','Treatment','Unnamed: 5'], axis=1)
df['Disease'] = df['Disease'].map({'dentin hypersensitivity':0.0,'cavity':1.0,'Tooth Erosion':2.0,'Mouth Sores':3.0,'Oral Cancer':4.0,'Periodontitis':5.0})
Y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
model1=LogisticRegression()
model1.fit(X,Y)
pickle.dump(model1, open('modelteeth.pkl','wb'))

model = pickle.load(open('modelteeth.pkl','rb'))