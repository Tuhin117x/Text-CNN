import pandas as pd
import os
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

#This script will be used for dividing training dataset and scoring models
#This model has not been trained on the complete dataset


#--------------------------------------
# Read Datasets
#--------------------------------------

path='E:\Storage Access\Workspaces\AMEX'
os.chdir(path)
df=pd.read_csv("train_2.csv")
rnd1_df=pd.read_csv("round1_test.csv")

#--------------------------------------
# Data Pre-processing
#--------------------------------------


df.dropna(inplace=True)
train=df.iloc[:35000,]
test=df.iloc[35000:,]
print(train)
print(test)

#X=train[['description','x0','y0','x1','y1','x2','y2','x3','y3']]
X=train[['description']]
y=train['label']


#label_encoder=LabelEncoder()
#X['imagename']=label_encoder.fit_transform(df['imagename'])

#--------------------------------------
# Model Definition
#--------------------------------------
model=RandomForestClassifier()
#model=GaussianNB()
#model=MultinomialNB()
#model=LogisticRegression(n_jobs=1,C=1e15)
#model=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
vectorizer1 = TfidfVectorizer()
column_transformer=ColumnTransformer([('tfidf1',vectorizer1,'description')],remainder='passthrough')

#--------------------------------------
# Model Fit
#--------------------------------------

print("************ New Model Run Starts ****************")
pipe=Pipeline([('tfidf',column_transformer),('classify',model)])
pipe.fit(X,y)
#pipe2=Pipeline([('tfidf',column_transformer),('classify',model)])

#--------------------------------------
# Test Dataset Preperation
#--------------------------------------

#x_test=rnd1_df[['description','x0','y0','x1','y1','x2','y2','x3','y3']]
#x_test=test[['description','x0','y0','x1','y1','x2','y2','x3','y3']]
x_test=test[['description']]
y_test=test[['label']]
#test_label_encoder=LabelEncoder()
#x_test['imagename']=test_label_encoder.fit_transform(rnd1_df['imagename'])

#--------------------------------------
# Prediction
# Change Filename on line 74 & 88
#--------------------------------------

output=pipe.predict(x_test)
print(type(output))
print(output)

output_df=x_test
output_df['predicted_label']=output
output_df['actual_label']=y_test
f1_score_value=f1_score(output_df['actual_label'],output_df['predicted_label'],average='weighted')
output_df['f1_score']=f1_score_value
output_df['model_name']='RandomForestClassifier'

print(output_df)
output_df.to_csv('model_testing.csv',index=False)

#rnd1_df['label']=output
#final_output=rnd1_df[['imagename','id','label']]
#final_output.to_csv('new_sol_v3.csv',index=False)

def recur_dictify(frame):
    if len(frame.columns) == 1:
        if frame.values.size == 1: return frame.values[0][0]
        return frame.values.squeeze()
    grouped = frame.groupby(frame.columns[0])
    d = {k: recur_dictify(g.iloc[:,1:]) for k,g in grouped}
    return d

#final_df=recur_dictify(final_output)
import json


#with open('final_output_v3.json','w') as filez:
#	filez.write(json.dumps(final_df))