import json
import pandas as pd
import numpy as np
import os

def wrangledata(df,name):
	df2=df['textAnnotations'].apply(pd.Series)
	#print(df2)
	dataframe1=pd.DataFrame(df2['description'])
	#return dataframe1
	df3=df2['boundingPoly'].apply(pd.Series)
	#print(df3)
	df4=df3['vertices'].apply(pd.Series)
	#print(df4)

	df5=df4[0].apply(pd.Series)
	dataframe2=pd.DataFrame(df5)
	dataframe2.columns=['x0','y0']
	#print(dataframe2)

	df6=df4[1].apply(pd.Series)
	dataframe3=pd.DataFrame(df6)
	dataframe3.columns=['x1','y1']
	#print(dataframe3)

	df7=df4[2].apply(pd.Series)
	dataframe4=pd.DataFrame(df7)
	dataframe4.columns=['x2','y2']
	#print(dataframe4)

	df8=df4[3].apply(pd.Series)
	dataframe5=pd.DataFrame(df8)
	dataframe5.columns=['x3','y3']
	#print(dataframe5)


	final_df=pd.concat([dataframe1,dataframe2,dataframe3,dataframe4,dataframe5],axis=1)
	#print(final_df)

	final_df['id']=np.arange(len(final_df))
	final_df['imagename']=name.split('.',1)[0]
	#print(final_df)
	return final_df


def wrangledata2(df,name):
	print("a")
	name2=name.split('.',1)[0]
	name3=name2.split('_',1)[0]+'_'+name2.split('_')[1]
	#df['imagename']=name.split('.',1)[0]
	df['imagename']=name3
	#df['imagename']=name.split('_',1)[0]+name.split('_',1)[1]
	dfk=df[['label','id','imagename','text']]
	return dfk

path="E:\Storage Access\Workspaces\AMEX\Round_1\Round1\input"
os.chdir(path)

basedf=pd.DataFrame(columns=['description','x0','y0','x1','y1','x2','y2','x3','y3','id','imagename'])
#print(basedf)


for file in os.listdir():
	if file.endswith(".json"):
		file_path=f"{path}\{file}"
		name=f"{file}"
		print(file_path)
		print(name)
		with open(name) as f:
			#id0=json.load(f)
			df=pd.read_json(name)
			#print(df)
			displaydf=wrangledata(df,name)
			#print(displaydf)
			basedf=pd.concat([basedf,displaydf],axis=0)
			#print(basedf)


path="E:\Storage Access\Workspaces\AMEX\Round_1"
os.chdir(path)

print(basedf)
basedf.to_csv('round1_test.csv',index=False)			
