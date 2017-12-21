import pandas as pd
import numpy as np
import datetime as dt

from sklearn import preprocessing as skp
from collections import defaultdict as dd
from fancyimpute import KNN
from scipy import stats



rawdata = pd.read_csv('exp.csv',header=0,low_memory=False,encoding='ISO-8859-1')

headers = [':CKDEPI','ALAT','ASAT','CHOL','CK','CRP','CTROP','ERY','HB','HDLC','HINDEX','HT','K','KR','LD','LDLC','LEU','LINDEX','MCH','MCHC','MCV1','NA','RATI','RDWC','TG','TRO','UR']
headerscard = ['Geslacht','DiagnoseCode','vrgeschiedenis_myochardinfarct','vrgeschiedenis_PCI','vrgeschiedenis_CABG','vrgeschiedenis_CVA_TIA','vrgeschiedenis_vaatlijden','vrgeschiedenis_hartfalen','vrgeschiedenis_maligniteit','vrgeschiedenis_COPD','vrgeschiedenis_atriumfibrilleren','TIA','CVA_Niet_Bloedig','CVA_Bloedig','dialyse','riscf_roken','riscf_familieanamnese','riscf_hypertensie','riscf_hypercholesterolemie','riscf_diabetes','roken','Radialis','Femoralis','Brachialis','vd_1','vd_2','vd_3']
passive = ['DATUM','TIJD','PATNR','SPOED','lbl']

out = pd.read_csv('wndinput.csv',header=0,low_memory=False,encoding='ISO-8859-1')
dfcard = pd.read_csv('/home/kerkt02/patdata/DM_CARDIOLOGIE.csv',header=0,low_memory=False,encoding='ISO-8859-1')


concatdict = {'CTROP':'TROP',':CKDEPI':':MDRD'}

print(rawdata)

def oneHotEncoder():
	catdf = pd.DataFrame()
	for feature in headerscard:
		dummies = pd.get_dummies(dfcard[feature],prefix=feature)
		catdf[dummies.columns] = dummies.astype(np.float32)
	catdf['PATNR'] = dfcard['PATNR']
	return catdf

def normalizer(ndf):
	normdf = pd.DataFrame()
	for feature in ndf:
		ndf[feature] = normalizeData(ndf[feature])
		ndf[feature] = ndf[feature].astype(np.float32)
	normdf = ndf
	return normdf

def normalizeData(df):
	x = df.values.astype(float)
	x = pd.DataFrame(x)
	mms = skp.MinMaxScaler()
	x_s = mms.fit_transform(x)
	df = pd.DataFrame(x_s)
	return df
def sift(df):
	sifting = dd(list)
	dfdict = dd(list)
	for patnr in set(df['PATNR']):
		sifting.clear()
		for index, row in df[df['PATNR'] == patnr].iterrows():
			for feature in df:
				sifting[feature].append(row[feature])
		for key in sifting.keys():
			appended = False
			for value in sifting[key]:
				if value != np.nan and not appended:
					dfdict[key].append(value)
					appended = True
			if not appended:
				dfdict[key].append(np.nan)
	dictdf = pd.DataFrame.from_dict(dfdict)
	dictdf.dropna(thresh=20, inplace=True)
	print(dictdf.apply(lambda x: x.count(), axis=0))
	return dictdf

def concatcolumns(df):
	urgency = []
	for index,item in df['CTROP'].astype(np.float32).iteritems():
		if np.isnan(item):
			urgency.append(1)
		else:
			urgency.append(0)
	print(urgency)
	df['SPOED'] = urgency
	for key in concatdict.keys():
		df[key].fillna(df[concatdict[key]],inplace=True)
		df.drop(concatdict[key],axis=1,inplace=True)
	return df
def imputer(predf):
	df = pd.DataFrame(KNN(3).complete(predf))
	df.columns = predf.columns
	df = df[(np.abs(stats.zscore(df)) < 10).all(axis=1)]
	return df

def finalizedata(df):
	df = imputer(df)
	df = normalizer(df)
	return df

def main():
	#dictdf = sift(rawdata)
	#print(dictdf)
	#dictdf.to_csv(path_or_buf='output.csv',index=False)
	#newdf = pd.DataFrame(dictdf[headers])
	#print(normalizer(newdf))
	#outskipstep = concatcolumns(out)

	out.dropna(thresh=28, inplace=True)
	print(out.apply(lambda x: x.count(), axis=0))
	for feature in out:
		print(feature)
		print(out[feature].dtype)

	finalout = finalizedata(out[headers])
	finalout = pd.concat([out[passive],finalout],axis=1,ignore_index=False)
	finalout.dropna(how='any',inplace=True)
	catdf = oneHotEncoder()
	catdf.drop_duplicates(subset='PATNR',inplace=True)
	next = pd.merge(finalout, catdf, on='PATNR')
	print(next)
	next.to_csv(path_or_buf='inputwnd.csv',index=False)

main()
