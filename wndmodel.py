import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dfinput = pd.read_csv('trainsetwnd.csv',header=0,low_memory=False,encoding='ISO-8859-1')
dfeval = pd.read_csv('testsetwnd.csv',header=0,low_memory=False,encoding='ISO-8859-1')

catheaders = ['SPOED','Geslacht_Man', 'Geslacht_Vrou','DiagnoseCode_203', 'DiagnoseCode_204', 'DiagnoseCode_205','vrgeschiedenis_myochardinfarct_Ja', 'vrgeschiedenis_PCI_Ja','vrgeschiedenis_CABG_Ja','vrgeschiedenis_atriumfibrilleren_Ja','riscf_familieanamnese_Ja', 'riscf_hypertensie_Ja','riscf_hypercholesterolemie_Ja', 'riscf_diabetes_Ja','Radialis_Ja', 'vd_1_Ja','vd_2_Ja', 'vd_3_Ja']
conheaders = ['CKDEPI','ALAT','ASAT','CHOL','CK','CRP','ERY','HB','HDLC','HINDEX','HT','K','KR','LD','LDLC','LEU','LINDEX','MCH','MCHC','MCV1','NA','RATI','RDWC','TG','TRO','UR']
target = 'lbl'
dfinput[catheaders] = dfinput[catheaders].astype(np.int64)
dfeval[catheaders] = dfeval[catheaders].astype(np.int64)

def initlbls():
	con = []
	cat = []
	for item in conheaders:
		con.append(tf.feature_column.numeric_column(item))
	for item in catheaders:
		cat.append(tf.feature_column.categorical_column_with_vocabulary_list(item,[1,0]))
	return con, cat


def input_fn(data, num_epochs=None, shuffle=True):
	return tf.estimator.inputs.pandas_input_fn(
		x=data[data.columns.difference([target])],
		y=data[target],
		num_epochs=num_epochs,
		shuffle=shuffle,
		batch_size=40)

def main():
	con, cat = initlbls()

	model = tf.estimator.DNNLinearCombinedClassifier(linear_feature_columns=cat,linear_optimizer='Adagrad', dnn_feature_columns=con,dnn_optimizer='Adagrad',dnn_hidden_units=[40,70,40],model_dir="/tmp/wnd")
	print(model)
	model.train(input_fn=input_fn(dfinput), steps=200)

	ev = model.evaluate(input_fn=input_fn(dfeval, num_epochs=1,shuffle=False))
	accuracy = ev["accuracy"]
	step = ev["global_step"]
	print("accuracy:  -- step:  ")
	print(str(accuracy), str(step))

main()
