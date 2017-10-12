import numpy as np

def accuracy_precision_recall_f1(yp,y_true):
	assert len(yp) == len(y_true)
	labels = set(y_true)
	metrics = np.zeros((4, len(labels)))
	for c in labels:
		tp_c = np.intersec1d(yp==c,y_true==c)
		fp_c = np.intersec1d(yp!=c,y_true==c)
		tn_c = np.sum(y_true!=c)
		tp_c = np.sum(y_true==c)
		metrics[:,c] = np.array([tp_c,fp_c,tn_c,tp_c])
