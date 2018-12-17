import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.manifold import Isomap
import os
np.random.seed(0)

model_name = os.path.basename(__file__).split(".")[0]

# Defining Evaluation Metric
def accuracy(y_true, y_pred):
	y_true[y_true==0]=-1
	y_pred[y_pred<0.5] = -1
	y_pred[y_pred>0.5] = 1
	y_pred[y_pred==0.5] = -1

	accuracy = accuracy_score(y_true, y_pred)
	eval_accuracy = 6*np.absolute(accuracy-0.5)
	return accuracy, eval_accuracy

# Defining Classifier
def Classifier(x_train,x_test,y_train,y_test, test):
	dtrain = xgb.DMatrix(x_train, y_train)
	dvalid = xgb.DMatrix(x_test, y_test)
	params = {
		"objective": "binary:logistic",
		"booster": "gbtree",
		# "eval_metric": "logloss",
		"eta": 0.000002,
		"max_depth": 9,
		"colsample_bytree": 0.50,
		"colsample_bylevel":0.5,
		"subsample":0.55,
		"gamma":0.05,
		"lambda":0.7,
		"tree_method":"exact",
		"silent": 1,
		"num_parallel_tree":3,
		"seed":321
	}
	watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
	gbm = xgb.train(params, dtrain, 1200, evals=watchlist, verbose_eval=True, early_stopping_rounds =30)
	return gbm.predict(xgb.DMatrix(x_test), ntree_limit = gbm.best_iteration), gbm.predict(xgb.DMatrix(test), ntree_limit = gbm.best_iteration)

# Laoding Data
train = pd.read_csv("../../data/train.csv")
test = pd.read_csv("../../data/test.csv")

target = train["Y"]
target[target == -1]=0
target = target[1:].values

test_time = test["Time"]

train.drop(["Y","Time"], axis=1, inplace=True)
test.drop(["Time"], axis=1, inplace=True)

data = pd.concat([train, test],axis=0)
print data.shape

# Adding Percentage change and Lag Varibels
changes = data.diff()/data
data = pd.concat([data, changes, changes.shift(-1),changes.shift(1)], axis=1)

train = data[1:3000].as_matrix()
test = data[3000:].as_matrix()
del data

print train.shape, test.shape

# Running KFold
prediction =[]
overall = np.zeros(train.shape[0])

kf = KFold(train.shape[0], n_folds =20, shuffle=True, random_state=0)
for i, (train_index, test_index) in enumerate(kf):
	print "Starting Round: ",i
	x_train , y_train = train[train_index], target[train_index]
	x_test, y_test = train[test_index], target[test_index]

	cv_pred, test_pred = Classifier(x_train,x_test,y_train,y_test, test)
	overall[test_index] = cv_pred
	print "ROC AUC Score",roc_auc_score(y_test, cv_pred)
	print "Accuracy, Eval Accuracy : ",accuracy(y_test, cv_pred)
	prediction.append(test_pred)
print "Overall ROC AUC Score and Mean: ", roc_auc_score(target, overall), np.mean(overall)

prediction = np.average(np.array(prediction), axis =0)
prediction[prediction<0.5] = -1
prediction[prediction>0.5] = 1
prediction[prediction==0.5] = -1


acc, eval_accuracy = accuracy(target, overall)
print "Overall Accuracy, Eval Accuracy : ", acc, eval_accuracy
eval_accuracy = str(eval_accuracy).replace(".","_")

pd.DataFrame({"Time":test_time, "Y": prediction}).to_csv("csv/"+eval_accuracy+"_"+model_name+".txt", index = False)
