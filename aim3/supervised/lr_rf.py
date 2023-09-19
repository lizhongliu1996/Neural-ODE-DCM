import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("../data2/sub_features.csv")

X = data[["interview_age", "traumascore", "efficiency", "assortativity", "transitivity"]]
y = data[["alc_sip_ever"]]

dim = data.shape[0]
index = round(dim*0.75)
idx_train = range(index)
idx_test = range(index, dim)

X_train = X.iloc[idx_train,:]
y_train = y.iloc[idx_train,:]
X_test = X.iloc[idx_test,:]
y_test = y.iloc[idx_test,:]

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
training_acc = log_reg.score(X_train, y_train)
test_acc = log_reg.score(X_test, y_test)
p_pred = log_reg.predict_proba(X_test)[:, 1]
auc_score = metrics.roc_auc_score(y_test, p_pred)

tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()
false_alarm_rate_lr = fp / (fp + tn)

precision_lr = tp / (tp + fp)
recall_lr = tp / (tp + fn)
f1_score_lr = 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr)

print("Logistic Regression:")
print("test_acc", test_acc)
print("auc_score",auc_score)
print("false_alarm_rate", false_alarm_rate_lr)
print("f1_score", f1_score_lr)

rf1 = RandomForestClassifier().fit(X_train, y_train)

pred_rf = rf1.predict(X_test)
test_acc_rf = rf1.score(X_test, y_test)
prob_rf = rf1.predict_proba(X_test)[:,1]
auc_rf = metrics.roc_auc_score(y_test, prob_rf)

tn_rf, fp_rf, fn_rf, tp_rf = metrics.confusion_matrix(y_test, pred_rf).ravel()
false_alarm_rate_rf = fp_rf / (fp_rf + tn_rf)

precision_rf = tp_rf / (tp_rf + fp_rf)
recall_rf = tp_rf / (tp_rf + fn_rf)
f1_score_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

print("\nRandom Forest:")
print("test_acc", test_acc_rf)
print("auc_score", auc_rf)
print("false_alarm_rate", false_alarm_rate_rf)
print("f1_score", f1_score_rf)