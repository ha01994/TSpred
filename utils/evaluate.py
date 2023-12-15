# Source: DeepAAI (https://github.com/enai4bio/DeepAAI/blob/main/metrics/evaluate.py)
import numpy as np
from sklearn.metrics import f1_score
from sklearn import metrics
from scipy import stats
from sklearn.metrics import confusion_matrix


def evaluation_metrics(predict_proba, label):
    auroc = metrics.roc_auc_score(y_true=label, y_score=predict_proba)
    auprc = metrics.average_precision_score(label, predict_proba)
        
    return auroc, auprc


def evaluation_metrics2(predict_proba, label):
    predicts = np.around(predict_proba)    
    f1 = metrics.f1_score(y_true=label, y_pred=predicts)
    acc = metrics.accuracy_score(y_true=label, y_pred=predicts)
    pre = metrics.precision_score(y_true=label, y_pred=predicts)
    recall = metrics.recall_score(y_true=label, y_pred=predicts)

    conf_matrix = confusion_matrix(label, predicts)   
    tn, fp, fn, tp = conf_matrix.ravel()    
    spe = tn / (tn + fp)
            
    return acc,pre,spe,recall,f1


