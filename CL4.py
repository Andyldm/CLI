import numpy as np
import pandas as pd
import copy
import random
import xlwt
import csv
from sklearn.model_selection import KFold, StratifiedKFold
from collections import Counter
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from cleanlab.classification import LearningWithNoisyLabels
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from cleanlab.pruning import get_noise_indices
import warnings
from cleanlab.latent_estimation import (
    compute_confident_joint,
    estimate_latent,
)

#不平衡：rus

def get_noise(label_new,pre_new,xall_new,X_test,y_test):
    label_new1=copy.deepcopy(label_new)
    pre_new1 = copy.deepcopy(pre_new)
    xall_new1 = copy.deepcopy(xall_new)
    XX = copy.deepcopy(X_test)
    y_test_no = copy.deepcopy(y_test)



    label_1 = label_new1.ravel()
    y_train2 = label_1.astype(np.int16)
    confident_joint = compute_confident_joint(
        s=y_train2,
        psx=pre_new1,  # P(s = k|x)
        thresholds=None
    )
    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=y_train2,
        py_method='cnt',
        converge_latent_estimates=False,
    )

    ordered_label_errors = get_noise_indices(
        s=y_train2,
        psx=pre_new1,
        inverse_noise_matrix=inv_noise_matrix,
        confident_joint=confident_joint,
        prune_method='prune_by_noise_rate',
    )
    # print(ordered_label_errors)

    x_mask = ~ordered_label_errors
    x_pruned = xall_new1[x_mask]
    # print(label_new_2)
    s_pruned = y_train2[x_mask]

    sample_weight = np.ones(np.shape(s_pruned))
    for k in range(2):
        sample_weight_k = 1.0 / noise_matrix[k][k]
        sample_weight[s_pruned == k] = sample_weight_k

    log_reg = LogisticRegression(solver='liblinear')
    # log_reg1 = LogisticRegression(solver='liblinear')
    log_reg.fit(x_pruned, s_pruned, sample_weight=sample_weight)
    pre1 = log_reg.predict(XX)
    y_test_11 = y_test_no.ravel()
    y_original = metrics.f1_score(y_test_11, pre1, pos_label=1, average="binary")

    prob = log_reg.predict_proba(XX)
    thresholds = metrics.roc_auc_score(y_test_11, prob[:, -1])
    acc = accuracy_score(y_test_11, pre1)

    mcc = matthews_corrcoef(y_test_11, pre1)

    return y_original, thresholds, acc, mcc




warnings.filterwarnings('ignore')

csv_order = {0: 'activemq', 1: 'camel', 2: 'derby', 3: 'geronimo', 4: 'hbase', 5: 'hcommon', 6: 'mahout', 7: 'openjpa',
             8: 'pig', 9: 'tuscany'}
csv_num = {0: 1245, 1: 2018, 2: 1153, 3: 1856, 4: 1681, 5: 2670, 6: 420, 7: 692, 8: 467, 9: 1506}


#置信学习例子，使用sklearner库，但没有使用封装的cleanlab函数
def con_learn():
    file_name = 'rus.csv'
    f = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["datasets", "f1-ori", "f1-cl", "f1-cli", "auc-ori", "auc-cl", "auc-cli", "acc-ori", "acc-cl",
                         "acc-cli", "mcc-ori", "mcc-cl", "mcc-cli"])
    #第一列是原始f值，第二列是运用置信学习去噪的结果，第三列是不平衡+置信学习
    for i in range(10):
        csv_string = csv_order[i]
        dataframe = pd.read_csv('data1/' + csv_string + '.csv')
        v = dataframe.iloc[:]

        train_v = np.array(v)
        # print(train_v )
        ori_all = []
        ori_all.append(csv_string)

        ob = train_v[:, 0:14]
        label = train_v[:, -1]
        label = label.reshape(-1, 1)

        yor_all = []
        yor_all1 = []
        yor_all2 = []

        auc_all = []
        auc_all1 = []
        auc_all2 = []

        acc_all = []
        acc_all1 = []
        acc_all2 = []

        mcc_all = []
        mcc_all1 = []
        mcc_all2 = []
        seed=[94733,16588,1761,59345,27886,80894,22367,65435,96636,89300]
        for ix in range(10):
            sfolder = StratifiedKFold(n_splits=5, shuffle=True,random_state= seed[ix])
            y_or = []            #原始f值
            y_or1 = []           #CL f值
            y_or2 = []           #CLI f值

            auc_or = []  # 原始auc值
            auc_or1 = []  # CL auc值
            auc_or2 = []  # CLI auc值

            acc_or = []  # 原始acc值
            acc_or1 = []  # CL acc值
            acc_or2 = []  # CLI acc值

            mcc_or = []  # 原始mcc值
            mcc_or1 = []  # CL mcc值
            mcc_or2 = []  # CLI mcc值
            for train_index, test_index in sfolder.split(ob, label):
                X_train, X_test = ob[train_index], ob[test_index]
                y_train, y_test = label[train_index], label[test_index]

                psx = np.zeros((len(y_train), 2))
                psx1 = np.zeros((len(y_train), 2))

                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed[ix])
                for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X_train, y_train)):

                    # Select the training and holdout cross-validated sets.
                    X_train_cv, X_holdout_cv = X_train[cv_train_idx], X_train[cv_holdout_idx]
                    s_train_cv, s_holdout_cv = y_train[cv_train_idx], y_train[cv_holdout_idx]

                    # Fit the clf classifier to the training set and
                    # predict on the holdout set and update psx.
                    log_reg = LogisticRegression(solver='liblinear')
                    log_reg.fit(X_train_cv, s_train_cv)
                    psx_cv = log_reg.predict_proba(X_holdout_cv)  # P(s = k|x) # [:,1]
                    psx[cv_holdout_idx] = psx_cv



                    rus = RandomUnderSampler(random_state=seed[ix])
                    X_resampled, y_resampled = rus.fit_sample(X_train_cv, s_train_cv)
                    #print(Counter(y_resampled))
                    log_reg = LogisticRegression(solver='liblinear')
                    log_reg.fit(X_resampled, y_resampled)
                    psx_cv1 = log_reg.predict_proba(X_holdout_cv)
                    psx1[cv_holdout_idx] = psx_cv1




                log_reg = LogisticRegression(solver='liblinear')
                log_reg.fit(X_train, y_train)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or.append(y_original)
                prob = log_reg.predict_proba(X_test)
                thresholds = metrics.roc_auc_score(y_test_1, prob[:, -1])
                auc_or.append(thresholds)
                acc = accuracy_score(y_test_1, pre1)
                acc_or.append(acc)
                mcc = matthews_corrcoef(y_test_1, pre1)
                mcc_or.append(mcc)

                y_original, thresholds1, acc1, mcc1 = get_noise(y_train, psx, X_train, X_test, y_test)
                y_or1.append(y_original)
                auc_or1.append(thresholds1)
                acc_or1.append(acc1)
                mcc_or1.append(mcc1)

                y_original, thresholds2, acc2, mcc2 = get_noise(y_train, psx1, X_train, X_test, y_test)
                y_or2.append(y_original)
                auc_or2.append(thresholds2)
                acc_or2.append(acc2)
                mcc_or2.append(mcc2)

            yor_all.append(np.mean(y_or))
            yor_all1.append(np.mean(y_or1))
            yor_all2.append(np.mean(y_or2))

            auc_all.append(np.mean(auc_or))
            auc_all1.append(np.mean(auc_or1))
            auc_all2.append(np.mean(auc_or2))

            acc_all.append(np.mean(acc_or))
            acc_all1.append(np.mean(acc_or1))
            acc_all2.append(np.mean(acc_or2))

            mcc_all.append(np.mean(mcc_or))
            mcc_all1.append(np.mean(mcc_or1))
            mcc_all2.append(np.mean(mcc_or2))

        ori_all.append(np.mean(yor_all))
        ori_all.append(np.mean(yor_all1))
        ori_all.append(np.mean(yor_all2))

        ori_all.append(np.mean(auc_all))
        ori_all.append(np.mean(auc_all1))
        ori_all.append(np.mean(auc_all2))

        ori_all.append(np.mean(acc_all))
        ori_all.append(np.mean(acc_all1))
        ori_all.append(np.mean(acc_all2))

        ori_all.append(np.mean(mcc_all))
        ori_all.append(np.mean(mcc_all1))
        ori_all.append(np.mean(mcc_all2))
        csv_writer.writerow(ori_all)
        print(csv_string + " is done~!")



if __name__ == '__main__':
    con_learn()