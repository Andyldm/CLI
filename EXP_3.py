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
from sklearn.metrics import accuracy_score
from cleanlab.classification import LearningWithNoisyLabels
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from cleanlab.pruning import get_noise_indices
import warnings
from cleanlab.latent_estimation import (
    compute_confident_joint,
    estimate_latent,
)

#实验6

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

def get_psx(g_x,g_y,seed_ix):
    psx = np.zeros((len(g_y), 2))
    psx1 = np.zeros((len(g_y), 2))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_ix)
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(g_x, g_y)):
        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv = g_x[cv_train_idx], g_x[cv_holdout_idx]
        s_train_cv, s_holdout_cv = g_y[cv_train_idx], g_y[cv_holdout_idx]

        # Fit the clf classifier to the training set and
        # predict on the holdout set and update psx.
        log_reg = LogisticRegression(solver='liblinear')
        log_reg.fit(X_train_cv, s_train_cv)
        psx_cv = log_reg.predict_proba(X_holdout_cv)  # P(s = k|x) # [:,1]
        psx[cv_holdout_idx] = psx_cv

        rus = RandomUnderSampler(random_state=seed_ix)
        X_resampled, y_resampled = rus.fit_sample(X_train_cv, s_train_cv)
        # print(Counter(y_resampled))
        log_reg = LogisticRegression(solver='liblinear')
        log_reg.fit(X_resampled, y_resampled)
        psx_cv1 = log_reg.predict_proba(X_holdout_cv)
        psx1[cv_holdout_idx] = psx_cv1
    return psx,psx1





warnings.filterwarnings('ignore')

csv_order = {0: 'activemq', 1: 'camel', 2: 'derby', 3: 'geronimo', 4: 'hbase', 5: 'hcommon', 6: 'mahout', 7: 'openjpa',
             8: 'pig', 9: 'tuscany'}
csv_num = {0: 1245, 1: 2018, 2: 1153, 3: 1856, 4: 1681, 5: 2670, 6: 420, 7: 692, 8: 467, 9: 1506}


#置信学习例子，使用sklearner库，但没有使用封装的cleanlab函数
def con_learn():
    #第一列是原始f值，第二列是运用置信学习去噪的结果，第三列是不平衡+置信学习
    for i in range(10):
        csv_string = csv_order[i]
        dataframe = pd.read_csv('data2/' + csv_string + '.csv')
        v = dataframe.iloc[:]

        train_v = np.array(v)

        ob = train_v[:, 0:14]
        #label = train_v[:,14: ]

        label_RA=train_v[:,-1 ]
        label_MA = train_v[:, -2]
        label_AG = train_v[:, -3]
        label_B = train_v[:, -4]

        yor_all = [[],[],[],[]]
        yor_all1 = [[],[],[],[]]
        yor_all2 = [[],[],[],[]]

        yor_all_B = [[],[],[],[]]
        yor_all1_B = [[],[],[],[]]
        yor_all2_B = [[],[],[],[]]

        yor_all_AG = [[],[],[],[]]
        yor_all1_AG = [[],[],[],[]]
        yor_all2_AG = [[],[],[],[]]

        yor_all_MA = [[],[],[],[]]
        yor_all1_MA = [[],[],[],[]]
        yor_all2_MA = [[],[],[],[]]

        seed=[94733,16588,1761,59345,27886,80894,22367,65435,96636,89300]
        for ix in range(10):
            sfolder = KFold(n_splits=5, shuffle=True,random_state= seed[ix])
            y_or = [[],[],[],[]]            #原始f值,AUC,ACC,MCC
            y_or1 = [[],[],[],[]]           #CL f值
            y_or2 = [[],[],[],[]]           #CLI f值

            y_or_B = [[],[],[],[]]          # 原始f值
            y_or1_B = [[],[],[],[]]         # CL f值
            y_or2_B = [[],[],[],[]]         # CLI f值

            y_or_AG = [[],[],[],[]]         # 原始f值
            y_or1_AG = [[],[],[],[]]        # CL f值
            y_or2_AG = [[],[],[],[]]        # CLI f值

            y_or_MA = [[],[],[],[]]         # 原始f值
            y_or1_MA = [[],[],[],[]]        # CL f值
            y_or2_MA = [[],[],[],[]]        # CLI f值
            for train_index, test_index in sfolder.split(ob, label_RA):
                X_train, X_test = ob[train_index], ob[test_index]           #划分训练和测试集
                y_train_RA, y_test = label_RA[train_index], label_RA[test_index]
                y_train_B = label_B[train_index]
                y_train_AG = label_AG[train_index]
                y_train_MA = label_MA[train_index]


                psx_B,psx1_B = get_psx(X_train,y_train_B,seed[ix])
                psx_AG, psx1_AG = get_psx(X_train, y_train_AG, seed[ix])
                psx_MA, psx1_MA = get_psx(X_train, y_train_MA, seed[ix])
                psx_RA, psx1_RA = get_psx(X_train, y_train_RA, seed[ix])



                log_reg = LogisticRegression(solver='liblinear')
                log_reg.fit(X_train, y_train_RA)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or[0].append(y_original)
                prob = log_reg.predict_proba(X_test)
                thresholds = metrics.roc_auc_score(y_test_1, prob[:, -1])
                y_or[1].append(thresholds)
                acc = accuracy_score(y_test_1, pre1)
                y_or[2].append(acc)
                mcc = matthews_corrcoef(y_test_1, pre1)
                y_or[3].append(mcc)
                y_original,auc,acc,mcc = get_noise(y_train_RA,psx_RA,X_train,X_test,y_test)
                y_or1[0].append(y_original)
                y_or1[1].append(auc)
                y_or1[2].append(acc)
                y_or1[3].append(mcc)
                y_original,auc,acc,mcc = get_noise(y_train_RA,psx1_RA,X_train,X_test,y_test)
                y_or2[0].append(y_original)
                y_or2[1].append(auc)
                y_or2[2].append(acc)
                y_or2[3].append(mcc)

                log_reg = LogisticRegression(solver='liblinear')
                log_reg.fit(X_train, y_train_B)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or_B[0].append(y_original)
                prob = log_reg.predict_proba(X_test)
                thresholds = metrics.roc_auc_score(y_test_1, prob[:, -1])
                y_or_B[1].append(thresholds)
                acc = accuracy_score(y_test_1, pre1)
                y_or_B[2].append(acc)
                mcc = matthews_corrcoef(y_test_1, pre1)
                y_or_B[3].append(mcc)
                y_original,auc,acc,mcc = get_noise(y_train_B, psx_B, X_train, X_test, y_test)
                y_or1_B[0].append(y_original)
                y_or1_B[1].append(auc)
                y_or1_B[2].append(acc)
                y_or1_B[3].append(mcc)
                y_original,auc,acc,mcc = get_noise(y_train_B, psx1_B, X_train, X_test, y_test)
                y_or2_B[0].append(y_original)
                y_or2_B[1].append(auc)
                y_or2_B[2].append(acc)
                y_or2_B[3].append(mcc)

                log_reg = LogisticRegression(solver='liblinear')
                log_reg.fit(X_train, y_train_AG)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or_AG[0].append(y_original)
                prob = log_reg.predict_proba(X_test)
                thresholds = metrics.roc_auc_score(y_test_1, prob[:, -1])
                y_or_AG[1].append(thresholds)
                acc = accuracy_score(y_test_1, pre1)
                y_or_AG[2].append(acc)
                mcc = matthews_corrcoef(y_test_1, pre1)
                y_or_AG[3].append(mcc)
                y_original,auc,acc,mcc  = get_noise(y_train_AG, psx_AG, X_train, X_test, y_test)
                y_or1_AG[0].append(y_original)
                y_or1_AG[1].append(auc)
                y_or1_AG[2].append(acc)
                y_or1_AG[3].append(mcc)
                y_original, auc, acc, mcc = get_noise(y_train_AG, psx1_AG, X_train, X_test, y_test)
                y_or2_AG[0].append(y_original)
                y_or2_AG[1].append(auc)
                y_or2_AG[2].append(acc)
                y_or2_AG[3].append(mcc)

                log_reg = LogisticRegression(solver='liblinear')
                log_reg.fit(X_train, y_train_MA)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or_MA[0].append(y_original)
                prob = log_reg.predict_proba(X_test)
                thresholds = metrics.roc_auc_score(y_test_1, prob[:, -1])
                y_or_MA[1].append(thresholds)
                acc = accuracy_score(y_test_1, pre1)
                y_or_MA[2].append(acc)
                mcc = matthews_corrcoef(y_test_1, pre1)
                y_or_MA[3].append(mcc)
                y_original,auc,acc,mcc = get_noise(y_train_MA, psx_MA, X_train, X_test, y_test)
                y_or1_MA[0].append(y_original)
                y_or1_MA[1].append(auc)
                y_or1_MA[2].append(acc)
                y_or1_MA[3].append(mcc)
                y_original,auc,acc,mcc  = get_noise(y_train_MA, psx1_MA, X_train, X_test, y_test)
                y_or2_MA[0].append(y_original)
                y_or2_MA[1].append(auc)
                y_or2_MA[2].append(acc)
                y_or2_MA[3].append(mcc)

            for index_i in range(4):
                yor_all[index_i].append(np.mean(y_or[index_i]))
            for index_i in range(4):
                yor_all1[index_i].append(np.mean(y_or1[index_i]))
            for index_i in range(4):
                yor_all2[index_i].append(np.mean(y_or2[index_i]))


            for index_i in range(4):
                yor_all_B[index_i].append(np.mean(y_or_B[index_i]))
            for index_i in range(4):
                yor_all1_B[index_i].append(np.mean(y_or1_B[index_i]))
            for index_i in range(4):
                yor_all2_B[index_i].append(np.mean(y_or2_B[index_i]))

            for index_i in range(4):
                yor_all_AG[index_i].append(np.mean(y_or_AG[index_i]))
            for index_i in range(4):
                yor_all1_AG[index_i].append(np.mean(y_or1_AG[index_i]))
            for index_i in range(4):
                yor_all2_AG[index_i].append(np.mean(y_or2_AG[index_i]))

            for index_i in range(4):
                yor_all_MA[index_i].append(np.mean(y_or_MA[index_i]))
            for index_i in range(4):
                yor_all1_MA[index_i].append(np.mean(y_or1_MA[index_i]))
            for index_i in range(4):
                yor_all2_MA[index_i].append(np.mean(y_or2_MA[index_i]))


        f = open("RA.csv", 'a')
        f.write("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (csv_string, np.mean(yor_all[0]),
            np.mean(yor_all1[0]),np.mean(yor_all2[0]),np.mean(yor_all[1]),np.mean(yor_all1[1]),np.mean(yor_all2[1]),
            np.mean(yor_all[2]),np.mean(yor_all1[2]),np.mean(yor_all2[2]),np.mean(yor_all[3]),np.mean(yor_all1[3]),
                                                                                      np.mean(yor_all2[3])))
        f.close()


        f = open("B.csv", 'a')
        f.write("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (csv_string, np.mean(yor_all_B[0]),
                                                                                      np.mean(yor_all1_B[0]),
                                                                                      np.mean(yor_all2_B[0]),
                                                                                      np.mean(yor_all_B[1]),
                                                                                      np.mean(yor_all1_B[1]),
                                                                                      np.mean(yor_all2_B[1]),
                                                                                      np.mean(yor_all_B[2]),
                                                                                      np.mean(yor_all1_B[2]),
                                                                                      np.mean(yor_all2_B[2]),
                                                                                      np.mean(yor_all_B[3]),
                                                                                      np.mean(yor_all1_B[3]),
                                                                                      np.mean(yor_all2_B[3])))
        f.close()


        f = open("AG.csv", 'a')
        f.write("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (csv_string, np.mean(yor_all_AG[0]),
                                                                                      np.mean(yor_all1_AG[0]),
                                                                                      np.mean(yor_all2_AG[0]),
                                                                                      np.mean(yor_all_AG[1]),
                                                                                      np.mean(yor_all1_AG[1]),
                                                                                      np.mean(yor_all2_AG[1]),
                                                                                      np.mean(yor_all_AG[2]),
                                                                                      np.mean(yor_all1_AG[2]),
                                                                                      np.mean(yor_all2_AG[2]),
                                                                                      np.mean(yor_all_AG[3]),
                                                                                      np.mean(yor_all1_AG[3]),
                                                                                      np.mean(yor_all2_AG[3])))
        f.close()


        f = open("MA.csv", 'a')
        f.write("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (csv_string, np.mean(yor_all_MA[0]),
                                                                                      np.mean(yor_all1_MA[0]),
                                                                                      np.mean(yor_all2_MA[0]),
                                                                                      np.mean(yor_all_MA[1]),
                                                                                      np.mean(yor_all1_MA[1]),
                                                                                      np.mean(yor_all2_MA[1]),
                                                                                      np.mean(yor_all_MA[2]),
                                                                                      np.mean(yor_all1_MA[2]),
                                                                                      np.mean(yor_all2_MA[2]),
                                                                                      np.mean(yor_all_MA[3]),
                                                                                      np.mean(yor_all1_MA[3]),
                                                                                      np.mean(yor_all2_MA[3])))
        f.close()

        print(csv_string+" is done~!")



if __name__ == '__main__':
    con_learn()