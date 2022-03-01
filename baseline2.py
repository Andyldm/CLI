import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import OneSidedSelection
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import warnings


#baseline1:RENN

warnings.filterwarnings('ignore')

csv_order = {0: 'activemq', 1: 'camel', 2: 'derby', 3: 'geronimo', 4: 'hbase', 5: 'hcommon', 6: 'mahout', 7: 'openjpa',
             8: 'pig', 9: 'tuscany'}
csv_num = {0: 1245, 1: 2018, 2: 1153, 3: 1856, 4: 1681, 5: 2670, 6: 420, 7: 692, 8: 467, 9: 1506}


#置信学习例子，使用sklearner库，但没有使用封装的cleanlab函数
def con_learn():
    file_name = 'OneSidedSelection.csv'
    f = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["datasets", "f1-ori", "f1-renn", "mcc-ori", "mcc-renn"])
    for i in range(10):
        csv_string = csv_order[i]
        dataframe = pd.read_csv('dataset/' + csv_string + '.csv')
        v = dataframe.iloc[:]

        train_v = np.array(v)
        ori_all = []
        ori_all.append(csv_string)

        ob = train_v[:, 0:14]
        label = train_v[:, -1]
        label = label.reshape(-1, 1)

        yor_all = []
        yor_all1 = []

        mcc_all = []
        mcc_all1 = []
        seed=[94733,16588,1761,59345,27886,80894,22367,65435,96636,89300]
        for ix in range(10):
            sfolder = StratifiedKFold(n_splits=5, shuffle=True,random_state= seed[ix])
            y_or = []            #原始f值
            y_or1 = []           #CL f值

            mcc_or = []  # 原始mcc值
            mcc_or1 = []  # CL mcc值
            for train_index, test_index in sfolder.split(ob, label):
                X_train, X_test = ob[train_index], ob[test_index]
                y_train, y_test = label[train_index], label[test_index]

                renn = OneSidedSelection(random_state=seed[ix])
                x_pruned,s_pruned = renn.fit_sample(X_train,y_train)

                log_reg = RandomForestClassifier(random_state=seed[ix])
                log_reg.fit(X_train, y_train)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or.append(y_original)
                print(y_original)
                mcc = matthews_corrcoef(y_test_1, pre1)
                mcc_or.append(mcc)

                log_reg1 = RandomForestClassifier(random_state=seed[ix])
                log_reg1.fit(x_pruned, s_pruned)
                pre1 = log_reg1.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or1.append(y_original)
                mcc = matthews_corrcoef(y_test_1, pre1)
                mcc_or1.append(mcc)


            yor_all.append(np.mean(y_or))
            yor_all1.append(np.mean(y_or1))

            mcc_all.append(np.mean(mcc_or))
            mcc_all1.append(np.mean(mcc_or1))

        ori_all.append(np.mean(yor_all))
        ori_all.append(np.mean(yor_all1))
        ori_all.append(np.mean(mcc_all))
        ori_all.append(np.mean(mcc_all1))
        csv_writer.writerow(ori_all)
        print(csv_string + " is done~!")



if __name__ == '__main__':
    con_learn()