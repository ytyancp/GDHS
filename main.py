import os
import csv
import numpy as np
from classifier import DecisionTree as CLF
from sklearn import metrics
from RBMetrics import MAvG as mGMcount
from RBMetrics import MAUC as MAUCcount
from sklearn.model_selection import StratifiedKFold

from GDHS_LC import GDHS_LC
# from GDHS_GL import GDHS_GL
# from GDHS_BA import GDHS_BA

if __name__ == '__main__':
    path = "D:\\14_dataset\\"
    files = os.listdir(path)
    filename = ['D:\\GDHS\\experiment\\GDHS_LC.xls']
    for filename1 in filename:
        dataset_excel = open(filename1, 'a+', encoding='gbk')

        dataset_excel.write('\t')
        dataset_excel.write('mGM')
        dataset_excel.write('\t')
        dataset_excel.write('MAUC')
        dataset_excel.write('\n')

        for s in range(len(files)):
            data = []

            name = files[s]
            name = name.replace('.csv', '')
            print(name)

            with open(path + "\\" + files[s], 'r') as myFile:
                lines = csv.reader(myFile)
                for line in lines:
                    data1 = []
                    for j in range(len(line)):
                        data1.append(float(line[j]))
                    data.append(data1)

            index = 0
            average_mGM = 0
            average_MAUC = 0

            for l in range(10):
                kf = StratifiedKFold(n_splits=5, shuffle=True)
                X = []
                y = []
                size = len(data)
                for j in range(len(data)):
                    pattern = data[j].copy()
                    label = pattern.pop()
                    y.append(label)
                    X.append(pattern)
                kf.get_n_splits(X, y)

                for train_index, test_index in kf.split(X, y):
                    train_data = []
                    test_data = []
                    for i in range(len(train_index)):
                        train_data.append(data[train_index[i]])
                    for i in range(len(test_index)):
                        test_data.append(data[test_index[i]])

                    train_label = []
                    train_pattern = []
                    for i in range(len(train_data)):
                        pattern = train_data[i].copy()
                        train_label.append(pattern[-1])
                        train_pattern.append(pattern[:-1])

                    test_label = []
                    test_pattern = []
                    for i in range(len(test_data)):
                        pattern = test_data[i].copy()
                        test_label.append(pattern[-1])
                        test_pattern.append(pattern[:-1])

                    gdhs = GDHS_LC(k1=5, k2=5, k3=5, w=0.8)

                    train_pattern_resampled, train_label_resampled = gdhs.fit_transform(np.array(train_pattern),
                                                                                        np.array(train_label))
                    result = CLF(train_pattern_resampled, train_label_resampled, test_pattern)

                    cm = metrics.confusion_matrix(test_label, result)
                    mGM = mGMcount(cm)
                    MAUC = MAUCcount(cm)

                    average_mGM = average_mGM + mGM
                    average_MAUC = average_MAUC + MAUC
                    index = index + 1

                    print('training over')

            average_mGM = average_mGM / index
            average_MAUC = average_MAUC / index

            dataset_excel = open(filename1, 'a+', encoding='gbk')
            dataset_excel.write(name)
            dataset_excel.write('\t')
            dataset_excel.write(str(average_mGM))
            dataset_excel.write('\t')
            dataset_excel.write(str(average_MAUC))
            dataset_excel.write('\n')

            print('ok')



