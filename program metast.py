from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

import pandas as pd
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor


Survival = pd.read_csv(r'C:\Users\redegator\PycharmProjects\hope\RNAs — MX.txt', delimiter = '\t', low_memory=False)
Survival.replace(',', '.', inplace=True, regex=True)


Survival = Survival.transpose()

Survival = Survival.drop(Survival.columns[0], axis=1)

# nx = pd.read_csv('NX working for Vlad.txt', header=None, names=['Nodes'], delimiter = '\t')
nx = pd.read_csv('metast.txt', delimiter = '\t', low_memory=False)


NUMBER_OF_RUNS = 20

with open('log.txt', 'a') as f:

    try:
        x = Survival.columns.get_loc('OR2B2')
        print(x)
        Survival = Survival.drop(Survival.columns[x], axis = 1)
    except:
        print("OR2B2 is not found")


    a = list(Survival)
    Y = nx['Survival']
    X = Survival[a]
    X = X[2::]          # delete Hugo_Symbol and Entrez_Gene_Id
    B = array(nx['Survival'])
    B = B.reshape(-1, 1)


    for i in range(NUMBER_OF_RUNS):
        print("----------------", file=f)
        print("RUN", i+1, file=f)

        print('Linear Regression R squared', file=f)
        print('Random Forest R squared', file=f)
        print('Random Forest RMSE', file=f)
        print('Decision Tree Accuracy', file=f)
        print('Decision Tree Sensitivity', file=f)
        print('Decision Tree Specificity', file=f)

        print('Random Forest Accuracy', file=f)
        print('Random Forest Precision', file=f)
        print('Random Forest F1-score:', file=f)

        # print(Survival.dtypes) #ПЕРЕВІРИТИ ТИП ДАНИХ

        # train and build a linear regression model
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
        from sklearn.linear_model import LinearRegression

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        # print('Linear Regression R squared": %.4f' % regressor.score(X_test, y_test), file=f)
        print('%.4f' % regressor.score(X_test, y_test), file=f)

        forest_reg = RandomForestRegressor(n_estimators=70, max_depth=5, random_state=42)
        forest_reg.fit(X_train, y_train)
        y_pred = forest_reg.predict(X_test)
        # print(y_pred, file=f)
        # print('Random Forest R squared": %.4f' % forest_reg.score(X_test, y_test), file=f)
        print('%.4f' % forest_reg.score(X_test, y_test), file=f)
        forest_mse = mean_squared_error(y_pred, y_test)
        forest_rmse = np.sqrt(forest_mse)
        # print('Random Forest RMSE: %.4f' % forest_rmse, file=f)
        print('%.4f' % forest_rmse, file=f)


        from sklearn.tree import DecisionTreeRegressor
        from sklearn.tree import DecisionTreeClassifier
        from sklearn import tree

        b = ['T1','T2', 'T3', 'T4']


        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(50, 50))
        tree.plot_tree(dtc, feature_names=a, class_names= b, label = 'root', filled=True)
        plt.savefig('Metast_run{}.pdf'.format(i+1))
        # plt.show()

        y_pred = dtc.predict(X_test)
        # print('Decision Tree Accuracy: %.4f' % accuracy_score(y_test, y_pred), file=f)
        print('%.4f' % accuracy_score(y_test, y_pred), file=f)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[1, 0]).ravel()

        sensitivity = np.nan_to_num(tp / (tp + fn))
        specificity = tn / (tn + fp)
        # print('Decision Tree Sensitivity: %.4f' % sensitivity, file=f)
        # print('Decision Tree Specificity: %.4f' % specificity, file=f)
        print('%.4f' % sensitivity, file=f)
        print('%.4f' % specificity, file=f)


        from sklearn.metrics import accuracy_score, precision_score, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_classification

        acc = accuracy_score(np.around(y_test), np.around(y_pred))
        prec = precision_score(np.around(y_test), np.around(y_pred), average='weighted')
        f1 = f1_score(np.around(y_test), np.around(y_pred), average='weighted')

        print('%.4f' % acc, file=f)
        print('%.4f' % prec, file=f)
        print('%.4f' % f1, file=f)



        print(f"Ready {i+1}/{NUMBER_OF_RUNS}")

print("FINISH FINISH FINISH!!!")