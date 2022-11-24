# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier



def categorical(train_df):
    """
    input: a pandas dataframe
    :param train_df:
    :return: categorical values
    """
    global cat_col
    cols = train_df.columns
    num_cols = train_df._get_numeric_data().columns
    cols = train_df.columns
    cat_col = list(set(cols) - set(num_cols))
    return cat_col


def numerical(train_df):
    """
    input: a pandas dataframe
    :param train_df:
    :return: numerical columns
    """
    global num_cols
    num_cols = train_df._get_numeric_data().columns
    return num_cols



def mis_Value(df: pd.DataFrame):
    """
    input:a pandas dataframe
    :param df:
    :return:values mean
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    val = df.values
    imputer = imputer.fit(val.reshape(-1, 1))
    val = imputer.transform(val.reshape(-1, 1))
    return val.mean()



def detect_outliers(df, features):
    """
    input: a pandas dataframe and dataframe features
    :param df:
    :param features:
    :return:multiple_outliers use in df.drop(df[multiple_outliers])
    """
    outlier_indices = []

    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c], 25)
        # 3rd quartile
        Q3 = np.percentile(df[c], 75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers



def prob_his(train_df, variable_x, variable_y):
    """

    :param train_df:
    :param variable_x:
    :param variable_y:
    :return:figure
    """
    fig = px.histogram(train_df, x=variable_x, marginal='box', title=variable_x, width=700, height=500,
                       color_discrete_sequence=['indianred'])
    fig.update_layout(bargap=0.1)
    fig.show()
    fig = px.histogram(train_df, x=variable_x, y=variable_y, color=variable_x, histfunc='avg', marginal='box',
                       barmode='overlay', title=variable_x, width=700, height=500)
    fig.update_layout(bargap=0.1, barmode='stack')
    fig.show()
    fig1 = px.box(train_df, x=variable_x, y=variable_y, color=variable_x, width=700, height=500, )
    fig1.update_traces(quartilemethod="exclusive")
    fig1.show()


def scatter_p(train_df, variable_x, variable_y):
    """

    :param train_df:
    :param variable_x:
    :param variable_y:
    :return:
    """
    fig = px.scatter(train_df, x=variable_x, y=variable_y, title=variable_x, width=700, height=500,
                     marginal_x="histogram", marginal_y="rug", trendline="ols", color=variable_x)
    fig.show()



def heatmap(train_df):
    """

    :param train_df:
    :return: heatmap for numerical variable
    :note:this function also uses numeric values that are categorical
    """
    sns.heatmap(train_df.corr(), annot=True, annot_kws={'size': 10}, fmt=".2f")
    plt.show()



def dummies(train_df: pd.DataFrame, columns):
    """

    :param train_df:
    :param  data frame columns:
    :return: dataframe=dummies(columns)
    """
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    train_df[columns] = le.fit_transform(train_df[columns])

    print(train_df)

    train_df = pd.get_dummies(train_df, columns=[columns])
    return train_df


def scaler(train_df: pd.DataFrame):
    """

    :param train_df:
    :return:dataframe=scaler(columns)
    """
    sc = StandardScaler()
    train_df = sc.fit_transform(train_df)
    return train_df



def normalzier(train_df: pd.DataFrame):
    """

    :param train_df:
    :return: dataframe=normalzier(columns)
    """
    from sklearn.preprocessing import Normalizer
    nr = Normalizer()
    train_df = nr.fit_transform(train_df)
    return train_df



def splits(train_df, train_df_len, droplabels):
    """

    :param train_df:
    :param train_df_len:
    :param droplabels:
    :return: X_train,X_test,y_train,y_test =splits(train_df, train_df_len, droplabels)
    """



    train = train_df[:train_df_len]
    X_t = train.drop(labels=droplabels, axis=1)
    y_t = train[droplabels]

    X_tr, X_te, y_tr, y_te = train_test_split(X_t, y_t, test_size=0.35, random_state=42)
    print("Now you can use the following variables")
    print("X_train", len(X_t))
    print("X_test", len(X_t))
    print("y_train", len(y_t))
    print("y_test", len(y_t))
    X_train = X_tr.copy()
    X_test = X_te.copy()
    y_train = y_tr.copy()
    y_test = y_te.copy()
    return X_train, X_test, y_train, y_test



def p_values(df, pred_df, row, col, liste: list):
    """

    :param df:
    :param prediction df() test_df:
    :param row df:
    :param col df:
    :param liste df columns number:
    :return X_l new train_df:
    """

    X = np.append(arr=np.ones((row, col)).astype(int), values=df, axis=1)
    X_l = df.iloc[:, liste].values
    X_l = pd.DataFrame(np.array(X_l, dtype=float))
    model = sm.OLS(pred_df, X_l).fit()
    print(model.summary())
    return X_l


# %%
def linearreg(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return y_head(round predict), y_predict(orj predict):
    """
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    y_head = pd.Series(y_predict).round(0).abs()
    print("Training Accuracy:  {}".format(lr.score(x_train, y_train) * 100, 2))
    print("Testing Accuracy:  {}".format(lr.score(x_test, y_test) * 100, 2))
    print("RMS: %r " % np.sqrt(np.mean((y_head - y_test) ** 2)))
    return y_head, y_predict


# %%
def logisticreg(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return y_head(round predict), y_predict(orj predict):
    """
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    y_predict = logreg.predict(x_test)
    y_head = pd.Series(y_predict).round(0).abs()
    acc_log_train = round(logreg.score(x_train, y_train) * 100, 2)
    acc_log_test = round(logreg.score(x_test, y_test) * 100, 2)
    print("Training Accuracy: % {}".format(acc_log_train))
    print("Testing Accuracy: % {}".format(acc_log_test))

    print("RMS: %r " % np.sqrt(np.mean((y_head - y_test) ** 2)))
    print("RMS: %r " % np.sqrt(np.mean((y_predict - y_test) ** 2)))

    print("RMSE", mean_squared_error(y_test, y_head, squared=False))
    print("MSE", mean_squared_error(y_test, y_head))

    return y_head, y_predict



# %%
def RFClassifier(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return y_head(round predict), y_predict(orj predict):
    """
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_predict = rf.predict(x_test)  # tahmin ediyoruz
    y_head = pd.Series(y_predict).round(0).abs()
    acc_rf_train = round(rf.score(x_train, y_train) * 100, 2)
    acc_rf_test = round(rf.score(x_test, y_test) * 100, 2)
    print("Training Accuracy: % {}".format(acc_rf_train))
    print("Testing Accuracy: % {}".format(acc_rf_test))
    print("RMS: %r " % np.sqrt(np.mean((y_head - y_test) ** 2)))
    print("RMS: %r " % np.sqrt(np.mean((y_predict - y_test) ** 2)))

    return y_predict, y_head


# %%
def RFRegresiions(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, estimators,
                  randomstate):
    """
    random forest regressions
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param estimators:
    :param randomstate:
    :return y_head(round predict), y_predict(orj predict):
    """
    rf = RandomForestRegressor(n_estimators=estimators, random_state=randomstate)
    rf.fit(x_train, y_train)
    y_predict = rf.predict(x_test)  # tahmin ediyoruz
    y_head = pd.Series(y_predict).round(0).abs()
    acc_rf_train = round(rf.score(x_train, y_train) * 100, 2)
    acc_rf_test = round(rf.score(x_test, y_test) * 100, 2)
    print("Training Accuracy: % {}".format(acc_rf_train))
    print("Testing Accuracy: % {}".format(acc_rf_test))
    print("RMS: %r " % np.sqrt(np.mean((y_head - y_test) ** 2)))
    print("RMS: %r " % np.sqrt(np.mean((y_predict - y_test) ** 2)))

    return y_predict, y_head


# %%
def polynominal(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, degree):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param degree:
    :return y_head(round predict), y_predict(orj predict):
    """
    polreg = PolynomialFeatures(degree=degree)

    x_train = polreg.fit_transform(x_train)
    x_test = polreg.fit_transform(x_test)

    lr = LinearRegression()

    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    y_head = pd.Series(y_predict).round(0).abs()

    acc_pl_train = round(lr.score(x_train, y_train) * 100, 2)
    acc_pl_test = round(lr.score(x_test, y_test) * 100, 2)
    print("Training Accuracy: % {}".format(acc_pl_train))
    print("Testing Accuracy: % {}".format(acc_pl_test))
    print("RMS: %r " % np.sqrt(np.mean((y_head - y_test) ** 2)))
    print("RMS: %r " % np.sqrt(np.mean((y_predict - y_test) ** 2)))

    return y_head, y_predict


# %%
def SvM(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, kernel: str):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param kernel:
    :return y_head(round predict), y_predict(orj predict) :
    """
    svm = SVC(random_state=1, kernel=kernel)
    svm.fit(x_train, y_train)
    y_predict = svm.predict(x_test)  # tahmin ediyoruz
    y_head = pd.Series(y_predict).round(0).abs()
    acc_svm_train = round(svm.score(x_train, y_train) * 100, 2)
    acc_svm_test = round(svm.score(x_test, y_test) * 100, 2)
    print("Training Accuracy: % {}".format(acc_svm_train))
    print("Testing Accuracy: % {}".format(acc_svm_test))
    print("head RMS: %r " % np.sqrt(np.mean((y_head - y_test) ** 2)))
    print("predict RMS: %r " % np.sqrt(np.mean((y_predict - y_test) ** 2)))
    return y_head, y_predict

# %%
def ridgeModel(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return y_head(round predict), y_predict(orj predict):
    """
    np.linspace(10, -2, 100)
    lambdalar = 10 ** np.linspace(10, -2, 100) * .05
    ridge_model = Ridge()
    katsayilar = []
    rmse = []
    for i in lambdalar:
        ## set params: parametreleri ayarlamak
        ridge_model.set_params(alpha=i)
        ridge_model.fit(X_train, y_train)
        y_predict = ridge_model.predict(X_test)
        y_predict = pd.Series(y_predict).round(0).abs()
        RMSE = np.sqrt(mean_squared_error(y_test, y_predict))
        rmse.append([RMSE, i])
        # her lambda bağımsız değişen sayısı kadar beta katsayılarını türemek
        katsayilar.append(ridge_model.coef_)
        # katsayilar
    a = min(rmse)

    ridge_model.set_params(alpha=a[1])
    ridge_model.fit(X_train, y_train)
    y_head = ridge_model.predict(X_test)
    y_predict = pd.Series(y_head).round(0).abs()
    RMSE1 = np.sqrt(mean_squared_error(y_test, y_predict))
    RMSE2 = np.sqrt(mean_squared_error(y_test, y_predict))

    acc_ridge_train = round(ridge_model.score(X_train, y_train) * 100, 2)
    acc_ridge_test = round(ridge_model.score(X_test, y_test) * 100, 2)
    print("Training Accuracy: % {}".format(acc_ridge_train))
    print("Testing Accuracy: % {}".format(acc_ridge_test))

    print("RMS: %r " % np.sqrt(np.mean((y_head - y_test) ** 2)))
    print("RMS: %r(yuvarlanmış) " % np.sqrt(np.mean((y_predict - y_test) ** 2)))
    return y_head,y_predict

# %%desicion tree
def destree(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, random_state=1):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param random_state:
    :return y_head(round predict), y_predict(orj predict):
    """
    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, y_train)
    y_predict = dtr.predict(X_test)
    y_head = pd.Series(y_predict).round(0).abs()
    acc_dtr_train = round(dtr.score(X_train, y_train) * 100, 2)
    acc_dtr_test = round(dtr.score(X_test, y_test) * 100, 2)
    print("Training Accuracy: % {}".format(acc_dtr_train))
    print("Testing Accuracy: % {}".format(acc_dtr_test))
    print("head RMS: %r " % np.sqrt(np.mean((y_head - y_test) ** 2)))
    print("predict RMS: %r " % np.sqrt(np.mean((y_predict - y_test) ** 2)))
    return y_head, y_predict