from unicodedata import numeric
import unittest
import mlf.mlf as mlf


class testsuit(unittest.TestCase):
    def test(self):
        train_df=mlf.pd.read_csv("C:/Users/Tolga/Desktop/mlf/tests/veriler.csv", sep="," )
        train_df["cinsiyet"] = [0 if i == "e" else 0 for i in train_df["cinsiyet"]]
        train_df["cinsiyet"].replace("e", 1, inplace=True)
        train_df["cinsiyet"].replace("k", 0, inplace=True)

        print(train_df.isnull().any())
        train_df.loc[mlf.detect_outliers(train_df, mlf.numerical(train_df))]

        categorical=mlf.categorical(train_df)

        numerical=mlf.numerical(train_df)

        print(mlf.mis_Value(train_df["yas"]))
        train_df = train_df.fillna(mlf.mis_Value(train_df["yas"]))

        train_df.loc[mlf.detect_outliers(train_df, numerical)]
        train_df = train_df.drop(mlf.detect_outliers(train_df, ["boy", "kilo", "yas"]), axis=0).reset_index(drop=True)

        for i in numerical:
            mlf.prob_his(train_df, i, "cinsiyet")

        for i in numerical:
            mlf.scatter_p(train_df, i, "cinsiyet")
        mlf.heatmap(train_df)

        train_df = mlf.dummies(train_df, "ulke")
        x_train, x_test, y_train, y_test = mlf.splits(train_df, 15, "yas")
        y_train = y_train.astype('int')

        print("------not scaler logistic---------")
        mlf.logisticreg(x_train,y_train,x_test,y_test)

        print("------not scaler linear---------")
        mlf.linearreg(x_train,y_train,x_test,y_test)

        print("------not scaler random forest clas---------")
        mlf.RFClassifier(x_train,y_train,x_test,y_test)

        print("------not scaler random forest reg---------")
        mlf.RFRegresiions(x_train,y_train,x_test,y_test, 4, 42)

        print("------not scaler support vector machine---------")
        mlf.SvM(x_train,y_train,x_test,y_test, "rbf")

        print("------not scaler ridge reg---------")
        mlf.ridgeModel(x_train,y_train,x_test,y_test)

        print("------not scaler desicion tree---------")
        mlf.destree(x_train,y_train,x_test,y_test)

        train_df_splt = train_df.drop(labels="cinsiyet", axis=1)
        train_df_splt_pred = mlf.pd.DataFrame(train_df["cinsiyet"])


        X_L=mlf.p_values(train_df_splt, train_df_splt_pred, 22, 1, [0, 1, 4])

        train_df_new = mlf.pd.concat([X_L, train_df_splt_pred], axis=1).reset_index(drop=True)

        xt,yt,xtest,ytest=mlf.splits(train_df_new, 15, "cinsiyet")

if __name__ == '__main__':
    unittest.main()

