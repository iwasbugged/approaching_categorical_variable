import pandas as pd 

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    # load the full data with folds
    df = pd.read_csv('../dataset/cat_train_folds.csv')

    # all columns are features except id , target  and kfolds columns
    features = [
        f for f in df.columns if f not in ('id' , 'target' , 'kfold')
    ]

    # fill all NaN values with NONE
    for col in features:
        df.loc[: , col] = df[col].astype(str).fillna('NONE')
        # we have converted all the column into string because all are categorical

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop = True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    # initialize OneHotEncoder from sklearn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation data set
    full_data = pd.concat(
        [df_train[features] , df_valid[features]] ,
        axis = 0
    )

    ohe.fit(full_data[features])

    # transform train data
    x_train = ohe.transform(df_train[features])

    # transform validation data
    x_valid = ohe.transform(df_valid[features])

    # initialize Logistic Regression model
    model = linear_model.LogisticRegression()

    # fit model with training data
    model.fit(x_train , df_train['target'].values)

    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[: , 1]

    # get roc auc score

    auc = metrics.roc_auc_score(df_valid.target.values , valid_preds)

    # print auc
    print(f"Fold = {fold} , AUC = {auc}")


if __name__ == "__main__":
    # run function for fold = 0
    for fold_ in range(5):
        run(fold_)