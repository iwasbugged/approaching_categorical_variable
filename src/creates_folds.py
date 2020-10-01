# create_folds.py
# import pandas and model_selection module from sklearn
import pandas as pd 
from sklearn import model_selection

if __name__ == "__main__":

    # Read training data
    df = pd.read_csv('../dataset/adult.csv')

    # we create a new column called kfold and fill it with -1
    df['kfold'] = -1

    # The step is to randomized the rows of the data
    df = df.sample(frac = 1).reset_index(drop = True)

    # fetch lebels
    y = df.income.values

    # initiate the kfold class from model_selection
    kf = model_selection.StratifiedKFold(n_splits= 5 )

    # fill the new kfold column
    for f , (t_ , v_) in enumerate(kf.split(X = df , y = y)):
        df.loc[v_ , 'kfold'] = f

    # save the new csv with kfold column
    df.to_csv('../dataset/adult_folds.csv' , index = False)