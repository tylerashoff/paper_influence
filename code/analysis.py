import numpy as np
import pandas as pd
import csv
import os

from sklearn.model_selection import train_test_split
from sklearn import metrics

import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class analysis():

    def __init__(self, verbose=False):
        self.verbose = verbose
        pass

    def neural_network(self, x_train, x_test, y_train):
        if self.verbose:
            print('neural network')
            pass
        
        model = Sequential()

        model.add(Dense(512, input_dim = x_train.shape[1], activation='relu'))
        
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        
        if not self.binary:
            model.add(Dense(1, activation='relu'))
            model.compile(loss='mean_squared_error', optimizer='adam',
                          metrics=['mean_squared_error'])
            pass
        else:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam',
                          metrics=['binary_accuracy'])
            pass
        
        model.fit(x_train, y_train, epochs=200, batch_size=32,
                  validation_split = 0.2, verbose=0)
        preds = model.predict(x_test)

        if not self.binary:
            return(preds.astype('float64').flatten())
        else:
            return(np.round(preds.flatten()))
        pass
    
    def random_forest(self, x_train, x_test, y_train):
        if self.verbose:
            print('random forests')
            pass

        if not self.binary:
            model = RandomForestRegressor()
            pass
        else:
            model = RandomForestClassifier()
            pass
        
        model.fit(x_train, y_train)
        
        # Get the mean absolute error on the validation data
        preds = model.predict(x_test)

        return(preds)
    
    def xgboost(self, x_train, x_test, y_train):
        if self.verbose:
            print('xgboost')
            pass

        if not self.binary:
            obj = 'reg:squarederror'
            pass
        else:
            obj = 'binary:hinge'
            pass
        
        params = {'max_depth': 5,
                  'eta': 0.5,
                  'objective': obj,
                  'eval_metric': ['logloss']}
        
        data_train = xgb.DMatrix(x_train,
                                 label=y_train)
        data_test = xgb.DMatrix(x_test,
                                label=y_test)
        
        bst = xgb.train(params, data_train)
        preds = bst.predict(data_test)

        return(preds)

    def gaussian_process(self, x_train, x_test, y_train):
        if self.verbose:
            print('gaussian_process')
            pass

        if not self.binary:
            model = GaussianProcessRegressor()
            pass
        else:
            model = GaussianProcessClassifier()
            pass

        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        return(preds)

    def scores(self, y_test, preds):

        if not self.binary:
            mse = metrics.mean_squared_error(y_test, preds)
            r2 = np.corrcoef(y_test, preds)[0, 1]
            ev = metrics.explained_variance_score(y_test, preds)
            return(mse, r2, ev)
        else:
            fpr, tpr, threshs = metrics.roc_curve(y_test, preds)
            auc =  metrics.auc(fpr, tpr)
            acc = metrics.accuracy_score(y_test, preds)
            f1 = metrics.f1_score(y_test, preds)
            return(auc, acc, f1)
        pass

    def write(self, model, scores):
        
        types = 'regression' if not self.binary else 'binary'
        csv.writer(open('results.csv', 'a')).writerow([str(self.embedding),
                                                       types, model,
                                                       scores[0], scores[1],
                                                       scores[2]])
        pass
        
    def main(self, x_train, x_test, y_train, y_test, binary=False, embedding=None):

        self.embedding = embedding
        self.binary = binary
        
        xg_preds = self.xgboost(x_train, x_test, y_train)
        xg_scores = self.scores(y_test, xg_preds)
        self.write('xgboost', xg_scores)
        print(xg_scores)
        '''
        gp_preds = self.gaussian_process(x_train, x_test, y_train)
        gp_scores = self.scores(y_test, gp_preds)
        self.write('gaussian_process', gp_scores)
        print(gp_scores)
        
        nn_preds = self.neural_network(x_train, x_test, y_train)
        nn_scores = self.scores(y_test, nn_preds)
        self.write('neural_net', nn_scores)
        print(nn_scores)
        '''
        rf_preds = self.random_forest(x_train, x_test, y_train)
        rf_scores = self.scores(y_test, rf_preds)
        self.write('random_forests', rf_scores)
        print(rf_scores)
        
        pass
    pass


embeddings = ['spectral', 'isomap', 'varenc', None]

for embedding in embeddings:
    print(embedding)
    if embedding is None:
        filename = './datasets/full_data.csv'
        pass
    else:
        filename = './datasets/'+embedding+'_full_data.csv'
        pass
    
    df = pd.read_csv(filename).to_numpy()
    
    # split into 2020 and not
    dates = pd.to_datetime(df[:, 1])
    mask = dates >= pd.to_datetime('2020')
    df_2020 = df[mask, :]
    df = df[np.invert(mask), :]
    
    if embedding is None:
        vec_dim = 100
        pass
    else:
        vec_dim = 5
        pass
    end_ind = vec_dim+8

    # set missing values
    df[:, 5:][np.isnan(list(df[:, 5:].flatten())).reshape(df[:, 5:].shape)] = 0
    df[:, 2][[isinstance(i, float) for i in df[:, 2]]] = 'NA'

    # enumerate journal entry
    journal_to_int = dict((c, i) for i, c in enumerate(np.unique(df[:, 2])))
    journal = np.array([journal_to_int[char] for char in df[:, 2]])

    # transforms
    small_const = 10**(-4)
    x = np.vstack([
        pd.to_datetime(df[:, 1]).year,
        journal,
        np.log(list(df[:, 5]+small_const)),
        np.log(list(df[:, 6]+small_const)),
        df[:, 7],
        #df[:, end_ind-5:end_ind+1].T,
        np.log(list(df[:, end_ind+1]+small_const)),
        np.log(list(df[:, end_ind+2]+small_const)),
        df[:, end_ind+3]
        #np.log(df[:, end_ind+4:].astype('float64')**2+small_const).T
    ]).T.astype('float64')

    y = np.log(list(df[:, 3]+10**(-4)))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.20,
                                                        random_state=42)
    
    anlys = analysis(verbose = True)

    anlys.main(x_train, x_test, y_train, y_test,
               binary=False, embedding=embedding)
    
    # binary predictions above a given number of citations
    cut = 5
    y_train = y_train > np.log(cut)
    y_test = y_test > np.log(cut)
    print('ratio: ', sum(y_test)/len(y_test))

    anlys.main(x_train, x_test, y_train, y_test,
               binary=True, embedding=embedding)

    pass

if anlys.verbose:
    os.system("say 'all done you frickin genius'")
    pass


