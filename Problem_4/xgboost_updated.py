from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss,accuracy_score,roc_auc_score
from sklearn import preprocessing

import numpy as np
import pandas as pd
import argparse
import os
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle
import sys
import xgboost
from xgboost import XGBClassifier 
import time


key_glob = ''
train_dict = {}
train_labels = {}
test_dict = {}
test_labels = {}


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', help='path argument..', required=True)
    args = parser.parse_args()
    return args


def load_train(args):
    stock_df = {}
    train_Y = {}
    for item in os.listdir(args.path):
        df = pd.read_csv(args.path+item,header = 0)
        df = df.drop(columns='datetime')
        train_Y[item.strip('.csv')] = df['Y']
        df = df.drop(columns = 'Y')
        stock_df[item.strip('.csv')] = df
    return stock_df,train_Y


def score(params):
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgboost.DMatrix(train_dict[key_glob], label=train_labels[key_glob])
    dvalid = xgboost.DMatrix(test_dict[key_glob], label=test_labels[key_glob])
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    print("\n\n\n.................\n\n\n")
    print(num_round, key_glob)
    gbm_model = xgboost.train(params, 
                              dtrain, 
                              num_round,
                              evals=watchlist,
                              verbose_eval=False)
    predictions = gbm_model.predict(dvalid, ntree_limit=gbm_model.best_iteration+1)
    score = roc_auc_score(test_labels[key_glob], predictions)
    # TODO: Add the importance for the selected features
    print("\n\n\n\tScore {0}\n\n\n\n\n\n".format(score))
    # The score function should return the loss (1-score)
    # since the optimize function looks for the minimum
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}
 
 
def optimize(evals, cores, trials, optimizer=tpe.suggest, random_state=0):
    space = {
        'n_estimators': hp.quniform('n_estimators', 200, 1500, 1),
        'eta': hp.quniform('eta', 0.005, 0.25, 0.025), # A problem with max_depth casted to float instead of int with the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 100, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'eval_metric': 'auc',
        'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
        'alpha' :  hp.quniform('alpha', 0, 10, 1),
        'lambda': hp.quniform('lambda', 1, 2, 0.1),
        'nthread': cores,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'seed': random_state
    }
    best = fmin(score, space, algo=tpe.suggest, max_evals=evals, trials = trials)
    return best


def main():
    args = parser()
    full_trainX, full_trainY = load_train(args)
    print ("Splitting data into train and valid ...\n\n")
    best_params = {}
    for key in full_trainX:
        X_train, X_test, y_train, y_test = train_test_split(full_trainX[key], full_trainY[key], test_size=0.2, random_state=1234)
        train_dict[key] = X_train
        train_labels[key] = y_train
        test_dict[key] = X_test
        test_labels[key] = y_test
        global key_glob
        key_glob = key 
        trials = Trials()
        best = optimize(10,24,trials,1234)
        best_params[key] = best
    
    xgb_model_dict = {}
    for key in train_dict:
        xgb = XGBClassifier(n_estimators = int(best_params[key]['n_estimators']),
                            learning_rate= best_params[key]['eta'],objective="binary:logistic",booster='gbtree',
                            gamma =  best_params[key]['gamma'],max_depth=best_params[key]['max_depth'],
                            min_child_weight=int(best_params[key]['min_child_weight']),subsample=best_params[key]['subsample'],
                            colsample_bytree=best_params[key]['colsample_bytree'],reg_alpha=best_params[key]['alpha'],
                            reg_lambda=best_params[key]['lambda'],nthread=24,random_state=1234)
        xgb.fit(train_dict[key],train_labels[key])
        xgb_model_dict[key] = xgb
        pred_probs = xgb_model_dict[key].predict_proba(test_dict[key])
        predictions = [np.argmax(value) for value in pred_probs]
        accuracy = accuracy_score(test_labels[key], predictions)
        print("\n\n\n\nAccuracy: %.2f%%\n\n\n\n" % (accuracy * 100.0))
    
    
    with open('final_xgboost_dict.pickle', 'wb') as handle:
        pickle.dump(xgb_model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__": main()