import numpy as np
import pandas as pd

import datetime
import random

from keras import models

from sklearn import model_selection


def LossReduction(previous_loss, current_loss):
    return (previous_loss - current_loss) / previous_loss


def train_model(X, y, model, ts_split=False, val_first=False, train_size=0.9, use_all_data=False, max_loss_stagnation_epochs=2):
    if ts_split:
        n_examples = X.shape[0]
        n_train = round(n_examples * train_size)

        if val_first:
            val_X = X[:-n_train,:]
            val_y = y[:-n_train]
            train_X = X[-n_train:,:]
            train_y = y[-n_train:]
        else:
            train_X = X[:n_train,:]
            train_y = y[:n_train]
            val_X = X[n_train:,:]
            val_y = y[n_train:]
    else:
        train_X, val_X, train_y, val_y = model_selection.train_test_split(
                X, y, train_size=train_size)
    if use_all_data:
        train_X = X
        train_y = y

    epoch = 1
    previous_val_loss = 0
    loss_stagnation_epochs = 0
    train_loss = model.evaluate(train_X, train_y, verbose=0)[0]
    val_loss = model.evaluate(val_X, val_y, verbose=0)[0]
    print('Before epoch 1: training loss = %.5f, validation loss = %.5f' % (
        train_loss, val_loss))
    model.save('Models/tmp.h5')
    best_loss = val_loss
    while True:
        history = model.fit(train_X, train_y, epochs=1, verbose=0)
        train_loss = history.history['loss'][0]
        val_loss = model.evaluate(val_X, val_y, verbose=0)[0]
        print('Epoch %d: training loss = %.5f, validation loss = %.5f' % (
            epoch, train_loss, val_loss))
        if epoch > 1:
            if LossReduction(previous_val_loss, val_loss) < 0.0001:
                loss_stagnation_epochs += 1
            else:
                loss_stagnation_epochs = 0
        if val_loss < best_loss:
            best_loss = val_loss
            model.save('Models/tmp.h5')
        if loss_stagnation_epochs >= max_loss_stagnation_epochs:
            break
        previous_val_loss = val_loss
        epoch += 1
    
    return models.load_model('Models/tmp.h5')


def split_by_year(X):
    folds = {}
    for i in X.index:
        year = i[:4]
        if year in folds:
            folds[year].append(i)
        else:
            folds[year] = [i]
    return folds


def split_by_month(X):
    folds = {}
    for i in X.index:
        month = i[:7]
        if month in folds:
            folds[month].append(i)
        else:
            folds[month] = [i]
    return folds


def split_by_4_months(X):
    folds = {}
    month_map = {1: '1-4', 2: '1-4', 3: '1-4', 4: '1-4',
                 5: '5-8', 6: '5-8', 7: '5-8', 8: '5-8',
                 9: '9-12', 10: '9-12', 11: '9-12', 12: '9-12'}
    for i in X.index:
        year = i[:4]
        month = int(i[5:7])
        fold_name = '%s_%s' % (year, month_map[month])
        if fold_name in folds:
            folds[fold_name].append(i)
        else:
            folds[fold_name] = [i]
    return folds


import random

def default_train_single_model(model_fn, eval_fn, train_X, train_y, val_X, val_y, max_stagnation_epochs=2,
        max_no_improvement_epochs=50, verbose=False):
    train_X = train_X.as_matrix()
    train_y = train_y.as_matrix()
    val_X = val_X.as_matrix()
    val_y = val_y.as_matrix()
    model = model_fn(train_X)
    
    epoch = 1
    loss_stagnation_epochs = 0
    
    train_loss = eval_fn(model, train_X, train_y).loss
    val_loss = eval_fn(model, val_X, val_y).loss
    previous_val_loss = val_loss
    previous_train_loss = train_loss
    if verbose:
        print('Before epoch 1: training loss = %.5f, validation loss = %.5f' % (
            train_loss, val_loss))
    
    model_file_path = 'Models/tmp%d.h5' % random.randint(0, 1000)
    model.save(model_file_path)
    best_train_loss = train_loss
    best_val_loss = val_loss
    epochs_since_last_improvement = 0
    while True:
        model.fit(train_X, train_y, epochs=1, verbose=0)
        train_loss = eval_fn(model, train_X, train_y).loss
        val_loss = eval_fn(model, val_X, val_y).loss
        if verbose:
            print('Epoch %d: training loss = %.5f, validation loss = %.5f' % (
                epoch, train_loss, val_loss))
        if epoch == 1:
            if LossReduction(previous_train_loss, train_loss) < 0.03:
                return None, 0.0
        else:
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                epochs_since_last_improvement = 0
                if val_loss < best_val_loss:
                    loss_stagnation_epochs = 0
                else:
                    loss_stagnation_epochs += 1
            else:
                epochs_since_last_improvement += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(model_file_path)
        if (loss_stagnation_epochs >= max_stagnation_epochs or
            epochs_since_last_improvement >= max_no_improvement_epochs):
            break
        #previous_val_loss = val_loss
        #previous_train_loss = train_loss
        epoch += 1
    
    return models.load_model(model_file_path), best_val_loss


def default_val_split(X, y, train_size=0.8):
    mdtp_buckets = {}
    for i in X.index:
        parts = [int(x) for x in i.split('@')[0].split('_')]
        dow = datetime.datetime(parts[0], parts[1], parts[2]).weekday()
        mdtp_key = (parts[1], dow, parts[-1])
        if mdtp_key in mdtp_buckets:
            mdtp_buckets[mdtp_key].append(i)
        else:
            mdtp_buckets[mdtp_key] = [i]
    
    train_indices = []
    val_indices = []
    for index_list in mdtp_buckets.values():
        random.shuffle(index_list)
        n_train = round(len(index_list) * train_size)
        train_indices += index_list[:n_train]
        val_indices += index_list[n_train:]

    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)

    return X.loc[train_indices], y.loc[train_indices], X.loc[val_indices], y.loc[val_indices]


def default_train_fn(model_fn, eval_fn, X, y, process_fn=None, n_models=10, verbose=False, train_size=0.8, **kwargs):
    
    train_X, train_y, val_X, val_y = default_val_split(X, y, train_size)
    #train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, train_size=0.8)
    #train_size = 0.8
    #n_train = round(train_size * X.shape[0])
    #train_X = X.iloc[:n_train, :]
    #train_y = y.iloc[:n_train]
    #val_X = X.iloc[n_train:, :]
    #val_y = y.iloc[n_train:]
    if process_fn:
        train_X, train_y, val_X, val_y = process_fn(train_X, train_y, val_X, val_y)
   
    best_model = None
    best_loss = 1e10
    n_successful_models = 0
    while n_successful_models < n_models:
        model, val_loss = default_train_single_model(model_fn, eval_fn, train_X, train_y, val_X, val_y, verbose=verbose,
                                                     **kwargs)
        if model:
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
            n_successful_models += 1

    if verbose:
        print('Best loss: %.5f' % best_loss)
    return best_model, best_loss


class Result:

    def __init__(self, model, predictions, targets, loss, metrics={}):
        self.model = model
        self.predictions = predictions
        self.targets = targets
        self.loss = loss
        self.metrics = metrics


def eval_mae(model, X, y):
    if X.__class__ == pd.DataFrame:
        X = X.as_matrix()
        y = y.as_matrix()
    pred = model.predict(X)[:,0]
    return Result(model, pred, y, np.mean(np.abs(pred - y)))


def default_performance_fn(performance, verbose=False):
    loss_sum = 0.0
    for fold_name, loss in sorted(performance.items()):
        if verbose:
            print('Peformance on %s: %.5f' % (fold_name, loss))
        loss_sum += loss
    return loss_sum / len(performance)


def cross_evaluate(X, y, model_fn,
                   split_fn=None,
                   train_fn=None,
                   eval_fn=None,
                   performance_fn=None,
                   process_fn=None,
                   train_size=0.8,
                   verbose=False):
    if not split_fn:
        split_fn = split_by_year
    if not eval_fn:
        eval_fn = eval_mae
    if not performance_fn:
        performance_fn = lambda performance_: default_performance_fn(performance_, verbose=verbose)
    if not train_fn:
        train_fn = lambda model_fn_, eval_fn_, X_, y_: default_train_fn(
                model_fn_, eval_fn_, X_, y_, process_fn=process_fn, train_size=train_size, verbose=verbose)

    if verbose:
        print('Cross evaluating model...')
    folds = split_fn(X)
    performance = {}
    results = {}
    for fold_name, fold_indices in sorted(folds.items()):
        fold_index_set = set(fold_indices)
        in_fold = [x in fold_index_set for x in X.index]
        not_in_fold = [not x for x in in_fold]
        if process_fn:
            train_X, train_y, test_X, test_y = process_fn(X[not_in_fold], y[not_in_fold], X[in_fold], y[in_fold])
        else:
            train_X = X[not_in_fold]
            train_y = y[not_in_fold]
            test_X = X[in_fold]
            test_y = y[in_fold]
        if verbose:
            print("Evaluating on fold '%s'..." % fold_name)
        model, loss = train_fn(model_fn, eval_fn, train_X, train_y, process_fn=process_fn, verbose=verbose)
        results[fold_name] = eval_fn(model, test_X, test_y)
        performance[fold_name] = results[fold_name].loss
        if verbose:
            print("Score on fold '%s': %.5f" % (fold_name, performance[fold_name]))
            for metric_name, metric_result in results[fold_name].metrics.items():
                print('%s: %.5f' % (metric_name, metric_result))
    final_performance = performance_fn(performance)
    if verbose:
        print('Final cross evaluation score: %.5f' % final_performance)
    return final_performance, results


def roll_model(test_X, test_y, model, increment=1000):
    pred = np.zeros(test_y.shape)
    chunks = list(range(0, test_X.shape[0], increment))
    chunks.append(test_X.shape[0])

    for i in range(len(chunks) - 1):
        start = chunks[i]
        end = chunks[i + 1]
        pred[start:end] = model.predict(test_X[start:end,:])[:,0]
        
        mae = np.mean(np.abs(pred[start:end] - test_y[start:end]))
        print('Increment %d: mae = %.5f' % (i + 1, mae))

        X_increment = test_X[start:end]
        y_increment = test_y[start:end]
        X_val = test_X[:end]
        y_val = test_y[:end]

        previous_val_loss = 0
        epoch = 1
        model.fit(X_val, y_val, epochs=2, verbose=0)
        #while True:
        #    model.fit(X_increment, y_increment, epochs=1, verbose=0)
        #    val_loss = model.evaluate(X_val,y_val, verbose=0)
        #    if epoch > 1:
        #        if val_loss > previous_val_loss:
        #            break
        #    previous_val_loss = val_loss
        #    epoch += 1

    return pred
