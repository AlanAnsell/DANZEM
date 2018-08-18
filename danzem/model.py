import numpy as np
import pandas as pd

import datetime
import random
import os
import tempfile

from keras import models
from keras import layers
from keras import models
from keras import initializers
from keras import regularizers

from sklearn import metrics
from sklearn import model_selection

from . import data


SKLEARN = 'sklearn'
KERAS = 'keras'


def LossReduction(previous_loss, current_loss):
    return (previous_loss - current_loss) / previous_loss


def Split(X, label_fn):
    folds = {}
    for i in X.index:
        label = label_fn(i)
        folds.setdefault(label, []).append(i)
    return folds
    

def SplitByYear(X):
    return Split(X, data.GetYearStrFromTPID)


def SplitByMonth(X):
    return Split(X, data.GetMonthStrFromTPID)


def SklearnTrainFn(model_fn, eval_fn, X, y, **kwargs):
    model_ = model_fn(X)
    model_.fit(X, y)
    return model_, 0.0


def KerasTrainFn(model_fn, eval_fn,
                 train_X, train_y, val_X, val_y,
                 max_val_stagnation_epochs=10,
                 progress_check_epoch=5,
                 progress_check_thresh=0.3,
                 batch_size=None,
                 verbose=False,
                 **kwargs):
    model_ = model_fn(train_X)
    train_X = train_X.as_matrix()
    train_y = train_y.as_matrix()
    val_X = val_X.as_matrix()
    val_y = val_y.as_matrix()
    
    train_loss = eval_fn(model_, train_X, train_y, **kwargs).loss
    val_loss = eval_fn(model_, val_X, val_y, **kwargs).loss
    initial_val_loss = val_loss
    if verbose:
        print('Before epoch 1: training loss = %.5f, validation loss = %.5f' % (
            train_loss, val_loss))

    model_file_dir = tempfile.mkdtemp()
    model_file_path = os.path.join(model_file_dir, 'model.h5')
    model_.save(model_file_path)
    
    epoch = 1
    val_stagnation_epochs = 0
    best_val_loss = val_loss
    while True:
        model_.fit(train_X, train_y, batch_size=batch_size, epochs=1, verbose=0)
        train_loss = eval_fn(model_, train_X, train_y, **kwargs).loss
        val_loss = eval_fn(model_, val_X, val_y, **kwargs).loss
        if verbose:
            print('Epoch %d: training loss = %.5f, validation loss = %.5f' % (
                epoch, train_loss, val_loss))
        
        if val_loss < best_val_loss:
            model_.save(model_file_path)
            best_val_loss = val_loss
            val_stagnation_epochs = 0
        else:
            val_stagnation_epochs += 1
       
        if val_stagnation_epochs >= max_val_stagnation_epochs:
            break

        if epoch == progress_check_epoch:
            if (LossReduction(initial_val_loss, best_val_loss) <
                    progress_check_thresh):
                return None, 0.0
        
        epoch += 1
    
    best_model = models.load_model(model_file_path)
    os.remove(model_file_path)
    os.rmdir(model_file_dir)

    if 'max_acceptable_loss' in kwargs and best_val_loss > kwargs['max_acceptable_loss']:
        return None, 0.0

    return best_model, best_val_loss


def DefaultTrainValSplitFn(X, y, train_size=0.8, **kwargs):
    day_buckets = {}
    for i in X.index:
        day = data.GetDayStrFromTPID(i)
        day_buckets.setdefault(day, []).append(i)

    days = list(day_buckets)
    random.shuffle(days)
    val_req = round((1.0 - train_size) * X.shape[0])
    val_tps = []
    i = 0
    while len(val_tps) < val_req and i < len(days):
        val_tps += day_buckets[days[i]]
        i += 1

    val_tps = sorted(val_tps)
    train_tps = sorted(list(set(X.index) - set(val_tps)))

    return (X.loc[train_tps], y.loc[train_tps],
            X.loc[val_tps], y.loc[val_tps])


_DefaultSingleModelTrainFn = KerasTrainFn

def DefaultTrainFn(model_fn, eval_fn, X, y,
                   single_model_train_fn=None,
                   train_val_split_fn=None,
                   transform_fn=None,
                   process_fn=None,
                   separate_transform_for_val=False,
                   n_models=1,
                   max_attempts=-1,
                   verbose=False,
                   **kwargs):

    if not single_model_train_fn:
        single_model_train_fn = _DefaultSingleModelTrainFn

    if not train_val_split_fn:
        train_val_split_fn = DefaultTrainValSplitFn
    
    train_X, train_y, val_X, val_y = train_val_split_fn(X, y, **kwargs)
    if transform_fn:
        if separate_transform_for_val:
            transform_fn = process_fn(train_X, train_y, **kwargs)
        train_X, train_y = transform_fn(train_X, train_y)
        val_X, val_y = transform_fn(val_X, val_y)
   
    best_model = None
    best_loss = 1e10
    n_successful_models = 0
    n_attempts = 0
    while ((n_successful_models < n_models) and
           (max_attempts == -1 or n_attempts < max_attempts)):
        model_, val_loss = single_model_train_fn(
                model_fn, eval_fn, train_X, train_y, val_X, val_y,
                verbose=verbose,
                **kwargs)
        if model_:
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model_
            n_successful_models += 1
        n_attempts += 1

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


_LOSS_FN_MAP = {'mae': metrics.mean_absolute_error,
                'mse': metrics.mean_squared_error,
                'accuracy': metrics.accuracy_score}


def PredictClasses(y):
    return np.argmax(y, axis=1)

def CategoricalAccuracy(y_true, y_pred):
    return np.mean(np.equal(PredictClasses(y_true), PredictClasses(y_pred)).astype(np.float32))

def Matthews(y_true, y_pred):
    return metrics.matthews_corrcoef(PredictClasses(y_true), PredictClasses(y_pred))


_PROBA_LOSS_FN_MAP = {'log_loss': metrics.log_loss,
                      'binary_crossentropy': metrics.log_loss,
                      'categorical_crossentropy': metrics.log_loss,
                      'categorical_accuracy': CategoricalAccuracy,
                      'matthews': Matthews}

class UnrecognisedLossException(Exception):
    pass


def DefaultPredictFn(model_, X, model_type=SKLEARN, classification=False,
                     proba=False, **kwargs):
    y = None
    if model_type == SKLEARN:
        if proba:
            y = model_.predict_proba(X)
        else:
            y = model_.predict(X)
    elif model_type == KERAS:
        if classification:
            if proba:
                y = model_.predict_proba(X)
            else:
                y = model_.predict_classes(X)
        else:
            y = model_.predict(X)
    if y.ndim == 2 and y.shape[1] == 1:
        return np.squeeze(y, axis=1) 
    return y

def DefaultEvalFn(model_, X, y, loss='mse', metrics=[], **kwargs):
    #print('Loss: %s' % loss)
    if isinstance(X, pd.DataFrame):
        X = X.as_matrix()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.as_matrix()
    
    classification = (loss in _PROBA_LOSS_FN_MAP)
    to_find = set([loss] + metrics)
    normal_loss_fns = []
    proba_loss_fns = []
    for loss_type in to_find:
        if loss_type in _LOSS_FN_MAP:
            normal_loss_fns.append(loss_type)
        elif loss_type in _PROBA_LOSS_FN_MAP:
            proba_loss_fns.append(loss_type)
        else:
            raise UnrecognisedLossException(loss_type)

    losses = {}
    pred = DefaultPredictFn(model_, X,
                            classification=classification,
                            proba=False,
                            **kwargs)
    for loss_type in normal_loss_fns:
        losses[loss_type] = _LOSS_FN_MAP[loss_type](y, pred)

    if proba_loss_fns:
        proba = DefaultPredictFn(model_, X,
                                 classification=classification,
                                 proba=True,
                                 **kwargs)
        for loss_type in proba_loss_fns:
            losses[loss_type] = _PROBA_LOSS_FN_MAP[loss_type](y, proba)
    
    loss_metrics = {loss_type: loss_val
                    for loss_type, loss_val in losses.items()
                    if loss_type != loss}
    return Result(model_, pred, y, losses[loss], loss_metrics)


def DefaultPerformanceFn(performance, verbose=False, **kwargs):
    loss_sum = 0.0
    for fold_name, loss in sorted(performance.items()):
        if verbose:
            print('Peformance on %s: %.5f' % (fold_name, loss))
        loss_sum += loss
    return loss_sum / len(performance)


def CrossEvaluate(X, y, model_fn, **kwargs):
    if 'fold_fn' not in kwargs:
        kwargs['fold_fn'] = SplitByYear
    fold_fn = kwargs['fold_fn']

    if 'eval_fn' in kwargs:
        eval_fn = kwargs['eval_fn']
        del kwargs['eval_fn']
    else:
        eval_fn = DefaultEvalFn

    if 'performance_fn' not in kwargs:
        kwargs['performance_fn'] = lambda performance: (
                DefaultPerformanceFn(performance, **kwargs))
    performance_fn = kwargs['performance_fn']
    
    if 'process_fn' not in kwargs:
        kwargs['process_fn'] = None
    process_fn = kwargs['process_fn']

    if 'train_fn' not in kwargs:
        kwargs['train_fn'] = DefaultTrainFn
    train_fn = kwargs['train_fn']

    if 'verbose' not in kwargs:
        kwargs['verbose'] = False
    verbose = kwargs['verbose']
    if verbose:
        print('Cross evaluating model...')
    
    folds = fold_fn(X)
    performance = {}
    results = {}
    for fold_name, fold_indices in sorted(folds.items()):
        if verbose:
            print("Evaluating on fold '%s'..." % fold_name)
        in_fold = sorted(fold_indices)
        not_in_fold = sorted(list(set(X.index) - set(in_fold)))
        train_X = X.loc[not_in_fold]
        train_y = y.loc[not_in_fold]
        test_X = X.loc[in_fold]
        test_y = y.loc[in_fold]
        if process_fn:
            transform_fn = process_fn(train_X, train_y, **kwargs)
            test_X, test_y = transform_fn(test_X, test_y, **kwargs)
        else:
            transform_fn = None
        kwargs['transform_fn'] = transform_fn

        model_, loss = train_fn(model_fn, eval_fn, train_X, train_y, **kwargs)
        
        if model_ is None:
            return None, None
        
        results[fold_name] = eval_fn(model_, test_X, test_y, **kwargs)
        performance[fold_name] = results[fold_name].loss
        if verbose:
            print("Score on fold '%s': %.5f" % (fold_name, performance[fold_name]))
            for metric_name, metric_result in results[fold_name].metrics.items():
                print('%s: %.5f' % (metric_name, metric_result))
    
    final_performance = performance_fn(performance)
    if verbose:
        print('Final cross evaluation score: %.5f' % final_performance)
    return final_performance, results


def MakeNN(hidden_layers, n_outputs=1, loss='mse', reg_strength=0.0, init_stddev=1.0, optimizer='adam'):
  
    if loss == 'log_loss':
        loss = 'binary_crossentropy'
    if loss  == 'binary_crossentropy':
        output_activation = 'sigmoid'
    elif loss == 'categorical_crossentropy':
        output_activation = 'softmax'
    else:
        output_activation = 'relu'

    def model_fn(X):
        nn = models.Sequential()

        nn.add(layers.Dense(units=hidden_layers[0], input_dim=X.shape[1],
                            kernel_initializer=initializers.RandomNormal(mean=0.0,
                                                                         stddev=init_stddev),
                            bias_initializer=initializers.RandomNormal(mean=0.0,
                                                                       stddev=init_stddev),
                            bias_regularizer=regularizers.l2(reg_strength),
                            kernel_regularizer=regularizers.l2(reg_strength),
                            activation='relu'))
        for i in range(1, len(hidden_layers)):
            nn.add(layers.Dense(units=hidden_layers[i],
                                kernel_initializer=initializers.RandomNormal(mean=0.0,
                                                                             stddev=init_stddev),
                                bias_initializer=initializers.RandomNormal(mean=0.0,
                                                                           stddev=init_stddev),
                                bias_regularizer=regularizers.l2(reg_strength),
                                kernel_regularizer=regularizers.l2(reg_strength),
                                activation='relu'))
        nn.add(layers.Dense(units=n_outputs,
                            kernel_initializer=initializers.RandomNormal(mean=0.0,
                                                                         stddev=init_stddev),
                            bias_initializer=initializers.RandomNormal(mean=0.0,
                                                                       stddev=init_stddev),
                            bias_regularizer=regularizers.l2(reg_strength),
                            kernel_regularizer=regularizers.l2(reg_strength),
                            activation=output_activation))

        nn.compile(loss=loss, optimizer=optimizer)

        return nn
    
    return model_fn
