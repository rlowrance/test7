import pdb

import pickle_utilities

all_columns = {
    'cusip': (9, '%9s', ('', 'cuisp'), 'identifies a particular security'),
    'feature_importance': (4, '%4.2f', ('feat', 'impt'), 'feature importance (fraction of decision trees using the feature)'),
    'feature_name': (20, '%20s', ('feature', 'name'), 'name of features'),
    'loss': (9, '%9.6f', ('', 'loss'), 'loss'),
    'max_abs_error': (9, '%9.6f', ('max', '|error|'), 'maximum absolute error'),
    'max_loss': (9, '%9.6f', ('max', 'loss'), 'maximum loss'),
    'model_name': (5, '%5s', ('model', 'name'), 'name of model'),
    'mean_feature_importance': (8, '%8.6f', ('mean', 'fimp'), 'mean of feature importance (fraction of decision trees using the feature)'),
    'mean_loss': (9, '%9.6f', ('mean', 'loss'), 'mean loss'),
    'median_abs_error': (9, '%9.6f', ('median', '|error|'), 'median absolute error'),
    'min_abs_error': (9, '%9.6f', ('min', '|error|'), 'minimum absolute error'),
    'min_loss': (9, '%9.6f', ('min', 'loss'), 'minimum loss'),
    'model_spec': (35, '%-35s', ('model', 'spec'), 'model specification'),
    'n_hp_sets': (7, '%7d', ('number', 'hp sets'), 'number of hyperparameter sets'),
    'n_samples': (7, '%7d', ('number', 'samples'), 'number of samples'),
    'n_trades_back': (7, '%7d', ('n trades', 'back'), 'number of trades before query trade used in training'),
    'n_predictions': (7, '%7d', ('number', 'predns'), 'number of predictions = (n_hp_sets x n_samples)'),
    'rmse': (9, '%9.6f', ('', 'rmse'), 'root mean squared error (square root of mean error)'),
    'std_abs_error': (9, '%9.6f', ('std', '|error|'), 'standard deviation of absolute error'),
    'std_loss': (9, '%9.6f', ('std', 'loss'), 'standard deviation of loss'),
    'ticker': (6, '%6s', ('', 'ticker'), 'identifies a particular security'),
    'trade_type': (5, '%5s', ('trade', 'type'), 'type of trade: B (dealer buy), D (dealer-to-dealer, S (dealer sell)'),
}


def columns(*args):
    result = []
    for column in args:
        if column in all_columns:
            column_def = [column]
            column_def.extend(all_columns[column])
            result.append(column_def)
        else:
            print 'column %s is not in all_columns' % column
            pdb.set_trace()
    return result


def iter_fit_predict_objects(path):
    def process_unpickled_object(obj):
        print 'process_unpickled_object', obj
        pdb.set_trace()
        yield obj

    pickle_utilities.unpickle_file(
        path=path,
        process_unpickled_object=process_unpickled_object,
    )
    return


if False:
    pdb
