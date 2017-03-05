import pdb

import pickle_utilities

all_columns = {
    'cusip': (9, '%9s', ('', 'cuisp'), 'identifies a particular security'),
    'feature_importance': (4, '%4.2f', ('feat', 'impt'), 'feature importance (fraction of decision trees using the feature)'),
    'feature_name': (20, '%20s', ('feature', 'name'), 'name of features'),
    'loss': (9, '%9.6f', ('', 'loss'), 'loss'),
    'max_loss': (9, '%9.6f', ('max', 'loss'), 'maximum loss'),
    'model_name': (5, '%5s', ('model', 'name'), 'name of model'),
    'mean_feature_importance': (4, '%4.2f', ('mean', 'fimp'), 'mean of feature importance (fraction of decision trees using the feature)'),
    'mean_loss': (9, '%9.6f', ('mean', 'loss'), 'mean loss'),
    'median_loss': (9, '%9.6f', ('median', 'loss'), 'median loss'),
    'min_loss': (9, '%9.6f', ('min', 'loss'), 'minimum loss'),
    'model_spec': (35, '%-35s', ('model', 'spec'), 'model specification'),
    'n_hp_sets': (7, '%7d', ('number', 'hp sets'), 'number of hyperparameter sets'),
    'n_samples': (7, '%7d', ('number', 'samples'), 'number of samples'),
    'std_loss': (9, '%9.6f', ('std', 'loss'), 'standard deviation of loss'),
    'ticker': (6, '%6s', ('', 'ticker'), 'identifies a particular security'),
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
