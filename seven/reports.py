import pdb

all_columns = {
    'ticker': (6, '%6s', ('', 'ticker'), 'identifies a particular security'),
    'cusip': (9, '%9s', ('', 'cuisp'), 'identifies a particular security'),
    'min_loss': (9, '%9.2f', ('min', 'loss'), 'minimum loss'),
    'mean_loss': (9, '%9.2f', ('mean', 'loss'), 'mean loss'),
    'max_loss': (9, '%9.2f', ('max', 'loss'), 'maximum loss'),
    'loss': (9, '%9.2f', ('', 'loss'), 'loss'),
    'model_spec': (20, '%-20s', ('model', 'spec'), 'model specification'),
}


def columns(*args):
    result = []
    for column in args:
        if column in all_columns:
            column_def = [column]
            column_def.extend(all_columns[column])
            result.append(column_def)
    return result


if False:
    pdb
