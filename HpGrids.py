'''define hyperparameter grids

Copyright 2017 Roy E. Lowrance, roy.lowrance@gmail.com

You may not use this file except in compliance with a License.
'''
import pdb
import sys
import unittest

import log
import ModelSpec


class HpGrid(object):

    def iter_model_specs(self):
        'yield each element: ModelSpec of the grid'
        for name in self.model_name_choices:
            if name == 'n':
                # naive model
                yield ModelSpec.ModelSpec(
                    name=name,
                )
            elif name == 'en':
                # elastic net model
                for n_trades_back in self.n_trades_back_choices:
                    for transform_x in self.transform_x_choices:
                        for transform_y in self.transform_y_choices:
                            for alpha in self.alpha_choices:
                                for l1_ratio in self.l1_ratio_choices:
                                    yield ModelSpec.ModelSpec(
                                        name=name,
                                        n_trades_back=n_trades_back,
                                        transform_x=transform_x,
                                        transform_y=transform_y,
                                        alpha=alpha,
                                        l1_ratio=l1_ratio,
                                    )
            elif name == 'rf':
                # random forests model
                for n_trades_back in self.n_trades_back_choices:
                    for n_estimators in self.n_estimators_choices:
                        for max_depth in self.max_depth_choices:
                            for max_features in self.max_features_choices:
                                yield ModelSpec.ModelSpec(
                                    name=name,
                                    n_trades_back=n_trades_back,
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    max_features=max_features,
                                )


class HpGrid0(HpGrid):
    'one choice for each hyperparameters'
    def __init__(self):
        self.model_name_choices = ('n', 'rf', 'en')
        self.n_trades_back_choices = (10,)
        self.transform_x_choices = ('log1p',)
        self.transform_y_choices = ('log',)
        self.alpha_choices = (0.50,)
        self.l1_ratio_choices = (0.50,)
        self.n_estimators_choices = (100,)
        self.max_depth_choices = (20,)
        self.max_features_choices = ('sqrt',)


class HpGrid1(HpGrid):
    'original test grid used during initial development of fit-predict'
    def __init__(self):
        self.model_name_choices = ('n', 'rf', 'en')

        # hyperparamater settings grid
        # For now, just enought to exercise fit-predict.py
        self.n_trades_back_choices = (1, 100)
        self.transform_x_choices = (None, 'log1p')
        self.transform_y_choices = (None, 'log')
        self.alpha_choices = (0.01, 1.00)  # weight on penalty term
        self.l1_ratio_choices = (0.01, 0.99)  # 0 ==> only L2 penalty, 1 ==> only L1 penalty
        self.n_estimators_choices = (1, 10)    # number of trees in the forest
        self.max_depth_choices = (2, None)     # max depth of a treeclass HpGrid1(HpGrid):
        self.max_features_choices = (1, 3)     # number of features to consider when looking for best split


class HpGrid2(HpGrid):
    'test grid 2'
    def __init__(self):
        self.model_name_choices = ('n', 'rf', 'en')

        # hyperparamater settings grid
        # For now, just enought to exercise fit-predict.py
        self.n_trades_back_choices = (1, 10, 30, 100, 300, 1000)
        self.transform_x_choices = (None, 'log1p')
        self.transform_y_choices = (None, 'log')
        self.alpha_choices = (0.001, 0.003, 0.1, .3, 1.00)  # weight on penalty term
        self.l1_ratio_choices = (0.01, 0.50, 0.99)  # 0 ==> only L2 penalty, 1 ==> only L1 penalty
        self.n_estimators_choices = (10, 100)    # number of trees in the forest
        self.max_depth_choices = (1, 3, None)     # max depth of a tree
        self.max_features_choices = ('sqrt', None)     # number of features to consider when looking for best split


class HpGrid3(HpGrid):
    'test grid 3'
    def __init__(self):
        self.model_name_choices = ('n', 'rf', 'en')

        # hyperparamater settings grid
        # For now, just enought to exercise fit-predict.py
        self.n_trades_back_choices = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 100, 300, 1000)
        self.transform_x_choices = (None, 'log1p')
        self.transform_y_choices = (None, 'log')
        self.alpha_choices = (0.001, 0.003, 0.1, .3, 1.00)  # weight on penalty term
        self.l1_ratio_choices = (0.01, 0.50, 0.99)  # 0 ==> only L2 penalty, 1 ==> only L1 penalty
        self.n_estimators_choices = (1, 3, 10, 30, 100)    # number of trees in the forest
        self.max_depth_choices = (1, 3, None)     # max depth of a tree
        self.max_features_choices = ('sqrt', None)     # number of features to consider when looking for best split


class HpGrid4(HpGrid):
    'test grid 4'
    def __init__(self):
        self.model_name_choices = ('n', 'rf', 'en')

        # hyperparamater settings grid
        # For now, just enought to exercise fit-predict.py
        self.n_trades_back_choices = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 100)
        self.transform_x_choices = (None, 'log1p')
        self.transform_y_choices = (None, 'log')
        self.alpha_choices = (0.001, 0.003, 0.1, .3, 1.00)  # weight on penalty term
        self.l1_ratio_choices = (0.01, 0.50, 0.99)  # 0 ==> only L2 penalty, 1 ==> only L1 penalty
        self.n_estimators_choices = (1, 3, 10, 30, 100)    # number of trees in the forest
        self.max_depth_choices = (1, 3, None)     # max depth of a tree
        self.max_features_choices = ('sqrt', None)     # number of features to consider when looking for best split


class HpGrid5(HpGrid):
    'test grid 5'
    def __init__(self):
        self.model_name_choices = ('n', 'rf', 'en')

        # hyperparamater settings grid
        # For now, just enought to exercise fit-predict.py
        self.n_trades_back_choices = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 100, 150, 200)
        self.transform_x_choices = (None, 'log1p')
        self.transform_y_choices = (None,)  # y is the oasspread, and it can be negative
        self.alpha_choices = (0.001, 0.003, 0.1, .3, 1.00)  # weight on penalty term
        self.l1_ratio_choices = (0.01, 0.50, 0.99)  # 0 ==> only L2 penalty, 1 ==> only L1 penalty
        self.n_estimators_choices = (1, 3, 10, 30, 100)    # number of trees in the forest
        self.max_depth_choices = (1, 3, None)     # max depth of a tree
        self.max_features_choices = ('sqrt', None)     # number of features to consider when looking for best split


def construct_HpGridN(hpset):
    'factory'
    if hpset == 'grid1':
        return HpGrid1()
    elif hpset == 'grid2':
        return HpGrid2()
    elif hpset == 'grid3':
        return HpGrid3()
    elif hpset == 'grid4':
        return HpGrid4()
    elif hpset == 'grid5':
        return HpGrid5()
    else:
        log.critical('bad hpset value %s' % hpset)
        sys.exit(1)


def common_to_from_str_test(Cls, self):
    c = Cls()
    for model_spec in c.iter_model_specs():
        s = str(model_spec)
        model_spec2 = ModelSpec.ModelSpec.make_from_str(s)
        self.assertEqual(model_spec, model_spec2)
        self.assertEqual(s, str(model_spec2))


class TestHpGrid0(unittest.TestCase):
    def test_to_from_str(self):
        common_to_from_str_test(HpGrid0, self)


class TestHpGrid1(unittest.TestCase):
    def test_to_from_str(self):
        common_to_from_str_test(HpGrid1, self)


class TestHpGrid2(unittest.TestCase):
    def test_to_from_str(self):
        common_to_from_str_test(HpGrid2, self)


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
