import unittest


def make_verbose_print(verbose: bool):
    if verbose:
        def verbose_print(*args):
            print(*args)
    else:
        def verbose_print(*args):
            pass
    return verbose_print


class Test(unittest.TestCase):
    def test_make_verbose_print(self):
        vp1 = make_verbose_print(True)
        vp1('a', 123)
        vp2 = make_verbose_print(False)
        vp2('a', 123)  # should print

if __name__ == '__main__':
    unittest.main()
