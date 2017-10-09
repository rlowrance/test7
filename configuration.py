import argparse
import copy
import json
import pdb
import pprint
from pprint import pprint as pp
import typing
import unittest


class Configuration:
    def __init__(self, d):
        assert isinstance(d, dict)
        self._d = copy.copy(d)
        self._pp = pprint.PrettyPrinter()

    @staticmethod
    def from_path(path):
        with open(path, 'r') as f:
            return Configuration(json.loads(f.read()))

    def __str__(self):
        return self._pp.pformat(self._d)

    def as_dict(self):
        return self._d

    def apply_overrides(self, overrides, program):
        'return a new Configuration, with overrides applied'
        def bounded_by_quote_chars(s: str, quote_char: str):
            assert len(quote_char) == 1
            return s[0] == quote_char and s[-1] == quote_char
        
        def value_of(s: str):
            try:
                # write floating point literals using a _ in place of the . (decimal point)
                assert '_' in s  # the _ signifies a literal floating point value
                return float(s.replace('_', '.'))
            except:
                pass
            try:
                return int(s)
            except:
                pass

            if s == 'True':
                return True
            elif s == 'False':
                return False
            elif bounded_by_quote_chars(s, '"') or bounded_by_quote_chars(s, "'"):
                # strip paired beginning and ending quote characters
                return s[1:-1]
            else:
                return s
            
        def apply_override(parts: typing.List[str], d: dict):
            'mutate d to reflect the overridden (or newly set) value'
            if len(parts) == 2:
                d[parts[0]] = value_of(parts[1])
                return
            else:
                apply_override(parts[1:], d[parts[0]])
            
            pass

        new_d = copy.copy(self._d)
        for override in overrides:
            apply_override(override.split('.'), new_d['programs'][program])
        return Configuration(new_d)
        

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('overrides', type=str, nargs='*')
    parser.add_argument('--config_path')
    args = parser.parse_args(argv)
    return args


def make_with_overrides(argv, path, program):
    args = parse_arguments(argv)
    c1 = Configuration.from_path(args.config_path)
    c = c1.apply_overrides(
        overrides=args.overrides,
        program=program,
    )
    return c


class Test(unittest.TestCase):
    def test(self): 
        path = 'configuration_unittest.json'
        argv = [
            'a_int.10',
            'a_float.1_23',
            'test.True',
            'c.new_string_value',
            'd."a string with spaces"',
            '--config_path',
            path,
        ]
        config_all = make_with_overrides(
            argv=argv,
            path=path,
            program='configuration.py',
        )
        config_all_dict = config_all.as_dict()
        self.assertTrue("queues" in config_all_dict)
        self.assertTrue("programs" in config_all_dict)
        
        queues = config_all_dict["queues"]
        self.assertTrue("events.cusip" in queues)

        programs = config_all_dict["programs"]
        me = programs["configuration.py"]
        self.assertEqual(me["a_int"], 10)
        self.assertAlmostEqual(me["a_float"], 1.23)
        self.assertEqual(me["b.1"], "abc")
        self.assertEqual(me["test"], True)
        self.assertEqual(me["c"], "new_string_value")
        self.assertEqual(me["d"], "a string with spaces")


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
        pp
