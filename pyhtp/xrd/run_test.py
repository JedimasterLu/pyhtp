# -*- coding: utf-8 -*-
"""
Filename: run_test.py
Author: Junyuan Lu
Contact: Lujunyuan@sjtu.edu.cn
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# pylint: disable=C0413,C0303
import unittest  # noqa: E402
from pyhtp.xrd.test import TestXrdDatabase, TestXrdProcess, TestXrdPlotter  # noqa: E402
# pylint: enable=C0413,C0303


def run_all_tests():
    '''Run all tests in the test suite.'''
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestXrdDatabase))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestXrdProcess))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestXrdPlotter))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all_tests()
