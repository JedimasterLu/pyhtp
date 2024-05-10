# -*- coding: utf-8 -*-
"""
Filename: test_database.py
Author: Junyuan Lu
Contact: Lujunyuan@sjtu.edu.cn
"""
import unittest
import numpy as np
from pyxrd.database import XrdDatabase


class TestXrdDatabase(unittest.TestCase):
    '''TestXrdDatabase is a class that test XrdDatabase.'''

    def test_init(self):
        '''test_init test __init__ method of XrdDatabase.'''
        file_dir = 'pyxrd/test/test_cif/'
        save_dir = 'pyxrd/test/test_cif/'
        xrd_database = XrdDatabase(file_dir, save_dir)
        self.assertEqual(xrd_database.file_dir, file_dir)
        self.assertEqual(xrd_database.save_dir, save_dir)
        xrd_database = XrdDatabase(file_dir)
        self.assertEqual(xrd_database.file_dir, file_dir)
        self.assertEqual(xrd_database.save_dir, file_dir)

    def test_process(self):
        '''test_process test process method of XrdDatabase.'''
        file_dir = 'pyxrd/test/test_cif/'
        save_dir = 'pyxrd/test/test_cif/'
        xrd_database = XrdDatabase(file_dir, save_dir)
        pattern_database, structure_database = xrd_database.process()
        self.assertEqual(len(pattern_database), 2)
        self.assertEqual(len(structure_database), 2)

    def test_get_pattern(self):
        '''test_get_pattern test get_pattern method of XrdDatabase.'''
        file_dir = 'pyxrd/test/test_cif/'
        save_dir = 'pyxrd/test/test_cif/'
        xrd_database = XrdDatabase(file_dir, save_dir)
        pattern_database, _ = xrd_database.process(if_save=False)
        pattern_database_ = xrd_database.get_pattern()
        self.assertEqual(len(pattern_database), len(pattern_database_))
        self.assertEqual(pattern_database[0]['name'], pattern_database_[0]['name'])
        self.assertEqual(np.allclose(pattern_database[0]['two_theta'], pattern_database_[0]['two_theta']), True)
        self.assertEqual(np.allclose(pattern_database[0]['intensity'], pattern_database_[0]['intensity']), True)
