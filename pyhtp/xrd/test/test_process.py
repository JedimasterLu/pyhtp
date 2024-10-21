# -*- coding: utf-8 -*-
"""
Filename: test_process.py
Author: Junyuan Lu
Contact: Lujunyuan@sjtu.edu.cn
"""
import unittest
import numpy as np
from pyhtp.xrd.process import XrdProcess


class TestXrdProcess(unittest.TestCase):
    '''TestXrdProcess is a class that test XrdProcess.'''

    def test_init(self):
        '''Test init function of XrdProcess.'''
        model = XrdProcess(
            file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                       'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy'],
        )
        self.assertIsNotNone(model.intensity)
        self.assertIsNotNone(model.two_theta)
        with self.assertRaises(ValueError):
            model = XrdProcess(
                file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy'],
            )
        with self.assertRaises(ValueError):
            model = XrdProcess(
                file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                           'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy'],
                two_theta=np.array([1, 2, 3]),
            )
        with self.assertRaises(ValueError):
            model = XrdProcess(
                intensity=np.array([1, 2, 3]),
                two_theta=np.array([10, 20, 30, 40])
            )

    def test_copy(self):
        '''Test copy function of XrdProcess.'''
        model = XrdProcess(
            file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                       'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy'],
            pattern_path='pyhtp/xrd/test/test_cif/pattern.pkl',
            structure_path='pyhtp/xrd/test/test_cif/structure.pkl',
        )
        model_copy = model.copy()
        self.assertEqual(np.allclose(model.intensity, model_copy.intensity), True)
        self.assertEqual(np.allclose(model.two_theta, model_copy.two_theta), True)
        self.assertEqual(model.pattern_path, model_copy.pattern_path)
        self.assertEqual(model.structure_path, model_copy.structure_path)
        model_copy.intensity = np.array([1, 2, 3])
        self.assertNotEqual(model_copy.intensity.shape, model.intensity.shape)

    def test_set_data(self):
        '''Test set_data function of XrdProcess.'''
        model = XrdProcess()
        model.set_data(
            intensity=np.array([1, 2, 3]),
            two_theta=np.array([10, 20, 30]),
        )
        self.assertEqual(np.allclose(model.intensity, np.array([1, 2, 3])), True)
        self.assertEqual(np.allclose(model.two_theta, np.array([10, 20, 30])), True)
        with self.assertRaises(ValueError):
            model.set_data(
                intensity=np.array([1, 2, 3]),
                two_theta=np.array([10, 20]),
            )
        with self.assertRaises(ValueError):
            model.set_data(
                intensity=np.array([1, 2, 3]),
                two_theta=np.array([10, 20, 30]),
                file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                           'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy'],
            )
        with self.assertRaises(ValueError):
            model.set_data(
                intensity=np.array([1, 2, 3]),
                two_theta=np.array([10, 20, 30]),
                file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy']
            )
        model = XrdProcess()
        model.set_data(
            file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                       'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy']
        )

    def test_set_database_path(self):
        '''Test set_database_path function of XrdProcess.'''
        model = XrdProcess()
        model.set_database_path(
            pattern_path='pyhtp/xrd/test/test_cif/pattern.pkl',
            structure_path='pyhtp/xrd/test/test_cif/structure.pkl',
        )
        self.assertEqual(model.pattern_path, 'pyhtp/xrd/test/test_cif/pattern.pkl')
        self.assertEqual(model.structure_path, 'pyhtp/xrd/test/test_cif/structure.pkl')

    def test_substract_baseline(self):
        '''Test substract_baseline function of XrdProcess.'''
        model = XrdProcess(
            file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                       'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy'],
        )
        original_intensity = model.intensity.copy()
        model = model.substract_baseline()
        self.assertIsNotNone(model.intensity)
        self.assertIsNotNone(model.two_theta)
        self.assertNotEqual(np.allclose(model.intensity, original_intensity), True)
        with self.assertRaises(ValueError):
            model = XrdProcess()
            model = model.substract_baseline()

    def test_get_baseline(self):
        '''Test get_baseline function of XrdProcess.'''
        model = XrdProcess(
            file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                       'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy'],
        )
        baseline = model.get_baseline()
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.shape, model.intensity.shape)
        with self.assertRaises(ValueError):
            model = XrdProcess()
            baseline = model.get_baseline()

    def test_smooth(self):
        '''Test smooth function of XrdProcess.'''
        model = XrdProcess(
            file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                       'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy'],
        )
        original_intensity = model.intensity.copy()
        model = model.smooth()
        self.assertIsNotNone(model.intensity)
        self.assertIsNotNone(model.two_theta)
        self.assertNotEqual(np.allclose(model.intensity, original_intensity), True)
        with self.assertRaises(ValueError):
            model = XrdProcess()
            model = model.smooth()

    def test_peaks(self):
        '''Test peaks function of XrdProcess.'''
        model = XrdProcess(
            file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                       'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy'],
        )
        peaks_value, peaks_index, properties = model.substract_baseline().smooth().peaks()
        self.assertIsInstance(peaks_value, np.ndarray)
        self.assertIsInstance(peaks_index, np.ndarray)
        self.assertIsInstance(properties, dict)
        self.assertEqual(peaks_value.shape, peaks_index.shape)
        peaks_value, peaks_index, properties = model.substract_baseline().smooth().peaks(mask=[[33, 35]])
        self.assertIsInstance(peaks_value, np.ndarray)
        self.assertIsInstance(peaks_index, np.ndarray)
        self.assertIsInstance(properties, dict)
        self.assertEqual(peaks_value.shape, peaks_index.shape)
        with self.assertRaises(ValueError):
            peaks_value, peaks_index, properties = model.peaks(mask=[33, 25, 27])
        with self.assertRaises(ValueError):
            peaks_value, peaks_index, properties = model.peaks(mask=[[33, 25, 27]])
        peaks_value, peaks_index, _ = model.substract_baseline().smooth().peaks(mask=[[33, 35.5]], mask_height=0.1, height=0.06)
        self.assertEqual(len(peaks_value), len(peaks_index))
        with self.assertRaises(ValueError):
            model = XrdProcess()
            peaks_value, peaks_index, properties = model.peaks()
        model = XrdProcess(
            file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                       'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy'],
        )
        peaks_value, peaks_index, properties = model.substract_baseline().smooth().peaks(
            mask=[[20, 40]], mask_height=0.07, height=0.06
        )

    def test_match(self):
        '''Test match function of XrdProcess.'''
        model = XrdProcess(
            file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                       'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy'],
        )
        model = model.substract_baseline().smooth()
        model.set_database_path(
            pattern_path='pyhtp/xrd/test/test_cif/pattern.pkl',
            structure_path='pyhtp/xrd/test/test_cif/structure.pkl',
        )
        match_result = model.match()
        self.assertIsInstance(match_result, list)
        self.assertIsInstance(match_result[0], str)
        self.assertEqual(len(match_result), 2)
        with self.assertRaises(ValueError):
            model = XrdProcess()
            model = model.match()
        model = XrdProcess(
            file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-208-0000_exported.xy',
                       'pyhtp/xrd/test/test_xy/46-analyze/46-208-0000_exported.xy'],
        )
        model = model.substract_baseline().smooth()
        model.set_database_path(
            pattern_path='pyhtp/xrd/test/test_cif/pattern.pkl',
            structure_path='pyhtp/xrd/test/test_cif/structure.pkl',
        )
        match_result = model.match()
        self.assertIsInstance(match_result, list)
        self.assertIsInstance(match_result[0], str)
        self.assertEqual(len(match_result), 2)

    def test_identify(self):
        '''Test identify function of XrdProcess.'''
        model = XrdProcess(
            file_path=['pyhtp/xrd/test/test_xy/22-analyze/22-000-0000_exported.xy',
                       'pyhtp/xrd/test/test_xy/46-analyze/46-000-0000_exported.xy'],
        )
        model = model.substract_baseline().smooth()
        model.set_database_path(
            pattern_path='pyhtp/xrd/test/test_cif/pattern.pkl',
            structure_path='pyhtp/xrd/test/test_cif/structure.pkl',
        )
        identify_result = model.identify(if_show=False, display_number=2)
        identify_result = model.identify(if_show=False, display_number=5, if_process=False, figure_title='test', save_path='test.png')
        self.assertIsInstance(identify_result, np.ndarray)

    def test_create_mask(self):
        '''Test create_mask function of XrdProcess.'''
        model = XrdProcess(
            two_theta=np.arange(0, 10, 1),
            intensity=np.ones(10),
        )
        # pylint: disable=W0212
        mask_condition = model._create_mask(left_angle=2, right_angle=5)
        self.assertEqual(np.allclose(mask_condition, np.array([False, False, True, True, True, False, False, False, False, False])), True)
        mask_condition = model._create_mask(left_angle=2, right_angle=5, current_mask=np.array([True, False, False, False, False, False, False, False, False, False]))
        self.assertEqual(np.allclose(mask_condition, np.array([True, False, True, True, True, False, False, False, False, False])), True)
        # pylint: enable=W0212

    def test_similar_peak_number(self):
        '''Test similar_peak_number function of XrdProcess.'''
        model = XrdProcess(
            two_theta=np.arange(0, 10, 1),
            intensity=np.ones(10),
        )
        # pylint: disable=W0212
        similar_peak_number = model._similar_peak_number([], [1])
        self.assertEqual(similar_peak_number, 0)
        similar_peak_number = model._similar_peak_number([1], [])
        self.assertEqual(similar_peak_number, 0)
        similar_peak_number = model._similar_peak_number([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], tolerance=0.01)
        self.assertEqual(similar_peak_number, 15)
        # pylint: enable=W0212

    def test_avg_min_lse(self):
        '''Test avg_min_lse function of XrdProcess.'''
        model = XrdProcess(
            two_theta=np.arange(0, 10, 1),
            intensity=np.ones(10),
        )
        # pylint: disable=W0212
        similar_peak_number = model._avg_min_lse([], [1])
        self.assertEqual(similar_peak_number, 0)
        similar_peak_number = model._avg_min_lse([1], [])
        self.assertEqual(similar_peak_number, 0)
        # pylint: enable=W0212
