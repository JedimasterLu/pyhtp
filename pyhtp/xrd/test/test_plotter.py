# -*- coding: utf-8 -*-
"""
Filename: test_plotter.py
Author: Junyuan Lu
Contact: Lujunyuan@sjtu.edu.cn
"""
import unittest
import cmcrameri
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyhtp.xrd.plotter import XrdPlotter, _get_width


class TestXrdPlotter(unittest.TestCase):
    '''TestXrdPlotter is a class that test XrdPlotter.'''

    def test_init(self):
        '''Test init function of XrdPlotter.'''
        plotter = XrdPlotter(
            file_dir=['pyhtp/xrd/test/test_xy/22-analyze/',
                      'pyhtp/xrd/test/test_xy/46-analyze/']
        )
        self.assertIsNotNone(plotter.file_dir)
        self.assertIsNotNone(plotter.left_xy)
        self.assertIsNotNone(plotter.right_xy)
        plotter = XrdPlotter(
            file_dir=['pyhtp/xrd/test/test_xy/22-analyze/',
                      'pyhtp/xrd/test/test_xy/46-analyze/'],
            save_dir='pyhtp/xrd/test/test_xy/',
            title='test_fig'
        )
        self.assertIsNotNone(plotter.file_dir)
        self.assertIsNotNone(plotter.left_xy)
        self.assertIsNotNone(plotter.right_xy)
        self.assertIsNotNone(plotter.save_dir)
        self.assertIsNotNone(plotter.title)
        # pylint: disable=no-member
        self.assertEqual(plotter.colormap, cmcrameri.cm.batlowS)
        # pylint: enable=no-member
        with self.assertRaises(ValueError):
            XrdPlotter(file_dir=['pyhtp/xrd/test/test_xy/22-analyze/'])

    def test_plot_animation(self):
        '''Test plot_animation function of XrdPlotter.'''
        plotter = XrdPlotter(
            file_dir=['pyhtp/xrd/test/test_xy/22-analyze/',
                      'pyhtp/xrd/test/test_xy/46-analyze/'],
            save_dir='pyhtp/xrd/test/test_xy/',
            title='test_fig'
        )
        plotter.plot_animation(dpi=100)

    def test_plot_spectrum(self):
        '''Test plot_spectrum function of XrdPlotter.'''
        plotter = XrdPlotter(
            file_dir=['pyhtp/xrd/test/test_xy/22-analyze/',
                      'pyhtp/xrd/test/test_xy/46-analyze/'],
            save_dir='pyhtp/xrd/test/test_xy/',
            title='test_fig'
        )
        ax = plotter.plot_spectrum(
            dpi=300,
            if_save=False,
            if_show=False
        )
        self.assertIsInstance(ax, mpl.axes.Axes)
        ax = plotter.plot_spectrum(
            dpi=300,
            index_to_plot=[0],
            if_save=False,
            if_show=False
        )
        self.assertIsInstance(ax, mpl.axes.Axes)
        with self.assertRaises(ValueError):
            plotter.plot_spectrum(
                dpi=300,
                index_to_plot=[0, 1, 2, 3],
                if_save=False,
                if_show=False
            )
        ax = plotter.plot_spectrum(
            dpi=300,
            plot_type='stack',
            if_show=False,
            if_save=False
        )
        self.assertIsInstance(ax, mpl.axes.Axes)

    def test_plot_spectrum_with_ref(self):
        '''Test plot_spectrum_with_ref function of XrdPlotter.'''
        plotter = XrdPlotter(
            file_dir=['pyhtp/xrd/test/test_xy/22-analyze/',
                      'pyhtp/xrd/test/test_xy/46-analyze/'],
            save_dir='pyhtp/xrd/test/test_xy/',
            title='test_fig'
        )
        ax = plotter.plot_spectrum_with_ref(
            dpi=300,
            if_save=False,
            if_show=False,
            database_dir='pyhtp/xrd/test/test_cif/',
            plot_type='stack',
            save_path='pyhtp/xrd/test/test_xy/test_fig.png'
        )
        self.assertIsInstance(ax, mpl.axes.Axes)
        _, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0] = plotter.plot_spectrum(ax=ax[0], index_to_plot=[0, 1], plot_type='stack', if_show=False)
        ax[1] = plotter.plot_spectrum(ax=ax[1], index_to_plot=[0, 1], plot_type='combine', if_show=False)
        ax[2] = plotter.plot_spectrum_with_ref(ax=ax[2], index_to_plot=[0, 1], plot_type='stack', if_show=False, database_dir='pyhtp/xrd/test/test_cif/')
        self.assertIsInstance(ax[0], mpl.axes.Axes)

    def test_set_save_dir(self):
        '''Test set_save_dir function of XrdPlotter.'''
        plotter = XrdPlotter(
            file_dir=['pyhtp/xrd/test/test_xy/22-analyze/',
                      'pyhtp/xrd/test/test_xy/46-analyze/'],
        )
        plotter.set_save_dir('pyhtp/xrd/test/test_xy/')
        self.assertIsNotNone(plotter.save_dir)

    def test_get_width(self):
        '''Test get_width function.'''
        index = [0, 1, 2, 3, 4, 5]
        width = _get_width(index)
        self.assertEqual(width, 3)
        with self.assertRaises(ValueError):
            _get_width([0, 1, 2, 3, 4, 5, 6])
