# -*- coding: utf-8 -*-
'''
This file plot all Ge-Sb-Sn figures in a combined plot.
'''

import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from pyhtp.xrd.plotter import plot_ternary_diagram, rotate_phase_index

# Load data
DATADIR = 'data/plot'
data_file = os.listdir(DATADIR)
fig, ax = plt.subplot_mosaic([['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']],
                             constrained_layout=False, figsize=(18, 10))
# fig, ax = plt.subplots(2, 3, figsize=(18, 10))
for data_name in data_file:
    with open(os.path.join(DATADIR, data_name), 'rb') as f:
        data = pickle.load(f)
    if data_name == 'SbSnGe_300C.pkl':
        plot_ternary_diagram(phase_type=data['ordered_custom_order'],
                             phase_index=rotate_phase_index(data['labels']),
                             labels=['Sn', 'Ge', 'Sb'],
                             title=data_name,
                             color={'amorphous': '#5EB89D',
                                    'SnSb (R-3m)': '#D1363C',
                                    'Sb (R-3m)': '#237AA6',
                                    'SnSb (I4_1/amd)': '#E89C3D',
                                    'Sn (I4_1/amd)': '#BD448E',
                                    'Ge (Fd-3m)': '#43624F',
                                    'Se (R-3)': '#F56056',
                                    'GeSe (Pnma)': '#40DB59',
                                    'Se (P3121)': '#DB7335',
                                    'SbSe (Pnma)': '#9CDC3A'},
                             if_show=False,
                             if_save=False,
                             if_legend=False,
                             rotation={'Ge (Fd-3m) + Sn (I4_1/amd)': 300,
                                       'Sb (R-3m) + SnSb (I4_1/amd)': 0},
                             ax=ax['(f)'])
    elif data_name == 'SbSnGe_150C.pkl':
        plot_ternary_diagram(phase_type=data['ordered_custom_order'],
                             phase_index=rotate_phase_index(data['labels']),
                             labels=['Sn', 'Ge', 'Sb'],
                             title=data_name,
                             color={'amorphous': '#5EB89D',
                                    'SnSb (R-3m)': '#D1363C',
                                    'Sb (R-3m)': '#237AA6',
                                    'SnSb (I4_1/amd)': '#E89C3D',
                                    'Sn (I4_1/amd)': '#BD448E',
                                    'Ge (Fd-3m)': '#43624F',
                                    'Se (R-3)': '#F56056',
                                    'GeSe (Pnma)': '#40DB59',
                                    'Se (P3121)': '#DB7335',
                                    'SbSe (Pnma)': '#9CDC3A'},
                             if_show=False,
                             if_save=False,
                             if_legend=False,
                             rotation={'Ge (Fd-3m) + Sn (I4_1/amd)': 300,
                                       'Sb (R-3m) + SnSb (I4_1/amd)': 0},
                             ax=ax['(e)'])
    elif data_name == 'GeSbSn_300C.pkl':
        plot_ternary_diagram(phase_type=data['ordered_custom_order'],
                             phase_index=data['labels'],
                             labels=['Sn', 'Ge', 'Sb'],
                             title=data_name,
                             color={'amorphous': '#5EB89D',
                                    'SnSb (R-3m)': '#D1363C',
                                    'Sb (R-3m)': '#237AA6',
                                    'SnSb (I4_1/amd)': '#E89C3D',
                                    'Sn (I4_1/amd)': '#BD448E',
                                    'Ge (Fd-3m)': '#43624F',
                                    'Se (R-3)': '#F56056',
                                    'GeSe (Pnma)': '#40DB59',
                                    'Se (P3121)': '#DB7335',
                                    'SbSe (Pnma)': '#9CDC3A'},
                             if_show=False,
                             if_save=False,
                             if_legend=True,
                             rotation={'Ge (Fd-3m) + Sn (I4_1/amd)': 300,
                                       'Sb (R-3m) + SnSb (I4_1/amd)': 0},
                             ax=ax['(c)'])
    elif data_name == 'GeSbSn_150C.pkl':
        plot_ternary_diagram(phase_type=data['ordered_custom_order'],
                             phase_index=data['labels'],
                             labels=['Sn', 'Ge', 'Sb'],
                             title=data_name,
                             color={'amorphous': '#5EB89D',
                                    'SnSb (R-3m)': '#D1363C',
                                    'Sb (R-3m)': '#237AA6',
                                    'SnSb (I4_1/amd)': '#E89C3D',
                                    'Sn (I4_1/amd)': '#BD448E',
                                    'Ge (Fd-3m)': '#43624F',
                                    'Se (R-3)': '#F56056',
                                    'GeSe (Pnma)': '#40DB59',
                                    'Se (P3121)': '#DB7335',
                                    'SbSe (Pnma)': '#9CDC3A'},
                             if_show=False,
                             if_save=False,
                             if_legend=False,
                             rotation={'Ge (Fd-3m) + Sn (I4_1/amd)': 300,
                                       'Sb (R-3m) + SnSb (I4_1/amd)': 0},
                             ax=ax['(b)'])
    else:
        plot_ternary_diagram(phase_type=data['ordered_custom_order'],
                             phase_index=data['labels'],
                             labels=['Sn', 'Ge', 'Sb'],
                             title='GeSnSb_As',
                             color={'amorphous': '#5EB89D',
                                    'SnSb (R-3m)': '#D1363C',
                                    'Sb (R-3m)': '#237AA6',
                                    'SnSb (I4_1/amd)': '#E89C3D',
                                    'Sn (I4_1/amd)': '#BD448E',
                                    'Ge (Fd-3m)': '#43624F',
                                    'Se (R-3)': '#F56056',
                                    'GeSe (Pnma)': '#40DB59',
                                    'Se (P3121)': '#DB7335',
                                    'SbSe (Pnma)': '#9CDC3A'},
                             if_show=False,
                             if_save=False,
                             if_legend=False,
                             rotation={'Ge (Fd-3m) + Sn (I4_1/amd)': 300,
                                       'Sb (R-3m) + SnSb (I4_1/amd)': 0},
                             ax=ax['(a)'])
        plot_ternary_diagram(phase_type=data['ordered_custom_order'],
                             phase_index=data['labels'],
                             labels=['Sn', 'Ge', 'Sb'],
                             title='GeSnSb_As',
                             color={'amorphous': '#5EB89D',
                                    'SnSb (R-3m)': '#D1363C',
                                    'Sb (R-3m)': '#237AA6',
                                    'SnSb (I4_1/amd)': '#E89C3D',
                                    'Sn (I4_1/amd)': '#BD448E',
                                    'Ge (Fd-3m)': '#43624F',
                                    'Se (R-3)': '#F56056',
                                    'GeSe (Pnma)': '#40DB59',
                                    'Se (P3121)': '#DB7335',
                                    'SbSe (Pnma)': '#9CDC3A'},
                             if_show=False,
                             if_save=False,
                             if_legend=False,
                             rotation={'Ge (Fd-3m) + Sn (I4_1/amd)': 300,
                                       'Sb (R-3m) + SnSb (I4_1/amd)': 0},
                             ax=ax['(d)'])
for label, ax in ax.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize=24, va='bottom', fontfamily='Times New Roman')
plt.show()
fig.savefig('combined_plot.png', dpi=600)
