'''
Process the XRD data for the quaternary samples.
'''
import numpy as np
from pyhtp.xrd import XrdDatabase, ICSD
from pyhtp.typing import SampleInfo, AngleRange, PeakParam
from pyhtp.quaternary import scatter_quaternary

sample1 = SampleInfo(
    name='GSTSe-2', element=['Se', 'Sb', 'Ge', 'Te'],
    angle_range=AngleRange(28, 52), temperature=350,
    film_thickness=[12.4, 7.6, 12.4, 7.6])
db1 = XrdDatabase(file_dir='data/GSTSe_XRD/GSTSe-2-350', info=sample1)
icsd = ICSD(file_dir='data/GSST ICSD')

label = np.zeros(400)

scatter_quaternary(
    value=label, label=('Ge', 'Sb', 'Se', 'Te'),
    database=db1,
    path_type='snakelike', interactive=True,
    lam=700, window=51,
    param=PeakParam(height=0.2, distance=0.5, prominence=0.5))
