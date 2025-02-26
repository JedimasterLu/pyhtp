import time
import multiprocessing

import matplotlib

from pyhtp.xrd import XRDDatabase
from pyhtp.typing import SampleInfo, AngleRange

if __name__ == '__main__':

    FILEPATH = 'data/GSTSe_XRD/GSTSe-1-300'

    multiprocessing.set_start_method('spawn')
    matplotlib.use('TkAgg')

    sample_info = SampleInfo(
        name='GSST-1-350',
        element=['Se', 'Sb', 'Ge', 'Te'],
        film_thickness=[12.4, 7.6, 12.4, 7.6],
        two_theta_range=AngleRange(28.5, 52),
        temperature=300)

    db = XRDDatabase(file_dir='data/GSTSe_XRD/GSTSe-1-350', info=sample_info)

    print("End of import.")

    db.data[0].postprocess(baseline_lam=700, window=151, spline_lam=0.1)

    def normal_process():
        print("Normal process")
        start_time = time.time()
        db.postprocess(baseline_lam=700, window=151, spline_lam=-1, concurrent=False)
        print(f'Normal: {time.time() - start_time:.2f} s')

    def concurrent_process():
        print("Concurrent process")
        start_time = time.time()
        db.postprocess(baseline_lam=700, window=151, spline_lam=-1, concurrent=True)
        print(f'Chunk: {time.time() - start_time:.2f} s')

    normal_process()
    concurrent_process()
