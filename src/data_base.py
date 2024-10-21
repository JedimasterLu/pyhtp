'''
This file is to process the icsd database by pyxrd
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pyhtp.xrd import XrdDatabase


db = XrdDatabase(file_dir='data/GeSbSn_icsd/')
db.process()
