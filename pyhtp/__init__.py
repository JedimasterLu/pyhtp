"""
A Python package for high-throughput data analysis in materials science.
Arthur: Junyuan Lu
"""
import os
import matplotlib

# For linux system, use 'TkAgg' backend
if os.name == 'posix':
    matplotlib.use('TkAgg')
