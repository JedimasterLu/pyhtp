# pyhtp: A Python package for high-throughput data analysis

The pyhtp is a package built for high-throughput data analysis in Materials Science. Nowadays, data-based and data-driven methods are getting more attention in the research of novel materials, and high-throughput methods have been activatly developed to accelerate materials design.

Although there exists many Python projects related to materials science, such as pymatgen, they are not for processing a large quantitiy of experimental or computational datasets. Pyhtp aims to solve this problem, enabling researchers to automatically process and analysis a large quantity of data with ready-to-use code, rather than coding new codes for new projects once and once again.

This repo is initially built for my graduation thesis on high-throughput screening of optical phase change material superlattice-like thin films. The package includes data analysis tools for XRD, XRF, and VASE. Some functions may not be general, since they are initially built for the characterization facilities in my lab. If you would like to implement this package in the workflow of your project, please do not hesitate to raise issues or pull requests. I would be glad to expand the generizability of pyhtp, and I believe it could only be achieved by collaborations of materials science researchers.

If you would like to commit to this repo, please note that:
- This repo obey almost all of the PEP8 style rule.
- The docstring is in Google type.

TODO:
- sphinx is deployed for doc generation, but it is not updated.
- The core functions are still under active development.
