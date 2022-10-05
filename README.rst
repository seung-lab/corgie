corgie: COnnectomics Registration Generalizable Inference Engine
================================================================

|Docs Badge| |Python Badge|

Welcome to corgie! corgie is a toolkit for registration of large 3D volumes.

The recommended installation method is `pip <https://pip.pypa.io/en/stable/>`_-installing into a `virtualenv <https://hynek.me/articles/virtualenv-lives/>`_:

.. code-block:: 

   git clone git@github.com:seung-lab/corgie.git
   cd corgie
   pip install -e .

Installation typically takes under 10 minutes.

To install the pinned versions of the package requirements, refer to the `requirements-release.txt`:

.. code-block::
   
   pip install -r requirements-release.txt


You can find demo walkthrough with links to the demo dataset and expected output in the `Documentation <https://corgie.readthedocs.io/en/latest/>`_.

Project Information
===================

- **License**: `MIT <https://choosealicense.com/licenses/mit/>`_
- **Source Code**: https://github.com/seung-lab/corgie
- **Documentation**:  https://corgie.readthedocs.io/en/latest/
- **Supported Python Versions**: 3.6 and later

Citation
========
.. code-block:: 

   @article {Popovych2022.03.25.485816,
      author = {Popovych, Sergiy and Macrina, Thomas and Kemnitz, Nico and Castro, Manuel and Nehoran, Barak and Jia, Zhen and Bae, J. Alexander and Mitchell, Eric and Mu, Shang and Trautman, Eric T. and Saalfeld, Stephan and Li, Kai and Seung, Sebastian},
      title = {Petascale pipeline for precise alignment of images from serial section electron microscopy},
      elocation-id = {2022.03.25.485816},
      year = {2022},
      doi = {10.1101/2022.03.25.485816},
      URL = {https://www.biorxiv.org/content/early/2022/03/27/2022.03.25.485816},
      eprint = {https://www.biorxiv.org/content/early/2022/03/27/2022.03.25.485816.full.pdf},
      journal = {bioRxiv}
   }



.. |Docs Badge| image:: https://readthedocs.org/projects/corgie/badge/?version=latest
    :target: https://corgie.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Python Badge| image:: https://img.shields.io/badge/python-3.6+-blue.svg

