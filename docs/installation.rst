Installation
============

SpaHDmap can be installed using pip or directly from the source code. Here are the different methods to install SpaHDmap:

Using pip
---------

The easiest way to install SpaHDmap is using pip:

.. code-block:: bash

   pip install SpaHDmap

This will install the latest stable version of SpaHDmap along with all its dependencies.

From Source
-----------

To install SpaHDmap from source, first clone the repository:

.. code-block:: bash

   git clone https://github.com/sldyns/SpaHDmap.git

Then, install the package using pip:

.. code-block:: bash

   pip install -e .

This will install SpaHDmap in editable mode, which is useful if you want to modify the source code.

Requirements
------------

SpaHDmap requires Python 3.9 or later. The main dependencies are:

- numpy>=1.22
- cython>=0.29.24
- torch>=1.12
- scikit-learn>=1.0.1
- squidpy
- matplotlib
- tqdm
- h5py
- scanpy
- anndata
- pandas
- scikit-image
- opencv-python
- imagecodecs
- scipy
- seaborn

These dependencies will be automatically installed when you install SpaHDmap using pip.


GPU Support
-----------

SpaHDmap can leverage GPU acceleration for faster computation. To use GPU support, make sure you have a CUDA-compatible GPU and the appropriate CUDA toolkit installed. Then, install the GPU version of PyTorch:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

Replace `cu113` with the appropriate CUDA version for your system.

Verifying Installation
----------------------

After installation, you can verify that SpaHDmap is correctly installed by running:

.. code-block:: python

   import SpaHDmap as map
   print(map.__version__)

This should print the version number of SpaHDmap without any errors.

Troubleshooting
---------------

If you encounter any issues during installation, please check the following:

1. Ensure you have the latest version of pip:

   .. code-block:: bash

      pip install --upgrade pip

2. If you're using a virtual environment, make sure it's activated.

3. On Windows, you might need to install Microsoft Visual C++ Build Tools.

If you still face problems, please open an issue on the `SpaHDmap GitHub repository <https://github.com/sldyns/SpaHDmap/issues>`_.
