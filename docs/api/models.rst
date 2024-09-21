Model
=====

The Model module of SpaHDmap encapsulates the model architecture, including the `SpaHDmapUnet` and the `GraphAutoEncoder`.

.. currentmodule:: SpaHDmap

.. autoclass:: SpaHDmap.model.SpaHDmapUnet
   :members:

.. autoclass:: SpaHDmap.model.GraphAutoEncoder
   :members:


.. note::
   For implementation of the UNet model, we refer to the model `HINet <https://arxiv.org/abs/2105.06086>`_ and its open-source `code <https://github.com/megvii-model/HINet/blob/main/basicsr/models/archs/hinet_arch.py>`_.

   Thanks for the authors for their great work.