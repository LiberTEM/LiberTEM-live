from typing import Iterable, Tuple

import numpy as np
from libertem.common import Shape


class LiveMeta:
    """
    Meta information for an acquisition

    An instance of this class is used to pass information about the current
    acquisition to the user-provided :code:`on_enter` and :code:`on_exit`
    functions of an acquisition.

    As an example, this can be used to configure the scan system of the
    microscope to match the desired acquisition shape and to set the number of
    detector frames accordingly.

    Parameters
    ----------

    dataset_shape : Shape
        :class:`libertem.common.Shape` object for the acquisition
    dataset_dtype : numpy.dtype
    udfs : Iterable
        Iterable with the udfs that will be executed.
    """
    def __init__(self, dataset_shape: Shape, dataset_dtype: np.dtype,
                 udfs: Iterable):
        self._dataset_shape = dataset_shape
        self._dataset_dtype = dataset_dtype
        self._udfs = udfs

    @property
    def dataset_shape(self) -> Shape:
        """
        Shape : The original shape of the whole dataset, not influenced by the ROI
        """
        return self._dataset_shape

    @property
    def dataset_dtype(self) -> np.dtype:
        """
        numpy.dtype : Native dtype of the dataset
        """
        return self._dataset_dtype

    @property
    def udfs(self) -> Tuple:
        """
        Tuple with the UDFs that will be run.
        """
        return tuple(self._udfs)
