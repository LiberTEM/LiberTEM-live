import numpy as np

from libertem.udf import UDF


class SignalMonitorUDF(UDF):
    '''
    Return the most recently processed signal space element (frame).

    This is useful for live processing as a beam monitor. Individual frames in
    an offline dataset are more easily accessible with
    :class:`~libertem.udf.raw.PickUDF`.

    The result is most likely the last frame of each partition for the
    individual merge steps. The end result depends on the merge order.
    '''

    def get_backends(self):
        ''
        return [
            backend for backend in self.BACKEND_ALL
            if backend not in {self.BACKEND_CUPY_SCIPY_COO, self.BACKEND_SCIPY_COO}
        ]

    # def get_preferred_input_dtype(self):
    #     ''
    #     return self.USE_NATIVE_DTYPE

    def get_result_buffers(self):
        ''
        return {
            "intensity": self.buffer(kind='sig', dtype=self.meta.input_dtype, where='device')
        }

    def process_tile(self, tile):
        ''
        # Assign the portion from the last frame within the tile
        # to the result buffer
        self.results.intensity[:] = self.forbuf(tile[-1], self.results.intensity)

    def merge(self, dest, src):
        ''
        dest.intensity[:] = src.intensity


# Largely copied from SumUDF, but separate implementation is
# desirable, for example since `merge_all()` doesn't make sense
# here.
class PartitionMonitorUDF(UDF):
    """
    Sum up frames in a partition and update result with
    the latest partition result.

    This is useful for live processing as a beam monitor if
    individual frames contain not enough signal. Partial sums of
    an offline dataset are more easily accessible with
    :class:`~libertem.udf.sum.SumUDF` with a ROI.

    The result is the sum of the partition that was merged last.
    The end result depends on the merge order.

    Parameters
    ----------
    dtype : numpy.dtype, optional
        Preferred dtype for computation, default 'float32'. The actual dtype will be determined
        from this value and the dataset's dtype using :meth:`numpy.result_type`.
        See also :ref:`udf dtype`.

    """
    def __init__(self, dtype='float32'):
        super().__init__(dtype=dtype)

    def get_preferred_input_dtype(self):
        ''
        return self.params.dtype

    def get_backends(self):
        ''
        return self.BACKEND_ALL

    def get_result_buffers(self):
        ''
        return {
            'intensity': self.buffer(
                kind='sig', dtype=self.meta.input_dtype, where='device'
            )
        }

    def process_tile(self, tile):
        ''
        self.results.intensity[:] += self.forbuf(
            np.sum(tile, axis=0),
            self.results.intensity
        )

    def merge(self, dest, src):
        ''
        dest.intensity[:] = src.intensity
