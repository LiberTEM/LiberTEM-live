import numpy as np


class LivePlot:
    def __init__(
            self, ds, kind='sig', dtype=np.float32,
            udf_index=0, postprocess=None, channel='intensity', extra_shape=(), title="",
    ):
        if kind == 'sig':
            shape = ds.shape.sig
        elif kind == 'nav':
            shape = ds.shape.nav
        elif kind == 'single':
            shape = extra_shape
        else:
            raise ValueError("unknown plot kind")
        self.shape = shape
        self.data = np.zeros(shape, dtype=dtype)
        self.udf_index = udf_index
        self.channel = channel
        self.pp = postprocess or (lambda x: x)

    def postprocess(self, udf_results):
        return self.pp(udf_results[self.udf_index][self.channel].data)

    def new_data(self, udf_results, force=False):
        self.data[:] = self.postprocess(udf_results)
        self.update(force=force)

    def update(self, force=False):
        """
        Update the plot based on `self.data`
        """
        raise NotImplementedError()
