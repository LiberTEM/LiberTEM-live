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

    def get_preferred_input_dtype(self):
        return self.USE_NATIVE_DTYPE

    def get_result_buffers(self):
        return {
            "intensity": self.buffer(kind='sig', dtype=self.meta.input_dtype)
        }

    def process_tile(self, tile):
        # Assign the portion from the last frame within the tile
        # to the result buffer
        self.results.intensity[:] = tile[-1]

    def merge(self, dest, src):
        dest.intensity[:] = src.intensity
