import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np


class TBBackend:
    """Dummy TensorBoard backend - disabled"""

    def __init__(self, logdir):
        self.logdir = logdir

    def scalar_summary(self, tag, scalar, step):
        pass

    def list_summary(self, tag, list, step, multi_graphs):
        pass

    def histogram_summary(self, tag, tensor, step):
        pass

    def add_summary(self, summary, step):
        pass

    def sync_to_file(self):
        pass

    def close(self):
        pass