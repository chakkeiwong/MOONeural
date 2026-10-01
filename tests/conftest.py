"""CPU-only tests that cannot accidentally resolve the former source packages."""

import importlib.abc
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


class RejectFormerDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"dsge_hmc", "common_utils"}:
            raise ImportError("MOONeural must not import its former source dependency: " + fullname)
        return None


sys.meta_path.insert(0, RejectFormerDependencies())
