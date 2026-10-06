"""Compatibility import; new users should import sgl_analytic.filaments."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("sgl_analytic.filaments")
