"""Compatibility import; new users should import sgl_analytic.pipeline."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("sgl_analytic.pipeline")
