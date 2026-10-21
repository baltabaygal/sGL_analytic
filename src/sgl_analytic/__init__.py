"""Sampling-free stochastic weak-lensing magnification PDFs."""
from .api import generate_pdf, generate_pdf_lnmu

__version__ = "0.2.0"
__all__ = ["generate_pdf", "generate_pdf_lnmu", "__version__"]
