"""Run after installing sgl-analytic; writes a PDF with model metadata."""
import json
from pathlib import Path

from sgl_analytic import generate_pdf_lnmu


def main():
    lnmu, pdf, info = generate_pdf_lnmu(
        z_s=1.0, h=0.674, Om=0.315, sigma8=0.811,
        Ob=0.0493, ns=0.965, zeq=3402.0, return_info=True,
    )
    dest = Path("pdf_zs1.json")
    dest.write_text(json.dumps(dict(lnmu=lnmu.tolist(), pdf=pdf.tolist(), **info),
                               indent=2, allow_nan=False))
    print(f"Wrote {dest}; diagnostics: {info['diagnostics']}")


if __name__ == "__main__":
    main()
