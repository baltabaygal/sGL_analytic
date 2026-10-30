# Changelog

## 0.2.0 (2026-10-21)

The defaults follow the research repository's chain as of 2026-10-21:

- halos and subhalo clumps are truncated at r200c (was r_vir);
- the background includes radiation, Omega_r = Om/(1 + zeq) (was none);
- linear growth from the ODE for every model, also in bias, M*, delta_c(z) and
  subhalo formation redshifts (was `"auto"`);
- a 101 x 101 coarse grid for the clustering term (was 51 x 51), finer
  inversion grids (72/36 points per sigma in kappa/gamma, 2400 integration
  nodes), a kink-split angle rule for filaments, the inferred number of
  ellipticity angles in the subhalo sector, block-wise inversion of very large
  grids, and a finer shear-axis grid for narrow PDFs;
- the dark-energy domain is wider (see [docs/details.md](docs/details.md)).


