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
- the optional positive clustering closure (`closure="vp"`), and speed-ups of
  the direct cell sum and the kappa transform (identical results up to
  floating-point roundoff);
- the dark-energy domain is wider (see [docs/details.md](docs/details.md)).

Combined effect on the Planck reference PDFs (new vs 0.1.0), z_s = 0.5 / 1 / 3 / 10:
sigma_kappa −2.3 / −2.0 / −1.6 / −1.4%, clipped variance of ln mu (|ln mu| <= 1)
−3.9 / −3.9 / −3.3 / −2.8%, peak +5.3 / +3.8 / +2.4 / +1.9%, and
max|dP| = 11 / 5.4 / 2.9 / 2.3% of the peak. `edge="rvir"`, `radiation=None`
and `growth_mode="auto"` restore the old model choices but not the old
numerical resolution.
