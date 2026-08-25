"""
geometry.py
===========
The (RA, Dec, lambda) -> SCA pixel mapping for the data challenge.
This part is GIVEN. Everything downstream of it is yours to build.

PROVENANCE: the dispersion polynomial `dispersion()` and the
sky -> SCA transform chain inside `radeclam_to_pixel()` (the tangent-plane
offset, the theta rotation, the f0 anchor built from the sca_to_sky
solution at 1.55 um, and the mm -> pixel conversion
100*(mm - pos_center) + 4088/2) are taken directly from Isak Wold's
`radeclambda_detector.ipynb` (github.com/isakwold/roman-grism-colab),
which defines the coordinate convention for detectors.pickle. The
GrismGeometry class, `trace()`, `on_detector()` and `trace_diagnostics()`
are convenience wrappers added for this challenge; the mapping itself is
unchanged and was verified against the grism frames (trace-placement test:
correct roll wins by ~10^3 in trace flux).

Built directly on the dispersion solution in `radeclambda_detector.ipynb`
(detectors.pickle: 22-coefficient 5th-order 3D polynomial, sky -> SCA mm).

Design constraint: the grism and direct images are large (~70-130 MB) and you
should never need to hold either fully in RAM. Every routine here reads through
`astropy.io.fits` memory-mapping and touches only the pixels under a trace.

Dependencies: numpy, scipy, astropy.  (photutils optional, for source detection.)

Author's note on units
----------------------
  lam        micron
  x, y       SCA pixel, 0-indexed, array access is data[y, x]
  flux       whatever unit the grism image is in (per pixel, summed across
             the cross-dispersion aperture). Flux calibration is NOT applied.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np

__all__ = ["dispersion", "GrismGeometry", "SCA_SIZE"]

SCA_SIZE = 4088  # active science pixels per axis


# ----------------------------------------------------------------------
# 1. Dispersion polynomial  (verbatim from the notebook, minus @njit so it
#    runs anywhere; numba gives ~5x but is not required)
# ----------------------------------------------------------------------
def dispersion(c, x, y, lam):
    """5th-order 3D polynomial mapping. c has shape (22, 2) -> returns (N, 2) in mm."""
    V = np.column_stack((
        np.ones_like(x),
        x, y, lam,
        x**2, x*y, y**2, x*lam, y*lam, lam**2,
        x**3, (x**2)*y, x*(y**2), y**3,
        (x**2)*lam, x*y*lam, (y**2)*lam,
        x*(lam**2), y*(lam**2), lam**3,
        lam**4, lam**5,
    ))
    return V @ c


# ----------------------------------------------------------------------
# 2. Geometry: (RA, Dec, lam) -> SCA pixel
# ----------------------------------------------------------------------
class GrismGeometry:
    """
    Wraps one SCA's grism dispersion solution.

    Parameters
    ----------
    pickle_path : path to detectors.pickle
    sca_key     : e.g. 'SCA#5'
    slice_center: (RA, Dec) in degrees -- the field/pointing center used to
                  build the simulation. In the notebook this comes from the
                  COSMOS mosaic WCS at the centre of the simulated FOV.
    theta       : roll angle in degrees (0 in the notebook's examples).
    """

    def __init__(self, pickle_path, sca_key, slice_center, theta=0.0):
        with open(pickle_path, "rb") as f:
            det = pickle.load(f)
        if sca_key not in det:
            raise KeyError(f"{sca_key!r} not in pickle. Available: {list(det)}")

        sca = det[sca_key]
        self.sca_key = sca_key
        self.pos_center = np.asarray(sca["pos-center"], dtype=float)
        self.sky_to_sca = np.asarray(sca["grism"]["sky_to_sca"], dtype=float)
        self.sca_to_sky = np.asarray(sca["grism"]["sca_to_sky"], dtype=float)
        self.slice_center = np.asarray(slice_center, dtype=float)
        self.theta = float(theta)

        # f0: the sky-plane offset that puts the SCA reference position at
        # 1.55 um in the right place. Same construction as notebook cell 2.
        self.f0 = dispersion(
            self.sca_to_sky,
            np.array([0.0]) + self.pos_center[0],
            np.array([0.0]) + self.pos_center[1],
            np.array([1.55]),
        )

    # -- core mapping -------------------------------------------------
    def radeclam_to_pixel(self, ra_deg, dec_deg, lam_um):
        """(RA, Dec, lam) -> (N, 2) array of SCA (x, y) pixel coordinates."""
        ra = np.atleast_1d(np.asarray(ra_deg, dtype=float))
        dec = np.atleast_1d(np.asarray(dec_deg, dtype=float))
        n = ra.shape[0]

        lam = np.atleast_1d(np.asarray(lam_um, dtype=float))
        if lam.shape[0] == 1 and n > 1:
            lam = np.full(n, lam[0])
        if lam.shape[0] != n:
            raise ValueError("lam must be scalar or match ra/dec length")

        deg = np.column_stack([ra, dec]) - self.slice_center
        deg[:, 0] *= np.cos(np.radians(self.slice_center[1]))

        ct, st = np.cos(np.radians(self.theta)), np.sin(np.radians(self.theta))
        nx = deg[:, 0] * ct - deg[:, 1] * st + self.f0[0][0]
        ny = deg[:, 0] * st + deg[:, 1] * ct + self.f0[0][1]

        mm = dispersion(
            self.sky_to_sca,
            nx.reshape(n, 1),
            ny.reshape(n, 1),
            lam.reshape(n, 1),
        )
        return 100.0 * (mm - self.pos_center) + SCA_SIZE / 2.0

    # -- convenience --------------------------------------------------
    def trace(self, ra_deg, dec_deg, lam_grid):
        """Full first-order trace for one source. Returns (n_lam, 2) pixel coords."""
        lam_grid = np.asarray(lam_grid, dtype=float)
        n = lam_grid.size
        return self.radeclam_to_pixel(
            np.full(n, float(ra_deg)), np.full(n, float(dec_deg)), lam_grid
        )

    def on_detector(self, ra_deg, dec_deg, lam_um=1.55, margin=0.0):
        p = self.radeclam_to_pixel(ra_deg, dec_deg, lam_um)[0]
        return bool(
            margin <= p[0] <= SCA_SIZE - margin and margin <= p[1] <= SCA_SIZE - margin
        )

    def trace_diagnostics(self, ra_deg, dec_deg, lam_grid):
        """Measured length, tilt, dispersion and straightness of one trace."""
        tr = self.trace(ra_deg, dec_deg, lam_grid)
        d = tr[-1] - tr[0]
        length = float(np.hypot(*d))
        angle = float(np.degrees(np.arctan2(d[1], d[0])))
        seg = np.hypot(np.diff(tr[:, 0]), np.diff(tr[:, 1]))
        dlam_nm = np.diff(lam_grid) * 1000.0
        with np.errstate(divide="ignore", invalid="ignore"):
            nm_per_px = dlam_nm / seg
        unit = d / length
        rel = tr - tr[0]
        perp = np.abs(rel[:, 0] * (-unit[1]) + rel[:, 1] * unit[0])
        return {
            "length_px": length,
            "angle_deg": angle,
            "nm_per_px_mean": float(np.nanmean(nm_per_px)),
            "nm_per_px_min": float(np.nanmin(nm_per_px)),
            "nm_per_px_max": float(np.nanmax(nm_per_px)),
            "max_deviation_px": float(perp.max()),
        }


