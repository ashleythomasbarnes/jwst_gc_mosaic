from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from mosaic.gc_mosaic import build_parser, discover_files, make_mosaic


def write_i2d(path: Path, value: float, ra: float, alt_value: float | None = None) -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [5.0, 5.0]
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
    wcs.wcs.crval = [ra, -29.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["BUNIT"] = "MJy/sr"
    data = np.full((9, 9), value, dtype=np.float32)
    data[0, 0] = np.nan
    hdus = [fits.PrimaryHDU(), fits.ImageHDU(data, header=header, name="SCI")]
    if alt_value is not None:
        hdus.append(
            fits.ImageHDU(np.full((9, 9), alt_value, dtype=np.float32), header=header, name="ALT")
        )
    fits.HDUList(hdus).writeto(path)


def test_required_cli_arguments():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_recursive_filter_discovery(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    write_i2d(nested / "a_miri_f770w_i2d.fits", 1, 266.4)
    write_i2d(nested / "b_miri_f2100w_i2d.fits", 1, 266.4)
    assert [path.name for path in discover_files(tmp_path, "F770W")] == [
        "a_miri_f770w_i2d.fits"
    ]


def test_alternate_extension_and_galactic_nan_output(tmp_path):
    write_i2d(tmp_path / "a_miri_f770w_i2d.fits", 1, 266.4, alt_value=7)
    output = make_mosaic(
        tmp_path, "f770w", extension="ALT", background_match=False
    )
    with fits.open(output) as hdul:
        data = hdul[0].data
        assert hdul[0].header["CTYPE1"].startswith("GLON")
        assert hdul[0].header["CTYPE2"].startswith("GLAT")
        assert hdul[0].header["BUNIT"] == "MJy/sr"
        assert data.dtype == np.dtype(">f4")
        assert np.nanmedian(data) == pytest.approx(7, abs=1e-4)
        assert np.isnan(data).any()


def test_background_matching_and_disconnected_tile(tmp_path):
    # The first two images overlap; the third is an isolated sky component.
    write_i2d(tmp_path / "a_miri_f770w_i2d.fits", 1, 266.4000)
    write_i2d(tmp_path / "b_miri_f770w_i2d.fits", 3, 266.4010)
    write_i2d(tmp_path / "c_miri_f770w_i2d.fits", 9, 266.4300)
    output = make_mosaic(tmp_path, "f770w", background_match=True)
    with fits.open(output) as hdul:
        data = hdul[0].data
        assert hdul[0].header["BGMATCH"]
        assert hdul[0].header["NBGCOMP"] == 2
        finite = data[np.isfinite(data)]
        assert np.any(np.isclose(finite, 2, atol=0.05))
        assert np.any(np.isclose(finite, 9, atol=0.05))


def test_no_background_matching_preserves_tile_levels(tmp_path):
    write_i2d(tmp_path / "a_miri_f770w_i2d.fits", 1, 266.4000)
    write_i2d(tmp_path / "b_miri_f770w_i2d.fits", 3, 266.4010)
    output = make_mosaic(tmp_path, "f770w", background_match=False)
    with fits.open(output) as hdul:
        finite = hdul[0].data[np.isfinite(hdul[0].data)]
        assert not hdul[0].header["BGMATCH"]
        assert np.any(np.isclose(finite, 1, atol=0.05))
        assert np.any(np.isclose(finite, 3, atol=0.05))


def test_existing_output_is_protected(tmp_path):
    write_i2d(tmp_path / "a_miri_f770w_i2d.fits", 1, 266.4)
    make_mosaic(tmp_path, "f770w", background_match=False)
    with pytest.raises(FileExistsError):
        make_mosaic(tmp_path, "f770w", background_match=False)


def test_inconsistent_units_are_rejected(tmp_path):
    write_i2d(tmp_path / "a_miri_f770w_i2d.fits", 1, 266.4000)
    second = tmp_path / "b_miri_f770w_i2d.fits"
    write_i2d(second, 1, 266.4010)
    with fits.open(second, mode="update") as hdul:
        del hdul["SCI"].header["BUNIT"]
    with pytest.raises(ValueError, match="inconsistent BUNIT"):
        make_mosaic(tmp_path, "f770w", background_match=False)
