import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from mosaic.program_10678_pipeline import (
    Product,
    build_parser,
    query_archive,
    run_pipeline,
    validate_product_identities,
)


def write_i2d(path: Path, value: float, ra: float) -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [5.0, 5.0]
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
    wcs.wcs.crval = [ra, -29.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["BUNIT"] = "MJy/sr"
    data = np.full((9, 9), value, dtype=np.float32)
    data[0, 0] = np.nan
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data, header=header, name="SCI")]
    ).writeto(path)


class MockObservations:
    def __init__(self, products: Table):
        self.products = products
        self.download_calls = 0

    def query_criteria(self, **criteria):
        assert criteria == {
            "proposal_id": "10678",
            "calib_level": 3,
            "dataproduct_type": "image",
        }
        return Table(
            rows=[("F770W",), ("F212N",), ("F480M",), ("F2100W",)],
            names=("filters",),
        )

    def get_product_list(self, observations):
        assert len(observations) == 3
        return self.products

    def filter_products(self, products, **filters):
        assert filters["mrp_only"]
        mask = [
            row["productGroupDescription"] == "Minimum Recommended Products"
            and row["productSubGroupDescription"] == "I2D"
            and row["productType"] == "SCIENCE"
            and str(row["productFilename"]).endswith(".fits")
            for row in products
        ]
        return products[mask]

    def download_products(self, products, download_dir, flat, cache):
        assert flat and cache
        self.download_calls += 1
        destination = Path(download_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for row in products:
            target = destination / str(row["productFilename"])
            if not target.exists():
                shutil.copyfile(str(row["source"]), target)


def product_table(rows: list[tuple[Path, str, int, str]]) -> Table:
    values = []
    for source, filename, level, group in rows:
        values.append(
            (
                filename.removesuffix("_i2d.fits"),
                filename,
                f"mast:JWST/product/{filename}",
                "SCIENCE",
                group,
                "I2D",
                level,
                source.stat().st_size,
                str(source),
            )
        )
    return Table(
        rows=values,
        names=(
            "obs_id",
            "productFilename",
            "dataURI",
            "productType",
            "productGroupDescription",
            "productSubGroupDescription",
            "calib_level",
            "size",
            "source",
        ),
    )


def association_row(source: Path, filename: str) -> tuple[Path, str, int, str]:
    return source, filename, 3, "Minimum Recommended Products"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_pipeline_cli_defaults_and_single_filter():
    args = build_parser().parse_args([])
    assert args.filter is None
    assert not args.fresh
    args = build_parser().parse_args(["--filter", "f770w", "--fresh"])
    assert args.filter == ["f770w"]
    assert args.fresh


def test_query_selects_only_level3_association_i2d(tmp_path):
    source = tmp_path / "source.fits"
    write_i2d(source, 1, 266.4)
    rows = [
        association_row(source, "jw10678-o001_t001_miri_f770w_i2d.fits"),
        (
            source,
            "jw10678001001_02101_00001_mirimage_i2d.fits",
            2,
            "Minimum Recommended Products",
        ),
        (
            source,
            "jw10678-o002_t002_miri_f770w_i2d.fits",
            3,
            "--",
        ),
    ]
    products, selected = query_archive(MockObservations(product_table(rows)))
    assert [product.filename for product in products] == [
        "jw10678-o001_t001_miri_f770w_i2d.fits"
    ]
    assert len(selected) == 1


def test_product_identity_collisions_are_rejected():
    products = [
        Product("f770w", "a", "same.fits", "mast:a", 1),
        Product("f770w", "b", "same.fits", "mast:b", 1),
    ]
    with pytest.raises(RuntimeError, match="filename collision"):
        validate_product_identities(products)


def test_initial_no_change_incremental_and_fresh_workflow(tmp_path):
    first_source = tmp_path / "first.fits"
    second_source = tmp_path / "second.fits"
    write_i2d(first_source, 1, 266.4000)
    write_i2d(second_source, 3, 266.4010)
    first_name = "jw10678-o001_t001_miri_f770w_i2d.fits"
    second_name = "jw10678-o002_t002_miri_f770w_i2d.fits"
    api = MockObservations(product_table([association_row(first_source, first_name)]))
    data_dir = tmp_path / "data"

    first = run_pipeline(data_dir, filters=["f770w"], observations_api=api)[0]
    output = data_dir / "gc_mosaic_f770w.fits.gz"
    manifest = data_dir / "logs" / "f770w_manifest.ecsv"
    run_log = data_dir / "logs" / "f770w_runs.jsonl"
    assert first["action"] == "initial_build"
    assert output.exists() and manifest.exists()
    assert not list((data_dir / "work").rglob("*_i2d.fits"))
    with fits.open(output) as hdul:
        assert hdul[0].header["NINPUT"] == 1
        assert not hdul[0].header["INCRMNT"]
        assert hdul["COVERAGE"].data.shape == hdul[0].data.shape

    unchanged = run_pipeline(
        data_dir, filters=["f770w"], observations_api=api
    )[0]
    assert unchanged["action"] == "no_change"
    assert api.download_calls == 1
    assert len(read_jsonl(run_log)) == 2

    api.products = product_table(
        [
            association_row(first_source, first_name),
            association_row(second_source, second_name),
        ]
    )
    updated = run_pipeline(data_dir, filters=["f770w"], observations_api=api)[0]
    assert updated["action"] == "incremental_update"
    assert updated["downloaded_files"] == [second_name]
    assert not list((data_dir / "work").rglob("*_i2d.fits"))
    with fits.open(output) as hdul:
        assert hdul[0].header["NINPUT"] == 2
        assert hdul[0].header["INCRMNT"]
        assert np.nanmax(hdul["COVERAGE"].data) > 1
        assert np.nanmedian(hdul[0].data) == pytest.approx(1, abs=0.1)
    assert len(Table.read(manifest, format="ascii.ecsv")) == 2
    assert len(read_jsonl(run_log)) == 3

    rebuilt = run_pipeline(
        data_dir, filters=["f770w"], fresh=True, observations_api=api
    )[0]
    assert rebuilt["action"] == "fresh_rebuild"
    history = read_jsonl(run_log)
    assert len(history) == 1
    assert history[0]["mode"] == "fresh"
    assert sorted(rebuilt["downloaded_files"]) == sorted([first_name, second_name])


def test_retracted_product_is_logged_but_retained_incrementally(tmp_path):
    first_source = tmp_path / "first.fits"
    second_source = tmp_path / "second.fits"
    write_i2d(first_source, 1, 266.4000)
    write_i2d(second_source, 2, 266.4010)
    first_name = "jw10678-o001_t001_miri_f770w_i2d.fits"
    second_name = "jw10678-o002_t002_miri_f770w_i2d.fits"
    api = MockObservations(
        product_table(
            [
                association_row(first_source, first_name),
                association_row(second_source, second_name),
            ]
        )
    )
    data_dir = tmp_path / "data"
    run_pipeline(data_dir, filters=["f770w"], observations_api=api)

    api.products = product_table([association_row(first_source, first_name)])
    result = run_pipeline(data_dir, filters=["f770w"], observations_api=api)[0]
    assert result["action"] == "no_change"
    assert result["archive_missing_previous_files"] == [second_name]
    with fits.open(data_dir / "gc_mosaic_f770w.fits.gz") as hdul:
        assert hdul[0].header["NINPUT"] == 2


def test_failed_download_preserves_staged_file_and_logs_failure(tmp_path):
    source = tmp_path / "source.fits"
    write_i2d(source, 1, 266.4)
    filename = "jw10678-o001_t001_miri_f770w_i2d.fits"
    api = MockObservations(product_table([association_row(source, filename)]))

    def fail_after_copy(products, download_dir, flat, cache):
        destination = Path(download_dir)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination / filename)
        raise RuntimeError("simulated interruption")

    api.download_products = fail_after_copy
    data_dir = tmp_path / "data"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_pipeline(data_dir, filters=["f770w"], observations_api=api)
    assert (data_dir / "work" / "f770w" / "downloads" / filename).exists()
    assert not (data_dir / "gc_mosaic_f770w.fits.gz").exists()
    record = read_jsonl(data_dir / "logs" / "f770w_runs.jsonl")[-1]
    assert record["status"] == "failed"
