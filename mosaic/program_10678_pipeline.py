#!/usr/bin/env python3
"""Incrementally maintain compressed Level-3 mosaics for JWST Program 10678."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tempfile
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel
from astroquery.mast import Observations
from reproject import reproject_interp

try:
    from .gc_mosaic import load_images, make_mosaic_from_paths
except ImportError:  # Direct execution: python mosaic/program_10678_pipeline.py
    from gc_mosaic import load_images, make_mosaic_from_paths


PROGRAM_ID = "10678"
SUPPORTED_FILTERS = ("f770w", "f212n", "f480m")
COPY_CHUNK = 16 * 1024 * 1024
ARRAY_ROWS = 256
BACKGROUND_SAMPLE_LIMIT = 1_000_000


@dataclass(frozen=True)
class Product:
    filter_name: str
    obs_id: str
    filename: str
    data_uri: str
    size: int

    @property
    def identity(self) -> tuple[str, str]:
        return self.data_uri, self.filename


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def filter_from_filename(filename: str) -> str | None:
    """Return the supported filter encoded in a Level-3 JWST filename."""
    lowered = filename.lower()
    for filter_name in SUPPORTED_FILTERS:
        if f"-{filter_name}_" in lowered or f"_{filter_name}_" in lowered:
            return filter_name
    return None


def _plain(value: object, default: str = "") -> str:
    """Convert an Astropy scalar, including masked values, to plain text."""
    if np.ma.is_masked(value):
        return default
    return str(value)


def query_archive(observations_api=Observations) -> tuple[list[Product], Table]:
    """Query only the association-level Level-3 I2D products we mosaic."""
    observations = observations_api.query_criteria(
        proposal_id=PROGRAM_ID,
        calib_level=3,
        dataproduct_type="image",
    )
    wanted_observations = observations[
        [
            _plain(value).lower() in SUPPORTED_FILTERS
            for value in observations["filters"]
        ]
    ]
    if len(wanted_observations) == 0:
        raise RuntimeError(f"MAST returned no supported Level-3 observations for {PROGRAM_ID}")

    products = observations_api.get_product_list(wanted_observations)
    selected = observations_api.filter_products(
        products,
        mrp_only=True,
        productSubGroupDescription="I2D",
        productType="SCIENCE",
        extension="fits",
    )
    selected = selected[
        [
            int(level) == 3 and filter_from_filename(_plain(filename)) is not None
            for level, filename in zip(
                selected["calib_level"], selected["productFilename"], strict=True
            )
        ]
    ]
    if len(selected) == 0:
        raise RuntimeError(f"MAST returned no Level-3 association I2D products for {PROGRAM_ID}")

    result = []
    for row in selected:
        filename = _plain(row["productFilename"])
        result.append(
            Product(
                filter_name=filter_from_filename(filename) or "",
                obs_id=_plain(row["obs_id"]),
                filename=filename,
                data_uri=_plain(row["dataURI"]),
                size=int(row["size"]),
            )
        )
    result.sort(key=lambda product: (product.filter_name, product.filename))
    validate_product_identities(result)
    return result, selected


def validate_product_identities(products: list[Product]) -> None:
    """Reject ambiguous MAST filename or URI collisions."""
    by_name: dict[str, str] = {}
    by_uri: dict[str, str] = {}
    for product in products:
        old_uri = by_name.setdefault(product.filename, product.data_uri)
        if old_uri != product.data_uri:
            raise RuntimeError(
                f"MAST filename collision for {product.filename}; run a fresh rebuild after inspection"
            )
        old_name = by_uri.setdefault(product.data_uri, product.filename)
        if old_name != product.filename:
            raise RuntimeError(
                f"MAST dataURI collision for {product.data_uri}; run a fresh rebuild after inspection"
            )


def read_manifest(path: Path) -> tuple[list[Product], dict[str, object]]:
    """Read the latest full product manifest for one mosaic."""
    table = Table.read(path, format="ascii.ecsv")
    products = [
        Product(
            filter_name=_plain(row["filter"]),
            obs_id=_plain(row["obs_id"]),
            filename=_plain(row["filename"]),
            data_uri=_plain(row["data_uri"]),
            size=int(row["size"]),
        )
        for row in table
    ]
    validate_product_identities(products)
    return products, dict(table.meta)


def write_manifest(
    path: Path,
    products: list[Product],
    run_id: str,
    completed_utc: str,
) -> None:
    """Write the complete provenance manifest for a mosaic."""
    table = Table(
        rows=[
            (
                product.filter_name,
                product.obs_id,
                product.filename,
                product.data_uri,
                product.size,
            )
            for product in sorted(products, key=lambda item: item.filename)
        ],
        names=("filter", "obs_id", "filename", "data_uri", "size"),
    )
    table.meta.update(
        {
            "program_id": PROGRAM_ID,
            "run_id": run_id,
            "completed_utc": completed_utc,
            "n_products": len(products),
        }
    )
    table.write(path, format="ascii.ecsv", overwrite=True)


def prepare_run_log(
    path: Path,
    record: dict[str, object],
    reset: bool,
    previous_path: Path | None = None,
) -> None:
    """Stage an atomically replaceable JSON-lines run log."""
    source = path if previous_path is None else previous_path
    old_text = "" if reset or not source.exists() else source.read_text()
    if old_text and not old_text.endswith("\n"):
        old_text += "\n"
    path.write_text(old_text + json.dumps(record, sort_keys=True) + "\n")


def append_failure_log(path: Path, record: dict[str, object]) -> None:
    """Append a failed attempt without disturbing successful history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def matching_rows(table: Table, products: list[Product]) -> Table:
    """Select product-table rows matching the requested identities."""
    identities = {product.identity for product in products}
    mask = [
        (_plain(row["dataURI"]), _plain(row["productFilename"])) in identities
        for row in table
    ]
    selected = table[mask]
    if len(selected) != len(identities):
        raise RuntimeError("Could not map every requested product back to the MAST table")
    return selected


def download_products(
    product_rows: Table,
    products: list[Product],
    destination: Path,
    observations_api=Observations,
) -> list[Path]:
    """Download products flat into a resumable staging directory."""
    destination.mkdir(parents=True, exist_ok=True)
    for product in products:
        existing = destination / product.filename
        if existing.exists() and existing.stat().st_size != product.size:
            existing.unlink()
    observations_api.download_products(
        product_rows,
        download_dir=str(destination),
        flat=True,
        cache=True,
    )
    paths = []
    for product in products:
        path = destination / product.filename
        if not path.is_file() or path.stat().st_size != product.size:
            raise RuntimeError(
                f"MAST download size mismatch for {path}: "
                f"expected {product.size}, found {path.stat().st_size if path.exists() else 0}"
            )
        paths.append(path)
    return paths


def gzip_fits(source: Path, destination: Path) -> None:
    """Create a lossless gzip-compressed FITS file without loading it in RAM."""
    with source.open("rb") as input_handle, gzip.open(
        destination, "wb", compresslevel=6
    ) as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=COPY_CHUNK)


def validate_compressed_mosaic(path: Path, expected_inputs: int) -> None:
    """Check gzip integrity and the required FITS mosaic structure."""
    with gzip.open(path, "rb") as handle:
        while handle.read(COPY_CHUNK):
            pass
    with fits.open(path, memmap=False) as hdul:
        hdul.verify("exception")
        if "COVERAGE" not in hdul:
            raise RuntimeError(f"Compressed mosaic has no COVERAGE extension: {path}")
        if hdul[0].shape != hdul["COVERAGE"].shape:
            raise RuntimeError(f"SCI and COVERAGE shapes differ in {path}")
        if int(hdul[0].header.get("NINPUT", -1)) != expected_inputs:
            raise RuntimeError(f"Unexpected NINPUT in {path}")


def decompress_fits(source: Path, destination: Path) -> None:
    """Expand a gzip FITS file to a disk-backed working copy."""
    with gzip.open(source, "rb") as input_handle, destination.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=COPY_CHUNK)


def image_unclipped_bounds(image, output_wcs: WCS) -> tuple[int, int, int, int]:
    """Return sampled edge bounds on an existing output pixel grid."""
    ny, nx = image.shape
    samples = max(11, min(64, max(nx, ny)))
    x = np.linspace(0, nx - 1, samples)
    y = np.linspace(0, ny - 1, samples)
    edge_x = np.concatenate((x, x, np.zeros_like(y), np.full_like(y, nx - 1)))
    edge_y = np.concatenate((np.zeros_like(x), np.full_like(x, ny - 1), y, y))
    out_x, out_y = pixel_to_pixel(image.wcs, output_wcs, edge_x, edge_y)
    valid = np.isfinite(out_x) & np.isfinite(out_y)
    if not np.any(valid):
        raise ValueError(f"{image.path.name}: WCS does not map into the output projection")
    return (
        int(np.floor(np.min(out_y[valid]))) - 2,
        int(np.ceil(np.max(out_y[valid]))) + 3,
        int(np.floor(np.min(out_x[valid]))) - 2,
        int(np.ceil(np.max(out_x[valid]))) + 3,
    )


def _copy_old_state(
    science: np.ndarray,
    old_coverage: np.ndarray,
    weighted_sum: np.memmap,
    coverage: np.memmap,
    y_offset: int,
    x_offset: int,
) -> None:
    """Copy an old mean mosaic into new disk-backed sum and weight arrays."""
    ny, nx = science.shape
    for start in range(0, ny, ARRAY_ROWS):
        stop = min(ny, start + ARRAY_ROWS)
        destination = np.s_[
            start + y_offset : stop + y_offset,
            x_offset : x_offset + nx,
        ]
        old_weight = np.asarray(old_coverage[start:stop], dtype=np.float32)
        old_science = np.asarray(science[start:stop], dtype=np.float32)
        coverage[destination] = old_weight
        weighted_sum[destination] = np.where(
            old_weight > 0, old_science * old_weight, 0
        )


def _background_offset(
    weighted_sum: np.ndarray,
    coverage: np.ndarray,
    tile: np.ndarray,
    footprint: np.ndarray,
) -> tuple[float, int]:
    """Estimate a robust additive offset from a bounded deterministic sample."""
    stride = max(
        1,
        int(np.ceil(np.sqrt(tile.size / BACKGROUND_SAMPLE_LIMIT))),
    )
    count = 0
    differences = []
    for start in range(0, tile.shape[0], ARRAY_ROWS):
        stop = min(tile.shape[0], start + ARRAY_ROWS)
        local_coverage = coverage[start:stop]
        local_footprint = footprint[start:stop]
        local_tile = tile[start:stop]
        local_sum = weighted_sum[start:stop]
        valid = (
            (local_coverage > 0)
            & (local_footprint > 0)
            & np.isfinite(local_tile)
            & np.isfinite(local_sum)
        )
        count += int(np.count_nonzero(valid))
        sampled_valid = valid[::stride, ::stride]
        if np.any(sampled_valid):
            sampled_coverage = local_coverage[::stride, ::stride][sampled_valid]
            old_mean = local_sum[::stride, ::stride][sampled_valid] / sampled_coverage
            differences.append(
                old_mean - local_tile[::stride, ::stride][sampled_valid]
            )
    if not differences:
        return 0.0, count
    return float(np.median(np.concatenate(differences))), count


def _add_tile(
    weighted_sum: np.memmap,
    coverage: np.memmap,
    tile: np.ndarray,
    footprint: np.ndarray,
    cutout: tuple[slice, slice],
    offset: float,
) -> None:
    """Add one reprojected tile to the accumulated weighted mean."""
    height = tile.shape[0]
    for start in range(0, height, ARRAY_ROWS):
        stop = min(height, start + ARRAY_ROWS)
        output_rows = slice(cutout[0].start + start, cutout[0].start + stop)
        output = np.s_[output_rows, cutout[1]]
        values = tile[start:stop]
        weights = footprint[start:stop]
        valid = (weights > 0) & np.isfinite(values)
        if not np.any(valid):
            continue
        sum_chunk = weighted_sum[output]
        coverage_chunk = coverage[output]
        sum_chunk[valid] += (values[valid] + offset) * weights[valid]
        coverage_chunk[valid] += weights[valid]


def write_incremental_mosaic(
    existing_gzip: Path,
    new_paths: list[Path],
    output_path: Path,
    filter_name: str,
    total_inputs: int,
    run_id: str,
    run_date: str,
    background_match: bool,
) -> list[dict[str, object]]:
    """Incrementally add new tiles while preserving the established pixel grid."""
    images, new_unit = load_images(new_paths, "SCI")
    offsets: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="program10678_incremental_") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        old_path = temp_dir / "old.fits"
        decompress_fits(existing_gzip, old_path)

        with fits.open(old_path, memmap=True) as old_hdul:
            if "COVERAGE" not in old_hdul:
                raise RuntimeError(
                    f"{existing_gzip} has no COVERAGE extension; rerun with --fresh --filter {filter_name}"
                )
            science = old_hdul[0].data
            old_coverage = old_hdul["COVERAGE"].data
            if science is None or science.ndim != 2 or old_coverage.shape != science.shape:
                raise RuntimeError(f"Invalid incremental mosaic structure: {existing_gzip}")
            old_wcs = WCS(old_hdul[0].header).celestial
            old_unit = old_hdul[0].header.get("BUNIT")
            if old_unit != new_unit:
                raise RuntimeError(
                    f"New input BUNIT {new_unit!r} differs from mosaic BUNIT {old_unit!r}"
                )

            raw_bounds = [image_unclipped_bounds(image, old_wcs) for image in images]
            old_ny, old_nx = science.shape
            ymin = min([0] + [item[0] for item in raw_bounds])
            ymax = max([old_ny] + [item[1] for item in raw_bounds])
            xmin = min([0] + [item[2] for item in raw_bounds])
            xmax = max([old_nx] + [item[3] for item in raw_bounds])
            shape_out = (ymax - ymin, xmax - xmin)
            y_offset, x_offset = -ymin, -xmin
            output_wcs = old_wcs.deepcopy()
            output_wcs.wcs.crpix += [x_offset, y_offset]

            weighted_sum = np.memmap(
                temp_dir / "weighted_sum.float32",
                mode="w+",
                dtype=np.float32,
                shape=shape_out,
            )
            coverage = np.memmap(
                temp_dir / "coverage.float32",
                mode="w+",
                dtype=np.float32,
                shape=shape_out,
            )
            weighted_sum[:] = 0
            coverage[:] = 0
            _copy_old_state(
                science,
                old_coverage,
                weighted_sum,
                coverage,
                y_offset,
                x_offset,
            )

            for number, (image, bounds) in enumerate(zip(images, raw_bounds, strict=True)):
                y0, y1, x0, x1 = bounds
                cutout = (
                    slice(y0 + y_offset, y1 + y_offset),
                    slice(x0 + x_offset, x1 + x_offset),
                )
                shape = (y1 - y0, x1 - x0)
                tile = np.memmap(
                    temp_dir / f"tile_{number}.float32",
                    mode="w+",
                    dtype=np.float32,
                    shape=shape,
                )
                footprint = np.memmap(
                    temp_dir / f"tile_footprint_{number}.float32",
                    mode="w+",
                    dtype=np.float32,
                    shape=shape,
                )
                reproject_interp(
                    str(image.path),
                    output_wcs[cutout],
                    shape_out=shape,
                    hdu_in=image.extension,
                    output_array=tile,
                    output_footprint=footprint,
                    return_footprint=True,
                    block_size="auto",
                )
                local_sum = weighted_sum[cutout]
                local_coverage = coverage[cutout]
                offset, overlap_pixels = (0.0, 0)
                if background_match:
                    offset, overlap_pixels = _background_offset(
                        local_sum, local_coverage, tile, footprint
                    )
                _add_tile(
                    weighted_sum,
                    coverage,
                    tile,
                    footprint,
                    cutout,
                    offset,
                )
                offsets.append(
                    {
                        "filename": image.path.name,
                        "offset": offset,
                        "overlap_pixels": overlap_pixels,
                    }
                )
                del tile, footprint
                (temp_dir / f"tile_{number}.float32").unlink()
                (temp_dir / f"tile_footprint_{number}.float32").unlink()

            for start in range(0, shape_out[0], ARRAY_ROWS):
                stop = min(shape_out[0], start + ARRAY_ROWS)
                sums = weighted_sum[start:stop]
                weights = coverage[start:stop]
                covered = weights > 0
                sums[covered] /= weights[covered]
                sums[~covered] = np.nan
            weighted_sum.flush()
            coverage.flush()

            header = old_hdul[0].header.copy()
            header.update(output_wcs.to_header(relax=True))
            header["PROGRAM"] = PROGRAM_ID
            header["FILTER"] = filter_name.upper()
            header["NINPUT"] = total_inputs
            header["RUNID"] = run_id
            header["RUNDATE"] = run_date
            header["INCRMNT"] = (True, "Existing weighted mosaic incrementally updated")
            header["COVHDU"] = ("COVERAGE", "Weight sum for incremental updates")
            header.add_history(
                "Incremental weighted mean; historical background offsets were not globally refit."
            )
            for item in offsets:
                header.add_history(
                    f"New input: {item['filename']}; additive offset={item['offset']:.8g}"
                )
            fits.HDUList(
                [
                    fits.PrimaryHDU(data=weighted_sum, header=header),
                    fits.ImageHDU(data=coverage, name="COVERAGE"),
                ]
            ).writeto(output_path, overwrite=True)
            del weighted_sum, coverage
    return offsets


def combine_products(previous: list[Product], new: list[Product]) -> list[Product]:
    """Return the full historical input set after an incremental update."""
    combined = {product.identity: product for product in previous}
    combined.update({product.identity: product for product in new})
    result = sorted(combined.values(), key=lambda product: product.filename)
    validate_product_identities(result)
    return result


def validate_existing_state(
    output_path: Path,
    manifest_path: Path,
    filter_name: str,
) -> tuple[list[Product], dict[str, object]]:
    """Require the mosaic and manifest to form a consistent pair."""
    if output_path.exists() != manifest_path.exists():
        raise RuntimeError(
            f"Mosaic/manifest state is incomplete; rerun with --fresh --filter {filter_name}"
        )
    if not output_path.exists():
        return [], {}
    products, metadata = read_manifest(manifest_path)
    if any(product.filter_name != filter_name for product in products):
        raise RuntimeError(
            f"Manifest contains the wrong filter; rerun with --fresh --filter {filter_name}"
        )
    if int(metadata.get("n_products", -1)) != len(products):
        raise RuntimeError(
            f"Manifest count is inconsistent; rerun with --fresh --filter {filter_name}"
        )
    return products, metadata


def process_filter(
    filter_name: str,
    archive_products: list[Product],
    archive_table: Table,
    data_dir: Path,
    fresh: bool,
    background_match: bool,
    observations_api=Observations,
    archive_queried_utc: str | None = None,
) -> dict[str, object]:
    """Update or rebuild one filter as a recoverable per-filter transaction."""
    started = utc_now()
    run_id = str(uuid.uuid4())
    logs_dir = data_dir / "logs"
    work_dir = data_dir / "work" / filter_name
    download_dir = work_dir / "downloads"
    output_path = data_dir / f"gc_mosaic_{filter_name}.fits.gz"
    manifest_path = logs_dir / f"{filter_name}_manifest.ecsv"
    log_path = logs_dir / f"{filter_name}_runs.jsonl"
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, object] = {
        "run_id": run_id,
        "program_id": PROGRAM_ID,
        "filter": filter_name,
        "mode": "fresh" if fresh else "update",
        "started_utc": started,
        "archive_queried_utc": archive_queried_utc or started,
        "archive_count": len(archive_products),
    }
    try:
        if fresh:
            previous, previous_meta = [], {}
        else:
            previous, previous_meta = validate_existing_state(
                output_path, manifest_path, filter_name
            )
        previous_identities = {product.identity for product in previous}
        archive_identities = {product.identity for product in archive_products}
        new = (
            archive_products
            if fresh or not previous
            else [
                product
                for product in archive_products
                if product.identity not in previous_identities
            ]
        )
        retracted = [
            product.filename
            for product in previous
            if product.identity not in archive_identities
        ]
        if retracted and not fresh:
            print(
                f"{filter_name.upper()}: warning: {len(retracted)} previously included "
                "product(s) are absent from the current MAST result and will be retained"
            )
        record.update(
            {
                "previous_count": len(previous),
                "new_files": [product.filename for product in new],
                "archive_missing_previous_files": retracted,
            }
        )

        if previous and not fresh and not new:
            validate_compressed_mosaic(output_path, len(previous))
            record.update(
                {
                    "status": "success",
                    "action": "no_change",
                    "completed_utc": utc_now(),
                    "output": str(output_path),
                    "included_files": [product.filename for product in previous],
                    "deleted_downloads": [],
                }
            )
            staged_log = work_dir / "runs.jsonl"
            prepare_run_log(
                staged_log, record, reset=False, previous_path=log_path
            )
            os.replace(staged_log, log_path)
            return record

        completed_products = (
            archive_products if fresh or not previous else combine_products(previous, new)
        )
        rows = matching_rows(archive_table, new)
        downloaded_paths = download_products(
            rows, new, download_dir, observations_api=observations_api
        )
        uncompressed = work_dir / f"gc_mosaic_{filter_name}.fits"
        compressed = work_dir / f"gc_mosaic_{filter_name}.fits.gz"
        uncompressed.unlink(missing_ok=True)
        compressed.unlink(missing_ok=True)

        if previous and not fresh:
            offsets = write_incremental_mosaic(
                output_path,
                downloaded_paths,
                uncompressed,
                filter_name,
                len(completed_products),
                run_id,
                started,
                background_match,
            )
            action = "incremental_update"
        else:
            make_mosaic_from_paths(
                downloaded_paths,
                uncompressed,
                filter_name,
                extension="SCI",
                background_match=background_match,
                overwrite=True,
                extra_header={
                    "PROGRAM": PROGRAM_ID,
                    "RUNID": run_id,
                    "RUNDATE": started,
                    "INCRMNT": (False, "Full mosaic recomputation"),
                    "COVHDU": ("COVERAGE", "Weight sum for incremental updates"),
                },
            )
            offsets = []
            action = "fresh_rebuild" if fresh else "initial_build"

        gzip_fits(uncompressed, compressed)
        validate_compressed_mosaic(compressed, len(completed_products))
        completed = utc_now()
        staged_manifest = work_dir / f"{filter_name}_manifest.ecsv"
        write_manifest(staged_manifest, completed_products, run_id, completed)
        staged_downloads = sorted(
            path.name for path in download_dir.iterdir() if path.is_file()
        )
        record.update(
            {
                "status": "success",
                "action": action,
                "completed_utc": completed,
                "output": str(output_path),
                "included_files": [
                    product.filename for product in completed_products
                ],
                "downloaded_files": [path.name for path in downloaded_paths],
                "deleted_downloads": staged_downloads,
                "background_offsets": offsets,
                "previous_manifest_run_id": previous_meta.get("run_id"),
            }
        )
        staged_log = work_dir / "runs.jsonl"
        prepare_run_log(
            staged_log, record, reset=fresh, previous_path=log_path
        )

        os.replace(compressed, output_path)
        os.replace(staged_manifest, manifest_path)
        os.replace(staged_log, log_path)
        if download_dir.exists():
            shutil.rmtree(download_dir)
        uncompressed.unlink(missing_ok=True)
        if work_dir.exists() and not any(work_dir.iterdir()):
            work_dir.rmdir()
        return record
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "completed_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        append_failure_log(log_path, record)
        raise


def run_pipeline(
    data_dir: str | Path,
    filters: list[str] | None = None,
    fresh: bool = False,
    background_match: bool = True,
    observations_api=Observations,
) -> list[dict[str, object]]:
    """Query MAST once, then update each requested filter."""
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    requested = list(dict.fromkeys(SUPPORTED_FILTERS if filters is None else filters))
    if not requested or any(item not in SUPPORTED_FILTERS for item in requested):
        raise ValueError(f"Filters must be selected from {', '.join(SUPPORTED_FILTERS)}")

    query_started = utc_now()
    try:
        products, table = query_archive(observations_api=observations_api)
    except Exception as exc:
        for filter_name in requested:
            append_failure_log(
                data_dir / "logs" / f"{filter_name}_runs.jsonl",
                {
                    "run_id": str(uuid.uuid4()),
                    "program_id": PROGRAM_ID,
                    "filter": filter_name,
                    "mode": "fresh" if fresh else "update",
                    "started_utc": query_started,
                    "completed_utc": utc_now(),
                    "status": "failed",
                    "error": f"MAST query failed: {type(exc).__name__}: {exc}",
                },
            )
        raise
    archive_queried_utc = utc_now()
    results = []
    failures = []
    for filter_name in requested:
        selected_products = [
            product for product in products if product.filter_name == filter_name
        ]
        if not selected_products:
            raise RuntimeError(f"MAST returned no {filter_name.upper()} products")
        selected_table = matching_rows(table, selected_products)
        print(f"{filter_name.upper()}: {len(selected_products)} Level-3 association product(s)")
        try:
            result = process_filter(
                filter_name,
                selected_products,
                selected_table,
                data_dir,
                fresh,
                background_match,
                observations_api=observations_api,
                archive_queried_utc=archive_queried_utc,
            )
            results.append(result)
            print(f"{filter_name.upper()}: {result['action']}")
        except Exception as exc:
            failures.append(
                f"{filter_name}: {exc}. Recovery: python "
                f"mosaic/program_10678_pipeline.py --fresh --filter {filter_name}"
            )
            traceback.print_exc()
    if failures:
        raise RuntimeError("One or more filters failed: " + "; ".join(failures))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Maintain compressed Program 10678 Level-3 mosaics from MAST."
    )
    parser.add_argument(
        "--filter",
        action="append",
        choices=SUPPORTED_FILTERS,
        help="Process one filter (repeatable; default: all three filters)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Output/log/work directory (default: PROJECT/data)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Download all current products and exactly rebuild selected mosaics",
    )
    parser.add_argument(
        "--no-bgmatch",
        action="store_true",
        help="Disable additive background matching",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        filters=args.filter,
        fresh=args.fresh,
        background_match=not args.no_bgmatch,
    )


if __name__ == "__main__":
    main()
