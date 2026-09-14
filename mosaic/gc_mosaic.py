#!/usr/bin/env python3
"""Combine JWST i2d images into one north-up Galactic mosaic."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel
from reproject import __version__ as reproject_version
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd


@dataclass
class Image:
    path: Path
    extension: str | int
    shape: tuple[int, int]
    wcs: WCS


def _close_memmap(array: np.memmap) -> None:
    """Flush and close a memmap before its temporary directory is removed."""
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def discover_files(input_dir: Path, filter_name: str) -> list[Path]:
    """Find matching i2d FITS files below input_dir."""
    suffix = f"_{filter_name.lower()}_i2d.fits"
    return sorted(
        path for path in input_dir.rglob("*.fits") if path.name.lower().endswith(suffix)
    )


def parse_extension(value: str) -> str | int:
    """Allow either a FITS extension name or a numeric index."""
    return int(value) if value.isdigit() else value


def load_images(paths: list[Path], extension: str | int) -> tuple[list[Image], str | None]:
    """Load and validate the requested 2-D image extension."""
    images = []
    units = set()

    for path in paths:
        with fits.open(path, memmap=True) as hdul:
            try:
                hdu = hdul[extension]
            except (KeyError, IndexError) as exc:
                raise ValueError(f"{path.name}: extension {extension!r} was not found") from exc

            if hdu.data is None or hdu.data.ndim != 2:
                shape = None if hdu.data is None else hdu.data.shape
                raise ValueError(f"{path.name}: extension {extension!r} is not 2-D (shape={shape})")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                wcs = WCS(hdu.header).celestial
            if not wcs.has_celestial or wcs.pixel_n_dim != 2:
                raise ValueError(f"{path.name}: extension {extension!r} has no 2-D celestial WCS")

            has_finite = any(
                np.any(np.isfinite(hdu.data[start : start + 256]))
                for start in range(0, hdu.data.shape[0], 256)
            )
            if not has_finite:
                raise ValueError(f"{path.name}: extension {extension!r} has no finite pixels")

            unit = hdu.header.get("BUNIT")
            units.add(None if unit is None else unit.strip())
            images.append(
                Image(path=path, extension=extension, shape=hdu.data.shape, wcs=wcs)
            )

    if len(units) > 1:
        values = sorted(repr(unit) for unit in units)
        raise ValueError(f"Input extensions have inconsistent BUNIT values: {values}")

    return images, next(iter(units), None)


def projected_bounds(image: Image, output_wcs: WCS, shape_out: tuple[int, int]) -> tuple[slice, slice]:
    """Return a safe output-pixel bounding box for one input image."""
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

    ymin = max(0, int(np.floor(np.min(out_y[valid]))) - 2)
    ymax = min(shape_out[0], int(np.ceil(np.max(out_y[valid]))) + 3)
    xmin = max(0, int(np.floor(np.min(out_x[valid]))) - 2)
    xmax = min(shape_out[1], int(np.ceil(np.max(out_x[valid]))) + 3)
    return slice(ymin, ymax), slice(xmin, xmax)


def bounds_intersect(first: tuple[slice, slice], second: tuple[slice, slice]) -> bool:
    """Return True when two output-pixel bounding boxes intersect."""
    ay, ax = first
    by, bx = second
    return ay.start < by.stop and by.start < ay.stop and ax.start < bx.stop and bx.start < ax.stop


def finite_footprints_overlap(
    first: Image,
    second: Image,
    first_bounds: tuple[slice, slice] | None = None,
    second_bounds: tuple[slice, slice] | None = None,
    output_wcs: WCS | None = None,
) -> bool:
    """Check overlap using the real finite-pixel masks, not just WCS rectangles."""
    if first_bounds is not None and second_bounds is not None and output_wcs is not None:
        cutout = (
            slice(
                max(first_bounds[0].start, second_bounds[0].start),
                min(first_bounds[0].stop, second_bounds[0].stop),
            ),
            slice(
                max(first_bounds[1].start, second_bounds[1].start),
                min(first_bounds[1].stop, second_bounds[1].stop),
            ),
        )
        target_wcs = output_wcs[cutout]
        target_shape = (
            cutout[0].stop - cutout[0].start,
            cutout[1].stop - cutout[1].start,
        )
    else:
        target_wcs = second.wcs
        target_shape = second.shape

    with fits.open(first.path, memmap=True) as first_hdul, fits.open(
        second.path, memmap=True
    ) as second_hdul:
        first_projected = reproject_interp(
            (first_hdul[first.extension].data, first.wcs),
            target_wcs,
            shape_out=target_shape,
            order="nearest-neighbor",
            return_footprint=False,
        )
        first_finite = np.isfinite(first_projected)
        del first_projected
        second_projected = reproject_interp(
            (second_hdul[second.extension].data, second.wcs),
            target_wcs,
            shape_out=target_shape,
            order="nearest-neighbor",
            return_footprint=False,
        )
        return bool(np.any(first_finite & np.isfinite(second_projected)))


def overlap_components(
    images: list[Image], bounds: list[tuple[slice, slice]], output_wcs: WCS | None = None
) -> list[list[int]]:
    """Find connected components of images with finite-pixel overlap."""
    neighbours = [set() for _ in images]
    for i, first in enumerate(images):
        for j in range(i + 1, len(images)):
            if bounds_intersect(bounds[i], bounds[j]) and finite_footprints_overlap(
                first, images[j], bounds[i], bounds[j], output_wcs
            ):
                neighbours[i].add(j)
                neighbours[j].add(i)

    components = []
    unseen = set(range(len(images)))
    while unseen:
        pending = [unseen.pop()]
        component = []
        while pending:
            index = pending.pop()
            component.append(index)
            new = neighbours[index] & unseen
            unseen.difference_update(new)
            pending.extend(new)
        components.append(sorted(component))
    return components


def component_bounds(
    component: list[int], bounds: list[tuple[slice, slice]]
) -> tuple[slice, slice]:
    """Return the union bounding box for one component."""
    return (
        slice(min(bounds[i][0].start for i in component), max(bounds[i][0].stop for i in component)),
        slice(min(bounds[i][1].start for i in component), max(bounds[i][1].stop for i in component)),
    )


def write_mosaic(
    images: list[Image],
    output_path: Path,
    filter_name: str,
    unit: str | None,
    background_match: bool,
    overwrite: bool,
    extra_header: dict[str, object] | None = None,
) -> Path:
    """Build the component mosaics and write the final FITS image."""
    input_shapes = [(image.shape, image.wcs) for image in images]
    output_wcs, shape_out = find_optimal_celestial_wcs(
        input_shapes, frame="galactic", auto_rotate=False
    )
    shape_out = tuple(int(value) for value in shape_out)
    bounds = [projected_bounds(image, output_wcs, shape_out) for image in images]
    components = overlap_components(images, bounds, output_wcs)

    print(f"Output shape: {shape_out[0]} x {shape_out[1]} pixels")
    print(f"Overlap-connected components: {len(components)}")
    for number, component in enumerate(components, start=1):
        names = ", ".join(images[i].path.name for i in component)
        state = "background matched" if background_match and len(component) > 1 else "unchanged zero level"
        print(f"  Component {number} ({state}): {names}")
    if background_match and len(components) > 1:
        print("Warning: relative backgrounds between disconnected components are unconstrained.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path} (use --overwrite to replace it)")

    with tempfile.TemporaryDirectory(
        prefix="gc_mosaic_", ignore_cleanup_errors=True
    ) as temp_dir:
        mosaic_path = Path(temp_dir) / "mosaic.float32"
        mosaic = np.memmap(mosaic_path, mode="w+", dtype=np.float32, shape=shape_out)
        mosaic[:] = np.nan
        coverage_path = Path(temp_dir) / "coverage.float32"
        coverage = np.memmap(
            coverage_path, mode="w+", dtype=np.float32, shape=shape_out
        )
        coverage[:] = 0

        for number, component in enumerate(components):
            cutout = component_bounds(component, bounds)
            cutout_wcs = output_wcs[cutout]
            cutout_shape = (
                cutout[0].stop - cutout[0].start,
                cutout[1].stop - cutout[1].start,
            )
            component_inputs = [str(images[i].path) for i in component]
            component_mosaic = np.memmap(
                Path(temp_dir) / f"component_{number}.float32",
                mode="w+",
                dtype=np.float32,
                shape=cutout_shape,
            )
            component_footprint = np.memmap(
                Path(temp_dir) / f"footprint_{number}.float32",
                mode="w+",
                dtype=np.float32,
                shape=cutout_shape,
            )
            component_mosaic[:] = 0
            component_footprint[:] = 0
            reproject_and_coadd(
                component_inputs,
                cutout_wcs,
                shape_out=cutout_shape,
                hdu_in=images[component[0]].extension,
                reproject_function=reproject_interp,
                combine_function="mean",
                match_background=background_match and len(component) > 1,
                output_array=component_mosaic,
                output_footprint=component_footprint,
                intermediate_memmap=True,
                blank_pixel_value=np.nan,
            )
            covered = component_footprint > 0
            destination = mosaic[cutout]
            destination[covered] = component_mosaic[covered]
            coverage_destination = coverage[cutout]
            coverage_destination[covered] = component_footprint[covered]
            del covered, destination, coverage_destination
            _close_memmap(component_mosaic)
            _close_memmap(component_footprint)
            del component_mosaic, component_footprint

        mosaic.flush()
        coverage.flush()
        header = output_wcs.to_header(relax=True)
        if unit is not None:
            header["BUNIT"] = unit
        header["FILTER"] = filter_name.upper()
        header["NINPUT"] = len(images)
        header["BGMATCH"] = (background_match, "Additive matching within overlap groups")
        header["NBGCOMP"] = (len(components), "Number of overlap-connected groups")
        header["REPROJ"] = (reproject_version, "reproject version")
        if extra_header:
            for key, value in extra_header.items():
                header[key] = value
        header.add_history("Mean coadd using reproject_interp; zero-coverage pixels are NaN.")
        for image in images:
            header.add_history(f"Input: {image.path}")

        partial_path = output_path.with_name(output_path.name + ".part")
        try:
            fits.HDUList(
                [
                    fits.PrimaryHDU(data=mosaic, header=header),
                    fits.ImageHDU(data=coverage, name="COVERAGE"),
                ]
            ).writeto(partial_path, overwrite=True)
            os.replace(partial_path, output_path)
        finally:
            partial_path.unlink(missing_ok=True)
            _close_memmap(mosaic)
            _close_memmap(coverage)
            del mosaic, coverage

    print(f"Wrote: {output_path}")
    return output_path


def make_mosaic(
    input_dir: str | Path,
    filter_name: str,
    output_dir: str | Path | None = None,
    extension: str | int = "SCI",
    background_match: bool = True,
    overwrite: bool = False,
) -> Path:
    """Discover inputs and create one Galactic FITS mosaic."""
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if not re.fullmatch(r"[A-Za-z0-9]+", filter_name):
        raise ValueError("Filter must contain only letters and numbers")

    output_dir = input_dir if output_dir is None else Path(output_dir).expanduser().resolve()
    output_path = output_dir / f"gc_mosaic_{filter_name.lower()}.fits"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path} (use --overwrite to replace it)")

    paths = discover_files(input_dir, filter_name)
    if not paths:
        raise FileNotFoundError(f"No *_{filter_name.lower()}_i2d.fits files found under {input_dir}")
    print(f"Matched {len(paths)} input file(s):")
    for path in paths:
        print(f"  {path}")

    if isinstance(extension, str):
        extension = parse_extension(extension)
    images, unit = load_images(paths, extension)
    return write_mosaic(
        images, output_path, filter_name, unit, background_match, overwrite
    )


def make_mosaic_from_paths(
    paths: list[str | Path],
    output_path: str | Path,
    filter_name: str,
    extension: str | int = "SCI",
    background_match: bool = True,
    overwrite: bool = False,
    extra_header: dict[str, object] | None = None,
) -> Path:
    """Create a mosaic from an explicit list of i2d products."""
    resolved = [Path(path).expanduser().resolve() for path in paths]
    if not resolved:
        raise ValueError("At least one input path is required")
    if isinstance(extension, str):
        extension = parse_extension(extension)
    images, unit = load_images(resolved, extension)
    return write_mosaic(
        images,
        Path(output_path).expanduser().resolve(),
        filter_name,
        unit,
        background_match,
        overwrite,
        extra_header=extra_header,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mosaic JWST i2d images onto a north-up Galactic WCS."
    )
    parser.add_argument("-filter", "--filter", required=True, help="Filter name, for example f770w")
    parser.add_argument("-inputdir", "--inputdir", required=True, help="Directory containing i2d FITS files")
    parser.add_argument("-outputdir", "--outputdir", help="Output directory (default: input directory)")
    parser.add_argument(
        "-extension", "--extension", default="SCI", type=parse_extension,
        help="Input image extension name or number (default: SCI)",
    )
    parser.add_argument(
        "-nobgmatch", "--no-bgmatch", action="store_true",
        help="Disable additive background matching",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    make_mosaic(
        input_dir=args.inputdir,
        filter_name=args.filter,
        output_dir=args.outputdir,
        extension=args.extension,
        background_match=not args.no_bgmatch,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
