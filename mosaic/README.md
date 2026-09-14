# JWST Program 10678 mosaics

This project queries MAST for the public Level-3 association mosaics from JWST
Program 10678 and maintains one north-up Galactic mosaic for each of these
filters:

- MIRI F770W
- NIRCam F212N
- NIRCam F480M

Only MAST minimum-recommended, science, Level-3 `I2D` FITS products are used.
This is important because a broad search for `I2D` products also returns many
large detector-level files that are not inputs to these mosaics.
The distant `jw10678-o138_t138*` and `jw10678-o139_t139*` products are excluded
by default because they make the mosaic footprint much larger.

## Run the pipeline

Use the `astro` conda environment. To check all three filters, download only
new products, and update mosaics where needed:

```bash
conda run -n astro --no-capture-output \
    python mosaic/program_10678_pipeline.py
```

Run only one filter with:

```bash
conda run -n astro --no-capture-output \
    python mosaic/program_10678_pipeline.py --filter f770w
```

To discard the incremental state and exactly rebuild every mosaic from the
current MAST products:

```bash
conda run -n astro --no-capture-output \
    python mosaic/program_10678_pipeline.py --fresh
```

A fresh rebuild can also be restricted to one filter:

```bash
conda run -n astro --no-capture-output \
    python mosaic/program_10678_pipeline.py --fresh --filter f212n
```

The default data directory is `./data` at the project root, independent of the
directory from which the command is launched. Use `--data-dir /path/to/data`
to override it. Add `--no-bgmatch` to disable additive background matching.
`--filter` may be repeated to process a chosen subset.

To include the two distant, mosaic-enlarging pointings, add:

```bash
conda run -n astro --no-capture-output \
    python mosaic/program_10678_pipeline.py --include-large-pointings
```

Normal incremental runs never remove products already in a mosaic. If an
existing mosaic includes these pointings and you want the smaller default
footprint, rebuild it with `--fresh` and omit `--include-large-pointings`.

## Outputs and disk cleanup

Successful runs leave only the compressed mosaics and small provenance logs:

```text
data/
  gc_mosaic_f770w.fits.gz
  gc_mosaic_f212n.fits.gz
  gc_mosaic_f480m.fits.gz
  logs/
    f770w_manifest.ecsv
    f770w_runs.jsonl
    f212n_manifest.ecsv
    f212n_runs.jsonl
    f480m_manifest.ecsv
    f480m_runs.jsonl
```

Downloads are staged below `data/work/<filter>/downloads` and retained after a
successful run so a later run can reuse them. A partial staged file whose size
does not match MAST is removed and downloaded again.

To remove downloaded FITS files after the new compressed mosaic, manifest, and
success log have been written, add `--remove-downloads`:

```bash
python mosaic/program_10678_pipeline.py --remove-downloads
```

Without that option, retained files remain below
`data/work/<filter>/downloads`. A later run checks their sizes against the MAST
metadata and reuses valid files. This is particularly useful with `--fresh`,
but requires enough disk space to keep the full selected archive inventory. If
a download or mosaic fails, staged files are retained regardless of the cleanup
option so the next run can resume.

Each `.fits.gz` file is lossless and can be opened directly by Astropy. The
primary HDU contains the surface-brightness mosaic and the `COVERAGE` extension
contains the summed reprojection weights needed for future incremental
updates. Uncovered pixels are `NaN`. The manifest is the authoritative list of
MAST filenames and data URIs included in the current mosaic. The JSON-lines run
log records UTC dates, queries, downloads, deletions, output action, missing
archive products, and failures.

A successful `--fresh --filter ...` resets the selected filter's manifest and
run history. An unfiltered `--fresh` resets all three independently. Failed
fresh runs retain the previous output and provenance.

## Incremental versus fresh results

A normal run compares MAST with the saved manifest. If nothing is new, it
validates the compressed mosaic, appends a no-change record, and downloads
nothing. New products are reprojected onto the established pixel grid and
combined with the existing mosaic using `COVERAGE` as the old weight. The grid
is extended by whole pixels when necessary, so existing pixels are not
resampled.

For background matching, a new tile that overlaps existing coverage is matched
to that established zero level. A disconnected new component remains at its
original zero level. Historical tile backgrounds cannot be globally refitted
after their individual files have been deleted, so `--fresh` is the
authoritative exact recomputation.

If a product previously used in a mosaic is no longer returned by MAST, a
normal incremental run warns through the log and retains it. A fresh rebuild
uses exactly the products currently returned by MAST. Reprocessed data that
keep both the same filename and data URI are also picked up only by `--fresh`.

The current archive inventory is roughly 34 GiB before mosaicking, dominated
by F212N. A fresh run therefore needs enough temporary space for all selected
downloads, the disk-backed reprojection arrays, an uncompressed output, and
the final compressed output. Do not interpret metadata-only or synthetic tests
as a completed production run.

## Standalone mosaicking utility

`gc_mosaic.py` remains available for arbitrary local `i2d` collections:

```bash
conda run -n astro --no-capture-output python mosaic/gc_mosaic.py \
    --filter f770w \
    --inputdir /path/to/JWST
```

It searches recursively for `*_<filter>_i2d.fits`, uses the `SCI` extension by
default, performs bilinear reprojection and a mean coadd, and writes
`gc_mosaic_<filter>.fits`. Existing outputs are protected unless `--overwrite`
is supplied. Alternate extensions can be selected with `--extension`.

Background matching is performed independently within each connected group of
finite image footprints. Isolated images remain unchanged, so relative zero
levels between disconnected groups are unconstrained. The result is a mean
surface-brightness mosaic, not an uncertainty-propagated product.

## Requirements and tests

The `astro` environment needs Python, NumPy, Astropy, astroquery, reproject,
and pytest. Run all focused tests with:

```bash
conda run -n astro --no-capture-output python -m pytest -q \
    mosaic/test_gc_mosaic.py mosaic/test_program_10678_pipeline.py
```

The pipeline tests use synthetic FITS files and a mocked MAST service. They do
not download the production dataset.
