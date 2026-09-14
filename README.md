# Continuous Registered Visible Human Viewer

WebGPU volume renderer, WebGL2 alignment viewer, and Python 3.12 preprocessing
pipeline for the Visible Human RGB and CT archives. Source slices are read-only.
Processed output defaults to `Visible-Human-Project/Processed/v1/<subject>`.

## Source data and project status

Courtesy of the U.S. National Library of Medicine.

The underlying [Visible Human datasets](https://www.nlm.nih.gov/research/visible/visible_human.html)
are described by NLM as public-domain. Its former access license was replaced
in 2019 by the linked [NLM Terms and Conditions](https://www.nlm.nih.gov/databases/download/terms_and_conditions.html).
This independent application is not endorsed by NLM. Its experimental,
modified registration does not represent NLM's most current or most accurate
data and is not validated for diagnosis or treatment.

The source data include real human cadaver photographs that may be disturbing.
This repository checkpoint contains source code and numerical/JSON metadata,
not photographs, CT pixels, screenshots, or volume bricks. Original code and
documentation in this directory are released under the [MIT License](LICENSE),
separately from the underlying images and third-party libraries; see [NOTICE.md](NOTICE.md).
The entry page requires 18+ self-confirmation and explicit cadaver-imagery
consent before loading the viewer, any subject data, or a GPU context. Consent
is remembered only in tab-scoped session storage, for up to eight hours.
It is a content warning, not verified identity or server-side access control.
“Hide imagery” immediately hides the viewer, clears consent, and reloads the
safe landing page. No date of birth, identity document, or analytics is collected.

See [RELEASE.md](RELEASE.md) for the read-only release configuration, lossless
data packaging, and browser/desktop distribution plan. Hosting and app-store
publication are separate, explicit operations.

This repository is self-contained and can be cloned directly. BallRoller uses
the same repository as the `tools/visible-human-viewer` Git submodule, so its
copy is a pinned checkout rather than duplicated source.

## Saved alignment checkpoints

[checkpoints/2026-09-14-public](checkpoints/2026-09-14-public) freezes the male
public-release inputs byte-for-byte: 39 accepted human knots, the generated
control curve, camera setup, geometry, volume inventories, and input hashes.
[checkpoints/2026-09-13](checkpoints/2026-09-13) is the earlier male/female work
checkpoint. Each index records file hashes. These are experimental snapshots,
not anatomical validation or candidate promotion. Source images, volume bricks,
caches, virtual environments, and review-history archives remain on local disk.
See [checkpoint notes](checkpoints/README.md) before restoring.

To archive a later state into a new directory:

```sh
.venv/bin/python export_checkpoint.py \
  --processed-root /path/to/Visible-Human-Project/Processed/v1 \
  --output checkpoints/YYYY-MM-DD
```

For a male-only public release checkpoint, also pass `--subject male
--include-volume-manifests`. That includes the CT/RGB geometry, input hashes,
and baked knot record, but still excludes every voxel and source image.

## Install and verify

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-py312.txt
.venv/bin/python preprocess.py --synthetic-test
.venv/bin/python -m pytest -q tests
node --test tests/test_*.mjs
```

## Metadata scan and bake

```sh
# Safe: parses names/GE headers and produces a review manifest only.
.venv/bin/python preprocess.py --subject female --dry-run

# Detect and solve discontinuities, producing QC without the large bake.
.venv/bin/python preprocess.py --subject female --analyse

# Expensive: performs analysis, writes corrected lossless PNG and U16BE CT.
.venv/bin/python preprocess.py --subject female --bake
.venv/bin/python preprocess.py --subject male --male-ct frozenCT --bake
```

If an `--analyse` run has already populated the target output directory, use
`--bake-existing` to reuse its reviewed `corrections.json` and skip analysis.
Add `--resume` after an interrupted bake; existing PNGs are verified before
reuse and density layers must have the exact expected byte length.
Use `--modality rgb` or `--modality density` with `--bake-existing` to rebuild
only one output stack.

Use `--root` and `--output` to point at a different archive/output directory.
Male density intentionally looks for `Radiological/frozenCT`; it is a complete
anatomical stack, unlike male normal CT. Female density uses normal CT.

Each bake writes `manifest.json`, `corrections.json`, `overrides.json`,
`qc.json`, and `qc.html`, plus `rgb/*.png` (2048×1216) and
`density/*.u16be` (1024×608). CT values are unsigned big-endian scanner values
(`HU = value - 1024`). Large CT gaps are recorded and are never interpolated.

## Register CT to color

After both corrected stacks exist, create the non-destructive cross-modality
profile. Color is the fixed `0.0–1.0` depth reference; the profile maps that
coordinate to fractional CT frames and maps color UV coordinates into CT UV
coordinates. It does not resample density values.

```sh
.venv/bin/python preprocess.py --subject male --male-ct frozenCT --register-modalities
.venv/bin/python preprocess.py --subject female --register-modalities
```

Use `--analyse-modalities` to print the proposed profile without writing it.
The resulting `alignment.json` records orientation, monotonic depth knots,
smooth regional inverse-UV transforms, confidence, and QC metrics. It also
hashes `manifest.json` and `corrections.json`; the server refuses to use the
profile after either input changes until registration is rerun.

Registration samples the head and neck more densely than the torso, keeps the
physical depth trend within a 1% local correction band, and validates disjoint
holdout slices by anatomical region. A profile is activated only when both its
outer silhouette and interior high-contrast edge distances pass the
region-specific thresholds recorded in `alignment-qc.json`.
These distances are image proxies, not corresponding anatomical landmarks;
passing those thresholds does not establish an anatomical registration.

### Six-curve candidate registration

The v2 optimizer writes a separate candidate and never overwrites the current
profile:

```sh
.venv/bin/python optimize_alignment.py \
  --subject male --memory-limit-gib 4 --candidate alignment-candidate-v2.json
```

It models `[z_position, y_position, x_position, rotation_deg, y_scale,
x_scale]`, fixes the known RGB 0 → CT 7 correspondence, and solves strictly
monotonic Z before in-plane parameters. Three feature pyramid levels are stored
as disk-backed uint8/float16 memory maps in `.alignment-cache-v2`; full RGB and
CT stacks remain on disk. Spatial fitting uses silhouette distance, normalized
internal gradients, field-of-view/body coverage, and local continuity rather
than cross-modality pixel intensity error. Existing X/Y-only anchors are review
locations only; complete reviewed frames become exact six-parameter constraints.

In the viewer choose **Registration → Candidate** for A/B comparison. The
review queue is capped to one fixed cohort of ten high-value pending control
points, chosen from Z ambiguity, failing held-out regions, and the largest gaps
between confirmed anchors. It does not reveal replacement points as reviews
are saved. The ranked Z details show score margins and alternate CT planes. **Save this
frame** confirms the exact Z plane and all five spatial values; **Reject** marks
a bad suggestion. Re-run the candidate after confirmations so the curves and
regional holdouts are refit. Promotion remains disabled until the adaptive Z
queue, regional median/95th-percentile checks, landmark RMS, monotonicity, hard
anchors, and memory limit pass. Promotion first preserves the old profile as
`alignment.previous.json`.

The primary **Saved human anchors** queue contains only persisted entries from
`reviewed-alignment.json`. It can open, step through, and delete those entries.
Any save or deletion invalidates candidate QC and disables promotion until the
candidate is rebuilt from the revised anchor set.

The bottom track marks the spatial knots actually used by the selected profile,
not acquisition-boundary warnings. The timeline presents saved and generated
control points as one unified green knot track and one matching **Working knots**
list. A saved correction replaces the generated values at the same frame without
adding another timeline notch. Deleting either kind removes that exact working
knot from navigation and interpolation, records the exclusion for subsequent
rebuilds, and removes any correction stored at that frame. Clicking a notch,
using either set of Previous/Next buttons, or selecting the list always opens the
same active RGB knot. The adjacent knot buttons and readout expose the two spatial and
Z knots bracketing the current slice, interpolation percentage, and endpoint
position/scale values. Opening a knot loads its absolute Z, X/Y position,
rotation, and X/Y scale into the left editor. Saving is allowed only for an
existing timeline knot, so loading an arbitrary slice cannot silently create a
new control point. A Z-only edit is stored as a Z-only constraint and does not
freeze the untouched spatial transform. Saved Z values must remain strictly
increasing. Candidate remains selected after a rebuild, nearby reviewed
interpolation is previewed by default, and every review-file mutation is backed
up under the processed subject's `review-history` directory.

Female optimization is intentionally blocked until preprocessing is rerun: GE
`.fre` binary image-header dimensions, FOV, and pixel spacing are now decoded,
which changes derived geometry and invalidates older input hashes.

## Medical volume rendering

The separate **Medical Rendering** tab is a full-window WebGPU renderer. It
combines the square, physically scaled 16-bit CT volume with a registered 8-bit
RGB volume on an independently positioned and rotated infinite intersection
plane. Build each subject's CT volume once:

```sh
.venv/bin/python build_volume.py \
  --subject male --brick-size 128 --levels 3 --memory-limit-gib 8
```

Output is written to `Processed/v1/<subject>/volume-v1`. The builder uses the
original 512×512 CT and the accepted CT-only corrections, not the 1024×608
alignment display raster. It emits little-endian 128³ bricks, a three-level
pyramid, physical spacing, exact source hashes, an HU histogram, and per-brick
HU bounds. Build the matching registered RGB volume after CT and alignment
knots are current:

```sh
.venv/bin/python build_rgb_volume.py \
  --subject male --brick-size 128 --memory-limit-gib 8
```

The RGB builder resamples source colour slices into the native CT grid, applies
the current accepted registration knots, and writes exact planar RGB8 bricks to
`Processed/v1/<subject>/rgb-volume-v1`. The male RGB volume is
`512×512×1877`, uses three bytes per voxel, and occupies about 1.38 GiB at L0.
It also derives RGB levels matching every CT pyramid level. The viewer loads
paired CT and RGB at the coarsest level first. It stops there until the user
explicitly chooses **More detail**, with the next level's transfer size shown.
Private local users can set `VISIBLE_HUMAN_AUTO_DETAIL=native` to restore
automatic refinement; public mode always requires explicit larger downloads.
On the current male stack,
L2 is `128×128×470` (about 37 MiB combined with CT), L1 is
`256×256×939`, and L0 is `512×512×1877`.

At 16 bits per voxel, 8 GiB can theoretically hold 4,294,967,296 voxels:
`1625³`, `1024×1024×4096`, or `512×512×16384`. Those figures leave no room for
the browser or render resources. The renderer therefore limits the main voxel
texture to 4 GiB, targets 6 GiB of renderer-owned allocations, and refuses to
cross an 8 GiB hard ceiling. It uploads one brick at a time, so it never holds
a second full CPU copy. The current native male and female volumes occupy about
0.917 GiB and 0.847 GiB respectively.

The top-left control selects the rendering preset. Drag to orbit, scroll to
zoom, Shift-drag to pan, Control-drag to rotate the intersection plane relative
to camera up/right, and Command-drag vertically
to move its position continuously. The plane is always enabled, its registered
RGB sampling scale is fixed at 1:1, and the cut is mathematically infinite.
W/S dolly, A/D move laterally, and Q/E move vertically. Selecting **Custom**
reveals an HU density slider; the first crossing of that density along a ray
samples the registered RGB volume, returns it as an opaque hit, and stops that
ray. Render quality is always adaptive:
interaction uses a smaller render target and longer ray steps, then refines
after input stops.

CT volume API: `GET /api/subjects/:subject/volume/manifest`,
`GET /api/subjects/:subject/volume/bricks/:level/:x/:y/:z`, and
`POST /api/subjects/:subject/volume/build`. Registered RGB uses the equivalent
`/volume/rgb/manifest`, `/volume/rgb/bricks/...`, and `/volume/rgb/build`
routes. This is a research visualization, not a diagnostic medical device.

### Live 3D alignment

Open `/?subject=male#align3d` or select **3D Alignment**. The same knot editor
and working-knot set used in the 2D alignment context are overlaid on the 3D
renderer. Previous/Next select that exact knot, centre its highlighted plane,
and load its original RGB photograph. Z, X/Y, rotation and independent scales
preview continuously. CT/RGB blends CT with the selected photograph; Custom
also exposes the remapped RGB throughout the CT density surface. Save updates
the selected existing knot, never creates an additional one. Reset reloads its
saved/working values. Edits do not replace the saved medical camera setup.

Editing streams paired L2 then L1 volumes (about 294 MiB of voxel textures for
male L1), a single 1024×608 RGB slice, and a roughly 60 KB native-depth lookup
table. The GPU maps the current alignment back into the RGB build's immutable
`alignment_knots`; no volume rebuild is needed for each edit. Shader sampling
does not cross acquisition seams or invalid intervals. Existing unchanged
baked regions remain unchanged. New corrections across long unsupported
intervals are explicitly flagged rather than silently using stale alignment.
Colour absent from the baked volume cannot be recovered by remapping; the
selected original photograph remains available for inspection.

`GET /api/subjects/:subject/volume/rgb/manifest?live=1` permits alignment-only
staleness when a valid build snapshot exists. Source/CT geometry changes still
require rebuilding. The normal RGB manifest endpoint remains strict. Normal
3D viewing first uses that endpoint, then retries the verified live endpoint
after an alignment-only stale response and applies the saved knot curve to the
baked RGB volume. Unsaved editor drafts are excluded. The saved camera,
infinite intersection plane, and optional refinement through native resolution
remain available; no knot editor or selected-slice highlight is shown.

“Auto Align · no reliable improvement” means the proposed local fit failed
the acceptance checks and the editor kept its original values. It does not
establish that the current alignment is correct.

## Serve

```sh
node server.mjs
```

Open <http://127.0.0.1:4173>. `VISIBLE_HUMAN_ROOT` selects the archive root;
`VISIBLE_HUMAN_PROCESSED_ROOT` selects `Processed/v1`. The viewer falls back to
raw source layers when no baked manifest exists. It uses two bracketing RGB and
two bracketing density textures: bilinear XY, linear depth interpolation, then
Color↔Density blend. `[` and `]` move exactly one colour plane.
When `alignment.json` is current, **align CT** applies the depth and spatial
profile before interpolation. The comparison selector provides a continuous
blend, colored RGB/CT edges, checkerboard, and side-by-side full CT views.
Outside registered depth coverage, available CT is shown as an explicitly
unregistered estimate using source frame spacing and the nearest spatial fit.
It must be verified with the CT-plane control. Missing CT or sampling outside
the CT canvas is marked with orange hatching; no missing source is synthesized.

**Problem areas** ranks sampled regions by body coverage, scale changes, canvas
exclusion and interior-edge disagreement. **Scan alignment** refreshes
`alignment-problems.json`; this audits the base profile, not reviewed previews.
Issue spans group warnings from sampled frames, not a verdict on every slice.

**Align this slice** edits any existing color frame. Choose the matching CT
plane first, then use **Suggest outline fit** or **Match landmarks**. Outline
suggestions filter conservative annotation/support components, retain a better
existing fit, and warn about worse interior agreement. They are never activated
without **Save this frame**. For matching landmarks, click a color point on the
left and the corresponding point in the full, oriented CT on the right. Use at
least three noncollinear pairs (preferably 4–6 spread across anatomy), then
**Fit points**. Check additional structures not used for the fit.

`Scale X/Y`, `Move X/Y`, and clockwise `Rotation°` adjust the current fitted overlay around image center.
Positive X moves right; positive Y moves down. Drag or arrow keys translate
(`Shift` + arrow is a quarter analysis pixel). New saves in
`reviewed-alignment.json` record absolute inverse-UV transforms, the explicit
six parameters, exact CT-frame correspondence, landmarks and input hashes. Stale saves are excluded after a
base-profile change. Original `manual-alignment.json` anchors remain selectable
for migration; their prior automatic-knot taper is no longer used.

Saved transforms apply at reviewed slices by default. The optional interpolation
preview blends absolute transforms between nearby reviewed frames (at most 4%
of stack depth), including their CT correspondence. It never crosses known
acquisition boundaries or extrapolates; unreviewed gaps retain the automatic fit.
This is a review preview, not a regenerated or validated full-stack profile.

Review API: `GET /api/review/:subject/problems`, `POST .../scan`,
`POST .../suggest/:frame`, and `GET .../frames`, `PUT|DELETE .../frames/:frame`.
Browser smoke test (Chrome, localhost server running):
`node tests/browser_review_smoke.mjs`.
It exercises previews only and does not save to the real dataset.
For unbaked male frozen CT, the local server decodes the archive's 16-bit PNG
to U16BE before streaming it, so the WebGL density path never silently falls
back to 8-bit PNG canvas values.

Boundary review is deliberately non-destructive: choose a flagged boundary,
edit `tx`, `ty`, `scale`, or rotation, then **Save + rebuild**. After the first
full bake, the local server rewrites only that boundary's right-hand segment;
the panel polls `/api/review/jobs/:job`. Enable raw/corrected flicker to inspect
the result in place.
Overrides are stored in the subject's `overrides.json` for auditability. The
API is `GET /api/subjects`, `GET /api/subjects/:subject/manifest`,
`GET /api/subjects/:subject/:modality/layers/:frame`,
`GET|PUT /api/review/:subject/boundaries/:boundary`,
`POST /api/review/:subject/rebuild`, and `GET /api/review/jobs/:job`.

`--dry-run` is the recommended first run. It parses all filenames and GE/frozen
header metadata without opening image payloads. The current mounted archives
scan as Male: 1,878 RGB / 1,877 frozen-density, Female: 5,186 RGB / 1,734
normal-density layers.
