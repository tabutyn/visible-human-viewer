# Image-free alignment checkpoints

These snapshots archive current Visible Human viewer metadata: geometry manifests,
corrections, overrides, alignment profiles/candidates, saved human corrections, camera
and plane preferences, and QC summaries when those files exist. They contain no
photographs, CT pixels, volume bricks, caches, or edit-history directories. References
to image filenames are metadata, not the images themselves.

The male alignment is experimental and not clinically validated. Female alignment
must be treated as stale until the GE pixel-spacing/FOV metadata and derived CT
geometry have been verified and rebuilt. Archiving a candidate does not promote it
or establish anatomical accuracy. QC scores are not clinical validation.

Each dated directory has an `index.json` containing relative paths, SHA-256 hashes,
byte counts, missing optional files, and subject-specific warnings. Source JSON is
copied byte for byte so embedded input-hash relationships are not silently rewritten.
The exporter reads the original dataset without changing it.

Create a new snapshot while the viewer is not saving changes:

```sh
python3 export_checkpoint.py \
  --processed-root /path/to/Visible-Human-Project/Processed/v1 \
  --output checkpoints/YYYY-MM-DD
```

For a male public-release checkpoint, add `--subject male
--include-volume-manifests`. This preserves the CT/RGB geometry, brick
inventory, baked-knot record, and input hashes, but still copies no brick
payloads. The 2026-09-14 public checkpoint uses that form.

An existing output is never intentionally replaced; choose a new dated name. An
exclusive export lock and staging-directory rename prevent cooperating exporters
from overwriting each other or publishing partially copied files. Source changes
detected during export abort it; pause editing for a consistent snapshot.

There is deliberately no automatic restore. Before restoring any metadata, back up
the current destination, verify the checkpoint hashes, and establish that the raw
images, preprocessing, CT geometry, and every input hash match the checkpoint's
dependencies. Only then copy the specifically needed files. Do not relabel a stale
profile as current or edit hashes to bypass provenance checks. This snapshot alone
cannot reconstruct images or recover excluded edit history.

Distributing source code or metadata does not, by itself, determine rights to
redistribute the underlying imagery. Check the applicable NLM dataset terms before
bundling images or publishing an image-serving application.
