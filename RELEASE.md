# MIT release preparation

The source is MIT-licensed; the image datasets and dependencies have separate
provenance and licensing. See LICENSE and NOTICE.md. This repository is also
consumed as a pinned BallRoller submodule; it does not require the parent game
repository. No dataset binaries or cadaver thumbnails belong in a default
README, social preview, or source archive.

## Browser demo and desktop edition

Use the same WebGPU renderer for both. The browser is a lightweight,
read-only entry point; a desktop/offline edition is the better fit for repeated
native-resolution work and large local datasets. Desktop packaging is a next
step, not an implemented installer. Neither packaging choice removes the
decoded RAM/GPU memory requirement. The present web edition is not an offline
PWA and does not promise persistent browser storage.

The default browser startup downloads only the coarsest CT+RGB pair. On the
male data, L2 is 128×128×470: 36.7 MiB decoded voxels and **18.9 MiB lossless
gzip transfer**, measured across all eight L2 bricks. Application scripts and
metadata are additional. L1 and native L0 are explicit **More detail** actions;
their size labels use measured compressed sizes where available and an
uncompressed upper bound otherwise. Compression is not a claim that a full
native dataset is a small download. Interactive render resolution remains
adaptive independently of the chosen data resolution.

Measured male CT+RGB brick sizes (lossless delivery copies, 2026-09-14):

| Detail | Voxel dimensions | Decoded voxels | Compressed transfer |
| --- | --- | --- | --- |
| L2: automatic preview | 128×128×470 | 36.7 MiB | 18.9 MiB |
| L1: optional | 256×256×939 | 293.4 MiB | 146.9 MiB |
| L0: optional native | 512×512×1877 | 2.29 GiB | 1.12 GiB |

Sizes are for that level alone, not cumulative downloads; upgrading reuses
browser-cached lower levels where available. GPU staging, browser state, and
working allocations add to decoded voxel memory. The 2D source-slice archive
is separate and is not needed to start the normal 3D viewer. L0 is the native
CT grid, with RGB resampled onto that grid; it is not the full-resolution
2048×1216 photographic archive.

## Read-only serving

```sh
VISIBLE_HUMAN_MODE=public \
VISIBLE_HUMAN_ROOT=/path/to/Visible-Human-Project \
VISIBLE_HUMAN_DELIVERY_ROOT=/path/to/Visible-Human-Project/Delivery/v1 \
node server.mjs
```

The server rejects all mutation methods before reading their request bodies.
This disables review saves/deletions, optimizers, preprocessing/build jobs,
promotion, and saved-camera writes. The browser opens the normal 3D context,
without alignment editors. Camera controls still work locally. Male is the
only public subject by default; do not expose female until its geometry and
alignment are verified. `VISIBLE_HUMAN_PUBLIC_SUBJECTS` explicitly overrides
that allowlist. Non-loopback binding requires public mode. Put an HTTPS reverse
proxy and appropriate resource/rate limits in front before internet hosting.

The consent page is intentionally image-free; renderer modules, subject APIs,
and textures start only after both unchecked confirmations are selected.
Public data API URLs remain directly downloadable: self-attestation is not
identity verification or an authorization system. Review applicable hosting,
age-assurance, accessibility, and medical-claims requirements before release.

Private editing remains available with the default server mode and `#alignment`
or `#align3d`. Do not expose that editing server to the internet.

## Lossless delivery copies

```sh
.venv/bin/python pack_delivery.py \
  --processed-root /path/to/Visible-Human-Project/Processed/v1 \
  --output /path/to/Visible-Human-Project/Delivery/v1 \
  --subject male --levels 2 1 0 --include-slices --workers 4
```

- CT volume bricks and density slices: gzip, preserving all 16 bits and byte
  order. RGB8 volume bricks: gzip, preserving every channel value.
- RGB PNG slices: lossless WebP, only when smaller. Decoded pixel equality is
  checked. No lossy HU quantization, JPEG conversion, or source replacement.
- Every gzip copy is decompressed and compared byte-for-byte. Content hashes,
  source sizes/timestamps, encoding, and transfer sizes go into an atomic index.
- Four bounded workers; reruns resume matching verified copies. Originals,
  geometry, alignment hashes, and saved knots are never rewritten. A lock
  prevents two packers from changing the same delivery index simultaneously.
- The server uses prepared copies only while their source matches; otherwise
  raw binary responses are gzip-streamed on demand. Browsers transparently
  decode HTTP gzip before the existing texture upload. WebP is image-decoded
  for slices. Missing/corrupt copies fall back or fail rather than silently
  changing CT density values.

For a quick preview-only pack use `--levels 2` without `--include-slices`.
For a representative slice trial add `--include-slices --max-slices 24`.
Generated files stay outside Git. The public release builder and R2 publisher
below turn only the 552 male CT/RGB L2/L1/L0 brick references into a pinned,
immutable browser release. Identical background bricks share a content-addressed
object. The 3,755 source slices remain private even when they are
present in an older delivery index.

## Immutable R2 and Pages release

The public browser is a static Cloudflare Pages bundle. It does not use the
Node API. Its exact release manifest is pinned in `release-config.json`, and
the manifest resolves content-addressed objects in the dedicated
`visible-human-public` R2 bucket. Never point it at `ballroller-map`, reuse the
map-service upload credential, or enable female/raw-slice publication.

Rebuild the registered RGB volume whenever the reviewed, candidate, baseline,
source, or CT manifest hash changes. Then update the lossless delivery copies:

```sh
.venv/bin/python build_rgb_volume.py \
  --processed /path/to/Visible-Human-Project/Processed/v1 --subject male --brick-size 128

.venv/bin/python pack_delivery.py \
  --processed-root /path/to/Visible-Human-Project/Processed/v1 \
  --output /path/to/Visible-Human-Project/Delivery/v1 \
  --subject male --levels 2 1 0 --workers 4
```

Do not add `--include-slices`. The public manifest builder re-hashes and
gzip-decodes every selected brick; it refuses stale RGB input hashes,
incomplete LoDs, unexpected object counts, changed source files, unsafe paths,
or corrupt delivery blobs. It emits no local paths, editor state, female data,
or mutable current pointer:

```sh
.venv/bin/python export_checkpoint.py \
  --processed-root /path/to/Visible-Human-Project/Processed/v1 \
  --output checkpoints/YYYY-MM-DD-public \
  --subject male --include-volume-manifests

.venv/bin/python build_public_release.py \
  --processed-root /path/to/Visible-Human-Project/Processed/v1 \
  --delivery-root /path/to/Visible-Human-Project/Delivery/v1 \
  --output /path/to/Visible-Human-Project/PublicRelease/v1 \
  --public-base-url https://visiblehuman-data.ballrollergames.com

.venv/bin/python build_public_site.py \
  --release-root /path/to/Visible-Human-Project/PublicRelease/v1 \
  --output dist/public-v1
```

The output manifest path is
`v1/releases/<release-sha256>/manifest.json`. Objects are
`v1/objects/<compressed-sha256>.gz`. Every brick records both compressed and
decoded byte counts and SHA-256 values. The browser verifies the manifest hash
and every decoded brick before GPU upload. Failed or missing data leaves an
explicit Retry action; it never falls back to another LoD.

The frozen 2026-09-14 male release has 552 brick references and 547 unique R2
objects: six native CT background bricks are identical, so content addressing
stores them once. Its referenced compressed payload is 1,372,938,027 bytes;
unique object storage is 1,372,846,412 bytes before the manifest.

Install the locked publisher dependencies, then provide a new R2 token scoped
only to `visible-human-public`:

```sh
npm ci

export VHP_R2_ENDPOINT=https://ACCOUNT_ID.r2.cloudflarestorage.com
export VHP_R2_BUCKET=visible-human-public
export VHP_R2_ACCESS_KEY_ID=REDACTED
export VHP_R2_SECRET_ACCESS_KEY=REDACTED
export VHP_R2_PUBLIC_BASE_URL=https://visiblehuman-data.ballrollergames.com
export VHP_PAGES_PROJECT=visible-human
```

Alternatively, put them in an ignored, mode-`600` `.env.vhp-public` file and
pass `--env-file .env.vhp-public` to the publisher. Do not commit these values.
Wrangler Pages deployment also needs a Wrangler
login or `CLOUDFLARE_ACCOUNT_ID` plus an API token scoped to Pages. A dry run
performs remote HEAD verification and reports present/missing objects, bytes,
per-LoD sizes, the release hash, cold-preview GET count, and estimated R2
requests. Apply uploads and verifies missing objects, uploads the immutable
manifest last, and only then deploys the prebuilt Pages directory:

```sh
node publish_visible_human_r2.mjs --dry-run \
  --env-file .env.vhp-public \
  --manifest /path/to/PublicRelease/v1/v1/releases/RELEASE_SHA/manifest.json \
  --delivery-root /path/to/Visible-Human-Project/Delivery/v1 \
  --site dist/public-v1

node publish_visible_human_r2.mjs --apply \
  --env-file .env.vhp-public \
  --manifest /path/to/PublicRelease/v1/v1/releases/RELEASE_SHA/manifest.json \
  --delivery-root /path/to/Visible-Human-Project/Delivery/v1 \
  --site dist/public-v1
```

One-time Cloudflare setup, using an account/zone token rather than the R2 S3
upload token:

```sh
npx wrangler r2 bucket create visible-human-public
npx wrangler r2 bucket cors set visible-human-public --file r2-cors.json --force
npx wrangler r2 bucket cors list visible-human-public
npx wrangler pages project create visible-human
```

In the Cloudflare dashboard, connect
`visiblehuman-data.ballrollergames.com` as the bucket custom domain and disable
its `r2.dev` URL. Attach `visiblehuman.ballrollergames.com` to the Pages
project. Add a Cache Rule matching host
`visiblehuman-data.ballrollergames.com` and path prefix `/v1/objects/`, mark it
eligible for caching, and enable Smart Tiered Cache. Enable R2 Data Access Logs,
review R2/cache analytics, and create account billing alerts. Purge the data
hostname cache after any CORS-policy change; immutable object contents are never
purged during an ordinary release.

Rollback is a Pages-only deployment pinned to a previous immutable manifest.
Keep all older manifests and objects until a separate inventory and cleanup
audit proves they are unreferenced. No cleanup command is part of the publisher.

After the custom domain is live, validate the public transport. The default
checks all eight L2 brick references; `--all` downloads and verifies every LoD:

```sh
node verify_visible_human_r2.mjs \
  --manifest https://visiblehuman-data.ballrollergames.com/v1/releases/RELEASE_SHA/manifest.json

node verify_visible_human_r2.mjs --all \
  --manifest https://visiblehuman-data.ballrollergames.com/v1/releases/RELEASE_SHA/manifest.json
```

This requires HTTPS and the exact content-addressed manifest SHA. It verifies
CORS and exposed headers, immutable cache policy, gzip metadata and automatic
decoding, compressed/decoded lengths, decoded SHA-256, and a byte-range response.

## Before a public launch

Keep the explicit graphic-content warning and a neutral landing preview;
respect the donors and avoid sensational presentation. Retain conspicuous
NLM credit, no-endorsement language, and the experimental/not-for-diagnosis
limitations. Describe current seams, source clipping, and validation status.
Audit dependency notices for any desktop binary bundle. Confirm the public
data-hosting budget and distribution targets before publishing; an MIT license
does not imply an App Store approval.

## Release regression checks

Run the Node and Python suites in README.md. With a read-only real-data server
running locally, this test opens its own headless Chrome profile and captures
both the image-free warning and the WebGPU preview:

```sh
VISIBLE_HUMAN_PUBLIC_URL=http://127.0.0.1:4173/ \
node tests/browser_public_release.mjs
```

It checks no renderer/dataset requests before consent, unchecked controls,
decline and expired confirmation, exactly eight L2 brick requests with no
automatic L1/L0 download, no public saves, session reuse, and Hide imagery.
It does not treat a successful render as anatomical validation.

The static-bundle/R2 emulator exercises the frozen release through automatic
gzip decoding and decoded SHA verification. It captures L2, L1, and L0 after
two explicit detail actions and checks that the browser cache prevents repeat
transfers. Optional modes test one corrupt object, one interrupted response,
and unavailable WebGPU:

```sh
node tests/browser_static_public_lods.mjs
VISIBLE_HUMAN_CORRUPT_ONCE=1 node tests/browser_static_public_lods.mjs
VISIBLE_HUMAN_INTERRUPT_ONCE=1 node tests/browser_static_public_lods.mjs
VISIBLE_HUMAN_NO_WEBGPU=1 node tests/browser_static_public_lods.mjs
```
