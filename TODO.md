# TODO

The roadmap gives release order. This file lists concrete work that can be picked up now.

## Release setup

- [x] Publish `wasmosaic@0.1.0` and create the matching GitHub release.
- [x] Confirm the npm package name.
- [x] Replace the `OWNER` placeholders in `CHANGELOG.md` after creating the repository.
- [x] Add repository, homepage, bugs, and author metadata to `package.json`.
- [ ] Add funding metadata if the project opens a funding channel.
- [ ] Configure npm trusted publishing for the release workflow.
- [ ] Run the package in Vite, webpack, and a direct browser import-map fixture.
- [ ] Record supported browser versions from those results.
- [x] Review the package archive produced by `npm pack --dry-run`.

## Correctness

- [x] Pin OpenCV.js 4.13.0 as the compatibility reference.
- [x] Independently author the 488-family browser binding inventory without copying its organized configuration.
- [x] Add unsigned 8-bit Rust `Mat` storage with channels, dimensions, stride, and zero-copy regions of interest.
- [x] Add deterministic TypeScript disposal for WASM matrix handles.
- [x] Add signed integer and floating-point matrix depths.
- [ ] Add reusable output buffers and in-WASM operation pipelines.
- [ ] Expand differential fixtures to cover all 55 partial operation families.
- [x] Verify and promote all 13 GFTT detector instance methods against the pinned browser call-contract matrix.
- [x] Verify and promote the five primitive AGAST and five primitive FAST instance methods against the pinned browser call-contract matrix.
- [x] Verify and promote the 11 non-enum KAZE instance methods against the pinned browser call-contract matrix.
- [x] Verify and promote the 11 non-enum AKAZE instance methods against the pinned browser call-contract matrix.
- [x] Verify and promote all ten enum-backed AKAZE, KAZE, AGAST, and FAST methods against the pinned browser call-contract matrix.
- [x] Verify and promote `arcLength`, `contourArea`, and `boundingRect` against the pinned browser contract.
- [x] Verify and promote `isContourConvex` and `pointPolygonTest` against the pinned browser contract.
- [x] Verify and promote `getRotationMatrix2D` against the pinned browser contract.
- [x] Verify and promote `determinant` against the pinned browser contract.
- [x] Verify and promote `setIdentity` against the pinned browser contract.
- [x] Verify and promote `getAffineTransform` against the pinned browser contract.
- [x] Verify and promote `getStructuringElement` against the pinned browser contract.
- [x] Verify and promote `invertAffineTransform` against the pinned browser contract.
- [x] Verify and promote `createHanningWindow` against the pinned browser contract.
- [x] Verify and promote `mean` and `minMaxLoc` against the pinned browser contract.
- [x] Verify and promote `trace` against the pinned browser contract.
- [x] Verify and promote `bitwiseNot` against the pinned browser contract.
- [x] Verify and promote all nine MSER configuration methods against the pinned browser contract.
- [x] Verify and promote all eleven ORB configuration methods against the pinned browser contract.
- [x] Verify and promote all sixteen Drago, Mantiuk, Reinhard, and inherited Tonemap state methods against the pinned browser contract.
- [ ] Pin the exact OpenCV reference version used to create fixtures.
- [ ] Decide whether grayscale should match OpenCV's integer rounding byte for byte.
- [ ] Add fuzz or property tests for dimensions, buffer lengths, and resize mappings.

## Performance

- [ ] Build the OpenCV.js comparison harness for 256 by 256, 1080p, and 4K inputs.
- [ ] Record p50, p95, allocation count, seam copies, WASM bytes, and packed bytes.
- [ ] Re-enable wasm-pack's `wasm-opt` pass after its bundled Binaryen supports current Rust bulk-memory output.
- [ ] Benchmark allocation and copy costs on one small and one large image.
- [ ] Measure scalar WASM before adding SIMD.
- [ ] Design reusable output buffers only after profiling real browser calls.
- [ ] Publish bundle and WASM byte sizes in CI.

## API work

- [ ] Decide how asynchronous worker execution should report cancellation.
- [ ] Specify border modes before adding convolution.
- [ ] Specify channel order and output format for general color conversion.
- [x] Add operation-specific parity fixtures before marking the first entries implemented.
- [ ] Add operation-specific parity fixtures before marking another entry implemented. Current checkpoint: 125 of 488 families, or 25.61%.
