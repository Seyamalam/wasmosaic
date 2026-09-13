# Changelog

This file records user-visible changes. The format follows Keep a Changelog, and versions follow Semantic Versioning as described in [the versioning policy](docs/VERSIONING.md).

## [Unreleased]

### Added

- Complete seven-mode `resize` implementation with all pinned depth/channel combinations, cubic/Lanczos and exact sampling, corrected fractional scale geometry, ROI/alias handling and 3,880 passing browser cases. Full parity is now 125 of 488 families, with 55 partial families.

- Original Rust `warpPerspective` with all-depth nearest sampling, five-depth linear/area sampling, five border modes, inverse maps, ROI destinations and alias-safe input snapshots.
- LU/QR perspective constructors and decomposition constants, plus a pinned-browser fixture covering 2,186 cases and two document-rectification workflows.

- Original Rust `approxPolyDP` for finite I32/F32 open and closed curves, with strict TypeScript binding, mutable destinations and source/destination alias support.
- Contour hole tracing and nested island hierarchy for `RETR_EXTERNAL`, `RETR_LIST`, `RETR_CCOMP` and `RETR_TREE`, with exact sibling ordering, SIMPLE/NONE chains, offsets and empty-output clearing.
- A dedicated shape-analysis browser fixture covering exhaustive 3×3 masks, larger nested/generated masks, polygon workflows and approximation contracts.

- Original Rust `matchTemplate` for U8/F32 images with one through four channels, all six matching methods, optional binary U8 and weighted F32 masks, and mutable F32 score maps. Strict TypeScript constants and overloads support matching followed by `minMaxLoc`.
- Dedicated pinned-browser template-matching fixtures and a reproducible scalar benchmark covering 256×256, 1080p, and 4K images with one/four channels and both input depths.

### Compatibility

- `getPerspectiveTransform` now requires continuous F32 points, matching the pinned browser; earlier F64 and strided point extensions are rejected. SVD/EIG/Cholesky and degenerate homogeneous fallback remain unsupported, with six frozen differential cases. Perspective warping remains partial.

- Shape families remain partial: polygon start/singleton choices and non-finite input behavior differ from OpenCV.js; contour source/hierarchy aliasing uses a source snapshot. Label input, flood-fill retrieval and Teh–Chin chains remain unsupported.

- Template matching remains partial. Inputs are snapshotted before destination writes, which differs from one pinned OpenCV.js template/result overlap case. Extreme floating-point behavior and complete error details remain unverified.

## [0.2.0] - 2026-08-28

### Changed

- Renamed the public project and npm package to WASMosaic (`wasmosaic`) so OpenCV remains only the stated compatibility target, not part of the product identity.
- Replaced the mandatory legal and per-operation patent release gates with checks for the reviewed package name and source-independent implementation.

### Added

- Independent inventory of 488 OpenCV.js 4.13.0 callable browser families with a checked 122-family target for 25% parity.
- Rust-owned matrices for U8, I8, U16, I16, I32, F32, and F64 with typed JavaScript factories and compact ROI exports.
- Twelve U8 matrix core slices: saturating arithmetic, absolute difference, bitwise operations, minimum, maximum, equality comparison, inclusive range masks, and non-zero reduction.
- All-depth matrix flip, transpose, rotation, and repeat operations with strided-region handling.
- All-depth count, sum, mean, extrema-location, and trace reductions with explicit NaN rules.
- All-depth identity, uniform random, and normal random matrix fills with reproducible package-owned seed control.
- Package-owned log-level state and 2-, 3-, and 5-smooth DFT size planning utilities.
- Per-element linear, affine, and perspective transforms with allocating and mutable destination forms.
- Dense determinant, inverse, and linear-system solvers with LU, Cholesky, and QR methods.
- Contour measurement, bounds, strict convexity, and signed point-to-polygon queries for continuous I32 and F32 point matrices.
- Structuring-element, Hanning-window, ellipse-polyline, and line-clipping helpers.
- Affine, inverse-affine, rotation, and perspective transform matrix constructors with F64 output.
- Alias-safe single-source `mixChannels` routing across all scalar depths.
- Sixteen partial AKAZE configuration families with typed defaults, validated Rust-owned state, and deterministic disposal, bringing the supported partial total to 85.
- Sixteen partial AGAST and FAST configuration families with signed 32-bit thresholds, typed detector modes, deterministic disposal, and browser-checked state mutation, bringing the supported partial total to 101.
- Fourteen partial KAZE configuration families with typed diffusivity, finite signed thresholds, deterministic disposal, and browser-checked state mutation, bringing the supported partial total to 115.
- One partial GFTT factory family and 13 fully implemented GFTT detector instance methods, bringing the totals to 116 partial, 13 implemented, and 129 supported families.
- A comprehensive pinned-browser GFTT matrix for exact arity, return values, scalar coercion, argument errors, deletion, repeat deletion, and calls after deletion.
- OpenCV.js-compatible `BindingError` identity and GFTT `delete()` lifecycle behavior, while retaining idempotent `dispose()` as a package convenience.
- Ten fully implemented primitive AGAST and FAST instance methods with exact arity, return values, scalar coercion, `BindingError` argument checks, and deletion behavior, bringing the totals to 106 partial, 23 implemented, and 129 supported families.
- Eleven fully implemented non-enum KAZE instance methods with exact arity, complete signed i32, F64, and boolean coercion, `BindingError` argument checks, and deletion behavior, bringing the totals to 95 partial, 34 implemented, and 129 supported families.
- Eleven fully implemented non-enum AKAZE instance methods with exact arity, complete signed i32 and F64 coercion, `BindingError` argument checks, and deletion behavior, bringing the totals to 84 partial, 45 implemented, and 129 supported families.
- OpenCV.js-compatible negative-input and upper-sentinel behavior for `getOptimalDFTSize`, plus a fail-closed numeric browser comparator for enum-backed values.
- A fully implemented `getOptimalDFTSize` family with exact arity, Embind signed i32 coercion and errors, exhaustive smooth-size boundary tests, and the pinned upper sentinel, bringing the totals to 83 partial, 46 implemented, and 129 supported families.
- Ten fully implemented enum-backed detector methods across AKAZE, KAZE, AGAST, and FAST. Their browser contract covers enum namespace descriptors, canonical and shared singleton identity, structural setter coercion, unknown signed i32 wire values, exact arity, and lifetime errors. The totals are now 73 partial, 56 implemented, and 129 supported families; detector factories remain partial.
- A fully implemented `transpose` family with exact two-argument calls, all scalar depths, OutputArray reallocation, empty and deleted matrices, in-place shapes, detached incompatible regions, and OpenCV-compatible overlapping aliases. The totals are now 72 partial, 57 implemented, and 129 supported families.
- A fully implemented `flip` family with exact three-argument calls, Embind signed-int conversion, all scalar depths, empty and deleted matrices, OutputArray reallocation, and live shared-region alias behavior. The totals are now 71 partial, 58 implemented, and 129 supported families.
- A fully implemented `countNonZero` family with exact one-argument calls, all seven scalar depths, empty matrices, non-contiguous regions, floating-point edge values, invalid inputs, and deleted-handle behavior. The totals are now 70 partial, 59 implemented, and 129 supported families.
- A fully implemented `repeat` family with exact four-argument calls, Embind signed-int conversion, all scalar depths, empty and deleted matrices, OutputArray replacement, exact in-place rejection, and live shared-region alias behavior. The totals are now 69 partial, 60 implemented, and 129 supported families.
- A fully implemented `rotate` family with exact three-argument calls and constants, Embind signed-int conversion, all scalar depths, empty and deleted matrices, OutputArray replacement, valid in-place operation, and live shared-region alias behavior. The totals are now 68 partial, 61 implemented, and 129 supported families.
- Fully implemented `exp`, `log`, `sqrt`, `pow`, and `magnitude` families with exact mutable destinations, typed empty headers, live shared-region traversal, all valid scalar depths, Embind F64 power conversion, integer saturation or wrapping, and pinned floating-point edge behavior. The totals are now 63 partial, 66 implemented, and 129 supported families.
- Fully implemented `cartToPolar` and `polarToCart` with exact overload dispatch, JavaScript boolean coercion, paired destination replacement, typed empty layouts, multichannel F32/F64 data, full-rotation angles, live overlapping regions, and the pinned F32 precision path. The totals are now 61 partial, 68 implemented, and 129 supported families.
- Fully implemented `multiply`, `divide`, `addWeighted`, and `convertScaleAbs` with exact overload dispatch, all scalar depths, explicit mixed-depth dtype conversion, OutputArray replacement, live shared-region traversal, typed empty layouts, and pinned CV_32S overflow behavior. The totals are now 57 partial, 72 implemented, and 129 supported families.
- Fully implemented `arcLength`, `contourArea`, and `boundingRect` with exact arity and runtime length, JavaScript truthiness, all three I32/F32 contour layouts, deleted and invalid inputs, canonical empty bounds, and safe typed-empty rejection. The totals are now 54 partial, 75 implemented, and 129 supported families.
- Fully implemented `isContourConvex` and `pointPolygonTest` with exact arity, strict convexity, structural Point2f conversion, float32 narrowing, one-point and segment contours, classification, signed distance, traversal-dependent zero signs, non-finite query sentinels, and complete invalid-input coverage. The totals are now 52 partial, 77 implemented, and 129 supported families.
- A fully implemented `getRotationMatrix2D` family with exact three-argument arity, ordered structural Point2f conversion, float32 center narrowing, strict F64 angle and scale conversion, boolean inputs, signed zero and non-finite propagation, bit-exact 2x3 F64 output, and independent result ownership. The totals are now 51 partial, 78 implemented, and 129 supported families.
- A fully implemented `determinant` family with exact one-argument Mat binding, nonempty square single-channel F32/F64 inputs, strided-region preservation, direct small-matrix behavior, depth-specific elimination and pivot cutoffs, signed zero, non-finite propagation, and exact Hilbert fixtures. The totals are now 50 partial, 79 implemented, and 129 supported families.
- A fully implemented `setIdentity` family with exact overload dispatch, Mat-before-Scalar conversion, structural four-lane Scalars, boolean and non-finite values, all seven shipped depths, integer conversion sentinels, typed empties, multichannel layouts, signed zero, and strided in-place ROI mutation. The totals are now 49 partial, 80 implemented, and 129 supported families.
- A fully implemented `getAffineTransform` family with exact two-Mat dispatch, continuous F32 point layouts, source preservation, singular zero matrices, non-finite propagation, bit-exact LU arithmetic, and fresh 2x3 F64 results. The totals are now 48 partial, 81 implemented, and 129 supported families.
- A fully implemented `getStructuringElement` family with exact overload dispatch, structural Size and Point conversion, Embind signed integers, default and explicit anchors, rectangle, cross, ellipse, and diamond kernels, plus complete invalid-input coverage. The totals are now 47 partial, 82 implemented, and 129 supported families.
- A fully implemented `invertAffineTransform` family with exact two-Mat dispatch, source-depth F32/F64 arithmetic, mutable destination replacement, compatible ROI writes, exact aliasing, singular signed zeros, non-finite propagation, and invalid-input coverage. The totals are now 46 partial, 83 implemented, and 129 supported families.
- A fully implemented `createHanningWindow` family with exact destination-first dispatch, structural Size and type conversion, F32/F64 destination replacement, compatible ROI writes, invalid-input preservation, and bit-exact asymmetric window arithmetic. The totals are now 45 partial, 84 implemented, and 129 supported families.
- Fully implemented `mean` and `minMaxLoc` families with exact optional-mask dispatch, all scalar depths, channel rules, compact and strided matrices, empty-header sentinels, row-major ties, and floating-point edge behavior. The totals are now 43 partial, 86 implemented, and 129 supported families.
- A fully implemented `trace` family with exact one-argument dispatch, all scalar depths, four-lane diagonal sums, rectangular and strided matrices, typed empties, F32 widening, and pinned floating-point accumulation. The totals are now 42 partial, 87 implemented, and 129 supported families.
- A fully implemented `bitwiseNot` family with exact destination overloads, all scalar depths, optional masks, typed empties, strided matrices, raw floating-point bits, aliases, and pinned live-overlap behavior. The totals are now 41 partial, 88 implemented, and 129 supported families.
- Nine fully implemented MSER configuration methods and eleven fully implemented ORB configuration methods with exact defaults, arity, scalar and enum coercion, validation, and lifetime behavior. The absent static factories remain partial, bringing the totals to 43 partial, 108 implemented, and 151 supported families.
- Sixteen fully implemented Drago, Mantiuk, Reinhard, and inherited Tonemap state methods with exact float32 defaults, conversion, arity, return values, and lifetime behavior. Three absent global factories remain partial, bringing the totals to 46 partial, 124 implemented, and 170 supported families.
- OpenCV.js-compatible `cvtColor` codes 0 through 11 for U8 `Mat` values, including RGB/BGR ordering, alpha insertion and removal, grayscale conversion, `dstCn` selection, strided inputs, mutable destination replacement, exact in-place use, and pinned-browser differential fixtures.
- OpenCV.js-compatible mutable `resize` with all-depth nearest-neighbor sampling, U8 half-pixel linear interpolation, U8 area shrinking, explicit or scale-derived dimensions, pinned interpolation constants, and browser differential fixtures.
- OpenCV.js-compatible mutable U8 `threshold` with five fixed modes, single-channel Otsu selection, returned threshold values, strict comparison boundaries, and browser differential fixtures.
- Original Rust neighborhood kernels for U8 `GaussianBlur`, seven `morphologyEx` operations, and 3x3 `Sobel` derivatives with mutable destinations, shared border handling, strict TypeScript constants, and pinned-browser fixtures.
- Original Rust `Canny` edge detection for U8 images with 3x3 gradients, L1/L2 magnitude, non-maximum suppression, double thresholds, hysteresis, and pinned-browser output evidence.
- Rust-owned `MatVector` support and `findContours` for U8 external/list component boundaries, simple or full chains, integer offsets, mutable I32 hierarchy output, and pinned-browser point-order evidence.
- Original Rust `warpAffine` sampling for U8 matrices with F32/F64 transforms, nearest and linear interpolation, constant and replicate borders, inverse-map support, and pinned-browser differential evidence.
- Original Rust single-channel U8 `equalizeHist` with cumulative-histogram lookup tables, constant-image preservation, mutable destinations, and pinned-browser differential evidence.
- Strict TypeScript matrix APIs and WASM adapters for the new depths and operations.

### Planned

- Build differential-test and browser-benchmark foundations against the pinned baseline.
- Complete the remaining overloads and depth forms for every partial core family.
- Add browser differential tests against trusted image fixtures.
- Reduce buffer copies after benchmark data justifies an ownership model.

## [0.1.0] - 2026-08-25

### Added

- Rust-owned unsigned 8-bit matrices with explicit disposal and zero-copy regions of interest.
- Source-independent implementation policy, per-operation provenance fields, third-party notices, and a publication clearance gate.
- Rust implementations for RGBA grayscale, invert, binary threshold, and nearest-neighbor resize.
- A strict TypeScript API with browser `ImageData` conversion helpers.
- A wasm-pack browser build and npm export map.
- Rust and TypeScript unit tests.
- Oxlint with the complete local anti-slop rule set.
- Generated JSON and human-readable OpenCV parity ledgers.
- CI, npm provenance release automation, documentation checks, and version consistency checks.

[Unreleased]: https://github.com/Seyamalam/wasmosaic/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Seyamalam/wasmosaic/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Seyamalam/wasmosaic/releases/tag/v0.1.0
