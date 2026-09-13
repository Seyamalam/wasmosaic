# Performance contract

The package is not faster because it uses Rust. OpenCV.js already runs optimized C++ through WebAssembly. The likely wins come from smaller specialized kernels, fewer copies, reusable memory, fused pipelines, SIMD, and browser-aware scheduling.

## Release targets

The 1.0 target is at least 2x the OpenCV.js geometric mean for warmed 1080p hot kernels and at least 4x for pipelines where this package can keep intermediates in WASM memory. No supported hot kernel should regress by more than 10% without a documented size, precision, or compatibility reason.

These numbers are goals. The project must not describe itself as faster until published benchmark artifacts meet them.

## Reference implementation

Benchmarks compare against the same pinned OpenCV.js release used by the parity ledger. Both libraries receive the same decoded pixels. Decode and network time stay outside operation timing unless a benchmark explicitly measures an end-to-end browser adapter.

## Required cases

- 256 by 256 images for small interactive work;
- 1920 by 1080 images for common camera and video frames;
- 3840 by 2160 images for memory pressure and parallel scheduling;
- one-channel and four-channel matrices where the operation supports both;
- cold initialization and warmed execution;
- scalar WASM, SIMD WASM, and threaded WASM when the browser supports them.

## Measurements

Each report records p50 and p95 elapsed time, warm-up policy, sample count, browser version, CPU, operating system, input type, input dimensions, allocation count, bytes copied across the JavaScript and WASM seam, WASM bytes, and total packed bytes.

Use enough iterations to produce at least one second of measured work per case. Report every case. Do not remove a slow input because it hurts the geometric mean.

## Optimization order

1. Prove numeric parity with differential fixtures.
2. Remove redundant JavaScript and WASM copies.
3. Reuse allocations and keep pipeline intermediates in WASM memory.
4. Measure scalar kernels and fix cache-unfriendly access.
5. Add SIMD for kernels with enough work to repay dispatch.
6. Add workers only when transfer, synchronization, and startup costs are lower than the saved compute time.
7. Consider WebGPU or WebNN for large operations with a measured win and a scalar fallback.

Every optimized implementation must pass the same fixtures as the scalar implementation.

## Template matching baseline

Run `bun run test:browser:prepare`, `bun run build`, and `bun run test:browser:serve`. Open `http://127.0.0.1:8766/test/browser/template-matching.html` to check numeric behavior first, then `http://127.0.0.1:8766/test/browser/template-matching-benchmark.html` to measure it. The benchmark exposes its JSON as `globalThis.templateMatchingBenchmark` and renders it on the page.

The benchmark uses independently generated pixels, an 8×8 template, and `TM_CCOEFF_NORMED`. It includes every combination of U8/F32, one/four channels, and 256×256/1920×1080/3840×2160 images. Separate workers initialize each library; measured calls run serially to avoid competition between the two libraries. Input creation, output inspection, and locating the planted template happen outside timing. Each case records its first call, three subsequent warm-ups, and at least five samples totaling at least one second. Results include all samples, p50, p95, and fresh-worker initialization time. That initialization measurement is not a fully cold browser start.

The [2026-09-13 baseline](../benchmarks/template-matching/2026-09-13-chromium.json) records Chrome 152 on an Apple M5 Pro with macOS 26.6.2. On the 1080p U8 cases, package p50 was 317.8 ms for one channel and 894.0 ms for four channels; the pinned reference measured 36.0 ms and 135.3 ms respectively. This direct scalar kernel is slower than OpenCV.js in every measured case. These results identify work to optimize, not a speed target achieved.

This is a scoped baseline, not a complete performance-contract report. It covers one matching method and unmasked inputs; internal allocation counts are uninstrumented, and machine-wide background load was not controlled. Timed calls transfer handles and scalars only, with zero pixel bytes copied across the JavaScript/WASM boundary. The package still snapshots inputs and allocates intermediates inside WASM. The saved artifact also records source/build hashes, WASM bytes, and the dry-run package size.
