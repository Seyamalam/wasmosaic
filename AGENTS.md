# WASMosaic agent notes

## Product goal

Build a browser-first image-processing library in original Rust, WebAssembly, and strict TypeScript. Match the pinned OpenCV.js 4.13.0 browser API, then beat it on measured hot paths. Full parity and better browser ergonomics are long-term goals. Speed claims require reproducible browser benchmarks.

## Current state

- Public npm package: `wasmosaic`
- Public repository: `https://github.com/Seyamalam/wasmosaic`
- Latest published version: `0.2.0`
- Published release tag: `v0.2.0`
- Read `README.md` and `docs/parity.json` for current parity counts; unreleased work can exceed the published checkpoint.
- Local checkout path: `/Users/seyam/Work/bun_opencv`

Do not rename the package back to a name containing `OpenCV`. OpenCV is the compatibility target, not the product identity.

## Local npm publishing

This machine stores the npm publishing credential in macOS Keychain:

- Keychain account: `seyamalam`
- Keychain service: `npm:wasmosaic:publish`
- Publish helper: `/Users/seyam/.local/bin/wasmosaic-publish`

Never print the Keychain value, place it in a command argument, commit it, write it to the repository, or leave it in `~/.npmrc`. The helper reads it into a temporary permission-restricted npm config and removes that file after publishing.

The current credential was pasted into a chat before being stored. Replace it with a fresh granular npm token when practical. Keep package write access and bypass-2FA enabled, use the shortest useful expiration, and update the same Keychain item. Never save the replacement token in this file.

Before publishing:

1. Update the version in `package.json`, the Cargo workspace, and `parity/manifest.ts`.
2. Update `CHANGELOG.md` and regenerate `docs/parity.json` with `bun run parity:write`.
3. Run `bun run check`, `bun run build`, and `bun run release:check`.
4. Commit and push the exact release state.
5. Run `wasmosaic-publish`.
6. Verify the registry with `npm view wasmosaic@<version>` and install it in a fresh temporary project.
7. Push the matching annotated Git tag and create the GitHub release.

## Source independence

Implement behavior from public API documentation, standards, papers, textbooks, and black-box differential tests. Do not copy or translate OpenCV kernels, control flow, generated bindings, configuration structure, tests, fixtures, tables, comments, or documentation prose. The OpenCV.js comparator stays test-only and outside the npm archive.

Each parity promotion needs an original implementation, a specification source, strict TypeScript types, Rust and TypeScript tests, pinned-browser differential evidence, and updated parity docs.

## Verification commands

```sh
bun run check
bun run build
bun run release:check
npm pack --dry-run --ignore-scripts
```

The real-browser differential fixture must also pass before promoting browser parity. Preserve the published denominator of 488 callable families unless the pinned upstream inventory changes with documented evidence.

## Working style

Make focused commits and push completed checkpoints frequently. Preserve unrelated user changes. Prefer Rust for pixel loops and compute-heavy code, TypeScript for validation and browser ergonomics, and WebAssembly as the boundary. Add SIMD, threads, workers, or GPU paths only after benchmarks identify a real bottleneck, and keep a portable fallback.

## Next API work

Use [the next API priorities](docs/NEXT_API_PRIORITIES.md) as the default implementation order. Favor complete user workflows over isolated configuration methods. Re-rank when dependency work or browser benchmarks change the cost-to-value ratio.

Template matching has a separate browser fixture at `test/browser/template-matching.html` and a scalar benchmark at `test/browser/template-matching-benchmark.html`. Serve them with `bun run test:browser:serve` after building and verifying the comparator cache. Keep the documented template/result alias difference visible when extending coverage.
