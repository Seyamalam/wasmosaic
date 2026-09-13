# OpenCV parity

This project reimplements OpenCV.js behavior in Rust and TypeScript. It does not compile OpenCV or wrap OpenCV.js.

## Baseline

Full operation parity means the 488 callable families selected by the OpenCV.js 4.13.0 browser configuration across `core`, `imgproc`, `objdetect`, `video`, `dnn`, `features2d`, `photo`, and `calib3d`. The independently authored [inventory](INVENTORY.md) is the checked denominator. The 25% milestone is 122 complete families.

The project pins OpenCV.js 4.13.0. A moving `4.x` branch is useful for discovery but cannot define a reproducible release gate. Callable namespace functions, selected constructors, and selected class methods count here. Constants, enum values, data-only types, basic `Mat` structures, generated vector wrappers, and browser helpers have separate compatibility work but do not change the 488-family denominator.

Desktop modules that the official OpenCV.js build disables are outside this parity denominator. This includes `highgui`, `videoio`, and `imgcodecs`. Browser-native image decoding, WebCodecs, WebRTC camera input, workers, and canvas conversion belong to a separate adapter ledger. Those adapters can make this package more useful than OpenCV.js without pretending a browser is a desktop process.

## Module status

| OpenCV.js module | Status      | Next dependency                                             |
| ---------------- | ----------- | ----------------------------------------------------------- |
| core             | Partial     | Typed `Mat`, arithmetic, reductions, transforms             |
| imgproc          | Partial     | Color conversion, interpolation, convolution                |
| objdetect        | Not started | Core matrices, features, model loading                      |
| video            | Not started | Core matrices, pyramids, motion kernels                     |
| dnn              | Not started | Tensor storage, model parser, execution planner             |
| features2d       | Partial     | Configuration only; gradients, pyramids, descriptors remain |
| photo            | Partial     | Tone-map pixel processing and numerical solvers             |
| calib3d          | Not started | Matrix algebra, feature matching, numerical solvers         |

## Fully implemented families

One hundred twenty-five families meet the full-family definition. Current full parity is 125 of 488, or 25.61%.

| Package methods                                                                                                      | OpenCV.js families                              | Verified contract                                                                    |
| -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------ |
| `GFTTDetector.getBlockSize`, `GFTTDetector.getDefaultName`, `GFTTDetector.getHarrisDetector`                         | Matching `cv.GFTTDetector` getters              | Exact arity, defaults, return values, and deleted-handle errors                      |
| `GFTTDetector.getK`, `GFTTDetector.getMaxFeatures`, `GFTTDetector.getMinDistance`, `GFTTDetector.getQualityLevel`    | Matching `cv.GFTTDetector` numeric getters      | Exact signed i32, F64, non-finite, arity, and lifecycle behavior                     |
| `GFTTDetector.setBlockSize`, `GFTTDetector.setMaxFeatures`                                                           | Matching `cv.GFTTDetector` integer setters      | Exact i32 coercion, undefined return, argument errors, and lifecycle                 |
| `GFTTDetector.setHarrisDetector`                                                                                     | `cv.GFTTDetector.setHarrisDetector`             | Exact boolean coercion, undefined return, argument errors, and lifecycle             |
| `GFTTDetector.setK`, `GFTTDetector.setMinDistance`, `GFTTDetector.setQualityLevel`                                   | Matching `cv.GFTTDetector` F64 setters          | Exact number coercion, non-finite values, argument errors, and lifecycle             |
| `MSER.getDefaultName`, `MSER.getDelta`, `MSER.getMinArea`, `MSER.getMaxArea`, `MSER.getPass2Only`                    | Matching `cv.MSER` getters                      | Exact defaults, return values, arity, and lifecycle                                  |
| `MSER.setDelta`, `MSER.setMinArea`, `MSER.setMaxArea`, `MSER.setPass2Only`                                           | Matching `cv.MSER` setters                      | Exact i32 or boolean coercion, return values, errors, and lifecycle                  |
| `ORB.getDefaultName`, `ORB.getFastThreshold`                                                                         | Matching `cv.ORB` getters                       | Exact defaults, return values, arity, and lifecycle                                  |
| `ORB.setEdgeThreshold`, `ORB.setFastThreshold`, `ORB.setFirstLevel`, `ORB.setMaxFeatures`, `ORB.setNLevels`          | Matching integer `cv.ORB` setters               | Exact i32 conversion, validation, return values, errors, and lifecycle               |
| `ORB.setPatchSize`, `ORB.setWTA_K`, `ORB.setScaleFactor`, `ORB.setScoreType`                                         | Matching remaining `cv.ORB` setters             | Exact i32, number, and structural enum conversion with lifecycle parity              |
| `AgastFeatureDetector.getDefaultName`, `FastFeatureDetector.getDefaultName`                                          | Matching AGAST and FAST name getters            | Exact arity, return values, argument errors, and lifecycle                           |
| `AgastFeatureDetector.getNonmaxSuppression`, `FastFeatureDetector.getNonmaxSuppression`                              | Matching AGAST and FAST boolean getters         | Exact arity, boolean state, argument errors, and lifecycle                           |
| `AgastFeatureDetector.getThreshold`, `FastFeatureDetector.getThreshold`                                              | Matching AGAST and FAST threshold getters       | Exact arity, signed i32 state, argument errors, and lifecycle                        |
| `AgastFeatureDetector.setNonmaxSuppression`, `FastFeatureDetector.setNonmaxSuppression`                              | Matching AGAST and FAST boolean setters         | Exact boolean coercion, undefined return, argument errors, and lifecycle             |
| `AgastFeatureDetector.setThreshold`, `FastFeatureDetector.setThreshold`                                              | Matching AGAST and FAST threshold setters       | Exact i32 coercion, undefined return, argument errors, and lifecycle                 |
| `KAZE.getDefaultName`, `KAZE.getExtended`, `KAZE.getUpright`                                                         | Matching non-enum `cv.KAZE` getters             | Exact arity, names or boolean state, argument errors, and lifecycle                  |
| `KAZE.getNOctaveLayers`, `KAZE.getNOctaves`, `KAZE.getThreshold`                                                     | Matching non-enum `cv.KAZE` numeric getters     | Exact signed i32 or F64 state, non-finite values, errors, and lifecycle              |
| `KAZE.setExtended`, `KAZE.setUpright`                                                                                | Matching non-enum `cv.KAZE` boolean setters     | Exact boolean coercion, undefined return, argument errors, and lifecycle             |
| `KAZE.setNOctaveLayers`, `KAZE.setNOctaves`                                                                          | Matching non-enum `cv.KAZE` integer setters     | Exact i32 coercion, undefined return, argument errors, and lifecycle                 |
| `KAZE.setThreshold`                                                                                                  | `cv.KAZE.setThreshold`                          | Exact F64 coercion, non-finite values, argument errors, and lifecycle                |
| `AKAZE.getDefaultName`                                                                                               | `cv.AKAZE.getDefaultName`                       | Exact arity, return value, argument errors, and lifecycle                            |
| `AKAZE.getDescriptorChannels`, `AKAZE.getDescriptorSize`                                                             | Matching non-enum `cv.AKAZE` descriptor getters | Exact signed i32 state, argument errors, and lifecycle                               |
| `AKAZE.getNOctaveLayers`, `AKAZE.getNOctaves`, `AKAZE.getThreshold`                                                  | Matching non-enum `cv.AKAZE` numeric getters    | Exact signed i32 or F64 state, non-finite values, errors, and lifecycle              |
| `AKAZE.setDescriptorChannels`, `AKAZE.setDescriptorSize`                                                             | Matching non-enum `cv.AKAZE` descriptor setters | Exact i32 coercion, undefined return, argument errors, and lifecycle                 |
| `AKAZE.setNOctaveLayers`, `AKAZE.setNOctaves`                                                                        | Matching non-enum `cv.AKAZE` octave setters     | Exact i32 coercion, undefined return, argument errors, and lifecycle                 |
| `AKAZE.setThreshold`                                                                                                 | `cv.AKAZE.setThreshold`                         | Exact F64 coercion, non-finite values, argument errors, and lifecycle                |
| `AKAZE.getDescriptorType`, `AKAZE.getDiffusivity`                                                                    | Matching enum-backed `cv.AKAZE` getters         | Canonical singleton identity, unknown wire values, arity, and lifecycle              |
| `AKAZE.setDescriptorType`, `AKAZE.setDiffusivity`                                                                    | Matching enum-backed `cv.AKAZE` setters         | Structural enum conversion, raw i32 state, errors, and lifecycle                     |
| `KAZE.getDiffusivity`, `KAZE.setDiffusivity`                                                                         | Matching enum-backed `cv.KAZE` methods          | Shared singleton identity, structural conversion, errors, and lifecycle              |
| `AgastFeatureDetector.getType`, `AgastFeatureDetector.setType`                                                       | Matching enum-backed `cv.AgastFeatureDetector`  | Canonical identity, structural conversion, raw i32 state, and lifecycle              |
| `FastFeatureDetector.getType`, `FastFeatureDetector.setType`                                                         | Matching enum-backed `cv.FastFeatureDetector`   | Canonical identity, structural conversion, raw i32 state, and lifecycle              |
| `getOptimalDFTSize`                                                                                                  | `cv.getOptimalDFTSize`                          | Exact arity, i32 coercion, smooth results, errors, and sentinel                      |
| `exp`, `log`, `sqrt`, `pow`, `magnitude`                                                                             | Matching `cv` float-math families               | Exact destinations, depths, aliases, empties, errors, and numeric edges              |
| `cartToPolar`, `polarToCart`                                                                                         | Matching `cv` coordinate-conversion families    | Exact overloads, paired outputs, aliases, empties, types, and precision              |
| `multiply`, `divide`, `addWeighted`, `convertScaleAbs`                                                               | Matching `cv` numeric families                  | Exact overloads, depths, dtype, outputs, aliases, empties, and overflow              |
| `arcLength`, `contourArea`, `boundingRect`                                                                           | Matching `cv` contour geometry families         | Exact I32/F32 layouts, arity, truthiness, empties, errors, and lifetime              |
| `isContourConvex`, `pointPolygonTest`                                                                                | Matching `cv` polygon-query families            | Exact strict convexity, Point2f, small contours, distance, errors, and ROI           |
| `getRotationMatrix2D`                                                                                                | `cv.getRotationMatrix2D`                        | Exact Point2f, F64 conversion, coefficients, and independent ownership               |
| `getAffineTransform`                                                                                                 | `cv.getAffineTransform`                         | Exact F32 point layouts, LU arithmetic, singular zeros, errors, and output           |
| `invertAffineTransform`                                                                                              | `cv.invertAffineTransform`                      | Exact F32/F64 mutable output, depth arithmetic, aliasing, ROI, and errors            |
| `resize`                                                                                                             | `cv.resize`                                     | All seven modes, depth/channel contracts, geometry, ROI, aliasing and binding errors |
| `getStructuringElement`                                                                                              | `cv.getStructuringElement`                      | Exact overloads, integer conversion, four kernel kinds, anchors, and errors          |
| `createHanningWindow`                                                                                                | `cv.createHanningWindow`                        | Exact mutable F32/F64 windows, Size/type conversion, ROI, and numeric bits           |
| `determinant`                                                                                                        | `cv.determinant`                                | Exact F32/F64 square-matrix arithmetic, errors, ROI, and preservation                |
| `setIdentity`                                                                                                        | `cv.setIdentity`                                | Exact overloads, Scalar conversion, all depths, empties, ROI, and aliases            |
| `transpose`                                                                                                          | `cv.transpose`                                  | Exact all-depth OutputArray, aliasing, empty, arity, and lifetime behavior           |
| `flip`                                                                                                               | `cv.flip`                                       | Exact all-depth OutputArray, signed codes, aliasing, errors, and lifetime            |
| `bitwiseNot`                                                                                                         | `cv.bitwise_not`                                | Exact all-depth OutputArray, masks, aliases, raw bits, and lifetime                  |
| `countNonZero`                                                                                                       | `cv.countNonZero`                               | Exact all-depth scalar reduction, empty, ROI, errors, and lifetime                   |
| `mean`, `minMaxLoc`                                                                                                  | `cv.mean`, `cv.minMaxLoc`                       | Exact optional masks, depths, channels, empties, numeric edges, and ROI              |
| `trace`                                                                                                              | `cv.trace`                                      | Exact four-lane diagonal Scalar, depths, empties, ROI, and numeric edges             |
| `repeat`                                                                                                             | `cv.repeat`                                     | Exact all-depth OutputArray, counts, empty, aliasing, and lifetime                   |
| `rotate`                                                                                                             | `cv.rotate`                                     | Exact all-depth OutputArray, codes, empty, aliasing, and lifetime                    |
| `Tonemap.getGamma`, `Tonemap.setGamma`                                                                               | Matching inherited `cv.Tonemap` methods         | Exact float32 state, scalar conversion, arity, returns, and lifecycle                |
| `TonemapDrago.getBias`, `TonemapDrago.getSaturation`, `TonemapDrago.setBias`, `TonemapDrago.setSaturation`           | Matching `cv.TonemapDrago` methods              | Exact defaults, float32 state, conversion, arity, and lifecycle                      |
| `TonemapMantiuk.getScale`, `TonemapMantiuk.getSaturation`, `TonemapMantiuk.setScale`, `TonemapMantiuk.setSaturation` | Matching `cv.TonemapMantiuk` methods            | Exact defaults, float32 state, conversion, arity, and lifecycle                      |
| `TonemapReinhard.getColorAdaptation`, `TonemapReinhard.getIntensity`, `TonemapReinhard.getLightAdaptation`           | Matching `cv.TonemapReinhard` getters           | Exact defaults, float32 state, arity, and lifecycle                                  |
| `TonemapReinhard.setColorAdaptation`, `TonemapReinhard.setIntensity`, `TonemapReinhard.setLightAdaptation`           | Matching `cv.TonemapReinhard` setters           | Exact float conversion, arity, returns, and lifecycle                                |

## Working partial families

Fifty-four families have useful original Rust/WASM slices but do not meet the full-family definition. The project supports 178 families in total.

| Package methods                                                       | OpenCV.js families                                        | Current limit                                             |
| --------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| `add`, `subtract`, `absdiff`, `min`, `max`                            | `cv.add`, `cv.subtract`, `cv.absdiff`, `cv.min`, `cv.max` | Matching U8 matrix operands                               |
| `bitwiseAnd`, `bitwiseOr`, `bitwiseXor`                               | `cv.bitwise_and`, `cv.bitwise_or`, `cv.bitwise_xor`       | U8 matrices without scalar or mask forms                  |
| `compareEqual`, `inRange`                                             | `cv.compare`, `cv.inRange`                                | Selected U8 forms                                         |
| `split`, `merge`                                                      | `cv.split`, `cv.merge`                                    | All depths; selected array call forms                     |
| `hconcat`, `vconcat`                                                  | `cv.hconcat`, `cv.vconcat`                                | All depths; two through four inputs                       |
| `copyMakeBorder`                                                      | `cv.copyMakeBorder`                                       | All depths and five border modes                          |
| `lut`                                                                 | `cv.LUT`                                                  | Byte sources and every table depth                        |
| `norm`, `normalize`                                                   | `cv.norm`, `cv.normalize`                                 | All depths, masks, and major norm modes                   |
| `meanStdDev`, `reduce`                                                | `cv.meanStdDev`, `cv.reduce`                              | All depths and mutable outputs                            |
| `mixChannels`                                                         | `cv.mixChannels`                                          | One source and destination; MatVector remains             |
| `randu`, `randn`, `setRNGSeed`                                        | Matching `cv` random families                             | Package RNG sequences differ from OpenCV                  |
| `getLogLevel`, `setLogLevel`                                          | Matching `cv` logging families                            | Log bindings stay absent from upstream browser artifact.  |
| `transform`, `perspectiveTransform`                                   | `cv.transform`, `cv.perspectiveTransform`                 | Selected channel and coefficient forms                    |
| `invert`, `solve`                                                     | `cv.invert`, `cv.solve`                                   | Selected dense single-channel methods                     |
| `ellipse2Poly`, `clipLine`                                            | Matching `cv` integer geometry helpers                    | Selected integer argument and return forms                |
| `getPerspectiveTransform`                                             | `cv.getPerspectiveTransform`                              | Continuous F32 points; LU/QR with F64 output              |
| `createAKAZE`                                                         | `cv.AKAZE.create`                                         | Static factory is absent from the pinned artifact         |
| `createKAZE`                                                          | `cv.KAZE.create`                                          | Static factory is absent from the pinned artifact         |
| `createAgastFeatureDetector`                                          | `cv.AgastFeatureDetector.create`                          | Static factory is absent from the pinned artifact         |
| `createFastFeatureDetector`                                           | `cv.FastFeatureDetector.create`                           | Static factory is absent from the pinned artifact         |
| `createGFTTDetector`                                                  | `cv.GFTTDetector.create`                                  | One factory shape; `gradientSize` overload remains        |
| `createMSER`                                                          | `cv.MSER.create`                                          | Static factory is absent; `detectRegions` remains         |
| `createORB`                                                           | `cv.ORB.create`                                           | Static factory is absent; detection remains               |
| `cvtColor`, `threshold`                                               | `cv.cvtColor`, `cv.threshold`                             | U8 color codes 0-11, five threshold modes/Otsu            |
| `GaussianBlur`, `morphologyEx`, `Sobel`                               | Matching `cv` neighborhood-filter families                | U8 separable blur, morphology, and 3x3 signed gradients   |
| `Canny`                                                               | `cv.Canny`                                                | U8 3x3 gradients, suppression, and hysteresis             |
| `approxPolyDP`                                                        | `cv.approxPolyDP`                                         | I32/F32 open/closed curves and mutable destinations       |
| `findContours`                                                        | `cv.findContours`                                         | U8 holes, nested hierarchy, SIMPLE/NONE chains            |
| `warpAffine`                                                          | `cv.warpAffine`                                           | U8 nearest/linear warps and constant/replicate borders    |
| `equalizeHist`                                                        | `cv.equalizeHist`                                         | Exact single-channel U8 histogram equalization            |
| `matchTemplate`                                                       | `cv.matchTemplate`                                        | Six U8/F32 methods, binary/weighted masks, F32 score maps |
| `createTonemapDrago`, `createTonemapMantiuk`, `createTonemapReinhard` | Matching global `cv.createTonemap*` functions             | Global factories absent; pixel processing remains         |

The fixture passes the complete pinned contracts for `arcLength`, `contourArea`, and `boundingRect`. It covers `arcLength`'s exact two-argument arity, `contourArea`'s runtime length of zero and one- or two-argument overloads, and `boundingRect`'s exact one-argument arity. It also checks JavaScript truthiness, I32 and F32 contours in `Nx1C2`, `1xNC2`, and `Nx2C1` layouts, deleted inputs, canonical empty bounds, and rejection of F64, U8, and invalid shapes. The package rejects typed empty contours before entering upstream paths that do not return a safe JavaScript error.

The fixture passes the complete pinned contracts for `isContourConvex` and `pointPolygonTest`. It checks exact one- and three-argument calls, strict clockwise and counter-clockwise convexity, collinear and duplicate vertices, concavity, self-crossing, continuous I32/F32 layouts, structural Point2f conversion, float32 narrowing, JavaScript truthiness, one-point and two-point contours, classification, signed distance, traversal-dependent signed zero, non-finite query sentinels, and rejected empty, deleted, invalid-depth, invalid-shape, and non-contiguous inputs.

The fixture passes the complete pinned contract for `getRotationMatrix2D`. It checks runtime length and exact three-argument arity before field access, structural Point2f field ordering and float32 narrowing, strict Embind double conversion for angle and scale, boolean inputs, signed zero, non-finite propagation, bit-exact 2x3 F64 output, and independent allocation and deletion.

The `determinant` fixture passes the exact one-argument Mat contract. It accepts only nonempty square single-channel F32 and F64 matrices, including non-contiguous regions, and verifies that the source and parent allocation do not change. Orders one through three match the direct formulas, stored-F32 widening, signed-zero results, and non-finite propagation. Larger matrices keep depth-specific elimination, absolute pivot cutoffs, exact cutoff acceptance, row-swap signs, singular positive zero, and Hilbert precision. Integer, multichannel, nonsquare, empty, deleted, and non-Mat inputs reject.

The fixture exposes the direct `AKAZE` constructor and all 15 instance methods. Its complete matrix checks exact arity, defaults, return values, scalar coercion, enum namespaces and singleton identity, structural enum setters, raw unknown wire values, deletion, repeat deletion, and calls after deletion. All 15 instance methods pass and count as implemented. The config-listed static `AKAZE.create` binding is absent from the artifact, so the package factory remains partial.

The fixture passes the complete pinned browser contract for `getOptimalDFTSize`. It checks exact arity, Embind signed i32 coercion and errors, negative and zero inputs, smooth-size results, the exclusive `2,125,764,000` upper sentinel, and the remaining signed i32 boundary. Exhaustive Rust tests additionally verify minimality and every representable 2-, 3-, and 5-smooth boundary. This family counts as implemented.

The fixture passes the complete pinned browser contract for `bitwiseNot`. It verifies both destination overloads, all scalar depths and one through five channels, raw floating-point bits, optional U8/I8 masks, compatible and replaced destinations, typed empties, compact and strided matrices, exact aliases, live unmasked overlapping regions, masked overlaps, argument conversion order, invalid inputs, and deleted handles.

The fixture passes the complete pinned browser contracts for `mean` and `minMaxLoc`. It verifies both optional-mask overloads, all scalar depths, the channel limits, compact and strided matrices, mask validation, canonical and typed empty headers, first row-major ties, non-finite values, signed zero, invalid inputs, and deleted handles.

The fixture passes the complete pinned browser contract for `trace`. It verifies exact arity, every scalar depth, one through four channel lanes, rectangular and strided matrices, typed empties, F32 widening, F64 accumulation order, signed zero, non-finite values, invalid inputs, and deleted handles.

The fixture exposes the direct `KAZE` constructor and all 13 instance methods. Its complete matrix checks exact arity, defaults, return values, scalar coercion, shared diffusivity singleton identity, structural enum setter behavior, raw unknown wire values, deletion, repeat deletion, and calls after deletion. All 13 instance methods pass and count as implemented. The config-listed static `KAZE.create` binding is absent from the artifact, so the package factory remains partial.

The same fixture passes the complete call contract for all seven AGAST and all seven FAST instance methods. It checks exact arity, return values, scalar coercion, enum namespaces and singleton identity, structural type setters, raw unknown wire values, deletion, repeat deletion, and calls after deletion. All 14 instance families count as implemented. The official artifact exposes direct constructors but omits the config-listed static `create` methods, so both package factories remain partial.

The fixture exposes the direct `GFTTDetector` constructor and all 13 instance methods. The complete pinned browser matrix checks exact method arity, defaults, return values, integer, number, and boolean coercion, missing and extra arguments, deletion, repeat deletion, and calls after deletion. All 13 instance methods pass and count as implemented. The artifact omits the config-listed static `GFTTDetector.create` method. The package factory remains partial because it covers one six-argument shape and omits the `gradientSize` overload.

The fixture exposes the direct `MSER` constructor and verifies nine configuration methods. It covers exact defaults, arity, signed i32 and boolean conversion, argument errors, deletion, repeat deletion, and calls after deletion. `detectRegions` remains outside this slice. The artifact omits the config-listed static `MSER.create`, so the package convenience factory remains partial.

The fixture exposes the direct `ORB` constructor and verifies eleven configuration methods. It covers constructor defaults, exact method arity, signed i32 and number conversion, the ORB score enum namespace and structural setter conversion, validation, deletion, repeat deletion, and calls after deletion. Detection and descriptors remain outside this slice. The artifact omits the config-listed static `ORB.create`, so the package convenience factory remains partial.

The fixture exposes the direct `TonemapDrago`, `TonemapMantiuk`, and `TonemapReinhard` constructors and the shared inherited gamma methods. Sixteen state methods pass the complete pinned browser matrix for defaults, exact arity, number and boolean conversion, float32 narrowing, signed zero and non-finite values, undefined setter returns, and const-versus-mutable deleted-pointer errors. Pixel processing remains outside this slice. The pinned artifact omits the three config-listed global factories, so the package conveniences remain partial.

### Template matching differential coverage

The dedicated template-matching fixture checks 259 numeric cases and 16 rejection cases against the pinned OpenCV.js 4.13.0 runtime. It covers all six methods, U8/F32 inputs, one through four channels, strided templates, binary and weighted masks, empty masks, zero norms, destination replacement, ROI writes, and finding a planted template with `minMaxLoc`. Numeric comparisons allow `0.0001 + 0.00002 * abs(reference)` because the original scalar F64 accumulation differs from the reference's F32 correlation rounding. Non-finite result comparisons require matching NaN or infinity classifications.

One template/result overlap case is recorded separately as a known difference. WASMosaic snapshots all inputs and returns the mathematical score map; the pinned build produces different values for that alias. This difference, unverified extreme/non-finite inputs, channel counts above four, and incomplete error-detail coverage keep the family partial. Run `bun run test:browser:prepare`, `bun run build`, and `bun run test:browser:serve`, then open `/test/browser/template-matching.html`. Run `/test/browser/differential.html` for the existing full regression fixture.

## Tracked planned sample

The machine-readable implementation ledger tracks `warpPerspective` as an original partial implementation. The upstream inventory already lists all 488 families. An inventory entry is missing until work starts; it does not need a duplicate planned implementation record.

## Definition of done

An operation counts as implemented only when all of these statements are true:

1. Rust implements every tracked overload for the pinned baseline.
2. Strict TypeScript describes inputs, outputs, errors, defaults, and ownership.
3. Differential fixtures compare results with the pinned OpenCV.js build.
4. Integer results match exactly. Floating-point operations declare and enforce a tolerance.
5. Tests cover empty, invalid, non-contiguous, and aliased inputs where the upstream operation accepts them.
6. The scalar WASM implementation runs in supported browsers.
7. SIMD and threaded implementations fall back without changing results.
8. The README, reference documentation, generated ledger, and package exports agree.

Performance does not decide parity. A slower correct implementation may count as parity, but it cannot satisfy the package's performance release gate.

## Update the ledger

1. Add the Rust export and its unit tests.
2. Add the TypeScript method, backend contract, and tests.
3. Add the operation to `src/operations.ts`.
4. Change or add the matching entry in `parity/manifest.ts`.
5. Run `bun run parity:write`.
6. Document numeric differences in this file.
7. Add the README parity row.
8. Run `bun run parity:check`.

CI rejects stale [generated parity JSON](parity.json). Status values are `implemented`, `partial`, and `planned`. Only `implemented` increments the 488-family parity numerator.

## Shape analysis differential coverage

Run `bun run test:browser:prepare`, `bun run build`, then `bun run test:browser:serve` and open `http://127.0.0.1:8766/test/browser/shape-analysis.html`. The comparator runs in a separate worker. The fixture compares complete contour point arrays, matrix headers, hierarchy indices, source preservation, and approximation outputs without sorting contours or rotating polygon vertices.

The September 13, 2026 Chromium run contains 4,637 cases: 4,587 exact result/contract comparisons, 13 matching rejection checks, and 37 explicitly recorded differences. It covers every binary 3×3 mask with all four supported retrieval modes and both chain encodings; nested holes/islands; 32 deterministic larger masks; single-row and empty inputs; strided sources; signed/fractional/wrapping offsets; and contour → perimeter → polygon → bounds/area workflows. Approximation cases cover both depths, all three input layouts, open/closed curves, repeated endpoints, collinear backtracking, compatible destination ROIs, in-place and overlapping output, epsilon validation and binding arity/coercion.

The [recorded differences](../test/browser/shape-analysis-known-differences.json) contain complete actual and reference results. Eight cases exercise source/hierarchy aliasing: WASMosaic snapshots the source, while the pinned comparator clears it before tracing. Twenty-eight approximation cases differ in closed-curve start or singleton vertex choice. One non-finite contour is rejected locally while OpenCV.js returns finite endpoints. The fixture requires those exact outcomes; it fails on new, changed, missing or resolved differences so the ledger must be deliberately updated.

`findContours` remains partial because I32 label input, `RETR_FLOODFILL`, Teh–Chin chains and the alias contract are incomplete. `approxPolyDP` remains partial because of the recorded vertex/non-finite differences and unverified wider numerical/error behavior. Full parity stays at 124/488; support increases to 179 families, including 55 partial families. This checkpoint makes no speed claim.

The implementations follow the public [shape API contract](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html) and the [Douglas–Peucker paper](https://doi.org/10.3138/FM57-6770-U75U-7727). Region labeling, boundary tracing, hierarchy construction and iterative simplification are original Rust code; the comparator is test-only. The [saved browser report](../test/browser/reports/shape-analysis-2026-09-13.json) records the fixture result and build hashes.

### Perspective rectification differential coverage

`test/browser/perspective.html` compares 2,186 cases against pinned OpenCV.js 4.13.0: 1,731 matching output cases and 449 matching rejection cases, plus six frozen solver differences. The fixture covers seven depths, one/four channels, nearest/linear/area interpolation, five borders, forward/inverse maps, singular and zero-denominator maps, regions, destination replacement, in-place and overlapping outputs, binding defaults and errors, transform constructors, and two corner-to-rectangle workflows.

Integer outputs compare exactly; F32 values use `2e-5 * max(1, abs(reference))` tolerance, and F64 values use `2e-9 * max(1, abs(reference))`. The observed maximum absolute difference is `1.52587890625e-5`. Serialized NaN/infinity categories are compared exactly; signed-zero equality is not audited here. The recorded report is [perspective-2026-09-13.json](../test/browser/reports/perspective-2026-09-13.json). The known-difference ledger freezes both actual and reference results, so new or changed differences fail the fixture.

`getPerspectiveTransform` still rejects two degenerate cases and four cases selecting unsupported SVD/EIG/Cholesky methods (including boolean-to-SVD coercion). `warpPerspective` remains partial: cubic/Lanczos interpolation, transparent borders, non-finite matrices and exhaustive numeric/error behavior remain. No family is promoted to full parity. Current totals are **124 implemented, 56 partial and 180 supported of 488**. This checkpoint makes no speed claim.

### Resize differential coverage

`test/browser/resize.html` passes **3,880 cases with no recorded differences**, covering all seven interpolation modes, all seven depths, one through five channels, up/down/mixed scaling, explicit versus scale-derived dimensions, extreme values, non-finite pixels, singleton axes, source and destination regions, exact in-place calls, overlapping views, argument ordering, arity, scalar conversion and errors. The report is [resize-2026-09-13.json](../test/browser/reports/resize-2026-09-13.json).

Integer values and matrix metadata compare exactly. F32 values use `2e-5 * max(1, abs(reference))`; F64 values use `2e-9 * max(1, abs(reference))`. NaN and infinity categories compare exactly. Unspecified pixel bytes after rejected calls are excluded; their output headers are checked. This fixture does not claim signed-zero bit equality for interpolated float output.

The separable sampler was written from pixel-centre/area geometry and public interpolation definitions. Impulse and rounding probes establish the cubic parameter, float32 coefficient precision, 16-bit exact-nearest step, fixed-point linear coefficients and quarter-unit U8 vertical contributions. Lanczos weights use the sinc product identity with its common sine factor cancelled before normalization. No upstream kernel, coefficient table or test code was used.

`resize` is promoted to implemented. Current totals are **125 implemented, 55 partial and 180 supported of 488**. This checkpoint makes no speed claim.
