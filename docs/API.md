# API reference

The package has RGBA image helpers and Rust-owned typed matrices. RGBA helper pixels occupy four bytes in red, green, blue, alpha order. Matrix operations declare their own depth and channel requirements.

## Initialization

### `initOpenCv()`

Loads the generated WebAssembly module and returns `Promise<OpenCv>`. Call it in a browser or bundler that can load emitted `.wasm` assets.

### `createOpenCv(backend)`

Creates a client from an object that implements `OpenCvBackend`. This is useful for alternate runtimes and tests. Application code should normally call `initOpenCv()`.

## Template matching

### `matchTemplate(image, template, result, method, mask?)`

Compares a template with every image window of the same size and writes a single-channel F32 score map into `result`. Image and template must have matching U8 or F32 depths and one through four channels. For an image of width W and height H and a template of width w and height h, the result has width W-w+1 and height H-h+1. An unmasked call also accepts the reversed size relationship when both dimensions fit after swapping.

| Method                          | Best match                                             |
| ------------------------------- | ------------------------------------------------------ |
| `TM_SQDIFF`, `TM_SQDIFF_NORMED` | Minimum squared difference                             |
| `TM_CCORR`, `TM_CCORR_NORMED`   | Maximum correlation                                    |
| `TM_CCOEFF`, `TM_CCOEFF_NORMED` | Maximum correlation after removing each channel's mean |

The constants are available on the initialized client and as named exports. `TemplateMatchMode` is their TypeScript union. The normalized methods divide by template and window norms.

A nonempty mask must have the template's dimensions and either one channel or the template's channel count. U8 masks treat every nonzero value as selected. F32 masks supply weights. An empty `Mat` selects the unmasked path; explicitly passing `undefined` as the fifth argument rejects. With a nonempty mask, zero norms can produce NaN or infinity. The unmasked path uses the pinned zero-norm conventions, including a score of one for a constant template under `TM_CCOEFF_NORMED`.

```ts
const cv = await initOpenCv();
const image = cv.matFromU8(3, 3, 1, new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
const template = cv.matFromU8(2, 2, 1, new Uint8Array([5, 6, 8, 9]));
const scores = cv.emptyMat();
try {
  cv.matchTemplate(image, template, scores, cv.TM_SQDIFF);
  const { minLoc } = cv.minMaxLoc(scores); // { x: 1, y: 1 }
} finally {
  image.dispose();
  template.dispose();
  scores.dispose();
}
```

Strided inputs and compatible destination ROIs work; incompatible destinations are replaced. WASMosaic snapshots inputs before writing any result. Template/result overlap therefore differs from the pinned OpenCV.js behavior in one recorded case. Use a separate result matrix when comparing libraries. The family remains partial, with finite-input browser coverage and documented floating-point tolerances; exhaustive extreme-value and error-detail parity is unfinished. This first scalar implementation is intended for correctness and benchmarking, with runtime proportional to the number of output windows times template area and channel count.

## Images

### `RgbaImage`

```ts
interface RgbaImage {
  readonly data: Uint8Array;
  readonly height: number;
  readonly width: number;
}
```

Width and height must be positive 32-bit integers. `data.byteLength` must equal `width * height * 4`. The complete byte length must also fit into the 32-bit WASM address space.

### `createRgbaImage(width, height, data)`

Validates the dimensions and copies `data`. Copying prevents a later caller mutation from changing an image that has already passed validation.

### `rgbaImageFromImageData(imageData)`

Copies browser `ImageData` into an `RgbaImage`.

### `imageDataFromRgbaImage(image)`

Validates the image and returns browser `ImageData` backed by a new `Uint8ClampedArray`.

## Operations

### `grayscale(image)`

Writes the same luma value to red, green, and blue. The integer formula is `(77R + 150G + 29B + 128) >> 8`. Alpha is unchanged.

### `invert(image)`

Replaces each RGB byte `x` with `255 - x`. Alpha is unchanged.

### `threshold(image, threshold)`

Calculates luma with the grayscale formula. Pixels whose luma is greater than or equal to `threshold` become white. The rest become black. The threshold must be an integer from 0 through 255. Alpha is unchanged.

### `threshold(source, destination, threshold, maximum, type)`

Applies a threshold to a U8 `Mat`, replaces the mutable destination, and returns the threshold used. The five fixed modes are `THRESH_BINARY`, `THRESH_BINARY_INV`, `THRESH_TRUNC`, `THRESH_TOZERO`, and `THRESH_TOZERO_INV`; comparisons use the OpenCV-compatible strict `source > threshold` boundary. Combining a fixed mode with `THRESH_OTSU` selects the threshold from a single-channel U8 histogram.

The package exports the pinned `THRESH_*` constants. `THRESH_TRIANGLE` and wider documented matrix depths are reserved but reject until their kernels and differential fixtures land.

### `GaussianBlur(source, destination, size, sigmaX, sigmaY?, borderType?)`

Applies an original separable Gaussian kernel to an interleaved U8 `Mat`. Kernel dimensions must be positive odd values. A non-positive sigma uses the documented size-derived default; 3-tap zero-sigma kernels use exact binomial weights. The default border is `BORDER_REFLECT_101`.

### `morphologyEx(source, destination, operation, kernel, anchor?, iterations?, borderType?, borderValue?)`

Applies U8 erosion, dilation, opening, closing, gradient, top-hat, or black-hat using nonzero kernel lanes. The default centered anchor, one iteration, constant border, and morphology-neutral border values match the common OpenCV.js call shape. `MORPH_HITMISS` and wider depths remain reserved.

### `Sobel(source, destination, ddepth, dx, dy, ksize?, scale?, delta?, borderType?)`

Computes first or second 3x3 derivatives from a U8 source. Destinations can retain U8 or use I16, F32, or F64 so negative gradients are preserved. The package exposes the pinned `BORDER_*` constants used by all three neighborhood APIs.

### `Canny(source, destination, threshold1, threshold2, apertureSize?, l2Gradient?)`

Produces a single-channel U8 edge map from a single-channel U8 source. The current Rust pipeline uses 3x3 Sobel gradients, directional non-maximum suppression, ordered double thresholds, and eight-neighbor hysteresis. Set `l2Gradient` for Euclidean gradient magnitude; the default uses the faster L1 magnitude.

### `createMatVector()` and `findContours(source, contours, hierarchy, mode, method, offset?)`

`createMatVector()` allocates a Rust-owned collection with `size()`, `get(index)`, `push_back(mat)`, `clear()`, and explicit `dispose()`/`delete()` lifetime methods. Retrieved matrices retain shared Rust storage independently of the vector wrapper.

`findContours` currently extracts external connected-component boundaries from single-channel U8 masks. It supports `RETR_EXTERNAL` and `RETR_LIST`, `CHAIN_APPROX_NONE` and `CHAIN_APPROX_SIMPLE`, integer offsets, mutable vector replacement, and an I32 four-channel sibling hierarchy. Hole relationships, tree retrieval, flood-fill input, and Teh-Chin approximation remain reserved.

### `warpAffine(source, destination, transform, size, flags?, borderType?, borderValue?)`

Warps an interleaved U8 matrix through a 2×3 F32 or F64 affine transform into a mutable destination. The current Rust sampler supports `INTER_NEAREST` and `INTER_LINEAR`, `BORDER_CONSTANT` and `BORDER_REPLICATE`, per-channel scalar border values, and `WARP_INVERSE_MAP`. Without the inverse flag, the forward transform is inverted before sampling. Wider source depths, additional border modes, and exact interpolation variants remain reserved.

### `equalizeHist(source, destination)`

Equalizes a single-channel U8 matrix with a Rust-owned 256-bin histogram, cumulative distribution, and lookup table. Constant images are preserved and the mutable destination is replaced when its layout is incompatible. CLAHE remains a separate, unimplemented family.

### `resizeNearest(image, targetWidth, targetHeight)`

Resizes an RGBA image with nearest-neighbor sampling. Both target dimensions must be positive 32-bit integers.

### `resize(source, destination, size, fx?, fy?, interpolation?)`

Resizes a Rust-owned `Mat` into a mutable destination. A positive `size.width` and `size.height` set the output dimensions. Passing `{ width: 0, height: 0 }` derives them from positive finite `fx` and `fy` values.

The current slice implements:

- `INTER_NEAREST` across every matrix depth and channel count
- `INTER_LINEAR` for U8 matrices using OpenCV half-pixel coordinates and nearest-even output rounding
- `INTER_AREA` for shrinking U8 matrices using source-pixel coverage weights

Nearest-neighbor mode copies complete scalar bytes without numeric conversion, so signed and floating-point matrices preserve their stored bit patterns. The operation compacts strided sources, replaces incompatible destinations, and snapshots the source before an exact in-place resize.

The package exports all pinned interpolation constants now so TypeScript code can use one stable namespace. Cubic, Lanczos, exact modes, linear interpolation for wider depths, and area enlargement remain unimplemented and reject instead of silently selecting another algorithm.

### `cvtColor(source, destination, code, dstCn?)`

Converts a Rust-owned U8 `Mat` into a mutable destination. The current slice matches pinned OpenCV.js color codes 0 through 11:

- RGB/BGR and RGBA/BGRA channel-order changes
- Opaque alpha insertion and alpha removal
- RGB, BGR, RGBA, or BGRA to one-channel grayscale
- One-channel grayscale to three or four channels

`dstCn` defaults to the conversion code's channel count. Passing `3` or `4` selects that output count for color and grayscale-expansion codes. Grayscale output remains one channel. The operation replaces an incompatible destination, compacts strided sources, and supports exact in-place conversion.

The `COLOR_*` constants for these codes are available both as named package exports and on the initialized client. This remains a partial family because U16, F32, HSV/HLS, YUV, packed-color, Bayer, and later conversion codes are not implemented yet.

### U8 matrix operations

The following methods accept Rust-owned U8 `Mat` values and return a new Rust-owned `Mat` unless noted otherwise:

- `add(left, right)` and `subtract(left, right)` use unsigned saturation.
- `absdiff(left, right)`, `min(left, right)`, and `max(left, right)` operate element by element.
- `bitwiseAnd(left, right)`, `bitwiseOr(left, right)`, and `bitwiseXor(left, right)` operate on every byte.
- `compareEqual(left, right)` returns 255 for equal elements and 0 otherwise.
- `inRange(source, lowerBound, upperBound)` applies inclusive per-channel bounds and returns a one-channel 255/0 mask.

Multi-input operations require identical rows, columns, and channels. These methods are working U8 slices, not yet complete OpenCV.js families; masks, scalar operands, optional destinations, and other depth forms remain tracked by the parity ledger.

`bitwiseNot(source, destination)` and `bitwiseNot(source, destination, mask)` match the pinned OpenCV.js destination forms and return `undefined`. They invert every stored bit for all seven scalar depths and any channel count. A nonempty mask must be single-channel U8 or I8 with the source dimensions; every nonzero mask byte selects the corresponding pixel across all channels. Compatible destinations preserve unselected pixels, while fresh or replaced destinations initialize them to zero. `bitwiseNotAlloc(source)` is the package's allocating convenience.

The browser fixture covers raw floating-point bits, typed empties, strided matrices, destination replacement, exact aliases, live unmasked overlapping regions, masked overlaps, argument conversion order, invalid masks, and deleted handles.

### Matrix layout operations

- `flipAlloc(source, code)` allocates a flipped matrix. Zero flips rows, positive signed 32-bit codes flip columns, and negative codes flip both axes.
- `transposeAlloc(source)` swaps rows and columns and allocates a new matrix.
- `rotateAlloc(source, code)` accepts `0`, `1`, or `2` for 90 degrees clockwise, 180 degrees, or 90 degrees counterclockwise and allocates its output.
- `repeatAlloc(source, rowRepeats, columnRepeats)` tiles a matrix and allocates its output.

These operations preserve all seven scalar depths and every interleaved channel. They return new Rust-owned matrices and compact non-contiguous regions before rearranging pixels.

Each layout operation also accepts an OpenCV-style destination form: `flip(source, destination, code)`, `transpose(source, destination)`, `rotate(source, destination, code)`, and `repeat(source, rows, columns, destination)`. All four use the exact OpenCV.js argument counts. They reuse compatible destinations, reallocate incompatible ordinary matrices, and detach incompatible regions. Compatible regions write through to shared parent storage.

The browser differential harness passes pinned OpenCV.js 4.13.0 all-depth fixtures for all four layout operations, including empty and deleted matrices, destination replacement, detached incompatible regions, and live overlapping aliases. Repeat matches the upstream exact in-place rejection contract; flip, transpose, and rotate match their valid in-place forms.

### Matrix channels

- `split(source)` returns one single-channel matrix per input channel.
- `merge([first, second])`, `merge([first, second, third])`, and `merge([first, second, third, fourth])` interleave compatible inputs without invalidating them.
- `extractChannel(source, channel)` returns one selected channel.
- `insertChannel(source, destination, channel)` writes a single-channel source into one destination channel.
- `mixChannels(source, destination, fromTo)` routes flattened source/destination channel pairs into an existing matrix while preserving unmapped destination channels.
- `hconcat([first, second, ...])` and `vconcat([first, second, ...])` join two through four compatible matrices without invalidating their inputs.

Channel operations preserve raw scalar bytes for all seven depths and compact strided source regions. Insertions and channel routing into a destination region update its shared parent storage. The current `mixChannels` slice accepts one source and one destination; MatVector forms remain.

### Floating-point math

- `exp(source, destination)`, `log(source, destination)`, and `sqrt(source, destination)` write element-wise results into exact mutable destinations. `expAlloc`, `logAlloc`, and `sqrtAlloc` allocate package-convenience outputs.
- `pow(source, exponent, destination)` writes element-wise powers into an exact mutable destination. `powAlloc` is the allocating convenience form.
- `magnitude(x, y, destination)` writes vector lengths into an exact mutable destination. `magnitudeAlloc` is the allocating convenience form.
- `cartToPolar(x, y, magnitude, angle, degrees)` writes lengths and angles into exact mutable destinations.
- `polarToCart(magnitude, angle, x, y, degrees)` writes cartesian components into exact mutable destinations.

`exp`, `log`, `sqrt`, and `magnitude` accept F32 and F64 matrices. `pow` accepts every scalar depth for valid integer powers and F32/F64 for the complete F64 exponent domain. The exact forms replace incompatible destinations, write through compatible regions, and traverse overlapping shared storage in pinned row-major order. Typed empty headers retain their rows, columns, channels, and depth. The package safely rejects integer `sqrt` and non-integral or non-finite integer `pow` calls because the pinned artifact exposes unsafe or uninitialized output for those inputs.

`cartToPolar` and `polarToCart` match the pinned four- and five-argument overloads. They accept matching F32 or F64 matrices with any channel count and replace both destinations when their metadata differs. Compatible regions write through shared storage, including the reference runtime's live row-major behavior when one output overlaps an input. Typed 0xN, Nx0, and 0x0 layouts retain their shape, channels, and depth. The optional `degrees` value defaults to `false` and follows JavaScript truthiness. Cartesian angles use a nonnegative full rotation. Both accepted depths follow the pinned F32 computation path. The package rejects shared paired outputs and native integer or canonical-empty calls before mutation because the reference artifact exposes assertion or transient numeric failures for those cases.

### Typed numeric operations

- `multiply(a, b, destination, scale?, dtype?)` and `divide(a, b, destination, scale?, dtype?)` match the pinned three-, four-, and five-argument matrix overloads. `multiplyAlloc` and `divideAlloc` are allocating conveniences.
- `addWeighted(a, alpha, b, beta, gamma, destination, dtype?)` matches the pinned six- and seven-argument overloads. `addWeightedAlloc` is the allocating convenience.
- `convertScaleAbs(source, destination, alpha?, beta?)` matches the pinned two-, three-, and four-argument overloads. `convertScaleAbsAlloc` is the allocating convenience.

The mutable forms support all seven scalar depths, strided and overlapping regions, incompatible destination replacement, and explicit mixed-depth output conversion. Integer outputs use nearest-even conversion and saturation except CV_32S, whose non-saturating overflow follows the pinned runtime. Integer division by zero produces zero. Floating-point outputs retain IEEE 754 behavior.

### Matrix borders

`copyMakeBorder(source, top, bottom, left, right, borderType, constant)` supports constant, replicate, reflect, wrap, and reflect-101 modes, with the optional isolated bit. It preserves every scalar depth and channel layout. Constant values use nearest-even rounding and saturation for integer destinations.

### Lookup tables

`lut(source, table)` returns a transformed matrix. `lut(source, table, destination)` writes into an exact destination. Sources may use U8 or I8 elements. The table contains exactly 256 pixels with one channel or the same channel count as the source. Tables and outputs may use any scalar depth.

### Norms and normalization

`norm` supports one matrix or a matrix difference, optional U8 masks, numeric norms, Hamming norms, and relative flags. `normalize(source, destination, alpha, beta, type, mask)` writes INF, L1, L2, or MINMAX normalization into a caller-owned matrix. It converts between all scalar depths and preserves unselected destination pixels when a mask is present.

### Statistics and dimensional reduction

`meanStdDev(source, means, standardDeviations, mask)` writes one F64 mean and population standard deviation per channel. `reduce(source, destination, axis, kind)` reduces rows or columns with sum, average, maximum, or minimum. Reduction converts between every scalar depth and uses nearest-even saturated integer output.

### Matrix initialization and random fills

- `setIdentity(destination, value?)` clears a matrix and writes the four-lane Scalar on its diagonal in place. The default is `[1, 0, 0, 0]`; all shipped depths, one through four channels, typed empties, and regions of interest are supported.
- `randu(destination, lower, upper)` fills each channel from a half-open uniform range.
- `randn(destination, mean, standardDeviation)` fills each channel from a normal distribution.
- `setRNGSeed(seed)` resets the random stream from a signed 32-bit seed.

These methods mutate existing matrices and support all seven scalar depths, strided regions, and one through four channels. Uniform integer draws use floor conversion. Normal integer draws and identity values use nearest-even rounding and saturation. `randn` accepts channel-specific standard deviations but not a covariance matrix.

The random functions use an independently authored SplitMix64 stream and Box-Muller normal sampler. Resetting a seed reproduces package results. It does not reproduce OpenCV's random sequence, so these families remain partial.

### Core runtime utilities

- `getLogLevel()` returns the package-owned log severity for the current WebAssembly instance.
- `setLogLevel(level)` updates that severity and returns the previous level.
- `getOptimalDFTSize(size)` returns the smallest integer at least as large as `size` whose only prime factors are 2, 3, and 5.

`LogLevel` is the integer union `0 | 1 | 2 | 3 | 4 | 5 | 6`, representing silent, fatal, error, warning, informational, debug, and verbose logging. The initial level is warning (`3`). Invalid levels throw `OpenCvInputError`. Logging state belongs to this package and one WebAssembly instance; it does not configure an installed OpenCV runtime.

OpenCV's 4.13.0 JavaScript binding configuration lists `getLogLevel` and `setLogLevel`, but the official documentation artifact used by the browser differential harness does not expose them at runtime. Their package behavior is covered by Rust and TypeScript tests; direct upstream runtime comparison is therefore unavailable for that pinned artifact.

`getOptimalDFTSize` requires exactly one argument; missing or extra arguments throw `BindingError`. Its plain-JavaScript boundary matches Embind signed i32 conversion: numeric fractions truncate toward zero, `NaN` becomes zero, and booleans become `1` or `0`. Numbers outside the signed 32-bit range, infinities, strings, `null`, and `undefined` throw `TypeError`.

After conversion, the function returns `-1` for negative values, `1` for zero and one, and otherwise returns the next 2-, 3-, and 5-smooth size. OpenCV.js treats `2,125,764,000` as an exclusive upper sentinel: `2,125,763,999` maps to it, while the sentinel itself and larger signed i32 inputs return `-1`. The pinned browser matrix verifies the call contract and boundary values, while exhaustive Rust tests verify minimality and every representable smooth-size transition.

### Per-element transforms

`transform(source, coefficients)` allocates a result. `transform(source, coefficients, destination)` writes into an exact destination. The source may use any scalar depth and one through four channels. Coefficients must be a single-channel F32 or F64 matrix with one row per output channel. It may contain one column per input channel for a linear transform or one extra column for an affine bias. The output keeps the source depth and may have a different channel count.

`perspectiveTransform(source, coefficients)` and its destination overload transform interleaved two- or three-component vectors. The source must use F32 or F64. Coefficients must be a single-channel square F32 or F64 matrix with one extra homogeneous row and column. The output keeps the source shape, channels, and depth.

Both methods compact strided inputs and snapshot inputs before destination writes. Differential fixtures against the pinned browser build remain.

### Contour geometry

```ts
interface Point {
  readonly x: number;
  readonly y: number;
}

interface Size {
  readonly height: number;
  readonly width: number;
}

interface Rect extends Size, Point {}

arcLength(contour: Mat, closed: boolean): number;
contourArea(contour: Mat, oriented?: boolean): number;
boundingRect(contour: Mat): Rect;
isContourConvex(contour: Mat): boolean;
pointPolygonTest(contour: Mat, point: Point, measureDistance: boolean): number;
```

`arcLength`, `contourArea`, and `boundingRect` accept I32 and F32 contours stored as `Nx1C2`, `1xNC2`, or `Nx2C1`. They reject F64, U8, invalid shapes, deleted inputs, non-finite coordinates, and numeric overflow. They read strided regions through their logical bytes. Other curve containers and points with more than two dimensions are not supported.

`arcLength` accepts exactly two arguments. `contourArea` has a JavaScript runtime length of zero and accepts one or two arguments. `boundingRect` accepts exactly one argument. The optional flags use JavaScript truthiness. `arcLength` measures an open or closed perimeter. `contourArea` returns unsigned area by default and signed, oriented area when `oriented` is truthy. Fewer than three points have zero area. `boundingRect` floors fractional coordinates and returns inclusive integer bounds, so one integer point produces a 1-by-1 rectangle. A canonical empty `Mat` has zero bounds. The package rejects typed empty contours before entering upstream paths that do not return a safe JavaScript error.

`isContourConvex` accepts exactly one argument and requires a continuous I32 or F32 contour. Every turn must be nonzero and have the same direction. Fewer than three points, collinear edge points, duplicate vertices, concavity, and self-intersection return false.

`pointPolygonTest` accepts exactly three arguments and the same continuous contour layouts. One-point and two-point contours are valid. The point may be any structural Point2f object with `x` and `y` fields. The binding narrows both fields to float32 and uses JavaScript truthiness for `measureDistance`. Classification returns `1` inside, `-1` outside, and zero on an edge. Distance mode returns the signed nearest-boundary distance and preserves the pinned traversal-dependent sign of zero. Non-finite query coordinates use the pinned browser sentinel behavior. Empty, deleted, F64, U8, invalid-shape, and non-contiguous contours are rejected before native computation.

### Image-processing helpers

```ts
type StructuringElementKind = 0 | 1 | 2 | 3;
type HanningWindowDepth = "f32" | "f64";

getStructuringElement(kind: StructuringElementKind, size: Size, anchor?: Point): Mat;
createHanningWindow(destination: Mat, size: Size, type: 5 | 6): void;
createHanningWindowAlloc(size: Size, depth: HanningWindowDepth): Mat;
ellipse2Poly(
  center: Point,
  axes: Size,
  rotationDegrees: number,
  arcStart: number,
  arcEnd: number,
  delta: number,
): Point[];
clipLine(rectangle: Rect, start: Point, end: Point): readonly [Point, Point] | undefined;
```

`getStructuringElement` returns a single-channel U8 kernel. Kind `0` is a rectangle, `1` is a cross, `2` is an ellipse, and `3` is a diamond. Width and height must be positive after Embind signed-integer conversion. The default anchor is `{ x: -1, y: -1 }`, which selects each dimension's center. A custom anchor moves the cross intersection. Rectangle, ellipse, and diamond geometry stay centered.

`createHanningWindow` accepts exactly three arguments and writes a single-channel F32 (`5`) or F64 (`6`) matrix into the supplied destination. Size is a structural value object whose fields use Embind signed i32 conversion, and each converted dimension must be at least two. Compatible regions are written through; other destinations are rebound. The pinned arithmetic takes the square root after multiplying the horizontal and vertical Hann coefficients, preserving asymmetric F64 rounding. `createHanningWindowAlloc` is the package-specific allocating convenience.

`ellipse2Poly` accepts non-negative integer axes. Arc bounds must satisfy `0 <= arcStart <= arcEnd <= 360`, and `delta` must be from 1 through 180. It always samples the exact end angle and removes consecutive points that round to the same signed 32-bit coordinate. It does not normalize or swap arc bounds.

`clipLine` accepts a positive-size integer rectangle and signed 32-bit segment coordinates. It clips against inclusive pixel bounds. A visible segment returns two points; a disjoint segment returns `undefined`. Unlike the OpenCV call shape, this package returns new points instead of mutating caller-owned point objects. Rectangle right and bottom edges must fit signed 32-bit coordinates.

### Transform matrix constructors

```ts
getRotationMatrix2D(center: Point, angleDegrees: number, scale: number): Mat;
getAffineTransform(source: Mat, destination: Mat): Mat;
invertAffineTransform(transform: Mat, destination: Mat): void;
invertAffineTransformAlloc(transform: Mat): Mat;
getPerspectiveTransform(source: Mat, destination: Mat): Mat;
```

`getRotationMatrix2D` accepts exactly three arguments. The center may be any structural Point2f object with `x` and `y` fields. The binding checks those fields in order and narrows them to float32. Angle and scale use strict Embind double conversion; numbers and booleans are accepted, while strings, boxed numbers, and generic coercion objects are rejected. Signed zero, `NaN`, and infinities propagate to a bit-exact `2x3C1` F64 result. Every call allocates an independent matrix.

`getAffineTransform` reads three source and destination points from continuous F32 matrices. Each point set may be `3x2C1`, `3x1C2`, or `1x3C2`. It returns a fresh 2x3 single-channel F64 matrix. Collinear source points produce six zeros, and non-finite coordinates propagate through the result.

`invertAffineTransform` accepts exactly two Mat arguments. The source must be `2x3C1` at F32 or F64 depth and may be a strided region. The destination is replaced when its layout differs and written through when it is a compatible region. Exact in-place operation is supported. Output depth and arithmetic match the source. Singular matrices produce the pinned signed-zero coefficients, while NaN and infinity propagate through the observed arithmetic. `invertAffineTransformAlloc` is the package-specific allocating convenience.

`getPerspectiveTransform` reads four source and destination points. Each point set may be `4x2C1`, `4x1C2`, or `1x4C2` at F32 or F64 depth. Strided regions are supported. It uses one scaled partial-pivoting solver, fixes the lower-right output coefficient to one, and returns a `3x3C1` F64 matrix. It rejects non-finite values, degenerate point configurations, and transforms that cannot use that normalization.

The three `get*Transform` constructors allocate their results. `invertAffineTransform` uses the upstream mutable destination contract, with an allocating package convenience. `getRotationMatrix2D`, `getAffineTransform`, and `invertAffineTransform` pass their complete pinned browser contracts. Complete differential fixtures remain for `getPerspectiveTransform`.

The pinned OpenCV.js 4.13.0 browser fixture passes the complete audited contracts for all five contour methods, `getRotationMatrix2D`, `getAffineTransform`, `invertAffineTransform`, and `getStructuringElement`. The audits cover exact overloads, structural value-object conversion, numeric edge behavior, mutable outputs, aliasing, source preservation, and result metadata.

### Dense matrix algebra

- `determinant(source)` requires exactly one live `Mat` and returns a JavaScript number. The source must be a nonempty square single-channel F32 or F64 matrix. Compact matrices and non-contiguous regions are accepted, and the call does not change the source or its parent allocation. Integer depths, multiple channels, nonsquare matrices, empty matrices, deleted handles, and non-Mat values throw.
- `invert(source, destination, method)` writes an inverse into an exact single-channel F32 or F64 destination. It returns `1` on success and `0` for a singular or rank-deficient source.
- `solve(coefficients, rightHandSides, destination, method)` solves one or more right-hand sides. It returns `false` on rank loss.

For orders one through three, `determinant` follows the pinned direct formulas. F32 inputs use their stored float32 values widened for those products. This preserves the reference runtime's signed-zero and non-finite results. For orders four and larger, F32 elimination stays in float32 while F64 elimination stays in float64. Both use partial pivoting and accumulate the returned product as F64. A pivot below `10 * 2^-23` for F32 or `100 * Number.EPSILON` for F64 returns positive zero; a pivot exactly at the cutoff remains valid. The browser audit covers both cutoff boundaries, row-swap signs, singular matrices, and Hilbert precision.

The decomposition method defaults to `0` for LU. Method `3` selects Cholesky and method `4` selects QR. LU and Cholesky require square coefficient matrices. QR also accepts overdetermined systems. SVD, eigen, normal-equation flags, pseudo-inverses, and underdetermined systems remain. Failed inverse and solve calls leave the destination unchanged. Inputs must be single-channel and finite. Inverse and solve destinations must use F32 or F64.

### Matrix reductions

- `countNonZero(source)` supports every scalar depth and requires one channel.
- `sum(source)` returns a four-number scalar and supports up to four channels. It is an extra convenience beyond the pinned 488-family browser ledger.
- `mean(source)` and `mean(source, mask)` return a four-number scalar for one through four channels. A U8 single-channel mask selects pixels by non-zero values.
- `minMaxLoc(source)` and `minMaxLoc(source, mask)` return extrema with their first row-major coordinates for a single-channel source. An all-zero mask returns zero extrema at `(-1, -1)`.
- `trace(source)` returns four diagonal-sum lanes for one through four channels.

Reducers compact non-contiguous regions before decoding values. `countNonZero` returns zero for an empty matrix, excludes both signs of zero, and counts NaN, infinities, and subnormal values as non-zero. Floating-point sums, means, and traces propagate NaN. `minMaxLoc` skips NaN; an all-NaN selection returns zero extrema at `(-1, -1)`. A fresh canonical empty header uses `(0, 0)`, while typed empty layouts use `(-1, -1)`.

The pinned browser fixture verifies the complete `mean` and `minMaxLoc` contracts, including optional masks, every scalar depth, channel limits, empty headers, compact and strided inputs, tie locations, floating-point edge values, argument errors, and deleted handles.

The same fixture verifies the complete `trace` contract across every scalar depth, one through four channels, rectangular and strided inputs, typed empties, exact accumulation order, floating-point edge values, argument errors, and deleted handles.

### AKAZE configuration

`cv.createAKAZE(options)` allocates a Rust-owned AKAZE configuration handle. The current slice stores configuration only. Feature detection, descriptor extraction, image input, and keypoint output are not implemented yet.

```ts
import { AKAZEDescriptorType, KAZEDiffusivity } from "wasmosaic";

const akaze = cv.createAKAZE({
  descriptorChannels: 3,
  descriptorSize: 0,
  descriptorType: AKAZEDescriptorType.MLDB,
  diffusivity: KAZEDiffusivity.PM_G2,
  maxPoints: -1,
  octaveLayers: 4,
  octaves: 4,
  threshold: 0.001,
});

try {
  akaze.setDescriptorType(cv.AKAZE_DescriptorType.DESCRIPTOR_MLDB_UPRIGHT);
  akaze.setDiffusivity(cv.KAZE_DiffusivityType.DIFF_WEICKERT);
  akaze.setThreshold(0.002);
  console.log(akaze.getDescriptorType(), akaze.getDiffusivity(), akaze.getThreshold());
} finally {
  akaze.dispose();
}
```

Omitting every option uses the OpenCV 4.13 defaults shown above. Factory options are deliberately stricter than instance mutation. Descriptor types accept KAZE upright, KAZE, MLDB upright, or MLDB. Descriptor channels range from 1 through 3, descriptor size is non-negative, octave counts are positive signed 32-bit integers, and thresholds are finite and non-negative. Diffusivity accepts PM G1, PM G2, Weickert, or Charbonnier. `maxPoints` accepts a signed 32-bit integer but has no getter or setter in the pinned AKAZE inventory.

The instance setters reproduce the pinned Embind boundary. Descriptor-channel, descriptor-size, octave, and octave-layer setters coerce numbers to signed i32 and store the complete signed range. The threshold setter preserves the complete F64 domain, including negative zero, `NaN`, and both infinities. Descriptor type and diffusivity setters read a structural object's `value` property and apply JavaScript signed 32-bit conversion. They accept canonical members, foreign enum members, inherited properties, and plain objects. Raw unknown codes remain in Rust-owned state; the matching getter returns `undefined` when no canonical singleton has that code. Missing or extra method arguments throw `BindingError`; unsupported scalar input and integer range failures throw `TypeError`. `getDefaultName()` returns `"Feature2D.AKAZE"`.

The instance exposes OpenCV.js-compatible `delete()` and idempotent `dispose()`. `delete()` releases the handle and a repeated call throws `BindingError`. Every getter or setter after either release path throws `BindingError`.

The pinned OpenCV.js 4.13.0 browser fixture exposes the direct `AKAZE` constructor and all 15 instance methods. The complete compatibility matrix passes for every instance method. It covers exact arity, defaults, return values, scalar coercion, enum namespace descriptors, canonical singleton identity, structural enum inputs, raw unknown wire values, argument errors, and lifetime behavior. Enum getters use const-pointer deletion errors; setters use mutable-pointer deletion errors. The artifact omits the config-listed static `AKAZE.create`, so the package factory has no direct runtime comparator for that baseline.

Call `dispose()` when the handle is no longer needed. Repeated disposal does nothing. Any getter or setter after disposal throws `BindingError`.

### KAZE configuration

`cv.createKAZE(options)` allocates a Rust-owned KAZE configuration handle. It stores detector and descriptor settings but does not accept images, detect keypoints, or compute descriptors yet.

```ts
import { KAZEDiffusivity } from "wasmosaic";

const kaze = cv.createKAZE({
  diffusivity: KAZEDiffusivity.PM_G2,
  extended: false,
  octaveLayers: 4,
  octaves: 4,
  threshold: 0.001,
  upright: false,
});

try {
  kaze.setThreshold(-1);
  kaze.setDiffusivity(cv.KAZE_DiffusivityType.DIFF_CHARBONNIER);
  console.log(kaze.getThreshold(), kaze.getDiffusivity());
} finally {
  kaze.dispose();
}
```

The OpenCV 4.13 defaults are `extended: false`, `upright: false`, threshold `0.0010000000474974513`, four octaves, four octave layers, and `KAZEDiffusivity.PM_G2`. Factory options are deliberately stricter than instance mutation: octave counts must be positive signed 32-bit integers, the threshold must be finite, and diffusivity accepts the typed PM G1, PM G2, Weickert, and Charbonnier enum members.

The instance exposes `getDefaultName()`, getters and setters for every option, OpenCV.js-compatible `delete()`, and idempotent `dispose()`. `getDefaultName()` returns `"Feature2D.KAZE"`. Octave setters coerce numbers to signed i32, threshold preserves the complete F64 domain including negative zero, `NaN`, and both infinities, and boolean setters use JavaScript truthiness. The diffusivity setter reads a structural object's `value` property and stores its signed i32 conversion without validating the code. A raw unknown code makes the getter return `undefined`; known codes return the same canonical singleton used by AKAZE. Missing or extra method arguments throw `BindingError`; unsupported scalar input and integer range failures throw `TypeError`.

`delete()` releases the handle and a repeated call throws `BindingError`, matching OpenCV.js. `dispose()` is the package convenience for idempotent cleanup. Every getter or setter after either release path throws `BindingError`.

The pinned OpenCV.js 4.13.0 browser fixture exposes the direct `KAZE` constructor and all 13 instance methods. The complete compatibility matrix passes for every instance method. It covers exact arity, defaults, return values, scalar coercion, shared enum singleton identity, structural enum inputs, raw unknown wire values, argument errors, and const-getter versus mutable-setter lifetime behavior. The artifact omits the config-listed static `KAZE.create`, so the package factory has no direct static-factory comparator for that baseline.

### MSER configuration

`cv.createMSER(options)` allocates a Rust-owned configuration handle. Defaults are delta `5`, minimum area `60`, maximum area `14400`, and second-pass-only disabled.

```ts
const mser = cv.createMSER({ delta: 8, minArea: 40, maxArea: 20_000 });

try {
  mser.setPass2Only(true);
  console.log(mser.getDelta(), mser.getPass2Only());
} finally {
  mser.dispose();
}
```

The nine implemented methods reproduce the pinned constructor state, exact method arity, signed i32 conversion, JavaScript boolean coercion, undefined setter returns, and deleted-object errors. Factory integer options are strictly validated before entering WASM. The pinned artifact omits the config-listed static `MSER.create`; the package factory is a convenience. Region detection and its output-vector types remain unimplemented.

### ORB configuration

`cv.createORB(options)` allocates a Rust-owned configuration handle. Its defaults are 500 maximum features, float32 `1.2` scale factor, eight levels, edge threshold `31`, first level `0`, WTA_K `2`, Harris scoring, patch size `31`, and FAST threshold `20`.

```ts
import { ORBScoreType } from "wasmosaic";

const orb = cv.createORB({ maxFeatures: 1000, scoreType: ORBScoreType.FAST_SCORE });

try {
  orb.setFastThreshold(12);
  orb.setScoreType(cv.ORB_ScoreType.HARRIS_SCORE);
  console.log(orb.getFastThreshold());
} finally {
  orb.dispose();
}
```

The eleven implemented methods reproduce exact arity, signed i32 and F64 input conversion, float32 scale storage, structural score-enum conversion, first-level validation, undefined setter returns, and const-versus-mutable deleted-object errors. The pinned artifact omits the config-listed static `ORB.create`; the package factory is a convenience. Image detection, keypoints, and descriptors remain unimplemented.

### AGAST and FAST configuration

`cv.createAgastFeatureDetector(options)` and `cv.createFastFeatureDetector(options)` allocate Rust-owned detector configuration handles. They do not accept images or detect keypoints yet.

```ts
import { AgastFeatureDetectorType, FastFeatureDetectorType } from "wasmosaic";

const agast = cv.createAgastFeatureDetector({
  nonmaxSuppression: true,
  threshold: 10,
  type: AgastFeatureDetectorType.OAST_9_16,
});
const fast = cv.createFastFeatureDetector({
  nonmaxSuppression: true,
  threshold: 10,
  type: FastFeatureDetectorType.TYPE_9_16,
});

try {
  agast.setThreshold(-1);
  agast.setType(cv.AgastFeatureDetector_DetectorType.AGAST_5_8);
  fast.setThreshold(256);
  fast.setType(cv.FastFeatureDetector_DetectorType.TYPE_5_8);
} finally {
  agast.dispose();
  fast.dispose();
}
```

AGAST uses these numeric neighborhood types:

| Enum member                            | Value |
| -------------------------------------- | ----: |
| `AgastFeatureDetectorType.AGAST_5_8`   |     0 |
| `AgastFeatureDetectorType.AGAST_7_12d` |     1 |
| `AgastFeatureDetectorType.AGAST_7_12s` |     2 |
| `AgastFeatureDetectorType.OAST_9_16`   |     3 |

FAST uses these numeric neighborhood types:

| Enum member                         | Value |
| ----------------------------------- | ----: |
| `FastFeatureDetectorType.TYPE_5_8`  |     0 |
| `FastFeatureDetectorType.TYPE_7_12` |     1 |
| `FastFeatureDetectorType.TYPE_9_16` |     2 |

Both factories default to threshold `10` with non-maximum suppression enabled. AGAST defaults to `OAST_9_16`, and FAST defaults to `TYPE_9_16`. `getDefaultName()` returns `"Feature2D.AgastFeatureDetector"` or `"Feature2D.FastFeatureDetector"`.

`setThreshold(value)` follows the pinned binding's signed i32 coercion and returns `undefined`. Fractional values truncate toward zero, and `NaN` becomes zero. The browser differential fixture also confirms that AGAST preserves `-1` and FAST preserves `256`. `setNonmaxSuppression(value)` applies JavaScript boolean coercion and returns `undefined`.

The five primitive methods on each class, `getDefaultName`, `getNonmaxSuppression`, `getThreshold`, `setNonmaxSuppression`, and `setThreshold`, enforce the pinned argument counts. Missing setter arguments and extra getter or setter arguments throw `BindingError`. `delete()` returns `undefined`; a second deletion and every primitive method call after deletion also throw `BindingError`. The package keeps idempotent `dispose()` as a convenience.

`getType()` returns the canonical Embind-compatible singleton for known codes. `setType(value)` reads a structural object's `value` property and stores its JavaScript signed 32-bit conversion. This accepts canonical or foreign enum members, inherited properties, plain objects, fractions, non-finite numbers, strings, and booleans with the same observable conversion as the pinned binding. Unknown raw codes remain in Rust-owned state and make the getter return `undefined`. Nullish inputs and throwing property getters propagate the matching errors without mutating state.

The pinned OpenCV.js 4.13.0 artifact exposes direct `AgastFeatureDetector` and `FastFeatureDetector` constructors and all seven instance methods on both. Every instance method passes the complete call-contract matrix, including enum namespace descriptors, canonical singleton identity, structural setter coercion, raw unknown wire values, exact arity, and const-getter versus mutable-setter lifetime errors. The factories remain partial because the artifact omits both config-listed static `create` methods.

### GFTT detector configuration

`cv.createGFTTDetector(options)` allocates a Rust-owned good-features-to-track detector configuration. This slice stores settings only. It does not accept images or detect keypoints.

```ts
const detector = cv.createGFTTDetector({
  blockSize: 3,
  k: 0.04,
  maxFeatures: 1_000,
  minDistance: 1,
  qualityLevel: 0.01,
  useHarrisDetector: false,
});

try {
  detector.setMaxFeatures(-1);
  detector.setQualityLevel(Number.POSITIVE_INFINITY);
  console.log(detector.getMaxFeatures(), detector.getQualityLevel());
} finally {
  detector.dispose();
}
```

Omitting options uses the exact OpenCV 4.13 defaults shown above. Factory options remain strictly typed and validate `maxFeatures` and `blockSize` as signed 32-bit integers.

The 13 instance methods match the pinned OpenCV.js binding contract. Every getter requires zero arguments, and every setter requires exactly one. Missing or extra arguments throw `BindingError`. `getDefaultName()` returns `"Feature2D.GFTTDetector"`.

The integer setters follow Embind's JavaScript coercion behavior. Numeric fractions truncate to signed i32, `NaN` becomes `0`, and booleans become `1` or `0`. Numbers outside the signed 32-bit range, infinity, strings, `null`, and `undefined` throw `TypeError`. The F64 setters preserve every number, including `NaN` and both infinities, and convert booleans to `1` or `0`. Other F64 inputs throw `TypeError`. `setHarrisDetector()` applies JavaScript truthiness through `Boolean(value)`.

Each setter returns `undefined`. Call `delete()` for OpenCV.js-compatible ownership. The first call returns `undefined`; a repeated call throws `BindingError`. Getters and setters also throw `BindingError` after deletion, with the pinned mutable or const pointer message. The package also keeps `dispose()` as an idempotent convenience.

The pinned OpenCV.js 4.13.0 fixture exposes the direct `GFTTDetector` constructor and all 13 instance methods. The browser matrix verifies method arity, defaults, return values, scalar coercion, missing and extra arguments, deletion, repeat deletion, and calls after deletion. All 13 instance methods count as implemented. The artifact omits the config-listed static `GFTTDetector.create` method. The package factory remains partial because it supports one six-argument shape and omits the `gradientSize` overload.

### Tone-map configuration

The three package factories allocate Rust-owned configuration handles with the pinned constructor argument order:

```ts
const drago = cv.createTonemapDrago(1, 1, 0.85);
const mantiuk = cv.createTonemapMantiuk(1, 0.7, 1);
const reinhard = cv.createTonemapReinhard(1, 0, 1, 0);

try {
  drago.setGamma(1.2);
  mantiuk.setScale(0.8);
  reinhard.setIntensity(-1);
} finally {
  drago.dispose();
  mantiuk.dispose();
  reinhard.dispose();
}
```

All three concrete types inherit exact `getGamma()` and `setGamma(value)` behavior. Drago adds bias and saturation getters and setters. Mantiuk adds scale and saturation getters and setters. Reinhard adds intensity, light-adaptation, and color-adaptation getters and setters. Constructor and setter inputs accept JavaScript numbers and booleans at the untyped binding boundary, narrow to float32, and preserve signed zero, `NaN`, and infinities. Missing or extra arguments throw `BindingError`; unsupported scalar inputs throw `TypeError` without mutation.

`delete()` matches the pinned repeated-delete errors, while `dispose()` is idempotent. Calls after release distinguish inherited `Tonemap const*` or `Tonemap` pointers from concrete const or mutable pointers. The pinned browser artifact exposes the three concrete constructors but omits the config-listed global `createTonemap*` functions, so the package factories remain partial. Pixel processing is not implemented or claimed yet.

## Errors

The TypeScript boundary throws `OpenCvInputError` for invalid dimensions, byte lengths, thresholds, and strictly validated factory options. Audited detector instance methods throw `BindingError` for argument-count and deleted-object failures, matching the pinned browser binding. Their scalar and enum-object boundary failures throw `TypeError` where the pinned binding does. Rust rejects invalid dimensions and byte lengths if a caller bypasses the TypeScript client.

## Matrices

### `cv.matFromU8(rows, columns, channels, data)`

Validates metadata once, copies the input into Rust-owned WASM memory, and returns a `Mat` with 1 through 512 interleaved channels.

Typed variants are `matFromI8`, `matFromU16`, `matFromI16`, `matFromI32`, `matFromF32`, and `matFromF64`. Each accepts the corresponding JavaScript typed array. Matrix depth codes follow OpenCV's scalar-depth order from U8 through F64.

### `cv.zerosU8(rows, columns, channels)`

Allocates a zero-filled matrix directly in Rust-owned WASM memory.

Typed variants are `zerosI8`, `zerosU16`, `zerosI16`, `zerosI32`, `zerosF32`, and `zerosF64`.

### `matrix.roi(row, column, rows, columns)`

Returns a matrix that shares its parent's Rust allocation. Creating a region does not copy pixels. A multi-row region may have a `rowStride` larger than its logical row width. `isContinuous` reports whether the logical bytes occupy one range.

### `matrix.toUint8Array()`

Copies logical bytes into compact JavaScript memory. For a U8 matrix those bytes are elements. For every other depth this method returns the compact raw byte representation. Strided regions omit bytes outside the region.

Typed element copies are available through `toInt8Array`, `toUint16Array`, `toInt16Array`, `toInt32Array`, `toFloat32Array`, and `toFloat64Array`. Calling a typed accessor that does not match the matrix depth throws.

### `matrix.copyFromBytes(data)`

Replaces the logical raw bytes of a matrix. The input length must equal `matrix.byteLength`. Writes are atomic and respect region stride, so overlapping regions and their parent observe the new values.

### `matrix.dispose()`

Releases the WASM handle. Calling it more than once is safe. Each region owns a separate handle, so disposing a parent does not invalidate an existing region. Accessing a disposed handle throws `OpenCvInputError`.
