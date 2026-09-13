import type {
  AKAZE,
  AKAZEOptions,
  AKAZE_DescriptorTypeNamespace,
  KAZE_DiffusivityTypeNamespace,
  WasmAKAZEFactory,
} from "./akaze.js";
import type { KAZE, KAZEOptions, WasmKAZEFactory } from "./kaze.js";
import type { ORB, ORBOptions, ORB_ScoreTypeNamespace, WasmORBFactory } from "./orb.js";
import type { GFTTDetector, GFTTDetectorOptions, WasmGFTTDetectorFactory } from "./gftt.js";
import type { MSER, MSEROptions, WasmMSERFactory } from "./mser.js";
import type {
  TonemapDrago,
  TonemapMantiuk,
  TonemapReinhard,
  WasmTonemapDragoFactory,
  WasmTonemapMantiukFactory,
  WasmTonemapReinhardFactory,
} from "./tonemap.js";
import type {
  AgastFeatureDetector,
  AgastFeatureDetector_DetectorTypeNamespace,
  AgastFeatureDetectorOptions,
  FastFeatureDetector,
  FastFeatureDetector_DetectorTypeNamespace,
  FastFeatureDetectorOptions,
  WasmAgastFeatureDetectorFactory,
  WasmFastFeatureDetectorFactory,
} from "./feature-detectors.js";
import type { Mat, WasmMatHandle } from "./mat.js";
import type { MatVector, WasmMatVectorHandle } from "./mat-vector.js";
import type { ColorConversionCode } from "./color.js";
import type { Interpolation } from "./interpolation.js";
import type { TemplateMatchMode } from "./template-matching.js";

/** An RGBA image whose data contains four bytes per pixel. */
export interface RgbaImage {
  readonly data: Uint8Array;
  readonly height: number;
  readonly width: number;
}

/** Four-channel scalar result used by OpenCV reductions. */
export type Scalar = readonly [number, number, number, number];

/** OpenCV-compatible border mode, optionally combined with the isolated bit. */
export type BorderType = 0 | 1 | 2 | 3 | 4 | 16 | 17 | 18 | 19 | 20;

/** Supported OpenCV numeric, binary, and relative norm flags. */
export type NormType = 1 | 2 | 4 | 5 | 6 | 7 | 9 | 10 | 12 | 13;

/** Norm modes accepted by normalization. */
export type NormalizeType = 1 | 2 | 4 | 32;

/** OpenCV reduce mode: sum, average, maximum, or minimum. */
export type ReduceKind = 0 | 1 | 2 | 3;

/** Implemented dense decomposition methods: LU, Cholesky, or QR. */
export type DecompositionMethod = 0 | 3 | 4;

/** Zero-based matrix coordinate. */
export interface Point {
  readonly x: number;
  readonly y: number;
}

/** Positive two-dimensional extent. */
export interface Size {
  readonly height: number;
  readonly width: number;
}

/** Integer rectangle with an inclusive pixel extent. */
export interface Rect extends Size, Point {}

/** Structuring-element kind: rectangle, cross, ellipse, or diamond. */
export type StructuringElementKind = 0 | 1 | 2 | 3;

/** Floating-point depths supported by Hanning windows. */
export type HanningWindowDepth = "f32" | "f64";

/** OpenCV matrix type code accepted by `createHanningWindow`. */
export type HanningWindowType = 5 | 6;

/** OpenCV-compatible log severity from silent through verbose. */
export type LogLevel = 0 | 1 | 2 | 3 | 4 | 5 | 6;

/** Extrema and first row-major locations returned by `minMaxLoc`. */
export interface MinMaxLocation {
  readonly maxLoc: Point;
  readonly maxVal: number;
  readonly minLoc: Point;
  readonly minVal: number;
}

/** Low-level contract implemented by the generated WebAssembly module. */
export interface OpenCvBackend {
  matMatchTemplateInto(
    image: WasmMatHandle,
    template: WasmMatHandle,
    result: WasmMatHandle,
    method: number,
  ): void;
  matMatchTemplateMaskedInto(
    image: WasmMatHandle,
    template: WasmMatHandle,
    result: WasmMatHandle,
    method: number,
    mask: WasmMatHandle,
  ): void;
  matEqualizeHistInto(source: WasmMatHandle, destination: WasmMatHandle): void;
  matWarpAffineInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    transform: WasmMatHandle,
    width: number,
    height: number,
    flags: number,
    borderType: number,
    borderValue: Float64Array,
  ): void;
  matVectorNew(): WasmMatVectorHandle;
  matFindContoursInto(
    source: WasmMatHandle,
    contours: WasmMatVectorHandle,
    hierarchy: WasmMatHandle,
    mode: number,
    method: number,
    offsetX: number,
    offsetY: number,
  ): void;
  matCannyInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    threshold1: number,
    threshold2: number,
    apertureSize: number,
    l2Gradient: boolean,
  ): void;
  readonly AgastFeatureDetector: WasmAgastFeatureDetectorFactory;
  readonly AKAZE: WasmAKAZEFactory;
  readonly FastFeatureDetector: WasmFastFeatureDetectorFactory;
  readonly GFTTDetector: WasmGFTTDetectorFactory;
  readonly KAZE: WasmKAZEFactory;
  readonly ORB: WasmORBFactory;
  readonly MSERConfig: WasmMSERFactory;
  readonly TonemapDrago: WasmTonemapDragoFactory;
  readonly TonemapMantiuk: WasmTonemapMantiukFactory;
  readonly TonemapReinhard: WasmTonemapReinhardFactory;
  clipLine(
    rectangleX: number,
    rectangleY: number,
    rectangleWidth: number,
    rectangleHeight: number,
    startX: number,
    startY: number,
    endX: number,
    endY: number,
  ): Int32Array;
  createHanningWindow(columns: number, rows: number, depth: number): WasmMatHandle;
  createHanningWindowInto(
    destination: WasmMatHandle,
    columns: number,
    rows: number,
    depth: number,
  ): void;
  ellipse2Poly(
    centerX: number,
    centerY: number,
    axisX: number,
    axisY: number,
    rotationDegrees: number,
    arcStart: number,
    arcEnd: number,
    delta: number,
  ): Int32Array;
  getStructuringElement(
    kind: number,
    columns: number,
    rows: number,
    anchorX: number,
    anchorY: number,
  ): WasmMatHandle;
  getLogLevel(): number;
  getOptimalDFTSize(size: number): number;
  grayscaleRgba(data: Uint8Array, width: number, height: number): Uint8Array;
  invertRgba(data: Uint8Array, width: number, height: number): Uint8Array;
  matEmpty(): WasmMatHandle;
  matFromF32(data: Float32Array, rows: number, columns: number, channels: number): WasmMatHandle;
  matFromF64(data: Float64Array, rows: number, columns: number, channels: number): WasmMatHandle;
  matFromI16(data: Int16Array, rows: number, columns: number, channels: number): WasmMatHandle;
  matFromI32(data: Int32Array, rows: number, columns: number, channels: number): WasmMatHandle;
  matFromI8(data: Int8Array, rows: number, columns: number, channels: number): WasmMatHandle;
  matFromU16(data: Uint16Array, rows: number, columns: number, channels: number): WasmMatHandle;
  matFromU8(data: Uint8Array, rows: number, columns: number, channels: number): WasmMatHandle;
  matCvtColorInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    code: number,
    destinationChannels: number,
  ): void;
  matResizeInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    targetWidth: number,
    targetHeight: number,
    scaleX: number,
    scaleY: number,
    interpolation: number,
  ): void;
  matGaussianBlurInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    width: number,
    height: number,
    sigmaX: number,
    sigmaY: number,
    borderType: number,
  ): void;
  matMorphologyExInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    operation: number,
    kernel: WasmMatHandle,
    anchorX: number,
    anchorY: number,
    iterations: number,
    borderType: number,
    borderValue: Float64Array,
    defaultBorderValue: boolean,
  ): void;
  matSobelInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    destinationDepth: number,
    dx: number,
    dy: number,
    kernelSize: number,
    scale: number,
    delta: number,
    borderType: number,
  ): void;
  matThresholdInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    threshold: number,
    maximum: number,
    thresholdType: number,
  ): number;
  matFlip(source: WasmMatHandle, flipCode: number): WasmMatHandle;
  matFlipInto(source: WasmMatHandle, destination: WasmMatHandle, flipCode: number): void;
  matRotate(source: WasmMatHandle, rotateCode: number): WasmMatHandle;
  matRotateInto(source: WasmMatHandle, destination: WasmMatHandle, rotateCode: number): void;
  matRepeat(source: WasmMatHandle, rowRepeats: number, columnRepeats: number): WasmMatHandle;
  matRepeatInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    rowRepeats: number,
    columnRepeats: number,
  ): void;
  matAbsdiffU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle;
  matAddU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle;
  matBitwiseAndU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle;
  matBitwiseNot(source: WasmMatHandle): WasmMatHandle;
  matBitwiseNotInto(source: WasmMatHandle, destination: WasmMatHandle): void;
  matBitwiseNotMaskedInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    mask: WasmMatHandle,
  ): void;
  matBitwiseNotU8(source: WasmMatHandle): WasmMatHandle;
  matBitwiseOrU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle;
  matBitwiseXorU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle;
  matCompareEqU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle;
  matCountNonZero(source: WasmMatHandle): number;
  matDeterminant(source: WasmMatHandle): number;
  matArcLength(contour: WasmMatHandle, closed: boolean): number;
  matBoundingRect(contour: WasmMatHandle): Int32Array;
  matContourArea(contour: WasmMatHandle, oriented: boolean): number;
  matInRangeU8(
    source: WasmMatHandle,
    lowerBound: WasmMatHandle,
    upperBound: WasmMatHandle,
  ): WasmMatHandle;
  matGetAffineTransform(source: WasmMatHandle, destination: WasmMatHandle): WasmMatHandle;
  matGetPerspectiveTransform(source: WasmMatHandle, destination: WasmMatHandle): WasmMatHandle;
  matGetRotationMatrix2D(
    centerX: number,
    centerY: number,
    angleDegrees: number,
    scale: number,
  ): WasmMatHandle;
  matInvertAffineTransform(transform: WasmMatHandle): WasmMatHandle;
  matInvertAffineTransformInto(transform: WasmMatHandle, destination: WasmMatHandle): void;
  matInvertInto(source: WasmMatHandle, destination: WasmMatHandle, method: number): number;
  matIsContourConvex(contour: WasmMatHandle): boolean;
  matSplit(source: WasmMatHandle): WasmMatHandle[];
  matMerge(first: WasmMatHandle, second: WasmMatHandle): WasmMatHandle;
  matMerge3(first: WasmMatHandle, second: WasmMatHandle, third: WasmMatHandle): WasmMatHandle;
  matMerge4(
    first: WasmMatHandle,
    second: WasmMatHandle,
    third: WasmMatHandle,
    fourth: WasmMatHandle,
  ): WasmMatHandle;
  matMixChannels(source: WasmMatHandle, destination: WasmMatHandle, fromTo: Uint16Array): void;
  matExtractChannel(source: WasmMatHandle, channel: number): WasmMatHandle;
  matInsertChannel(source: WasmMatHandle, destination: WasmMatHandle, channel: number): void;
  matLut(source: WasmMatHandle, table: WasmMatHandle): WasmMatHandle;
  matLutInto(source: WasmMatHandle, table: WasmMatHandle, destination: WasmMatHandle): void;
  matNorm(source: WasmMatHandle, normType: number): number;
  matNormMasked(source: WasmMatHandle, normType: number, mask: WasmMatHandle): number;
  matNormDiff(first: WasmMatHandle, second: WasmMatHandle, normType: number): number;
  matNormDiffMasked(
    first: WasmMatHandle,
    second: WasmMatHandle,
    normType: number,
    mask: WasmMatHandle,
  ): number;
  matNormalizeInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    alpha: number,
    beta: number,
    normType: number,
  ): void;
  matNormalizeMaskedInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    alpha: number,
    beta: number,
    normType: number,
    mask: WasmMatHandle,
  ): void;
  matMeanStdDevInto(
    source: WasmMatHandle,
    means: WasmMatHandle,
    standardDeviations: WasmMatHandle,
  ): void;
  matMeanStdDevMaskedInto(
    source: WasmMatHandle,
    means: WasmMatHandle,
    standardDeviations: WasmMatHandle,
    mask: WasmMatHandle,
  ): void;
  matReduceInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    axis: number,
    kind: number,
  ): void;
  matRandn(destination: WasmMatHandle, mean: Float64Array, standardDeviation: Float64Array): void;
  matRandu(destination: WasmMatHandle, lower: Float64Array, upper: Float64Array): void;
  matSetIdentity(destination: WasmMatHandle, value: Float64Array): void;
  matSolveInto(
    coefficients: WasmMatHandle,
    rightHandSides: WasmMatHandle,
    destination: WasmMatHandle,
    method: number,
  ): boolean;
  matHconcat2(first: WasmMatHandle, second: WasmMatHandle): WasmMatHandle;
  matHconcat3(first: WasmMatHandle, second: WasmMatHandle, third: WasmMatHandle): WasmMatHandle;
  matHconcat4(
    first: WasmMatHandle,
    second: WasmMatHandle,
    third: WasmMatHandle,
    fourth: WasmMatHandle,
  ): WasmMatHandle;
  matVconcat2(first: WasmMatHandle, second: WasmMatHandle): WasmMatHandle;
  matVconcat3(first: WasmMatHandle, second: WasmMatHandle, third: WasmMatHandle): WasmMatHandle;
  matVconcat4(
    first: WasmMatHandle,
    second: WasmMatHandle,
    third: WasmMatHandle,
    fourth: WasmMatHandle,
  ): WasmMatHandle;
  matExp(source: WasmMatHandle): WasmMatHandle;
  matExpInto(source: WasmMatHandle, destination: WasmMatHandle): void;
  matLog(source: WasmMatHandle): WasmMatHandle;
  matLogInto(source: WasmMatHandle, destination: WasmMatHandle): void;
  matSqrt(source: WasmMatHandle): WasmMatHandle;
  matSqrtInto(source: WasmMatHandle, destination: WasmMatHandle): void;
  matPow(source: WasmMatHandle, exponent: number): WasmMatHandle;
  matPowInto(source: WasmMatHandle, exponent: number, destination: WasmMatHandle): void;
  matMagnitude(x: WasmMatHandle, y: WasmMatHandle): WasmMatHandle;
  matMagnitudeInto(x: WasmMatHandle, y: WasmMatHandle, destination: WasmMatHandle): void;
  matCartToPolar(
    x: WasmMatHandle,
    y: WasmMatHandle,
    magnitude: WasmMatHandle,
    angle: WasmMatHandle,
    degrees: boolean,
  ): void;
  matPolarToCart(
    magnitude: WasmMatHandle,
    angle: WasmMatHandle,
    x: WasmMatHandle,
    y: WasmMatHandle,
    degrees: boolean,
  ): void;
  matMultiply(a: WasmMatHandle, b: WasmMatHandle, scale: number): WasmMatHandle;
  matMultiplyInto(
    a: WasmMatHandle,
    b: WasmMatHandle,
    destination: WasmMatHandle,
    scale: number,
    dtype: number,
  ): void;
  matDivide(a: WasmMatHandle, b: WasmMatHandle, scale: number): WasmMatHandle;
  matDivideInto(
    a: WasmMatHandle,
    b: WasmMatHandle,
    destination: WasmMatHandle,
    scale: number,
    dtype: number,
  ): void;
  matAddWeighted(
    a: WasmMatHandle,
    alpha: number,
    b: WasmMatHandle,
    beta: number,
    gamma: number,
  ): WasmMatHandle;
  matAddWeightedInto(
    a: WasmMatHandle,
    alpha: number,
    b: WasmMatHandle,
    beta: number,
    gamma: number,
    destination: WasmMatHandle,
    dtype: number,
  ): void;
  matConvertScaleAbs(source: WasmMatHandle, alpha: number, beta: number): WasmMatHandle;
  matConvertScaleAbsInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    alpha: number,
    beta: number,
  ): void;
  matCopyMakeBorder(
    source: WasmMatHandle,
    top: number,
    bottom: number,
    left: number,
    right: number,
    borderType: number,
    constant: Float64Array,
  ): WasmMatHandle;
  matCopyMakeBorderInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    top: number,
    bottom: number,
    left: number,
    right: number,
    borderType: number,
    constant: Float64Array,
  ): void;
  matMaxU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle;
  matMean(source: WasmMatHandle): Float64Array;
  matMeanMasked(source: WasmMatHandle, mask: WasmMatHandle): Float64Array;
  matMinMaxLoc(source: WasmMatHandle): Float64Array;
  matMinMaxLocMasked(source: WasmMatHandle, mask: WasmMatHandle): Float64Array;
  matMinU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle;
  matSubtractU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle;
  matSum(source: WasmMatHandle): Float64Array;
  matTranspose(source: WasmMatHandle): WasmMatHandle;
  matTransposeInto(source: WasmMatHandle, destination: WasmMatHandle): void;
  matTrace(source: WasmMatHandle): Float64Array;
  matTransform(source: WasmMatHandle, coefficients: WasmMatHandle): WasmMatHandle;
  matTransformInto(
    source: WasmMatHandle,
    coefficients: WasmMatHandle,
    destination: WasmMatHandle,
  ): void;
  matPerspectiveTransform(source: WasmMatHandle, coefficients: WasmMatHandle): WasmMatHandle;
  matPerspectiveTransformInto(
    source: WasmMatHandle,
    coefficients: WasmMatHandle,
    destination: WasmMatHandle,
  ): void;
  matPointPolygonTest(
    contour: WasmMatHandle,
    x: number,
    y: number,
    measureDistance: boolean,
  ): number;
  matZerosF32(rows: number, columns: number, channels: number): WasmMatHandle;
  matZerosF64(rows: number, columns: number, channels: number): WasmMatHandle;
  matZerosI16(rows: number, columns: number, channels: number): WasmMatHandle;
  matZerosI32(rows: number, columns: number, channels: number): WasmMatHandle;
  matZerosI8(rows: number, columns: number, channels: number): WasmMatHandle;
  matZerosU16(rows: number, columns: number, channels: number): WasmMatHandle;
  matZerosU8(rows: number, columns: number, channels: number): WasmMatHandle;
  setRNGSeed(seed: number): void;
  setLogLevel(level: number): number;
  resizeNearestRgba(
    data: Uint8Array,
    width: number,
    height: number,
    targetWidth: number,
    targetHeight: number,
  ): Uint8Array;
  thresholdRgba(data: Uint8Array, width: number, height: number, threshold: number): Uint8Array;
}

/** Initialized image processing client. */
export interface OpenCv {
  readonly TM_SQDIFF: 0;
  readonly TM_SQDIFF_NORMED: 1;
  readonly TM_CCORR: 2;
  readonly TM_CCORR_NORMED: 3;
  readonly TM_CCOEFF: 4;
  readonly TM_CCOEFF_NORMED: 5;
  /** Writes a single-channel F32 score map for U8/F32 images with one through four channels. */
  matchTemplate(image: Mat, template: Mat, result: Mat, method: TemplateMatchMode): void;
  /** U8 masks select nonzero pixels; F32 masks supply weights, shared or per channel. */
  matchTemplate(image: Mat, template: Mat, result: Mat, method: TemplateMatchMode, mask: Mat): void;
  readonly RETR_EXTERNAL: 0;
  readonly RETR_LIST: 1;
  readonly RETR_CCOMP: 2;
  readonly RETR_TREE: 3;
  readonly RETR_FLOODFILL: 4;
  readonly CHAIN_APPROX_NONE: 1;
  readonly CHAIN_APPROX_SIMPLE: 2;
  readonly CHAIN_APPROX_TC89_L1: 3;
  readonly CHAIN_APPROX_TC89_KCOS: 4;
  readonly WARP_FILL_OUTLIERS: 8;
  readonly WARP_INVERSE_MAP: 16;
  readonly WARP_RELATIVE_MAP: 32;
  readonly BORDER_CONSTANT: 0;
  readonly BORDER_REPLICATE: 1;
  readonly BORDER_REFLECT: 2;
  readonly BORDER_WRAP: 3;
  readonly BORDER_REFLECT_101: 4;
  readonly BORDER_TRANSPARENT: 5;
  readonly BORDER_DEFAULT: 4;
  readonly BORDER_ISOLATED: 16;
  readonly MORPH_ERODE: 0;
  readonly MORPH_DILATE: 1;
  readonly MORPH_OPEN: 2;
  readonly MORPH_CLOSE: 3;
  readonly MORPH_GRADIENT: 4;
  readonly MORPH_TOPHAT: 5;
  readonly MORPH_BLACKHAT: 6;
  readonly MORPH_HITMISS: 7;
  readonly COLOR_BGR2BGRA: 0;
  readonly COLOR_RGB2RGBA: 0;
  readonly COLOR_BGRA2BGR: 1;
  readonly COLOR_RGBA2RGB: 1;
  readonly COLOR_BGR2RGBA: 2;
  readonly COLOR_RGB2BGRA: 2;
  readonly COLOR_RGBA2BGR: 3;
  readonly COLOR_BGRA2RGB: 3;
  readonly COLOR_BGR2RGB: 4;
  readonly COLOR_RGB2BGR: 4;
  readonly COLOR_BGRA2RGBA: 5;
  readonly COLOR_RGBA2BGRA: 5;
  readonly COLOR_BGR2GRAY: 6;
  readonly COLOR_RGB2GRAY: 7;
  readonly COLOR_GRAY2BGR: 8;
  readonly COLOR_GRAY2RGB: 8;
  readonly COLOR_GRAY2BGRA: 9;
  readonly COLOR_GRAY2RGBA: 9;
  readonly COLOR_BGRA2GRAY: 10;
  readonly COLOR_RGBA2GRAY: 11;
  readonly INTER_NEAREST: 0;
  readonly INTER_LINEAR: 1;
  readonly INTER_CUBIC: 2;
  readonly INTER_AREA: 3;
  readonly INTER_LANCZOS4: 4;
  readonly INTER_LINEAR_EXACT: 5;
  readonly INTER_NEAREST_EXACT: 6;
  readonly THRESH_BINARY: 0;
  readonly THRESH_BINARY_INV: 1;
  readonly THRESH_TRUNC: 2;
  readonly THRESH_TOZERO: 3;
  readonly THRESH_TOZERO_INV: 4;
  readonly THRESH_MASK: 7;
  readonly THRESH_OTSU: 8;
  readonly THRESH_TRIANGLE: 16;
  readonly THRESH_DRYRUN: 128;
  readonly AKAZE_DescriptorType: AKAZE_DescriptorTypeNamespace;
  readonly AgastFeatureDetector_DetectorType: AgastFeatureDetector_DetectorTypeNamespace;
  readonly FastFeatureDetector_DetectorType: FastFeatureDetector_DetectorTypeNamespace;
  readonly KAZE_DiffusivityType: KAZE_DiffusivityTypeNamespace;
  readonly ORB_FAST_SCORE: 1;
  readonly ORB_HARRIS_SCORE: 0;
  readonly ORB_ScoreType: ORB_ScoreTypeNamespace;
  readonly ROTATE_90_CLOCKWISE: 0;
  readonly ROTATE_180: 1;
  readonly ROTATE_90_COUNTERCLOCKWISE: 2;
  absdiff(left: Mat, right: Mat): Mat;
  add(left: Mat, right: Mat): Mat;
  addWeighted(
    a: Mat,
    alpha: number,
    b: Mat,
    beta: number,
    gamma: number,
    destination: Mat,
    dtype?: number,
  ): void;
  addWeightedAlloc(a: Mat, alpha: number, b: Mat, beta: number, gamma: number): Mat;
  bitwiseAnd(left: Mat, right: Mat): Mat;
  bitwiseNot(source: Mat, destination: Mat): void;
  bitwiseNot(source: Mat, destination: Mat, mask: Mat): void;
  bitwiseNotAlloc(source: Mat): Mat;
  bitwiseOr(left: Mat, right: Mat): Mat;
  bitwiseXor(left: Mat, right: Mat): Mat;
  arcLength(contour: Mat, closed: boolean): number;
  boundingRect(contour: Mat): Rect;
  clipLine(rectangle: Rect, start: Point, end: Point): readonly [Point, Point] | undefined;
  compareEqual(left: Mat, right: Mat): Mat;
  cartToPolar(x: Mat, y: Mat, magnitude: Mat, angle: Mat, degrees?: boolean): void;
  Canny(
    source: Mat,
    destination: Mat,
    threshold1: number,
    threshold2: number,
    apertureSize?: number,
    l2Gradient?: boolean,
  ): void;
  countNonZero(source: Mat): number;
  contourArea(contour: Mat, oriented?: boolean): number;
  createHanningWindow(destination: Mat, size: Size, type: HanningWindowType): void;
  createHanningWindowAlloc(size: Size, depth: HanningWindowDepth): Mat;
  createAKAZE(options?: AKAZEOptions): AKAZE;
  createKAZE(options?: KAZEOptions): KAZE;
  createORB(options?: ORBOptions): ORB;
  createAgastFeatureDetector(options?: AgastFeatureDetectorOptions): AgastFeatureDetector;
  createFastFeatureDetector(options?: FastFeatureDetectorOptions): FastFeatureDetector;
  createGFTTDetector(options?: GFTTDetectorOptions): GFTTDetector;
  createMSER(options?: MSEROptions): MSER;
  createMatVector(): MatVector;
  createTonemapDrago(gamma?: number, saturation?: number, bias?: number): TonemapDrago;
  createTonemapMantiuk(gamma?: number, scale?: number, saturation?: number): TonemapMantiuk;
  createTonemapReinhard(
    gamma?: number,
    intensity?: number,
    lightAdaptation?: number,
    colorAdaptation?: number,
  ): TonemapReinhard;
  determinant(source: Mat): number;
  convertScaleAbs(source: Mat, destination: Mat, alpha?: number, beta?: number): void;
  convertScaleAbsAlloc(source: Mat, alpha?: number, beta?: number): Mat;
  copyMakeBorder(
    source: Mat,
    top: number,
    bottom: number,
    left: number,
    right: number,
    borderType: BorderType,
    constant?: Scalar,
  ): Mat;
  cvtColor(
    source: Mat,
    destination: Mat,
    code: ColorConversionCode,
    destinationChannels?: number,
  ): void;
  divide(a: Mat, b: Mat, destination: Mat, scale?: number, dtype?: number): void;
  divideAlloc(a: Mat, b: Mat, scale?: number): Mat;
  extractChannel(source: Mat, channel: number): Mat;
  ellipse2Poly(
    center: Point,
    axes: Size,
    rotationDegrees: number,
    arcStart: number,
    arcEnd: number,
    delta: number,
  ): Point[];
  exp(source: Mat, destination: Mat): void;
  expAlloc(source: Mat): Mat;
  emptyMat(): Mat;
  equalizeHist(source: Mat, destination: Mat): void;
  flip(source: Mat, destination: Mat, flipCode: number): void;
  flipAlloc(source: Mat, flipCode: number): Mat;
  findContours(
    source: Mat,
    contours: MatVector,
    hierarchy: Mat,
    mode: number,
    method: number,
    offset?: Point,
  ): void;
  grayscale(image: RgbaImage): RgbaImage;
  hconcat(
    sources: readonly [Mat, Mat] | readonly [Mat, Mat, Mat] | readonly [Mat, Mat, Mat, Mat],
  ): Mat;
  getAffineTransform(source: Mat, destination: Mat): Mat;
  getLogLevel(): LogLevel;
  getOptimalDFTSize(size: number): number;
  GaussianBlur(
    source: Mat,
    destination: Mat,
    size: Size,
    sigmaX: number,
    sigmaY?: number,
    borderType?: BorderType,
  ): void;
  getPerspectiveTransform(source: Mat, destination: Mat): Mat;
  getRotationMatrix2D(center: Point, angleDegrees: number, scale: number): Mat;
  getStructuringElement(kind: StructuringElementKind, size: Size, anchor?: Point): Mat;
  invert(image: RgbaImage): RgbaImage;
  matFromF32(rows: number, columns: number, channels: number, data: Float32Array): Mat;
  matFromF64(rows: number, columns: number, channels: number, data: Float64Array): Mat;
  matFromI16(rows: number, columns: number, channels: number, data: Int16Array): Mat;
  matFromI32(rows: number, columns: number, channels: number, data: Int32Array): Mat;
  matFromI8(rows: number, columns: number, channels: number, data: Int8Array): Mat;
  matFromU16(rows: number, columns: number, channels: number, data: Uint16Array): Mat;
  matFromU8(rows: number, columns: number, channels: number, data: Uint8Array): Mat;
  inRange(source: Mat, lowerBound: Mat, upperBound: Mat): Mat;
  insertChannel(source: Mat, destination: Mat, channel: number): void;
  invertAffineTransform(transform: Mat, destination: Mat): void;
  invertAffineTransformAlloc(transform: Mat): Mat;
  isContourConvex(contour: Mat): boolean;
  invert(source: Mat, destination: Mat, method?: DecompositionMethod): number;
  log(source: Mat, destination: Mat): void;
  logAlloc(source: Mat): Mat;
  lut(source: Mat, table: Mat): Mat;
  lut(source: Mat, table: Mat, destination: Mat): void;
  magnitude(x: Mat, y: Mat, destination: Mat): void;
  magnitudeAlloc(x: Mat, y: Mat): Mat;
  max(left: Mat, right: Mat): Mat;
  mean(source: Mat): Scalar;
  mean(source: Mat, mask: Mat): Scalar;
  meanStdDev(source: Mat, means: Mat, standardDeviations: Mat, mask?: Mat): void;
  merge(
    sources: readonly [Mat, Mat] | readonly [Mat, Mat, Mat] | readonly [Mat, Mat, Mat, Mat],
  ): Mat;
  minMaxLoc(source: Mat): MinMaxLocation;
  minMaxLoc(source: Mat, mask: Mat): MinMaxLocation;
  min(left: Mat, right: Mat): Mat;
  mixChannels(source: Mat, destination: Mat, fromTo: Uint16Array): void;
  multiply(a: Mat, b: Mat, destination: Mat, scale?: number, dtype?: number): void;
  multiplyAlloc(a: Mat, b: Mat, scale?: number): Mat;
  morphologyEx(
    source: Mat,
    destination: Mat,
    operation: number,
    kernel: Mat,
    anchor?: Point,
    iterations?: number,
    borderType?: BorderType,
    borderValue?: Scalar,
  ): void;
  norm(source: Mat, normType?: NormType, mask?: Mat): number;
  norm(first: Mat, second: Mat, normType?: NormType, mask?: Mat): number;
  normalize(
    source: Mat,
    destination: Mat,
    alpha: number,
    beta: number,
    normType: NormalizeType,
    mask?: Mat,
  ): void;
  polarToCart(magnitude: Mat, angle: Mat, x: Mat, y: Mat, degrees?: boolean): void;
  pointPolygonTest(contour: Mat, point: Point, measureDistance: boolean): number;
  perspectiveTransform(source: Mat, coefficients: Mat): Mat;
  perspectiveTransform(source: Mat, coefficients: Mat, destination: Mat): void;
  pow(source: Mat, exponent: number, destination: Mat): void;
  powAlloc(source: Mat, exponent: number): Mat;
  randn(destination: Mat, mean: Scalar, standardDeviation: Scalar): void;
  randu(destination: Mat, lower: Scalar, upper: Scalar): void;
  resizeNearest(image: RgbaImage, targetWidth: number, targetHeight: number): RgbaImage;
  resize(
    source: Mat,
    destination: Mat,
    size: Size,
    scaleX?: number,
    scaleY?: number,
    interpolation?: Interpolation,
  ): void;
  repeat(source: Mat, rowRepeats: number, columnRepeats: number, destination: Mat): void;
  repeatAlloc(source: Mat, rowRepeats: number, columnRepeats: number): Mat;
  rotate(source: Mat, destination: Mat, rotateCode: number): void;
  rotateAlloc(source: Mat, rotateCode: number): Mat;
  setIdentity(destination: Mat, value?: Scalar): void;
  setLogLevel(level: LogLevel): LogLevel;
  setRNGSeed(seed: number): void;
  solve(
    coefficients: Mat,
    rightHandSides: Mat,
    destination: Mat,
    method?: DecompositionMethod,
  ): boolean;
  Sobel(
    source: Mat,
    destination: Mat,
    destinationDepth: number,
    dx: number,
    dy: number,
    kernelSize?: number,
    scale?: number,
    delta?: number,
    borderType?: BorderType,
  ): void;
  reduce(source: Mat, destination: Mat, axis: 0 | 1, kind: ReduceKind): void;
  threshold(image: RgbaImage, threshold: number): RgbaImage;
  threshold(
    source: Mat,
    destination: Mat,
    threshold: number,
    maximum: number,
    type: number,
  ): number;
  subtract(left: Mat, right: Mat): Mat;
  split(source: Mat): Mat[];
  sqrt(source: Mat, destination: Mat): void;
  sqrtAlloc(source: Mat): Mat;
  sum(source: Mat): Scalar;
  transpose(source: Mat, destination: Mat): void;
  transposeAlloc(source: Mat): Mat;
  trace(source: Mat): Scalar;
  transform(source: Mat, coefficients: Mat): Mat;
  transform(source: Mat, coefficients: Mat, destination: Mat): void;
  vconcat(
    sources: readonly [Mat, Mat] | readonly [Mat, Mat, Mat] | readonly [Mat, Mat, Mat, Mat],
  ): Mat;
  warpAffine(
    source: Mat,
    destination: Mat,
    transform: Mat,
    size: Size,
    flags?: number,
    borderType?: BorderType,
    borderValue?: Scalar,
  ): void;
  zerosF32(rows: number, columns: number, channels: number): Mat;
  zerosF64(rows: number, columns: number, channels: number): Mat;
  zerosI16(rows: number, columns: number, channels: number): Mat;
  zerosI32(rows: number, columns: number, channels: number): Mat;
  zerosI8(rows: number, columns: number, channels: number): Mat;
  zerosU16(rows: number, columns: number, channels: number): Mat;
  zerosU8(rows: number, columns: number, channels: number): Mat;
}
