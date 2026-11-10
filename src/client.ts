import {
  DECOMP_LU,
  DECOMP_SVD,
  DECOMP_EIG,
  DECOMP_CHOLESKY,
  DECOMP_QR,
  DECOMP_NORMAL,
} from "./decomposition.js";
import {
  AKAZE,
  AKAZE_DescriptorType,
  KAZE_DiffusivityType,
  validateAKAZEOptions,
} from "./akaze.js";
import type { AKAZEOptions } from "./akaze.js";
import { KAZE, validateKAZEOptions } from "./kaze.js";
import type { KAZEOptions } from "./kaze.js";
import { ORB, ORB_FAST_SCORE, ORB_HARRIS_SCORE, ORB_ScoreType, validateORBOptions } from "./orb.js";
import type { ORBOptions } from "./orb.js";
import { GFTTDetector, validateGFTTDetectorOptions } from "./gftt.js";
import type { GFTTDetectorOptions } from "./gftt.js";
import { MSER, validateMSEROptions } from "./mser.js";
import type { MSEROptions } from "./mser.js";
import { TonemapDrago, TonemapMantiuk, TonemapReinhard } from "./tonemap.js";
import {
  AgastFeatureDetector,
  AgastFeatureDetector_DetectorType,
  FastFeatureDetector,
  FastFeatureDetector_DetectorType,
  validateAgastFeatureDetectorOptions,
  validateFastFeatureDetectorOptions,
} from "./feature-detectors.js";
import type {
  AgastFeatureDetectorOptions,
  FastFeatureDetectorOptions,
} from "./feature-detectors.js";
import {
  createRgbaImage,
  validateDimension,
  validateRgbaImage,
  validateThreshold,
} from "./image.js";
import { BindingError, OpenCvInputError } from "./error.js";
import { Mat, validateMatrixDimension, validateMatrixInput } from "./mat.js";
import { MatVector } from "./mat-vector.js";
import { PointVector } from "./point-vector.js";
import {
  COLOR_BGR2BGRA,
  COLOR_BGR2GRAY,
  COLOR_BGR2RGB,
  COLOR_BGR2RGBA,
  COLOR_BGRA2BGR,
  COLOR_BGRA2GRAY,
  COLOR_BGRA2RGB,
  COLOR_BGRA2RGBA,
  COLOR_GRAY2BGR,
  COLOR_GRAY2BGRA,
  COLOR_GRAY2RGB,
  COLOR_GRAY2RGBA,
  COLOR_RGB2BGR,
  COLOR_RGB2BGRA,
  COLOR_RGB2GRAY,
  COLOR_RGB2RGBA,
  COLOR_RGBA2BGR,
  COLOR_RGBA2BGRA,
  COLOR_RGBA2GRAY,
  COLOR_RGBA2RGB,
} from "./color.js";
import type { ColorConversionCode } from "./color.js";
import {
  CHAIN_APPROX_NONE,
  CHAIN_APPROX_SIMPLE,
  CHAIN_APPROX_TC89_KCOS,
  CHAIN_APPROX_TC89_L1,
  RETR_CCOMP,
  RETR_EXTERNAL,
  RETR_FLOODFILL,
  RETR_LIST,
  RETR_TREE,
} from "./contours.js";
import {
  BORDER_CONSTANT,
  BORDER_DEFAULT,
  BORDER_ISOLATED,
  BORDER_REFLECT,
  BORDER_REFLECT_101,
  BORDER_REPLICATE,
  BORDER_TRANSPARENT,
  BORDER_WRAP,
  MORPH_BLACKHAT,
  MORPH_CLOSE,
  MORPH_DILATE,
  MORPH_ERODE,
  MORPH_GRADIENT,
  MORPH_HITMISS,
  MORPH_OPEN,
  MORPH_TOPHAT,
  WARP_FILL_OUTLIERS,
  WARP_INVERSE_MAP,
  WARP_RELATIVE_MAP,
} from "./filtering.js";
import {
  INTER_AREA,
  INTER_CUBIC,
  INTER_LANCZOS4,
  INTER_LINEAR,
  INTER_LINEAR_EXACT,
  INTER_NEAREST,
  INTER_NEAREST_EXACT,
} from "./interpolation.js";
import type { Interpolation } from "./interpolation.js";
import {
  TM_SQDIFF,
  TM_SQDIFF_NORMED,
  TM_CCORR,
  TM_CCORR_NORMED,
  TM_CCOEFF,
  TM_CCOEFF_NORMED,
} from "./template-matching.js";
import type { TemplateMatchMode } from "./template-matching.js";
import {
  THRESH_BINARY,
  THRESH_BINARY_INV,
  THRESH_DRYRUN,
  THRESH_MASK,
  THRESH_OTSU,
  THRESH_TOZERO,
  THRESH_TOZERO_INV,
  THRESH_TRIANGLE,
  THRESH_TRUNC,
} from "./threshold.js";
import type {
  AdaptiveThresholdType,
  BorderType,
  Circle,
  ColormapType,
  DecompositionMethod,
  DistType,
  HanningWindowDepth,
  HanningWindowType,
  LogLevel,
  MinMaxLocation,
  Moments,
  NormalizeType,
  NormType,
  OpenCv,
  OpenCvBackend,
  Point,
  Rect,
  RgbaImage,
  RotatedRect,
  ReduceKind,
  Scalar,
  Size,
  StructuringElementKind,
  TermCriteria,
  ThresholdType,
} from "./types.js";

function contourPoints(pts: MatVector | Mat): Int32Array {
  const mat = pts instanceof MatVector ? pts.get(0) : pts;
  return Int32Array.from(mat.toInt32Array());
}

class WasmOpenCv implements OpenCv {
  readonly TM_SQDIFF = TM_SQDIFF;
  readonly TM_SQDIFF_NORMED = TM_SQDIFF_NORMED;
  readonly TM_CCORR = TM_CCORR;
  readonly TM_CCORR_NORMED = TM_CCORR_NORMED;
  readonly TM_CCOEFF = TM_CCOEFF;
  readonly TM_CCOEFF_NORMED = TM_CCOEFF_NORMED;
  readonly DECOMP_LU = DECOMP_LU;
  readonly DECOMP_SVD = DECOMP_SVD;
  readonly DECOMP_EIG = DECOMP_EIG;
  readonly DECOMP_CHOLESKY = DECOMP_CHOLESKY;
  readonly DECOMP_QR = DECOMP_QR;
  readonly DECOMP_NORMAL = DECOMP_NORMAL;
  readonly RETR_EXTERNAL = RETR_EXTERNAL;
  readonly RETR_LIST = RETR_LIST;
  readonly RETR_CCOMP = RETR_CCOMP;
  readonly RETR_TREE = RETR_TREE;
  readonly RETR_FLOODFILL = RETR_FLOODFILL;
  readonly CHAIN_APPROX_NONE = CHAIN_APPROX_NONE;
  readonly CHAIN_APPROX_SIMPLE = CHAIN_APPROX_SIMPLE;
  readonly CHAIN_APPROX_TC89_L1 = CHAIN_APPROX_TC89_L1;
  readonly CHAIN_APPROX_TC89_KCOS = CHAIN_APPROX_TC89_KCOS;
  readonly BORDER_CONSTANT = BORDER_CONSTANT;
  readonly BORDER_REPLICATE = BORDER_REPLICATE;
  readonly BORDER_REFLECT = BORDER_REFLECT;
  readonly BORDER_WRAP = BORDER_WRAP;
  readonly BORDER_REFLECT_101 = BORDER_REFLECT_101;
  readonly BORDER_TRANSPARENT = BORDER_TRANSPARENT;
  readonly BORDER_DEFAULT = BORDER_DEFAULT;
  readonly BORDER_ISOLATED = BORDER_ISOLATED;
  readonly MORPH_ERODE = MORPH_ERODE;
  readonly MORPH_DILATE = MORPH_DILATE;
  readonly MORPH_OPEN = MORPH_OPEN;
  readonly MORPH_CLOSE = MORPH_CLOSE;
  readonly MORPH_GRADIENT = MORPH_GRADIENT;
  readonly MORPH_TOPHAT = MORPH_TOPHAT;
  readonly MORPH_BLACKHAT = MORPH_BLACKHAT;
  readonly MORPH_HITMISS = MORPH_HITMISS;
  readonly WARP_FILL_OUTLIERS = WARP_FILL_OUTLIERS;
  readonly WARP_INVERSE_MAP = WARP_INVERSE_MAP;
  readonly WARP_RELATIVE_MAP = WARP_RELATIVE_MAP;
  readonly COLOR_BGR2BGRA = COLOR_BGR2BGRA;
  readonly COLOR_RGB2RGBA = COLOR_RGB2RGBA;
  readonly COLOR_BGRA2BGR = COLOR_BGRA2BGR;
  readonly COLOR_RGBA2RGB = COLOR_RGBA2RGB;
  readonly COLOR_BGR2RGBA = COLOR_BGR2RGBA;
  readonly COLOR_RGB2BGRA = COLOR_RGB2BGRA;
  readonly COLOR_RGBA2BGR = COLOR_RGBA2BGR;
  readonly COLOR_BGRA2RGB = COLOR_BGRA2RGB;
  readonly COLOR_BGR2RGB = COLOR_BGR2RGB;
  readonly COLOR_RGB2BGR = COLOR_RGB2BGR;
  readonly COLOR_BGRA2RGBA = COLOR_BGRA2RGBA;
  readonly COLOR_RGBA2BGRA = COLOR_RGBA2BGRA;
  readonly COLOR_BGR2GRAY = COLOR_BGR2GRAY;
  readonly COLOR_RGB2GRAY = COLOR_RGB2GRAY;
  readonly COLOR_GRAY2BGR = COLOR_GRAY2BGR;
  readonly COLOR_GRAY2RGB = COLOR_GRAY2RGB;
  readonly COLOR_GRAY2BGRA = COLOR_GRAY2BGRA;
  readonly COLOR_GRAY2RGBA = COLOR_GRAY2RGBA;
  readonly COLOR_BGRA2GRAY = COLOR_BGRA2GRAY;
  readonly COLOR_RGBA2GRAY = COLOR_RGBA2GRAY;
  readonly INTER_NEAREST = INTER_NEAREST;
  readonly INTER_LINEAR = INTER_LINEAR;
  readonly INTER_CUBIC = INTER_CUBIC;
  readonly INTER_AREA = INTER_AREA;
  readonly INTER_LANCZOS4 = INTER_LANCZOS4;
  readonly INTER_LINEAR_EXACT = INTER_LINEAR_EXACT;
  readonly INTER_NEAREST_EXACT = INTER_NEAREST_EXACT;
  readonly THRESH_BINARY = THRESH_BINARY;
  readonly THRESH_BINARY_INV = THRESH_BINARY_INV;
  readonly THRESH_TRUNC = THRESH_TRUNC;
  readonly THRESH_TOZERO = THRESH_TOZERO;
  readonly THRESH_TOZERO_INV = THRESH_TOZERO_INV;
  readonly THRESH_MASK = THRESH_MASK;
  readonly THRESH_OTSU = THRESH_OTSU;
  readonly THRESH_TRIANGLE = THRESH_TRIANGLE;
  readonly THRESH_DRYRUN = THRESH_DRYRUN;
  readonly AKAZE_DescriptorType = AKAZE_DescriptorType;
  readonly AgastFeatureDetector_DetectorType = AgastFeatureDetector_DetectorType;
  readonly FastFeatureDetector_DetectorType = FastFeatureDetector_DetectorType;
  readonly KAZE_DiffusivityType = KAZE_DiffusivityType;
  readonly ORB_FAST_SCORE = ORB_FAST_SCORE;
  readonly ORB_HARRIS_SCORE = ORB_HARRIS_SCORE;
  readonly ORB_ScoreType = ORB_ScoreType;
  readonly ROTATE_90_CLOCKWISE = 0 as const;
  readonly ROTATE_180 = 1 as const;
  readonly ROTATE_90_COUNTERCLOCKWISE = 2 as const;
  readonly #backend: OpenCvBackend;

  constructor(backend: OpenCvBackend) {
    this.#backend = backend;
  }

  absdiff(source1: Mat, source2: Mat, destination: Mat): void {
    requireExactArity(arguments.length, 3, "absdiff");
    this.#backend.matAbsdiffInto(
      matHandleForBinding(source1),
      matHandleForBinding(source2),
      matHandleForBinding(destination),
    );
  }

  add(
    ...arguments_:
      | [source1: Mat, source2: Mat, destination: Mat]
      | [source1: Mat, source2: Mat, destination: Mat, mask: Mat]
      | [source1: Mat, source2: Mat, destination: Mat, mask: Mat, dtype: number]
  ): void {
    requireArityRange(arguments_.length, 3, 5, "add");
    const source1 = matHandleForBinding(arguments_[0]);
    const source2 = matHandleForBinding(arguments_[1]);
    const destination = matHandleForBinding(arguments_[2]);
    if (arguments_.length === 3) {
      this.#backend.matAddInto(source1, source2, destination);
      return;
    }
    const mask = matHandleForBinding(arguments_[3]);
    if (arguments_.length === 4) {
      this.#backend.matAddMaskedInto(source1, source2, destination, mask);
      return;
    }
    this.#backend.matAddMaskedDtypeInto(
      source1,
      source2,
      destination,
      mask,
      toWasmI32(arguments_[4]),
    );
  }

  approxPolyDP(curve: Mat, approximation: Mat, epsilon: number, closed: boolean): void {
    requireExactArity(arguments.length, 4, "approxPolyDP");
    this.#backend.matApproxPolyDPInto(
      matHandleForBinding(curve),
      matHandleForBinding(approximation),
      toWasmF64(epsilon),
      coerceBoolean(closed),
    );
  }

  arcLength(contour: Mat, closed: boolean): number {
    requireExactArity(arguments.length, 2, "arcLength");
    return this.#backend.matArcLength(matHandleForBinding(contour), coerceBoolean(closed));
  }

  addWeighted(
    ...arguments_: [
      a: Mat,
      alpha: number,
      b: Mat,
      beta: number,
      gamma: number,
      destination: Mat,
      dtype?: number,
    ]
  ): void {
    requireOverloadArity(arguments_.length, 6, 7, "addWeighted");
    const [a, alpha, b, beta, gamma, destination] = arguments_;
    const dtype = arguments_.length === 7 ? toWasmI32(arguments_[6]) : -1;
    this.#backend.matAddWeightedInto(
      matHandleForBinding(a),
      toWasmF64(alpha),
      matHandleForBinding(b),
      toWasmF64(beta),
      toWasmF64(gamma),
      matHandleForBinding(destination),
      dtype,
    );
  }

  addWeightedAlloc(a: Mat, alpha: number, b: Mat, beta: number, gamma: number): Mat {
    return new Mat(
      this.#backend.matAddWeighted(
        a.handleForBackend(),
        toWasmF64(alpha),
        b.handleForBackend(),
        toWasmF64(beta),
        toWasmF64(gamma),
      ),
    );
  }

  bitwiseAnd(
    ...arguments_:
      | [source1: Mat, source2: Mat, destination: Mat]
      | [source1: Mat, source2: Mat, destination: Mat, mask: Mat]
  ): void {
    requireOverloadArity(arguments_.length, 3, 4, "bitwise_and");
    const source1 = matHandleForBinding(arguments_[0]);
    const source2 = matHandleForBinding(arguments_[1]);
    const destination = matHandleForBinding(arguments_[2]);
    if (arguments_.length === 3) {
      this.#backend.matBitwiseAndInto(source1, source2, destination);
      return;
    }
    this.#backend.matBitwiseAndMaskedInto(
      source1,
      source2,
      destination,
      matHandleForBinding(arguments_[3]),
    );
  }

  bitwiseNot(
    ...arguments_: [source: Mat, destination: Mat] | [source: Mat, destination: Mat, mask: Mat]
  ): void {
    requireOverloadArity(arguments_.length, 2, 3, "bitwise_not");
    const source = matHandleForBinding(arguments_[0]);
    const destination = matHandleForBinding(arguments_[1]);
    if (arguments_.length === 2) {
      this.#backend.matBitwiseNotInto(source, destination);
      return;
    }
    this.#backend.matBitwiseNotMaskedInto(source, destination, matHandleForBinding(arguments_[2]));
  }

  bitwiseNotAlloc(source: Mat): Mat {
    return new Mat(this.#backend.matBitwiseNot(source.handleForBackend()));
  }

  bitwiseOr(
    ...arguments_:
      | [source1: Mat, source2: Mat, destination: Mat]
      | [source1: Mat, source2: Mat, destination: Mat, mask: Mat]
  ): void {
    requireOverloadArity(arguments_.length, 3, 4, "bitwise_or");
    const source1 = matHandleForBinding(arguments_[0]);
    const source2 = matHandleForBinding(arguments_[1]);
    const destination = matHandleForBinding(arguments_[2]);
    if (arguments_.length === 3) {
      this.#backend.matBitwiseOrInto(source1, source2, destination);
      return;
    }
    this.#backend.matBitwiseOrMaskedInto(
      source1,
      source2,
      destination,
      matHandleForBinding(arguments_[3]),
    );
  }

  bitwiseXor(
    ...arguments_:
      | [source1: Mat, source2: Mat, destination: Mat]
      | [source1: Mat, source2: Mat, destination: Mat, mask: Mat]
  ): void {
    requireOverloadArity(arguments_.length, 3, 4, "bitwise_xor");
    const source1 = matHandleForBinding(arguments_[0]);
    const source2 = matHandleForBinding(arguments_[1]);
    const destination = matHandleForBinding(arguments_[2]);
    if (arguments_.length === 3) {
      this.#backend.matBitwiseXorInto(source1, source2, destination);
      return;
    }
    this.#backend.matBitwiseXorMaskedInto(
      source1,
      source2,
      destination,
      matHandleForBinding(arguments_[3]),
    );
  }

  boundingRect(contour: Mat): Rect {
    requireExactArity(arguments.length, 1, "boundingRect");
    return rectangleFromArray(this.#backend.matBoundingRect(matHandleForBinding(contour)));
  }

  clipLine(rectangle: Rect, start: Point, end: Point): boolean {
    requireExactArity(arguments.length, 3, "clipLine");
    const rect = rectForBinding(rectangle);
    const startPoint = pointForBinding(start, "start");
    const endPoint = pointForBinding(end, "end");
    return this.#backend.clipLineBool(
      rect.x,
      rect.y,
      rect.width,
      rect.height,
      startPoint.x,
      startPoint.y,
      endPoint.x,
      endPoint.y,
    );
  }

  clipLinePoints(rectangle: Rect, start: Point, end: Point): readonly [Point, Point] | undefined {
    validateRect(rectangle);
    validateIntegerPoint(start, "start");
    validateIntegerPoint(end, "end");
    return lineFromArray(
      this.#backend.clipLine(
        rectangle.x,
        rectangle.y,
        rectangle.width,
        rectangle.height,
        start.x,
        start.y,
        end.x,
        end.y,
      ),
    );
  }

  compare(source1: Mat, source2: Mat, destination: Mat, compareOp: number): void {
    requireExactArity(arguments.length, 4, "compare");
    this.#backend.matCompareInto(
      matHandleForBinding(source1),
      matHandleForBinding(source2),
      matHandleForBinding(destination),
      toWasmI32(compareOp),
    );
  }

  compareEqual(left: Mat, right: Mat): Mat {
    return new Mat(this.#backend.matCompareEqU8(left.handleForBackend(), right.handleForBackend()));
  }

  cartToPolar(
    ...arguments_: [x: Mat, y: Mat, magnitude: Mat, angle: Mat, degrees?: boolean]
  ): void {
    requireOverloadArity(arguments_.length, 4, 5, "cartToPolar");
    const [x, y, magnitude, angle, degrees = false] = arguments_;
    this.#backend.matCartToPolar(
      matHandleForBinding(x),
      matHandleForBinding(y),
      matHandleForBinding(magnitude),
      matHandleForBinding(angle),
      coerceBoolean(degrees),
    );
  }

  Canny(
    source: Mat,
    destination: Mat,
    threshold1: number,
    threshold2: number,
    apertureSize = 3,
    l2Gradient = false,
  ): void {
    requireArityRange(arguments.length, 4, 6, "Canny");
    this.#backend.matCannyInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmF64(threshold1),
      toWasmF64(threshold2),
      toWasmI32(apertureSize),
      coerceBoolean(l2Gradient),
    );
  }

  countNonZero(source: Mat): number {
    requireExactArity(arguments.length, 1, "countNonZero");
    return this.#backend.matCountNonZero(matHandleForBinding(source));
  }

  contourArea(...arguments_: [contour: Mat, oriented?: boolean]): number {
    requireOverloadArity(arguments_.length, 1, 2, "contourArea");
    const [contour, oriented = false] = arguments_;
    return this.#backend.matContourArea(matHandleForBinding(contour), coerceBoolean(oriented));
  }

  createHanningWindow(destination: Mat, size: Size, type: HanningWindowType): void {
    requireExactArity(arguments.length, 3, "createHanningWindow");
    const destinationHandle = matHandleForBinding(destination);
    const convertedSize = size2iForBinding(size);
    const convertedType = toWasmI32(type);
    this.#backend.createHanningWindowInto(
      destinationHandle,
      convertedSize.width,
      convertedSize.height,
      convertedType,
    );
  }

  createHanningWindowAlloc(size: Size, depth: HanningWindowDepth): Mat {
    validateMinimumSize(size, 2, "Hanning window");
    return new Mat(this.#backend.createHanningWindow(size.width, size.height, depthCode(depth)));
  }

  createAKAZE(options: AKAZEOptions = {}): AKAZE {
    validateAKAZEOptions(options);
    return new AKAZE(
      this.#backend.AKAZE.create(
        options.descriptorType,
        options.descriptorSize,
        options.descriptorChannels,
        options.threshold,
        options.octaves,
        options.octaveLayers,
        options.diffusivity,
        options.maxPoints,
      ),
    );
  }

  createKAZE(options: KAZEOptions = {}): KAZE {
    validateKAZEOptions(options);
    return new KAZE(
      this.#backend.KAZE.create(
        options.extended,
        options.upright,
        options.threshold,
        options.octaves,
        options.octaveLayers,
        options.diffusivity,
      ),
    );
  }

  createORB(options: ORBOptions = {}): ORB {
    validateORBOptions(options);
    return new ORB(
      this.#backend.ORB.create(
        options.maxFeatures,
        options.scaleFactor === undefined ? undefined : Math.fround(options.scaleFactor),
        options.nLevels,
        options.edgeThreshold,
        options.firstLevel,
        options.wtaK,
        options.scoreType,
        options.patchSize,
        options.fastThreshold,
      ),
    );
  }

  createGFTTDetector(options: GFTTDetectorOptions = {}): GFTTDetector {
    validateGFTTDetectorOptions(options);
    return new GFTTDetector(
      this.#backend.GFTTDetector.create(
        options.maxFeatures,
        options.qualityLevel,
        options.minDistance,
        options.blockSize,
        options.useHarrisDetector,
        options.k,
      ),
    );
  }

  createMSER(options: MSEROptions = {}): MSER {
    validateMSEROptions(options);
    return new MSER(
      this.#backend.MSERConfig.create(
        options.delta,
        options.minArea,
        options.maxArea,
        options.pass2Only,
      ),
    );
  }

  createMatVector(): MatVector {
    return new MatVector(this.#backend.matVectorNew());
  }

  createPointVector(): PointVector {
    return new PointVector();
  }

  createTonemapDrago(
    ...arguments_:
      | []
      | [gamma: number]
      | [gamma: number, saturation: number]
      | [gamma: number, saturation: number, bias: number]
  ): TonemapDrago {
    requireArityRange(arguments_.length, 0, 3, "TonemapDrago");
    const gamma = arguments_.length >= 1 ? toWasmF32(arguments_[0]) : undefined;
    const saturation = arguments_.length >= 2 ? toWasmF32(arguments_[1]) : undefined;
    const bias = arguments_.length === 3 ? toWasmF32(arguments_[2]) : undefined;
    return new TonemapDrago(this.#backend.TonemapDrago.create(gamma, saturation, bias));
  }

  createTonemapMantiuk(
    ...arguments_:
      | []
      | [gamma: number]
      | [gamma: number, scale: number]
      | [gamma: number, scale: number, saturation: number]
  ): TonemapMantiuk {
    requireArityRange(arguments_.length, 0, 3, "TonemapMantiuk");
    const gamma = arguments_.length >= 1 ? toWasmF32(arguments_[0]) : undefined;
    const scale = arguments_.length >= 2 ? toWasmF32(arguments_[1]) : undefined;
    const saturation = arguments_.length === 3 ? toWasmF32(arguments_[2]) : undefined;
    return new TonemapMantiuk(this.#backend.TonemapMantiuk.create(gamma, scale, saturation));
  }

  createTonemapReinhard(
    ...arguments_:
      | []
      | [gamma: number]
      | [gamma: number, intensity: number]
      | [gamma: number, intensity: number, lightAdaptation: number]
      | [gamma: number, intensity: number, lightAdaptation: number, colorAdaptation: number]
  ): TonemapReinhard {
    requireArityRange(arguments_.length, 0, 4, "TonemapReinhard");
    const gamma = arguments_.length >= 1 ? toWasmF32(arguments_[0]) : undefined;
    const intensity = arguments_.length >= 2 ? toWasmF32(arguments_[1]) : undefined;
    const lightAdaptation = arguments_.length >= 3 ? toWasmF32(arguments_[2]) : undefined;
    const colorAdaptation = arguments_.length === 4 ? toWasmF32(arguments_[3]) : undefined;
    return new TonemapReinhard(
      this.#backend.TonemapReinhard.create(gamma, intensity, lightAdaptation, colorAdaptation),
    );
  }

  createAgastFeatureDetector(options: AgastFeatureDetectorOptions = {}): AgastFeatureDetector {
    validateAgastFeatureDetectorOptions(options);
    return new AgastFeatureDetector(
      this.#backend.AgastFeatureDetector.create(
        options.threshold,
        options.nonmaxSuppression,
        options.type,
      ),
    );
  }

  createFastFeatureDetector(options: FastFeatureDetectorOptions = {}): FastFeatureDetector {
    validateFastFeatureDetectorOptions(options);
    return new FastFeatureDetector(
      this.#backend.FastFeatureDetector.create(
        options.threshold,
        options.nonmaxSuppression,
        options.type,
      ),
    );
  }

  determinant(source: Mat): number {
    requireExactArity(arguments.length, 1, "determinant");
    return this.#backend.matDeterminant(matHandleForBinding(source));
  }

  convertScaleAbs(
    ...arguments_: [source: Mat, destination: Mat, alpha?: number, beta?: number]
  ): void {
    requireArityRange(arguments_.length, 2, 4, "convertScaleAbs");
    const [source, destination] = arguments_;
    const alpha = arguments_.length >= 3 ? toWasmF64(arguments_[2]) : 1;
    const beta = arguments_.length === 4 ? toWasmF64(arguments_[3]) : 0;
    this.#backend.matConvertScaleAbsInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      alpha,
      beta,
    );
  }

  convertScaleAbsAlloc(source: Mat, alpha = 1, beta = 0): Mat {
    return new Mat(
      this.#backend.matConvertScaleAbs(
        source.handleForBackend(),
        toWasmF64(alpha),
        toWasmF64(beta),
      ),
    );
  }

  copyMakeBorder(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      top: number,
      bottom: number,
      left: number,
      right: number,
      borderType: BorderType,
      value?: Scalar,
    ]
  ): void {
    requireArityRange(arguments_.length, 7, 8, "copyMakeBorder");
    const [source, destination, top, bottom, left, right, borderType] = arguments_;
    const constant = arguments_.length === 8 ? scalarForBinding(arguments_[7]) : [0, 0, 0, 0];
    validateBorderSize(top, "top");
    validateBorderSize(bottom, "bottom");
    validateBorderSize(left, "left");
    validateBorderSize(right, "right");
    this.#backend.matCopyMakeBorderInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      top,
      bottom,
      left,
      right,
      toWasmI32(borderType),
      Float64Array.from(constant),
    );
  }

  copyMakeBorderAlloc(
    source: Mat,
    top: number,
    bottom: number,
    left: number,
    right: number,
    borderType: BorderType,
    constant: Scalar = [0, 0, 0, 0],
  ): Mat {
    validateBorderSize(top, "top");
    validateBorderSize(bottom, "bottom");
    validateBorderSize(left, "left");
    validateBorderSize(right, "right");
    validateFiniteNumbers({
      constant0: constant[0],
      constant1: constant[1],
      constant2: constant[2],
      constant3: constant[3],
    });
    return new Mat(
      this.#backend.matCopyMakeBorder(
        source.handleForBackend(),
        top,
        bottom,
        left,
        right,
        borderType,
        Float64Array.from(constant),
      ),
    );
  }

  cvtColor(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      code: ColorConversionCode,
      destinationChannels?: number,
    ]
  ): void {
    requireArityRange(arguments_.length, 3, 4, "cvtColor");
    const [source, destination, code] = arguments_;
    const destinationChannels = arguments_.length === 4 ? toWasmI32(arguments_[3]) : 0;
    this.#backend.matCvtColorInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(code),
      destinationChannels,
    );
  }

  divide(...arguments_: [a: Mat, b: Mat, destination: Mat, scale?: number, dtype?: number]): void {
    requireArityRange(arguments_.length, 3, 5, "divide");
    const [a, b, destination] = arguments_;
    const scale = arguments_.length >= 4 ? toWasmF64(arguments_[3]) : 1;
    const dtype = arguments_.length === 5 ? toWasmI32(arguments_[4]) : -1;
    this.#backend.matDivideInto(
      matHandleForBinding(a),
      matHandleForBinding(b),
      matHandleForBinding(destination),
      scale,
      dtype,
    );
  }

  divideAlloc(a: Mat, b: Mat, scale = 1): Mat {
    return new Mat(
      this.#backend.matDivide(a.handleForBackend(), b.handleForBackend(), toWasmF64(scale)),
    );
  }

  extractChannel(source: Mat, channel: number): Mat {
    validateChannelIndex(channel);
    return new Mat(this.#backend.matExtractChannel(source.handleForBackend(), channel));
  }

  ellipse2Poly(
    center: Point,
    axes: Size,
    rotationDegrees: number,
    arcStart: number,
    arcEnd: number,
    delta: number,
    points: PointVector,
  ): void {
    requireExactArity(arguments.length, 7, "ellipse2Poly");
    const flat = this.ellipsePolyFlat(center, axes, rotationDegrees, arcStart, arcEnd, delta);
    const vector = pointVectorHandleForBinding(points);
    for (let index = 0; index < flat.length; index += 2) {
      vector.push_back({ x: flat[index] ?? 0, y: flat[index + 1] ?? 0 });
    }
  }

  ellipsePolyFlat(
    center: Point,
    axes: Size,
    rotationDegrees: number,
    arcStart: number,
    arcEnd: number,
    delta: number,
  ): Int32Array {
    return this.#backend.ellipse2Poly(
      toWasmI32(center?.x),
      toWasmI32(center?.y),
      toWasmI32(axes?.width),
      toWasmI32(axes?.height),
      toWasmI32(rotationDegrees),
      toWasmI32(arcStart),
      toWasmI32(arcEnd),
      toWasmI32(delta),
    );
  }

  ellipse2PolyPoints(
    center: Point,
    axes: Size,
    rotationDegrees: number,
    arcStart: number,
    arcEnd: number,
    delta: number,
  ): Point[] {
    return pointsFromArray(
      this.ellipsePolyFlat(center, axes, rotationDegrees, arcStart, arcEnd, delta),
    );
  }

  exp(source: Mat, destination: Mat): void {
    requireExactArity(arguments.length, 2, "exp");
    this.#backend.matExpInto(matHandleForBinding(source), matHandleForBinding(destination));
  }

  expAlloc(source: Mat): Mat {
    return new Mat(this.#backend.matExp(source.handleForBackend()));
  }

  emptyMat(): Mat {
    return new Mat(this.#backend.matEmpty());
  }

  equalizeHist(source: Mat, destination: Mat): void {
    requireExactArity(arguments.length, 2, "equalizeHist");
    this.#backend.matEqualizeHistInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
    );
  }

  matchTemplate(image: Mat, template: Mat, result: Mat, method: TemplateMatchMode): void;
  matchTemplate(image: Mat, template: Mat, result: Mat, method: TemplateMatchMode, mask: Mat): void;
  matchTemplate(
    ...args:
      | [image: Mat, template: Mat, result: Mat, method: TemplateMatchMode]
      | [image: Mat, template: Mat, result: Mat, method: TemplateMatchMode, mask: Mat]
  ): void {
    requireArityRange(args.length, 4, 5, "matchTemplate");
    const image = matHandleForBinding(args[0]);
    const template = matHandleForBinding(args[1]);
    const result = matHandleForBinding(args[2]);
    const method = toWasmI32(args[3]);
    if (args.length === 4) {
      this.#backend.matMatchTemplateInto(image, template, result, method);
      return;
    }
    this.#backend.matMatchTemplateMaskedInto(
      image,
      template,
      result,
      method,
      matHandleForBinding(args[4]),
    );
  }

  flip(source: Mat, destination: Mat, flipCode: number): void {
    requireExactArity(arguments.length, 3, "flip");
    this.#backend.matFlipInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(flipCode),
    );
  }

  flipAlloc(source: Mat, flipCode: number): Mat {
    return new Mat(this.#backend.matFlip(source.handleForBackend(), toWasmI32(flipCode)));
  }

  findContours(
    ...arguments_:
      | [source: Mat, contours: MatVector, hierarchy: Mat, mode: number, method: number]
      | [
          source: Mat,
          contours: MatVector,
          hierarchy: Mat,
          mode: number,
          method: number,
          offset: Point,
        ]
  ): void {
    requireArityRange(arguments_.length, 5, 6, "findContours");
    const [source, contours, hierarchy, mode, method] = arguments_;
    const convertedOffset =
      arguments_.length === 6 ? point2iForBinding(arguments_[5]) : { x: 0, y: 0 };
    this.#backend.matFindContoursInto(
      matHandleForBinding(source),
      contours.handleForBackend(),
      matHandleForBinding(hierarchy),
      toWasmI32(mode),
      toWasmI32(method),
      convertedOffset.x,
      convertedOffset.y,
    );
  }

  rotate(source: Mat, destination: Mat, rotateCode: number): void {
    requireExactArity(arguments.length, 3, "rotate");
    const sourceHandle = matHandleForBinding(source);
    const destinationHandle = matHandleForBinding(destination);
    const code = toWasmI32(rotateCode);
    if (code < 0 || code > 2) return;
    this.#backend.matRotateInto(sourceHandle, destinationHandle, code);
  }

  rotateAlloc(source: Mat, rotateCode: number): Mat {
    return new Mat(this.#backend.matRotate(source.handleForBackend(), toWasmI32(rotateCode)));
  }

  grayscale(image: RgbaImage): RgbaImage {
    validateRgbaImage(image);
    const data = this.#backend.grayscaleRgba(image.data, image.width, image.height);
    return createRgbaImage(image.width, image.height, data);
  }

  GaussianBlur(
    source: Mat,
    destination: Mat,
    size: Size,
    sigmaX: number,
    sigmaY = 0,
    borderType: BorderType = BORDER_DEFAULT,
  ): void {
    requireArityRange(arguments.length, 4, 6, "GaussianBlur");
    const convertedSize = size2iForBinding(size);
    this.#backend.matGaussianBlurInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      convertedSize.width,
      convertedSize.height,
      toWasmF64(sigmaX),
      toWasmF64(sigmaY),
      toWasmI32(borderType),
    );
  }

  hconcat(sources: MatVector, destination: Mat): void {
    requireExactArity(arguments.length, 2, "hconcat");
    this.#backend.matHconcatVecInto(
      matVectorHandleForBinding(sources),
      matHandleForBinding(destination),
    );
  }

  hconcatPlanes(
    sources: readonly [Mat, Mat] | readonly [Mat, Mat, Mat] | readonly [Mat, Mat, Mat, Mat],
  ): Mat {
    return this.#concat(sources, "horizontal");
  }

  getAffineTransform(source: Mat, destination: Mat): Mat {
    requireExactArity(arguments.length, 2, "getAffineTransform");
    const sourceHandle = matHandleForBinding(source);
    const destinationHandle = matHandleForBinding(destination);
    return new Mat(this.#backend.matGetAffineTransform(sourceHandle, destinationHandle));
  }

  getLogLevel(): LogLevel {
    return logLevelFromNumber(this.#backend.getLogLevel());
  }

  getOptimalDFTSize(size: number): number {
    requireExactArity(arguments.length, 1, "getOptimalDFTSize");
    return this.#backend.getOptimalDFTSize(toWasmI32(size));
  }

  getPerspectiveTransform(
    ...arguments_:
      [source: Mat, destination: Mat] | [source: Mat, destination: Mat, solveMethod: number]
  ): Mat {
    requireArityRange(arguments_.length, 2, 3, "getPerspectiveTransform");
    return new Mat(
      this.#backend.matGetPerspectiveTransform(
        matHandleForBinding(arguments_[0]),
        matHandleForBinding(arguments_[1]),
        arguments_.length === 3 ? toWasmI32(arguments_[2]) : 0,
      ),
    );
  }

  getRotationMatrix2D(center: Point, angleDegrees: number, scale: number): Mat {
    requireExactArity(arguments.length, 3, "getRotationMatrix2D");
    const bindingCenter = point2fForBinding(center);
    return new Mat(
      this.#backend.matGetRotationMatrix2D(
        bindingCenter.x,
        bindingCenter.y,
        toWasmF64(angleDegrees),
        toWasmF64(scale),
      ),
    );
  }

  getStructuringElement(
    ...args:
      | [kind: StructuringElementKind, size: Size]
      | [kind: StructuringElementKind, size: Size, anchor: Point]
  ): Mat {
    requireOverloadArity(args.length, 2, 3, "getStructuringElement");
    const kind = toWasmI32(args[0]);
    const size = size2iForBinding(args[1]);
    const anchor = args.length === 2 ? { x: -1, y: -1 } : point2iForBinding(args[2]);
    if (size.width < 1 || size.height < 1) {
      throw new OpenCvInputError("structuring element dimensions must be at least 1");
    }
    return new Mat(
      this.#backend.getStructuringElement(kind, size.width, size.height, anchor.x, anchor.y),
    );
  }

  invert(image: RgbaImage): RgbaImage;
  invert(source: Mat, destination: Mat, method?: DecompositionMethod): number;
  invert(
    ...arguments_: [imageOrSource: RgbaImage | Mat, destination?: Mat, method?: DecompositionMethod]
  ): RgbaImage | number {
    const [imageOrSource, destination, method = 0] = arguments_;
    if (imageOrSource instanceof Mat) {
      if (destination === undefined) {
        throw new OpenCvInputError("matrix inverse requires a destination");
      }
      return this.#backend.matInvertInto(
        imageOrSource.handleForBackend(),
        destination.handleForBackend(),
        method,
      );
    }
    validateRgbaImage(imageOrSource);
    const data = this.#backend.invertRgba(
      imageOrSource.data,
      imageOrSource.width,
      imageOrSource.height,
    );
    return createRgbaImage(imageOrSource.width, imageOrSource.height, data);
  }

  matFromF32(rows: number, columns: number, channels: number, data: Float32Array): Mat {
    validateMatrixInput(rows, columns, channels, data.byteLength, Float32Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matFromF32(data, rows, columns, channels));
  }

  matFromF64(rows: number, columns: number, channels: number, data: Float64Array): Mat {
    validateMatrixInput(rows, columns, channels, data.byteLength, Float64Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matFromF64(data, rows, columns, channels));
  }

  matFromI16(rows: number, columns: number, channels: number, data: Int16Array): Mat {
    validateMatrixInput(rows, columns, channels, data.byteLength, Int16Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matFromI16(data, rows, columns, channels));
  }

  matFromI32(rows: number, columns: number, channels: number, data: Int32Array): Mat {
    validateMatrixInput(rows, columns, channels, data.byteLength, Int32Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matFromI32(data, rows, columns, channels));
  }

  matFromI8(rows: number, columns: number, channels: number, data: Int8Array): Mat {
    validateMatrixInput(rows, columns, channels, data.byteLength, Int8Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matFromI8(data, rows, columns, channels));
  }

  matFromU16(rows: number, columns: number, channels: number, data: Uint16Array): Mat {
    validateMatrixInput(rows, columns, channels, data.byteLength, Uint16Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matFromU16(data, rows, columns, channels));
  }

  matFromU8(rows: number, columns: number, channels: number, data: Uint8Array): Mat {
    validateMatrixInput(rows, columns, channels, data.byteLength);
    return new Mat(this.#backend.matFromU8(data, rows, columns, channels));
  }

  inRange(source: Mat, lowerBound: Mat, upperBound: Mat, destination: Mat): void {
    requireExactArity(arguments.length, 4, "inRange");
    this.#backend.matInRangeInto(
      matHandleForBinding(source),
      matHandleForBinding(lowerBound),
      matHandleForBinding(upperBound),
      matHandleForBinding(destination),
    );
  }

  insertChannel(source: Mat, destination: Mat, channel: number): void {
    validateChannelIndex(channel);
    this.#backend.matInsertChannel(
      source.handleForBackend(),
      destination.handleForBackend(),
      channel,
    );
  }

  invertAffineTransform(transform: Mat, destination: Mat): void {
    requireExactArity(arguments.length, 2, "invertAffineTransform");
    this.#backend.matInvertAffineTransformInto(
      matHandleForBinding(transform),
      matHandleForBinding(destination),
    );
  }

  invertAffineTransformAlloc(transform: Mat): Mat {
    return new Mat(this.#backend.matInvertAffineTransform(transform.handleForBackend()));
  }

  isContourConvex(contour: Mat): boolean {
    requireExactArity(arguments.length, 1, "isContourConvex");
    return this.#backend.matIsContourConvex(matHandleForBinding(contour));
  }

  log(source: Mat, destination: Mat): void {
    requireExactArity(arguments.length, 2, "log");
    this.#backend.matLogInto(matHandleForBinding(source), matHandleForBinding(destination));
  }

  logAlloc(source: Mat): Mat {
    return new Mat(this.#backend.matLog(source.handleForBackend()));
  }

  LUT(source: Mat, table: Mat, destination: Mat): void {
    this.#backend.matLutInto(
      source.handleForBackend(),
      table.handleForBackend(),
      destination.handleForBackend(),
    );
  }

  lut(source: Mat, table: Mat): Mat;
  lut(source: Mat, table: Mat, destination: Mat): void;
  lut(source: Mat, table: Mat, destination?: Mat): Mat | void {
    if (destination !== undefined) {
      this.#backend.matLutInto(
        source.handleForBackend(),
        table.handleForBackend(),
        destination.handleForBackend(),
      );
      return;
    }
    return new Mat(this.#backend.matLut(source.handleForBackend(), table.handleForBackend()));
  }

  magnitude(x: Mat, y: Mat, destination: Mat): void {
    requireExactArity(arguments.length, 3, "magnitude");
    this.#backend.matMagnitudeInto(
      matHandleForBinding(x),
      matHandleForBinding(y),
      matHandleForBinding(destination),
    );
  }

  magnitudeAlloc(x: Mat, y: Mat): Mat {
    return new Mat(this.#backend.matMagnitude(x.handleForBackend(), y.handleForBackend()));
  }

  max(source1: Mat, source2: Mat, destination: Mat): void {
    requireExactArity(arguments.length, 3, "max");
    this.#backend.matMaxInto(
      matHandleForBinding(source1),
      matHandleForBinding(source2),
      matHandleForBinding(destination),
    );
  }

  mean(...arguments_: [source: Mat] | [source: Mat, mask: Mat]): Scalar {
    requireOverloadArity(arguments_.length, 1, 2, "mean");
    const source = matHandleForBinding(arguments_[0]);
    const values =
      arguments_.length === 1
        ? this.#backend.matMean(source)
        : this.#backend.matMeanMasked(source, matHandleForBinding(arguments_[1]));
    return scalarFromArray(values);
  }

  meanStdDev(
    ...arguments_:
      | [source: Mat, means: Mat, standardDeviations: Mat]
      | [source: Mat, means: Mat, standardDeviations: Mat, mask: Mat]
  ): void {
    requireOverloadArity(arguments_.length, 3, 4, "meanStdDev");
    const [source, means, standardDeviations, mask] = arguments_;
    if (mask === undefined) {
      this.#backend.matMeanStdDevInto(
        matHandleForBinding(source),
        matHandleForBinding(means),
        matHandleForBinding(standardDeviations),
      );
      return;
    }
    this.#backend.matMeanStdDevMaskedInto(
      matHandleForBinding(source),
      matHandleForBinding(means),
      matHandleForBinding(standardDeviations),
      matHandleForBinding(mask),
    );
  }

  merge(sources: MatVector, destination: Mat): void {
    requireExactArity(arguments.length, 2, "merge");
    this.#backend.matMergeVecInto(
      matVectorHandleForBinding(sources),
      matHandleForBinding(destination),
    );
  }

  mergePlanes(
    sources: readonly [Mat, Mat] | readonly [Mat, Mat, Mat] | readonly [Mat, Mat, Mat, Mat],
  ): Mat {
    const handles = sources.map((source) => source.handleForBackend());
    const first = handles[0];
    const second = handles[1];
    if (first === undefined || second === undefined) {
      throw new OpenCvInputError("merge requires two through four matrices");
    }
    if (handles.length === 2) {
      return new Mat(this.#backend.matMerge(first, second));
    }
    const third = handles[2];
    if (third === undefined) {
      throw new OpenCvInputError("merge requires a third matrix");
    }
    if (handles.length === 3) {
      return new Mat(this.#backend.matMerge3(first, second, third));
    }
    const fourth = handles[3];
    if (fourth === undefined) {
      throw new OpenCvInputError("merge requires a fourth matrix");
    }
    return new Mat(this.#backend.matMerge4(first, second, third, fourth));
  }

  min(source1: Mat, source2: Mat, destination: Mat): void {
    requireExactArity(arguments.length, 3, "min");
    this.#backend.matMinInto(
      matHandleForBinding(source1),
      matHandleForBinding(source2),
      matHandleForBinding(destination),
    );
  }

  mixChannels(sources: MatVector, destinations: MatVector, fromTo: ArrayLike<number>): void {
    requireExactArity(arguments.length, 3, "mixChannels");
    this.#backend.matMixChannelsVec(
      matVectorHandleForBinding(sources),
      matVectorHandleForBinding(destinations),
      Uint16Array.from(fromTo),
    );
  }

  mixChannelsMat(source: Mat, destination: Mat, fromTo: Uint16Array): void {
    if (fromTo.length % 2 !== 0) {
      throw new OpenCvInputError("fromTo must contain source/destination channel pairs");
    }
    this.#backend.matMixChannels(source.handleForBackend(), destination.handleForBackend(), fromTo);
  }

  multiply(
    ...arguments_: [a: Mat, b: Mat, destination: Mat, scale?: number, dtype?: number]
  ): void {
    requireArityRange(arguments_.length, 3, 5, "multiply");
    const [a, b, destination] = arguments_;
    const scale = arguments_.length >= 4 ? toWasmF64(arguments_[3]) : 1;
    const dtype = arguments_.length === 5 ? toWasmI32(arguments_[4]) : -1;
    this.#backend.matMultiplyInto(
      matHandleForBinding(a),
      matHandleForBinding(b),
      matHandleForBinding(destination),
      scale,
      dtype,
    );
  }

  multiplyAlloc(a: Mat, b: Mat, scale = 1): Mat {
    return new Mat(
      this.#backend.matMultiply(a.handleForBackend(), b.handleForBackend(), toWasmF64(scale)),
    );
  }

  morphologyEx(
    source: Mat,
    destination: Mat,
    operation: number,
    kernel: Mat,
    anchor: Point = { x: -1, y: -1 },
    iterations = 1,
    borderType: BorderType = BORDER_CONSTANT,
    borderValue?: Scalar,
  ): void {
    requireArityRange(arguments.length, 4, 8, "morphologyEx");
    const convertedAnchor = point2iForBinding(anchor);
    this.#backend.matMorphologyExInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(operation),
      matHandleForBinding(kernel),
      convertedAnchor.x,
      convertedAnchor.y,
      toWasmI32(iterations),
      toWasmI32(borderType),
      borderValue === undefined ? new Float64Array() : scalarForBinding(borderValue),
      borderValue === undefined,
    );
  }

  erode(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      kernel: Mat,
      anchor?: Point,
      iterations?: number,
      borderType?: BorderType,
      borderValue?: Scalar,
    ]
  ): void {
    requireArityRange(arguments_.length, 3, 7, "erode");
    const [
      source,
      destination,
      kernel,
      anchor = { x: -1, y: -1 },
      iterations = 1,
      borderType = BORDER_CONSTANT,
      borderValue,
    ] = arguments_;
    const convertedAnchor = point2iForBinding(anchor);
    this.#backend.matErodeInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      matHandleForBinding(kernel),
      convertedAnchor.x,
      convertedAnchor.y,
      toWasmI32(iterations),
      toWasmI32(borderType),
      borderValue === undefined ? new Float64Array() : scalarForBinding(borderValue),
      borderValue === undefined,
    );
  }

  dilate(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      kernel: Mat,
      anchor?: Point,
      iterations?: number,
      borderType?: BorderType,
      borderValue?: Scalar,
    ]
  ): void {
    requireArityRange(arguments_.length, 3, 7, "dilate");
    const [
      source,
      destination,
      kernel,
      anchor = { x: -1, y: -1 },
      iterations = 1,
      borderType = BORDER_CONSTANT,
      borderValue,
    ] = arguments_;
    const convertedAnchor = point2iForBinding(anchor);
    this.#backend.matDilateInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      matHandleForBinding(kernel),
      convertedAnchor.x,
      convertedAnchor.y,
      toWasmI32(iterations),
      toWasmI32(borderType),
      borderValue === undefined ? new Float64Array() : scalarForBinding(borderValue),
      borderValue === undefined,
    );
  }

  blur(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      kernelSize: Size,
      anchor?: Point,
      borderType?: BorderType,
    ]
  ): void {
    requireArityRange(arguments_.length, 3, 5, "blur");
    const [
      source,
      destination,
      kernelSize,
      anchor = { x: -1, y: -1 },
      borderType = BORDER_DEFAULT,
    ] = arguments_;
    const convertedSize = size2iForBinding(kernelSize);
    const convertedAnchor = point2iForBinding(anchor);
    this.#backend.matBlurInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      convertedSize.width,
      convertedSize.height,
      convertedAnchor.x,
      convertedAnchor.y,
      toWasmI32(borderType),
    );
  }

  boxFilter(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      destinationDepth: number,
      kernelSize: Size,
      anchor?: Point,
      normalize?: boolean,
      borderType?: BorderType,
    ]
  ): void {
    requireArityRange(arguments_.length, 4, 7, "boxFilter");
    const [
      source,
      destination,
      destinationDepth,
      kernelSize,
      anchor = { x: -1, y: -1 },
      normalize = true,
      borderType = BORDER_DEFAULT,
    ] = arguments_;
    const convertedSize = size2iForBinding(kernelSize);
    const convertedAnchor = point2iForBinding(anchor);
    this.#backend.matBoxFilterInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(destinationDepth),
      convertedSize.width,
      convertedSize.height,
      convertedAnchor.x,
      convertedAnchor.y,
      normalize,
      toWasmI32(borderType),
    );
  }

  medianBlur(source: Mat, destination: Mat, kernelSize: number): void {
    requireExactArity(arguments.length, 3, "medianBlur");
    this.#backend.matMedianBlurInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(kernelSize),
    );
  }

  filter2D(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      destinationDepth: number,
      kernel: Mat,
      anchor?: Point,
      delta?: number,
      borderType?: BorderType,
    ]
  ): void {
    requireArityRange(arguments_.length, 4, 7, "filter2D");
    const [
      source,
      destination,
      destinationDepth,
      kernel,
      anchor = { x: -1, y: -1 },
      delta = 0,
      borderType = BORDER_DEFAULT,
    ] = arguments_;
    const convertedAnchor = point2iForBinding(anchor);
    this.#backend.matFilter2DInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(destinationDepth),
      matHandleForBinding(kernel),
      convertedAnchor.x,
      convertedAnchor.y,
      toWasmF64(delta),
      toWasmI32(borderType),
    );
  }

  sepFilter2D(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      destinationDepth: number,
      kernelX: Mat,
      kernelY: Mat,
      anchor?: Point,
      delta?: number,
      borderType?: BorderType,
    ]
  ): void {
    requireArityRange(arguments_.length, 5, 8, "sepFilter2D");
    const [
      source,
      destination,
      destinationDepth,
      kernelX,
      kernelY,
      anchor = { x: -1, y: -1 },
      delta = 0,
      borderType = BORDER_DEFAULT,
    ] = arguments_;
    const convertedAnchor = point2iForBinding(anchor);
    this.#backend.matSepFilter2DInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(destinationDepth),
      matHandleForBinding(kernelX),
      matHandleForBinding(kernelY),
      convertedAnchor.x,
      convertedAnchor.y,
      toWasmF64(delta),
      toWasmI32(borderType),
    );
  }

  sqrBoxFilter(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      destinationDepth: number,
      kernelSize: Size,
      anchor?: Point,
      normalize?: boolean,
      borderType?: BorderType,
    ]
  ): void {
    requireArityRange(arguments_.length, 4, 7, "sqrBoxFilter");
    const [
      source,
      destination,
      destinationDepth,
      kernelSize,
      anchor = { x: -1, y: -1 },
      normalize = true,
      borderType = BORDER_DEFAULT,
    ] = arguments_;
    const convertedSize = size2iForBinding(kernelSize);
    const convertedAnchor = point2iForBinding(anchor);
    this.#backend.matSqrBoxFilterInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(destinationDepth),
      convertedSize.width,
      convertedSize.height,
      convertedAnchor.x,
      convertedAnchor.y,
      normalize,
      toWasmI32(borderType),
    );
  }

  stackBlur(source: Mat, destination: Mat, kernelSize: Size): void {
    requireExactArity(arguments.length, 3, "stackBlur");
    const convertedSize = size2iForBinding(kernelSize);
    this.#backend.matStackBlurInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      convertedSize.width,
      convertedSize.height,
    );
  }

  bilateralFilter(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      diameter: number,
      sigmaColor: number,
      sigmaSpace: number,
      borderType?: BorderType,
    ]
  ): void {
    requireArityRange(arguments_.length, 5, 6, "bilateralFilter");
    const [source, destination, diameter, sigmaColor, sigmaSpace, borderType = BORDER_DEFAULT] =
      arguments_;
    this.#backend.matBilateralFilterInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(diameter),
      toWasmF64(sigmaColor),
      toWasmF64(sigmaSpace),
      toWasmI32(borderType),
    );
  }

  pyrDown(
    ...arguments_: [source: Mat, destination: Mat, destinationSize?: Size, borderType?: BorderType]
  ): void {
    requireArityRange(arguments_.length, 2, 4, "pyrDown");
    const [source, destination, destinationSize, borderType = BORDER_DEFAULT] = arguments_;
    const convertedSize =
      destinationSize === undefined ? { width: -1, height: -1 } : size2iForBinding(destinationSize);
    this.#backend.matPyrDownInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      convertedSize.height,
      convertedSize.width,
      toWasmI32(borderType),
    );
  }

  pyrUp(
    ...arguments_: [source: Mat, destination: Mat, destinationSize?: Size, borderType?: BorderType]
  ): void {
    requireArityRange(arguments_.length, 2, 4, "pyrUp");
    const [source, destination, destinationSize, borderType = BORDER_DEFAULT] = arguments_;
    const convertedSize =
      destinationSize === undefined ? { width: -1, height: -1 } : size2iForBinding(destinationSize);
    this.#backend.matPyrUpInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      convertedSize.height,
      convertedSize.width,
      toWasmI32(borderType),
    );
  }

  Laplacian(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      destinationDepth: number,
      kernelSize?: number,
      scale?: number,
      delta?: number,
      borderType?: BorderType,
    ]
  ): void {
    requireArityRange(arguments_.length, 3, 7, "Laplacian");
    const [
      source,
      destination,
      destinationDepth,
      kernelSize = 1,
      scale = 1,
      delta = 0,
      borderType = BORDER_DEFAULT,
    ] = arguments_;
    this.#backend.matLaplacianInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(destinationDepth),
      toWasmI32(kernelSize),
      toWasmF64(scale),
      toWasmF64(delta),
      toWasmI32(borderType),
    );
  }

  Scharr(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      destinationDepth: number,
      dx: number,
      dy: number,
      scale?: number,
      delta?: number,
      borderType?: BorderType,
    ]
  ): void {
    requireArityRange(arguments_.length, 5, 8, "Scharr");
    const [
      source,
      destination,
      destinationDepth,
      dx,
      dy,
      scale = 1,
      delta = 0,
      borderType = BORDER_DEFAULT,
    ] = arguments_;
    this.#backend.matScharrInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(destinationDepth),
      toWasmI32(dx),
      toWasmI32(dy),
      toWasmF64(scale),
      toWasmF64(delta),
      toWasmI32(borderType),
    );
  }

  spatialGradient(
    ...arguments_: [
      source: Mat,
      destinationX: Mat,
      destinationY: Mat,
      kernelSize?: number,
      borderType?: BorderType,
    ]
  ): void {
    requireArityRange(arguments_.length, 3, 5, "spatialGradient");
    const [source, destinationX, destinationY, kernelSize = 3, borderType = BORDER_DEFAULT] =
      arguments_;
    if (kernelSize !== 3) {
      throw new Error("spatialGradient only supports a 3x3 kernel");
    }
    this.#backend.matSpatialGradientInto(
      matHandleForBinding(source),
      matHandleForBinding(destinationX),
      matHandleForBinding(destinationY),
      toWasmI32(borderType),
    );
  }

  preCornerDetect(
    ...arguments_: [source: Mat, destination: Mat, kernelSize: number, borderType?: BorderType]
  ): void {
    requireArityRange(arguments_.length, 3, 4, "preCornerDetect");
    const [source, destination, kernelSize, borderType = BORDER_DEFAULT] = arguments_;
    this.#backend.matPreCornerDetectInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(kernelSize),
      toWasmI32(borderType),
    );
  }

  cornerHarris(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      blockSize: number,
      kernelSize: number,
      k: number,
      borderType?: BorderType,
    ]
  ): void {
    requireArityRange(arguments_.length, 5, 6, "cornerHarris");
    const [source, destination, blockSize, kernelSize, k, borderType = BORDER_DEFAULT] = arguments_;
    this.#backend.matCornerHarrisInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(blockSize),
      toWasmI32(kernelSize),
      toWasmF64(k),
      toWasmI32(borderType),
    );
  }

  cornerMinEigenVal(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      blockSize: number,
      kernelSize?: number,
      borderType?: BorderType,
    ]
  ): void {
    requireArityRange(arguments_.length, 3, 5, "cornerMinEigenVal");
    const [source, destination, blockSize, kernelSize = 3, borderType = BORDER_DEFAULT] =
      arguments_;
    this.#backend.matCornerMinEigenValInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(blockSize),
      toWasmI32(kernelSize),
      toWasmI32(borderType),
    );
  }

  adaptiveThreshold(
    source: Mat,
    destination: Mat,
    maxValue: number,
    adaptiveMethod: AdaptiveThresholdType,
    thresholdType: ThresholdType,
    blockSize: number,
    delta: number,
  ): void {
    this.#backend.matAdaptiveThresholdInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      maxValue,
      toWasmI32(adaptiveMethod),
      toWasmI32(thresholdType),
      toWasmI32(blockSize),
      delta,
    );
  }

  applyColorMap(source: Mat, destination: Mat, colormap: ColormapType): void {
    this.#backend.matApplyColorMapInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(colormap),
    );
  }

  convexHull(
    ...arguments_: [points: Mat, hull: Mat, clockwise?: boolean, returnPoints?: boolean]
  ): void {
    requireArityRange(arguments_.length, 2, 4, "convexHull");
    const [points, hull, clockwise = false, returnPoints = true] = arguments_;
    this.#backend.matConvexHullInto(
      matHandleForBinding(points),
      matHandleForBinding(hull),
      clockwise,
      returnPoints,
    );
  }

  minAreaRect(points: Mat): RotatedRect {
    const rect = this.#backend.matMinAreaRect(matHandleForBinding(points));
    const [x, y, width, height, angle] = [
      requiredFloat(rect, 0),
      requiredFloat(rect, 1),
      requiredFloat(rect, 2),
      requiredFloat(rect, 3),
      requiredFloat(rect, 4),
    ];
    return { center: { x, y }, size: { width, height }, angle };
  }

  minEnclosingCircle(points: Mat): Circle {
    const circle = this.#backend.matMinEnclosingCircle(matHandleForBinding(points));
    const [x, y, radius] = [
      requiredFloat(circle, 0),
      requiredFloat(circle, 1),
      requiredFloat(circle, 2),
    ];
    return { center: { x, y }, radius };
  }

  fitEllipse(points: Mat): RotatedRect {
    const ellipse = this.#backend.matFitEllipse(matHandleForBinding(points));
    const [x, y, width, height, angle] = [
      requiredFloat(ellipse, 0),
      requiredFloat(ellipse, 1),
      requiredFloat(ellipse, 2),
      requiredFloat(ellipse, 3),
      requiredFloat(ellipse, 4),
    ];
    return { center: { x, y }, size: { width, height }, angle };
  }

  fitLine(
    points: Mat,
    line: Mat,
    distType: DistType,
    param: number,
    reps: number,
    aeps: number,
  ): void {
    this.#backend.matFitLineInto(
      matHandleForBinding(points),
      matHandleForBinding(line),
      toWasmI32(distType),
      param,
      reps,
      aeps,
    );
  }

  remap(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      map1: Mat,
      map2: Mat,
      interpolation: Interpolation,
      borderType?: BorderType,
      borderValue?: Scalar,
    ]
  ): void {
    requireArityRange(arguments_.length, 5, 7, "remap");
    const [source, destination, map1, map2, interpolation, _borderType, borderValue] = arguments_;
    this.#backend.matRemapInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      matHandleForBinding(map1),
      matHandleForBinding(map2),
      toWasmI32(interpolation),
      borderValue === undefined ? 0 : requiredFloat(scalarForBinding(borderValue), 0),
    );
  }

  convertMaps(
    ...arguments_: [
      map1: Mat,
      map2: Mat,
      dst1: Mat,
      dst2: Mat,
      destinationType?: number,
      nninterpolation?: boolean,
    ]
  ): void {
    requireArityRange(arguments_.length, 4, 6, "convertMaps");
    const [map1, map2, dst1, dst2] = arguments_;
    this.#backend.matConvertMapsInto(
      matHandleForBinding(map1),
      matHandleForBinding(map2),
      matHandleForBinding(dst1),
      matHandleForBinding(dst2),
    );
  }

  getRectSubPix(
    ...arguments_: [
      source: Mat,
      patchSize: Size,
      center: Point,
      destination: Mat,
      patchType?: number,
    ]
  ): void {
    requireArityRange(arguments_.length, 4, 5, "getRectSubPix");
    const [source, patchSize, center, destination] = arguments_;
    this.#backend.matGetRectSubPixInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(patchSize.width),
      toWasmI32(patchSize.height),
      center.x,
      center.y,
    );
  }

  demosaicing(...arguments_: [source: Mat, destination: Mat, code: number, dstCn?: number]): void {
    requireArityRange(arguments_.length, 3, 4, "demosaicing");
    const [source, destination, code] = arguments_;
    this.#backend.matDemosaicingInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(code),
    );
  }

  calcHist(
    ...arguments_: [
      image: Mat,
      channels: number[],
      mask: Mat,
      hist: Mat,
      histSize: number[],
      ranges: number[],
      accumulate?: boolean,
    ]
  ): void {
    requireArityRange(arguments_.length, 6, 7, "calcHist");
    const [image, _channels, _mask, hist, histSize, ranges, accumulate = false] = arguments_;
    const source = image instanceof MatVector ? image.get(0) : image;
    this.#backend.matCalcHistInto(
      matHandleForBinding(source),
      matHandleForBinding(hist),
      toWasmI32(histSize[0] ?? 0),
      ranges[0] ?? 0,
      ranges[1] ?? 0,
      accumulate,
    );
  }

  kmeans(
    ...arguments_: [
      data: Mat,
      k: number,
      bestLabels: Mat,
      criteria: TermCriteria,
      attempts: number,
      flags: number,
      centers?: Mat,
    ]
  ): number {
    requireArityRange(arguments_.length, 6, 7, "kmeans");
    const [data, k, bestLabels, criteria, _attempts, _flags, centers] = arguments_;
    return this.#backend.matKmeansInto(
      matHandleForBinding(data),
      toWasmI32(k),
      matHandleForBinding(bestLabels),
      centers === undefined ? this.#backend.matEmpty() : matHandleForBinding(centers),
      toWasmI32(criteria.maxCount),
      criteria.epsilon,
      1,
    );
  }

  dft(...arguments_: [source: Mat, destination: Mat, flags?: number, nonzeroRows?: number]): void {
    requireArityRange(arguments_.length, 2, 4, "dft");
    const [source, destination, flags = 0] = arguments_;
    this.#backend.matDftInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(flags),
    );
  }

  calcBackProject(
    image: Mat | MatVector,
    channels: number[],
    hist: Mat,
    destination: Mat,
    ranges: number[],
    scale: number,
  ): void {
    const source = image instanceof MatVector ? image.get(0) : image;
    this.#backend.matCalcBackProjectInto(
      matHandleForBinding(source),
      channels[0] ?? 0,
      matHandleForBinding(hist),
      matHandleForBinding(destination),
      ranges[0] ?? 0,
      ranges[1] ?? 0,
      scale,
    );
  }

  warpPolar(
    source: Mat,
    destination: Mat,
    dsize: Size,
    center: Point,
    maxRadius: number,
    flags: number,
  ): void {
    this.#backend.matWarpPolarInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(dsize.width),
      toWasmI32(dsize.height),
      center.x,
      center.y,
      maxRadius,
      toWasmI32(flags),
    );
  }

  groupRectangles(
    ...arguments_: [rects: Mat, weights: Mat, groupThreshold: number, eps?: number]
  ): number {
    requireArityRange(arguments_.length, 3, 4, "groupRectangles");
    const [rects, _weights, groupThreshold, eps = 0.2] = arguments_;
    return this.#backend.matGroupRectanglesInto(
      matHandleForBinding(rects),
      this.#backend.matEmpty(),
      toWasmI32(groupThreshold),
      eps,
    );
  }

  line(
    ...arguments_: [
      image: Mat,
      pt1: Point,
      pt2: Point,
      color: Scalar,
      thickness?: number,
      lineType?: number,
      shift?: number,
    ]
  ): void {
    requireArityRange(arguments_.length, 4, 7, "line");
    const [image, pt1, pt2, color, thickness = 1, lineType = 8, shift = 0] = arguments_;
    this.#backend.matLineInto(
      matHandleForBinding(image),
      toWasmI32(pt1.x),
      toWasmI32(pt1.y),
      toWasmI32(pt2.x),
      toWasmI32(pt2.y),
      scalarForBinding(color),
      toWasmI32(thickness),
      toWasmI32(lineType),
      toWasmI32(shift),
    );
  }

  rectangle(
    ...arguments_: [
      image: Mat,
      pt1: Point,
      pt2: Point,
      color: Scalar,
      thickness?: number,
      lineType?: number,
      shift?: number,
    ]
  ): void {
    requireArityRange(arguments_.length, 4, 7, "rectangle");
    const [image, pt1, pt2, color, thickness = 1] = arguments_;
    this.#backend.matRectangleInto(
      matHandleForBinding(image),
      toWasmI32(pt1.x),
      toWasmI32(pt1.y),
      toWasmI32(pt2.x),
      toWasmI32(pt2.y),
      scalarForBinding(color),
      toWasmI32(thickness),
    );
  }

  circle(
    ...arguments_: [
      image: Mat,
      center: Point,
      radius: number,
      color: Scalar,
      thickness?: number,
      lineType?: number,
      shift?: number,
    ]
  ): void {
    requireArityRange(arguments_.length, 4, 7, "circle");
    const [image, center, radius, color, thickness = 1] = arguments_;
    this.#backend.matCircleInto(
      matHandleForBinding(image),
      toWasmI32(center.x),
      toWasmI32(center.y),
      toWasmI32(radius),
      scalarForBinding(color),
      toWasmI32(thickness),
    );
  }

  ellipse(
    ...arguments_: [
      image: Mat,
      center: Point,
      axes: Size,
      angle: number,
      startAngle: number,
      endAngle: number,
      color: Scalar,
      thickness?: number,
      lineType?: number,
      shift?: number,
    ]
  ): void {
    requireArityRange(arguments_.length, 7, 10, "ellipse");
    const [image, center, axes, angle, startAngle, endAngle, color, thickness = 1] = arguments_;
    this.#backend.matEllipseInto(
      matHandleForBinding(image),
      center.x,
      center.y,
      axes.width / 2,
      axes.height / 2,
      angle,
      startAngle,
      endAngle,
      scalarForBinding(color),
      toWasmI32(thickness),
    );
  }

  fillPoly(
    ...arguments_: [
      image: Mat,
      pts: MatVector | Mat,
      color: Scalar,
      lineType?: number,
      shift?: number,
      offset?: Point,
    ]
  ): void {
    requireArityRange(arguments_.length, 3, 6, "fillPoly");
    const [image, pts, color] = arguments_;
    this.#backend.matFillPolyInto(
      matHandleForBinding(image),
      contourPoints(pts),
      true,
      scalarForBinding(color),
    );
  }

  drawMarker(
    ...arguments_: [
      image: Mat,
      position: Point,
      color: Scalar,
      markerType?: number,
      markerSize?: number,
      thickness?: number,
      lineType?: number,
    ]
  ): void {
    requireArityRange(arguments_.length, 3, 7, "drawMarker");
    const [image, position, color, markerType = 0, markerSize = 20, thickness = 1] = arguments_;
    this.#backend.matDrawMarkerInto(
      matHandleForBinding(image),
      toWasmI32(position.x),
      toWasmI32(position.y),
      scalarForBinding(color),
      toWasmI32(markerType),
      toWasmI32(markerSize),
      toWasmI32(thickness),
    );
  }

  polylines(
    ...arguments_: [
      image: Mat,
      pts: MatVector | Mat,
      isClosed: boolean,
      color: Scalar,
      thickness?: number,
      lineType?: number,
      shift?: number,
    ]
  ): void {
    requireArityRange(arguments_.length, 4, 7, "polylines");
    const [image, pts, isClosed, color, thickness = 1] = arguments_;
    this.#backend.matPolylinesInto(
      matHandleForBinding(image),
      contourPoints(pts),
      isClosed,
      scalarForBinding(color),
      toWasmI32(thickness),
    );
  }

  compareHist(h1: Mat, h2: Mat, method: number): number {
    return this.#backend.matCompareHist(
      matHandleForBinding(h1),
      matHandleForBinding(h2),
      toWasmI32(method),
    );
  }

  gemm(
    ...arguments_: [
      src1: Mat,
      src2: Mat,
      alpha: number,
      src3: Mat,
      beta: number,
      destination: Mat,
      flags?: number,
    ]
  ): void {
    requireArityRange(arguments_.length, 6, 7, "gemm");
    const [src1, src2, alpha, src3, beta, destination, flags = 0] = arguments_;
    this.#backend.matGemmInto(
      matHandleForBinding(src1),
      matHandleForBinding(src2),
      alpha,
      matHandleForBinding(src3),
      beta,
      matHandleForBinding(destination),
      toWasmI32(flags),
    );
  }

  eigen(...arguments_: [source: Mat, eigenvalues: Mat, eigenvectors?: Mat]): boolean {
    requireArityRange(arguments_.length, 2, 3, "eigen");
    const [source, eigenvalues, eigenvectors] = arguments_;
    return this.#backend.matEigenInto(
      matHandleForBinding(source),
      matHandleForBinding(eigenvalues),
      eigenvectors === undefined ? this.#backend.matEmpty() : matHandleForBinding(eigenvectors),
    );
  }

  solvePoly(...arguments_: [coeffs: Mat, roots: Mat, maxIters?: number]): number {
    requireArityRange(arguments_.length, 2, 3, "solvePoly");
    const [coeffs, roots, maxIters = 300] = arguments_;
    return this.#backend.matSolvePolyInto(
      matHandleForBinding(coeffs),
      matHandleForBinding(roots),
      toWasmI32(maxIters),
    );
  }

  connectedComponents(
    ...arguments_: [image: Mat, labels: Mat, connectivity?: number, labelType?: number]
  ): number {
    requireArityRange(arguments_.length, 2, 4, "connectedComponents");
    const [image, labels, connectivity = 8, labelType = 4] = arguments_;
    return this.#backend.matConnectedComponentsInto(
      matHandleForBinding(image),
      matHandleForBinding(labels),
      toWasmI32(connectivity),
      toWasmI32(labelType),
    );
  }

  connectedComponentsWithStats(
    ...arguments_: [
      image: Mat,
      labels: Mat,
      stats: Mat,
      centroids: Mat,
      connectivity?: number,
      labelType?: number,
    ]
  ): number {
    requireArityRange(arguments_.length, 4, 6, "connectedComponentsWithStats");
    const [image, labels, stats, centroids, connectivity = 8, labelType = 4] = arguments_;
    return this.#backend.matConnectedComponentsWithStatsInto(
      matHandleForBinding(image),
      matHandleForBinding(labels),
      matHandleForBinding(stats),
      matHandleForBinding(centroids),
      toWasmI32(connectivity),
      toWasmI32(labelType),
    );
  }

  distanceTransform(
    ...arguments_: [source: Mat, destination: Mat, distanceType?: number, maskSize?: number]
  ): void {
    requireArityRange(arguments_.length, 2, 4, "distanceTransform");
    const [source, destination, distanceType = 2, maskSize = 3] = arguments_;
    this.#backend.matDistanceTransformInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(distanceType),
      toWasmI32(maskSize),
    );
  }

  floodFill(
    ...arguments_: [
      image: Mat,
      mask: Mat,
      seedPoint: Point,
      newValue: Scalar,
      rect?: Rect,
      loDiff?: Scalar,
      upDiff?: Scalar,
      flags?: number,
    ]
  ): number {
    requireArityRange(arguments_.length, 4, 8, "floodFill");
    const [
      image,
      mask,
      seedPoint,
      newValue,
      _rect,
      loDiff = [0, 0, 0, 0],
      upDiff = [0, 0, 0, 0],
      flags = 4,
    ] = arguments_;
    return this.#backend.matFloodFillInto(
      matHandleForBinding(image),
      matHandleForBinding(mask),
      toWasmI32(seedPoint.x),
      toWasmI32(seedPoint.y),
      requiredFloat(scalarForBinding(newValue), 0),
      requiredFloat(scalarForBinding(loDiff), 0),
      requiredFloat(scalarForBinding(upDiff), 0),
      toWasmI32(flags),
    );
  }

  integral(
    ...arguments_: [source: Mat, sum: Mat, sumDepth?: number, squaredSumDepth?: number]
  ): void {
    requireArityRange(arguments_.length, 2, 4, "integral");
    const [source, sum, sumDepth = -1, squaredSumDepth = -1] = arguments_;
    this.#backend.matIntegralInto(
      matHandleForBinding(source),
      matHandleForBinding(sum),
      toWasmI32(sumDepth),
      toWasmI32(squaredSumDepth),
    );
  }

  integral2(
    ...arguments_: [
      source: Mat,
      sum: Mat,
      squaredSum: Mat,
      sumDepth?: number,
      squaredSumDepth?: number,
    ]
  ): void {
    requireArityRange(arguments_.length, 3, 5, "integral2");
    const [source, sum, squaredSum, sumDepth = -1, squaredSumDepth = -1] = arguments_;
    this.#backend.matIntegral2Into(
      matHandleForBinding(source),
      matHandleForBinding(sum),
      matHandleForBinding(squaredSum),
      toWasmI32(sumDepth),
      toWasmI32(squaredSumDepth),
    );
  }

  moments(...arguments_: [source: Mat, binaryImage?: boolean]): Moments {
    requireArityRange(arguments_.length, 1, 2, "moments");
    const [source, binaryImage = false] = arguments_;
    const values = this.#backend.matMoments(matHandleForBinding(source), binaryImage);
    const moment = (index: number) => requiredFloat(values, index);
    return {
      m00: moment(0),
      m10: moment(1),
      m01: moment(2),
      m20: moment(3),
      m11: moment(4),
      m02: moment(5),
      m30: moment(6),
      m21: moment(7),
      m12: moment(8),
      m03: moment(9),
      mu20: moment(10),
      mu11: moment(11),
      mu02: moment(12),
      mu30: moment(13),
      mu21: moment(14),
      mu12: moment(15),
      mu03: moment(16),
      nu20: moment(17),
      nu11: moment(18),
      nu02: moment(19),
      nu30: moment(20),
      nu21: moment(21),
      nu12: moment(22),
      nu03: moment(23),
    };
  }

  HuMoments(source: Moments, destination: Mat): void {
    requireExactArity(arguments.length, 2, "HuMoments");
    this.#backend.matHuMomentsInto(momentsForBinding(source), matHandleForBinding(destination));
  }

  // oxlint-disable-next-line anti-slop/no-shape-in-symbol-names -- Upstream binding name.
  matchShapes(first: Mat, second: Mat, method: number, parameter: number): number {
    requireExactArity(arguments.length, 4, "matchShapes");
    // oxlint-disable-next-line anti-slop/no-shape-in-symbol-names -- Upstream binding name.
    return this.#backend.matMatchShapes(
      matHandleForBinding(first),
      matHandleForBinding(second),
      toWasmI32(method),
      toWasmF64(parameter),
    );
  }

  getFontScaleFromHeight(
    ...arguments_: [fontFace: number, pixelHeight: number, thickness: number]
  ): number {
    requireArityRange(arguments_.length, 3, 3, "getFontScaleFromHeight");
    const [fontFace, pixelHeight, thickness] = arguments_;
    return this.#backend.fontScaleFromHeight(
      toWasmI32(fontFace),
      toWasmI32(pixelHeight),
      toWasmI32(thickness),
    );
  }

  norm(
    ...arguments_:
      [source: Mat, normType?: NormType] | [source: Mat, normType?: NormType, mask?: Mat]
  ): number {
    requireArityRange(arguments_.length, 1, 3, "norm");
    const [source, normType = 4, mask] = arguments_;
    return mask === undefined
      ? this.#backend.matNorm(matHandleForBinding(source), normType)
      : this.#backend.matNormMasked(
          matHandleForBinding(source),
          normType,
          matHandleForBinding(mask),
        );
  }

  normDiff(first: Mat, second: Mat, normType: NormType = 4, mask?: Mat): number {
    return mask === undefined
      ? this.#backend.matNormDiff(first.handleForBackend(), second.handleForBackend(), normType)
      : this.#backend.matNormDiffMasked(
          first.handleForBackend(),
          second.handleForBackend(),
          normType,
          mask.handleForBackend(),
        );
  }

  normalize(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      alpha?: number,
      beta?: number,
      normType?: NormalizeType,
      dtype?: number,
      mask?: Mat,
    ]
  ): void {
    requireArityRange(arguments_.length, 2, 7, "normalize");
    const [source, destination, alpha = 1, beta = 0, normType = 4, dtype = -1, mask] = arguments_;
    if (mask === undefined) {
      this.#backend.matNormalizeInto(
        matHandleForBinding(source),
        matHandleForBinding(destination),
        alpha,
        beta,
        normType,
        dtype,
      );
      return;
    }
    this.#backend.matNormalizeMaskedInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      alpha,
      beta,
      normType,
      dtype,
      matHandleForBinding(mask),
    );
  }

  polarToCart(
    ...arguments_: [magnitude: Mat, angle: Mat, x: Mat, y: Mat, degrees?: boolean]
  ): void {
    requireOverloadArity(arguments_.length, 4, 5, "polarToCart");
    const [magnitude, angle, x, y, degrees = false] = arguments_;
    this.#backend.matPolarToCart(
      matHandleForBinding(magnitude),
      matHandleForBinding(angle),
      matHandleForBinding(x),
      matHandleForBinding(y),
      coerceBoolean(degrees),
    );
  }

  pointPolygonTest(contour: Mat, point: Point, measureDistance: boolean): number {
    requireExactArity(arguments.length, 3, "pointPolygonTest");
    const contourHandle = matHandleForBinding(contour);
    const bindingPoint = point2fForBinding(point);
    return this.#backend.matPointPolygonTest(
      contourHandle,
      bindingPoint.x,
      bindingPoint.y,
      coerceBoolean(measureDistance),
    );
  }

  perspectiveTransform(source: Mat, destination: Mat, coefficients: Mat): void {
    this.#backend.matPerspectiveTransformInto(
      source.handleForBackend(),
      coefficients.handleForBackend(),
      destination.handleForBackend(),
    );
  }

  pow(source: Mat, exponent: number, destination: Mat): void {
    requireExactArity(arguments.length, 3, "pow");
    const sourceHandle = matHandleForBinding(source);
    const power = toWasmF64(exponent);
    const destinationHandle = matHandleForBinding(destination);
    this.#backend.matPowInto(sourceHandle, power, destinationHandle);
  }

  powAlloc(source: Mat, exponent: number): Mat {
    return new Mat(this.#backend.matPow(source.handleForBackend(), toWasmF64(exponent)));
  }

  randn(destination: Mat, mean: Scalar, standardDeviation: Scalar): void {
    this.#backend.matRandn(
      destination.handleForBackend(),
      validatedScalar(mean, "mean"),
      validatedScalar(standardDeviation, "standardDeviation"),
    );
  }

  randu(destination: Mat, lower: Scalar, upper: Scalar): void {
    this.#backend.matRandu(
      destination.handleForBackend(),
      validatedScalar(lower, "lower"),
      validatedScalar(upper, "upper"),
    );
  }

  minMaxLoc(...arguments_: [source: Mat] | [source: Mat, mask: Mat]): MinMaxLocation {
    requireOverloadArity(arguments_.length, 1, 2, "minMaxLoc");
    const source = matHandleForBinding(arguments_[0]);
    const values =
      arguments_.length === 1
        ? this.#backend.matMinMaxLoc(source)
        : this.#backend.matMinMaxLocMasked(source, matHandleForBinding(arguments_[1]));
    return minMaxLocationFromArray(values);
  }

  resizeNearest(image: RgbaImage, targetWidth: number, targetHeight: number): RgbaImage {
    validateRgbaImage(image);
    validateDimension(targetWidth, "targetWidth");
    validateDimension(targetHeight, "targetHeight");
    const data = this.#backend.resizeNearestRgba(
      image.data,
      image.width,
      image.height,
      targetWidth,
      targetHeight,
    );
    return createRgbaImage(targetWidth, targetHeight, data);
  }

  resize(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      size: Size,
      scaleX?: number,
      scaleY?: number,
      interpolation?: Interpolation,
    ]
  ): void {
    requireArityRange(arguments_.length, 3, 6, "resize");
    const [source, destination, size] = arguments_;
    const sourceHandle = matHandleForBinding(source);
    const destinationHandle = matHandleForBinding(destination);
    const convertedSize = size2iForBinding(size);
    const scaleX = arguments_.length >= 4 ? toWasmF64(arguments_[3]) : 0;
    const scaleY = arguments_.length >= 5 ? toWasmF64(arguments_[4]) : 0;
    const interpolation = arguments_.length === 6 ? toWasmI32(arguments_[5]) : INTER_LINEAR;
    this.#backend.matResizeInto(
      sourceHandle,
      destinationHandle,
      convertedSize.width,
      convertedSize.height,
      scaleX,
      scaleY,
      interpolation,
    );
  }

  repeat(source: Mat, rowRepeats: number, columnRepeats: number, destination: Mat): void {
    requireExactArity(arguments.length, 4, "repeat");
    const sourceHandle = matHandleForBinding(source);
    const rows = toWasmI32(rowRepeats);
    const columns = toWasmI32(columnRepeats);
    const destinationHandle = matHandleForBinding(destination);
    this.#backend.matRepeatInto(sourceHandle, destinationHandle, rows, columns);
  }

  repeatAlloc(source: Mat, rowRepeats: number, columnRepeats: number): Mat {
    return new Mat(
      this.#backend.matRepeat(
        source.handleForBackend(),
        toWasmI32(rowRepeats),
        toWasmI32(columnRepeats),
      ),
    );
  }

  reduce(
    ...arguments_: [source: Mat, destination: Mat, axis: 0 | 1, kind: ReduceKind, dtype?: number]
  ): void {
    requireOverloadArity(arguments_.length, 4, 5, "reduce");
    const [source, destination, axis, kind, dtype = -1] = arguments_;
    this.#backend.matReduceInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      axis,
      kind,
      dtype,
    );
  }

  setIdentity(...args: [destination: Mat] | [destination: Mat, value: Scalar]): void {
    requireOverloadArity(args.length, 1, 2, "setIdentity");
    const destinationHandle = matHandleForBinding(args[0]);
    const value = args.length === 1 ? [1, 0, 0, 0] : args[1];
    this.#backend.matSetIdentity(destinationHandle, scalarForBinding(value));
  }

  setLogLevel(level: LogLevel): LogLevel {
    return logLevelFromNumber(this.#backend.setLogLevel(level));
  }

  setRNGSeed(seed: number): void {
    if (!Number.isSafeInteger(seed) || seed < -2_147_483_648 || seed > 2_147_483_647) {
      throw new OpenCvInputError("seed must be a signed 32-bit integer");
    }
    this.#backend.setRNGSeed(seed);
  }

  solve(
    ...arguments_: [
      coefficients: Mat,
      rightHandSides: Mat,
      destination: Mat,
      method?: DecompositionMethod,
    ]
  ): boolean {
    requireOverloadArity(arguments_.length, 3, 4, "solve");
    const [coefficients, rightHandSides, destination, method = 0] = arguments_;
    return this.#backend.matSolveInto(
      matHandleForBinding(coefficients),
      matHandleForBinding(rightHandSides),
      matHandleForBinding(destination),
      method,
    );
  }

  Sobel(
    source: Mat,
    destination: Mat,
    destinationDepth: number,
    dx: number,
    dy: number,
    kernelSize = 3,
    scale = 1,
    delta = 0,
    borderType: BorderType = BORDER_DEFAULT,
  ): void {
    requireArityRange(arguments.length, 5, 9, "Sobel");
    this.#backend.matSobelInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmI32(destinationDepth),
      toWasmI32(dx),
      toWasmI32(dy),
      toWasmI32(kernelSize),
      toWasmF64(scale),
      toWasmF64(delta),
      toWasmI32(borderType),
    );
  }

  threshold(image: RgbaImage, threshold: number): RgbaImage;
  threshold(
    source: Mat,
    destination: Mat,
    threshold: number,
    maximum: number,
    type: number,
  ): number;
  threshold(
    source: Mat | RgbaImage,
    destination: Mat | number,
    threshold?: number,
    maximum?: number,
    type?: number,
  ): RgbaImage | number {
    requireOverloadArity(arguments.length, 2, 5, "threshold");
    if (arguments.length === 2 && !(source instanceof Mat)) {
      validateRgbaImage(source);
      // oxlint-disable-next-line anti-slop/no-runtime-typeof -- This is the JS-to-Embind scalar parser boundary.
      if (typeof destination !== "number") {
        throw new OpenCvInputError("threshold must be an integer from 0 through 255");
      }
      validateThreshold(destination);
      const data = this.#backend.thresholdRgba(
        source.data,
        source.width,
        source.height,
        destination,
      );
      return createRgbaImage(source.width, source.height, data);
    }
    return this.#backend.matThresholdInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmF64(threshold),
      toWasmF64(maximum),
      toWasmI32(type),
    );
  }

  subtract(
    ...arguments_:
      | [source1: Mat, source2: Mat, destination: Mat]
      | [source1: Mat, source2: Mat, destination: Mat, mask: Mat]
      | [source1: Mat, source2: Mat, destination: Mat, mask: Mat, dtype: number]
  ): void {
    requireArityRange(arguments_.length, 3, 5, "subtract");
    const source1 = matHandleForBinding(arguments_[0]);
    const source2 = matHandleForBinding(arguments_[1]);
    const destination = matHandleForBinding(arguments_[2]);
    if (arguments_.length === 3) {
      this.#backend.matSubtractInto(source1, source2, destination);
      return;
    }
    const mask = matHandleForBinding(arguments_[3]);
    if (arguments_.length === 4) {
      this.#backend.matSubtractMaskedInto(source1, source2, destination, mask);
      return;
    }
    this.#backend.matSubtractMaskedDtypeInto(
      source1,
      source2,
      destination,
      mask,
      toWasmI32(arguments_[4]),
    );
  }

  split(source: Mat, destination: MatVector): void {
    requireExactArity(arguments.length, 2, "split");
    this.#backend.matSplitVecInto(
      matHandleForBinding(source),
      matVectorHandleForBinding(destination),
    );
  }

  splitPlanes(source: Mat): Mat[] {
    return this.#backend.matSplit(source.handleForBackend()).map((handle) => new Mat(handle));
  }

  sqrt(source: Mat, destination: Mat): void {
    requireExactArity(arguments.length, 2, "sqrt");
    this.#backend.matSqrtInto(matHandleForBinding(source), matHandleForBinding(destination));
  }

  sqrtAlloc(source: Mat): Mat {
    return new Mat(this.#backend.matSqrt(source.handleForBackend()));
  }

  sum(source: Mat): Scalar {
    return scalarFromArray(this.#backend.matSum(source.handleForBackend()));
  }

  transpose(source: Mat, destination: Mat): void {
    requireExactArity(arguments.length, 2, "transpose");
    this.#backend.matTransposeInto(source.handleForBackend(), destination.handleForBackend());
  }

  transposeAlloc(source: Mat): Mat {
    return new Mat(this.#backend.matTranspose(source.handleForBackend()));
  }

  trace(source: Mat): Scalar {
    requireExactArity(arguments.length, 1, "trace");
    return scalarFromArray(this.#backend.matTrace(matHandleForBinding(source)));
  }

  transform(source: Mat, destination: Mat, coefficients: Mat): void {
    this.#backend.matTransformInto(
      source.handleForBackend(),
      coefficients.handleForBackend(),
      destination.handleForBackend(),
    );
  }

  vconcat(sources: MatVector, destination: Mat): void {
    requireExactArity(arguments.length, 2, "vconcat");
    this.#backend.matVconcatVecInto(
      matVectorHandleForBinding(sources),
      matHandleForBinding(destination),
    );
  }

  vconcatPlanes(
    sources: readonly [Mat, Mat] | readonly [Mat, Mat, Mat] | readonly [Mat, Mat, Mat, Mat],
  ): Mat {
    return this.#concat(sources, "vertical");
  }

  warpPerspective(
    ...arguments_: [
      source: Mat,
      destination: Mat,
      transform: Mat,
      size: Size,
      flags?: number,
      borderType?: BorderType,
      borderValue?: Scalar,
    ]
  ): void {
    requireArityRange(arguments_.length, 4, 7, "warpPerspective");
    const [source, destination, transform, size] = arguments_;
    const sourceHandle = matHandleForBinding(source);
    const destinationHandle = matHandleForBinding(destination);
    const transformHandle = matHandleForBinding(transform);
    const convertedSize = size2iForBinding(size);
    this.#backend.matWarpPerspectiveInto(
      sourceHandle,
      destinationHandle,
      transformHandle,
      convertedSize.width,
      convertedSize.height,
      arguments_.length >= 5 ? toWasmI32(arguments_[4]) : INTER_LINEAR,
      arguments_.length >= 6 ? toWasmI32(arguments_[5]) : BORDER_CONSTANT,
      arguments_.length >= 7 ? scalarForBinding(arguments_[6]) : new Float64Array(4),
    );
  }

  warpAffine(
    source: Mat,
    destination: Mat,
    transform: Mat,
    size: Size,
    flags: number = INTER_LINEAR,
    borderType: BorderType = BORDER_CONSTANT,
    borderValue: Scalar = [0, 0, 0, 0],
  ): void {
    requireArityRange(arguments.length, 4, 7, "warpAffine");
    const convertedSize = size2iForBinding(size);
    this.#backend.matWarpAffineInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      matHandleForBinding(transform),
      convertedSize.width,
      convertedSize.height,
      toWasmI32(flags),
      toWasmI32(borderType),
      scalarForBinding(borderValue),
    );
  }

  #concat(
    sources: readonly [Mat, Mat] | readonly [Mat, Mat, Mat] | readonly [Mat, Mat, Mat, Mat],
    direction: "horizontal" | "vertical",
  ): Mat {
    const handles = sources.map((source) => source.handleForBackend());
    const [first, second, third, fourth] = handles;
    if (first === undefined || second === undefined) {
      throw new OpenCvInputError("concat requires two through four matrices");
    }
    const prefix = direction === "horizontal" ? "matHconcat" : "matVconcat";
    if (third === undefined) {
      return new Mat(this.#backend[`${prefix}2`](first, second));
    }
    if (fourth === undefined) {
      return new Mat(this.#backend[`${prefix}3`](first, second, third));
    }
    return new Mat(this.#backend[`${prefix}4`](first, second, third, fourth));
  }

  zerosF32(rows: number, columns: number, channels: number): Mat {
    validateZeroAllocation(rows, columns, channels, Float32Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matZerosF32(rows, columns, channels));
  }

  zerosF64(rows: number, columns: number, channels: number): Mat {
    validateZeroAllocation(rows, columns, channels, Float64Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matZerosF64(rows, columns, channels));
  }

  zerosI16(rows: number, columns: number, channels: number): Mat {
    validateZeroAllocation(rows, columns, channels, Int16Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matZerosI16(rows, columns, channels));
  }

  zerosI32(rows: number, columns: number, channels: number): Mat {
    validateZeroAllocation(rows, columns, channels, Int32Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matZerosI32(rows, columns, channels));
  }

  zerosI8(rows: number, columns: number, channels: number): Mat {
    validateZeroAllocation(rows, columns, channels, Int8Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matZerosI8(rows, columns, channels));
  }

  zerosU16(rows: number, columns: number, channels: number): Mat {
    validateZeroAllocation(rows, columns, channels, Uint16Array.BYTES_PER_ELEMENT);
    return new Mat(this.#backend.matZerosU16(rows, columns, channels));
  }

  zerosU8(rows: number, columns: number, channels: number): Mat {
    validateMatrixDimension(rows, "rows");
    validateMatrixDimension(columns, "columns");
    validateMatrixInput(rows, columns, channels, rows * columns * channels);
    return new Mat(this.#backend.matZerosU8(rows, columns, channels));
  }
}

function validateZeroAllocation(
  rows: number,
  columns: number,
  channels: number,
  byteWidth: number,
): void {
  validateMatrixInput(rows, columns, channels, rows * columns * channels * byteWidth, byteWidth);
}

function scalarFromArray(values: Float64Array): Scalar {
  if (values.length !== 4) {
    throw new OpenCvInputError(`WASM scalar has ${values.length} lanes; expected 4`);
  }
  const lane0 = values[0];
  const lane1 = values[1];
  const lane2 = values[2];
  const lane3 = values[3];
  if (lane0 === undefined || lane1 === undefined || lane2 === undefined || lane3 === undefined) {
    throw new OpenCvInputError("WASM scalar is missing a lane");
  }
  return [lane0, lane1, lane2, lane3];
}

function minMaxLocationFromArray(values: Float64Array): MinMaxLocation {
  if (values.length !== 6) {
    throw new OpenCvInputError(`WASM minMaxLoc result has ${values.length} lanes; expected 6`);
  }
  const minVal = requiredFloat(values, 0);
  const maxVal = requiredFloat(values, 1);
  const minX = requiredFloat(values, 2);
  const minY = requiredFloat(values, 3);
  const maxX = requiredFloat(values, 4);
  const maxY = requiredFloat(values, 5);
  return {
    maxLoc: { x: maxX, y: maxY },
    maxVal,
    minLoc: { x: minX, y: minY },
    minVal,
  };
}

function rectangleFromArray(values: Int32Array): Rect {
  if (values.length !== 4) {
    throw new OpenCvInputError(`WASM rectangle has ${values.length} lanes; expected 4`);
  }
  return {
    x: requiredInteger(values, 0),
    y: requiredInteger(values, 1),
    width: requiredInteger(values, 2),
    height: requiredInteger(values, 3),
  };
}

function lineFromArray(values: Int32Array): readonly [Point, Point] | undefined {
  if (values.length === 0) return undefined;
  if (values.length !== 4) {
    throw new OpenCvInputError(`WASM clipped line has ${values.length} lanes; expected 0 or 4`);
  }
  return [
    { x: requiredInteger(values, 0), y: requiredInteger(values, 1) },
    { x: requiredInteger(values, 2), y: requiredInteger(values, 3) },
  ];
}

function pointsFromArray(values: Int32Array): Point[] {
  if (values.length % 2 !== 0) {
    throw new OpenCvInputError(`WASM point array has ${values.length} lanes; expected pairs`);
  }
  const points: Point[] = [];
  for (let index = 0; index < values.length; index += 2) {
    points.push({ x: requiredInteger(values, index), y: requiredInteger(values, index + 1) });
  }
  return points;
}

function requiredFloat(values: Float64Array, index: number): number {
  const value = values[index];
  if (value === undefined) {
    throw new OpenCvInputError(`WASM result is missing lane ${index}`);
  }
  return value;
}

function requiredInteger(values: Int32Array, index: number): number {
  const value = values[index];
  if (value === undefined) {
    throw new OpenCvInputError(`WASM integer result is missing lane ${index}`);
  }
  return value;
}

function logLevelFromNumber(value: number): LogLevel {
  switch (value) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
      return value;
    default:
      throw new OpenCvInputError(`WASM log level ${value} is outside 0 through 6`);
  }
}

function validateChannelIndex(channel: number): void {
  if (!Number.isSafeInteger(channel) || channel < 0 || channel > 511) {
    throw new OpenCvInputError("channel must be a non-negative integer below 512");
  }
}

function validateFiniteNumbers(values: Readonly<Record<string, number>>): void {
  for (const [name, value] of Object.entries(values)) {
    if (!Number.isFinite(value)) throw new OpenCvInputError(`${name} must be finite`);
  }
}

function validateIntegerPoint(point: Point, name: string): void {
  validateSignedInteger(point.x, `${name}.x`);
  validateSignedInteger(point.y, `${name}.y`);
}

function validateSignedInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < -2_147_483_648 || value > 2_147_483_647) {
    throw new OpenCvInputError(`${name} must be a signed 32-bit integer`);
  }
}

interface EmbindObjectInput {
  toString(): string;
}

interface EmbindPointObject extends EmbindObjectInput {
  readonly x?: EmbindPointInput;
  readonly y?: EmbindPointInput;
}

type EmbindScalarInput = boolean | number | EmbindObjectInput | string | null | undefined;
type EmbindPointInput = EmbindScalarInput | bigint | symbol;

function toWasmI32(value: EmbindScalarInput): number {
  if (value === true) return 1;
  if (value === false) return 0;
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- This is the JS-to-Embind scalar parser boundary.
  if (typeof value !== "number") {
    throw new TypeError(`Cannot convert "${String(value)}" to int`);
  }
  if (value < -2_147_483_648 || value > 2_147_483_647) {
    throw new TypeError(
      `Passing a number "${String(value)}" from JS side to C/C++ side to an argument of type "int", which is outside the valid range [-2147483648, 2147483647]!`,
    );
  }
  return value | 0;
}

// oxlint-disable-next-line anti-slop/no-unknown-parameters -- This parser is the JS-to-Embind double boundary.
function toWasmF64(value: unknown): number {
  if (value === true) return 1;
  if (value === false) return 0;
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- This is the JS-to-Embind scalar parser boundary.
  if (typeof value !== "number") {
    throw new TypeError(`Cannot convert "${String(value)}" to double`);
  }
  return value;
}

// oxlint-disable-next-line anti-slop/no-unknown-parameters -- This validator is the JS-to-Embind Mat boundary.
function matVectorHandleForBinding(value: unknown): ReturnType<MatVector["handleForBackend"]> {
  if (value === null) {
    throw new BindingError("null is not a valid MatVector");
  }
  if (value === undefined) {
    throw new TypeError("Cannot read properties of undefined (reading '$$')");
  }
  if (!(value instanceof MatVector)) {
    throw new BindingError(
      // oxlint-disable-next-line typescript/no-base-to-string -- Embind error text formats the rejected value verbatim.
      `Expected null or instance of MatVector, got an instance of ${String(value)}`,
    );
  }
  return value.handleForBackend();
}

function rectForBinding(rectangle: Rect): Rect {
  if (
    rectangle === null ||
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- This is the JS-to-Embind Rect boundary.
    typeof rectangle !== "object"
  ) {
    throw new BindingError(`Cannot convert "${String(rectangle)}" to Rect`);
  }
  for (const field of ["x", "y", "width", "height"] as const) {
    if (!(field in rectangle)) {
      throw new BindingError(`Rect is missing field ${field}`);
    }
  }
  return {
    x: toWasmI32(rectangle.x),
    y: toWasmI32(rectangle.y),
    width: toWasmI32(rectangle.width),
    height: toWasmI32(rectangle.height),
  };
}

function pointForBinding(point: Point, name: string): Point {
  if (
    point === null ||
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- This is the JS-to-Embind Point boundary.
    typeof point !== "object"
  ) {
    throw new BindingError(`Cannot convert "${String(point)}" to Point`);
  }
  if (!("x" in point) || !("y" in point)) {
    throw new BindingError(`${name} is missing field x or y`);
  }
  return { x: toWasmI32(point.x), y: toWasmI32(point.y) };
}

// oxlint-disable-next-line anti-slop/no-unknown-parameters -- This validator is the JS-to-Embind PointVector boundary.
function pointVectorHandleForBinding(value: unknown): PointVector {
  if (value === null) {
    throw new BindingError("null is not a valid PointVector");
  }
  if (value === undefined) {
    throw new TypeError("Cannot read properties of undefined (reading '$$')");
  }
  if (!(value instanceof PointVector)) {
    throw new BindingError(
      // oxlint-disable-next-line typescript/no-base-to-string -- Embind error text formats the rejected value verbatim.
      `Expected null or instance of PointVector, got an instance of ${String(value)}`,
    );
  }
  return value;
}

// oxlint-disable-next-line anti-slop/no-unknown-parameters -- This validator is the JS-to-Embind Mat boundary.
function matHandleForBinding(value: unknown): ReturnType<Mat["handleForBackend"]> {
  if (value === null) {
    throw new BindingError("null is not a valid Mat");
  }
  if (value === undefined) {
    throw new TypeError("Cannot read properties of undefined (reading '$$')");
  }
  if (!(value instanceof Mat)) {
    // oxlint-disable-next-line typescript/no-base-to-string -- Embind error text formats the rejected value verbatim.
    throw new BindingError(`Cannot pass "${String(value)}" as a Mat`);
  }
  return value.handleForBackend();
}

function point2fForBinding(value: EmbindPointInput): Point {
  if (!isEmbindPointObject(value)) {
    throw new BindingError(`Cannot convert "${String(value)}" to Point2f`);
  }
  if (!("x" in value)) {
    throw new BindingError("Point2f is missing field x");
  }
  if (!("y" in value)) {
    throw new BindingError("Point2f is missing field y");
  }
  const x = toWasmF32(value.x);
  const y = toWasmF32(value.y);
  return { x, y };
}

function point2iForBinding(value: Point): Point {
  const isObject =
    value !== null &&
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- This is the JS-to-Embind Point boundary.
    (typeof value === "object" || typeof value === "function");
  if (!isObject) {
    throw new BindingError(`Cannot convert "${String(value)}" to Point`);
  }
  if (!("x" in value)) throw new BindingError("Point is missing field x");
  if (!("y" in value)) throw new BindingError("Point is missing field y");
  // SAFETY: The field checks above prove both binding fields exist; each field is parsed next.
  // oxlint-disable-next-line typescript/no-unsafe-type-assertion -- This assertion models the untyped Embind value-object boundary.
  const fields = value as { readonly x: EmbindScalarInput; readonly y: EmbindScalarInput };
  return { x: toWasmI32(fields.x), y: toWasmI32(fields.y) };
}

function size2iForBinding(value: Size): Size {
  const isObject =
    value !== null &&
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- This is the JS-to-Embind Size boundary.
    (typeof value === "object" || typeof value === "function");
  if (!isObject) {
    throw new BindingError(`Cannot convert "${String(value)}" to Size`);
  }
  if (!("width" in value)) throw new BindingError("Size is missing field width");
  if (!("height" in value)) throw new BindingError("Size is missing field height");
  // SAFETY: The field checks above prove both binding fields exist; each field is parsed next.
  // oxlint-disable-next-line typescript/no-unsafe-type-assertion -- This assertion models the untyped Embind value-object boundary.
  const fields = value as {
    readonly width: EmbindScalarInput;
    readonly height: EmbindScalarInput;
  };
  return { width: toWasmI32(fields.width), height: toWasmI32(fields.height) };
}

function isEmbindPointObject(value: EmbindPointInput): value is EmbindPointObject {
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- This is the JS-to-Embind Point2f parser boundary.
  return value !== null && (typeof value === "object" || typeof value === "function");
}

function toWasmF32(value: EmbindPointInput): number {
  if (value === true) return 1;
  if (value === false) return 0;
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- This is the JS-to-Embind scalar parser boundary.
  if (typeof value !== "number") {
    throw new TypeError(`Cannot convert "${String(value)}" to float`);
  }
  return Math.fround(value);
}

function coerceBoolean(
  value: boolean | number | bigint | string | symbol | null | undefined,
): boolean {
  return Boolean(value);
}

function requireExactArity(actual: number, expected: number, method: string): void {
  if (actual !== expected) {
    throw new BindingError(
      `function ${method} called with ${actual} arguments, expected ${expected} args!`,
    );
  }
}

function requireOverloadArity(
  actual: number,
  firstExpected: number,
  secondExpected: number,
  method: string,
): void {
  if (actual !== firstExpected && actual !== secondExpected) {
    throw new BindingError(
      `Function '${method}' called with an invalid number of arguments (${actual}); expected ${firstExpected} or ${secondExpected}`,
    );
  }
}

function requireArityRange(actual: number, minimum: number, maximum: number, method: string): void {
  if (actual < minimum || actual > maximum) {
    throw new BindingError(
      `function ${method} called with ${actual} arguments, expected ${minimum} to ${maximum} args!`,
    );
  }
}

function validateMinimumSize(size: Size, minimum: number, name: string): void {
  validateDimension(size.width, `${name}.width`);
  validateDimension(size.height, `${name}.height`);
  if (size.width < minimum || size.height < minimum) {
    throw new OpenCvInputError(`${name} dimensions must be at least ${minimum}`);
  }
}

function validateRect(rectangle: Rect): void {
  validateSignedInteger(rectangle.x, "rectangle.x");
  validateSignedInteger(rectangle.y, "rectangle.y");
  validateDimension(rectangle.width, "rectangle.width");
  validateDimension(rectangle.height, "rectangle.height");
}

function depthCode(depth: HanningWindowDepth): number {
  return depth === "f32" ? 5 : 6;
}

function validatedScalar(value: Scalar, name: string): Float64Array {
  validateFiniteNumbers({
    [`${name}[0]`]: value[0],
    [`${name}[1]`]: value[1],
    [`${name}[2]`]: value[2],
    [`${name}[3]`]: value[3],
  });
  return Float64Array.from(value);
}

// oxlint-disable-next-line anti-slop/no-unknown-parameters -- This parser is the JS-to-Embind Scalar boundary.
function scalarForBinding(value: unknown): Float64Array {
  const isObject =
    value !== null &&
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- This is the JS-to-Embind Scalar boundary.
    (typeof value === "object" || typeof value === "function");
  // SAFETY: The object/function check above proves property access is valid at this binding boundary.
  if (!isObject || (value as { length?: unknown }).length !== 4) {
    throw new BindingError(`Cannot convert "${String(value)}" to Scalar`);
  }
  // SAFETY: Scalar conversion requires an array-like value with four indexed lanes.
  // oxlint-disable-next-line anti-slop/no-unsafe-dictionary-type, typescript/no-unsafe-type-assertion -- Each lane is parsed immediately as an Embind double.
  const indexed = value as Record<number, unknown>;
  return new Float64Array([
    toWasmF64(indexed[0]),
    toWasmF64(indexed[1]),
    toWasmF64(indexed[2]),
    toWasmF64(indexed[3]),
  ]);
}

const MOMENT_FIELDS = [
  "m00",
  "m10",
  "m01",
  "m20",
  "m11",
  "m02",
  "m30",
  "m21",
  "m12",
  "m03",
  "mu20",
  "mu11",
  "mu02",
  "mu30",
  "mu21",
  "mu12",
  "mu03",
  "nu20",
  "nu11",
  "nu02",
  "nu30",
  "nu21",
  "nu12",
  "nu03",
] as const;

function momentsForBinding(value: Moments): Float64Array {
  // SAFETY: Each field is parsed immediately as an Embind double, so a missing or
  // nonnumeric field becomes NaN and is rejected by toWasmF64.
  const values = MOMENT_FIELDS.map((field) => toWasmF64(value[field]));
  return new Float64Array(values);
}

function validateBorderSize(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > 4_294_967_295) {
    throw new OpenCvInputError(`${name} border must be a non-negative 32-bit integer`);
  }
}

/** Creates a client from a compatible backend. Most callers should use `initOpenCv`. */
export function createOpenCv(backend: OpenCvBackend): OpenCv {
  return new WasmOpenCv(backend);
}
