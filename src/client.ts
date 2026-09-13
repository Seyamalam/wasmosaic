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
  BorderType,
  DecompositionMethod,
  HanningWindowDepth,
  HanningWindowType,
  LogLevel,
  MinMaxLocation,
  NormalizeType,
  NormType,
  OpenCv,
  OpenCvBackend,
  Point,
  Rect,
  RgbaImage,
  ReduceKind,
  Scalar,
  Size,
  StructuringElementKind,
} from "./types.js";

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

  absdiff(left: Mat, right: Mat): Mat {
    return new Mat(this.#backend.matAbsdiffU8(left.handleForBackend(), right.handleForBackend()));
  }

  add(left: Mat, right: Mat): Mat {
    return new Mat(this.#backend.matAddU8(left.handleForBackend(), right.handleForBackend()));
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

  bitwiseAnd(left: Mat, right: Mat): Mat {
    return new Mat(
      this.#backend.matBitwiseAndU8(left.handleForBackend(), right.handleForBackend()),
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

  bitwiseOr(left: Mat, right: Mat): Mat {
    return new Mat(this.#backend.matBitwiseOrU8(left.handleForBackend(), right.handleForBackend()));
  }

  bitwiseXor(left: Mat, right: Mat): Mat {
    return new Mat(
      this.#backend.matBitwiseXorU8(left.handleForBackend(), right.handleForBackend()),
    );
  }

  boundingRect(contour: Mat): Rect {
    requireExactArity(arguments.length, 1, "boundingRect");
    return rectangleFromArray(this.#backend.matBoundingRect(matHandleForBinding(contour)));
  }

  clipLine(rectangle: Rect, start: Point, end: Point): readonly [Point, Point] | undefined {
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
  ): Point[] {
    validateIntegerPoint(center, "center");
    validateNonNegativeInteger(axes.width, "axes.width");
    validateNonNegativeInteger(axes.height, "axes.height");
    validateSignedInteger(rotationDegrees, "rotationDegrees");
    validateSignedInteger(arcStart, "arcStart");
    validateSignedInteger(arcEnd, "arcEnd");
    validateSignedInteger(delta, "delta");
    return pointsFromArray(
      this.#backend.ellipse2Poly(
        center.x,
        center.y,
        axes.width,
        axes.height,
        rotationDegrees,
        arcStart,
        arcEnd,
        delta,
      ),
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

  hconcat(
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
    imageOrSource: RgbaImage | Mat,
    destination?: Mat,
    method: DecompositionMethod = 0,
  ): RgbaImage | number {
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

  inRange(source: Mat, lowerBound: Mat, upperBound: Mat): Mat {
    return new Mat(
      this.#backend.matInRangeU8(
        source.handleForBackend(),
        lowerBound.handleForBackend(),
        upperBound.handleForBackend(),
      ),
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

  max(left: Mat, right: Mat): Mat {
    return new Mat(this.#backend.matMaxU8(left.handleForBackend(), right.handleForBackend()));
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

  meanStdDev(source: Mat, means: Mat, standardDeviations: Mat, mask?: Mat): void {
    if (mask === undefined) {
      this.#backend.matMeanStdDevInto(
        source.handleForBackend(),
        means.handleForBackend(),
        standardDeviations.handleForBackend(),
      );
      return;
    }
    this.#backend.matMeanStdDevMaskedInto(
      source.handleForBackend(),
      means.handleForBackend(),
      standardDeviations.handleForBackend(),
      mask.handleForBackend(),
    );
  }

  merge(
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

  min(left: Mat, right: Mat): Mat {
    return new Mat(this.#backend.matMinU8(left.handleForBackend(), right.handleForBackend()));
  }

  mixChannels(source: Mat, destination: Mat, fromTo: Uint16Array): void {
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

  norm(source: Mat, normType?: NormType, mask?: Mat): number;
  norm(first: Mat, second: Mat, normType?: NormType, mask?: Mat): number;
  norm(
    first: Mat,
    secondOrType: Mat | NormType = 4,
    typeOrMask?: NormType | Mat,
    mask?: Mat,
  ): number {
    if (secondOrType instanceof Mat) {
      const resolvedMask = typeOrMask instanceof Mat ? typeOrMask : mask;
      const normType = typeOrMask instanceof Mat ? 4 : (typeOrMask ?? 4);
      return resolvedMask === undefined
        ? this.#backend.matNormDiff(
            first.handleForBackend(),
            secondOrType.handleForBackend(),
            normType,
          )
        : this.#backend.matNormDiffMasked(
            first.handleForBackend(),
            secondOrType.handleForBackend(),
            normType,
            resolvedMask.handleForBackend(),
          );
    }
    const resolvedMask = typeOrMask instanceof Mat ? typeOrMask : undefined;
    return resolvedMask === undefined
      ? this.#backend.matNorm(first.handleForBackend(), secondOrType)
      : this.#backend.matNormMasked(
          first.handleForBackend(),
          secondOrType,
          resolvedMask.handleForBackend(),
        );
  }

  normalize(
    source: Mat,
    destination: Mat,
    alpha: number,
    beta: number,
    normType: NormalizeType,
    mask?: Mat,
  ): void {
    validateFiniteNumbers({ alpha, beta });
    if (mask === undefined) {
      this.#backend.matNormalizeInto(
        source.handleForBackend(),
        destination.handleForBackend(),
        alpha,
        beta,
        normType,
      );
      return;
    }
    this.#backend.matNormalizeMaskedInto(
      source.handleForBackend(),
      destination.handleForBackend(),
      alpha,
      beta,
      normType,
      mask.handleForBackend(),
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

  perspectiveTransform(source: Mat, coefficients: Mat): Mat;
  perspectiveTransform(source: Mat, coefficients: Mat, destination: Mat): void;
  perspectiveTransform(source: Mat, coefficients: Mat, destination?: Mat): Mat | void {
    if (destination !== undefined) {
      this.#backend.matPerspectiveTransformInto(
        source.handleForBackend(),
        coefficients.handleForBackend(),
        destination.handleForBackend(),
      );
      return;
    }
    return new Mat(
      this.#backend.matPerspectiveTransform(
        source.handleForBackend(),
        coefficients.handleForBackend(),
      ),
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
    const convertedSize = size2iForBinding(size);
    const scaleX = arguments_.length >= 4 ? toWasmF64(arguments_[3]) : 0;
    const scaleY = arguments_.length >= 5 ? toWasmF64(arguments_[4]) : 0;
    const interpolation = arguments_.length === 6 ? toWasmI32(arguments_[5]) : INTER_LINEAR;
    this.#backend.matResizeInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
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

  reduce(source: Mat, destination: Mat, axis: 0 | 1, kind: ReduceKind): void {
    this.#backend.matReduceInto(
      source.handleForBackend(),
      destination.handleForBackend(),
      axis,
      kind,
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
    coefficients: Mat,
    rightHandSides: Mat,
    destination: Mat,
    method: DecompositionMethod = 0,
  ): boolean {
    return this.#backend.matSolveInto(
      coefficients.handleForBackend(),
      rightHandSides.handleForBackend(),
      destination.handleForBackend(),
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
    ...arguments_:
      | [image: RgbaImage, threshold: number]
      | [source: Mat, destination: Mat, threshold: number, maximum: number, type: number]
  ): RgbaImage | number {
    requireOverloadArity(arguments_.length, 2, 5, "threshold");
    if (arguments_.length === 2) {
      const [image, threshold] = arguments_;
      validateRgbaImage(image);
      validateThreshold(threshold);
      const data = this.#backend.thresholdRgba(image.data, image.width, image.height, threshold);
      return createRgbaImage(image.width, image.height, data);
    }
    const [source, destination, threshold, maximum, type] = arguments_;
    return this.#backend.matThresholdInto(
      matHandleForBinding(source),
      matHandleForBinding(destination),
      toWasmF64(threshold),
      toWasmF64(maximum),
      toWasmI32(type),
    );
  }

  subtract(left: Mat, right: Mat): Mat {
    return new Mat(this.#backend.matSubtractU8(left.handleForBackend(), right.handleForBackend()));
  }

  split(source: Mat): Mat[] {
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

  transform(source: Mat, coefficients: Mat): Mat;
  transform(source: Mat, coefficients: Mat, destination: Mat): void;
  transform(source: Mat, coefficients: Mat, destination?: Mat): Mat | void {
    if (destination !== undefined) {
      this.#backend.matTransformInto(
        source.handleForBackend(),
        coefficients.handleForBackend(),
        destination.handleForBackend(),
      );
      return;
    }
    return new Mat(
      this.#backend.matTransform(source.handleForBackend(), coefficients.handleForBackend()),
    );
  }

  vconcat(
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

function matHandleForBinding(value: Mat): ReturnType<Mat["handleForBackend"]> {
  if (value === null) {
    throw new BindingError("null is not a valid Mat");
  }
  if (value === undefined) {
    throw new TypeError("Cannot read properties of undefined (reading '$$')");
  }
  if (!(value instanceof Mat)) {
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

function validateNonNegativeInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > 2_147_483_647) {
    throw new OpenCvInputError(`${name} must be a non-negative signed 32-bit integer`);
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

function validateBorderSize(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > 4_294_967_295) {
    throw new OpenCvInputError(`${name} border must be a non-negative 32-bit integer`);
  }
}

/** Creates a client from a compatible backend. Most callers should use `initOpenCv`. */
export function createOpenCv(backend: OpenCvBackend): OpenCv {
  return new WasmOpenCv(backend);
}
