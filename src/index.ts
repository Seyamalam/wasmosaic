import { createOpenCv } from "./client.js";
import type { OpenCvBackend } from "./types.js";

export { createOpenCv } from "./client.js";
export {
  AKAZE,
  AKAZE_DEFAULTS,
  AKAZE_DescriptorType,
  AKAZEDescriptorType,
  KAZE_DiffusivityType,
  KAZEDiffusivity,
} from "./akaze.js";
export type {
  AKAZE_DescriptorTypeNamespace,
  AKAZE_DescriptorTypeValue,
  AKAZEOptions,
  KAZE_DiffusivityTypeNamespace,
  KAZE_DiffusivityTypeValue,
  WasmAKAZEFactory,
  WasmAKAZEHandle,
} from "./akaze.js";
export { KAZE, KAZE_DEFAULTS } from "./kaze.js";
export type { KAZEOptions, WasmKAZEFactory, WasmKAZEHandle } from "./kaze.js";
export {
  ORB,
  ORB_DEFAULTS,
  ORB_FAST_SCORE,
  ORB_HARRIS_SCORE,
  ORB_ScoreType,
  ORBScoreType,
} from "./orb.js";
export type {
  ORBOptions,
  ORB_ScoreTypeNamespace,
  ORB_ScoreTypeValue,
  WasmORBFactory,
  WasmORBHandle,
} from "./orb.js";
export { GFTT_DETECTOR_DEFAULTS, GFTTDetector } from "./gftt.js";
export type {
  GFTTDetectorOptions,
  WasmGFTTDetectorFactory,
  WasmGFTTDetectorHandle,
} from "./gftt.js";
export { MSER, MSER_DEFAULTS } from "./mser.js";
export type { MSEROptions, WasmMSERFactory, WasmMSERHandle } from "./mser.js";
export { Tonemap, TonemapDrago, TonemapMantiuk, TonemapReinhard } from "./tonemap.js";
export type {
  WasmTonemapDragoFactory,
  WasmTonemapDragoHandle,
  WasmTonemapHandle,
  WasmTonemapMantiukFactory,
  WasmTonemapMantiukHandle,
  WasmTonemapReinhardFactory,
  WasmTonemapReinhardHandle,
} from "./tonemap.js";
export { BindingError, OpenCvInputError } from "./error.js";
export {
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
export type { ColorConversionCode } from "./color.js";
export {
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
export {
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
export type { MorphologyOperation } from "./filtering.js";
export {
  INTER_AREA,
  INTER_CUBIC,
  INTER_LANCZOS4,
  INTER_LINEAR,
  INTER_LINEAR_EXACT,
  INTER_NEAREST,
  INTER_NEAREST_EXACT,
} from "./interpolation.js";
export type { Interpolation } from "./interpolation.js";
export {
  TM_SQDIFF,
  TM_SQDIFF_NORMED,
  TM_CCORR,
  TM_CCORR_NORMED,
  TM_CCOEFF,
  TM_CCOEFF_NORMED,
} from "./template-matching.js";
export type { TemplateMatchMode } from "./template-matching.js";
export {
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
export type { ThresholdMode } from "./threshold.js";
export type { EmbindEnumInput, EmbindEnumValue } from "./embind-enum.js";
export {
  AGAST_FEATURE_DETECTOR_DEFAULTS,
  AgastFeatureDetector,
  AgastFeatureDetector_DetectorType,
  AgastFeatureDetectorType,
  FAST_FEATURE_DETECTOR_DEFAULTS,
  FastFeatureDetector,
  FastFeatureDetector_DetectorType,
  FastFeatureDetectorType,
} from "./feature-detectors.js";
export type {
  AgastFeatureDetector_DetectorTypeNamespace,
  AgastFeatureDetector_DetectorTypeValue,
  AgastFeatureDetectorOptions,
  FastFeatureDetector_DetectorTypeNamespace,
  FastFeatureDetector_DetectorTypeValue,
  FastFeatureDetectorOptions,
  WasmAgastFeatureDetectorFactory,
  WasmAgastFeatureDetectorHandle,
  WasmFastFeatureDetectorFactory,
  WasmFastFeatureDetectorHandle,
} from "./feature-detectors.js";
export { createRgbaImage, imageDataFromRgbaImage, rgbaImageFromImageData } from "./image.js";
export { OPENCV_OPERATIONS } from "./operations.js";
export type { OpenCvOperation } from "./operations.js";
export { Mat } from "./mat.js";
export type { MatDepth, WasmMatHandle } from "./mat.js";
export { MatVector } from "./mat-vector.js";
export type { WasmMatVectorHandle } from "./mat-vector.js";
export type {
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

/** Loads the package WebAssembly module and returns an initialized client. */
export async function initOpenCv(): Promise<ReturnType<typeof createOpenCv>> {
  const backend: OpenCvBackend & { default(): Promise<void> } = await import("#wasm");
  await backend.default();
  return createOpenCv(backend);
}
