import type { WasmMatHandle } from "./mat.js";
import type { WasmMatVectorHandle } from "./mat-vector.js";
import type { WasmAKAZEHandle } from "./akaze.js";
import type { WasmKAZEHandle } from "./kaze.js";
import type { WasmORBHandle } from "./orb.js";
import type { WasmGFTTDetectorHandle } from "./gftt.js";
import type { WasmMSERHandle } from "./mser.js";
import type {
  WasmTonemapDragoHandle,
  WasmTonemapMantiukHandle,
  WasmTonemapReinhardHandle,
} from "./tonemap.js";
import type {
  WasmAgastFeatureDetectorHandle,
  WasmFastFeatureDetectorHandle,
} from "./feature-detectors.js";

declare module "#wasm" {
  export class AgastFeatureDetector implements WasmAgastFeatureDetectorHandle {
    private constructor();
    static create(
      threshold?: number | null,
      nonmaxSuppression?: boolean | null,
      type?: number | null,
    ): AgastFeatureDetector;
    free(): void;
    getDefaultName(): string;
    getNonmaxSuppression(): boolean;
    getThreshold(): number;
    getType(): number;
    setNonmaxSuppression(value: boolean): void;
    setThreshold(value: number): void;
    setType(value: number): void;
  }

  export class FastFeatureDetector implements WasmFastFeatureDetectorHandle {
    private constructor();
    static create(
      threshold?: number | null,
      nonmaxSuppression?: boolean | null,
      type?: number | null,
    ): FastFeatureDetector;
    free(): void;
    getDefaultName(): string;
    getNonmaxSuppression(): boolean;
    getThreshold(): number;
    getType(): number;
    setNonmaxSuppression(value: boolean): void;
    setThreshold(value: number): void;
    setType(value: number): void;
  }

  export class AKAZE implements WasmAKAZEHandle {
    private constructor();
    static create(
      descriptorType?: number | null,
      descriptorSize?: number | null,
      descriptorChannels?: number | null,
      threshold?: number | null,
      octaves?: number | null,
      octaveLayers?: number | null,
      diffusivity?: number | null,
      maxPoints?: number | null,
    ): AKAZE;
    free(): void;
    getDefaultName(): string;
    getDescriptorChannels(): number;
    getDescriptorSize(): number;
    getDescriptorType(): number;
    getDiffusivity(): number;
    getNOctaveLayers(): number;
    getNOctaves(): number;
    getThreshold(): number;
    setDescriptorChannels(value: number): void;
    setDescriptorSize(value: number): void;
    setDescriptorType(value: number): void;
    setDiffusivity(value: number): void;
    setNOctaveLayers(value: number): void;
    setNOctaves(value: number): void;
    setThreshold(value: number): void;
  }

  export class KAZE implements WasmKAZEHandle {
    private constructor();
    static create(
      extended?: boolean | null,
      upright?: boolean | null,
      threshold?: number | null,
      octaves?: number | null,
      octaveLayers?: number | null,
      diffusivity?: number | null,
    ): KAZE;
    free(): void;
    getDefaultName(): string;
    getDiffusivity(): number;
    getExtended(): boolean;
    getNOctaveLayers(): number;
    getNOctaves(): number;
    getThreshold(): number;
    getUpright(): boolean;
    setDiffusivity(value: number): void;
    setExtended(value: boolean): void;
    setNOctaveLayers(value: number): void;
    setNOctaves(value: number): void;
    setThreshold(value: number): void;
    setUpright(value: boolean): void;
  }

  export class ORB implements WasmORBHandle {
    private constructor();
    static create(
      maxFeatures?: number | null,
      scaleFactor?: number | null,
      nLevels?: number | null,
      edgeThreshold?: number | null,
      firstLevel?: number | null,
      wtaK?: number | null,
      scoreType?: number | null,
      patchSize?: number | null,
      fastThreshold?: number | null,
    ): ORB;
    free(): void;
    getDefaultName(): string;
    getFastThreshold(): number;
    setEdgeThreshold(value: number): void;
    setFastThreshold(value: number): void;
    setFirstLevel(value: number): void;
    setMaxFeatures(value: number): void;
    setNLevels(value: number): void;
    setPatchSize(value: number): void;
    setScaleFactor(value: number): void;
    setScoreType(value: number): void;
    setWTA_K(value: number): void;
  }

  export class GFTTDetector implements WasmGFTTDetectorHandle {
    private constructor();
    static create(
      maxFeatures?: number | null,
      qualityLevel?: number | null,
      minDistance?: number | null,
      blockSize?: number | null,
      useHarrisDetector?: boolean | null,
      k?: number | null,
    ): GFTTDetector;
    free(): void;
    getBlockSize(): number;
    getDefaultName(): string;
    getHarrisDetector(): boolean;
    getK(): number;
    getMaxFeatures(): number;
    getMinDistance(): number;
    getQualityLevel(): number;
    setBlockSize(value: number): void;
    setHarrisDetector(value: boolean): void;
    setK(value: number): void;
    setMaxFeatures(value: number): void;
    setMinDistance(value: number): void;
    setQualityLevel(value: number): void;
  }

  export class MSERConfig implements WasmMSERHandle {
    private constructor();
    static create(
      delta?: number | null,
      minArea?: number | null,
      maxArea?: number | null,
      pass2Only?: boolean | null,
    ): MSERConfig;
    free(): void;
    getDefaultName(): string;
    getDelta(): number;
    getMaxArea(): number;
    getMinArea(): number;
    getPass2Only(): boolean;
    setDelta(value: number): void;
    setMaxArea(value: number): void;
    setMinArea(value: number): void;
    setPass2Only(value: boolean): void;
  }

  export class TonemapDrago implements WasmTonemapDragoHandle {
    private constructor();
    static create(
      gamma?: number | null,
      saturation?: number | null,
      bias?: number | null,
    ): TonemapDrago;
    free(): void;
    getBias(): number;
    getGamma(): number;
    getSaturation(): number;
    setBias(value: number): void;
    setGamma(value: number): void;
    setSaturation(value: number): void;
  }

  export class TonemapMantiuk implements WasmTonemapMantiukHandle {
    private constructor();
    static create(
      gamma?: number | null,
      scale?: number | null,
      saturation?: number | null,
    ): TonemapMantiuk;
    free(): void;
    getGamma(): number;
    getSaturation(): number;
    getScale(): number;
    setGamma(value: number): void;
    setSaturation(value: number): void;
    setScale(value: number): void;
  }

  export class TonemapReinhard implements WasmTonemapReinhardHandle {
    private constructor();
    static create(
      gamma?: number | null,
      intensity?: number | null,
      lightAdaptation?: number | null,
      colorAdaptation?: number | null,
    ): TonemapReinhard;
    free(): void;
    getColorAdaptation(): number;
    getGamma(): number;
    getIntensity(): number;
    getLightAdaptation(): number;
    setColorAdaptation(value: number): void;
    setGamma(value: number): void;
    setIntensity(value: number): void;
    setLightAdaptation(value: number): void;
  }

  export default function initialize(): Promise<void>;
  export function initSync(input: { module: BufferSource | WebAssembly.Module }): void;

  export function grayscaleRgba(data: Uint8Array, width: number, height: number): Uint8Array;
  export function matCannyInto(
    source: import("./mat.js").WasmMatHandle,
    destination: import("./mat.js").WasmMatHandle,
    threshold1: number,
    threshold2: number,
    apertureSize: number,
    l2Gradient: boolean,
  ): void;
  export function matEqualizeHistInto(source: WasmMatHandle, destination: WasmMatHandle): void;
  export function matMatchTemplateInto(
    image: WasmMatHandle,
    template: WasmMatHandle,
    result: WasmMatHandle,
    method: number,
  ): void;
  export function matMatchTemplateMaskedInto(
    image: WasmMatHandle,
    template: WasmMatHandle,
    result: WasmMatHandle,
    method: number,
    mask: WasmMatHandle,
  ): void;
  export function matWarpAffineInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    transform: WasmMatHandle,
    width: number,
    height: number,
    flags: number,
    borderType: number,
    borderValue: Float64Array,
  ): void;
  export function matVectorNew(): WasmMatVectorHandle;
  export function matApproxPolyDPInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    epsilon: number,
    closed: boolean,
  ): void;
  export function matFindContoursInto(
    source: WasmMatHandle,
    contours: WasmMatVectorHandle,
    hierarchy: WasmMatHandle,
    mode: number,
    method: number,
    offsetX: number,
    offsetY: number,
  ): void;
  export function matCvtColorInto(
    source: import("./mat.js").WasmMatHandle,
    destination: import("./mat.js").WasmMatHandle,
    code: number,
    destinationChannels: number,
  ): void;
  export function matResizeInto(
    source: import("./mat.js").WasmMatHandle,
    destination: import("./mat.js").WasmMatHandle,
    targetWidth: number,
    targetHeight: number,
    scaleX: number,
    scaleY: number,
    interpolation: number,
  ): void;
  export function matThresholdInto(
    source: import("./mat.js").WasmMatHandle,
    destination: import("./mat.js").WasmMatHandle,
    threshold: number,
    maximum: number,
    thresholdType: number,
  ): number;
  export function matGaussianBlurInto(
    source: import("./mat.js").WasmMatHandle,
    destination: import("./mat.js").WasmMatHandle,
    width: number,
    height: number,
    sigmaX: number,
    sigmaY: number,
    borderType: number,
  ): void;
  export function matMorphologyExInto(
    source: import("./mat.js").WasmMatHandle,
    destination: import("./mat.js").WasmMatHandle,
    operation: number,
    kernel: import("./mat.js").WasmMatHandle,
    anchorX: number,
    anchorY: number,
    iterations: number,
    borderType: number,
    borderValue: Float64Array,
    defaultBorderValue: boolean,
  ): void;
  export function matSobelInto(
    source: import("./mat.js").WasmMatHandle,
    destination: import("./mat.js").WasmMatHandle,
    destinationDepth: number,
    dx: number,
    dy: number,
    kernelSize: number,
    scale: number,
    delta: number,
    borderType: number,
  ): void;
  export function invertRgba(data: Uint8Array, width: number, height: number): Uint8Array;
  export function matEmpty(): WasmMatHandle;
  export function matFromU8(
    data: Uint8Array,
    rows: number,
    columns: number,
    channels: number,
  ): WasmMatHandle;
  export function matZerosU8(rows: number, columns: number, channels: number): WasmMatHandle;
  export function resizeNearestRgba(
    data: Uint8Array,
    width: number,
    height: number,
    targetWidth: number,
    targetHeight: number,
  ): Uint8Array;
  export function thresholdRgba(
    data: Uint8Array,
    width: number,
    height: number,
    threshold: number,
  ): Uint8Array;
}
