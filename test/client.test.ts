import { describe, expect, test } from "bun:test";

import {
  AGAST_FEATURE_DETECTOR_DEFAULTS,
  AgastFeatureDetector_DetectorType,
  AKAZE_DEFAULTS,
  AKAZE_DescriptorType,
  AKAZEDescriptorType,
  BindingError,
  COLOR_GRAY2RGBA,
  COLOR_RGBA2BGRA,
  COLOR_RGBA2GRAY,
  COLOR_RGB2RGBA,
  createOpenCv,
  createRgbaImage,
  FAST_FEATURE_DETECTOR_DEFAULTS,
  FastFeatureDetector_DetectorType,
  GFTT_DETECTOR_DEFAULTS,
  INTER_AREA,
  INTER_LINEAR,
  INTER_NEAREST,
  KAZE_DEFAULTS,
  KAZE_DiffusivityType,
  KAZEDiffusivity,
  MSER,
  MSER_DEFAULTS,
  OpenCvInputError,
  ORB_DEFAULTS,
  ORB_FAST_SCORE,
  ORB_HARRIS_SCORE,
  ORB_ScoreType,
  ORBScoreType,
  THRESH_BINARY,
  THRESH_OTSU,
  BORDER_CONSTANT,
  CHAIN_APPROX_SIMPLE,
  MORPH_DILATE,
  MORPH_ERODE,
  RETR_EXTERNAL,
  TM_SQDIFF,
  TM_SQDIFF_NORMED,
  TM_CCORR,
  TM_CCORR_NORMED,
  TM_CCOEFF,
  TM_CCOEFF_NORMED,
} from "../src/index.js";
import type {
  Mat,
  OpenCvBackend,
  WasmAgastFeatureDetectorFactory,
  WasmAgastFeatureDetectorHandle,
  WasmAKAZEFactory,
  WasmAKAZEHandle,
  WasmFastFeatureDetectorFactory,
  WasmFastFeatureDetectorHandle,
  WasmGFTTDetectorFactory,
  WasmGFTTDetectorHandle,
  WasmKAZEFactory,
  WasmKAZEHandle,
  WasmMSERFactory,
  WasmMSERHandle,
  WasmMatHandle,
  WasmMatVectorHandle,
  WasmORBFactory,
  WasmORBHandle,
  WasmTonemapDragoFactory,
  WasmTonemapDragoHandle,
  WasmTonemapMantiukFactory,
  WasmTonemapMantiukHandle,
  WasmTonemapReinhardFactory,
  WasmTonemapReinhardHandle,
} from "../src/index.js";

class CopyingMatHandle implements WasmMatHandle {
  byteLength: number;
  channels: number;
  columns: number;
  data: Uint8Array;
  depth: number;
  isContinuous: boolean;
  rowStride: number;
  rows: number;

  constructor(
    rows: number,
    columns: number,
    channels: number,
    data: Uint8Array,
    depth = 0,
    emptyIsContinuous = true,
  ) {
    this.rows = rows;
    this.columns = columns;
    this.channels = channels;
    this.data = data;
    this.depth = depth;
    const byteWidth = depthByteWidth(depth);
    this.byteLength = rows * columns * channels * byteWidth;
    this.isContinuous = rows > 0 && columns > 0 ? true : emptyIsContinuous;
    this.rowStride = columns * channels * byteWidth;
  }

  replaceFrom(source: WasmMatHandle, data: Uint8Array): void {
    this.rows = source.rows;
    this.columns = source.columns;
    this.channels = source.channels;
    this.depth = source.depth;
    this.data = new Uint8Array(data);
    this.byteLength = data.byteLength;
    this.isContinuous = source.rows > 0 && source.columns > 0 ? true : source.isContinuous;
    this.rowStride = source.columns * source.channels * depthByteWidth(source.depth);
  }

  free(): void {}

  copyFromBytes(data: Uint8Array): void {
    if (data.byteLength !== this.byteLength) {
      throw new OpenCvInputError("matrix buffer length mismatch");
    }
    this.data.set(data);
  }

  roi(row: number, column: number, rows: number, columns: number): WasmMatHandle {
    const byteWidth = depthByteWidth(this.depth);
    const output = new Uint8Array(rows * columns * this.channels * byteWidth);
    for (let targetRow = 0; targetRow < rows; targetRow += 1) {
      const sourceStart = (row + targetRow) * this.rowStride + column * this.channels * byteWidth;
      const sourceEnd = sourceStart + columns * this.channels * byteWidth;
      const targetStart = targetRow * columns * this.channels * byteWidth;
      output.set(this.data.subarray(sourceStart, sourceEnd), targetStart);
    }
    return new CopyingMatHandle(rows, columns, this.channels, output, this.depth);
  }

  toFloat32Array(): Float32Array {
    return new Float32Array(this.data.slice().buffer);
  }

  toFloat64Array(): Float64Array {
    return new Float64Array(this.data.slice().buffer);
  }

  toInt16Array(): Int16Array {
    return new Int16Array(this.data.slice().buffer);
  }

  toInt32Array(): Int32Array {
    return new Int32Array(this.data.slice().buffer);
  }

  toInt8Array(): Int8Array {
    return new Int8Array(this.data.slice().buffer);
  }

  toUint16Array(): Uint16Array {
    return new Uint16Array(this.data.slice().buffer);
  }

  toUint8Array(): Uint8Array {
    return new Uint8Array(this.data);
  }
}

class CopyingMatVectorHandle implements WasmMatVectorHandle {
  #values: WasmMatHandle[] = [];

  clear(): void {
    this.#values = [];
  }

  free(): void {
    this.clear();
  }

  get(index: number): WasmMatHandle {
    const value = this.#values[index];
    if (value === undefined)
      throw new OpenCvInputError(`MatVector index ${index} is out of bounds`);
    return new CopyingMatHandle(
      value.rows,
      value.columns,
      value.channels,
      value.toUint8Array(),
      value.depth,
      value.isContinuous,
    );
  }

  push_back(value: WasmMatHandle): void {
    this.#values.push(value);
  }

  replace(values: WasmMatHandle[]): void {
    this.#values = values;
  }

  size(): number {
    return this.#values.length;
  }
}

class CopyingAKAZEHandle implements WasmAKAZEHandle {
  #descriptorChannels: number;
  #descriptorSize: number;
  #descriptorType: number;
  #diffusivity: number;
  #freed = false;
  #octaveLayers: number;
  #octaves: number;
  #threshold: number;

  constructor(
    descriptorType: number,
    descriptorSize: number,
    descriptorChannels: number,
    threshold: number,
    octaves: number,
    octaveLayers: number,
    diffusivity: number,
    readonly onFree: () => void,
  ) {
    this.#descriptorType = descriptorType;
    this.#descriptorSize = descriptorSize;
    this.#descriptorChannels = descriptorChannels;
    this.#threshold = threshold;
    this.#octaves = octaves;
    this.#octaveLayers = octaveLayers;
    this.#diffusivity = diffusivity;
  }

  free(): void {
    if (this.#freed) return;
    this.#freed = true;
    this.onFree();
  }

  getDefaultName(): string {
    return "Feature2D.AKAZE";
  }

  getDescriptorChannels(): number {
    return this.#descriptorChannels;
  }

  getDescriptorSize(): number {
    return this.#descriptorSize;
  }

  getDescriptorType(): number {
    return this.#descriptorType;
  }

  getDiffusivity(): number {
    return this.#diffusivity;
  }

  getNOctaveLayers(): number {
    return this.#octaveLayers;
  }

  getNOctaves(): number {
    return this.#octaves;
  }

  getThreshold(): number {
    return this.#threshold;
  }

  setDescriptorChannels(value: number): void {
    this.#descriptorChannels = value;
  }

  setDescriptorSize(value: number): void {
    this.#descriptorSize = value;
  }

  setDescriptorType(value: number): void {
    if (!Number.isInteger(value) || value < 2 || value > 5) {
      throw new OpenCvInputError("invalid AKAZE descriptor type");
    }
    this.#descriptorType = value;
  }

  setDiffusivity(value: number): void {
    if (!Number.isInteger(value) || value < 0 || value > 3) {
      throw new OpenCvInputError("invalid AKAZE diffusivity");
    }
    this.#diffusivity = value;
  }

  setNOctaveLayers(value: number): void {
    this.#octaveLayers = value;
  }

  setNOctaves(value: number): void {
    this.#octaves = value;
  }

  setThreshold(value: number): void {
    this.#threshold = value;
  }
}

class CopyingGFTTDetectorHandle implements WasmGFTTDetectorHandle {
  #blockSize: number;
  #freed = false;
  #harrisDetector: boolean;
  #k: number;
  #maxFeatures: number;
  #minDistance: number;
  #qualityLevel: number;

  constructor(
    maxFeatures: number,
    qualityLevel: number,
    minDistance: number,
    blockSize: number,
    harrisDetector: boolean,
    k: number,
    readonly onFree: () => void,
  ) {
    this.#maxFeatures = maxFeatures;
    this.#qualityLevel = qualityLevel;
    this.#minDistance = minDistance;
    this.#blockSize = blockSize;
    this.#harrisDetector = harrisDetector;
    this.#k = k;
  }

  free(): void {
    if (this.#freed) return;
    this.#freed = true;
    this.onFree();
  }

  getBlockSize(): number {
    return this.#blockSize;
  }

  getDefaultName(): string {
    return "Feature2D.GFTTDetector";
  }

  getHarrisDetector(): boolean {
    return this.#harrisDetector;
  }

  getK(): number {
    return this.#k;
  }

  getMaxFeatures(): number {
    return this.#maxFeatures;
  }

  getMinDistance(): number {
    return this.#minDistance;
  }

  getQualityLevel(): number {
    return this.#qualityLevel;
  }

  setBlockSize(value: number): void {
    this.#blockSize = value;
  }

  setHarrisDetector(value: boolean): void {
    this.#harrisDetector = value;
  }

  setK(value: number): void {
    this.#k = value;
  }

  setMaxFeatures(value: number): void {
    this.#maxFeatures = value;
  }

  setMinDistance(value: number): void {
    this.#minDistance = value;
  }

  setQualityLevel(value: number): void {
    this.#qualityLevel = value;
  }
}

class CopyingMSERHandle implements WasmMSERHandle {
  #delta: number;
  #freed = false;
  #maxArea: number;
  #minArea: number;
  #pass2Only: boolean;

  constructor(
    delta: number,
    minArea: number,
    maxArea: number,
    pass2Only: boolean,
    readonly onFree: () => void,
  ) {
    this.#delta = delta;
    this.#minArea = minArea;
    this.#maxArea = maxArea;
    this.#pass2Only = pass2Only;
  }

  free(): void {
    if (this.#freed) return;
    this.#freed = true;
    this.onFree();
  }

  getDefaultName(): string {
    return "Feature2D.MSER";
  }

  getDelta(): number {
    return this.#delta;
  }

  getMaxArea(): number {
    return this.#maxArea;
  }

  getMinArea(): number {
    return this.#minArea;
  }

  getPass2Only(): boolean {
    return this.#pass2Only;
  }

  setDelta(value: number): void {
    this.#delta = value;
  }

  setMaxArea(value: number): void {
    this.#maxArea = value;
  }

  setMinArea(value: number): void {
    this.#minArea = value;
  }

  setPass2Only(value: boolean): void {
    this.#pass2Only = value;
  }
}

class CopyingKAZEHandle implements WasmKAZEHandle {
  #diffusivity: number;
  #extended: boolean;
  #freed = false;
  #octaveLayers: number;
  #octaves: number;
  #threshold: number;
  #upright: boolean;

  constructor(
    extended: boolean,
    upright: boolean,
    threshold: number,
    octaves: number,
    octaveLayers: number,
    diffusivity: number,
    readonly onFree: () => void,
  ) {
    this.#extended = extended;
    this.#upright = upright;
    this.#threshold = threshold;
    this.#octaves = octaves;
    this.#octaveLayers = octaveLayers;
    this.#diffusivity = diffusivity;
  }

  free(): void {
    if (this.#freed) return;
    this.#freed = true;
    this.onFree();
  }

  getDefaultName(): string {
    return "Feature2D.KAZE";
  }

  getDiffusivity(): number {
    return this.#diffusivity;
  }

  getExtended(): boolean {
    return this.#extended;
  }

  getNOctaveLayers(): number {
    return this.#octaveLayers;
  }

  getNOctaves(): number {
    return this.#octaves;
  }

  getThreshold(): number {
    return this.#threshold;
  }

  getUpright(): boolean {
    return this.#upright;
  }

  setDiffusivity(value: number): void {
    if (!Number.isInteger(value) || value < 0 || value > 3) {
      throw new OpenCvInputError("invalid KAZE diffusivity");
    }
    this.#diffusivity = value;
  }

  setExtended(value: boolean): void {
    this.#extended = value;
  }

  setNOctaveLayers(value: number): void {
    this.#octaveLayers = value;
  }

  setNOctaves(value: number): void {
    this.#octaves = value;
  }

  setThreshold(value: number): void {
    this.#threshold = value;
  }

  setUpright(value: boolean): void {
    this.#upright = value;
  }
}

class CopyingORBHandle implements WasmORBHandle {
  #edgeThreshold: number;
  #fastThreshold: number;
  #firstLevel: number;
  #freed = false;
  #maxFeatures: number;
  #nLevels: number;
  #patchSize: number;
  #scaleFactor: number;
  #scoreType: number;
  #wtaK: number;

  constructor(
    maxFeatures: number,
    scaleFactor: number,
    nLevels: number,
    edgeThreshold: number,
    firstLevel: number,
    wtaK: number,
    scoreType: number,
    patchSize: number,
    fastThreshold: number,
    readonly onFree: () => void,
  ) {
    this.#maxFeatures = maxFeatures;
    this.#scaleFactor = scaleFactor;
    this.#nLevels = nLevels;
    this.#edgeThreshold = edgeThreshold;
    this.#firstLevel = firstLevel;
    this.#wtaK = wtaK;
    this.#scoreType = scoreType;
    this.#patchSize = patchSize;
    this.#fastThreshold = fastThreshold;
  }

  free(): void {
    if (this.#freed) return;
    this.#freed = true;
    this.onFree();
  }

  getDefaultName(): string {
    return "Feature2D.ORB";
  }

  getFastThreshold(): number {
    return this.#fastThreshold;
  }

  configuration(): readonly [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ] {
    return [
      this.#maxFeatures,
      this.#scaleFactor,
      this.#nLevels,
      this.#edgeThreshold,
      this.#firstLevel,
      this.#wtaK,
      this.#scoreType,
      this.#patchSize,
      this.#fastThreshold,
    ];
  }

  setEdgeThreshold(value: number): void {
    this.#edgeThreshold = value;
  }

  setFastThreshold(value: number): void {
    this.#fastThreshold = value;
  }

  setFirstLevel(value: number): void {
    if (value < 0) throw new OpenCvInputError("ORB first level must be zero or greater");
    this.#firstLevel = value;
  }

  setMaxFeatures(value: number): void {
    this.#maxFeatures = value;
  }

  setNLevels(value: number): void {
    this.#nLevels = value;
  }

  setPatchSize(value: number): void {
    this.#patchSize = value;
  }

  setScaleFactor(value: number): void {
    this.#scaleFactor = value;
  }

  setScoreType(value: number): void {
    this.#scoreType = value;
  }

  setWTA_K(value: number): void {
    this.#wtaK = value;
  }
}

class CopyingTonemapHandle
  implements WasmTonemapDragoHandle, WasmTonemapMantiukHandle, WasmTonemapReinhardHandle
{
  #bias: number;
  #colorAdaptation: number;
  #freed = false;
  #gamma: number;
  #intensity: number;
  #lightAdaptation: number;
  #saturation: number;
  #scale: number;

  constructor(
    state: {
      readonly bias?: number;
      readonly colorAdaptation?: number;
      readonly gamma: number;
      readonly intensity?: number;
      readonly lightAdaptation?: number;
      readonly saturation?: number;
      readonly scale?: number;
    },
    readonly onFree: () => void,
  ) {
    this.#bias = state.bias ?? Math.fround(0.85);
    this.#colorAdaptation = state.colorAdaptation ?? 0;
    this.#gamma = state.gamma;
    this.#intensity = state.intensity ?? 0;
    this.#lightAdaptation = state.lightAdaptation ?? 1;
    this.#saturation = state.saturation ?? 1;
    this.#scale = state.scale ?? Math.fround(0.7);
  }

  free(): void {
    if (this.#freed) return;
    this.#freed = true;
    this.onFree();
  }

  getBias(): number {
    return this.#bias;
  }

  getColorAdaptation(): number {
    return this.#colorAdaptation;
  }

  getGamma(): number {
    return this.#gamma;
  }

  getIntensity(): number {
    return this.#intensity;
  }

  getLightAdaptation(): number {
    return this.#lightAdaptation;
  }

  getSaturation(): number {
    return this.#saturation;
  }

  getScale(): number {
    return this.#scale;
  }

  setBias(value: number): void {
    this.#bias = value;
  }

  setColorAdaptation(value: number): void {
    this.#colorAdaptation = value;
  }

  setGamma(value: number): void {
    this.#gamma = value;
  }

  setIntensity(value: number): void {
    this.#intensity = value;
  }

  setLightAdaptation(value: number): void {
    this.#lightAdaptation = value;
  }

  setSaturation(value: number): void {
    this.#saturation = value;
  }

  setScale(value: number): void {
    this.#scale = value;
  }
}

class CopyingAgastFeatureDetectorHandle implements WasmAgastFeatureDetectorHandle {
  #freed = false;
  #nonmaxSuppression: boolean;
  #threshold: number;
  #type: number;

  constructor(
    threshold: number,
    nonmaxSuppression: boolean,
    type: number,
    readonly onFree: () => void,
  ) {
    this.#threshold = threshold;
    this.#nonmaxSuppression = nonmaxSuppression;
    this.#type = type;
  }

  free(): void {
    if (this.#freed) return;
    this.#freed = true;
    this.onFree();
  }

  getDefaultName(): string {
    return "Feature2D.AgastFeatureDetector";
  }

  getNonmaxSuppression(): boolean {
    return this.#nonmaxSuppression;
  }

  getThreshold(): number {
    return this.#threshold;
  }

  getType(): number {
    return this.#type;
  }

  setNonmaxSuppression(value: boolean): void {
    this.#nonmaxSuppression = value;
  }

  setThreshold(value: number): void {
    this.#threshold = value;
  }

  setType(value: number): void {
    this.#type = value;
  }
}

class CopyingFastFeatureDetectorHandle implements WasmFastFeatureDetectorHandle {
  #freed = false;
  #nonmaxSuppression: boolean;
  #threshold: number;
  #type: number;

  constructor(
    threshold: number,
    nonmaxSuppression: boolean,
    type: number,
    readonly onFree: () => void,
  ) {
    this.#threshold = threshold;
    this.#nonmaxSuppression = nonmaxSuppression;
    this.#type = type;
  }

  free(): void {
    if (this.#freed) return;
    this.#freed = true;
    this.onFree();
  }

  getDefaultName(): string {
    return "Feature2D.FastFeatureDetector";
  }

  getNonmaxSuppression(): boolean {
    return this.#nonmaxSuppression;
  }

  getThreshold(): number {
    return this.#threshold;
  }

  getType(): number {
    return this.#type;
  }

  setNonmaxSuppression(value: boolean): void {
    this.#nonmaxSuppression = value;
  }

  setThreshold(value: number): void {
    this.#threshold = value;
  }

  setType(value: number): void {
    this.#type = value;
  }
}

function depthByteWidth(depth: number): number {
  if (depth === 0 || depth === 1) {
    return 1;
  }
  if (depth === 2 || depth === 3) {
    return 2;
  }
  if (depth === 4 || depth === 5) {
    return 4;
  }
  return 8;
}

function copyViewBytes(data: ArrayBufferView): Uint8Array {
  return new Uint8Array(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
}

function mergeHandles(sources: readonly WasmMatHandle[]): WasmMatHandle {
  const first = sources[0];
  if (first === undefined) {
    throw new OpenCvInputError("merge requires a source");
  }
  const scalarWidth = depthByteWidth(first.depth);
  const channels = sources.reduce((total, source) => total + source.channels, 0);
  const output = new Uint8Array(first.rows * first.columns * channels * scalarWidth);
  for (let pixel = 0; pixel < first.rows * first.columns; pixel += 1) {
    let targetChannel = 0;
    for (const source of sources) {
      const bytes = source.toUint8Array();
      const sourceStart = pixel * source.channels * scalarWidth;
      const targetStart = (pixel * channels + targetChannel) * scalarWidth;
      output.set(
        bytes.subarray(sourceStart, sourceStart + source.channels * scalarWidth),
        targetStart,
      );
      targetChannel += source.channels;
    }
  }
  return new CopyingMatHandle(first.rows, first.columns, channels, output, first.depth);
}

function concatHandles(
  sources: readonly WasmMatHandle[],
  direction: "horizontal" | "vertical",
): WasmMatHandle {
  const first = sources[0];
  if (first === undefined) throw new OpenCvInputError("concat requires a source");
  const rowBytes = (matrix: WasmMatHandle) =>
    matrix.columns * matrix.channels * depthByteWidth(matrix.depth);
  if (direction === "vertical") {
    const output = new Uint8Array(sources.reduce((total, source) => total + source.byteLength, 0));
    let offset = 0;
    for (const source of sources) {
      output.set(source.toUint8Array(), offset);
      offset += source.byteLength;
    }
    return new CopyingMatHandle(
      sources.reduce((total, source) => total + source.rows, 0),
      first.columns,
      first.channels,
      output,
      first.depth,
    );
  }
  const outputColumns = sources.reduce((total, source) => total + source.columns, 0);
  const output = new Uint8Array(
    first.rows * outputColumns * first.channels * depthByteWidth(first.depth),
  );
  let offset = 0;
  for (let row = 0; row < first.rows; row += 1) {
    for (const source of sources) {
      const bytes = source.toUint8Array();
      const width = rowBytes(source);
      output.set(bytes.subarray(row * width, (row + 1) * width), offset);
      offset += width;
    }
  }
  return new CopyingMatHandle(first.rows, outputColumns, first.channels, output, first.depth);
}

function floatValues(source: WasmMatHandle): Float32Array | Float64Array {
  return source.depth === 5 ? source.toFloat32Array() : source.toFloat64Array();
}

function floatHandle(source: WasmMatHandle, values: readonly number[]): WasmMatHandle {
  const typed = source.depth === 5 ? new Float32Array(values) : new Float64Array(values);
  return new CopyingMatHandle(
    source.rows,
    source.columns,
    source.channels,
    copyViewBytes(typed),
    source.depth,
  );
}

function mapFloatHandle(
  source: WasmMatHandle,
  operation: (value: number) => number,
): WasmMatHandle {
  return floatHandle(source, Array.from(floatValues(source), operation));
}

function zipFloatHandles(
  left: WasmMatHandle,
  right: WasmMatHandle,
  operation: (left: number, right: number) => number,
): WasmMatHandle {
  const leftValues = floatValues(left);
  const rightValues = floatValues(right);
  return floatHandle(
    left,
    Array.from(leftValues, (value, index) => operation(value, rightValues[index] ?? Number.NaN)),
  );
}

function binaryNumericU8(
  left: WasmMatHandle,
  right: WasmMatHandle,
  operation: (left: number, right: number) => number,
): WasmMatHandle {
  const rightBytes = right.toUint8Array();
  const output = Uint8Array.from(left.toUint8Array(), (value, index) =>
    Math.min(255, Math.max(0, Math.round(operation(value, rightBytes[index] ?? 0)))),
  );
  return new CopyingMatHandle(left.rows, left.columns, left.channels, output);
}

function normBytes(values: Uint8Array, normType: number): number {
  const baseType = normType & 7;
  if (baseType === 1) return Math.max(0, ...values);
  if (baseType === 2) return values.reduce((sum, value) => sum + Math.abs(value), 0);
  if (baseType === 5) return values.reduce((sum, value) => sum + value * value, 0);
  return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
}

function maskedBytes(source: WasmMatHandle, mask: WasmMatHandle): Uint8Array {
  const input = source.toUint8Array();
  const selected = mask.toUint8Array();
  const output: number[] = [];
  for (let pixel = 0; pixel < source.rows * source.columns; pixel += 1) {
    if (selected[pixel] === 0) continue;
    const offset = pixel * source.channels;
    output.push(...input.subarray(offset, offset + source.channels));
  }
  return Uint8Array.from(output);
}

function writeMeanStdDev(
  source: WasmMatHandle,
  means: WasmMatHandle,
  deviations: WasmMatHandle,
  mask?: WasmMatHandle,
): void {
  const input = source.toUint8Array();
  const selected = mask?.toUint8Array();
  const channels: number[][] = Array.from({ length: source.channels }, () => []);
  for (let pixel = 0; pixel < source.rows * source.columns; pixel += 1) {
    if (selected !== undefined && selected[pixel] === 0) continue;
    for (let channel = 0; channel < source.channels; channel += 1) {
      channels[channel]?.push(input[pixel * source.channels + channel] ?? 0);
    }
  }
  const average = channels.map((values) =>
    values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length,
  );
  const standardDeviation = channels.map((values, channel) => {
    const mean = average[channel] ?? 0;
    return values.length === 0
      ? 0
      : Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
  });
  means.copyFromBytes(copyViewBytes(new Float64Array(average)));
  deviations.copyFromBytes(copyViewBytes(new Float64Array(standardDeviation)));
}

class CopyingBackend implements OpenCvBackend {
  readonly templateMatchCalls: Array<{
    image: WasmMatHandle;
    template: WasmMatHandle;
    result: WasmMatHandle;
    method: number;
    mask: WasmMatHandle | undefined;
  }> = [];

  matMatchTemplateInto(
    image: WasmMatHandle,
    template: WasmMatHandle,
    result: WasmMatHandle,
    method: number,
  ): void {
    this.templateMatchCalls.push({ image, template, result, method, mask: undefined });
  }

  matMatchTemplateMaskedInto(
    image: WasmMatHandle,
    template: WasmMatHandle,
    result: WasmMatHandle,
    method: number,
    mask: WasmMatHandle,
  ): void {
    this.templateMatchCalls.push({ image, template, result, method, mask });
  }
  #agastFeatureDetectorFreeCount = 0;
  #akazeFreeCount = 0;
  #fastFeatureDetectorFreeCount = 0;
  #gfttDetectorFreeCount = 0;
  #kazeFreeCount = 0;
  #mserFreeCount = 0;
  #orbFreeCount = 0;
  #tonemapFreeCount = 0;
  #logLevel = 3;
  #randomState = 0;
  readonly cartToPolarDegreeFlags: boolean[] = [];
  readonly numericIntoCalls: Array<{
    readonly method: string;
    readonly scale: number;
    readonly dtype: number;
  }> = [];
  readonly optimalDftSizeInputs: number[] = [];
  readonly pointPolygonTestInputs: Array<{
    readonly x: number;
    readonly y: number;
    readonly measureDistance: boolean;
  }> = [];
  readonly rotationMatrixInputs: Array<{
    readonly centerX: number;
    readonly centerY: number;
    readonly angleDegrees: number;
    readonly scale: number;
  }> = [];
  readonly polarToCartDegreeFlags: boolean[] = [];

  readonly AKAZE: WasmAKAZEFactory = {
    create: (
      descriptorType,
      descriptorSize,
      descriptorChannels,
      threshold,
      octaves,
      octaveLayers,
      diffusivity,
      maxPoints,
    ): WasmAKAZEHandle => {
      const resolvedMaxPoints = maxPoints ?? AKAZE_DEFAULTS.maxPoints;
      if (!Number.isInteger(resolvedMaxPoints)) {
        throw new OpenCvInputError("invalid AKAZE maximum point count");
      }
      return new CopyingAKAZEHandle(
        descriptorType ?? AKAZE_DEFAULTS.descriptorType,
        descriptorSize ?? AKAZE_DEFAULTS.descriptorSize,
        descriptorChannels ?? AKAZE_DEFAULTS.descriptorChannels,
        threshold ?? AKAZE_DEFAULTS.threshold,
        octaves ?? AKAZE_DEFAULTS.octaves,
        octaveLayers ?? AKAZE_DEFAULTS.octaveLayers,
        diffusivity ?? AKAZE_DEFAULTS.diffusivity,
        () => {
          this.#akazeFreeCount += 1;
        },
      );
    },
  };

  readonly KAZE: WasmKAZEFactory = {
    create: (extended, upright, threshold, octaves, octaveLayers, diffusivity): WasmKAZEHandle =>
      new CopyingKAZEHandle(
        extended ?? KAZE_DEFAULTS.extended,
        upright ?? KAZE_DEFAULTS.upright,
        threshold ?? KAZE_DEFAULTS.threshold,
        octaves ?? KAZE_DEFAULTS.octaves,
        octaveLayers ?? KAZE_DEFAULTS.octaveLayers,
        diffusivity ?? KAZE_DEFAULTS.diffusivity,
        () => {
          this.#kazeFreeCount += 1;
        },
      ),
  };

  readonly ORB: WasmORBFactory = {
    create: (
      maxFeatures,
      scaleFactor,
      nLevels,
      edgeThreshold,
      firstLevel,
      wtaK,
      scoreType,
      patchSize,
      fastThreshold,
    ): WasmORBHandle =>
      new CopyingORBHandle(
        maxFeatures ?? ORB_DEFAULTS.maxFeatures,
        scaleFactor ?? ORB_DEFAULTS.scaleFactor,
        nLevels ?? ORB_DEFAULTS.nLevels,
        edgeThreshold ?? ORB_DEFAULTS.edgeThreshold,
        firstLevel ?? ORB_DEFAULTS.firstLevel,
        wtaK ?? ORB_DEFAULTS.wtaK,
        scoreType ?? ORB_DEFAULTS.scoreType,
        patchSize ?? ORB_DEFAULTS.patchSize,
        fastThreshold ?? ORB_DEFAULTS.fastThreshold,
        () => {
          this.#orbFreeCount += 1;
        },
      ),
  };

  readonly GFTTDetector: WasmGFTTDetectorFactory = {
    create: (
      maxFeatures,
      qualityLevel,
      minDistance,
      blockSize,
      useHarrisDetector,
      k,
    ): WasmGFTTDetectorHandle =>
      new CopyingGFTTDetectorHandle(
        maxFeatures ?? GFTT_DETECTOR_DEFAULTS.maxFeatures,
        qualityLevel ?? GFTT_DETECTOR_DEFAULTS.qualityLevel,
        minDistance ?? GFTT_DETECTOR_DEFAULTS.minDistance,
        blockSize ?? GFTT_DETECTOR_DEFAULTS.blockSize,
        useHarrisDetector ?? GFTT_DETECTOR_DEFAULTS.useHarrisDetector,
        k ?? GFTT_DETECTOR_DEFAULTS.k,
        () => {
          this.#gfttDetectorFreeCount += 1;
        },
      ),
  };

  readonly MSERConfig: WasmMSERFactory = {
    create: (delta, minArea, maxArea, pass2Only): WasmMSERHandle =>
      new CopyingMSERHandle(
        delta ?? MSER_DEFAULTS.delta,
        minArea ?? MSER_DEFAULTS.minArea,
        maxArea ?? MSER_DEFAULTS.maxArea,
        pass2Only ?? MSER_DEFAULTS.pass2Only,
        () => {
          this.#mserFreeCount += 1;
        },
      ),
  };

  readonly TonemapDrago: WasmTonemapDragoFactory = {
    create: (gamma, saturation, bias): WasmTonemapDragoHandle =>
      new CopyingTonemapHandle(
        {
          bias: bias ?? Math.fround(0.85),
          gamma: gamma ?? 1,
          saturation: saturation ?? 1,
        },
        () => {
          this.#tonemapFreeCount += 1;
        },
      ),
  };

  readonly TonemapMantiuk: WasmTonemapMantiukFactory = {
    create: (gamma, scale, saturation): WasmTonemapMantiukHandle =>
      new CopyingTonemapHandle(
        {
          gamma: gamma ?? 1,
          saturation: saturation ?? 1,
          scale: scale ?? Math.fround(0.7),
        },
        () => {
          this.#tonemapFreeCount += 1;
        },
      ),
  };

  readonly TonemapReinhard: WasmTonemapReinhardFactory = {
    create: (gamma, intensity, lightAdaptation, colorAdaptation): WasmTonemapReinhardHandle =>
      new CopyingTonemapHandle(
        {
          colorAdaptation: colorAdaptation ?? 0,
          gamma: gamma ?? 1,
          intensity: intensity ?? 0,
          lightAdaptation: lightAdaptation ?? 1,
        },
        () => {
          this.#tonemapFreeCount += 1;
        },
      ),
  };

  readonly AgastFeatureDetector: WasmAgastFeatureDetectorFactory = {
    create: (threshold, nonmaxSuppression, type): WasmAgastFeatureDetectorHandle =>
      new CopyingAgastFeatureDetectorHandle(
        threshold ?? AGAST_FEATURE_DETECTOR_DEFAULTS.threshold,
        nonmaxSuppression ?? AGAST_FEATURE_DETECTOR_DEFAULTS.nonmaxSuppression,
        type ?? AGAST_FEATURE_DETECTOR_DEFAULTS.type,
        () => {
          this.#agastFeatureDetectorFreeCount += 1;
        },
      ),
  };

  get agastFeatureDetectorFreeCount(): number {
    return this.#agastFeatureDetectorFreeCount;
  }

  get kazeFreeCount(): number {
    return this.#kazeFreeCount;
  }

  get orbFreeCount(): number {
    return this.#orbFreeCount;
  }

  get mserFreeCount(): number {
    return this.#mserFreeCount;
  }

  get tonemapFreeCount(): number {
    return this.#tonemapFreeCount;
  }

  get gfttDetectorFreeCount(): number {
    return this.#gfttDetectorFreeCount;
  }

  readonly FastFeatureDetector: WasmFastFeatureDetectorFactory = {
    create: (threshold, nonmaxSuppression, type): WasmFastFeatureDetectorHandle =>
      new CopyingFastFeatureDetectorHandle(
        threshold ?? FAST_FEATURE_DETECTOR_DEFAULTS.threshold,
        nonmaxSuppression ?? FAST_FEATURE_DETECTOR_DEFAULTS.nonmaxSuppression,
        type ?? FAST_FEATURE_DETECTOR_DEFAULTS.type,
        () => {
          this.#fastFeatureDetectorFreeCount += 1;
        },
      ),
  };

  get fastFeatureDetectorFreeCount(): number {
    return this.#fastFeatureDetectorFreeCount;
  }

  get akazeFreeCount(): number {
    return this.#akazeFreeCount;
  }

  clipLine(
    rectangleX: number,
    rectangleY: number,
    rectangleWidth: number,
    rectangleHeight: number,
    startX: number,
    startY: number,
    endX: number,
    endY: number,
  ): Int32Array {
    const minimumX = rectangleX;
    const maximumX = rectangleX + rectangleWidth - 1;
    const minimumY = rectangleY;
    const maximumY = rectangleY + rectangleHeight - 1;
    if (startY === endY && startY >= minimumY && startY <= maximumY) {
      const clippedStart = Math.max(minimumX, Math.min(maximumX, startX));
      const clippedEnd = Math.max(minimumX, Math.min(maximumX, endX));
      if (Math.max(startX, endX) < minimumX || Math.min(startX, endX) > maximumX) {
        return new Int32Array();
      }
      return new Int32Array([clippedStart, startY, clippedEnd, endY]);
    }
    return new Int32Array();
  }

  createHanningWindow(columns: number, rows: number, depth: number): WasmMatHandle {
    const values = Array.from({ length: rows * columns }, (_, index) => {
      const row = Math.floor(index / columns);
      const column = index % columns;
      const vertical = 0.5 * (1 - Math.cos((2 * Math.PI * row) / (rows - 1)));
      const horizontal = 0.5 * (1 - Math.cos((2 * Math.PI * column) / (columns - 1)));
      return cleanTiny(Math.sqrt(vertical * horizontal));
    });
    const typed = depth === 5 ? new Float32Array(values) : new Float64Array(values);
    return new CopyingMatHandle(rows, columns, 1, copyViewBytes(typed), depth);
  }

  createHanningWindowInto(
    destination: WasmMatHandle,
    columns: number,
    rows: number,
    depth: number,
  ): void {
    if (columns < 2 || rows < 2 || (depth !== 5 && depth !== 6)) {
      throw new OpenCvInputError("invalid Hanning window arguments");
    }
    destination.copyFromBytes(this.createHanningWindow(columns, rows, depth).toUint8Array());
  }

  ellipse2Poly(
    centerX: number,
    centerY: number,
    axisX: number,
    axisY: number,
    rotationDegrees: number,
    arcStart: number,
    arcEnd: number,
    delta: number,
  ): Int32Array {
    const rotation = (rotationDegrees * Math.PI) / 180;
    const output: number[] = [];
    for (let angle = arcStart; angle <= arcEnd; angle += delta) {
      const radians = (Math.min(angle, arcEnd) * Math.PI) / 180;
      const x = axisX * Math.cos(radians);
      const y = axisY * Math.sin(radians);
      output.push(
        Math.round(centerX + x * Math.cos(rotation) - y * Math.sin(rotation)),
        Math.round(centerY + x * Math.sin(rotation) + y * Math.cos(rotation)),
      );
      if (angle + delta > arcEnd && angle !== arcEnd) angle = arcEnd - delta;
    }
    return new Int32Array(output);
  }

  getStructuringElement(
    kind: number,
    columns: number,
    rows: number,
    anchorX: number,
    anchorY: number,
  ): WasmMatHandle {
    const resolvedX = anchorX === -1 ? Math.floor(columns / 2) : anchorX;
    const resolvedY = anchorY === -1 ? Math.floor(rows / 2) : anchorY;
    const output = Uint8Array.from({ length: rows * columns }, (_, index) => {
      if (kind === 0) return 1;
      const row = Math.floor(index / columns);
      const column = index % columns;
      return kind === 1
        ? Number(row === resolvedY || column === resolvedX)
        : Number(
            ((column - resolvedX) / Math.max(resolvedX, 1)) ** 2 +
              ((row - resolvedY) / Math.max(resolvedY, 1)) ** 2 <=
              1,
          );
    });
    return new CopyingMatHandle(rows, columns, 1, output);
  }

  getLogLevel(): number {
    return this.#logLevel;
  }

  getOptimalDFTSize(size: number): number {
    this.optimalDftSizeInputs.push(size);
    if (size < 0 || size === 2_125_764_000) return -1;
    for (let candidate = Math.max(size, 1); candidate <= 2_125_764_000; candidate += 1) {
      let remainder = candidate;
      for (const factor of [2, 3, 5]) {
        while (remainder % factor === 0) remainder /= factor;
      }
      if (remainder === 1) return candidate;
    }
    return -1;
  }

  grayscaleRgba(data: Uint8Array): Uint8Array {
    return new Uint8Array(data);
  }

  invertRgba(data: Uint8Array): Uint8Array {
    return new Uint8Array(data);
  }

  matFromF32(data: Float32Array, rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(rows, columns, channels, copyViewBytes(data), 5);
  }

  matCannyInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    threshold1: number,
    threshold2: number,
    apertureSize: number,
    _l2Gradient: boolean,
  ): void {
    if (source.depth !== 0 || source.channels !== 1 || apertureSize !== 3) {
      throw new OpenCvInputError("mock Canny supports single-channel U8 aperture 3 only");
    }
    const input = source.toUint8Array();
    const output = new Uint8Array(input.length);
    const high = Math.max(threshold1, threshold2);
    for (let row = 0; row < source.rows; row += 1) {
      for (let column = 0; column + 1 < source.columns; column += 1) {
        const index = row * source.columns + column;
        if (Math.abs(input[index + 1]! - input[index]!) * 4 > high) output[index] = 255;
      }
    }
    replaceMockDestination(destination, source.rows, source.columns, 1, output, 0);
  }

  matFromF64(data: Float64Array, rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(rows, columns, channels, copyViewBytes(data), 6);
  }

  matFromI16(data: Int16Array, rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(rows, columns, channels, copyViewBytes(data), 3);
  }

  matFromI32(data: Int32Array, rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(rows, columns, channels, copyViewBytes(data), 4);
  }

  matFromI8(data: Int8Array, rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(rows, columns, channels, copyViewBytes(data), 1);
  }

  matFromU16(data: Uint16Array, rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(rows, columns, channels, copyViewBytes(data), 2);
  }

  matFromU8(data: Uint8Array, rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(rows, columns, channels, new Uint8Array(data));
  }

  matVectorNew(): WasmMatVectorHandle {
    return new CopyingMatVectorHandle();
  }

  approximationCalls: { epsilon: number; closed: boolean }[] = [];

  matApproxPolyDPInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    epsilon: number,
    closed: boolean,
  ): void {
    this.approximationCalls.push({ epsilon, closed });
    replaceMockDestination(
      destination,
      source.rows,
      source.columns,
      source.channels,
      source.toUint8Array(),
      source.depth,
    );
  }

  matFindContoursInto(
    _source: WasmMatHandle,
    contours: WasmMatVectorHandle,
    hierarchy: WasmMatHandle,
    mode: number,
    method: number,
    offsetX: number,
    offsetY: number,
  ): void {
    if (mode !== 0 || method !== 2 || !(contours instanceof CopyingMatVectorHandle)) {
      throw new OpenCvInputError("mock findContours supports external simple contours only");
    }
    const points = new Int32Array([
      1 + offsetX,
      1 + offsetY,
      1 + offsetX,
      3 + offsetY,
      3 + offsetX,
      3 + offsetY,
      3 + offsetX,
      1 + offsetY,
    ]);
    contours.replace([new CopyingMatHandle(4, 1, 2, copyViewBytes(points), 4)]);
    replaceMockDestination(hierarchy, 1, 1, 4, copyViewBytes(new Int32Array([-1, -1, -1, -1])), 4);
  }

  matCvtColorInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    code: number,
    destinationChannels: number,
  ): void {
    const input = source.toUint8Array();
    let output: Uint8Array;
    let channels: number;
    if (code === 11) {
      channels = 1;
      output = Uint8Array.from({ length: source.rows * source.columns }, (_, index) => {
        const offset = index * 4;
        return Math.round(
          input[offset]! * 0.299 + input[offset + 1]! * 0.587 + input[offset + 2]! * 0.114,
        );
      });
    } else if (code === 5) {
      channels = 4;
      output = new Uint8Array(input.byteLength);
      for (let offset = 0; offset < input.byteLength; offset += 4) {
        output.set(
          [input[offset + 2]!, input[offset + 1]!, input[offset]!, input[offset + 3]!],
          offset,
        );
      }
    } else if (code === 0) {
      channels = destinationChannels === 3 ? 3 : 4;
      output = Uint8Array.from({ length: source.rows * source.columns * channels }, (_, index) => {
        const channel = index % channels;
        if (channel === 3) return 255;
        return input[Math.floor(index / channels) * 3 + channel]!;
      });
    } else if (code === 9) {
      channels = destinationChannels === 3 ? 3 : 4;
      output = Uint8Array.from({ length: source.rows * source.columns * channels }, (_, index) => {
        const channel = index % channels;
        return channel === 3 ? 255 : input[Math.floor(index / channels)]!;
      });
    } else {
      throw new OpenCvInputError(`unsupported mock color code ${code}`);
    }
    if (!(destination instanceof CopyingMatHandle)) {
      throw new OpenCvInputError("mock destination must use CopyingMatHandle");
    }
    destination.replaceFrom(
      new CopyingMatHandle(source.rows, source.columns, channels, output),
      output,
    );
  }

  matEqualizeHistInto(source: WasmMatHandle, destination: WasmMatHandle): void {
    const input = source.toUint8Array();
    const histogram = new Uint32Array(256);
    for (const value of input) histogram[value] = (histogram[value] ?? 0) + 1;
    const first = histogram.findIndex((count) => count !== 0);
    const firstCount = histogram[first] ?? 0;
    let cumulative = 0;
    const lookup = new Uint8Array(256);
    for (let value = 0; value < histogram.length; value += 1) {
      cumulative += histogram[value] ?? 0;
      if (value > first) {
        lookup[value] = roundNearestEven(
          ((cumulative - firstCount) * 255) / (input.length - firstCount),
        );
      }
    }
    replaceMockDestination(
      destination,
      source.rows,
      source.columns,
      1,
      input.map((value) => lookup[value]!),
      0,
    );
  }

  perspectiveCalls: { flags: number; borderType: number; width: number; height: number }[] = [];
  matWarpPerspectiveInto(
    _source: WasmMatHandle,
    _destination: WasmMatHandle,
    _transform: WasmMatHandle,
    width: number,
    height: number,
    flags: number,
    borderType: number,
    _borderValue: Float64Array,
  ): void {
    this.perspectiveCalls.push({ flags, borderType, width, height });
  }
  matWarpAffineInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    transform: WasmMatHandle,
    width: number,
    height: number,
    flags: number,
    borderType: number,
    borderValue: Float64Array,
  ): void {
    if (flags !== 0 || borderType !== 0 || transform.depth !== 6) {
      throw new OpenCvInputError("mock warpAffine supports tested nearest constant F64 transforms");
    }
    const matrix = transform.toFloat64Array();
    const translateX = matrix[2] ?? 0;
    const translateY = matrix[5] ?? 0;
    const input = source.toUint8Array();
    const output = new Uint8Array(width * height * source.channels);
    for (let row = 0; row < height; row += 1) {
      for (let column = 0; column < width; column += 1) {
        const sourceX = column - translateX;
        const sourceY = row - translateY;
        for (let channel = 0; channel < source.channels; channel += 1) {
          const outputIndex = (row * width + column) * source.channels + channel;
          output[outputIndex] =
            sourceX >= 0 && sourceX < source.columns && sourceY >= 0 && sourceY < source.rows
              ? input[(sourceY * source.columns + sourceX) * source.channels + channel]!
              : roundNearestEven(borderValue[channel] ?? 0);
        }
      }
    }
    replaceMockDestination(destination, height, width, source.channels, output, 0);
  }

  matResizeInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    targetWidth: number,
    targetHeight: number,
    scaleX: number,
    scaleY: number,
    interpolation: number,
  ): void {
    if (interpolation !== 0 && interpolation !== 1 && interpolation !== 3) {
      throw new OpenCvInputError("mock supports nearest, linear, and area resize only");
    }
    const width = targetWidth === 0 ? Math.round(source.columns * scaleX) : targetWidth;
    const height = targetHeight === 0 ? Math.round(source.rows * scaleY) : targetHeight;
    const pixelBytes = source.channels * depthByteWidth(source.depth);
    const input = source.toUint8Array();
    const output = new Uint8Array(width * height * pixelBytes);
    for (let row = 0; row < height; row += 1) {
      for (let column = 0; column < width; column += 1) {
        const targetOffset = (row * width + column) * pixelBytes;
        if (interpolation === 0) {
          const sourceRow = Math.floor((row * source.rows) / height);
          const sourceColumn = Math.floor((column * source.columns) / width);
          const sourceOffset = (sourceRow * source.columns + sourceColumn) * pixelBytes;
          output.set(input.subarray(sourceOffset, sourceOffset + pixelBytes), targetOffset);
          continue;
        }
        if (interpolation === 3) {
          const sourceTop = (row * source.rows) / height;
          const sourceBottom = ((row + 1) * source.rows) / height;
          const sourceLeft = (column * source.columns) / width;
          const sourceRight = ((column + 1) * source.columns) / width;
          const area = (sourceBottom - sourceTop) * (sourceRight - sourceLeft);
          for (let channel = 0; channel < pixelBytes; channel += 1) {
            let sum = 0;
            for (
              let sourceRow = Math.floor(sourceTop);
              sourceRow < Math.ceil(sourceBottom);
              sourceRow += 1
            ) {
              const vertical =
                Math.min(sourceBottom, sourceRow + 1) - Math.max(sourceTop, sourceRow);
              for (
                let sourceColumn = Math.floor(sourceLeft);
                sourceColumn < Math.ceil(sourceRight);
                sourceColumn += 1
              ) {
                const horizontal =
                  Math.min(sourceRight, sourceColumn + 1) - Math.max(sourceLeft, sourceColumn);
                sum +=
                  input[(sourceRow * source.columns + sourceColumn) * pixelBytes + channel]! *
                  vertical *
                  horizontal;
              }
            }
            output[targetOffset + channel] = roundNearestEven(sum / area);
          }
          continue;
        }
        const sourceY = ((row + 0.5) * source.rows) / height - 0.5;
        const top = Math.max(0, Math.min(source.rows - 1, Math.floor(sourceY)));
        const bottom = Math.max(0, Math.min(source.rows - 1, Math.floor(sourceY) + 1));
        const vertical = sourceY - Math.floor(sourceY);
        const sourceX = ((column + 0.5) * source.columns) / width - 0.5;
        const left = Math.max(0, Math.min(source.columns - 1, Math.floor(sourceX)));
        const right = Math.max(0, Math.min(source.columns - 1, Math.floor(sourceX) + 1));
        const horizontal = sourceX - Math.floor(sourceX);
        for (let channel = 0; channel < pixelBytes; channel += 1) {
          const at = (sourceRow: number, sourceColumn: number) =>
            input[(sourceRow * source.columns + sourceColumn) * pixelBytes + channel]!;
          const topValue = at(top, left) * (1 - horizontal) + at(top, right) * horizontal;
          const bottomValue = at(bottom, left) * (1 - horizontal) + at(bottom, right) * horizontal;
          output[targetOffset + channel] = roundNearestEven(
            topValue * (1 - vertical) + bottomValue * vertical,
          );
        }
      }
    }
    if (!(destination instanceof CopyingMatHandle)) {
      throw new OpenCvInputError("mock destination must use CopyingMatHandle");
    }
    destination.replaceFrom(
      new CopyingMatHandle(height, width, source.channels, output, source.depth),
      output,
    );
  }

  matThresholdInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    threshold: number,
    maximum: number,
    thresholdType: number,
  ): number {
    if (source.depth !== 0) {
      throw new OpenCvInputError("mock threshold supports U8 matrices only");
    }
    const mode = thresholdType & 7;
    const input = source.toUint8Array();
    const usedThreshold = (thresholdType & 8) === 0 ? threshold : mockOtsuThreshold(input);
    const maximumU8 = roundNearestEven(Math.max(0, Math.min(255, maximum)));
    const truncatedU8 = Math.max(0, Math.min(255, Math.floor(usedThreshold)));
    const output = input.map((value) => {
      switch (mode) {
        case 0:
          return value > usedThreshold ? maximumU8 : 0;
        case 1:
          return value > usedThreshold ? 0 : maximumU8;
        case 2:
          return Math.min(value, truncatedU8);
        case 3:
          return value > usedThreshold ? value : 0;
        case 4:
          return value > usedThreshold ? 0 : value;
        default:
          throw new OpenCvInputError(`unsupported mock threshold mode ${mode}`);
      }
    });
    if (!(destination instanceof CopyingMatHandle)) {
      throw new OpenCvInputError("mock destination must use CopyingMatHandle");
    }
    destination.replaceFrom(
      new CopyingMatHandle(source.rows, source.columns, source.channels, output),
      output,
    );
    return usedThreshold;
  }

  matGaussianBlurInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    width: number,
    height: number,
    _sigmaX: number,
    _sigmaY: number,
    borderType: number,
  ): void {
    if (width !== 3 || height !== 1 || source.depth !== 0) {
      throw new OpenCvInputError("mock GaussianBlur supports a 3x1 U8 kernel only");
    }
    const input = source.toUint8Array();
    const output = new Uint8Array(input.length);
    for (let row = 0; row < source.rows; row += 1) {
      for (let column = 0; column < source.columns; column += 1) {
        for (let channel = 0; channel < source.channels; channel += 1) {
          let sum = 0;
          for (let offset = -1; offset <= 1; offset += 1) {
            const mapped = mockBorderIndex(column + offset, source.columns, borderType);
            if (mapped !== undefined) {
              const weight = offset === 0 ? 0.5 : 0.25;
              sum += input[(row * source.columns + mapped) * source.channels + channel]! * weight;
            }
          }
          output[(row * source.columns + column) * source.channels + channel] =
            roundNearestEven(sum);
        }
      }
    }
    replaceMockDestination(destination, source.rows, source.columns, source.channels, output, 0);
  }

  matMorphologyExInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    operation: number,
    kernel: WasmMatHandle,
    anchorX: number,
    anchorY: number,
    _iterations: number,
    borderType: number,
    _borderValue: Float64Array,
    _defaultBorderValue: boolean,
  ): void {
    if (source.depth !== 0 || kernel.depth !== 0 || kernel.channels !== 1) {
      throw new OpenCvInputError("mock morphology supports U8 matrices only");
    }
    const input = source.toUint8Array();
    const mask = kernel.toUint8Array();
    const resolvedX = anchorX < 0 ? Math.floor(kernel.columns / 2) : anchorX;
    const resolvedY = anchorY < 0 ? Math.floor(kernel.rows / 2) : anchorY;
    const primitive = (erode: boolean) => {
      const output = new Uint8Array(input.length);
      for (let row = 0; row < source.rows; row += 1) {
        for (let column = 0; column < source.columns; column += 1) {
          for (let channel = 0; channel < source.channels; channel += 1) {
            let selected = erode ? 255 : 0;
            for (let kernelY = 0; kernelY < kernel.rows; kernelY += 1) {
              for (let kernelX = 0; kernelX < kernel.columns; kernelX += 1) {
                if (mask[kernelY * kernel.columns + kernelX] === 0) continue;
                const mappedY = mockBorderIndex(row + kernelY - resolvedY, source.rows, borderType);
                const mappedX = mockBorderIndex(
                  column + kernelX - resolvedX,
                  source.columns,
                  borderType,
                );
                const value =
                  mappedY === undefined || mappedX === undefined
                    ? erode
                      ? 255
                      : 0
                    : input[(mappedY * source.columns + mappedX) * source.channels + channel]!;
                selected = erode ? Math.min(selected, value) : Math.max(selected, value);
              }
            }
            output[(row * source.columns + column) * source.channels + channel] = selected;
          }
        }
      }
      return output;
    };
    if (operation !== 0 && operation !== 1) {
      throw new OpenCvInputError("mock morphology supports erode and dilate only");
    }
    replaceMockDestination(
      destination,
      source.rows,
      source.columns,
      source.channels,
      primitive(operation === 0),
      0,
    );
  }

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
  ): void {
    if (source.depth !== 0 || destinationDepth !== 3 || dx !== 1 || dy !== 0 || kernelSize !== 3) {
      throw new OpenCvInputError("mock Sobel supports the tested U8-to-I16 X derivative only");
    }
    const input = source.toUint8Array();
    const horizontal = [-1, 0, 1];
    const vertical = [1, 2, 1];
    const output = new Int16Array(input.length);
    for (let row = 0; row < source.rows; row += 1) {
      for (let column = 0; column < source.columns; column += 1) {
        for (let channel = 0; channel < source.channels; channel += 1) {
          let sum = 0;
          for (let kernelY = 0; kernelY < 3; kernelY += 1) {
            const mappedY = mockBorderIndex(row + kernelY - 1, source.rows, borderType);
            for (let kernelX = 0; kernelX < 3; kernelX += 1) {
              const mappedX = mockBorderIndex(column + kernelX - 1, source.columns, borderType);
              if (mappedY !== undefined && mappedX !== undefined) {
                sum +=
                  input[(mappedY * source.columns + mappedX) * source.channels + channel]! *
                  vertical[kernelY]! *
                  horizontal[kernelX]!;
              }
            }
          }
          output[(row * source.columns + column) * source.channels + channel] = roundNearestEven(
            sum * scale + delta,
          );
        }
      }
    }
    replaceMockDestination(
      destination,
      source.rows,
      source.columns,
      source.channels,
      copyViewBytes(output),
      3,
    );
  }

  matEmpty(): WasmMatHandle {
    return new CopyingMatHandle(0, 0, 1, new Uint8Array(), 0, false);
  }

  matFlip(source: WasmMatHandle, flipCode: number): WasmMatHandle {
    const input = source.toUint8Array();
    const output = new Uint8Array(input.byteLength);
    const pixelBytes = source.channels * depthByteWidth(source.depth);
    for (let row = 0; row < source.rows; row += 1) {
      for (let column = 0; column < source.columns; column += 1) {
        const sourceRow = flipCode <= 0 ? source.rows - row - 1 : row;
        const sourceColumn = flipCode !== 0 ? source.columns - column - 1 : column;
        const sourceOffset = (sourceRow * source.columns + sourceColumn) * pixelBytes;
        const outputOffset = (row * source.columns + column) * pixelBytes;
        output.set(input.subarray(sourceOffset, sourceOffset + pixelBytes), outputOffset);
      }
    }
    return new CopyingMatHandle(source.rows, source.columns, source.channels, output, source.depth);
  }

  matFlipInto(source: WasmMatHandle, destination: WasmMatHandle, flipCode: number): void {
    destination.copyFromBytes(this.matFlip(source, flipCode).toUint8Array());
  }

  matSplit(source: WasmMatHandle): WasmMatHandle[] {
    const scalarWidth = depthByteWidth(source.depth);
    const input = source.toUint8Array();
    return Array.from({ length: source.channels }, (_, channel) => {
      const output = new Uint8Array(source.rows * source.columns * scalarWidth);
      for (let pixel = 0; pixel < source.rows * source.columns; pixel += 1) {
        const inputOffset = (pixel * source.channels + channel) * scalarWidth;
        output.set(input.subarray(inputOffset, inputOffset + scalarWidth), pixel * scalarWidth);
      }
      return new CopyingMatHandle(source.rows, source.columns, 1, output, source.depth);
    });
  }

  matMerge(first: WasmMatHandle, second: WasmMatHandle): WasmMatHandle {
    return mergeHandles([first, second]);
  }

  matMerge3(first: WasmMatHandle, second: WasmMatHandle, third: WasmMatHandle): WasmMatHandle {
    return mergeHandles([first, second, third]);
  }

  matMerge4(
    first: WasmMatHandle,
    second: WasmMatHandle,
    third: WasmMatHandle,
    fourth: WasmMatHandle,
  ): WasmMatHandle {
    return mergeHandles([first, second, third, fourth]);
  }

  matMixChannels(source: WasmMatHandle, destination: WasmMatHandle, fromTo: Uint16Array): void {
    const scalarWidth = depthByteWidth(source.depth);
    const input = source.toUint8Array();
    const output = destination.toUint8Array();
    for (let pixel = 0; pixel < source.rows * source.columns; pixel += 1) {
      for (let index = 0; index < fromTo.length; index += 2) {
        const sourceChannel = fromTo[index] ?? 0;
        const destinationChannel = fromTo[index + 1] ?? 0;
        const sourceOffset = (pixel * source.channels + sourceChannel) * scalarWidth;
        const destinationOffset = (pixel * destination.channels + destinationChannel) * scalarWidth;
        output.set(input.subarray(sourceOffset, sourceOffset + scalarWidth), destinationOffset);
      }
    }
    destination.copyFromBytes(output);
  }

  matExtractChannel(source: WasmMatHandle, channel: number): WasmMatHandle {
    const output = this.matSplit(source)[channel];
    if (output === undefined) {
      throw new OpenCvInputError("channel is out of bounds");
    }
    return output;
  }

  matInsertChannel(source: WasmMatHandle, destination: WasmMatHandle, channel: number): void {
    const scalarWidth = depthByteWidth(destination.depth);
    const input = source.toUint8Array();
    const output = destination.toUint8Array();
    for (let pixel = 0; pixel < destination.rows * destination.columns; pixel += 1) {
      const sourceOffset = pixel * scalarWidth;
      const destinationOffset = (pixel * destination.channels + channel) * scalarWidth;
      output.set(input.subarray(sourceOffset, sourceOffset + scalarWidth), destinationOffset);
    }
    destination.copyFromBytes(output);
  }

  matLut(source: WasmMatHandle, table: WasmMatHandle): WasmMatHandle {
    const input = source.toUint8Array();
    const lookup = table.toUint8Array();
    const output = Uint8Array.from(input, (value, index) => {
      const channel = index % source.channels;
      const tableChannel = table.channels === 1 ? 0 : channel;
      return lookup[value * table.channels + tableChannel] ?? 0;
    });
    return new CopyingMatHandle(source.rows, source.columns, source.channels, output, table.depth);
  }

  matLutInto(source: WasmMatHandle, table: WasmMatHandle, destination: WasmMatHandle): void {
    destination.copyFromBytes(this.matLut(source, table).toUint8Array());
  }

  matNorm(source: WasmMatHandle, normType: number): number {
    return normBytes(source.toUint8Array(), normType);
  }

  matNormMasked(source: WasmMatHandle, normType: number, mask: WasmMatHandle): number {
    return normBytes(maskedBytes(source, mask), normType);
  }

  matNormDiff(first: WasmMatHandle, second: WasmMatHandle, normType: number): number {
    const right = second.toUint8Array();
    return normBytes(
      Uint8Array.from(first.toUint8Array(), (value, index) =>
        Math.abs(value - (right[index] ?? 0)),
      ),
      normType,
    );
  }

  matNormDiffMasked(
    first: WasmMatHandle,
    second: WasmMatHandle,
    normType: number,
    mask: WasmMatHandle,
  ): number {
    const right = second.toUint8Array();
    const difference = Uint8Array.from(first.toUint8Array(), (value, index) =>
      Math.abs(value - (right[index] ?? 0)),
    );
    const handle = new CopyingMatHandle(first.rows, first.columns, first.channels, difference);
    return normBytes(maskedBytes(handle, mask), normType);
  }

  matNormalizeInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    alpha: number,
    beta: number,
    normType: number,
  ): void {
    const input = source.toUint8Array();
    const denominator = normBytes(input, normType);
    destination.copyFromBytes(
      Uint8Array.from(input, (value) =>
        Math.round(denominator === 0 ? 0 : (value * alpha) / denominator + beta),
      ),
    );
  }

  matNormalizeMaskedInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    alpha: number,
    beta: number,
    normType: number,
    mask: WasmMatHandle,
  ): void {
    const before = destination.toUint8Array();
    const temporary = new CopyingMatHandle(
      source.rows,
      source.columns,
      source.channels,
      new Uint8Array(source.byteLength),
    );
    this.matNormalizeInto(source, temporary, alpha, beta, normType);
    const normalized = temporary.toUint8Array();
    const maskBytes = mask.toUint8Array();
    for (let pixel = 0; pixel < source.rows * source.columns; pixel += 1) {
      if (maskBytes[pixel] === 0) continue;
      const offset = pixel * source.channels;
      before.set(normalized.subarray(offset, offset + source.channels), offset);
    }
    destination.copyFromBytes(before);
  }

  matMeanStdDevInto(
    source: WasmMatHandle,
    means: WasmMatHandle,
    standardDeviations: WasmMatHandle,
  ): void {
    writeMeanStdDev(source, means, standardDeviations);
  }

  matMeanStdDevMaskedInto(
    source: WasmMatHandle,
    means: WasmMatHandle,
    standardDeviations: WasmMatHandle,
    mask: WasmMatHandle,
  ): void {
    writeMeanStdDev(source, means, standardDeviations, mask);
  }

  matReduceInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    axis: number,
    kind: number,
  ): void {
    const input = source.toUint8Array();
    const output: number[] = [];
    const major = axis === 0 ? source.columns : source.rows;
    const count = axis === 0 ? source.rows : source.columns;
    for (let position = 0; position < major; position += 1) {
      for (let channel = 0; channel < source.channels; channel += 1) {
        const values: number[] = [];
        for (let index = 0; index < count; index += 1) {
          const row = axis === 0 ? index : position;
          const column = axis === 0 ? position : index;
          values.push(input[(row * source.columns + column) * source.channels + channel] ?? 0);
        }
        const sum = values.reduce((total, value) => total + value, 0);
        output.push(
          kind === 0
            ? sum
            : kind === 1
              ? Math.round(sum / values.length)
              : kind === 2
                ? Math.max(...values)
                : Math.min(...values),
        );
      }
    }
    destination.copyFromBytes(Uint8Array.from(output));
  }

  matRandn(destination: WasmMatHandle, mean: Float64Array, standardDeviation: Float64Array): void {
    const meanValue = mean[0] ?? 0;
    const deviation = standardDeviation[0] ?? 0;
    if (destination.depth === 6 && deviation === 0) {
      destination.copyFromBytes(
        copyViewBytes(new Float64Array(destination.rows * destination.columns).fill(meanValue)),
      );
      return;
    }
    throw new OpenCvInputError("fake backend only implements constant F64 normal fills");
  }

  matRandu(destination: WasmMatHandle, lower: Float64Array, upper: Float64Array): void {
    const low = lower[0] ?? 0;
    const high = upper[0] ?? 0;
    destination.copyFromBytes(
      Uint8Array.from({ length: destination.byteLength }, () => {
        this.#randomState = (Math.imul(this.#randomState, 1_664_525) + 1_013_904_223) | 0;
        const unit = (this.#randomState >>> 0) / 4_294_967_296;
        return Math.floor(low + (high - low) * unit);
      }),
    );
  }

  matSetIdentity(destination: WasmMatHandle, value: Float64Array): void {
    const output = new Uint8Array(destination.byteLength);
    const diagonal = Math.min(destination.rows, destination.columns);
    for (let position = 0; position < diagonal; position += 1) {
      output[(position * destination.columns + position) * destination.channels] = Math.round(
        value[0] ?? 0,
      );
    }
    destination.copyFromBytes(output);
  }

  matHconcat2(first: WasmMatHandle, second: WasmMatHandle): WasmMatHandle {
    return concatHandles([first, second], "horizontal");
  }

  matHconcat3(first: WasmMatHandle, second: WasmMatHandle, third: WasmMatHandle): WasmMatHandle {
    return concatHandles([first, second, third], "horizontal");
  }

  matHconcat4(
    first: WasmMatHandle,
    second: WasmMatHandle,
    third: WasmMatHandle,
    fourth: WasmMatHandle,
  ): WasmMatHandle {
    return concatHandles([first, second, third, fourth], "horizontal");
  }

  matVconcat2(first: WasmMatHandle, second: WasmMatHandle): WasmMatHandle {
    return concatHandles([first, second], "vertical");
  }

  matVconcat3(first: WasmMatHandle, second: WasmMatHandle, third: WasmMatHandle): WasmMatHandle {
    return concatHandles([first, second, third], "vertical");
  }

  matVconcat4(
    first: WasmMatHandle,
    second: WasmMatHandle,
    third: WasmMatHandle,
    fourth: WasmMatHandle,
  ): WasmMatHandle {
    return concatHandles([first, second, third, fourth], "vertical");
  }

  matExp(source: WasmMatHandle): WasmMatHandle {
    return mapFloatHandle(source, Math.exp);
  }

  matExpInto(source: WasmMatHandle, destination: WasmMatHandle): void {
    destination.copyFromBytes(this.matExp(source).toUint8Array());
  }

  matLog(source: WasmMatHandle): WasmMatHandle {
    return mapFloatHandle(source, Math.log);
  }

  matLogInto(source: WasmMatHandle, destination: WasmMatHandle): void {
    destination.copyFromBytes(this.matLog(source).toUint8Array());
  }

  matSqrt(source: WasmMatHandle): WasmMatHandle {
    return mapFloatHandle(source, Math.sqrt);
  }

  matSqrtInto(source: WasmMatHandle, destination: WasmMatHandle): void {
    destination.copyFromBytes(this.matSqrt(source).toUint8Array());
  }

  matPow(source: WasmMatHandle, exponent: number): WasmMatHandle {
    return mapFloatHandle(source, (value) => value ** exponent);
  }

  matPowInto(source: WasmMatHandle, exponent: number, destination: WasmMatHandle): void {
    destination.copyFromBytes(this.matPow(source, exponent).toUint8Array());
  }

  matMagnitude(x: WasmMatHandle, y: WasmMatHandle): WasmMatHandle {
    return zipFloatHandles(x, y, Math.hypot);
  }

  matMagnitudeInto(x: WasmMatHandle, y: WasmMatHandle, destination: WasmMatHandle): void {
    destination.copyFromBytes(this.matMagnitude(x, y).toUint8Array());
  }

  matCartToPolar(
    x: WasmMatHandle,
    y: WasmMatHandle,
    magnitude: WasmMatHandle,
    angle: WasmMatHandle,
    degrees: boolean,
  ): void {
    this.cartToPolarDegreeFlags.push(degrees);
    magnitude.copyFromBytes(this.matMagnitude(x, y).toUint8Array());
    const scale = degrees ? 180 / Math.PI : 1;
    angle.copyFromBytes(
      zipFloatHandles(x, y, (xValue, yValue) => Math.atan2(yValue, xValue) * scale).toUint8Array(),
    );
  }

  matPolarToCart(
    magnitude: WasmMatHandle,
    angle: WasmMatHandle,
    x: WasmMatHandle,
    y: WasmMatHandle,
    degrees: boolean,
  ): void {
    this.polarToCartDegreeFlags.push(degrees);
    const scale = degrees ? Math.PI / 180 : 1;
    x.copyFromBytes(
      zipFloatHandles(
        magnitude,
        angle,
        (length, direction) => length * Math.cos(direction * scale),
      ).toUint8Array(),
    );
    y.copyFromBytes(
      zipFloatHandles(
        magnitude,
        angle,
        (length, direction) => length * Math.sin(direction * scale),
      ).toUint8Array(),
    );
  }

  matMultiply(a: WasmMatHandle, b: WasmMatHandle, scale: number): WasmMatHandle {
    return binaryNumericU8(a, b, (left, right) => left * right * scale);
  }

  matMultiplyInto(
    a: WasmMatHandle,
    b: WasmMatHandle,
    destination: WasmMatHandle,
    scale: number,
    dtype: number,
  ): void {
    this.numericIntoCalls.push({ method: "multiply", scale, dtype });
    destination.copyFromBytes(this.matMultiply(a, b, scale).toUint8Array());
  }

  matDivide(a: WasmMatHandle, b: WasmMatHandle, scale: number): WasmMatHandle {
    return binaryNumericU8(a, b, (left, right) => (right === 0 ? 0 : (left * scale) / right));
  }

  matDivideInto(
    a: WasmMatHandle,
    b: WasmMatHandle,
    destination: WasmMatHandle,
    scale: number,
    dtype: number,
  ): void {
    this.numericIntoCalls.push({ method: "divide", scale, dtype });
    destination.copyFromBytes(this.matDivide(a, b, scale).toUint8Array());
  }

  matAddWeighted(
    a: WasmMatHandle,
    alpha: number,
    b: WasmMatHandle,
    beta: number,
    gamma: number,
  ): WasmMatHandle {
    return binaryNumericU8(a, b, (left, right) => left * alpha + right * beta + gamma);
  }

  matAddWeightedInto(
    a: WasmMatHandle,
    alpha: number,
    b: WasmMatHandle,
    beta: number,
    gamma: number,
    destination: WasmMatHandle,
    dtype: number,
  ): void {
    this.numericIntoCalls.push({ method: "addWeighted", scale: 1, dtype });
    destination.copyFromBytes(this.matAddWeighted(a, alpha, b, beta, gamma).toUint8Array());
  }

  matConvertScaleAbs(source: WasmMatHandle, alpha: number, beta: number): WasmMatHandle {
    const output = Uint8Array.from(source.toUint8Array(), (value) =>
      Math.min(255, Math.max(0, Math.round(Math.abs(value * alpha + beta)))),
    );
    return new CopyingMatHandle(source.rows, source.columns, source.channels, output);
  }

  matConvertScaleAbsInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    alpha: number,
    beta: number,
  ): void {
    destination.copyFromBytes(this.matConvertScaleAbs(source, alpha, beta).toUint8Array());
  }

  matCopyMakeBorder(
    source: WasmMatHandle,
    top: number,
    bottom: number,
    left: number,
    right: number,
    borderType: number,
    constant: Float64Array,
  ): WasmMatHandle {
    if (borderType !== 0 && borderType !== 16) {
      throw new OpenCvInputError("fake backend only implements constant borders");
    }
    const rows = source.rows + top + bottom;
    const columns = source.columns + left + right;
    const output = new Uint8Array(rows * columns * source.channels);
    for (let index = 0; index < output.length; index += 1) {
      output[index] = Math.round(constant[index % source.channels] ?? 0);
    }
    const input = source.toUint8Array();
    for (let row = 0; row < source.rows; row += 1) {
      const target = ((row + top) * columns + left) * source.channels;
      const start = row * source.columns * source.channels;
      output.set(input.subarray(start, start + source.columns * source.channels), target);
    }
    return new CopyingMatHandle(rows, columns, source.channels, output, source.depth);
  }

  matCopyMakeBorderInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    top: number,
    bottom: number,
    left: number,
    right: number,
    borderType: number,
    constant: Float64Array,
  ): void {
    destination.copyFromBytes(
      this.matCopyMakeBorder(source, top, bottom, left, right, borderType, constant).toUint8Array(),
    );
  }

  matAbsdiffU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle {
    return binaryU8(left, right, (leftValue, rightValue) => Math.abs(leftValue - rightValue));
  }

  matAddU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle {
    return binaryU8(left, right, (leftValue, rightValue) => Math.min(leftValue + rightValue, 255));
  }

  matBitwiseAndU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle {
    return binaryU8(left, right, (leftValue, rightValue) => leftValue & rightValue);
  }

  matBitwiseNotU8(source: WasmMatHandle): WasmMatHandle {
    return unaryU8(source, (value) => ~value & 255);
  }

  matBitwiseNot(source: WasmMatHandle): WasmMatHandle {
    return new CopyingMatHandle(
      source.rows,
      source.columns,
      source.channels,
      source.toUint8Array().map((value) => ~value & 255),
      source.depth,
      source.isContinuous,
    );
  }

  matBitwiseNotInto(source: WasmMatHandle, destination: WasmMatHandle): void {
    this.#bitwiseNotInto(source, destination);
  }

  matBitwiseNotMaskedInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    mask: WasmMatHandle,
  ): void {
    this.#bitwiseNotInto(source, destination, mask);
  }

  matBitwiseOrU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle {
    return binaryU8(left, right, (leftValue, rightValue) => leftValue | rightValue);
  }

  matBitwiseXorU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle {
    return binaryU8(left, right, (leftValue, rightValue) => leftValue ^ rightValue);
  }

  #bitwiseNotInto(source: WasmMatHandle, destination: WasmMatHandle, mask?: WasmMatHandle): void {
    const sourceBytes = source.toUint8Array();
    const compatible =
      destination.rows === source.rows &&
      destination.columns === source.columns &&
      destination.channels === source.channels &&
      destination.depth === source.depth;
    const output = compatible ? destination.toUint8Array() : new Uint8Array(sourceBytes.length);
    const maskBytes = mask?.byteLength === 0 ? undefined : mask?.toUint8Array();
    const pixelBytes = source.channels * depthByteWidth(source.depth);
    for (let pixel = 0; pixel < source.rows * source.columns; pixel += 1) {
      if (maskBytes !== undefined && byteAt(maskBytes, pixel) === 0) continue;
      const first = pixel * pixelBytes;
      for (let offset = 0; offset < pixelBytes; offset += 1) {
        output[first + offset] = ~byteAt(sourceBytes, first + offset) & 255;
      }
    }
    if (destination instanceof CopyingMatHandle) {
      destination.replaceFrom(source, output);
      return;
    }
    destination.copyFromBytes(output);
  }

  matCompareEqU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle {
    return binaryU8(left, right, (leftValue, rightValue) => (leftValue === rightValue ? 255 : 0));
  }

  matCountNonZero(source: WasmMatHandle): number {
    return source.toUint8Array().reduce((count, value) => count + Number(value !== 0), 0);
  }

  matArcLength(contour: WasmMatHandle, closed: boolean): number {
    const points = contourPoints(contour);
    const pairCount = closed ? points.length : Math.max(points.length - 1, 0);
    let total = 0;
    for (let index = 0; index < pairCount; index += 1) {
      const start = requiredPoint(points, index);
      const end = requiredPoint(points, (index + 1) % points.length);
      total += Math.hypot(end.x - start.x, end.y - start.y);
    }
    return total;
  }

  matBoundingRect(contour: WasmMatHandle): Int32Array {
    const points = contourPoints(contour);
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    const minimumX = Math.floor(Math.min(...xs));
    const minimumY = Math.floor(Math.min(...ys));
    const maximumX = Math.floor(Math.max(...xs));
    const maximumY = Math.floor(Math.max(...ys));
    return new Int32Array([minimumX, minimumY, maximumX - minimumX + 1, maximumY - minimumY + 1]);
  }

  matContourArea(contour: WasmMatHandle, oriented: boolean): number {
    const points = contourPoints(contour);
    let twiceArea = 0;
    for (let index = 0; index < points.length; index += 1) {
      const current = requiredPoint(points, index);
      const next = requiredPoint(points, (index + 1) % points.length);
      twiceArea += current.x * next.y - next.x * current.y;
    }
    const area = twiceArea / 2;
    return oriented ? area : Math.abs(area);
  }

  matDeterminant(source: WasmMatHandle): number {
    const values = source.toFloat64Array();
    return (values[0] ?? 0) * (values[3] ?? 0) - (values[1] ?? 0) * (values[2] ?? 0);
  }

  matInRangeU8(
    source: WasmMatHandle,
    lowerBound: WasmMatHandle,
    upperBound: WasmMatHandle,
  ): WasmMatHandle {
    const values = source.toUint8Array();
    const lower = lowerBound.toUint8Array();
    const upper = upperBound.toUint8Array();
    const output = new Uint8Array(source.rows * source.columns);
    for (let pixel = 0; pixel < output.length; pixel += 1) {
      let inside = true;
      for (let channel = 0; channel < source.channels; channel += 1) {
        const index = pixel * source.channels + channel;
        const value = byteAt(values, index);
        inside &&= value >= byteAt(lower, index) && value <= byteAt(upper, index);
      }
      output[pixel] = inside ? 255 : 0;
    }
    return new CopyingMatHandle(source.rows, source.columns, 1, output);
  }

  matGetAffineTransform(source: WasmMatHandle, destination: WasmMatHandle): WasmMatHandle {
    const from = contourPoints(source);
    const to = contourPoints(destination);
    const origin = requiredPoint(to, 0);
    const sourceX = requiredPoint(from, 1).x - requiredPoint(from, 0).x;
    const sourceY = requiredPoint(from, 2).y - requiredPoint(from, 0).y;
    const scaleX = (requiredPoint(to, 1).x - origin.x) / sourceX;
    const scaleY = (requiredPoint(to, 2).y - origin.y) / sourceY;
    return f64Handle(2, 3, [scaleX, 0, origin.x, 0, scaleY, origin.y]);
  }

  matGetPerspectiveTransform(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    _method: number,
  ): WasmMatHandle {
    const affine = this.matGetAffineTransform(source, destination).toFloat64Array();
    return f64Handle(3, 3, [
      affine[0] ?? 0,
      affine[1] ?? 0,
      affine[2] ?? 0,
      affine[3] ?? 0,
      affine[4] ?? 0,
      affine[5] ?? 0,
      0,
      0,
      1,
    ]);
  }

  matGetRotationMatrix2D(
    centerX: number,
    centerY: number,
    angleDegrees: number,
    scale: number,
  ): WasmMatHandle {
    this.rotationMatrixInputs.push({ centerX, centerY, angleDegrees, scale });
    const radians = (angleDegrees * Math.PI) / 180;
    const alpha = scale * Math.cos(radians);
    const beta = scale * Math.sin(radians);
    return f64Handle(2, 3, [
      alpha,
      beta,
      (1 - alpha) * centerX - beta * centerY,
      -beta,
      alpha,
      beta * centerX + (1 - alpha) * centerY,
    ]);
  }

  matInvertAffineTransform(transform: WasmMatHandle): WasmMatHandle {
    const values = transform.depth === 5 ? transform.toFloat32Array() : transform.toFloat64Array();
    const a = values[0] ?? 0;
    const b = values[1] ?? 0;
    const c = values[2] ?? 0;
    const d = values[3] ?? 0;
    const e = values[4] ?? 0;
    const f = values[5] ?? 0;
    const determinant = a * e - b * d;
    const inverseA = e / determinant;
    const inverseB = -b / determinant;
    const inverseD = -d / determinant;
    const inverseE = a / determinant;
    const output = [
      inverseA,
      inverseB,
      -(inverseA * c + inverseB * f),
      inverseD,
      inverseE,
      -(inverseD * c + inverseE * f),
    ].map(cleanTiny);
    return transform.depth === 5
      ? new CopyingMatHandle(2, 3, 1, copyViewBytes(new Float32Array(output)), 5)
      : f64Handle(2, 3, output);
  }

  matInvertAffineTransformInto(transform: WasmMatHandle, destination: WasmMatHandle): void {
    const output = this.matInvertAffineTransform(transform);
    destination.copyFromBytes(output.toUint8Array());
  }

  matInvertInto(source: WasmMatHandle, destination: WasmMatHandle, _method: number): number {
    const values = source.toFloat64Array();
    const determinant = this.matDeterminant(source);
    if (determinant === 0) return 0;
    destination.copyFromBytes(
      copyViewBytes(
        new Float64Array([
          (values[3] ?? 0) / determinant,
          -(values[1] ?? 0) / determinant,
          -(values[2] ?? 0) / determinant,
          (values[0] ?? 0) / determinant,
        ]),
      ),
    );
    return 1;
  }

  matIsContourConvex(contour: WasmMatHandle): boolean {
    const points = contourPoints(contour);
    let direction = 0;
    for (let index = 0; index < points.length; index += 1) {
      const a = requiredPoint(points, index);
      const b = requiredPoint(points, (index + 1) % points.length);
      const c = requiredPoint(points, (index + 2) % points.length);
      const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
      if (cross === 0) continue;
      const sign = Math.sign(cross);
      if (direction !== 0 && sign !== direction) return false;
      direction = sign;
    }
    return direction !== 0;
  }

  matPointPolygonTest(
    contour: WasmMatHandle,
    x: number,
    y: number,
    measureDistance: boolean,
  ): number {
    this.pointPolygonTestInputs.push({ x, y, measureDistance });
    const points = contourPoints(contour);
    let inside = false;
    let nearest = Number.POSITIVE_INFINITY;
    for (let index = 0; index < points.length; index += 1) {
      const start = requiredPoint(points, index);
      const end = requiredPoint(points, (index + 1) % points.length);
      nearest = Math.min(nearest, pointSegmentDistance(x, y, start, end));
      if (start.y > y !== end.y > y) {
        const crossingX = ((end.x - start.x) * (y - start.y)) / (end.y - start.y) + start.x;
        if (x < crossingX) inside = !inside;
      }
    }
    if (nearest === 0) return 0;
    if (!measureDistance) return inside ? 1 : -1;
    return inside ? nearest : -nearest;
  }

  matSolveInto(
    coefficients: WasmMatHandle,
    rightHandSides: WasmMatHandle,
    destination: WasmMatHandle,
    method: number,
  ): boolean {
    const inverse = new CopyingMatHandle(2, 2, 1, new Uint8Array(32), 6);
    if (this.matInvertInto(coefficients, inverse, method) === 0) return false;
    const inverseValues = inverse.toFloat64Array();
    const right = rightHandSides.toFloat64Array();
    destination.copyFromBytes(
      copyViewBytes(
        new Float64Array([
          (inverseValues[0] ?? 0) * (right[0] ?? 0) + (inverseValues[1] ?? 0) * (right[1] ?? 0),
          (inverseValues[2] ?? 0) * (right[0] ?? 0) + (inverseValues[3] ?? 0) * (right[1] ?? 0),
        ]),
      ),
    );
    return true;
  }

  matMaxU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle {
    return binaryU8(left, right, Math.max);
  }

  matMean(source: WasmMatHandle): Float64Array {
    const totals = this.matSum(source);
    const output = new Float64Array(4);
    const pixels = source.rows * source.columns;
    if (pixels === 0) return output;
    for (let channel = 0; channel < source.channels; channel += 1) {
      output[channel] = floatAt(totals, channel) * (1 / pixels);
    }
    return output;
  }

  matMeanMasked(source: WasmMatHandle, mask: WasmMatHandle): Float64Array {
    if (mask.byteLength === 0) return this.matMean(source);
    const output = new Float64Array(4);
    const data = source.toUint8Array();
    const maskData = mask.toUint8Array();
    let selected = 0;
    for (let pixel = 0; pixel < source.rows * source.columns; pixel += 1) {
      if (byteAt(maskData, pixel) === 0) continue;
      selected += 1;
      for (let channel = 0; channel < source.channels; channel += 1) {
        const index = pixel * source.channels + channel;
        output[channel] = floatAt(output, channel) + byteAt(data, index);
      }
    }
    if (selected !== 0) {
      for (let channel = 0; channel < source.channels; channel += 1) {
        output[channel] = floatAt(output, channel) * (1 / selected);
      }
    }
    return output;
  }

  matMinU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle {
    return binaryU8(left, right, Math.min);
  }

  matMinMaxLoc(source: WasmMatHandle): Float64Array {
    const data = source.toUint8Array();
    if (source.rows === 0 || source.columns === 0) {
      const coordinate = source.isContinuous ? -1 : 0;
      return new Float64Array([0, 0, coordinate, coordinate, coordinate, coordinate]);
    }
    let minimum = byteAt(data, 0);
    let maximum = minimum;
    let minimumIndex = 0;
    let maximumIndex = 0;
    for (let index = 1; index < data.length; index += 1) {
      const value = byteAt(data, index);
      if (value < minimum) {
        minimum = value;
        minimumIndex = index;
      }
      if (value > maximum) {
        maximum = value;
        maximumIndex = index;
      }
    }
    return new Float64Array([
      minimum,
      maximum,
      minimumIndex % source.columns,
      Math.floor(minimumIndex / source.columns),
      maximumIndex % source.columns,
      Math.floor(maximumIndex / source.columns),
    ]);
  }

  matMinMaxLocMasked(source: WasmMatHandle, mask: WasmMatHandle): Float64Array {
    if (mask.byteLength === 0) return this.matMinMaxLoc(source);
    const data = source.toUint8Array();
    const maskData = mask.toUint8Array();
    let minimum = 0;
    let maximum = 0;
    let minimumIndex = -1;
    let maximumIndex = -1;
    for (let index = 0; index < source.rows * source.columns; index += 1) {
      if (byteAt(maskData, index) === 0) continue;
      const value = byteAt(data, index);
      if (minimumIndex === -1) {
        minimum = value;
        maximum = value;
        minimumIndex = index;
        maximumIndex = index;
        continue;
      }
      if (value < minimum) {
        minimum = value;
        minimumIndex = index;
      }
      if (value > maximum) {
        maximum = value;
        maximumIndex = index;
      }
    }
    return new Float64Array([
      minimum,
      maximum,
      minimumIndex === -1 ? -1 : minimumIndex % source.columns,
      minimumIndex === -1 ? -1 : Math.floor(minimumIndex / source.columns),
      maximumIndex === -1 ? -1 : maximumIndex % source.columns,
      maximumIndex === -1 ? -1 : Math.floor(maximumIndex / source.columns),
    ]);
  }

  matSubtractU8(left: WasmMatHandle, right: WasmMatHandle): WasmMatHandle {
    return binaryU8(left, right, (leftValue, rightValue) => Math.max(leftValue - rightValue, 0));
  }

  matSum(source: WasmMatHandle): Float64Array {
    const output = new Float64Array(4);
    const data = source.toUint8Array();
    for (let index = 0; index < data.length; index += 1) {
      const channel = index % source.channels;
      output[channel] = floatAt(output, channel) + byteAt(data, index);
    }
    return output;
  }

  matTranspose(source: WasmMatHandle): WasmMatHandle {
    const input = source.toUint8Array();
    const output = new Uint8Array(input.byteLength);
    const pixelBytes = source.channels * depthByteWidth(source.depth);
    for (let row = 0; row < source.columns; row += 1) {
      for (let column = 0; column < source.rows; column += 1) {
        const sourceOffset = (column * source.columns + row) * pixelBytes;
        const outputOffset = (row * source.rows + column) * pixelBytes;
        output.set(input.subarray(sourceOffset, sourceOffset + pixelBytes), outputOffset);
      }
    }
    return new CopyingMatHandle(source.columns, source.rows, source.channels, output, source.depth);
  }

  matTransposeInto(source: WasmMatHandle, destination: WasmMatHandle): void {
    destination.copyFromBytes(this.matTranspose(source).toUint8Array());
  }

  matZerosU8(rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(rows, columns, channels, new Uint8Array(rows * columns * channels));
  }

  matZerosF32(rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(
      rows,
      columns,
      channels,
      new Uint8Array(rows * columns * channels * Float32Array.BYTES_PER_ELEMENT),
      5,
    );
  }

  matTrace(source: WasmMatHandle): Float64Array {
    const data: ArrayLike<number> = (() => {
      switch (source.depth) {
        case 0:
          return source.toUint8Array();
        case 1:
          return source.toInt8Array();
        case 2:
          return source.toUint16Array();
        case 3:
          return source.toInt16Array();
        case 4:
          return source.toInt32Array();
        case 5:
          return source.toFloat32Array();
        case 6:
          return source.toFloat64Array();
        default:
          throw new OpenCvInputError("unsupported depth");
      }
    })();
    const diagonal = Math.min(source.rows, source.columns);
    const output = new Float64Array(4);
    for (let position = 0; position < diagonal; position += 1) {
      const first = (position * source.columns + position) * source.channels;
      for (let channel = 0; channel < source.channels; channel += 1) {
        output[channel] = floatAt(output, channel) + requiredNumber(data, first + channel);
      }
    }
    return output;
  }

  matTransform(source: WasmMatHandle, coefficients: WasmMatHandle): WasmMatHandle {
    const input = source.toUint8Array();
    const weights = coefficients.toFloat64Array();
    const output = new Uint8Array(source.rows * source.columns * coefficients.rows);
    for (let pixel = 0; pixel < source.rows * source.columns; pixel += 1) {
      for (let outputChannel = 0; outputChannel < coefficients.rows; outputChannel += 1) {
        let value = 0;
        for (let inputChannel = 0; inputChannel < source.channels; inputChannel += 1) {
          value +=
            (input[pixel * source.channels + inputChannel] ?? 0) *
            (weights[outputChannel * coefficients.columns + inputChannel] ?? 0);
        }
        if (coefficients.columns === source.channels + 1) {
          value += weights[outputChannel * coefficients.columns + source.channels] ?? 0;
        }
        output[pixel * coefficients.rows + outputChannel] = Math.round(value);
      }
    }
    return new CopyingMatHandle(source.rows, source.columns, coefficients.rows, output);
  }

  matTransformInto(
    source: WasmMatHandle,
    coefficients: WasmMatHandle,
    destination: WasmMatHandle,
  ): void {
    destination.copyFromBytes(this.matTransform(source, coefficients).toUint8Array());
  }

  matPerspectiveTransform(source: WasmMatHandle, _coefficients: WasmMatHandle): WasmMatHandle {
    return new CopyingMatHandle(
      source.rows,
      source.columns,
      source.channels,
      source.toUint8Array(),
      source.depth,
    );
  }

  matPerspectiveTransformInto(
    source: WasmMatHandle,
    coefficients: WasmMatHandle,
    destination: WasmMatHandle,
  ): void {
    destination.copyFromBytes(this.matPerspectiveTransform(source, coefficients).toUint8Array());
  }

  matRotate(source: WasmMatHandle, rotateCode: number): WasmMatHandle {
    if (rotateCode === 1) {
      return this.matFlip(source, -1);
    }
    const transposed = this.matTranspose(source);
    return this.matFlip(transposed, rotateCode === 0 ? 1 : 0);
  }

  matRotateInto(source: WasmMatHandle, destination: WasmMatHandle, rotateCode: number): void {
    destination.copyFromBytes(this.matRotate(source, rotateCode).toUint8Array());
  }

  matRepeat(source: WasmMatHandle, rowRepeats: number, columnRepeats: number): WasmMatHandle {
    const input = source.toUint8Array();
    const rows = source.rows * rowRepeats;
    const columns = source.columns * columnRepeats;
    const pixelBytes = source.channels * depthByteWidth(source.depth);
    const output = new Uint8Array(rows * columns * pixelBytes);
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const sourceOffset =
          ((row % source.rows) * source.columns + (column % source.columns)) * pixelBytes;
        const outputOffset = (row * columns + column) * pixelBytes;
        output.set(input.subarray(sourceOffset, sourceOffset + pixelBytes), outputOffset);
      }
    }
    return new CopyingMatHandle(rows, columns, source.channels, output, source.depth);
  }

  matRepeatInto(
    source: WasmMatHandle,
    destination: WasmMatHandle,
    rowRepeats: number,
    columnRepeats: number,
  ): void {
    destination.copyFromBytes(this.matRepeat(source, rowRepeats, columnRepeats).toUint8Array());
  }

  matZerosF64(rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(
      rows,
      columns,
      channels,
      new Uint8Array(rows * columns * channels * Float64Array.BYTES_PER_ELEMENT),
      6,
    );
  }

  matZerosI16(rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(
      rows,
      columns,
      channels,
      new Uint8Array(rows * columns * channels * Int16Array.BYTES_PER_ELEMENT),
      3,
    );
  }

  matZerosI32(rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(
      rows,
      columns,
      channels,
      new Uint8Array(rows * columns * channels * Int32Array.BYTES_PER_ELEMENT),
      4,
    );
  }

  matZerosI8(rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(
      rows,
      columns,
      channels,
      new Uint8Array(rows * columns * channels),
      1,
    );
  }

  matZerosU16(rows: number, columns: number, channels: number): WasmMatHandle {
    return new CopyingMatHandle(
      rows,
      columns,
      channels,
      new Uint8Array(rows * columns * channels * Uint16Array.BYTES_PER_ELEMENT),
      2,
    );
  }

  setRNGSeed(seed: number): void {
    this.#randomState = seed;
  }

  setLogLevel(level: number): number {
    const previous = this.#logLevel;
    this.#logLevel = level;
    return previous;
  }

  resizeNearestRgba(
    _data: Uint8Array,
    _width: number,
    _height: number,
    targetWidth: number,
    targetHeight: number,
  ): Uint8Array {
    return new Uint8Array(targetWidth * targetHeight * 4);
  }

  thresholdRgba(data: Uint8Array): Uint8Array {
    return new Uint8Array(data);
  }
}

function binaryU8(
  left: WasmMatHandle,
  right: WasmMatHandle,
  operation: (leftValue: number, rightValue: number) => number,
): WasmMatHandle {
  const leftData = left.toUint8Array();
  const rightData = right.toUint8Array();
  const output = leftData.map((value, index) => operation(value, byteAt(rightData, index)));
  return new CopyingMatHandle(left.rows, left.columns, left.channels, output);
}

function mockOtsuThreshold(input: Uint8Array): number {
  const histogram = new Uint32Array(256);
  for (const value of input) histogram[value] = (histogram[value] ?? 0) + 1;
  const totalSum = histogram.reduce((sum, count, value) => sum + value * count, 0);
  let backgroundCount = 0;
  let backgroundSum = 0;
  let bestVariance = -1;
  let bestThreshold = 0;
  for (let threshold = 0; threshold < histogram.length; threshold += 1) {
    const count = histogram[threshold] ?? 0;
    backgroundCount += count;
    if (backgroundCount === 0) continue;
    const foregroundCount = input.length - backgroundCount;
    if (foregroundCount === 0) break;
    backgroundSum += threshold * count;
    const difference =
      backgroundSum / backgroundCount - (totalSum - backgroundSum) / foregroundCount;
    const variance = backgroundCount * foregroundCount * difference * difference;
    if (variance > bestVariance) {
      bestVariance = variance;
      bestThreshold = threshold;
    }
  }
  return bestThreshold;
}

function mockBorderIndex(index: number, length: number, borderType: number): number | undefined {
  if (index >= 0 && index < length) return index;
  switch (borderType & ~16) {
    case 0:
      return undefined;
    case 1:
      return Math.max(0, Math.min(length - 1, index));
    case 4: {
      if (length === 1) return 0;
      const period = 2 * length - 2;
      const position = ((index % period) + period) % period;
      return position < length ? position : period - position;
    }
    default:
      throw new OpenCvInputError(`unsupported mock border type ${borderType}`);
  }
}

function replaceMockDestination(
  destination: WasmMatHandle,
  rows: number,
  columns: number,
  channels: number,
  output: Uint8Array,
  depth: number,
): void {
  if (!(destination instanceof CopyingMatHandle)) {
    throw new OpenCvInputError("mock destination must use CopyingMatHandle");
  }
  destination.replaceFrom(new CopyingMatHandle(rows, columns, channels, output, depth), output);
}

function f64Handle(rows: number, columns: number, values: readonly number[]): WasmMatHandle {
  return new CopyingMatHandle(rows, columns, 1, copyViewBytes(new Float64Array(values)), 6);
}

function contourPoints(source: WasmMatHandle): Array<{ readonly x: number; readonly y: number }> {
  const values =
    source.depth === 4
      ? source.toInt32Array()
      : source.depth === 5
        ? source.toFloat32Array()
        : source.toFloat64Array();
  const points: Array<{ readonly x: number; readonly y: number }> = [];
  for (let index = 0; index < values.length; index += 2) {
    points.push({ x: requiredNumber(values, index), y: requiredNumber(values, index + 1) });
  }
  return points;
}

function requiredPoint(
  points: ReadonlyArray<{ readonly x: number; readonly y: number }>,
  index: number,
): { readonly x: number; readonly y: number } {
  const point = points[index];
  if (point === undefined) throw new RangeError(`missing point at index ${index}`);
  return point;
}

function requiredNumber(values: ArrayLike<number>, index: number): number {
  const value = values[index];
  if (value === undefined) throw new RangeError(`missing number at index ${index}`);
  return value;
}

function pointSegmentDistance(
  x: number,
  y: number,
  start: { readonly x: number; readonly y: number },
  end: { readonly x: number; readonly y: number },
): number {
  const deltaX = end.x - start.x;
  const deltaY = end.y - start.y;
  const lengthSquared = deltaX * deltaX + deltaY * deltaY;
  const ratio =
    lengthSquared === 0
      ? 0
      : Math.max(0, Math.min(1, ((x - start.x) * deltaX + (y - start.y) * deltaY) / lengthSquared));
  return Math.hypot(x - (start.x + ratio * deltaX), y - (start.y + ratio * deltaY));
}

function cleanTiny(value: number): number {
  return Math.abs(value) < Number.EPSILON ? 0 : value;
}

function byteAt(data: Uint8Array, index: number): number {
  const value = data[index];
  if (value === undefined) {
    throw new RangeError(`missing byte at index ${index}`);
  }
  return value;
}

function floatAt(data: Float64Array, index: number): number {
  const value = data[index];
  if (value === undefined) {
    throw new RangeError(`missing float at index ${index}`);
  }
  return value;
}

function unaryU8(source: WasmMatHandle, operation: (value: number) => number): WasmMatHandle {
  return new CopyingMatHandle(
    source.rows,
    source.columns,
    source.channels,
    source.toUint8Array().map(operation),
  );
}

function roundNearestEven(value: number): number {
  const lower = Math.floor(value);
  return value - lower === 0.5 ? lower + (lower % 2) : Math.round(value);
}

describe("createRgbaImage", () => {
  test("copies caller-owned data", () => {
    const input = new Uint8Array([1, 2, 3, 4]);
    const image = createRgbaImage(1, 1, input);
    input[0] = 99;
    expect(image.data).toEqual(new Uint8Array([1, 2, 3, 4]));
  });

  test("rejects a mismatched RGBA buffer", () => {
    expect(() => createRgbaImage(2, 1, new Uint8Array(4))).toThrow(OpenCvInputError);
  });
});

describe("OpenCv client", () => {
  test("template matching exposes six constants and dispatches both overloads", () => {
    const backend = new CopyingBackend();
    const cv = createOpenCv(backend);
    const image = cv.matFromU8(1, 3, 1, new Uint8Array([1, 2, 3]));
    const template = cv.matFromU8(1, 2, 1, new Uint8Array([1, 2]));
    const result = cv.emptyMat();
    const mask = cv.emptyMat();
    expect([
      cv.TM_SQDIFF,
      cv.TM_SQDIFF_NORMED,
      cv.TM_CCORR,
      cv.TM_CCORR_NORMED,
      cv.TM_CCOEFF,
      cv.TM_CCOEFF_NORMED,
    ]).toEqual([
      TM_SQDIFF,
      TM_SQDIFF_NORMED,
      TM_CCORR,
      TM_CCORR_NORMED,
      TM_CCOEFF,
      TM_CCOEFF_NORMED,
    ]);
    expect(cv.matchTemplate.length).toBe(0);
    expect(cv.matchTemplate(image, template, result, TM_SQDIFF)).toBeUndefined();
    expect(cv.matchTemplate(image, template, result, TM_CCOEFF_NORMED, mask)).toBeUndefined();
    expect(backend.templateMatchCalls).toEqual([
      {
        image: image.handleForBackend(),
        template: template.handleForBackend(),
        result: result.handleForBackend(),
        method: 0,
        mask: undefined,
      },
      {
        image: image.handleForBackend(),
        template: template.handleForBackend(),
        result: result.handleForBackend(),
        method: 5,
        mask: mask.handleForBackend(),
      },
    ]);
    for (const mat of [image, template, result, mask]) mat.dispose();
  });

  test("template matching checks arity, scalar conversion, and matrix lifetime before dispatch", () => {
    const backend = new CopyingBackend();
    const cv = createOpenCv(backend);
    const image = cv.emptyMat();
    const template = cv.emptyMat();
    const result = cv.emptyMat();
    // @ts-expect-error Deliberately exercise the JavaScript caller boundary.
    expect(() => cv.matchTemplate(image, template, result)).toThrow(BindingError);
    // @ts-expect-error Deliberately exercise the JavaScript caller boundary.
    expect(() => cv.matchTemplate(image, template, result, 0, image, image)).toThrow(BindingError);
    // @ts-expect-error Explicit undefined selects the five-argument Mat overload.
    expect(() => cv.matchTemplate(image, template, result, 0, undefined)).toThrow(TypeError);
    // @ts-expect-error Null is not a valid Mat binding.
    expect(() => cv.matchTemplate(image, template, result, 0, null)).toThrow(BindingError);
    // @ts-expect-error Method strings do not use JavaScript numeric coercion.
    expect(() => cv.matchTemplate(image, template, result, "2")).toThrow(TypeError);
    // @ts-expect-error Embind truncates fractional numeric input at runtime.
    cv.matchTemplate(image, template, result, 2.9);
    // @ts-expect-error Embind converts NaN to signed integer zero.
    cv.matchTemplate(image, template, result, Number.NaN);
    expect(backend.templateMatchCalls.map((call) => call.method)).toEqual([2, 0]);
    template.dispose();
    expect(() => cv.matchTemplate(image, template, result, TM_CCORR)).toThrow();
    expect(backend.templateMatchCalls).toHaveLength(2);
    image.dispose();
    result.dispose();
  });
  const client = createOpenCv(new CopyingBackend());
  const image = createRgbaImage(1, 1, new Uint8Array([1, 2, 3, 255]));

  test("cvtColor converts browser RGBA pixels through the OpenCV-compatible Mat API", () => {
    const source = client.matFromU8(1, 2, 4, new Uint8Array([255, 0, 0, 7, 1, 2, 3, 4]));
    const gray = client.emptyMat();
    const reordered = client.emptyMat();

    client.cvtColor(source, gray, COLOR_RGBA2GRAY);
    client.cvtColor(source, reordered, COLOR_RGBA2BGRA);

    expect(gray.rows).toBe(1);
    expect(gray.columns).toBe(2);
    expect(gray.channels).toBe(1);
    expect(gray.toUint8Array()).toEqual(new Uint8Array([76, 2]));
    expect(reordered.channels).toBe(4);
    expect(reordered.toUint8Array()).toEqual(new Uint8Array([0, 0, 255, 7, 3, 2, 1, 4]));

    source.dispose();
    gray.dispose();
    reordered.dispose();
  });

  test("cvtColor honors dstCn for RGB alpha insertion and grayscale expansion", () => {
    const rgb = client.matFromU8(1, 1, 3, new Uint8Array([10, 20, 30]));
    const gray = client.matFromU8(1, 1, 1, new Uint8Array([9]));
    const threeChannels = client.emptyMat();
    const fourChannels = client.emptyMat();

    client.cvtColor(rgb, threeChannels, COLOR_RGB2RGBA, 3);
    client.cvtColor(gray, fourChannels, COLOR_GRAY2RGBA, 4);

    expect(threeChannels.toUint8Array()).toEqual(new Uint8Array([10, 20, 30]));
    expect(fourChannels.toUint8Array()).toEqual(new Uint8Array([9, 9, 9, 255]));

    rgb.dispose();
    gray.dispose();
    threeChannels.dispose();
    fourChannels.dispose();
  });

  test("resize applies nearest-neighbor sampling through a mutable Mat destination", () => {
    const source = client.matFromU8(2, 2, 1, new Uint8Array([1, 2, 3, 4]));
    const destination = client.emptyMat();

    client.resize(source, destination, { width: 4, height: 2 }, 0, 0, INTER_NEAREST);

    expect(destination.rows).toBe(2);
    expect(destination.columns).toBe(4);
    expect(destination.channels).toBe(1);
    expect(destination.toUint8Array()).toEqual(new Uint8Array([1, 1, 2, 2, 3, 3, 4, 4]));

    source.dispose();
    destination.dispose();
  });

  test("resize uses OpenCV half-pixel linear interpolation by default", () => {
    const source = client.matFromU8(2, 2, 1, new Uint8Array([0, 100, 150, 255]));
    const destination = client.emptyMat();

    client.resize(source, destination, { width: 3, height: 3 });

    expect(client.INTER_LINEAR).toBe(INTER_LINEAR);
    expect(destination.toUint8Array()).toEqual(
      new Uint8Array([0, 50, 100, 75, 126, 178, 150, 202, 255]),
    );

    source.dispose();
    destination.dispose();
  });

  test("resize averages covered U8 pixels when shrinking with INTER_AREA", () => {
    const source = client.matFromU8(
      4,
      4,
      1,
      new Uint8Array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]),
    );
    const destination = client.emptyMat();

    client.resize(source, destination, { width: 2, height: 2 }, 0, 0, INTER_AREA);

    expect(destination.toUint8Array()).toEqual(new Uint8Array([25, 45, 105, 125]));

    source.dispose();
    destination.dispose();
  });

  test("threshold writes the strict binary Mat result and returns the used threshold", () => {
    const source = client.matFromU8(1, 5, 1, new Uint8Array([0, 99, 100, 101, 255]));
    const destination = client.emptyMat();

    const used = client.threshold(source, destination, 100, 200, THRESH_BINARY);

    expect(used).toBe(100);
    expect(destination.toUint8Array()).toEqual(new Uint8Array([0, 0, 0, 200, 200]));

    source.dispose();
    destination.dispose();
  });

  test("threshold computes an Otsu split for a single-channel U8 Mat", () => {
    const source = client.matFromU8(1, 6, 1, new Uint8Array([10, 10, 10, 200, 200, 200]));
    const destination = client.emptyMat();

    const used = client.threshold(source, destination, 0, 255, THRESH_BINARY | THRESH_OTSU);

    expect(used).toBe(10);
    expect(destination.toUint8Array()).toEqual(new Uint8Array([0, 0, 0, 255, 255, 255]));

    source.dispose();
    destination.dispose();
  });

  test("GaussianBlur applies a separable U8 kernel into a mutable destination", () => {
    const source = client.matFromU8(1, 5, 1, new Uint8Array([0, 0, 255, 0, 0]));
    const destination = client.emptyMat();

    client.GaussianBlur(source, destination, { width: 3, height: 1 }, 0, 0, BORDER_CONSTANT);

    expect(destination.toUint8Array()).toEqual(new Uint8Array([0, 64, 128, 64, 0]));
    source.dispose();
    destination.dispose();
  });

  test("morphologyEx shares erosion and dilation primitives", () => {
    const source = client.matFromU8(1, 5, 1, new Uint8Array([0, 255, 255, 255, 0]));
    const kernel = client.matFromU8(1, 3, 1, new Uint8Array([1, 1, 1]));
    const eroded = client.emptyMat();
    const dilated = client.emptyMat();

    client.morphologyEx(source, eroded, MORPH_ERODE, kernel);
    client.morphologyEx(source, dilated, MORPH_DILATE, kernel);

    expect(eroded.toUint8Array()).toEqual(new Uint8Array([0, 0, 255, 0, 0]));
    expect(dilated.toUint8Array()).toEqual(new Uint8Array([255, 255, 255, 255, 255]));
    source.dispose();
    kernel.dispose();
    eroded.dispose();
    dilated.dispose();
  });

  test("Sobel emits signed U8 gradients into an I16 destination", () => {
    const source = client.matFromU8(3, 3, 1, new Uint8Array([0, 10, 20, 0, 10, 20, 0, 10, 20]));
    const destination = client.emptyMat();

    client.Sobel(source, destination, 3, 1, 0, 3, 1, 0, BORDER_CONSTANT);

    expect(destination.depth).toBe("i16");
    expect(destination.toInt16Array()).toEqual(
      new Int16Array([30, 60, -30, 40, 80, -40, 30, 60, -30]),
    );
    source.dispose();
    destination.dispose();
  });

  test("Canny traces a one-pixel vertical edge from a U8 image", () => {
    const source = client.matFromU8(
      5,
      5,
      1,
      new Uint8Array([
        0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0,
        255, 255, 255,
      ]),
    );
    const edges = client.emptyMat();

    client.Canny(source, edges, 50, 100);

    expect(edges.toUint8Array()).toEqual(
      new Uint8Array([
        0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0,
      ]),
    );
    source.dispose();
    edges.dispose();
  });

  test("findContours writes simple external contours into a Rust-owned MatVector", () => {
    const source = client.matFromU8(
      5,
      5,
      1,
      new Uint8Array([
        0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0,
      ]),
    );
    const contours = client.createMatVector();
    const hierarchy = client.emptyMat();

    client.findContours(source, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    expect(client.findContours.length).toBe(0);
    expect(contours.size()).toBe(1);
    const contour = contours.get(0);
    expect(contour.toInt32Array()).toEqual(new Int32Array([1, 1, 1, 3, 3, 3, 3, 1]));
    expect(hierarchy.toInt32Array()).toEqual(new Int32Array([-1, -1, -1, -1]));
    contour.dispose();
    contours.dispose();
    source.dispose();
    hierarchy.dispose();
  });

  test("warpAffine inverse-maps a U8 translation with a constant border", () => {
    const source = client.matFromU8(2, 3, 1, new Uint8Array([1, 2, 3, 4, 5, 6]));
    const transform = client.matFromF64(2, 3, 1, new Float64Array([1, 0, 1, 0, 1, 0]));
    const destination = client.emptyMat();

    client.warpAffine(
      source,
      destination,
      transform,
      { width: 3, height: 2 },
      INTER_NEAREST,
      BORDER_CONSTANT,
      [9, 0, 0, 0],
    );

    expect(destination.toUint8Array()).toEqual(new Uint8Array([9, 1, 2, 9, 4, 5]));
    source.dispose();
    transform.dispose();
    destination.dispose();
  });

  test("equalizeHist applies a cumulative U8 histogram transform", () => {
    const source = client.matFromU8(1, 8, 1, new Uint8Array([0, 0, 1, 1, 2, 3, 3, 3]));
    const destination = client.emptyMat();

    client.equalizeHist(source, destination);

    expect(destination.toUint8Array()).toEqual(new Uint8Array([0, 0, 85, 85, 128, 255, 255, 255]));
    source.dispose();
    destination.dispose();
  });

  test("returns validated output", () => {
    expect(client.grayscale(image)).toEqual(image);
    expect(client.invert(image)).toEqual(image);
    expect(client.threshold(image, 127)).toEqual(image);
  });

  test("validates threshold values before calling WASM", () => {
    expect(() => client.threshold(image, 256)).toThrow(OpenCvInputError);
    expect(() => client.threshold(image, 1.5)).toThrow(OpenCvInputError);
  });

  test("uses target dimensions for resized output", () => {
    const resized = client.resizeNearest(image, 2, 3);
    expect(resized.width).toBe(2);
    expect(resized.height).toBe(3);
    expect(resized.data.byteLength).toBe(24);
  });

  test("creates Rust-owned matrix handles and regions", () => {
    const matrix = client.matFromU8(2, 4, 1, new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]));
    const region = matrix.roi(0, 1, 2, 2);

    expect(matrix.depth).toBe("u8");
    expect(region.rows).toBe(2);
    expect(region.columns).toBe(2);
    expect(region.toUint8Array()).toEqual(new Uint8Array([2, 3, 6, 7]));

    matrix.dispose();
    expect(() => matrix.rows).toThrow(
      new BindingError("Cannot pass deleted object as a pointer of type Mat const*"),
    );
    expect(region.toUint8Array()).toEqual(new Uint8Array([2, 3, 6, 7]));
    region.dispose();
  });

  test("creates a canonical empty matrix for OutputArray destinations", () => {
    const matrix = client.emptyMat();

    expect(matrix.rows).toBe(0);
    expect(matrix.columns).toBe(0);
    expect(matrix.channels).toBe(1);
    expect(matrix.depth).toBe("u8");
    expect(matrix.byteLength).toBe(0);
    expect(matrix.rowStride).toBe(0);
    expect(matrix.isContinuous).toBe(false);
    expect(matrix.toUint8Array()).toEqual(new Uint8Array());

    matrix.dispose();
    expect(() => matrix.rows).toThrow(
      new BindingError("Cannot pass deleted object as a pointer of type Mat const*"),
    );
  });

  test("allocates zero-filled matrices", () => {
    const matrix = client.zerosU8(2, 3, 4);
    expect(matrix.byteLength).toBe(24);
    expect(matrix.toUint8Array()).toEqual(new Uint8Array(24));
    matrix.dispose();
  });

  test("exposes factories for every scalar matrix depth", () => {
    expect(client).toHaveProperty("matFromI8");
    expect(client).toHaveProperty("matFromU16");
    expect(client).toHaveProperty("matFromI16");
    expect(client).toHaveProperty("matFromI32");
    expect(client).toHaveProperty("matFromF32");
    expect(client).toHaveProperty("matFromF64");

    const signed = client.matFromI16(1, 3, 1, new Int16Array([-32_768, 7, 32_767]));
    expect(signed.depth).toBe("i16");
    expect(signed.toInt16Array()).toEqual(new Int16Array([-32_768, 7, 32_767]));
    signed.dispose();

    const floating = client.zerosF32(2, 2, 1);
    expect(floating.depth).toBe("f32");
    expect(floating.toFloat32Array()).toEqual(new Float32Array(4));
    floating.dispose();
  });

  test("constructs typed empty matrix headers", () => {
    const zeroRows = client.matFromF32(0, 3, 2, new Float32Array());
    expect([zeroRows.rows, zeroRows.columns, zeroRows.channels, zeroRows.depth]).toEqual([
      0,
      3,
      2,
      "f32",
    ]);
    expect(zeroRows.byteLength).toBe(0);

    const zeroColumns = client.matFromF64(2, 0, 3, new Float64Array());
    expect([
      zeroColumns.rows,
      zeroColumns.columns,
      zeroColumns.channels,
      zeroColumns.depth,
    ]).toEqual([2, 0, 3, "f64"]);
    expect(zeroColumns.byteLength).toBe(0);

    zeroColumns.dispose();
    zeroRows.dispose();
  });

  test("initializes matrices and controls deterministic random fills", () => {
    const identity = client.zerosU8(2, 3, 1);
    client.setIdentity(identity);
    expect(identity.toUint8Array()).toEqual(new Uint8Array([1, 0, 0, 0, 1, 0]));

    const first = client.zerosU8(1, 8, 1);
    const second = client.zerosU8(1, 8, 1);
    client.setRNGSeed(42);
    client.randu(first, [10, 0, 0, 0], [20, 0, 0, 0]);
    client.setRNGSeed(42);
    client.randu(second, [10, 0, 0, 0], [20, 0, 0, 0]);
    expect(first.toUint8Array()).toEqual(second.toUint8Array());
    expect(Array.from(first.toUint8Array()).every((value) => value >= 10 && value < 20)).toBeTrue();

    const normal = client.zerosF64(1, 4, 1);
    client.randn(normal, [3, 0, 0, 0], [0, 0, 0, 0]);
    expect(normal.toFloat64Array()).toEqual(new Float64Array([3, 3, 3, 3]));

    expect(() => client.setRNGSeed(2 ** 31)).toThrow(OpenCvInputError);
    for (const matrix of [normal, second, first, identity]) matrix.dispose();
  });

  test("matches the setIdentity binding and Scalar conversion contract", () => {
    expect(client.setIdentity.bind(client)).toHaveLength(0);

    const backend = new CopyingBackend();
    let received: Float64Array | undefined;
    backend.matSetIdentity = (_destination, value) => {
      received = value;
    };
    const localClient = createOpenCv(backend);
    const destination = localClient.zerosF64(2, 2, 4);
    const arrayLike = {
      0: true,
      1: -0,
      2: Number.POSITIVE_INFINITY,
      3: Number.NaN,
      length: 4,
    };
    // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises the untyped JavaScript binding boundary.
    Reflect.apply(localClient.setIdentity, localClient, [destination, arrayLike]);
    expect(received?.[0]).toBe(1);
    expect(Object.is(received?.[1], -0)).toBeTrue();
    expect(received?.[2]).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isNaN(received?.[3])).toBeTrue();

    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(localClient.setIdentity, localClient, []);
    }).toThrow(BindingError);
    expect(() =>
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(localClient.setIdentity, localClient, [destination, [1, 2, 3, 4], "extra"]),
    ).toThrow(BindingError);
    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises an invalid runtime Scalar.
      Reflect.apply(localClient.setIdentity, localClient, [destination, [1, 2, 3]]);
    }).toThrow(BindingError);
    expect(() =>
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises an invalid runtime Scalar lane.
      Reflect.apply(localClient.setIdentity, localClient, [destination, [1, 2, "3", 4]]),
    ).toThrow(TypeError);

    destination.dispose();
  });

  test("controls logging and computes optimal DFT sizes", () => {
    const initial = client.getLogLevel();
    expect(client.setLogLevel(5)).toBe(initial);
    expect(client.getLogLevel()).toBe(5);
    expect(client.setLogLevel(initial)).toBe(5);
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    expect(localClient.getOptimalDFTSize.bind(localClient)).toHaveLength(1);
    expect(localClient.getOptimalDFTSize(7)).toBe(8);
    expect(localClient.getOptimalDFTSize(25)).toBe(25);
    expect(localClient.getOptimalDFTSize(-1)).toBe(-1);
    expect(localClient.getOptimalDFTSize(2_125_763_999)).toBe(2_125_764_000);
    expect(localClient.getOptimalDFTSize(2_125_764_000)).toBe(-1);

    // SAFETY: This widens only the plain-JavaScript call surface exercised by the binding audit.
    const javascriptClient = localClient as typeof localClient & {
      getOptimalDFTSize(size?: boolean | number | string | null, extra?: number): number;
    };
    expect(() => javascriptClient.getOptimalDFTSize()).toThrow(
      new BindingError("function getOptimalDFTSize called with 0 arguments, expected 1 args!"),
    );
    expect(() => javascriptClient.getOptimalDFTSize(7, 1)).toThrow(
      new BindingError("function getOptimalDFTSize called with 2 arguments, expected 1 args!"),
    );
    expect(javascriptClient.getOptimalDFTSize(7.9)).toBe(8);
    expect(backend.optimalDftSizeInputs.at(-1)).toBe(7);
    expect(javascriptClient.getOptimalDFTSize(Number.NaN)).toBe(1);
    expect(backend.optimalDftSizeInputs.at(-1)).toBe(0);
    expect(javascriptClient.getOptimalDFTSize(true)).toBe(1);
    expect(backend.optimalDftSizeInputs.at(-1)).toBe(1);

    const callCountBeforeInvalidInputs = backend.optimalDftSizeInputs.length;
    const rejected: ReadonlyArray<readonly [boolean | number | string | null | undefined, string]> =
      [
        [null, 'Cannot convert "null" to int'],
        [undefined, 'Cannot convert "undefined" to int'],
        ["7", 'Cannot convert "7" to int'],
        [
          Number.POSITIVE_INFINITY,
          'Passing a number "Infinity" from JS side to C/C++ side to an argument of type "int", which is outside the valid range [-2147483648, 2147483647]!',
        ],
        [
          2_147_483_648,
          'Passing a number "2147483648" from JS side to C/C++ side to an argument of type "int", which is outside the valid range [-2147483648, 2147483647]!',
        ],
      ];
    for (const [input, message] of rejected) {
      expect(() => javascriptClient.getOptimalDFTSize(input)).toThrow(new TypeError(message));
    }
    expect(backend.optimalDftSizeInputs).toHaveLength(callCountBeforeInvalidInputs);
  });

  test("applies linear and perspective transforms", () => {
    const source = client.matFromU8(1, 2, 1, new Uint8Array([1, 2]));
    const coefficients = client.matFromF64(1, 2, 1, new Float64Array([2, 1]));
    const transformed = client.transform(source, coefficients);
    const transformedDestination = client.zerosU8(1, 2, 1);
    client.transform(source, coefficients, transformedDestination);
    expect(transformed.toUint8Array()).toEqual(new Uint8Array([3, 5]));
    expect(transformedDestination.toUint8Array()).toEqual(new Uint8Array([3, 5]));

    const points = client.matFromF64(1, 2, 2, new Float64Array([1, 2, 3, 4]));
    const identity = client.matFromF64(3, 3, 1, new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
    const projected = client.perspectiveTransform(points, identity);
    const projectedDestination = client.zerosF64(1, 2, 2);
    client.perspectiveTransform(points, identity, projectedDestination);
    expect(projected.toFloat64Array()).toEqual(new Float64Array([1, 2, 3, 4]));
    expect(projectedDestination.toFloat64Array()).toEqual(new Float64Array([1, 2, 3, 4]));

    for (const matrix of [
      projectedDestination,
      projected,
      identity,
      points,
      transformedDestination,
      transformed,
      coefficients,
      source,
    ]) {
      matrix.dispose();
    }
  });

  test("measures contours and classifies polygon points", () => {
    const contour = client.matFromI32(4, 1, 2, new Int32Array([0, 0, 4, 0, 4, 3, 0, 3]));

    expect(client.arcLength(contour, false)).toBe(11);
    expect(client.arcLength(contour, true)).toBe(14);
    expect(client.contourArea(contour)).toBe(12);
    expect(client.boundingRect(contour)).toEqual({ x: 0, y: 0, width: 5, height: 4 });
    expect(client.isContourConvex(contour)).toBeTrue();
    expect(client.pointPolygonTest(contour, { x: 2, y: 1 }, true)).toBe(1);

    contour.dispose();
  });

  test("matches contour measurement call contracts", () => {
    const contour = client.matFromI32(4, 1, 2, new Int32Array([0, 0, 4, 0, 4, 3, 0, 3]));

    expect(client.arcLength.length).toBe(2);
    expect(client.contourArea.length).toBe(0);
    expect(client.boundingRect.length).toBe(1);
    // @ts-expect-error Runtime parity uses JavaScript boolean coercion.
    expect(client.arcLength(contour, "closed")).toBe(14);
    // @ts-expect-error Runtime parity uses JavaScript boolean coercion.
    expect(client.contourArea(contour, 0)).toBe(12);
    // @ts-expect-error Runtime parity rejects missing arguments.
    expect(() => client.arcLength(contour)).toThrow(BindingError);
    // @ts-expect-error Runtime parity rejects extra arguments.
    expect(() => client.boundingRect(contour, 1)).toThrow(BindingError);

    contour.dispose();
  });

  test("matches polygon query call contracts", () => {
    type JavascriptBindingValue =
      boolean | number | bigint | string | symbol | object | null | undefined;
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const contour = localClient.matFromI32(4, 1, 2, new Int32Array([0, 0, 4, 0, 4, 3, 0, 3]));
    // SAFETY: This widens only the plain-JavaScript call shapes exercised by the binding audit.
    const javascriptClient = localClient as typeof localClient & {
      isContourConvex(contour?: JavascriptBindingValue, extra?: JavascriptBindingValue): boolean;
      pointPolygonTest(
        contour?: JavascriptBindingValue,
        point?: JavascriptBindingValue,
        measureDistance?: JavascriptBindingValue,
        extra?: JavascriptBindingValue,
      ): number;
    };

    expect(localClient.isContourConvex.length).toBe(1);
    expect(localClient.pointPolygonTest.length).toBe(3);
    expect(() => javascriptClient.isContourConvex()).toThrow(BindingError);
    expect(() => javascriptClient.isContourConvex(contour, 1)).toThrow(BindingError);
    expect(() => javascriptClient.isContourConvex(null)).toThrow(BindingError);
    expect(() => javascriptClient.pointPolygonTest()).toThrow(BindingError);
    expect(() => javascriptClient.pointPolygonTest(contour)).toThrow(BindingError);
    expect(() => javascriptClient.pointPolygonTest(contour, { x: 1, y: 1 })).toThrow(BindingError);
    expect(() => javascriptClient.pointPolygonTest(contour, { x: 1, y: 1 }, false, 1)).toThrow(
      BindingError,
    );
    expect(() => javascriptClient.pointPolygonTest(null, { x: 1, y: 1 }, false)).toThrow(
      BindingError,
    );

    const propertyReads: string[] = [];
    const point = {
      get x(): number {
        propertyReads.push("x");
        return 16_777_217;
      },
      get y(): boolean {
        propertyReads.push("y");
        return true;
      },
      get ignored(): never {
        throw new Error("point extras must not be read");
      },
    };
    expect(javascriptClient.pointPolygonTest(contour, point, "distance")).toBeLessThan(0);
    expect(propertyReads).toEqual(["x", "y"]);
    expect(backend.pointPolygonTestInputs.at(-1)).toEqual({
      x: Math.fround(16_777_217),
      y: 1,
      measureDistance: true,
    });

    const missingFieldReads: string[] = [];
    const missingY = {
      get x(): number {
        missingFieldReads.push("x");
        return 1;
      },
    };
    expect(() => javascriptClient.pointPolygonTest(contour, missingY, false)).toThrow(BindingError);
    expect(missingFieldReads).toEqual([]);

    const arrayWithFields = Object.assign([], { x: 2, y: 1.5 });
    const functionWithFields = Object.assign(() => undefined, { x: 2, y: 1.5 });
    expect(javascriptClient.pointPolygonTest(contour, arrayWithFields, false)).toBe(1);
    expect(javascriptClient.pointPolygonTest(contour, functionWithFields, false)).toBe(1);

    expect(
      javascriptClient.pointPolygonTest(
        contour,
        { x: Number.NaN, y: Number.POSITIVE_INFINITY },
        0n,
      ),
    ).toBe(-1);
    const nonFiniteInput = backend.pointPolygonTestInputs.at(-1);
    expect(nonFiniteInput?.x).toBeNaN();
    expect(nonFiniteInput?.y).toBe(Number.POSITIVE_INFINITY);
    expect(nonFiniteInput?.measureDistance).toBeFalse();

    const boxedNumber: object = Object(1);
    const rejectedPoints: JavascriptBindingValue[] = [
      "point",
      null,
      undefined,
      boxedNumber,
      1n,
      [1, 2],
      {},
      { x: 1 },
      { y: 1 },
    ];
    const rejectedComponents: JavascriptBindingValue[] = [
      "1",
      null,
      undefined,
      boxedNumber,
      1n,
      [],
    ];
    const callCountBeforeRejections = backend.pointPolygonTestInputs.length;
    for (const rejectedPoint of rejectedPoints) {
      expect(() => javascriptClient.pointPolygonTest(contour, rejectedPoint, false)).toThrow();
    }
    for (const rejectedComponent of rejectedComponents) {
      expect(() =>
        javascriptClient.pointPolygonTest(contour, { x: rejectedComponent, y: 1 }, false),
      ).toThrow();
      expect(() =>
        javascriptClient.pointPolygonTest(contour, { x: 1, y: rejectedComponent }, false),
      ).toThrow();
    }
    expect(backend.pointPolygonTestInputs).toHaveLength(callCountBeforeRejections);

    contour.dispose();
  });

  test("creates image-processing helpers with structured point results", () => {
    const kernel = client.getStructuringElement(1, { width: 3, height: 3 }, { x: 1, y: 1 });
    expect(kernel.toUint8Array()).toEqual(new Uint8Array([0, 1, 0, 1, 1, 1, 0, 1, 0]));

    const window = client.zerosF64(3, 3, 1);
    expect(client.createHanningWindow(window, { width: 3, height: 3 }, 6)).toBeUndefined();
    expect(window.toFloat64Array()).toEqual(new Float64Array([0, 0, 0, 0, 1, 0, 0, 0, 0]));
    const allocatedWindow = client.createHanningWindowAlloc({ width: 3, height: 3 }, "f64");
    expect(allocatedWindow.toFloat64Array()).toEqual(new Float64Array([0, 0, 0, 0, 1, 0, 0, 0, 0]));

    expect(client.ellipse2Poly({ x: 0, y: 0 }, { width: 10, height: 5 }, 0, 0, 90, 90)).toEqual([
      { x: 10, y: 0 },
      { x: 0, y: 5 },
    ]);
    expect(
      client.clipLine({ x: 10, y: 20, width: 5, height: 4 }, { x: 8, y: 21 }, { x: 16, y: 21 }),
    ).toEqual([
      { x: 10, y: 21 },
      { x: 14, y: 21 },
    ]);
    expect(
      client.clipLine({ x: 10, y: 20, width: 5, height: 4 }, { x: 0, y: 0 }, { x: 1, y: 1 }),
    ).toBeUndefined();
    expect(() => client.createHanningWindow(window, { width: 1, height: 3 }, 5)).toThrow();

    allocatedWindow.dispose();
    window.dispose();
    kernel.dispose();
  });

  test("matches createHanningWindow destination, conversion, and arity contracts", () => {
    expect(client.createHanningWindow.bind(client)).toHaveLength(3);
    const destination = client.zerosF32(2, 3, 1);
    // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises the untyped JavaScript binding boundary.
    Reflect.apply(client.createHanningWindow, client, [
      destination,
      { width: 3.9, height: 2.9 },
      5.9,
    ]);
    expect(destination.toFloat32Array()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));

    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.createHanningWindow, client, [destination, { width: 3, height: 3 }]);
    }).toThrow(BindingError);
    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.createHanningWindow, client, [
        destination,
        { width: 3, height: 3 },
        5,
        1,
      ]);
    }).toThrow(BindingError);
    destination.dispose();
  });

  test("matches getStructuringElement overload and integer conversion contracts", () => {
    expect(client.getStructuringElement.bind(client)).toHaveLength(0);
    // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises the untyped JavaScript binding boundary.
    const converted = Reflect.apply(client.getStructuringElement, client, [
      true,
      { width: 3.9, height: 2.9 },
      { x: true, y: false },
    ]);
    expect([converted.rows, converted.columns]).toEqual([2, 3]);
    converted.dispose();

    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.getStructuringElement, client, [0]);
    }).toThrow(BindingError);
    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.getStructuringElement, client, [
        0,
        { width: 3, height: 3 },
        { x: 1, y: 1 },
        0,
      ]);
    }).toThrow(BindingError);
  });

  test("constructs affine and perspective matrices", () => {
    const rotation = client.getRotationMatrix2D({ x: 1, y: 2 }, 90, 1);
    expect(Array.from(rotation.toFloat64Array())).toEqual([
      6.123_233_995_736_766e-17, 1, -1, -1, 6.123_233_995_736_766e-17, 3,
    ]);

    const affineSource = client.matFromF64(3, 2, 1, new Float64Array([0, 0, 1, 0, 0, 1]));
    const affineDestination = client.matFromF64(3, 2, 1, new Float64Array([2, 3, 4, 3, 2, 6]));
    const affine = client.getAffineTransform(affineSource, affineDestination);
    expect(Array.from(affine.toFloat64Array())).toEqual([2, 0, 2, 0, 3, 3]);

    const inverse = client.zerosF64(2, 3, 1);
    expect(client.invertAffineTransform(affine, inverse)).toBeUndefined();
    expect(Array.from(inverse.toFloat64Array())).toEqual([0.5, 0, -1, 0, 1 / 3, -1]);

    const inverseAlloc = client.invertAffineTransformAlloc(affine);
    expect(Array.from(inverseAlloc.toFloat64Array())).toEqual([0.5, 0, -1, 0, 1 / 3, -1]);

    const perspectiveSource = client.matFromF32(
      4,
      2,
      1,
      new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    );
    const perspectiveDestination = client.matFromF32(
      4,
      2,
      1,
      new Float32Array([2, 3, 4, 3, 4, 6, 2, 6]),
    );
    const perspective = client.getPerspectiveTransform(perspectiveSource, perspectiveDestination);
    expect(Array.from(perspective.toFloat64Array())).toEqual([2, 0, 2, 0, 3, 3, 0, 0, 1]);
    for (const matrix of [
      perspective,
      perspectiveDestination,
      perspectiveSource,
      inverseAlloc,
      inverse,
      affine,
      affineDestination,
      affineSource,
      rotation,
    ]) {
      matrix.dispose();
    }
  });

  test("matches invertAffineTransform destination and arity contracts", () => {
    expect(client.invertAffineTransform.bind(client)).toHaveLength(2);
    const source = client.matFromF32(2, 3, 1, new Float32Array([2, 0, 4, 0, 3, -6]));
    const destination = client.zerosF32(2, 3, 1);
    client.invertAffineTransform(source, destination);
    expect(destination.depth).toBe("f32");
    expect(Array.from(destination.toFloat32Array())).toEqual([
      0.5,
      0,
      -2,
      0,
      Math.fround(1 / 3),
      2,
    ]);

    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.invertAffineTransform, client, [source]);
    }).toThrow(BindingError);
    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.invertAffineTransform, client, [source, destination, 1]);
    }).toThrow(BindingError);

    destination.dispose();
    source.dispose();
  });

  test("matches getAffineTransform exact arity", () => {
    expect(client.getAffineTransform.bind(client)).toHaveLength(2);
    const source = client.matFromF32(3, 2, 1, new Float32Array([0, 0, 1, 0, 0, 1]));
    const destination = client.matFromF32(3, 2, 1, new Float32Array([2, 3, 4, 3, 2, 6]));

    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.getAffineTransform, client, [source]);
    }).toThrow(BindingError);
    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.getAffineTransform, client, [source, destination, 1]);
    }).toThrow(BindingError);

    destination.dispose();
    source.dispose();
  });

  test("matches getRotationMatrix2D binding contracts", () => {
    type JavascriptBindingValue =
      boolean | number | bigint | string | symbol | object | null | undefined;
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    // SAFETY: This widens only the plain-JavaScript call shapes exercised by the binding audit.
    const javascriptClient = localClient as typeof localClient & {
      getRotationMatrix2D(
        center?: JavascriptBindingValue,
        angleDegrees?: JavascriptBindingValue,
        scale?: JavascriptBindingValue,
        extra?: JavascriptBindingValue,
      ): Mat;
    };

    expect(localClient.getRotationMatrix2D.length).toBe(3);
    expect(() => javascriptClient.getRotationMatrix2D()).toThrow(
      new BindingError("function getRotationMatrix2D called with 0 arguments, expected 3 args!"),
    );
    expect(() => javascriptClient.getRotationMatrix2D({ x: 1, y: 2 }, 30)).toThrow(
      new BindingError("function getRotationMatrix2D called with 2 arguments, expected 3 args!"),
    );
    const arityReads: string[] = [];
    const unreadCenter = {
      get x(): never {
        arityReads.push("x");
        throw new Error("arity must be checked first");
      },
      y: 2,
    };
    expect(() => javascriptClient.getRotationMatrix2D(unreadCenter, 30, 2, 1)).toThrow(
      new BindingError("function getRotationMatrix2D called with 4 arguments, expected 3 args!"),
    );
    expect(arityReads).toEqual([]);

    const propertyReads: string[] = [];
    const center = {
      get x(): number {
        propertyReads.push("x");
        return 16_777_217;
      },
      get y(): boolean {
        propertyReads.push("y");
        return true;
      },
      get ignored(): never {
        throw new Error("point extras must not be read");
      },
    };
    const first = javascriptClient.getRotationMatrix2D(center, true, false);
    expect(propertyReads).toEqual(["x", "y"]);
    expect(backend.rotationMatrixInputs.at(-1)).toEqual({
      centerX: Math.fround(16_777_217),
      centerY: 1,
      angleDegrees: 1,
      scale: 0,
    });
    expect([first.rows, first.columns, first.channels, first.depth]).toEqual([2, 3, 1, "f64"]);

    const second = javascriptClient.getRotationMatrix2D(
      { x: Number.NaN, y: Number.POSITIVE_INFINITY },
      Number.NEGATIVE_INFINITY,
      Number.NaN,
    );
    const nonFinite = backend.rotationMatrixInputs.at(-1);
    expect(nonFinite?.centerX).toBeNaN();
    expect(nonFinite?.centerY).toBe(Number.POSITIVE_INFINITY);
    expect(nonFinite?.angleDegrees).toBe(Number.NEGATIVE_INFINITY);
    expect(nonFinite?.scale).toBeNaN();

    const signedZero = javascriptClient.getRotationMatrix2D({ x: -0, y: -0 }, -0, -0);
    const signedZeroInput = backend.rotationMatrixInputs.at(-1);
    expect(Object.is(signedZeroInput?.centerX, -0)).toBeTrue();
    expect(Object.is(signedZeroInput?.centerY, -0)).toBeTrue();
    expect(Object.is(signedZeroInput?.angleDegrees, -0)).toBeTrue();
    expect(Object.is(signedZeroInput?.scale, -0)).toBeTrue();

    expect(() => javascriptClient.getRotationMatrix2D({ x: 1, y: 2 }, "30", 2)).toThrow(
      new TypeError('Cannot convert "30" to double'),
    );
    expect(() => javascriptClient.getRotationMatrix2D({ x: "1", y: 2 }, 30, 2)).toThrow(
      new TypeError('Cannot convert "1" to float'),
    );
    expect(() => javascriptClient.getRotationMatrix2D({ x: 1 }, 30, 2)).toThrow(BindingError);

    first.dispose();
    second.dispose();
    signedZero.dispose();
  });

  test("computes determinants, inverses, and linear solves", () => {
    const coefficients = client.matFromF64(2, 2, 1, new Float64Array([4, 7, 2, 6]));
    expect(client.determinant(coefficients)).toBeCloseTo(10);

    const inverse = client.zerosF64(2, 2, 1);
    expect(client.invert(coefficients, inverse)).toBe(1);
    expect(Array.from(inverse.toFloat64Array())).toEqual([0.6, -0.7, -0.2, 0.4]);

    const rightHandSide = client.matFromF64(2, 1, 1, new Float64Array([1, 0]));
    const solution = client.zerosF64(2, 1, 1);
    expect(client.solve(coefficients, rightHandSide, solution)).toBeTrue();
    expect(Array.from(solution.toFloat64Array())).toEqual([0.6, -0.2]);

    for (const matrix of [solution, rightHandSide, inverse, coefficients]) matrix.dispose();
  });

  test("matches determinant call and Mat binding contracts", () => {
    type JavascriptBindingValue =
      boolean | number | bigint | string | symbol | object | null | undefined;
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const source = localClient.matFromF64(2, 2, 1, new Float64Array([1, 2, 3, 4]));
    // SAFETY: This widens only the plain-JavaScript call shapes exercised by the binding audit.
    const javascriptClient = localClient as typeof localClient & {
      determinant(source?: JavascriptBindingValue, extra?: JavascriptBindingValue): number;
    };

    expect(localClient.determinant.length).toBe(1);
    const sourceBefore = source.toFloat64Array();
    expect(localClient.determinant(source)).toBe(-2);
    expect(source.toFloat64Array()).toEqual(sourceBefore);
    expect(() => javascriptClient.determinant()).toThrow(BindingError);
    expect(() => javascriptClient.determinant(source, 1)).toThrow(BindingError);
    expect(() => javascriptClient.determinant(null)).toThrow(
      new BindingError("null is not a valid Mat"),
    );
    expect(() => javascriptClient.determinant({})).toThrow(BindingError);

    source.dispose();
    expect(() => localClient.determinant(source)).toThrow(
      new BindingError("Cannot pass deleted object as a pointer of type Mat"),
    );
  });

  test("exposes matrix-based core operations", () => {
    expect(client).toHaveProperty("add");
    expect(client).toHaveProperty("subtract");
    expect(client).toHaveProperty("absdiff");
    expect(client).toHaveProperty("bitwiseAnd");
    expect(client).toHaveProperty("bitwiseOr");
    expect(client).toHaveProperty("bitwiseXor");
    expect(client).toHaveProperty("bitwiseNot");
    expect(client).toHaveProperty("min");
    expect(client).toHaveProperty("max");
    expect(client).toHaveProperty("compareEqual");
    expect(client).toHaveProperty("inRange");
    expect(client).toHaveProperty("countNonZero");

    const left = client.matFromU8(1, 3, 1, new Uint8Array([250, 2, 3]));
    const right = client.matFromU8(1, 3, 1, new Uint8Array([10, 5, 3]));
    const added = client.add(left, right);
    const subtracted = client.subtract(left, right);
    const difference = client.absdiff(left, right);
    const equal = client.compareEqual(left, right);
    const inverted = client.bitwiseNotAlloc(left);

    expect(added.toUint8Array()).toEqual(new Uint8Array([255, 7, 6]));
    expect(subtracted.toUint8Array()).toEqual(new Uint8Array([240, 0, 0]));
    expect(difference.toUint8Array()).toEqual(new Uint8Array([240, 3, 0]));
    expect(equal.toUint8Array()).toEqual(new Uint8Array([0, 0, 255]));
    expect(inverted.toUint8Array()).toEqual(new Uint8Array([5, 253, 252]));
    expect(client.countNonZero(left)).toBe(3);

    for (const matrix of [added, subtracted, difference, equal, inverted, left, right]) {
      matrix.dispose();
    }
  });

  test("matches the exact one-argument countNonZero call contract", () => {
    const source = client.matFromU8(2, 3, 1, new Uint8Array([0, 1, 2, 0, 3, 0]));

    expect(client.countNonZero.length).toBe(1);
    expect(() => {
      // @ts-expect-error Runtime parity requires testing missing arguments from plain JavaScript.
      client.countNonZero();
    }).toThrow(new BindingError("function countNonZero called with 0 arguments, expected 1 args!"));
    expect(client.countNonZero(source)).toBe(3);
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an extra argument from plain JavaScript.
      client.countNonZero(source, 1);
    }).toThrow(new BindingError("function countNonZero called with 2 arguments, expected 1 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a null Mat from plain JavaScript.
      client.countNonZero(null);
    }).toThrow(new BindingError("null is not a valid Mat"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an undefined Mat from plain JavaScript.
      client.countNonZero(undefined);
    }).toThrow(new TypeError("Cannot read properties of undefined (reading '$$')"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a structural object from plain JavaScript.
      client.countNonZero({});
    }).toThrow(new BindingError('Cannot pass "[object Object]" as a Mat'));

    source.dispose();
    expect(() => client.countNonZero(source)).toThrow(
      new BindingError("Cannot pass deleted object as a pointer of type Mat"),
    );
  });

  test("matches bitwiseNot destination and mask contracts", () => {
    expect(client.bitwiseNot.length).toBe(0);

    const source = client.matFromU8(2, 2, 1, new Uint8Array([0, 1, 2, 3]));
    const mask = client.matFromI8(2, 2, 1, new Int8Array([1, 0, 2, 0]));
    const populated = client.matFromU8(2, 2, 1, new Uint8Array([9, 9, 9, 9]));
    expect(client.bitwiseNot(source, populated, mask)).toBeUndefined();
    expect(populated.toUint8Array()).toEqual(new Uint8Array([255, 9, 253, 9]));

    const fresh = client.emptyMat();
    client.bitwiseNot(source, fresh, mask);
    expect(fresh.toUint8Array()).toEqual(new Uint8Array([255, 0, 253, 0]));

    const unmasked = client.emptyMat();
    client.bitwiseNot(source, unmasked);
    expect(unmasked.toUint8Array()).toEqual(new Uint8Array([255, 254, 253, 252]));

    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.bitwiseNot, client, [source]);
    }).toThrow(BindingError);
    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.bitwiseNot, client, [source, fresh, mask, mask]);
    }).toThrow(BindingError);

    for (const matrix of [unmasked, fresh, populated, mask, source]) matrix.dispose();
  });

  test("exposes matrix layout operations", () => {
    expect(client).toHaveProperty("flip");
    expect(client).toHaveProperty("transpose");
    expect(client).toHaveProperty("rotate");
    expect(client).toHaveProperty("repeat");

    const source = client.matFromU8(2, 3, 1, new Uint8Array([1, 2, 3, 4, 5, 6]));
    const horizontal = client.flipAlloc(source, 1);
    const transposed = client.transposeAlloc(source);
    const clockwise = client.rotateAlloc(source, 0);
    const repeated = client.repeatAlloc(source, 2, 1);
    const flippedDestination = client.zerosU8(2, 3, 1);
    const transposedDestination = client.zerosU8(3, 2, 1);
    const rotatedDestination = client.zerosU8(3, 2, 1);
    const repeatedDestination = client.zerosU8(4, 3, 1);
    client.flip(source, flippedDestination, 1);
    expect(client.transpose(source, transposedDestination)).toBeUndefined();
    client.rotate(source, rotatedDestination, 0);
    client.repeat(source, 2, 1, repeatedDestination);
    expect(horizontal.toUint8Array()).toEqual(new Uint8Array([3, 2, 1, 6, 5, 4]));
    expect(transposed.rows).toBe(3);
    expect(transposed.columns).toBe(2);
    expect(transposed.toUint8Array()).toEqual(new Uint8Array([1, 4, 2, 5, 3, 6]));
    expect(clockwise.rows).toBe(3);
    expect(clockwise.columns).toBe(2);
    expect(clockwise.toUint8Array()).toEqual(new Uint8Array([4, 1, 5, 2, 6, 3]));
    expect(repeated.rows).toBe(4);
    expect(repeated.toUint8Array()).toEqual(new Uint8Array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]));
    expect(flippedDestination.toUint8Array()).toEqual(new Uint8Array([3, 2, 1, 6, 5, 4]));
    expect(transposedDestination.toUint8Array()).toEqual(new Uint8Array([1, 4, 2, 5, 3, 6]));
    expect(rotatedDestination.toUint8Array()).toEqual(new Uint8Array([4, 1, 5, 2, 6, 3]));
    expect(repeatedDestination.toUint8Array()).toEqual(
      new Uint8Array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]),
    );
    repeatedDestination.dispose();
    rotatedDestination.dispose();
    transposedDestination.dispose();
    flippedDestination.dispose();
    repeated.dispose();
    clockwise.dispose();
    transposed.dispose();
    horizontal.dispose();
    source.dispose();
  });

  test("matches the exact two-argument transpose call contract", () => {
    const source = client.matFromU8(2, 1, 1, new Uint8Array([1, 2]));
    const destination = client.zerosU8(1, 2, 1);

    expect(client.transpose.length).toBe(2);
    expect(() => {
      // @ts-expect-error Runtime parity requires testing missing arguments from plain JavaScript.
      client.transpose();
    }).toThrow(new BindingError("function transpose called with 0 arguments, expected 2 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a missing destination from plain JavaScript.
      client.transpose(source);
    }).toThrow(new BindingError("function transpose called with 1 arguments, expected 2 args!"));
    expect(client.transpose(source, destination)).toBeUndefined();
    expect(destination.toUint8Array()).toEqual(new Uint8Array([1, 2]));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an extra argument from plain JavaScript.
      client.transpose(source, destination, 1);
    }).toThrow(new BindingError("function transpose called with 3 arguments, expected 2 args!"));

    destination.dispose();
    source.dispose();
  });

  test("matches the exact three-argument flip call contract", () => {
    const source = client.matFromU8(2, 3, 1, new Uint8Array([1, 2, 3, 4, 5, 6]));
    const destination = client.zerosU8(2, 3, 1);

    expect(client.flip.length).toBe(3);
    expect(() => {
      // @ts-expect-error Runtime parity requires testing missing arguments from plain JavaScript.
      client.flip();
    }).toThrow(new BindingError("function flip called with 0 arguments, expected 3 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a missing destination and code.
      client.flip(source);
    }).toThrow(new BindingError("function flip called with 1 arguments, expected 3 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a missing code.
      client.flip(source, destination);
    }).toThrow(new BindingError("function flip called with 2 arguments, expected 3 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an extra argument.
      client.flip(source, destination, 1, 2);
    }).toThrow(new BindingError("function flip called with 4 arguments, expected 3 args!"));

    expect(client.flip(source, destination, 2)).toBeUndefined();
    expect(destination.toUint8Array()).toEqual(new Uint8Array([3, 2, 1, 6, 5, 4]));
    expect(client.flip(source, destination, -2)).toBeUndefined();
    expect(destination.toUint8Array()).toEqual(new Uint8Array([6, 5, 4, 3, 2, 1]));
    expect(client.flip(source, destination, Number.NaN)).toBeUndefined();
    expect(destination.toUint8Array()).toEqual(new Uint8Array([4, 5, 6, 1, 2, 3]));
    expect(() => client.flip(source, destination, 2_147_483_648)).toThrow(
      new TypeError(
        'Passing a number "2147483648" from JS side to C/C++ side to an argument of type "int", which is outside the valid range [-2147483648, 2147483647]!',
      ),
    );
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a null Mat from plain JavaScript.
      client.flip(null, destination, 1);
    }).toThrow(new BindingError("null is not a valid Mat"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an undefined Mat from plain JavaScript.
      client.flip(undefined, destination, 1);
    }).toThrow(new TypeError("Cannot read properties of undefined (reading '$$')"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a structural object from plain JavaScript.
      client.flip({}, destination, 1);
    }).toThrow(new BindingError('Cannot pass "[object Object]" as a Mat'));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a null destination from plain JavaScript.
      client.flip(source, null, 1);
    }).toThrow(new BindingError("null is not a valid Mat"));

    destination.dispose();
    source.dispose();
  });

  test("matches the exact four-argument repeat call contract", () => {
    const source = client.matFromU8(1, 2, 1, new Uint8Array([1, 2]));
    const destination = client.zerosU8(2, 4, 1);

    expect(client.repeat.length).toBe(4);
    expect(() => {
      // @ts-expect-error Runtime parity requires testing missing arguments from plain JavaScript.
      client.repeat();
    }).toThrow(new BindingError("function repeat called with 0 arguments, expected 4 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing missing arguments from plain JavaScript.
      client.repeat(source);
    }).toThrow(new BindingError("function repeat called with 1 arguments, expected 4 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing missing arguments from plain JavaScript.
      client.repeat(source, 1);
    }).toThrow(new BindingError("function repeat called with 2 arguments, expected 4 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a missing destination from plain JavaScript.
      client.repeat(source, 1, 2);
    }).toThrow(new BindingError("function repeat called with 3 arguments, expected 4 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an extra argument from plain JavaScript.
      client.repeat(source, 1, 2, destination, 5);
    }).toThrow(new BindingError("function repeat called with 5 arguments, expected 4 args!"));
    expect(client.repeat(source, 2, 2, destination)).toBeUndefined();
    expect(destination.toUint8Array()).toEqual(new Uint8Array([1, 2, 1, 2, 1, 2, 1, 2]));

    const fractionalDestination = client.zerosU8(1, 4, 1);
    expect(client.repeat(source, 1.9, 2.9, fractionalDestination)).toBeUndefined();
    expect(fractionalDestination.toUint8Array()).toEqual(new Uint8Array([1, 2, 1, 2]));

    const booleanDestination = client.zerosU8(1, 2, 1);
    expect(() => {
      // @ts-expect-error Runtime parity requires boolean-to-int conversion from plain JavaScript.
      client.repeat(source, true, true, booleanDestination);
    }).not.toThrow();
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a null Mat from plain JavaScript.
      client.repeat(null, 1, 1, destination);
    }).toThrow(new BindingError("null is not a valid Mat"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an undefined Mat from plain JavaScript.
      client.repeat(undefined, 1, 1, destination);
    }).toThrow(new TypeError("Cannot read properties of undefined (reading '$$')"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a structural Mat from plain JavaScript.
      client.repeat({}, 1, 1, destination);
    }).toThrow(new BindingError('Cannot pass "[object Object]" as a Mat'));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a null destination from plain JavaScript.
      client.repeat(source, 1, 1, null);
    }).toThrow(new BindingError("null is not a valid Mat"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an invalid repeat count.
      client.repeat(source, "1", 1, destination);
    }).toThrow(new TypeError('Cannot convert "1" to int'));
    expect(() => client.repeat(source, 2_147_483_648, 1, destination)).toThrow(
      new TypeError(
        'Passing a number "2147483648" from JS side to C/C++ side to an argument of type "int", which is outside the valid range [-2147483648, 2147483647]!',
      ),
    );

    const deleted = client.matFromU8(1, 1, 1, new Uint8Array([1]));
    deleted.dispose();
    expect(() => client.repeat(deleted, 1, 1, destination)).toThrow(
      new BindingError("Cannot pass deleted object as a pointer of type Mat"),
    );

    booleanDestination.dispose();
    fractionalDestination.dispose();
    destination.dispose();
    source.dispose();
  });

  test("matches the exact three-argument rotate call contract", () => {
    const source = client.matFromU8(2, 3, 1, new Uint8Array([1, 2, 3, 4, 5, 6]));
    const destination = client.zerosU8(3, 2, 1);

    expect(client.ROTATE_90_CLOCKWISE).toBe(0);
    expect(client.ROTATE_180).toBe(1);
    expect(client.ROTATE_90_COUNTERCLOCKWISE).toBe(2);
    expect(client.rotate.length).toBe(3);
    expect(() => {
      // @ts-expect-error Runtime parity requires testing missing arguments from plain JavaScript.
      client.rotate();
    }).toThrow(new BindingError("function rotate called with 0 arguments, expected 3 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing missing arguments from plain JavaScript.
      client.rotate(source);
    }).toThrow(new BindingError("function rotate called with 1 arguments, expected 3 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a missing code from plain JavaScript.
      client.rotate(source, destination);
    }).toThrow(new BindingError("function rotate called with 2 arguments, expected 3 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an extra argument from plain JavaScript.
      client.rotate(source, destination, 0, 1);
    }).toThrow(new BindingError("function rotate called with 4 arguments, expected 3 args!"));
    expect(client.rotate(source, destination, 0.9)).toBeUndefined();
    expect(destination.toUint8Array()).toEqual(new Uint8Array([4, 1, 5, 2, 6, 3]));

    destination.dispose();
    source.dispose();
  });

  test("matches transpose errors for deleted matrices", () => {
    const deletedSource = client.matFromU8(1, 1, 1, new Uint8Array([7]));
    const liveDestination = client.emptyMat();
    deletedSource.dispose();

    expect(() => client.transpose(deletedSource, liveDestination)).toThrow(
      new BindingError("Cannot pass deleted object as a pointer of type Mat"),
    );
    expect(() => deletedSource.rows).toThrow(
      new BindingError("Cannot pass deleted object as a pointer of type Mat const*"),
    );

    const liveSource = client.matFromU8(1, 1, 1, new Uint8Array([8]));
    const deletedDestination = client.emptyMat();
    deletedDestination.dispose();
    expect(() => client.transpose(liveSource, deletedDestination)).toThrow(
      new BindingError("Cannot pass deleted object as a pointer of type Mat"),
    );

    liveSource.dispose();
    liveDestination.dispose();
  });

  test("exposes typed matrix reductions", () => {
    expect(client).toHaveProperty("sum");
    expect(client).toHaveProperty("mean");
    expect(client).toHaveProperty("minMaxLoc");
    expect(client).toHaveProperty("trace");

    const source = client.matFromU8(1, 2, 3, new Uint8Array([1, 10, 100, 3, 30, 200]));
    expect(client.sum(source)).toEqual([4, 40, 300, 0]);
    expect(client.mean(source)).toEqual([2, 20, 150, 0]);
    source.dispose();

    const extremaSource = client.matFromU8(2, 3, 1, new Uint8Array([5, 2, 9, 2, 9, 4]));
    expect(client.minMaxLoc(extremaSource)).toEqual({
      maxLoc: { x: 2, y: 0 },
      maxVal: 9,
      minLoc: { x: 1, y: 0 },
      minVal: 2,
    });
    expect(client.trace(extremaSource)).toEqual([14, 0, 0, 0]);
    extremaSource.dispose();
  });

  test("matches the exact four-lane trace contract", () => {
    expect(client.trace.bind(client)).toHaveLength(1);

    const source = client.matFromU8(
      2,
      3,
      4,
      new Uint8Array([
        1, 10, 100, 200, 2, 20, 101, 201, 3, 30, 102, 202, 4, 40, 103, 203, 5, 50, 104, 204, 6, 60,
        105, 205,
      ]),
    );
    expect(client.trace(source)).toEqual([6, 60, 204, 404]);

    const typedEmpty = client.matFromF64(0, 3, 4, new Float64Array());
    expect(client.trace(typedEmpty)).toEqual([0, 0, 0, 0]);

    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.trace, client, []);
    }).toThrow(BindingError);
    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.trace, client, [source, source]);
    }).toThrow(BindingError);

    typedEmpty.dispose();
    source.dispose();
  });

  test("matches mean and minMaxLoc overload, mask, and empty contracts", () => {
    expect(client.mean.bind(client)).toHaveLength(0);
    expect(client.minMaxLoc.bind(client)).toHaveLength(0);

    const source = client.matFromU8(1, 2, 3, new Uint8Array([1, 10, 100, 3, 30, 200]));
    const firstPixelMask = client.matFromU8(1, 2, 1, new Uint8Array([1, 0]));
    expect(client.mean(source, firstPixelMask)).toEqual([1, 10, 100, 0]);

    const extrema = client.matFromU8(2, 3, 1, new Uint8Array([5, 2, 9, 2, 9, 4]));
    const selectiveMask = client.matFromU8(2, 3, 1, new Uint8Array([0, 1, 0, 1, 0, 1]));
    expect(client.minMaxLoc(extrema, selectiveMask)).toEqual({
      maxLoc: { x: 2, y: 1 },
      maxVal: 4,
      minLoc: { x: 1, y: 0 },
      minVal: 2,
    });
    const zeroMask = client.zerosU8(2, 3, 1);
    expect(client.minMaxLoc(extrema, zeroMask)).toEqual({
      maxLoc: { x: -1, y: -1 },
      maxVal: 0,
      minLoc: { x: -1, y: -1 },
      minVal: 0,
    });

    const empty = client.emptyMat();
    expect(client.mean(empty)).toEqual([0, 0, 0, 0]);
    expect(client.minMaxLoc(empty)).toEqual({
      maxLoc: { x: 0, y: 0 },
      maxVal: 0,
      minLoc: { x: 0, y: 0 },
      minVal: 0,
    });
    const typedEmpty = client.matFromF32(0, 3, 1, new Float32Array());
    expect(client.minMaxLoc(typedEmpty)).toEqual({
      maxLoc: { x: -1, y: -1 },
      maxVal: 0,
      minLoc: { x: -1, y: -1 },
      minVal: 0,
    });

    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.mean, client, []);
    }).toThrow(BindingError);
    expect(() => {
      // oxlint-disable-next-line anti-slop/no-reflect-apply, typescript/unbound-method -- The test exercises invalid JavaScript arity.
      Reflect.apply(client.minMaxLoc, client, [extrema, selectiveMask, zeroMask]);
    }).toThrow(BindingError);

    typedEmpty.dispose();
    empty.dispose();
    zeroMask.dispose();
    selectiveMask.dispose();
    extrema.dispose();
    firstPixelMask.dispose();
    source.dispose();
  });

  test("mutates shared matrix destinations", () => {
    const matrix = client.matFromU8(2, 3, 1, new Uint8Array([1, 2, 3, 4, 5, 6]));
    matrix.copyFromBytes(new Uint8Array([6, 5, 4, 3, 2, 1]));
    expect(matrix.toUint8Array()).toEqual(new Uint8Array([6, 5, 4, 3, 2, 1]));
    expect(() => matrix.copyFromBytes(new Uint8Array([1]))).toThrow(OpenCvInputError);
    matrix.dispose();
  });

  test("splits, merges, extracts, and inserts channels", () => {
    const source = client.matFromU8(1, 2, 3, new Uint8Array([1, 10, 100, 2, 20, 200]));
    const planes = client.split(source);
    expect(planes.map((plane) => plane.toUint8Array())).toEqual([
      new Uint8Array([1, 2]),
      new Uint8Array([10, 20]),
      new Uint8Array([100, 200]),
    ]);
    const merged = client.merge([planes[0]!, planes[1]!, planes[2]!]);
    expect(merged.toUint8Array()).toEqual(source.toUint8Array());
    const extracted = client.extractChannel(source, 1);
    expect(extracted.toUint8Array()).toEqual(new Uint8Array([10, 20]));
    const destination = client.zerosU8(1, 2, 3);
    client.insertChannel(extracted, destination, 2);
    expect(destination.toUint8Array()).toEqual(new Uint8Array([0, 0, 10, 0, 0, 20]));
    const routed = client.matFromU8(1, 2, 3, new Uint8Array([7, 8, 9, 70, 80, 90]));
    client.mixChannels(source, routed, new Uint16Array([2, 0, 0, 2]));
    expect(routed.toUint8Array()).toEqual(new Uint8Array([100, 8, 1, 200, 80, 2]));
    routed.dispose();
    destination.dispose();
    extracted.dispose();
    merged.dispose();
    for (const plane of planes) plane.dispose();
    source.dispose();
  });

  test("runs floating-point math and cartesian conversions", () => {
    const x = client.matFromF32(1, 2, 1, new Float32Array([3, 0]));
    const y = client.matFromF32(1, 2, 1, new Float32Array([4, 2]));
    const exponential = client.expAlloc(x);
    const logarithm = client.logAlloc(exponential);
    const squareRoot = client.sqrtAlloc(y);
    const squared = client.powAlloc(x, 2);
    const vectorLength = client.magnitudeAlloc(x, y);
    expect(Array.from(exponential.toFloat32Array())).toEqual([Math.fround(Math.exp(3)), 1]);
    expect(Array.from(logarithm.toFloat32Array())[0]).toBeCloseTo(3, 5);
    expect(Array.from(squareRoot.toFloat32Array())).toEqual([2, Math.fround(Math.sqrt(2))]);
    expect(Array.from(squared.toFloat32Array())).toEqual([9, 0]);
    expect(Array.from(vectorLength.toFloat32Array())).toEqual([5, 2]);

    const lengths = client.zerosF32(1, 2, 1);
    const angles = client.zerosF32(1, 2, 1);
    client.cartToPolar(x, y, lengths, angles);
    const roundTripX = client.zerosF32(1, 2, 1);
    const roundTripY = client.zerosF32(1, 2, 1);
    client.polarToCart(lengths, angles, roundTripX, roundTripY);
    expect(Array.from(roundTripX.toFloat32Array())[0]).toBeCloseTo(3, 5);
    expect(Array.from(roundTripY.toFloat32Array())[0]).toBeCloseTo(4, 5);

    for (const matrix of [
      roundTripY,
      roundTripX,
      angles,
      lengths,
      vectorLength,
      squared,
      squareRoot,
      exponential,
      logarithm,
      y,
      x,
    ])
      matrix.dispose();
  });

  test("matches float-math destination call contracts", () => {
    const source = client.matFromF32(1, 2, 1, new Float32Array([1, 4]));
    const other = client.matFromF32(1, 2, 1, new Float32Array([2, 3]));
    const destination = client.zerosF32(1, 2, 1);

    expect(client.exp.length).toBe(2);
    expect(client.log.length).toBe(2);
    expect(client.sqrt.length).toBe(2);
    expect(client.pow.length).toBe(3);
    expect(client.magnitude.length).toBe(3);

    expect(() => {
      // @ts-expect-error Runtime parity requires testing missing arguments from plain JavaScript.
      client.exp();
    }).toThrow(new BindingError("function exp called with 0 arguments, expected 2 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a missing destination.
      client.log(source);
    }).toThrow(new BindingError("function log called with 1 arguments, expected 2 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an extra argument.
      client.sqrt(source, destination, destination);
    }).toThrow(new BindingError("function sqrt called with 3 arguments, expected 2 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing missing arguments from plain JavaScript.
      client.pow();
    }).toThrow(new BindingError("function pow called with 0 arguments, expected 3 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a missing destination.
      client.pow(source, 2);
    }).toThrow(new BindingError("function pow called with 2 arguments, expected 3 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an extra argument.
      client.pow(source, 2, destination, destination);
    }).toThrow(new BindingError("function pow called with 4 arguments, expected 3 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing missing arguments from plain JavaScript.
      client.magnitude();
    }).toThrow(new BindingError("function magnitude called with 0 arguments, expected 3 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a missing destination.
      client.magnitude(source, other);
    }).toThrow(new BindingError("function magnitude called with 2 arguments, expected 3 args!"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an extra argument.
      client.magnitude(source, other, destination, destination);
    }).toThrow(new BindingError("function magnitude called with 4 arguments, expected 3 args!"));

    expect(client.exp(source, destination)).toBeUndefined();
    expect(client.log(source, destination)).toBeUndefined();
    expect(client.sqrt(source, destination)).toBeUndefined();
    expect(client.pow(source, 2, destination)).toBeUndefined();
    expect(client.magnitude(source, other, destination)).toBeUndefined();
    expect(() => {
      // @ts-expect-error Runtime parity requires boolean-to-double conversion.
      client.pow(source, true, destination);
    }).not.toThrow();
    expect(() => {
      // @ts-expect-error Runtime parity requires testing Embind double rejection.
      client.pow(source, "2", destination);
    }).toThrow(new TypeError('Cannot convert "2" to double'));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a null Mat.
      client.exp(null, destination);
    }).toThrow(new BindingError("null is not a valid Mat"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an undefined Mat.
      client.log(undefined, destination);
    }).toThrow(new TypeError("Cannot read properties of undefined (reading '$$')"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a structural Mat.
      client.sqrt({}, destination);
    }).toThrow(new BindingError('Cannot pass "[object Object]" as a Mat'));

    const deleted = client.matFromF32(1, 1, 1, new Float32Array([1]));
    deleted.dispose();
    expect(() => {
      // @ts-expect-error Source conversion must fail before power conversion.
      client.pow(deleted, "bad", destination);
    }).toThrow(new BindingError("Cannot pass deleted object as a pointer of type Mat"));
    expect(() => {
      // @ts-expect-error Power conversion must fail before destination conversion.
      client.pow(source, "bad", null);
    }).toThrow(new TypeError('Cannot convert "bad" to double'));

    destination.dispose();
    other.dispose();
    source.dispose();
  });

  test("matches coordinate-conversion overloads and boolean coercion", () => {
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const first = localClient.matFromF32(1, 1, 1, new Float32Array([1]));
    const second = localClient.matFromF32(1, 1, 1, new Float32Array([0]));
    const firstOutput = localClient.zerosF32(1, 1, 1);
    const secondOutput = localClient.zerosF32(1, 1, 1);

    expect(localClient.cartToPolar.length).toBe(0);
    expect(localClient.polarToCart.length).toBe(0);
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a missing output.
      localClient.cartToPolar(first, second, firstOutput);
    }).toThrow(BindingError);
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an extra argument.
      localClient.cartToPolar(first, second, firstOutput, secondOutput, true, 1);
    }).toThrow(BindingError);
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a missing output.
      localClient.polarToCart(first, second, firstOutput);
    }).toThrow(BindingError);
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an extra argument.
      localClient.polarToCart(first, second, firstOutput, secondOutput, true, 1);
    }).toThrow(BindingError);
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a null Mat.
      localClient.cartToPolar(null, second, firstOutput, secondOutput);
    }).toThrow(new BindingError("null is not a valid Mat"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing an undefined Mat.
      localClient.polarToCart(first, undefined, firstOutput, secondOutput);
    }).toThrow(new TypeError("Cannot read properties of undefined (reading '$$')"));
    expect(() => {
      // @ts-expect-error Runtime parity requires testing a structural Mat.
      localClient.cartToPolar(first, second, {}, secondOutput);
    }).toThrow(new BindingError('Cannot pass "[object Object]" as a Mat'));

    localClient.cartToPolar(first, second, firstOutput, secondOutput);
    // @ts-expect-error Runtime parity requires JavaScript boolean coercion.
    localClient.cartToPolar(first, second, firstOutput, secondOutput, "degrees");
    localClient.polarToCart(first, second, firstOutput, secondOutput);
    // @ts-expect-error Runtime parity requires JavaScript boolean coercion.
    localClient.polarToCart(first, second, firstOutput, secondOutput, 0);
    expect(backend.cartToPolarDegreeFlags).toEqual([false, true]);
    expect(backend.polarToCartDegreeFlags).toEqual([false, false]);

    secondOutput.dispose();
    firstOutput.dispose();
    second.dispose();
    first.dispose();
  });

  test("concatenates matrices in both axes", () => {
    const left = client.matFromU8(2, 1, 1, new Uint8Array([1, 2]));
    const right = client.matFromU8(2, 2, 1, new Uint8Array([3, 4, 5, 6]));
    const horizontal = client.hconcat([left, right]);
    expect(horizontal.rows).toBe(2);
    expect(horizontal.columns).toBe(3);
    expect(horizontal.toUint8Array()).toEqual(new Uint8Array([1, 3, 4, 2, 5, 6]));

    const top = client.matFromU8(1, 2, 1, new Uint8Array([7, 8]));
    const bottom = client.matFromU8(2, 2, 1, new Uint8Array([9, 10, 11, 12]));
    const vertical = client.vconcat([top, bottom]);
    expect(vertical.rows).toBe(3);
    expect(vertical.columns).toBe(2);
    expect(vertical.toUint8Array()).toEqual(new Uint8Array([7, 8, 9, 10, 11, 12]));
    for (const matrix of [vertical, bottom, top, horizontal, right, left]) matrix.dispose();
  });

  test("runs scaled all-depth numeric operations", () => {
    const left = client.matFromU8(1, 3, 1, new Uint8Array([10, 20, 250]));
    const right = client.matFromU8(1, 3, 1, new Uint8Array([2, 0, 2]));
    const product = client.multiplyAlloc(left, right, 0.5);
    const quotient = client.divideAlloc(left, right, 2);
    const blended = client.addWeightedAlloc(left, 0.5, right, 0.5, 1);
    const absolute = client.convertScaleAbsAlloc(left, -1, 5);
    const destination = client.zerosU8(1, 3, 1);
    client.multiply(left, right, destination, 0.5);
    expect(client.multiply.length).toBe(0);
    expect(client.divide.length).toBe(0);
    expect(client.addWeighted.length).toBe(0);
    expect(client.convertScaleAbs.length).toBe(0);
    expect(product.toUint8Array()).toEqual(new Uint8Array([10, 0, 250]));
    expect(quotient.toUint8Array()).toEqual(new Uint8Array([10, 0, 250]));
    expect(blended.toUint8Array()).toEqual(new Uint8Array([7, 11, 127]));
    expect(absolute.toUint8Array()).toEqual(new Uint8Array([5, 15, 245]));
    expect(destination.toUint8Array()).toEqual(product.toUint8Array());
    for (const matrix of [destination, absolute, blended, quotient, product, right, left])
      matrix.dispose();
  });

  test("matches numeric destination overloads and dtype forwarding", () => {
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const left = localClient.matFromU8(1, 1, 1, new Uint8Array([6]));
    const right = localClient.matFromU8(1, 1, 1, new Uint8Array([3]));
    const destination = localClient.zerosU8(1, 1, 1);

    localClient.multiply(left, right, destination);
    localClient.multiply(left, right, destination, 0.5, 6);
    localClient.divide(left, right, destination, 2, 5);
    localClient.addWeighted(left, 0.25, right, 0.75, 1, destination, 4);

    expect(backend.numericIntoCalls).toEqual([
      { method: "multiply", scale: 1, dtype: -1 },
      { method: "multiply", scale: 0.5, dtype: 6 },
      { method: "divide", scale: 2, dtype: 5 },
      { method: "addWeighted", scale: 1, dtype: 4 },
    ]);
    // @ts-expect-error Runtime parity rejects missing destination arguments.
    expect(() => localClient.multiply(left, right)).toThrow(BindingError);
    // @ts-expect-error Runtime parity rejects missing destination arguments.
    expect(() => localClient.convertScaleAbs(left)).toThrow(BindingError);
    expect(() => localClient.multiply(left, right, destination, undefined)).toThrow(TypeError);
    expect(() => localClient.divide(left, right, destination, undefined)).toThrow(TypeError);
    expect(() => localClient.addWeighted(left, 1, right, 1, 0, destination, undefined)).toThrow(
      TypeError,
    );
    expect(() => localClient.convertScaleAbs(left, destination, undefined)).toThrow(TypeError);

    destination.dispose();
    right.dispose();
    left.dispose();
  });

  test("adds typed matrix borders", () => {
    const source = client.matFromU8(1, 2, 1, new Uint8Array([7, 8]));
    const bordered = client.copyMakeBorder(source, 1, 1, 1, 1, 0, [9, 0, 0, 0]);
    expect(bordered.rows).toBe(3);
    expect(bordered.columns).toBe(4);
    expect(bordered.toUint8Array()).toEqual(new Uint8Array([9, 9, 9, 9, 9, 7, 8, 9, 9, 9, 9, 9]));
    bordered.dispose();
    source.dispose();
  });

  test("applies a typed lookup table", () => {
    const values = Uint8Array.from({ length: 256 }, (_, value) => value);
    values[1] = 99;
    const table = client.matFromU8(256, 1, 1, values);
    const source = client.matFromU8(1, 3, 1, new Uint8Array([0, 1, 255]));
    const result = client.lut(source, table);
    expect(result.toUint8Array()).toEqual(new Uint8Array([0, 99, 255]));
    const destination = client.zerosU8(1, 3, 1);
    client.lut(source, table, destination);
    expect(destination.toUint8Array()).toEqual(result.toUint8Array());
    destination.dispose();
    result.dispose();
    source.dispose();
    table.dispose();
  });

  test("computes norms and normalizes into destinations", () => {
    const source = client.matFromU8(1, 2, 1, new Uint8Array([3, 4]));
    expect(client.norm(source, 4)).toBe(5);
    const other = client.matFromU8(1, 2, 1, new Uint8Array([0, 0]));
    expect(client.norm(source, other, 2)).toBe(7);
    const destination = client.zerosU8(1, 2, 1);
    client.normalize(source, destination, 10, 0, 4);
    expect(destination.toUint8Array()).toEqual(new Uint8Array([6, 8]));
    destination.dispose();
    other.dispose();
    source.dispose();
  });

  test("writes channel statistics and dimensional reductions", () => {
    const source = client.matFromU8(2, 2, 1, new Uint8Array([1, 3, 5, 7]));
    const means = client.zerosF64(1, 1, 1);
    const deviations = client.zerosF64(1, 1, 1);
    client.meanStdDev(source, means, deviations);
    expect(means.toFloat64Array()[0]).toBe(4);
    expect(deviations.toFloat64Array()[0]).toBeCloseTo(Math.sqrt(5), 12);
    const reduced = client.zerosU8(1, 2, 1);
    client.reduce(source, reduced, 0, 0);
    expect(reduced.toUint8Array()).toEqual(new Uint8Array([6, 10]));
    reduced.dispose();
    deviations.dispose();
    means.dispose();
    source.dispose();
  });

  test("owns AKAZE configuration handles with documented defaults", () => {
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const akaze = localClient.createAKAZE();

    expect(akaze.getDefaultName()).toBe("Feature2D.AKAZE");
    expect(akaze.getDescriptorType()).toBe(AKAZE_DescriptorType.DESCRIPTOR_MLDB);
    expect(akaze.getDescriptorSize()).toBe(AKAZE_DEFAULTS.descriptorSize);
    expect(akaze.getDescriptorChannels()).toBe(AKAZE_DEFAULTS.descriptorChannels);
    expect(akaze.getThreshold()).toBeCloseTo(AKAZE_DEFAULTS.threshold, 9);
    expect(akaze.getNOctaves()).toBe(AKAZE_DEFAULTS.octaves);
    expect(akaze.getNOctaveLayers()).toBe(AKAZE_DEFAULTS.octaveLayers);
    expect(akaze.getDiffusivity()).toBe(KAZE_DiffusivityType.DIFF_PM_G2);

    akaze.dispose();
    expect(backend.akazeFreeCount).toBe(1);
    akaze.dispose();
    expect(backend.akazeFreeCount).toBe(1);
    expect(() => akaze.getThreshold()).toThrow(BindingError);
  });

  test("owns an AgastFeatureDetector configuration with OpenCV defaults", () => {
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const detector = localClient.createAgastFeatureDetector();

    expect(detector.getDefaultName()).toBe("Feature2D.AgastFeatureDetector");
    expect(detector.getThreshold()).toBe(AGAST_FEATURE_DETECTOR_DEFAULTS.threshold);
    expect(detector.getNonmaxSuppression()).toBe(AGAST_FEATURE_DETECTOR_DEFAULTS.nonmaxSuppression);
    expect(detector.getType()).toBe(AgastFeatureDetector_DetectorType.OAST_9_16);

    detector.setThreshold(-1);
    detector.setNonmaxSuppression(false);
    detector.setType(AgastFeatureDetector_DetectorType.AGAST_7_12s);
    expect(detector.getThreshold()).toBe(-1);
    expect(detector.getNonmaxSuppression()).toBe(false);
    expect(detector.getType()).toBe(AgastFeatureDetector_DetectorType.AGAST_7_12s);

    detector.dispose();
    detector.dispose();
    expect(backend.agastFeatureDetectorFreeCount).toBe(1);
    expect(() => detector.getThreshold()).toThrow(BindingError);
  });

  test("validates AGAST construction and coerces instance thresholds", () => {
    const localClient = createOpenCv(new CopyingBackend());

    expect(() => localClient.createAgastFeatureDetector({ threshold: -2_147_483_649 })).toThrow(
      OpenCvInputError,
    );
    expect(() => localClient.createAgastFeatureDetector({ threshold: 2_147_483_648 })).toThrow(
      OpenCvInputError,
    );
    const detector = localClient.createAgastFeatureDetector({ threshold: -2_147_483_648 });
    expect(detector.getThreshold()).toBe(-2_147_483_648);
    detector.setThreshold(2_147_483_647);
    expect(detector.getThreshold()).toBe(2_147_483_647);
    expect(detector.setThreshold(1.5)).toBeUndefined();
    expect(detector.getThreshold()).toBe(1);
    detector.dispose();
  });

  test("owns a FastFeatureDetector configuration with OpenCV defaults", () => {
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const detector = localClient.createFastFeatureDetector();

    expect(detector.getDefaultName()).toBe("Feature2D.FastFeatureDetector");
    expect(detector.getThreshold()).toBe(FAST_FEATURE_DETECTOR_DEFAULTS.threshold);
    expect(detector.getNonmaxSuppression()).toBe(FAST_FEATURE_DETECTOR_DEFAULTS.nonmaxSuppression);
    expect(detector.getType()).toBe(FastFeatureDetector_DetectorType.TYPE_9_16);

    detector.setThreshold(256);
    detector.setNonmaxSuppression(false);
    detector.setType(FastFeatureDetector_DetectorType.TYPE_7_12);
    expect(detector.getThreshold()).toBe(256);
    expect(detector.getNonmaxSuppression()).toBe(false);
    expect(detector.getType()).toBe(FastFeatureDetector_DetectorType.TYPE_7_12);

    detector.dispose();
    detector.dispose();
    expect(backend.fastFeatureDetectorFreeCount).toBe(1);
    expect(() => detector.setNonmaxSuppression(true)).toThrow(BindingError);
  });

  test("validates FAST construction and coerces instance thresholds", () => {
    const localClient = createOpenCv(new CopyingBackend());

    expect(() => localClient.createFastFeatureDetector({ threshold: -2_147_483_649 })).toThrow(
      OpenCvInputError,
    );
    expect(() => localClient.createFastFeatureDetector({ threshold: 2_147_483_648 })).toThrow(
      OpenCvInputError,
    );
    const detector = localClient.createFastFeatureDetector({ threshold: -2_147_483_648 });
    expect(detector.getThreshold()).toBe(-2_147_483_648);
    detector.setThreshold(2_147_483_647);
    expect(detector.getThreshold()).toBe(2_147_483_647);
    expect(detector.setThreshold(Number.NaN)).toBeUndefined();
    expect(detector.getThreshold()).toBe(0);
    detector.dispose();
  });

  test("creates and mutates an explicit AKAZE configuration", () => {
    const localClient = createOpenCv(new CopyingBackend());
    const akaze = localClient.createAKAZE({
      descriptorType: AKAZEDescriptorType.MLDB_UPRIGHT,
      descriptorSize: 96,
      descriptorChannels: 2,
      threshold: 0.05,
      octaves: 5,
      octaveLayers: 6,
      diffusivity: KAZEDiffusivity.WEICKERT,
      maxPoints: 300,
    });

    expect(akaze.getDescriptorType()).toBe(AKAZE_DescriptorType.DESCRIPTOR_MLDB_UPRIGHT);
    expect(akaze.getDescriptorSize()).toBe(96);
    expect(akaze.getDescriptorChannels()).toBe(2);
    expect(akaze.getThreshold()).toBe(0.05);
    expect(akaze.getNOctaves()).toBe(5);
    expect(akaze.getNOctaveLayers()).toBe(6);
    expect(akaze.getDiffusivity()).toBe(KAZE_DiffusivityType.DIFF_WEICKERT);

    akaze.setDescriptorType(AKAZE_DescriptorType.DESCRIPTOR_KAZE);
    akaze.setDescriptorSize(128);
    akaze.setDescriptorChannels(3);
    akaze.setThreshold(0.1);
    akaze.setNOctaves(7);
    akaze.setNOctaveLayers(8);
    akaze.setDiffusivity(KAZE_DiffusivityType.DIFF_CHARBONNIER);
    expect(akaze.getDescriptorType()).toBe(AKAZE_DescriptorType.DESCRIPTOR_KAZE);
    expect(akaze.getDescriptorSize()).toBe(128);
    expect(akaze.getDescriptorChannels()).toBe(3);
    expect(akaze.getThreshold()).toBe(0.1);
    expect(akaze.getNOctaves()).toBe(7);
    expect(akaze.getNOctaveLayers()).toBe(8);
    expect(akaze.getDiffusivity()).toBe(KAZE_DiffusivityType.DIFF_CHARBONNIER);

    expect(akaze.setDescriptorSize(-1)).toBeUndefined();
    expect(akaze.getDescriptorSize()).toBe(-1);
    akaze.dispose();
    expect(() => akaze.setThreshold(0.2)).toThrow(BindingError);
    expect(() => localClient.createAKAZE({ descriptorSize: -1 })).toThrow(OpenCvInputError);
  });

  test("owns a GFTTDetector configuration with OpenCV 4.13 defaults", () => {
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const detector = localClient.createGFTTDetector();

    expect(detector.getDefaultName()).toBe("Feature2D.GFTTDetector");
    expect(detector.getMaxFeatures()).toBe(GFTT_DETECTOR_DEFAULTS.maxFeatures);
    expect(detector.getQualityLevel()).toBe(GFTT_DETECTOR_DEFAULTS.qualityLevel);
    expect(detector.getMinDistance()).toBe(GFTT_DETECTOR_DEFAULTS.minDistance);
    expect(detector.getBlockSize()).toBe(GFTT_DETECTOR_DEFAULTS.blockSize);
    expect(detector.getHarrisDetector()).toBe(GFTT_DETECTOR_DEFAULTS.useHarrisDetector);
    expect(detector.getK()).toBe(GFTT_DETECTOR_DEFAULTS.k);

    detector.dispose();
    detector.dispose();
    expect(backend.gfttDetectorFreeCount).toBe(1);
    expect(() => detector.getMaxFeatures()).toThrow(BindingError);
    expect(() => detector.setHarrisDetector(true)).toThrow(BindingError);
  });

  test("creates and mutates the full GFTTDetector configuration", () => {
    const localClient = createOpenCv(new CopyingBackend());
    const detector = localClient.createGFTTDetector({
      blockSize: -1,
      k: -1,
      maxFeatures: -1,
      minDistance: -1,
      qualityLevel: -1,
      useHarrisDetector: true,
    });

    expect(detector.getBlockSize()).toBe(-1);
    expect(detector.getK()).toBe(-1);
    expect(detector.getMaxFeatures()).toBe(-1);
    expect(detector.getMinDistance()).toBe(-1);
    expect(detector.getQualityLevel()).toBe(-1);
    expect(detector.getHarrisDetector()).toBe(true);

    detector.setBlockSize(-2_147_483_648);
    detector.setK(Number.NaN);
    detector.setMaxFeatures(2_147_483_647);
    detector.setMinDistance(Number.NEGATIVE_INFINITY);
    detector.setQualityLevel(Number.POSITIVE_INFINITY);
    detector.setHarrisDetector(false);
    expect(detector.getBlockSize()).toBe(-2_147_483_648);
    expect(detector.getK()).toBeNaN();
    expect(detector.getMaxFeatures()).toBe(2_147_483_647);
    expect(detector.getMinDistance()).toBe(Number.NEGATIVE_INFINITY);
    expect(detector.getQualityLevel()).toBe(Number.POSITIVE_INFINITY);
    expect(detector.getHarrisDetector()).toBe(false);
    detector.dispose();
  });

  test("validates GFTTDetector constructor integers and coerces instance setters", () => {
    const localClient = createOpenCv(new CopyingBackend());

    expect(() => localClient.createGFTTDetector({ maxFeatures: -2_147_483_649 })).toThrow(
      OpenCvInputError,
    );
    expect(() => localClient.createGFTTDetector({ blockSize: 2_147_483_648 })).toThrow(
      OpenCvInputError,
    );
    expect(() => localClient.createGFTTDetector({ blockSize: 1.5 })).toThrow(OpenCvInputError);
    const detector = localClient.createGFTTDetector();
    expect(detector.setMaxFeatures(Number.NaN)).toBeUndefined();
    expect(() => detector.setBlockSize(Number.POSITIVE_INFINITY)).toThrow(
      new TypeError(
        'Passing a number "Infinity" from JS side to C/C++ side to an argument of type "int", which is outside the valid range [-2147483648, 2147483647]!',
      ),
    );
    expect(detector.getMaxFeatures()).toBe(0);
    expect(detector.getBlockSize()).toBe(GFTT_DETECTOR_DEFAULTS.blockSize);
    detector.dispose();
  });

  test("creates a Rust-owned MSER configuration without exposing a static factory", () => {
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const detector = localClient.createMSER({
      delta: 7,
      minArea: 61,
      maxArea: 14_401,
      pass2Only: true,
    });

    expect(Reflect.has(MSER, "create")).toBe(false);
    expect(detector.getDefaultName()).toBe("Feature2D.MSER");
    expect(detector.getDelta()).toBe(7);
    expect(detector.getMinArea()).toBe(61);
    expect(detector.getMaxArea()).toBe(14_401);
    expect(detector.getPass2Only()).toBe(true);

    detector.dispose();
    detector.dispose();
    expect(backend.mserFreeCount).toBe(1);
  });

  test("uses MSER defaults and validates convenience-factory integers", () => {
    const localClient = createOpenCv(new CopyingBackend());
    const detector = localClient.createMSER();

    expect(detector.getDelta()).toBe(MSER_DEFAULTS.delta);
    expect(detector.getMinArea()).toBe(MSER_DEFAULTS.minArea);
    expect(detector.getMaxArea()).toBe(MSER_DEFAULTS.maxArea);
    expect(detector.getPass2Only()).toBe(MSER_DEFAULTS.pass2Only);
    detector.dispose();

    expect(() => localClient.createMSER({ delta: 1.5 })).toThrow(OpenCvInputError);
    expect(() => localClient.createMSER({ minArea: -2_147_483_649 })).toThrow(OpenCvInputError);
    expect(() => localClient.createMSER({ maxArea: 2_147_483_648 })).toThrow(OpenCvInputError);
  });

  test("creates all tone-mapping state families from their pinned constructor defaults", () => {
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const drago = localClient.createTonemapDrago();
    const mantiuk = localClient.createTonemapMantiuk();
    const reinhard = localClient.createTonemapReinhard();

    expect([drago.getGamma(), drago.getSaturation(), drago.getBias()]).toEqual([
      1,
      1,
      Math.fround(0.85),
    ]);
    expect([mantiuk.getGamma(), mantiuk.getScale(), mantiuk.getSaturation()]).toEqual([
      1,
      Math.fround(0.7),
      1,
    ]);
    expect([
      reinhard.getGamma(),
      reinhard.getIntensity(),
      reinhard.getLightAdaptation(),
      reinhard.getColorAdaptation(),
    ]).toEqual([1, 0, 1, 0]);

    drago.delete();
    mantiuk.delete();
    reinhard.delete();
    expect(backend.tonemapFreeCount).toBe(3);
  });

  // oxlint-disable anti-slop/no-reflect-apply, typescript/unbound-method -- This test exercises untyped JavaScript constructor-call shapes.
  test("matches concrete Tonemap constructor arity and float coercion", () => {
    const localClient = createOpenCv(new CopyingBackend());

    expect(localClient.createTonemapDrago).toHaveLength(0);
    expect(localClient.createTonemapMantiuk).toHaveLength(0);
    expect(localClient.createTonemapReinhard).toHaveLength(0);

    const drago = Reflect.apply(localClient.createTonemapDrago, localClient, [true, false, 1e40]);
    const mantiuk = localClient.createTonemapMantiuk(0.25, 1.25, 2.25);
    const reinhard = localClient.createTonemapReinhard(0.25, 1.25, 2.25, 3.25);
    expect([drago.getGamma(), drago.getSaturation(), drago.getBias()]).toEqual([
      1,
      0,
      Number.POSITIVE_INFINITY,
    ]);
    expect([mantiuk.getGamma(), mantiuk.getScale(), mantiuk.getSaturation()]).toEqual([
      0.25, 1.25, 2.25,
    ]);
    expect([
      reinhard.getGamma(),
      reinhard.getIntensity(),
      reinhard.getLightAdaptation(),
      reinhard.getColorAdaptation(),
    ]).toEqual([0.25, 1.25, 2.25, 3.25]);

    expect(() => Reflect.apply(localClient.createTonemapDrago, localClient, [undefined])).toThrow(
      new TypeError('Cannot convert "undefined" to float'),
    );
    expect(() =>
      Reflect.apply(localClient.createTonemapMantiuk, localClient, [1, 2, 3, 4]),
    ).toThrow(BindingError);
    expect(() =>
      Reflect.apply(localClient.createTonemapReinhard, localClient, [1, 2, 3, 4, 5]),
    ).toThrow(BindingError);

    drago.delete();
    mantiuk.delete();
    reinhard.delete();
  });
  // oxlint-enable anti-slop/no-reflect-apply, typescript/unbound-method

  test("owns a KAZE configuration with OpenCV 4.13 defaults", () => {
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const kaze = localClient.createKAZE();

    expect(kaze.getDefaultName()).toBe("Feature2D.KAZE");
    expect(kaze.getExtended()).toBe(KAZE_DEFAULTS.extended);
    expect(kaze.getUpright()).toBe(KAZE_DEFAULTS.upright);
    expect(kaze.getThreshold()).toBe(0.0010000000474974513);
    expect(kaze.getNOctaves()).toBe(KAZE_DEFAULTS.octaves);
    expect(kaze.getNOctaveLayers()).toBe(KAZE_DEFAULTS.octaveLayers);
    expect(kaze.getDiffusivity()).toBe(KAZE_DiffusivityType.DIFF_PM_G2);

    kaze.dispose();
    expect(backend.kazeFreeCount).toBe(1);
    kaze.dispose();
    expect(backend.kazeFreeCount).toBe(1);
    expect(() => kaze.getThreshold()).toThrow(BindingError);
  });

  test("owns an ORB configuration with the pinned constructor defaults", () => {
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const orb = localClient.createORB();

    expect(orb.getDefaultName()).toBe("Feature2D.ORB");
    expect(orb.getFastThreshold()).toBe(ORB_DEFAULTS.fastThreshold);
    expect(ORB_ScoreType.HARRIS_SCORE.value).toBe(0);
    expect(ORB_ScoreType.FAST_SCORE.value).toBe(1);

    orb.dispose();
    expect(backend.orbFreeCount).toBe(1);
  });

  // oxlint-disable anti-slop/no-reflect-apply, typescript/unbound-method -- This test exercises untyped JavaScript Embind call shapes.
  test("matches ORB method arity, coercion, enum, and lifetime contracts", () => {
    const backend = new CopyingBackend();
    const localClient = createOpenCv(backend);
    const orb = localClient.createORB({ fastThreshold: 20, scoreType: ORBScoreType.HARRIS_SCORE });

    expect(orb.getDefaultName).toHaveLength(0);
    expect(orb.getFastThreshold).toHaveLength(0);
    for (const setter of [
      orb.setEdgeThreshold,
      orb.setFastThreshold,
      orb.setFirstLevel,
      orb.setMaxFeatures,
      orb.setNLevels,
      orb.setPatchSize,
      orb.setScaleFactor,
      orb.setScoreType,
      orb.setWTA_K,
    ]) {
      expect(setter).toHaveLength(1);
      expect(() => Reflect.apply(setter, orb, [])).toThrow(BindingError);
      expect(() => Reflect.apply(setter, orb, [1, 2])).toThrow(BindingError);
    }
    expect(() => Reflect.apply(orb.getFastThreshold, orb, [1])).toThrow(BindingError);

    expect(orb.setFastThreshold(37.9)).toBeUndefined();
    expect(orb.getFastThreshold()).toBe(37);
    orb.setFastThreshold(Number.NaN);
    expect(orb.getFastThreshold()).toBe(0);
    expect(() => orb.setFastThreshold(Number.POSITIVE_INFINITY)).toThrow(TypeError);
    expect(() => Reflect.apply(orb.setFastThreshold, orb, [null])).toThrow(TypeError);
    expect(orb.getFastThreshold()).toBe(0);

    expect(orb.setScaleFactor(Number.NEGATIVE_INFINITY)).toBeUndefined();
    expect(() => Reflect.apply(orb.setScaleFactor, orb, ["1.2"])).toThrow(TypeError);
    expect(() => orb.setFirstLevel(-1)).toThrow(OpenCvInputError);
    expect(orb.setFirstLevel(2)).toBeUndefined();

    expect(orb.setScoreType(ORB_ScoreType.FAST_SCORE)).toBeUndefined();
    expect(Reflect.apply(orb.setScoreType, orb, [{ value: 19.9 }])).toBeUndefined();
    expect(Reflect.apply(orb.setScoreType, orb, [1])).toBeUndefined();
    expect(() => Reflect.apply(orb.setScoreType, orb, [null])).toThrow(TypeError);

    expect(ORB_HARRIS_SCORE).toBe(0);
    expect(ORB_FAST_SCORE).toBe(1);
    expect(localClient.ORB_HARRIS_SCORE).toBe(0);
    expect(localClient.ORB_FAST_SCORE).toBe(1);
    expect(localClient.ORB_ScoreType).toBe(ORB_ScoreType);
    expect(ORB_ScoreType.values[0]).toBe(ORB_ScoreType.HARRIS_SCORE);
    expect(ORB_ScoreType.values[1]).toBe(ORB_ScoreType.FAST_SCORE);
    expect(ORB_ScoreType.HARRIS_SCORE).not.toBe(ORB_HARRIS_SCORE);

    expect(Reflect.apply(orb.delete, orb, [1, 2])).toBeUndefined();
    expect(backend.orbFreeCount).toBe(1);
    expect(() => orb.delete()).toThrow("ORB instance already deleted");
    expect(() => orb.getDefaultName()).toThrow(
      "Cannot pass deleted object as a pointer of type ORB",
    );
    expect(() => orb.getFastThreshold()).toThrow(
      "Cannot pass deleted object as a pointer of type ORB const*",
    );
    expect(() => orb.setFastThreshold(1)).toThrow(
      "Cannot pass deleted object as a pointer of type ORB",
    );
  });
  // oxlint-enable anti-slop/no-reflect-apply, typescript/unbound-method

  test("creates and mutates an explicit KAZE configuration", () => {
    const localClient = createOpenCv(new CopyingBackend());
    const kaze = localClient.createKAZE({
      diffusivity: KAZEDiffusivity.WEICKERT,
      extended: true,
      octaveLayers: 6,
      octaves: 5,
      threshold: -1,
      upright: true,
    });

    expect(kaze.getExtended()).toBe(true);
    expect(kaze.getUpright()).toBe(true);
    expect(kaze.getThreshold()).toBe(-1);
    expect(kaze.getNOctaves()).toBe(5);
    expect(kaze.getNOctaveLayers()).toBe(6);
    expect(kaze.getDiffusivity()).toBe(KAZE_DiffusivityType.DIFF_WEICKERT);

    kaze.setExtended(false);
    kaze.setUpright(false);
    kaze.setThreshold(-0.25);
    kaze.setNOctaves(7);
    kaze.setNOctaveLayers(8);
    kaze.setDiffusivity(KAZE_DiffusivityType.DIFF_CHARBONNIER);
    expect(kaze.getExtended()).toBe(false);
    expect(kaze.getUpright()).toBe(false);
    expect(kaze.getThreshold()).toBe(-0.25);
    expect(kaze.getNOctaves()).toBe(7);
    expect(kaze.getNOctaveLayers()).toBe(8);
    expect(kaze.getDiffusivity()).toBe(KAZE_DiffusivityType.DIFF_CHARBONNIER);

    kaze.dispose();
    expect(() => kaze.setExtended(true)).toThrow(BindingError);
  });

  test("rejects invalid KAZE configuration before calling WASM", () => {
    const localClient = createOpenCv(new CopyingBackend());

    expect(() => localClient.createKAZE({ threshold: Number.NaN })).toThrow(OpenCvInputError);
    expect(() => localClient.createKAZE({ octaves: 0 })).toThrow(OpenCvInputError);
    expect(() => localClient.createKAZE({ octaveLayers: 2_147_483_648 })).toThrow(OpenCvInputError);
    const kaze = localClient.createKAZE();
    kaze.setNOctaves(1.5);
    kaze.setThreshold(Number.POSITIVE_INFINITY);
    expect(kaze.getDiffusivity()).toBe(KAZE_DiffusivityType.DIFF_PM_G2);
    expect(kaze.getNOctaves()).toBe(1);
    expect(kaze.getThreshold()).toBe(Number.POSITIVE_INFINITY);
    kaze.dispose();
  });
});

describe("approxPolyDP binding", () => {
  test("forwards finite numeric epsilon, boolean closure and matrix destinations", () => {
    const backend = new CopyingBackend();
    const cv = createOpenCv(backend);
    const source = cv.matFromI32(3, 1, 2, new Int32Array([0, 0, 1, 2, 4, 0]));
    const destination = cv.emptyMat();
    cv.approxPolyDP(source, destination, 0.5, true);
    expect(backend.approximationCalls).toEqual([{ epsilon: 0.5, closed: true }]);
    expect(destination.toInt32Array()).toEqual(source.toInt32Array());
    expect(cv.approxPolyDP.length).toBe(4);
    // @ts-expect-error exercise pinned binding arity rejection
    expect(() => cv.approxPolyDP(source, destination, 1)).toThrow(BindingError);
    source.dispose();
    expect(() => cv.approxPolyDP(source, destination, 1, false)).toThrow();
    destination.dispose();
  });
});

describe("perspective bindings", () => {
  test("forwards warp defaults, converts integer fields, and checks arity and lifetimes", () => {
    const backend = new CopyingBackend();
    const cv = createOpenCv(backend);
    const source = cv.matFromU8(1, 1, 1, new Uint8Array([5]));
    const destination = cv.emptyMat();
    const transform = cv.matFromF64(3, 3, 1, new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
    cv.warpPerspective(source, destination, transform, { width: 3.9, height: 2.1 });
    expect(backend.perspectiveCalls).toEqual([{ flags: 1, borderType: 0, width: 3, height: 2 }]);
    expect(cv.warpPerspective.length).toBe(0);
    expect(cv.getPerspectiveTransform.length).toBe(0);
    expect([cv.DECOMP_LU, cv.DECOMP_QR, cv.DECOMP_NORMAL]).toEqual([0, 4, 16]);
    // @ts-expect-error test binding arity rejection
    expect(() => cv.warpPerspective(source, destination, transform)).toThrow(BindingError);
    // @ts-expect-error test binding arity rejection
    expect(() => cv.getPerspectiveTransform(source)).toThrow(BindingError);
    source.dispose();
    expect(() =>
      cv.warpPerspective(source, destination, transform, { width: 1, height: 1 }),
    ).toThrow(BindingError);
    destination.dispose();
    transform.dispose();
  });
});

describe("resize binding", () => {
  test("converts dimensions, distinguishes omitted arguments, and forwards every mode", () => {
    const backend = new CopyingBackend();
    const calls: number[][] = [];
    backend.matResizeInto = (_source, _destination, width, height, fx, fy, mode) => {
      calls.push([width, height, fx, fy, mode]);
    };
    const cv = createOpenCv(backend);
    const source = cv.matFromU8(1, 1, 1, new Uint8Array([7]));
    const destination = cv.emptyMat();
    cv.resize(source, destination, { width: 3.9, height: 2.1 });
    expect(calls[0]).toEqual([3, 2, 0, 0, 1]);
    for (const mode of [0, 1, 2, 3, 4, 5, 6] as const)
      cv.resize(source, destination, { width: 2, height: 3 }, 0.7, 1.3, mode);
    expect(calls.slice(1).map((call) => call[4])).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(() => cv.resize(source, destination, { width: 2, height: 3 }, 0, 0, undefined)).toThrow(
      TypeError,
    );
    expect(() => cv.resize(source, destination, { width: 2, height: 3 }, undefined)).toThrow();
    expect(cv.resize.length).toBe(0);
    // @ts-expect-error audit runtime binding arity
    expect(() => cv.resize(source, destination)).toThrow(BindingError);
    // @ts-expect-error audit runtime binding arity
    expect(() => cv.resize(source, destination, { width: 1, height: 1 }, 0, 0, 0, 0)).toThrow(
      BindingError,
    );
    source.dispose();
    destination.dispose();
  });
  test("validates matrix lifetime before reading structural size fields", () => {
    const cv = createOpenCv(new CopyingBackend());
    const source = cv.emptyMat(),
      destination = cv.emptyMat();
    const reads: string[] = [];
    const size = {
      get width() {
        reads.push("width");
        return 2;
      },
      get height() {
        reads.push("height");
        return 2;
      },
    };
    source.dispose();
    expect(() => cv.resize(source, destination, size)).toThrow(BindingError);
    expect(reads).toEqual([]);
    destination.dispose();
  });
});
