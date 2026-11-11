import { BindingError, OpenCvInputError } from "./error.js";
import { Mat } from "./mat.js";
import type { WasmMatHandle } from "./mat.js";
import type { Size } from "./types.js";

/** Optional configuration accepted by `OpenCv.createCLAHE`. */
export interface CLAHEOptions {
  readonly clipLimit?: number;
  readonly tileGridSize?: Size;
}

/** OpenCV 4.13 defaults exposed by the CLAHE configuration methods. */
export const CLAHE_DEFAULTS = Object.freeze({
  clipLimit: 40,
  tileGridSize: Object.freeze({ width: 8, height: 8 }),
});

/** Low-level CLAHE processor returned by the generated WebAssembly module. */
export interface WasmCLAHEHandle {
  apply(source: WasmMatHandle, destination: WasmMatHandle): void;
  collectGarbage(): void;
  free(): void;
  getClipLimit(): number;
  getTilesGridSize(): ArrayLike<number>;
  setClipLimit(value: number): void;
  setTilesGridSize(width: number, height: number): void;
}

/** Internal constructor contract exposed by wasm-bindgen's CLAHE processor class. */
export interface WasmCLAHEFactory {
  create(
    clipLimit?: number | null,
    tilesX?: number | null,
    tilesY?: number | null,
  ): WasmCLAHEHandle;
}

/** Rust-owned CLAHE processor with an explicit lifetime. */
export class CLAHE {
  #handle: WasmCLAHEHandle | undefined;

  /** Low-level adapters may construct a processor from a compatible WASM handle. */
  constructor(handle: WasmCLAHEHandle) {
    this.#handle = handle;
  }

  apply(source: Mat, destination: Mat): void {
    requireArityRange(arguments.length, 1, 2, "CLAHE.apply");
    this.#owned().apply(matHandleForBinding(source), matHandleForBinding(destination));
  }

  collectGarbage(): void {
    requireExactArity(arguments.length, 0, "CLAHE.collectGarbage");
    this.#owned().collectGarbage();
  }

  getClipLimit(): number {
    requireExactArity(arguments.length, 0, "CLAHE.getClipLimit");
    return this.#owned().getClipLimit();
  }

  getTilesGridSize(): Size {
    requireExactArity(arguments.length, 0, "CLAHE.getTilesGridSize");
    const grid = this.#owned().getTilesGridSize();
    return { width: grid[0] ?? 0, height: grid[1] ?? 0 };
  }

  setClipLimit(clipLimit: number): void {
    requireExactArity(arguments.length, 1, "CLAHE.setClipLimit");
    this.#owned().setClipLimit(toWasmF64(clipLimit));
  }

  setTilesGridSize(tileGridSize: Size): void {
    requireExactArity(arguments.length, 1, "CLAHE.setTilesGridSize");
    const converted = size2iForBinding(tileGridSize, "CLAHE tile grid size");
    this.#owned().setTilesGridSize(converted.width, converted.height);
  }

  /** Releases the WASM handle with OpenCV.js-compatible repeated-delete behavior. */
  delete(): void {
    if (this.#handle === undefined) {
      throw new BindingError("CLAHE instance already deleted");
    }
    this.dispose();
  }

  /** Releases the WASM handle. Repeated calls do nothing. */
  dispose(): void {
    const handle = this.#handle;
    if (handle === undefined) return;
    this.#handle = undefined;
    handle.free();
  }

  #owned(): WasmCLAHEHandle {
    const handle = this.#handle;
    if (handle === undefined) {
      throw new BindingError("Cannot pass deleted object as a pointer of type CLAHE");
    }
    return handle;
  }
}

export function validateCLAHEOptions(options: CLAHEOptions): void {
  if (options.clipLimit !== undefined && !Number.isFinite(options.clipLimit)) {
    throw new OpenCvInputError("CLAHE clip limit must be a finite number");
  }
  if (options.tileGridSize !== undefined) {
    size2iForBinding(options.tileGridSize, "CLAHE tile grid size");
  }
}

function requireArityRange(actual: number, minimum: number, maximum: number, name: string): void {
  if (actual < minimum || actual > maximum) {
    throw new BindingError(`${name} expects between ${minimum} and ${maximum} arguments`);
  }
}

function requireExactArity(actual: number, expected: number, name: string): void {
  if (actual !== expected) {
    throw new BindingError(`${name} expects ${expected} arguments, got ${actual}`);
  }
}

// oxlint-disable-next-line anti-slop/no-unknown-parameters -- JS-to-Embind Mat boundary.
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

function size2iForBinding(value: Size, name: string) {
  if (
    !Number.isSafeInteger(value.width) ||
    !Number.isSafeInteger(value.height) ||
    value.width < 1 ||
    value.height < 1 ||
    value.width > 2_147_483_647 ||
    value.height > 2_147_483_647
  ) {
    throw new OpenCvInputError(`${name} must contain positive signed 32-bit integer dimensions`);
  }
  return { width: value.width, height: value.height };
}

function toWasmF64(value: number): number {
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- JS-to-Embind scalar boundary.
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`Cannot convert "${String(value)}" to double`);
  }
  return value;
}
