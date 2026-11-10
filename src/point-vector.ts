import { OpenCvInputError } from "./error.js";
import type { Point } from "./types.js";

/** JavaScript-owned point vector matching the upstream `cv.PointVector` output API. */
export class PointVector {
  #points: Point[] = [];
  #disposed = false;

  size(): number {
    return this.#points.length;
  }

  get(index: number): Point {
    if (!Number.isSafeInteger(index) || index < 0 || index > 4_294_967_295) {
      throw new OpenCvInputError("PointVector index must be a non-negative 32-bit integer");
    }
    const point = this.#points[index];
    if (point === undefined) {
      throw new OpenCvInputError(`PointVector index ${index} is out of bounds`);
    }
    return point;
  }

  push_back(point: Point): void {
    this.#points.push(point);
  }

  clear(): void {
    this.#points = [];
  }

  dispose(): void {
    this.#points = [];
    this.#disposed = true;
  }

  delete(): void {
    this.dispose();
  }

  /** Reports whether `dispose` released the vector contents. */
  get disposed(): boolean {
    return this.#disposed;
  }
}
