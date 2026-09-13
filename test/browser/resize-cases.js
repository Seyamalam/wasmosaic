import { perspectiveApi } from "./perspective-adapters.js";

export function resizeMatrixApi(cv, reference) {
  const base = perspectiveApi(cv, reference);
  return reference
    ? {
        ...base,
        mat(rows, columns, channels, depth, values) {
          const matrix = new cv.Mat(rows, columns, depth + ((channels - 1) << 3));
          const pointers = [
            "ucharPtr",
            "charPtr",
            "ushortPtr",
            "shortPtr",
            "intPtr",
            "floatPtr",
            "doublePtr",
          ];
          for (let row = 0; row < rows; row++)
            matrix[pointers[depth]](row).set(
              values.slice(row * columns * channels, (row + 1) * columns * channels),
            );
          return matrix;
        },
      }
    : base;
}

export function runResizeCases(cv, reference) {
  const api = resizeMatrixApi(cv, reference);
  const cases = [];
  const geometries = [
    [3, 5, 7, 8],
    [3, 5, 2, 3],
    [4, 6, 2, 3],
    [2, 2, 3, 3],
    [1, 4, 3, 7],
    [5, 1, 8, 3],
    [4, 6, 7, 2],
    [1, 1, 2, 2],
    [3, 5, 3, 5],
    [5, 7, 2, 3],
    [9, 13, 17, 19],
    [13, 17, 5, 7],
    [7, 9, 3, 4],
    [10, 14, 5, 7],
  ];
  for (let depth = 0; depth < 7; depth++) {
    for (const channels of [1, 2, 3, 4, 5]) {
      for (let mode = 0; mode <= 6; mode++) {
        for (const [rows, columns, height, width] of geometries) {
          cases.push({
            name: `resize/${depth}/${channels}/${mode}/${rows}x${columns}-${height}x${width}`,
            depth,
            channels,
            mode,
            rows,
            columns,
            height,
            width,
          });
        }
      }
    }
  }
  for (const [fx, fy] of [
    [1.3, 1.7],
    [0.7, 0.6],
    [2.1, 0.51],
    [0, 0],
    [-1, 1],
    [NaN, 1],
    [Infinity, 1],
  ]) {
    for (let mode = 0; mode <= 6; mode++)
      cases.push({
        name: `scale/${fx}/${fy}/${mode}`,
        depth: 0,
        channels: 1,
        mode,
        rows: 3,
        columns: 5,
        height: 0,
        width: 0,
        fx,
        fy,
      });
  }
  for (const width of [-1, 0, 2])
    for (const height of [-1, 0, 2])
      cases.push({
        name: `size/${width}/${height}`,
        depth: 0,
        channels: 1,
        mode: 0,
        rows: 3,
        columns: 5,
        width,
        height,
        fx: 1.3,
        fy: 1.7,
      });
  const edgeValues = [
    [0, 255, 1, 254],
    [-128, 127, -1, 0],
    [0, 65535, 32767, 32768],
    [-32768, 32767, -1, 0],
    [-2147483648, 2147483647, -1, 0],
    [1e30, -1e30, 1e-30, -1e-30],
    [1e150, -1e150, 1e-150, -1e-150],
  ];
  for (let depth = 0; depth < 7; depth++)
    for (let mode = 0; mode <= 6; mode++) {
      cases.push({
        name: `extremes/${depth}/${mode}`,
        depth,
        channels: 1,
        mode,
        rows: 3,
        columns: 5,
        height: 7,
        width: 8,
        pattern: edgeValues[depth],
      });
    }
  return [
    { name: "arity", result: cv.resize.length },
    ...cases.map((entry) => {
      const handles = [];
      try {
        const values = Array.from(
          { length: entry.rows * entry.columns * entry.channels },
          (_, i) => {
            if (entry.pattern) return entry.pattern[i % entry.pattern.length];
            const value = (i * 73 + 19) % 251;
            if (entry.depth === 1 || entry.depth === 3 || entry.depth === 4) return value - 127;
            if (entry.depth >= 5) return (value - 100) / 7;
            if (entry.depth === 2) return value * 199;
            return value;
          },
        );
        const source = api.mat(entry.rows, entry.columns, entry.channels, entry.depth, values);
        handles.push(source);
        const destination = api.empty();
        handles.push(destination);
        cv.resize(
          source,
          destination,
          { width: entry.width, height: entry.height },
          entry.fx ?? 0,
          entry.fy ?? 0,
          entry.mode,
        );
        return { name: entry.name, result: api.summary(destination) };
      } catch {
        return { name: entry.name, error: true };
      } finally {
        for (const handle of handles.toReversed()) api.dispose(handle);
      }
    }),
  ];
}

// oxlint-disable anti-slop/no-runtime-typeof -- Serialized browser boundary values contain mixed scalar and matrix results.
export function compareResize(actual, reference) {
  if (actual.length !== reference.length) throw new Error("Resize case count differs");
  const differences = [];
  let numericCases = 0,
    errorCases = 0,
    maximumError = 0,
    maximumRelativeError = 0;
  const equal = (a, b, depth) => {
    if (a === b) return true;
    if (typeof a === "number" && typeof b === "number") {
      const delta = Math.abs(a - b);
      maximumError = Math.max(maximumError, delta);
      maximumRelativeError = Math.max(maximumRelativeError, delta / Math.max(1, Math.abs(b)));
      const tolerance = depth === 5 ? 2e-5 : depth === 6 ? 2e-9 : 0;
      return delta <= tolerance * Math.max(1, Math.abs(b));
    }
    if (Array.isArray(a) && Array.isArray(b))
      return a.length === b.length && a.every((v, i) => equal(v, b[i], depth));
    if (a && b && typeof a === "object" && typeof b === "object")
      return (
        Object.keys(a).length === Object.keys(b).length &&
        Object.keys(a).every((key) => equal(a[key], b[key], a.depth ?? depth))
      );
    return false;
  };
  for (let index = 0; index < actual.length; index++) {
    const a = actual[index],
      b = reference[index];
    if (a.name !== b.name) throw new Error("Resize case ordering differs");
    if (!equal(a, b)) differences.push({ actual: a, expected: b });
    else if (a.error) errorCases++;
    else numericCases++;
  }
  return {
    cases: actual.length,
    numericCases,
    errorCases,
    maximumError,
    maximumRelativeError,
    differences,
  };
}
