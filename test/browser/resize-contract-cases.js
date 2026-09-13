import { resizeMatrixApi } from "./resize-cases.js";

export function runResizeContracts(cv, reference) {
  const api = resizeMatrixApi(cv, reference);
  const entries = [];
  for (let mode = 0; mode <= 6; mode++) {
    for (const depth of [0, 1, 2, 3, 4, 5, 6]) {
      for (const layout of ["source-roi", "destination-roi", "in-place", "overlap"])
        entries.push({ name: `layout/${mode}/${depth}/${layout}`, mode, depth, layout });
    }
  }
  for (const mode of [-1, 7, 8, 16, 999, undefined, NaN, Infinity, true, false, "1", null]) {
    for (const sameSize of [false, true])
      entries.push({ name: `mode/${String(mode)}/${sameSize}`, mode, sameSize });
  }
  for (const field of ["width", "height", "fx", "fy"]) {
    for (const value of [
      undefined,
      NaN,
      Infinity,
      -Infinity,
      -1.5,
      1.5,
      true,
      false,
      "2",
      null,
      4294967299,
    ])
      entries.push({ name: `coercion/${field}/${String(value)}`, field, value });
  }
  for (let arity = 0; arity <= 8; arity++) entries.push({ name: `arity/${arity}`, arity });
  for (const layout of [
    "source-deleted",
    "destination-deleted",
    "source-null",
    "destination-null",
    "source-object",
    "destination-object",
    "empty-source",
    "size-null",
    "size-object",
    "size-array",
    "argument-order-source",
    "argument-order-destination",
    "argument-order-live",
  ])
    entries.push({ name: layout, layout });
  for (const depth of [5, 6])
    for (const mode of [0, 1, 2, 3, 4, 5, 6]) {
      for (const value of [NaN, Infinity, -Infinity, -0])
        entries.push({
          name: `nonfinite/${depth}/${mode}/${String(value)}`,
          depth,
          mode,
          value,
          nonfinite: true,
        });
    }
  return entries.map((entry) => {
    const owned = new Set();
    const own = (matrix) => {
      owned.add(matrix);
      return matrix;
    };
    const release = (matrix) => {
      api.dispose(matrix);
      owned.delete(matrix);
    };
    try {
      const rows = 5,
        columns = 7,
        channels = 3,
        depth = entry.depth ?? 0;
      const values = Array.from({ length: rows * columns * channels }, (_, i) =>
        entry.nonfinite && i % 7 === 0 ? entry.value : (i * 43 + 17) % 251,
      );
      let source = own(api.mat(rows, columns, channels, depth, values));
      let destination = own(api.empty());
      let parent;
      let width = 9,
        height = 8,
        fx = 0,
        fy = 0;
      if (entry.layout === "source-roi") source = own(api.roi(source, 1, 1, 3, 5));
      if (entry.layout === "destination-roi") {
        parent = own(
          api.mat(
            10,
            12,
            channels,
            depth,
            Array.from({ length: 360 }, () => 123),
          ),
        );
        destination = own(api.roi(parent, 1, 2, 8, 9));
      }
      if (entry.layout === "in-place") destination = source;
      if (entry.layout === "overlap") {
        destination = own(api.roi(source, 1, 1, 3, 5));
        width = 5;
        height = 3;
      }
      if (entry.layout === "empty-source") source = own(api.empty());
      if (entry.sameSize) {
        width = columns;
        height = rows;
      }
      if (entry.field === "width") width = entry.value;
      if (entry.field === "height") height = entry.value;
      if (entry.field === "fx") fx = entry.value;
      if (entry.field === "fy") fy = entry.value;
      const trace = [];
      let size = { width, height };
      if (entry.layout?.startsWith("argument-order"))
        size = {
          get width() {
            trace.push("width");
            return width;
          },
          get height() {
            trace.push("height");
            return height;
          },
        };
      if (entry.layout === "size-null") size = null;
      if (entry.layout === "size-object") size = {};
      if (entry.layout === "size-array") size = [width, height];
      let sourceArgument = source,
        destinationArgument = destination;
      if (entry.layout === "source-deleted" || entry.layout === "argument-order-source")
        release(source);
      if (entry.layout === "destination-deleted" || entry.layout === "argument-order-destination")
        release(destination);
      if (entry.layout === "source-null") sourceArgument = null;
      if (entry.layout === "destination-null") destinationArgument = null;
      if (entry.layout === "source-object") sourceArgument = {};
      if (entry.layout === "destination-object") destinationArgument = {};
      const args = [
        sourceArgument,
        destinationArgument,
        size,
        fx,
        fy,
        Object.hasOwn(entry, "mode") ? entry.mode : 1,
      ];
      if (entry.arity !== undefined) {
        args.length = entry.arity;
      }
      let error = false;
      try {
        cv.resize(...args);
      } catch {
        error = true;
      }
      const summary = (matrix) => {
        const result = api.summary(matrix);
        if (error) delete result.values; // A rejected native operation may allocate uninitialized output bytes.
        return result;
      };
      const result = { name: entry.name, error, trace };
      if (owned.has(source)) result.source = summary(source);
      if (owned.has(destination)) result.destination = summary(destination);
      if (parent) result.parent = summary(parent);
      return result;
    } finally {
      for (const matrix of [...owned].toReversed()) api.dispose(matrix);
    }
  });
}
