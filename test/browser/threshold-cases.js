import { resizeMatrixApi } from "./resize-cases.js";

const scalar = (value) =>
  Object.is(value, -0) ? "-0" : Number.isFinite(value) ? value : String(value);
export function runThresholdCases(cv, reference) {
  const api = resizeMatrixApi(cv, reference);
  const entries = [];
  const valuesByDepth = [
    [0, 1, 2, 50, 99, 100, 101, 127, 128, 200, 254, 255],
    [-128, -127, -1, 0, 1, 2, 50, 99, 100, 101, 126, 127],
    [0, 1, 2, 50, 99, 100, 101, 32767, 32768, 60000, 65534, 65535],
    [-32768, -32767, -101, -1, 0, 1, 2, 99, 100, 101, 32766, 32767],
    [-2147483648, -2147483647, -101, -1, 0, 1, 2, 99, 100, 101, 2147483646, 2147483647],
    [-Infinity, -1, -0, 0, 0.5, 99.9, 100, 100.1, 101, Infinity, NaN, 1e30],
    [-Infinity, -1, -0, 0, 0.5, 99.9, 100, 100.1, 101, Infinity, NaN, 1e150],
  ];
  for (let depth = 0; depth < 7; depth++)
    for (const channels of [1, 3])
      for (let mode = 0; mode < 5; mode++) {
        for (const threshold of [
          -Infinity,
          -32769.5,
          -1.5,
          -0,
          0.5,
          100.8,
          255,
          65535,
          2147483648,
          Infinity,
          NaN,
        ]) {
          entries.push({
            name: `fixed/${depth}/${channels}/${mode}/${scalar(threshold)}`,
            depth,
            channels,
            mode,
            threshold,
            maximum: 200.5,
          });
        }
        for (const maximum of [
          -Infinity,
          -100,
          -0,
          0,
          0.5,
          127.5,
          255.5,
          65535.5,
          2147483648,
          Infinity,
          NaN,
        ]) {
          entries.push({
            name: `maximum/${depth}/${channels}/${mode}/${scalar(maximum)}`,
            depth,
            channels,
            mode,
            threshold: 100,
            maximum,
          });
        }
      }
  for (let depth = 0; depth < 7; depth++)
    for (const channels of [1, 3])
      for (const mode of [
        5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 24, 32, 128, 129, 130, 131, 132, 136, 144,
        152, 256, -1,
      ]) {
        entries.push({
          name: `flags/${depth}/${channels}/${mode}`,
          depth,
          channels,
          mode,
          threshold: 100.8,
          maximum: 200.5,
        });
      }
  for (const mode of [8, 16])
    for (const distribution of [
      "zero",
      "one",
      "bimodal",
      "ramp",
      "spike",
      "left-tail",
      "right-tail",
    ]) {
      entries.push({
        name: `auto/${mode}/${distribution}`,
        depth: 0,
        channels: 1,
        mode,
        distribution,
        threshold: 77,
        maximum: 255,
      });
    }
  for (let depth = 0; depth < 7; depth++)
    for (const mode of [0, 1, 2, 3, 4, 8, 16, 128, 136, 144])
      for (const layout of ["source-roi", "destination-roi", "in-place", "overlap", "empty"]) {
        entries.push({
          name: `layout/${depth}/${mode}/${layout}`,
          depth,
          channels: 1,
          mode,
          layout,
          threshold: 100.8,
          maximum: 200.5,
        });
      }
  for (let arity = 0; arity <= 7; arity++)
    entries.push({
      name: `arity/${arity}`,
      arity,
      depth: 0,
      channels: 1,
      mode: 0,
      threshold: 100,
      maximum: 200,
    });
  return [
    {
      name: "threshold-contract",
      result: {
        arity: cv.threshold.length,
        dryrun: cv.THRESH_DRYRUN,
        triangle: cv.THRESH_TRIANGLE,
      },
    },
    ...entries.map((entry) => {
      const owned = new Set();
      const own = (matrix) => {
        owned.add(matrix);
        return matrix;
      };
      const summary = (matrix) => {
        const result = api.summary(matrix);
        result.values = result.values.map((value) => (Object.is(value, -0) ? "-0" : value));
        return result;
      };
      try {
        let values = valuesByDepth[entry.depth];
        if (entry.distribution === "zero") values = Array.from({ length: 128 }, () => 0);
        if (entry.distribution === "one") values = Array.from({ length: 128 }, () => 77);
        if (entry.distribution === "bimodal")
          values = Array.from({ length: 128 }, (_, i) => (i < 64 ? 10 : 200));
        if (entry.distribution === "ramp") values = Array.from({ length: 256 }, (_, i) => i);
        if (entry.distribution === "spike")
          values = Array.from({ length: 256 }, (_, i) => (i < 250 ? 60 : 200));
        if (entry.distribution === "left-tail")
          values = Array.from({ length: 256 }, (_, i) => (i < 200 ? 200 : i - 200));
        if (entry.distribution === "right-tail")
          values = Array.from({ length: 256 }, (_, i) => (i < 200 ? 40 : i));
        const depth = entry.depth,
          channels = entry.channels;
        let source = own(
          api.mat(
            1,
            values.length,
            channels,
            depth,
            values.flatMap((value) => Array.from({ length: channels }, () => value)),
          ),
        );
        let destination = own(api.empty()),
          parent;
        if (entry.layout === "source-roi")
          source = own(api.roi(source, 0, 1, 1, values.length - 2));
        if (entry.layout === "destination-roi") {
          parent = own(
            api.mat(
              1,
              values.length + 2,
              channels,
              depth,
              Array.from({ length: (values.length + 2) * channels }, () => 66),
            ),
          );
          destination = own(api.roi(parent, 0, 1, 1, values.length));
        }
        if (entry.layout === "in-place") destination = source;
        if (entry.layout === "overlap") {
          const full = source;
          source = own(api.roi(full, 0, 0, 1, values.length - 1));
          destination = own(api.roi(full, 0, 1, 1, values.length - 1));
          parent = full;
        }
        if (entry.layout === "empty") source = own(api.empty());
        const args = [source, destination, entry.threshold, entry.maximum, entry.mode];
        if (entry.arity !== undefined) args.length = entry.arity;
        let returned,
          error = false;
        try {
          returned = scalar(cv.threshold(...args));
        } catch {
          error = true;
        }
        const result = { name: entry.name, error, returned };
        result.source = summary(source);
        result.destination = summary(destination);
        if (parent) result.parent = summary(parent);
        if (error) {
          delete result.source.values;
          delete result.destination.values;
          if (result.parent) delete result.parent.values;
        }
        return result;
      } finally {
        for (const matrix of [...owned].toReversed()) api.dispose(matrix);
      }
    }),
  ];
}
