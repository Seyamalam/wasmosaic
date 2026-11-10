import { resizeMatrixApi } from "./resize-cases.js";

function trackMat(handles, mat) {
  handles.push(mat);
  return mat;
}

const depths = [0, 1, 2, 3, 4, 5, 6];
const arrays = {
  0: (n) => Array.from({ length: n }, (_, i) => (i * 37 + 11) % 256),
  1: (n) => Array.from({ length: n }, (_, i) => ((i * 31 + 7) % 255) - 128),
  2: (n) => Array.from({ length: n }, (_, i) => (i * 911 + 17) % 65536),
  3: (n) => Array.from({ length: n }, (_, i) => ((i * 577 + 13) % 65535) - 32768),
  4: (n) => Array.from({ length: n }, (_, i) => ((i * 2147 + 23) % 2 ** 31) - 2 ** 30),
  5: (n) => Array.from({ length: n }, (_, i) => i * 0.5 - 3.25 + (i % 7) * 1e-4),
  6: (n) => Array.from({ length: n }, (_, i) => i * 0.5 - 3.25 + (i % 13) * 1e-9),
};

export function runCoreOpsCases(cv, reference) {
  const api = resizeMatrixApi(cv, reference);
  const entries = [];
  const LUT = (...args) => cv.LUT(...args);
  const run = (name, thunk) => {
    const handles = [];
    try {
      entries.push({ name, ...thunk(handles) });
    } catch {
      entries.push({ name, error: true });
    } finally {
      for (const handle of handles.toReversed()) {
        try {
          api.dispose(handle);
        } catch {
          /* already released */
        }
      }
    }
  };
  const track = trackMat;

  // ---- meanStdDev: depths, channels, masks, ROI sources
  for (const depth of depths)
    for (const channels of [1, 3])
      for (const maskMode of ["none", "full", "partial", "badDepth"]) {
        run(`msd/${depth}/${channels}/${maskMode}`, (handles) => {
          const rows = 3,
            columns = 4;
          const count = rows * columns * channels;
          const source = track(
            handles,
            api.mat(rows, columns, channels, depth, arrays[depth](count)),
          );
          const means = track(handles, api.empty());
          const deviations = track(handles, api.empty());
          let mask;
          if (maskMode !== "none") {
            const values =
              maskMode === "badDepth"
                ? arrays[5](rows * columns)
                : maskMode === "full"
                  ? rows * columns === 1
                    ? [255]
                    : Array.from({ length: rows * columns }, () => 255)
                  : Array.from({ length: rows * columns }, (_, i) => (i % 2 ? 255 : 0));
            mask = track(
              handles,
              api.mat(rows, columns, 1, maskMode === "badDepth" ? 5 : 0, values),
            );
            cv.meanStdDev(source, means, deviations, mask);
          } else {
            cv.meanStdDev(source, means, deviations);
          }
          return { means: api.summary(means), deviations: api.summary(deviations) };
        });
      }
  // ROI source case
  run("msd/roi", (handles) => {
    const parent = track(handles, api.mat(4, 6, 1, 0, arrays[0](24)));
    const source = api.roi(parent, 1, 2, 2, 3);
    handles.push(source);
    const means = track(handles, api.empty());
    const deviations = track(handles, api.empty());
    cv.meanStdDev(source, means, deviations);
    return { means: api.summary(means), deviations: api.summary(deviations) };
  });

  // ---- reduce: axes, kinds, dtype, depths
  for (const depth of depths)
    for (const axis of [0, 1])
      for (const kind of [0, 1, 2, 3, 4, 99])
        for (const dtype of [-1, 0, 5, 6]) {
          run(`reduce/${depth}/${axis}/${kind}/${dtype}`, (handles) => {
            const source = track(handles, api.mat(3, 4, 1, depth, arrays[depth](12)));
            const destination = track(handles, api.empty());
            const call =
              dtype === -1
                ? () => cv.reduce(source, destination, axis, kind)
                : () => cv.reduce(source, destination, axis, kind, dtype);
            call();
            return { destination: api.summary(destination) };
          });
        }
  for (const channels of [2, 3, 4])
    run(`reduce/chan/${channels}`, (handles) => {
      const source = track(handles, api.mat(2, 3, channels, 0, arrays[0](2 * 3 * channels)));
      const destination = track(handles, api.empty());
      cv.reduce(source, destination, 0, 0);
      return { destination: api.summary(destination) };
    });

  // ---- norm: types, single/masked/invalid-arg-count forms
  for (const depth of depths)
    for (const normType of [1, 2, 4, 5, 6, 7, 8, 9, 12, 32, 33, 34, 36])
      for (const form of ["single", "masked", "extra", "matAsType"]) {
        run(`norm/${depth}/${normType}/${form}`, (handles) => {
          const first = track(handles, api.mat(2, 3, 1, depth, arrays[depth](6)));
          const second = track(
            handles,
            api.mat(
              2,
              3,
              1,
              depth,
              arrays[depth](6).map((v) => v / 2),
            ),
          );
          const mask = track(handles, api.mat(2, 3, 1, 0, [255, 0, 255, 0, 255, 0]));
          let value;
          if (form === "single") {
            value = cv.norm(first, normType);
          } else if (form === "masked") {
            value = cv.norm(first, normType, mask);
          } else if (form === "extra") {
            value = cv.norm(first, second, normType, mask);
          } else {
            value = cv.norm(first, second, normType);
          }
          return { value, depth: 6 };
        });
      }
  run("norm/one-arg", (handles) => {
    const first = track(handles, api.mat(2, 2, 1, 0, [3, 4, 0, 0]));
    return { value: cv.norm(first), depth: 6 };
  });

  // ---- normalize: norm types, alpha/beta, dtype, masks
  for (const depth of depths)
    for (const normType of [1, 2, 4, 32])
      for (const [alpha, beta] of [
        [10, 0],
        [0, 200],
        [-5, 3],
        [0.5, -2],
        [Infinity, 0],
        [0, NaN],
      ])
        for (const dtype of [-1, 0, 5, 6]) {
          run(`normalize/${depth}/${normType}/${alpha}/${beta}/${dtype}`, (handles) => {
            const source = track(handles, api.mat(2, 3, 1, depth, arrays[depth](6)));
            const destination = track(handles, api.empty());
            const call =
              dtype === -1
                ? () => cv.normalize(source, destination, alpha, beta, normType)
                : () => cv.normalize(source, destination, alpha, beta, normType, dtype);
            call();
            return { destination: api.summary(destination) };
          });
        }
  for (const maskMode of ["full", "partial", "empty"]) {
    run(`normalize/mask/${maskMode}`, (handles) => {
      const source = track(handles, api.mat(2, 3, 1, 0, [10, 20, 30, 40, 50, 60]));
      const destination = track(handles, api.mat(2, 3, 1, 0, [99, 99, 99, 99, 99, 99]));
      const maskValues =
        maskMode === "full"
          ? [255, 255, 255, 255, 255, 255]
          : maskMode === "partial"
            ? [255, 0, 255, 0, 255, 0]
            : [0, 0, 0, 0, 0, 0];
      const mask = track(handles, api.mat(2, 3, 1, 0, maskValues));
      cv.normalize(source, destination, 10, 0, 2, -1, mask);
      return { destination: api.summary(destination) };
    });
  }

  // ---- LUT: source depths, channel tables, output depths
  for (const sourceDepth of [0, 1])
    for (const tableChannels of [1, 3])
      for (const sourceChannels of [1, tableChannels])
        for (const tableDepth of depths) {
          run(`lut/${sourceDepth}/${sourceChannels}/${tableChannels}/${tableDepth}`, (handles) => {
            const count = 6 * sourceChannels;
            const source = track(
              handles,
              api.mat(2, 3, sourceChannels, sourceDepth, arrays[sourceDepth](count)),
            );
            const table = track(
              handles,
              api.mat(1, 256, tableChannels, tableDepth, arrays[tableDepth](256 * tableChannels)),
            );
            const destination = track(handles, api.empty());
            LUT(source, table, destination);
            return { destination: api.summary(destination) };
          });
        }

  // ---- transform: channels, coefficient shapes/depths
  for (const sourceDepth of depths)
    for (const channels of [1, 2, 3, 4])
      for (const coeffDepth of [5, 6])
        for (const coeffRows of [channels, channels + 1]) {
          run(`transform/${sourceDepth}/${channels}/${coeffDepth}/${coeffRows}`, (handles) => {
            const source = track(
              handles,
              api.mat(2, 2, channels, sourceDepth, arrays[sourceDepth](4 * channels)),
            );
            const coefficient = track(
              handles,
              api.mat(
                coeffRows,
                channels,
                1,
                coeffDepth,
                Array.from({ length: coeffRows * channels }, (_, i) => ((i % 3) + 1) * 0.5),
              ),
            );
            const destination = track(handles, api.empty());
            cv.transform(source, destination, coefficient);
            return { destination: api.summary(destination) };
          });
        }

  // ---- perspectiveTransform: F32/F64, 2/3 channels
  for (const depth of [5, 6])
    for (const channels of [2, 3])
      for (const coeffSize of [channels + 1]) {
        run(`persp/${depth}/${channels}`, (handles) => {
          const source = track(
            handles,
            api.mat(
              2,
              1,
              channels,
              depth,
              Array.from({ length: 2 * channels }, (_, i) => i + 0.5),
            ),
          );
          const coefficient = track(
            handles,
            api.mat(
              coeffSize,
              coeffSize,
              1,
              depth,
              Array.from({ length: coeffSize * coeffSize }, (_, i) =>
                i % (coeffSize + 1) === 0 ? 1 : 0.25 * ((i / coeffSize) | 0) + 0.01 * i,
              ),
            ),
          );
          const destination = track(handles, api.empty());
          cv.perspectiveTransform(source, destination, coefficient);
          return { destination: api.summary(destination) };
        });
      }

  // ---- invert: sizes, depths, methods
  for (const depth of depths)
    for (const size of [1, 2, 3])
      for (const method of [0, 1, 3, 4, 99]) {
        run(`invert/${depth}/${size}/${method}`, (handles) => {
          const source = track(
            handles,
            api.mat(
              size,
              size,
              1,
              depth,
              Array.from({ length: size * size }, (_, i) =>
                i % (size + 1) === 0 ? 2 : 0.5 * ((i % size) + 1),
              ),
            ),
          );
          const destination = track(handles, api.empty());
          const determinant = cv.invert(source, destination, method);
          return { determinant, destination: api.summary(destination) };
        });
      }

  // ---- solve: square systems, right-hand sides, methods
  for (const depth of depths)
    for (const rhsColumns of [1, 2])
      for (const method of [0, 1, 3, 4]) {
        run(`solve/${depth}/${rhsColumns}/${method}`, (handles) => {
          const n = 2;
          const coefficients = track(
            handles,
            api.mat(n, n, 1, depth, depth === 5 || depth === 6 ? [2, 0.5, 0.25, 3] : [2, 1, 0, 3]),
          );
          const rhs = track(
            handles,
            api.mat(
              n,
              rhsColumns,
              1,
              depth,
              Array.from({ length: n * rhsColumns }, (_, i) => i + 2),
            ),
          );
          const destination = track(handles, api.empty());
          const solved = cv.solve(coefficients, rhs, destination, method);
          return { solved, destination: api.summary(destination) };
        });
      }

  // ---- elementwise binary ops: add/subtract/absdiff/max/min across depths, channels,
  //      masks, dtype, and error paths
  const bop = {
    add: (...a) => cv.add(...a),
    subtract: (...a) => cv.subtract(...a),
    absdiff: (...a) => cv.absdiff(...a),
    max: (...a) => cv.max(...a),
    min: (...a) => cv.min(...a),
  };
  for (const name of ["add", "subtract", "absdiff", "max", "min"])
    for (const depth of depths)
      for (const channels of [1, 3]) {
        const rows = 2,
          columns = 3,
          count = rows * columns * channels;
        run(`${name}/${depth}/${channels}/plain`, (handles) => {
          const s1 = track(handles, api.mat(rows, columns, channels, depth, arrays[depth](count)));
          const s2 = track(
            handles,
            api.mat(
              rows,
              columns,
              channels,
              depth,
              arrays[depth](count).map((v) => (Number.isInteger(v) ? v * 2 + 3 : v * -0.5 + 1)),
            ),
          );
          const destination = track(handles, api.empty());
          bop[name](s1, s2, destination);
          return { destination: api.summary(destination) };
        });
      }
  for (const name of ["add", "subtract"])
    for (const depth of depths)
      for (const variant of ["mask", "dtype5", "maskDtype", "badDtype", "badShape", "badDepth"])
        run(`${name}/${depth}/${variant}`, (handles) => {
          const rows = 2,
            columns = 3,
            count = rows * columns * 1;
          const source1 = track(handles, api.mat(rows, columns, 1, depth, arrays[depth](count)));
          const source2 = track(
            handles,
            api.mat(
              variant === "badShape" ? rows + 1 : rows,
              columns,
              1,
              variant === "badDepth" ? 6 : depth,
              (variant === "badDepth" ? arrays[6] : arrays[depth])(
                (variant === "badShape" ? rows + 1 : rows) * columns,
              ),
            ),
          );
          const mask = track(handles, api.mat(rows, columns, 1, 0, [255, 0, 255, 0, 255, 0]));
          const destination = track(handles, api.empty());
          const s2 = variant === "badShape" || variant === "badDepth" ? source2 : source2;
          if (variant === "mask") bop[name](source1, s2, destination, mask);
          else if (variant === "dtype5") bop[name](source1, s2, destination, api.empty(), 5);
          else if (variant === "maskDtype") bop[name](source1, s2, destination, mask, 6);
          else if (variant === "badDtype") bop[name](source1, s2, destination, mask, 9);
          else bop[name](source1, s2, destination);
          return { destination: api.summary(destination) };
        });

  // ---- absdiff/max/min strict 3-arg errors
  for (const name of ["absdiff", "max", "min"]) {
    run(`${name}/extra-arg`, (handles) => {
      const s = track(handles, api.mat(1, 2, 1, 0, [1, 2]));
      const d = track(handles, api.empty());
      const mask = track(handles, api.mat(1, 2, 1, 0, [255, 255]));
      bop[name](s, s, d, mask);
      return { destination: api.summary(d) };
    });
    run(`${name}/mismatch-depth`, (handles) => {
      const s1 = track(handles, api.mat(1, 2, 1, 0, [1, 2]));
      const s2 = track(handles, api.mat(1, 2, 1, 5, [1, 2]));
      const d = track(handles, api.empty());
      bop[name](s1, s2, d);
      return { destination: api.summary(d) };
    });
  }

  // ---- bitwise ops: all depths (raw bit patterns), masks
  const bops = {
    and: (...a) => (reference ? cv.bitwise_and(...a) : cv.bitwiseAnd(...a)),
    or: (...a) => (reference ? cv.bitwise_or(...a) : cv.bitwiseOr(...a)),
    xor: (...a) => (reference ? cv.bitwise_xor(...a) : cv.bitwiseXor(...a)),
  };
  for (const op of ["and", "or", "xor"])
    for (const depth of depths)
      for (const variant of ["plain", "mask", "badDepth"]) {
        const rows = 2,
          columns = 3;
        run(`bitwise_${op}/${depth}/${variant}`, (handles) => {
          const source1 = track(handles, api.mat(rows, columns, 1, depth, arrays[depth](6)));
          const source2 = track(
            handles,
            api.mat(
              variant === "badDepth" ? rows : rows,
              columns,
              1,
              variant === "badDepth" ? 0 : depth,
              arrays[variant === "badDepth" ? 0 : depth](6).map((v) =>
                Number.isInteger(v) ? v + 7 : v * 2,
              ),
            ),
          );
          const destination = track(handles, api.empty());
          if (variant === "mask") {
            const mask = track(handles, api.mat(rows, columns, 1, 0, [0, 255, 0, 255, 0, 255]));
            bops[op](source1, source2, destination, mask);
          } else {
            bops[op](source1, source2, destination);
          }
          return { destination: api.summary(destination) };
        });
      }

  // ---- compare: all ops, depths, scalar edge values
  for (const depth of depths)
    for (const op of [0, 1, 2, 3, 4, 5, 9])
      run(`compare/${depth}/${op}`, (handles) => {
        const rows = 2,
          columns = 3;
        const source1 = track(handles, api.mat(rows, columns, 1, depth, arrays[depth](6)));
        const source2 = track(
          handles,
          api.mat(
            rows,
            columns,
            1,
            depth,
            arrays[depth](6).map((v, i) =>
              i % 2 === 0 ? v : Number.isInteger(v) ? v + 1 : v + 0.5,
            ),
          ),
        );
        const destination = track(handles, api.empty());
        cv.compare(source1, source2, destination, op);
        return { destination: api.summary(destination) };
      });

  // ---- inRange: depths, single/per-channel bounds
  for (const depth of depths)
    for (const boundChannels of [1, 3])
      for (const sourceChannels of [1, 3]) {
        const rows = 2,
          columns = 3,
          boundCount = rows * columns * boundChannels,
          sourceCount = rows * columns * sourceChannels;
        const boundValues = (offset) =>
          Array.from({ length: boundCount }, (_, i) => {
            const v = arrays[depth](boundCount)[i];
            return Number.isInteger(v) ? v + offset : v + offset * 0.1;
          });
        run(`inRange/${depth}/${sourceChannels}/${boundChannels}`, (handles) => {
          const source = track(
            handles,
            api.mat(rows, columns, sourceChannels, depth, arrays[depth](sourceCount)),
          );
          const lower = track(
            handles,
            api.mat(rows, columns, boundChannels, depth, boundValues(-20)),
          );
          const upper = track(
            handles,
            api.mat(rows, columns, boundChannels, depth, boundValues(20)),
          );
          const destination = track(handles, api.empty());
          cv.inRange(source, lower, upper, destination);
          return { destination: api.summary(destination) };
        });
      }
  run("inRange/bad-bound-depth", (handles) => {
    const source = track(handles, api.mat(2, 3, 1, 0, [1, 2, 3, 4, 5, 6]));
    const lower = track(handles, api.mat(2, 3, 1, 5, [0, 0, 0, 0, 0, 0]));
    const upper = track(handles, api.mat(2, 3, 1, 0, [9, 9, 9, 9, 9, 9]));
    const destination = track(handles, api.empty());
    cv.inRange(source, lower, upper, destination);
    return { destination: api.summary(destination) };
  });

  // ---- merge/split/hconcat/vconcat/mixChannels vector forms
  const vec = () => (reference ? new cv.MatVector() : cv.createMatVector());
  const vecSummary = (vector) => {
    const planes = [];
    for (let index = 0; index < vector.size(); index += 1) {
      const mat = vector.get(index);
      planes.push(api.summary(mat));
      api.dispose(mat);
    }
    return planes;
  };

  for (const depth of depths)
    for (const variant of ["two", "ch31", "five", "badRows", "badDepth"]) {
      run(`merge/${depth}/${variant}`, (handles) => {
        const rows = 2,
          columns = 3;
        const vector = track(handles, vec());
        const make = (rowsArg, chs, depthArg, offset) =>
          track(
            handles,
            api.mat(
              rowsArg,
              columns,
              chs,
              depthArg,
              arrays[depthArg](rowsArg * columns * chs).map((v) =>
                Number.isInteger(v) ? v + offset : v + offset * 0.01,
              ),
            ),
          );
        if (variant === "two") {
          vector.push_back(make(rows, 1, depth, 0));
          vector.push_back(make(rows, 1, depth, 7));
        } else if (variant === "ch31") {
          vector.push_back(make(rows, 3, depth, 0));
          vector.push_back(make(rows, 1, depth, 7));
        } else if (variant === "five") {
          for (let index = 0; index < 5; index += 1)
            vector.push_back(make(rows, 1, depth, index * 3));
        } else if (variant === "badRows") {
          vector.push_back(make(rows, 1, depth, 0));
          vector.push_back(make(rows + 1, 1, depth, 0));
        } else {
          vector.push_back(make(rows, 1, depth, 0));
          vector.push_back(make(rows, 1, depth === 0 ? 5 : 0, 0));
        }
        const destination = track(handles, api.empty());
        cv.merge(vector, destination);
        return { destination: api.summary(destination) };
      });
    }

  for (const depth of depths)
    for (const channels of [1, 3, 4]) {
      run(`split/${depth}/${channels}`, (handles) => {
        const source = track(
          handles,
          api.mat(2, 3, channels, depth, arrays[depth](2 * 3 * channels)),
        );
        const destination = track(handles, vec());
        if (channels === 4)
          destination.push_back(track(handles, api.mat(9, 9, 9, depth, arrays[depth](9 * 9 * 9))));
        cv.split(source, destination);
        return { planes: vecSummary(destination) };
      });
    }

  for (const depth of depths)
    for (const axis of ["h", "v"])
      for (const variant of ["two", "three", "badDims", "badDepth", "badChannels", "empty"]) {
        run(`concat/${axis}/${depth}/${variant}`, (handles) => {
          const fn = axis === "h" ? (a, b) => cv.hconcat(a, b) : (a, b) => cv.vconcat(a, b);
          const vector = track(handles, vec());
          const make = (rowsArg, colsArg, chs, depthArg, offset) =>
            track(
              handles,
              api.mat(
                rowsArg,
                colsArg,
                chs,
                depthArg,
                arrays[depthArg](rowsArg * colsArg * chs).map((v) =>
                  Number.isInteger(v) ? v + offset : v + offset * 0.01,
                ),
              ),
            );
          const baseRows = 2,
            baseCols = 3;
          const otherDim = axis === "h" ? baseRows : baseCols;
          if (variant === "two" || variant === "three") {
            vector.push_back(make(baseRows, baseCols, 1, depth, 0));
            vector.push_back(make(baseRows, baseCols, 1, depth, 5));
            if (variant === "three") vector.push_back(make(baseRows, baseCols, 1, depth, 9));
          } else if (variant === "badDims") {
            vector.push_back(make(baseRows, baseCols, 1, depth, 0));
            vector.push_back(
              axis === "h"
                ? make(otherDim + 1, baseCols, 1, depth, 0)
                : make(baseRows, otherDim + 1, 1, depth, 0),
            );
          } else if (variant === "badDepth") {
            vector.push_back(make(baseRows, baseCols, 1, depth, 0));
            vector.push_back(make(baseRows, baseCols, 1, depth === 0 ? 2 : 0, 0));
          } else if (variant === "badChannels") {
            vector.push_back(make(baseRows, baseCols, 1, depth, 0));
            vector.push_back(make(baseRows, baseCols, 3, depth, 0));
          }
          const destination = track(handles, api.empty());
          fn(vector, destination);
          return { destination: api.summary(destination) };
        });
      }

  for (const depth of [0, 1, 4, 5, 6])
    for (const variant of ["reroute", "partial", "oddPairs", "badChannel", "badSize", "emptyDst"]) {
      run(`mixChannels/${depth}/${variant}`, (handles) => {
        const sourceVector = track(handles, vec());
        sourceVector.push_back(track(handles, api.mat(2, 2, 3, depth, arrays[depth](12))));
        sourceVector.push_back(
          track(
            handles,
            api.mat(
              2,
              2,
              1,
              depth,
              arrays[depth](4).map((v) => (Number.isInteger(v) ? v + 40 : v + 1)),
            ),
          ),
        );
        const destinationVector = track(handles, vec());
        if (variant !== "emptyDst") {
          destinationVector.push_back(
            track(
              handles,
              api.mat(
                variant === "badSize" ? 3 : 2,
                variant === "badSize" ? 3 : 2,
                1,
                depth,
                arrays[depth](variant === "badSize" ? 9 : 4),
              ),
            ),
          );
          destinationVector.push_back(track(handles, api.mat(2, 2, 2, depth, arrays[depth](8))));
        }
        const pairs = {
          reroute: [2, 0, 3, 1, 0, 2],
          partial: [2, 0],
          oddPairs: [2, 0, 1],
          badChannel: [9, 0],
          badSize: [2, 0],
          emptyDst: [2, 0],
        }[variant];
        cv.mixChannels(sourceVector, destinationVector, pairs);
        return { planes: vecSummary(destinationVector) };
      });
    }

  // ---- contract arity audit
  run("contract", () => ({
    lengths: {
      meanStdDev: cv.meanStdDev.length,
      reduce: cv.reduce.length,
      norm: cv.norm.length,
      normalize: cv.normalize.length,
      LUT: cv.LUT.length,
      transform: cv.transform.length,
      perspectiveTransform: cv.perspectiveTransform.length,
      invert: cv.invert.length,
      solve: cv.solve.length,
      add: cv.add.length,
      subtract: cv.subtract.length,
      absdiff: cv.absdiff.length,
      max: cv.max.length,
      min: cv.min.length,
      compare: cv.compare.length,
      inRange: cv.inRange.length,
      bitwiseAnd: (reference ? cv.bitwise_and : cv.bitwiseAnd).length,
      bitwiseOr: (reference ? cv.bitwise_or : cv.bitwiseOr).length,
      bitwiseXor: (reference ? cv.bitwise_xor : cv.bitwiseXor).length,
      merge: cv.merge.length,
      split: cv.split.length,
      hconcat: cv.hconcat.length,
      vconcat: cv.vconcat.length,
      mixChannels: cv.mixChannels.length,
    },
  }));

  return entries;
}
