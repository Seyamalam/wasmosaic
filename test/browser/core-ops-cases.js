import { resizeMatrixApi } from "./resize-cases.js";

function trackMat(handles, mat) {
  handles.push(mat);
  return mat;
}

const depths = [0, 1, 2, 3, 4, 5, 6];
const pt = (x, y) => ({ x, y });
const vecPoints = (vector) => {
  const points = [];
  for (let index = 0; index < vector.size(); index += 1) {
    const p = vector.get(index);
    points.push([p.x, p.y]);
  }
  return points;
};
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

  // ---- clipLine: rectangle/point acceptance and boolean result
  for (const [name, rect, a, b] of /** @type {[string, object, object, object][]} */ ([
    ["inside", { x: 0, y: 0, width: 10, height: 10 }, pt(1, 1), pt(5, 5)],
    ["outside", { x: 0, y: 0, width: 10, height: 10 }, pt(-5, -5), pt(-1, -1)],
    ["cross", { x: 0, y: 0, width: 10, height: 10 }, pt(-5, 5), pt(15, 5)],
    ["neg-rect", { x: -2, y: -2, width: 5, height: 5 }, pt(-5, 0), pt(10, 0)],
    ["float-rect", { x: 0.5, y: 0.5, width: 9.7, height: 9.3 }, pt(-5, 5), pt(15, 5)],
    ["float-pts", { x: 0, y: 0, width: 10, height: 10 }, pt(1.5, 1.5), pt(8.9, 8.9)],
    ["degenerate", { x: 0, y: 0, width: 10, height: 10 }, pt(5, 5), pt(5, 5)],
    ["bad-width", { x: 0, y: 0, width: -10, height: 10 }, pt(1, 1), pt(5, 5)],
    ["zero-size", { x: 0, y: 0, width: 0, height: 0 }, pt(1, 1), pt(5, 5)],
    ["bad-rect", { width: 10, height: 10 }, pt(1, 1), pt(5, 5)],
    ["bad-point", { x: 0, y: 0, width: 10, height: 10 }, pt(1, 1), { y: 2 }],
  ])) {
    run(`clipLine/${name}`, () => ({ result: cv.clipLine(rect, a, b) }));
  }

  // ---- copyMakeBorder: dst form, all depths, five modes, optional value
  const borderTypes = [0, 1, 2, 3, 4];
  for (const depth of depths)
    for (const borderType of borderTypes)
      for (const variant of ["value", "default", "asymmetric", "isolated"]) {
        run(`copyMakeBorder/${depth}/${borderType}/${variant}`, (handles) => {
          const source = track(
            handles,
            api.mat(2, 3, depth === 6 ? 1 : 2, depth, arrays[depth](2 * 3 * (depth === 6 ? 1 : 2))),
          );
          const destination = track(handles, api.empty());
          const top = variant === "asymmetric" ? 2 : 1;
          const left = variant === "asymmetric" ? 0 : 1;
          const type = variant === "isolated" ? borderType | 16 : borderType;
          if (variant === "value") {
            cv.copyMakeBorder(source, destination, top, 1, left, 1, type, [9, 8, 7, 6]);
          } else {
            cv.copyMakeBorder(source, destination, top, 1, left, 1, type);
          }
          return { destination: api.summary(destination) };
        });
      }

  // ---- ellipse2Poly: arcs, steps, swaps, rotation
  const ptsvec = () => (reference ? new cv.PointVector() : cv.createPointVector());
  for (const variant of [
    "full-90",
    "quarter-30",
    "swapped",
    "rotated",
    "delta-1",
    "big-delta",
    "empty-arc",
    "zero-axes",
    "neg-center",
    "float-args",
    "bad-delta",
  ]) {
    run(`ellipse2Poly/${variant}`, (handles) => {
      const vector = track(handles, ptsvec());
      const args = {
        "full-90": [{ x: 0, y: 0 }, { width: 6, height: 4 }, 0, 0, 360, 90],
        "quarter-30": [{ x: 5, y: 5 }, { width: 4, height: 2 }, 0, 0, 90, 30],
        swapped: [{ x: 0, y: 0 }, { width: 4, height: 2 }, 0, 300, 60, 30],
        rotated: [{ x: 0, y: 0 }, { width: 4, height: 2 }, 45, 0, 360, 90],
        "delta-1": [{ x: 0, y: 0 }, { width: 3, height: 2 }, 0, 0, 360, 1],
        "big-delta": [{ x: 0, y: 0 }, { width: 4, height: 2 }, 0, 0, 360, 400],
        "empty-arc": [{ x: 0, y: 0 }, { width: 4, height: 2 }, 0, 10, 10, 30],
        "zero-axes": [{ x: 0, y: 0 }, { width: 0, height: 0 }, 0, 0, 360, 90],
        "neg-center": [{ x: -5, y: -5 }, { width: 4, height: 2 }, 0, 0, 360, 90],
        "float-args": [{ x: 1.9, y: 2.1 }, { width: 4.6, height: 2.4 }, 0.9, 0.2, 360.7, 90.3],
        "bad-delta": [{ x: 0, y: 0 }, { width: 4, height: 2 }, 0, 0, 360, 0],
      }[variant];
      cv.ellipse2Poly(args[0], args[1], args[2], args[3], args[4], args[5], vector);
      return { points: vecPoints(vector) };
    });
  }

  // ---- equalizeHist: histogram shapes, constants, channel/empty errors
  for (const [
    name,
    values,
    rows,
    columns,
    channels,
  ] of /** @type {[string, number[], number, number, number][]} */ ([
    ["gradient", [0, 1, 2, 3, 4, 5, 6, 7], 1, 8, 1],
    ["clustered", [0, 0, 1, 1, 2, 3, 3, 3], 1, 8, 1],
    ["constant", [7, 7, 7, 7], 1, 4, 1],
    ["two-peak", [0, 0, 0, 255, 255, 255], 2, 3, 1],
    ["wide", arrays[0](48), 6, 8, 1],
    ["three-channel", arrays[0](24), 2, 4, 3],
    ["bad-depth", arrays[5](8), 2, 4, 1],
    ["roi", arrays[0](36), 6, 6, 1],
  ])) {
    run(`equalizeHist/${name}`, (handles) => {
      let source = track(
        handles,
        api.mat(rows, columns, channels, name === "bad-depth" ? 5 : 0, values),
      );
      if (name === "roi") {
        source = api.roi(track(handles, source), 1, 1, 4, 4);
        handles.push(source);
      }
      const destination = track(handles, api.empty());
      cv.equalizeHist(source, destination);
      return { destination: api.summary(destination) };
    });
  }

  // ---- erode/dilate: depths, kernel masks, anchors, iterations, borders
  const kernelMasks = {
    full: [1, 1, 1, 1, 1, 1, 1, 1, 1],
    cross: [0, 1, 0, 1, 1, 1, 0, 1, 0],
    diag: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    row: [0, 0, 0, 1, 1, 1, 0, 0, 0],
  };
  for (const op of ["erode", "dilate"])
    for (const depth of depths)
      for (const [kernelName, mask] of Object.entries(kernelMasks))
        for (const variant of ["base", "anchor", "iter2", "badDepth"])
          run(`${op}/${depth}/${kernelName}/${variant}`, (handles) => {
            const source = track(handles, api.mat(4, 5, 1, depth, arrays[depth](20)));
            const kernelDepth = variant === "badDepth" ? 4 : 0;
            const kernel = track(
              handles,
              api.mat(
                3,
                3,
                1,
                kernelDepth,
                mask.map((v) => v * 100),
              ),
            );
            const destination = track(handles, api.empty());
            if (variant === "anchor") {
              cv[op](source, destination, kernel, pt(0, 0));
            } else if (variant === "iter2") {
              cv[op](source, destination, kernel, pt(-1, -1), 2);
            } else {
              cv[op](source, destination, kernel);
            }
            return { destination: api.summary(destination) };
          });
  for (const op of ["erode", "dilate"])
    for (const borderType of [0, 1, 2, 4, 3, 99])
      for (const borderValue of ["default", "zero", "mid"]) {
        run(`${op}/border/${borderType}/${borderValue}`, (handles) => {
          const source = track(handles, api.mat(4, 5, 1, 0, arrays[0](20)));
          const kernel = track(
            handles,
            api.mat(
              3,
              3,
              1,
              0,
              kernelMasks.full.map((v) => v * 100),
            ),
          );
          const destination = track(handles, api.empty());
          if (borderValue === "default") {
            cv[op](source, destination, kernel, pt(-1, -1), 1, borderType);
          } else {
            cv[op](
              source,
              destination,
              kernel,
              pt(-1, -1),
              1,
              borderType,
              borderValue === "zero" ? [0, 0, 0, 0] : [50, 60, 70, 80],
            );
          }
          return { destination: api.summary(destination) };
        });
      }
  for (const op of ["erode", "dilate"])
    for (const channels of [3, 4])
      run(`${op}/channels/${channels}`, (handles) => {
        const source = track(handles, api.mat(4, 5, channels, 0, arrays[0](4 * 5 * channels)));
        const kernel = track(handles, api.mat(3, 3, 1, 0, kernelMasks.cross));
        const destination = track(handles, api.empty());
        cv[op](source, destination, kernel);
        return { destination: api.summary(destination) };
      });

  // ---- blur/boxFilter: depths, kernel sizes, anchors, borders, ddepth, normalize
  for (const depth of depths)
    for (const channels of [1, 3])
      for (const ksize of [
        [3, 3],
        [5, 1],
        [2, 4],
      ])
        run(`blur/${depth}/${channels}/${ksize[0]}x${ksize[1]}`, (handles) => {
          const source = track(
            handles,
            api.mat(5, 6, channels, depth, arrays[depth](5 * 6 * channels)),
          );
          const destination = track(handles, api.empty());
          cv.blur(source, destination, { width: ksize[0], height: ksize[1] });
          return { destination: api.summary(destination) };
        });
  for (const borderType of [0, 1, 2, 4, 3, 99])
    run(`blur/border/${borderType}`, (handles) => {
      const source = track(handles, api.mat(4, 5, 1, 0, arrays[0](20)));
      const destination = track(handles, api.empty());
      cv.blur(source, destination, { width: 3, height: 3 }, pt(-1, -1), borderType);
      return { destination: api.summary(destination) };
    });
  for (const anchorP of [
    [-1, -1],
    [0, 0],
    [2, 1],
    [3, 0],
    [0, 3],
  ])
    run(`blur/anchor/${anchorP[0]}x${anchorP[1]}`, (handles) => {
      const source = track(handles, api.mat(4, 5, 1, 0, arrays[0](20)));
      const destination = track(handles, api.empty());
      cv.blur(source, destination, { width: 3, height: 3 }, pt(anchorP[0], anchorP[1]));
      return { destination: api.summary(destination) };
    });
  for (const depth of depths)
    for (const ddepth of [-1, 0, 3, 5, 6])
      for (const normalize of [true, false])
        run(`boxFilter/${depth}/d${ddepth}/n${normalize ? 1 : 0}`, (handles) => {
          const source = track(handles, api.mat(4, 5, 1, depth, arrays[depth](20)));
          const destination = track(handles, api.empty());
          cv.boxFilter(source, destination, ddepth, { width: 3, height: 3 }, pt(-1, -1), normalize);
          return { destination: api.summary(destination) };
        });
  for (const ddepth of [9, 99])
    run(`boxFilter/badDepth/${ddepth}`, (handles) => {
      const source = track(handles, api.mat(4, 5, 1, 0, arrays[0](20)));
      const destination = track(handles, api.empty());
      cv.boxFilter(source, destination, ddepth, { width: 3, height: 3 });
      return { destination: api.summary(destination) };
    });

  // ---- medianBlur: depths, kernel sizes, channels
  for (const depth of depths)
    for (const ksize of [1, 3, 5, 7])
      for (const channels of [1, 3])
        run(`medianBlur/${depth}/${ksize}/${channels}`, (handles) => {
          const source = track(
            handles,
            api.mat(9, 8, channels, depth, arrays[depth](9 * 8 * channels)),
          );
          const destination = track(handles, api.empty());
          cv.medianBlur(source, destination, ksize);
          return { destination: api.summary(destination) };
        });
  for (const ksize of [0, 2, -3])
    run(`medianBlur/bad/${ksize}`, (handles) => {
      const source = track(handles, api.mat(4, 5, 1, 0, arrays[0](20)));
      const destination = track(handles, api.empty());
      cv.medianBlur(source, destination, ksize);
      return { destination: api.summary(destination) };
    });

  // ---- filter2D/sepFilter2D: depths, kernels, anchors, delta, borders
  for (const depth of depths)
    for (const ddepth of [-1, 3, 5])
      for (const [kernelDepth, kernelData] of /** @type {[number, number[]][]} */ ([
        [5, [1, 1, 1, 1, -1, 1, 1, 1, 1]],
        [6, [0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, 0.25]],
      ]))
        run(`filter2D/${depth}/d${ddepth}/k${kernelDepth}`, (handles) => {
          const source = track(handles, api.mat(4, 5, 1, depth, arrays[depth](20)));
          const kernel = track(handles, api.mat(3, 3, 1, kernelDepth, kernelData));
          const destination = track(handles, api.empty());
          cv.filter2D(source, destination, ddepth, kernel);
          return { destination: api.summary(destination) };
        });
  for (const [variant, args] of /** @type {[string, unknown[]][]} */ ([
    ["anchor", [pt(0, 0)]],
    ["delta", [pt(-1, -1), 12.5]],
    ["border3", [pt(-1, -1), 0, 3]],
    ["border1", [pt(-1, -1), 0, 1]],
  ]))
    run(`filter2D/variant/${variant}`, (handles) => {
      const source = track(handles, api.mat(4, 5, 1, 5, arrays[5](20)));
      const kernel = track(handles, api.mat(3, 3, 1, 5, [1, 1, 1, 1, 1, 1, 1, 1, 1]));
      const destination = track(handles, api.empty());
      cv.filter2D(source, destination, -1, kernel, ...args);
      return { destination: api.summary(destination) };
    });
  for (const depth of depths)
    for (const [kx, ky] of /** @type {[number[], number[]][]} */ ([
      [
        [1, 2, 1],
        [1, 2, 1],
      ],
      [
        [-1, 0, 1],
        [1, 2, 1],
      ],
    ]))
      run(`sepFilter2D/${depth}/${kx[0]}`, (handles) => {
        const source = track(handles, api.mat(4, 5, 1, depth, arrays[depth](20)));
        const kxMat = track(handles, api.mat(1, 3, 1, 6, kx));
        const kyMat = track(handles, api.mat(3, 1, 1, 6, ky));
        const destination = track(handles, api.empty());
        cv.sepFilter2D(source, destination, -1, kxMat, kyMat);
        return { destination: api.summary(destination) };
      });
  for (const depth of depths)
    for (const normalize of [true, false])
      for (const ddepth of [-1, 5])
        run(`sqrBoxFilter/${depth}/n${normalize ? 1 : 0}/d${ddepth}`, (handles) => {
          const source = track(handles, api.mat(4, 5, 1, depth, arrays[depth](20)));
          const destination = track(handles, api.empty());
          cv.sqrBoxFilter(
            source,
            destination,
            ddepth,
            { width: 3, height: 3 },
            pt(-1, -1),
            normalize,
          );
          return { destination: api.summary(destination) };
        });
  for (const depth of depths)
    for (const [kw, kh] of /** @type {[number, number][]} */ ([
      [3, 3],
      [5, 3],
    ]))
      run(`stackBlur/${depth}/${kw}x${kh}`, (handles) => {
        const source = track(handles, api.mat(6, 7, 1, depth, arrays[depth](42)));
        const destination = track(handles, api.empty());
        cv.stackBlur(source, destination, { width: kw, height: kh });
        return { destination: api.summary(destination) };
      });
  for (const depth of [0, 5])
    for (const [d, sc, ss] of /** @type {[number, number, number][]} */ ([
      [3, 50, 50],
      [5, 80, 30],
      [-1, 50, 50],
    ]))
      run(`bilateralFilter/${depth}/${d}/${sc}/${ss}`, (handles) => {
        const source = track(handles, api.mat(5, 6, 1, depth, arrays[depth](30)));
        const destination = track(handles, api.empty());
        cv.bilateralFilter(source, destination, d, sc, ss);
        return { destination: api.summary(destination) };
      });
  for (const depth of [1, 2])
    run(`bilateralFilter/badDepth/${depth}`, (handles) => {
      const source = track(handles, api.mat(4, 5, 1, depth, arrays[depth](20)));
      const destination = track(handles, api.empty());
      cv.bilateralFilter(source, destination, 3, 50, 50);
      return { destination: api.summary(destination) };
    });

  // ---- pyrDown/pyrUp: depths, sizes, borders
  for (const depth of depths)
    for (const op of ["pyrDown", "pyrUp"])
      run(`${op}/${depth}`, (handles) => {
        const source = track(handles, api.mat(6, 7, 1, depth, arrays[depth](42)));
        const destination = track(handles, api.empty());
        cv[op](source, destination);
        return { destination: api.summary(destination) };
      });
  for (const [name, size] of /** @type {[string, { width: number, height: number }][]} */ ([
    ["exact", { width: 4, height: 3 }],
    ["loose", { width: 3, height: 3 }],
    ["bad", { width: 8, height: 4 }],
  ]))
    run(`pyrDown/size/${name}`, (handles) => {
      const source = track(handles, api.mat(6, 7, 1, 0, arrays[0](42)));
      const destination = track(handles, api.empty());
      cv.pyrDown(source, destination, size);
      return { destination: api.summary(destination) };
    });
  for (const [name, size] of /** @type {[string, { width: number, height: number }][]} */ ([
    ["exact", { width: 14, height: 12 }],
    ["loose", { width: 13, height: 11 }],
    ["bad", { width: 20, height: 12 }],
  ]))
    run(`pyrUp/size/${name}`, (handles) => {
      const source = track(handles, api.mat(6, 7, 1, 0, arrays[0](42)));
      const destination = track(handles, api.empty());
      cv.pyrUp(source, destination, size);
      return { destination: api.summary(destination) };
    });
  for (const borderType of [0, 1, 2, 4])
    run(`pyrDown/border/${borderType}`, (handles) => {
      const source = track(handles, api.mat(6, 7, 1, 0, arrays[0](42)));
      const destination = track(handles, api.empty());
      cv.pyrDown(source, destination, { width: -1, height: -1 }, borderType);
      return { destination: api.summary(destination) };
    });

  // ---- Laplacian/Scharr: depths, ksizes, ddepth, scale/delta, borders
  for (const depth of depths)
    for (const ddepth of [-1, 3, 6])
      for (const ksize of [1, 3, 5])
        run(`Laplacian/${depth}/d${ddepth}/k${ksize}`, (handles) => {
          const source = track(handles, api.mat(5, 6, 1, depth, arrays[depth](30)));
          const destination = track(handles, api.empty());
          cv.Laplacian(source, destination, ddepth, ksize);
          return { destination: api.summary(destination) };
        });
  for (const [name, ksize, scale, delta] of [
    ["scaled", 3, 0.5, 0],
    ["delta", 1, 1, 7.5],
    ["bad", 4, 1, 0],
  ])
    run(`Laplacian/variant/${name}`, (handles) => {
      const source = track(handles, api.mat(5, 6, 1, 5, arrays[5](30)));
      const destination = track(handles, api.empty());
      cv.Laplacian(source, destination, -1, ksize, scale, delta);
      return { destination: api.summary(destination) };
    });
  for (const depth of depths)
    for (const [dx, dy] of /** @type {[number, number][]} */ ([
      [1, 0],
      [0, 1],
    ]))
      for (const ddepth of [-1, 3])
        run(`Scharr/${depth}/${dx}${dy}/d${ddepth}`, (handles) => {
          const source = track(handles, api.mat(5, 6, 1, depth, arrays[depth](30)));
          const destination = track(handles, api.empty());
          cv.Scharr(source, destination, ddepth, dx, dy);
          return { destination: api.summary(destination) };
        });
  for (const [dx, dy] of /** @type {[number, number][]} */ ([
    [1, 1],
    [2, 0],
    [-1, 0],
  ]))
    run(`Scharr/bad/${dx}${dy}`, (handles) => {
      const source = track(handles, api.mat(5, 6, 1, 0, arrays[0](30)));
      const destination = track(handles, api.empty());
      cv.Scharr(source, destination, -1, dx, dy);
      return { destination: api.summary(destination) };
    });
  for (const depth of depths)
    run(`spatialGradient/${depth}`, (handles) => {
      const source = track(handles, api.mat(5, 6, 1, depth, arrays[depth](30)));
      const dxMat = track(handles, api.empty());
      const dyMat = track(handles, api.empty());
      cv.spatialGradient(source, dxMat, dyMat);
      return { dx: api.summary(dxMat), dy: api.summary(dyMat) };
    });

  // ---- corner detectors: depths, ksizes
  for (const depth of depths)
    for (const ksize of [3, 5])
      run(`preCornerDetect/${depth}/k${ksize}`, (handles) => {
        const source = track(handles, api.mat(5, 6, 1, depth, arrays[depth](30)));
        const destination = track(handles, api.empty());
        cv.preCornerDetect(source, destination, ksize);
        return { destination: api.summary(destination) };
      });
  for (const depth of depths)
    for (const [block, ksize, k] of /** @type {[number, number, number][]} */ ([
      [2, 3, 0.04],
      [3, 3, 0.05],
    ]))
      run(`cornerHarris/${depth}/b${block}/k${ksize}`, (handles) => {
        const source = track(handles, api.mat(6, 7, 1, depth, arrays[depth](42)));
        const destination = track(handles, api.empty());
        cv.cornerHarris(source, destination, block, ksize, k);
        return { destination: api.summary(destination) };
      });
  for (const depth of depths)
    for (const [block, ksize] of /** @type {[number, number][]} */ ([
      [3, 3],
      [5, 3],
    ]))
      run(`cornerMinEigenVal/${depth}/b${block}/k${ksize}`, (handles) => {
        const source = track(handles, api.mat(6, 7, 1, depth, arrays[depth](42)));
        const destination = track(handles, api.empty());
        cv.cornerMinEigenVal(source, destination, block, ksize);
        return { destination: api.summary(destination) };
      });

  // ---- integral/integral2: depths, channels, requested depths
  for (const depth of depths)
    for (const channels of [1, 3])
      for (const sdepth of [-1, 4, 5, 6])
        run(`integral/${depth}/${channels}/d${sdepth}`, (handles) => {
          const source = track(
            handles,
            api.mat(3, 4, channels, depth, arrays[depth](3 * 4 * channels)),
          );
          const sum = track(handles, api.empty());
          cv.integral(source, sum, sdepth);
          return { sum: api.summary(sum) };
        });
  for (const depth of depths)
    for (const sdepth of [-1, 4])
      for (const sqdepth of [-1, 5])
        run(`integral2/${depth}/d${sdepth}/q${sqdepth}`, (handles) => {
          const source = track(handles, api.mat(3, 4, 1, depth, arrays[depth](12)));
          const sum = track(handles, api.empty());
          const squared = track(handles, api.empty());
          cv.integral2(source, sum, squared, sdepth, sqdepth);
          return { sum: api.summary(sum), squared: api.summary(squared) };
        });

  // ---- moments: depths, binary flag, channel/empty errors
  for (const depth of depths)
    for (const channels of [1, 3])
      for (const binary of [false, true])
        run(`moments/${depth}/${channels}/b${binary ? 1 : 0}`, (handles) => {
          const source = track(
            handles,
            api.mat(4, 5, channels, depth, arrays[depth](4 * 5 * channels)),
          );
          const m = cv.moments(source, binary);
          return {
            m00: m.m00,
            m10: m.m10,
            m01: m.m01,
            m20: m.m20,
            m11: m.m11,
            m02: m.m02,
            m30: m.m30,
            m21: m.m21,
            m12: m.m12,
            m03: m.m03,
            mu20: m.mu20,
            mu11: m.mu11,
            mu02: m.mu02,
            mu30: m.mu30,
            mu21: m.mu21,
            mu12: m.mu12,
            mu03: m.mu03,
            nu20: m.nu20,
            nu11: m.nu11,
            nu02: m.nu02,
            nu30: m.nu30,
            nu21: m.nu21,
            nu12: m.nu12,
            nu03: m.nu03,
          };
        });
  for (const depth of depths)
    run(`HuMoments/${depth}`, (handles) => {
      const source = track(handles, api.mat(4, 5, 1, depth, arrays[depth](20)));
      const m = cv.moments(source);
      const hu = track(handles, api.empty());
      cv.HuMoments(m, hu);
      return { hu: api.summary(hu) };
    });

  // ---- matchShapes: contour pairs, methods, images
  const contourA = [0, 0, 10, 0, 10, 10, 0, 10];
  const contourB = [0, 0, 20, 0, 20, 20, 0, 20];
  const contourC = [0, 0, 10, 0, 20, 5, 10, 10, 0, 10];
  for (const method of [1, 2, 3, 0, 7])
    for (const [name, pair] of /** @type {[string, number[][]][]} */ ([
      ["same", [contourA, contourA]],
      ["scaled", [contourA, contourB]],
      ["different", [contourA, contourC]],
    ]))
      run(`matchShapes/${method}/${name}`, (handles) => {
        const first = track(handles, api.mat(pair[0].length / 2, 1, 2, 4, pair[0]));
        const second = track(handles, api.mat(pair[1].length / 2, 1, 2, 4, pair[1]));
        // oxlint-disable-next-line anti-slop/no-shape-in-symbol-names -- Upstream binding name.
        return { score: cv.matchShapes(first, second, method, 0) };
      });
  for (const depth of [4, 5])
    run(`matchShapes/float/${depth}`, (handles) => {
      const first = track(handles, api.mat(5, 1, 2, depth, [0, 0, 4, 0, 4, 4, 0, 4, 2, 2]));
      const second = track(handles, api.mat(5, 1, 2, depth, [0, 0, 4, 0, 4, 4, 0, 4, 3, 3]));
      // oxlint-disable-next-line anti-slop/no-shape-in-symbol-names -- Upstream binding name.
      return { score: cv.matchShapes(first, second, 2, 0.5) };
    });

  // ---- getFontScaleFromHeight: faces, heights, thickness
  for (const face of [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 24, -1])
    for (const height of [0, 12, 20, -5])
      run(`fontScale/${face}/${height}`, () => ({
        scale: cv.getFontScaleFromHeight(face, height, -1),
      }));
  for (const thickness of [-1, 0, 1, 2, 3, 10])
    run(`fontScale/thickness/${thickness}`, () => ({
      scale: cv.getFontScaleFromHeight(0, 20, thickness),
    }));

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
      clipLine: cv.clipLine.length,
      copyMakeBorder: cv.copyMakeBorder.length,
      ellipse2Poly: cv.ellipse2Poly.length,
      equalizeHist: cv.equalizeHist.length,
      erode: cv.erode.length,
      dilate: cv.dilate.length,
      blur: cv.blur.length,
      boxFilter: cv.boxFilter.length,
      medianBlur: cv.medianBlur.length,
      integral: cv.integral.length,
      integral2: cv.integral2.length,
      moments: cv.moments.length,
      HuMoments: cv.HuMoments.length,
      // oxlint-disable-next-line anti-slop/no-shape-in-symbol-names -- Upstream binding name.
      matchShapes: cv.matchShapes.length,
      getFontScaleFromHeight: cv.getFontScaleFromHeight.length,
      filter2D: cv.filter2D.length,
      sepFilter2D: cv.sepFilter2D.length,
      sqrBoxFilter: cv.sqrBoxFilter.length,
      stackBlur: cv.stackBlur.length,
      bilateralFilter: cv.bilateralFilter.length,
      pyrDown: cv.pyrDown.length,
      pyrUp: cv.pyrUp.length,
      Laplacian: cv.Laplacian.length,
      Scharr: cv.Scharr.length,
      spatialGradient: cv.spatialGradient.length,
      preCornerDetect: cv.preCornerDetect.length,
      cornerHarris: cv.cornerHarris.length,
      cornerMinEigenVal: cv.cornerMinEigenVal.length,
    },
  }));

  return entries;
}
