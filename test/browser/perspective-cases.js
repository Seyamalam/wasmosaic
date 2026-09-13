const square = [0, 0, 6, 0, 6, 4, 0, 4];
const quadrilateral = [1, 0, 5, 1, 6, 4, 0, 3];
const transforms = {
  identity: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  translation: [1, 0, 1, 0, 1, -1, 0, 0, 1],
  fractional: [1, 0, 0.5, 0, 1, -0.25, 0, 0, 1],
  projective: [1.1, 0.1, 0.2, -0.08, 0.9, 0.4, 0.02, -0.01, 1],
  horizon: [1, 0, 0, 0, 1, 0, 0.25, 0, -1],
};
export function perspectiveCases() {
  const entries = [];
  for (const depth of [0, 1, 2, 3, 4, 5, 6])
    for (const channels of [1, 4])
      for (const interpolation of [0, 1, 3])
        for (const border of [0, 1, 2, 3, 4])
          for (const [transformName, transform] of Object.entries(transforms))
            for (const inverse of [0, 16])
              entries.push({
                kind: "warp",
                name: `warp/${depth}/${channels}/${interpolation}/${border}/${transformName}/${inverse}`,
                depth,
                channels,
                flags: interpolation | inverse,
                border,
                transform,
              });
  for (const layout of [
    "source-roi",
    "destination-roi",
    "in-place",
    "overlap",
    "transform-roi",
    "transform-f32",
    "zero-size",
    "negative-size",
    "default",
    "omitted-border",
    "omitted-value",
  ])
    entries.push({
      kind: "warp",
      name: `warp/layout/${layout}`,
      depth: 0,
      channels: 4,
      flags: 1,
      border: 0,
      transform: transforms.projective,
      layout,
    });
  entries.push({
    kind: "warp",
    name: "warp/singular",
    depth: 0,
    channels: 1,
    flags: 1,
    border: 0,
    transform: [0, 0, 0, 0, 0, 0, 0, 0, 0],
  });
  for (const interpolation of [5, 6, 7])
    entries.push({
      kind: "warp",
      name: `warp/invalid-interpolation/${interpolation}`,
      depth: 0,
      channels: 1,
      flags: interpolation,
      border: 0,
      transform: transforms.identity,
    });
  for (const border of [-1, 16, 99])
    entries.push({
      kind: "warp",
      name: `warp/invalid-border/${border}`,
      depth: 0,
      channels: 1,
      flags: 1,
      border,
      transform: transforms.translation,
    });
  for (const arity of [3, 8])
    entries.push({
      kind: "warp",
      name: `warp/arity/${arity}`,
      depth: 0,
      channels: 1,
      flags: 1,
      border: 0,
      transform: transforms.identity,
      arity,
    });
  for (const depth of [5, 6, 4])
    for (const layout of ["column", "row", "pairs", "strided"])
      for (const method of [0, 4, 16, 20])
        entries.push({
          kind: "get",
          name: `get/${depth}/${layout}/${method}`,
          depth,
          layout,
          method,
          source: square,
          target: quadrilateral,
        });
  for (const name of [
    "default",
    "same-input",
    "zero",
    "collinear",
    "nan",
    "infinity",
    "missing",
    "extra",
    "undefined",
    "string",
    "boolean",
    "svd",
    "eigen",
    "cholesky",
    "invalid-method",
  ]) {
    const methods = {
      svd: 1,
      eigen: 2,
      cholesky: 3,
      "invalid-method": 99,
      undefined: undefined,
      string: "0",
      boolean: true,
    };
    entries.push({
      kind: "get",
      name: `get/${name}`,
      depth: 5,
      layout: "column",
      method: name in methods ? methods[name] : 0,
      source:
        name === "zero"
          ? Array.from({ length: 8 }, () => 0)
          : name === "collinear"
            ? [0, 0, 1, 0, 2, 0, 3, 0]
            : square,
      target:
        name === "same-input"
          ? square
          : name === "nan"
            ? [NaN, ...quadrilateral.slice(1)]
            : name === "infinity"
              ? [Infinity, ...quadrilateral.slice(1)]
              : quadrilateral,
      special: name,
    });
  }
  for (const method of [0, 4])
    entries.push({ kind: "workflow", name: `rectification/${method}`, method });
  return entries;
}

export function runPerspectiveCases(api) {
  const results = perspectiveCases().map((entry) => {
    const owned = [];
    const own = (mat) => {
      owned.push(mat);
      return mat;
    };
    const mat = (...args) => own(api.mat(...args));
    try {
      if (entry.kind === "get") {
        const count = entry.source.length / 2;
        const make = (values) =>
          mat(
            entry.layout === "row" ? 1 : count,
            entry.layout === "row" ? count : entry.layout === "pairs" ? 2 : 1,
            entry.layout === "pairs" ? 1 : 2,
            entry.depth,
            values,
          );
        let source = make(entry.source);
        const target = make(entry.target);
        if (entry.layout === "strided")
          source = own(
            api.roi(
              mat(
                4,
                2,
                2,
                entry.depth,
                entry.source.flatMap((v, i) => (i % 2 ? [v, 99, 99] : [v])),
              ),
              0,
              0,
              4,
              1,
            ),
          );
        const args = [source, target, entry.method];
        if (entry.special === "default") args.pop();
        if (entry.special === "missing") args.splice(1);
        if (entry.special === "extra") args.push(99);
        return {
          name: entry.name,
          result: api.summary(own(api.get(...args))),
          source: api.summary(source),
          target: api.summary(target),
        };
      }
      const depth = entry.depth ?? 0,
        channels = entry.channels ?? 4;
      const values = Array.from(
        { length: 5 * 7 * channels },
        (_, i) => ((i * 29 + 17) % 127) + (depth >= 5 ? (i % 3) * 0.13 : 0),
      );
      let source = mat(5, 7, channels, depth, values);
      let destination = own(api.empty());
      let transform =
        entry.kind === "workflow"
          ? own(api.get(mat(4, 1, 2, 5, quadrilateral), mat(4, 1, 2, 5, square), entry.method))
          : mat(3, 3, 1, entry.layout === "transform-f32" ? 5 : 6, entry.transform);
      let size = { width: 8, height: 6 };
      if (entry.layout === "source-roi") {
        const parent = mat(
          7,
          9,
          channels,
          depth,
          Array.from({ length: 7 * 9 * channels }, (_, i) => {
            const pixel = Math.floor(i / channels),
              x = (pixel % 9) - 1,
              y = Math.floor(pixel / 9) - 1;
            return x >= 0 && x < 7 && y >= 0 && y < 5
              ? values[(y * 7 + x) * channels + (i % channels)]
              : 99;
          }),
        );
        source = own(api.roi(parent, 1, 1, 5, 7));
      }
      let parent;
      if (entry.layout === "destination-roi") {
        parent = mat(
          8,
          10,
          channels,
          depth,
          Array.from({ length: 80 * channels }, () => 99),
        );
        destination = own(api.roi(parent, 1, 1, 6, 8));
      }
      if (entry.layout === "in-place") {
        destination = source;
        size = { width: 7, height: 5 };
      }
      if (entry.layout === "overlap") {
        destination = own(api.roi(source, 0, 0, 4, 6));
        size = { width: 6, height: 4 };
      }
      if (entry.layout === "transform-roi") {
        const p = mat(
          3,
          4,
          1,
          6,
          entry.transform.flatMap((v, i) => (i % 3 === 2 ? [v, 99] : [v])),
        );
        transform = own(api.roi(p, 0, 0, 3, 3));
      }
      if (entry.layout === "zero-size") size = { width: 0, height: 3 };
      if (entry.layout === "negative-size") size = { width: -3, height: 2 };
      const args = [
        source,
        destination,
        transform,
        size,
        entry.flags ?? 1,
        entry.border ?? 0,
        [9.8, 33, -7, 254.8],
      ];
      if (entry.layout === "default") args.splice(4);
      if (entry.layout === "omitted-border") args.splice(5);
      if (entry.layout === "omitted-value") args.splice(6);
      if (entry.arity === 3) args.splice(3);
      if (entry.arity === 8) args.push(99);
      api.warp(...args);
      const result = {
        name: entry.name,
        result: api.summary(destination),
        source: api.summary(source),
      };
      if (parent) result.parent = api.summary(parent);
      return result;
    } catch {
      return { name: entry.name, error: true };
    } finally {
      for (const handle of [...new Set(owned)].toReversed()) api.dispose(handle);
    }
  });
  return [{ name: "binding-contract", result: api.contract() }, ...results];
}

export function comparePerspective(actual, reference, known) {
  const mismatches = [];
  let numericCases = 0,
    errorCases = 0,
    maximumError = 0;
  // oxlint-disable anti-slop/no-runtime-typeof -- Compare serialized browser boundary results, including heterogeneous headers and numeric arrays.
  const equal = (a, b, depth) => {
    if (typeof a === "number" && typeof b === "number") {
      const delta = Math.abs(a - b);
      maximumError = Math.max(maximumError, delta);
      return delta <= (depth === 5 ? 2e-5 : depth === 6 ? 2e-9 : 0) * Math.max(1, Math.abs(b));
    }
    if (a === b) return true;
    if (Array.isArray(a) && Array.isArray(b))
      return a.length === b.length && a.every((v, i) => equal(v, b[i], depth));
    if (a && b && typeof a === "object" && typeof b === "object")
      return (
        Object.keys(a).length === Object.keys(b).length &&
        Object.keys(a).every((key) => equal(a[key], b[key], a.depth ?? depth))
      );
    return false;
  };
  if (actual.length !== reference.length) throw new Error("Case counts differ");
  for (let i = 0; i < actual.length; i++) {
    if (actual[i].name !== reference[i].name) throw new Error("Case ordering differs");
    if (!equal(actual[i], reference[i]))
      mismatches.push({ actual: actual[i], expected: reference[i] });
    else if (actual[i].error) errorCases++;
    else numericCases++;
  }
  const recorded = new Map(known.map((entry) => [entry.actual.name, entry]));
  const knownDifferences = mismatches.map((entry) => {
    const expected = recorded.get(entry.actual.name);
    if (
      !expected ||
      JSON.stringify(entry.actual) !== JSON.stringify(expected.actual) ||
      JSON.stringify(entry.expected) !== JSON.stringify(expected.expected)
    ) {
      throw new Error(`Unrecorded difference: ${JSON.stringify(entry)}`);
    }
    recorded.delete(entry.actual.name);
    return { name: entry.actual.name, reason: expected.reason };
  });
  if (recorded.size) throw new Error("Recorded differences changed; update the ledger");
  return { cases: actual.length, numericCases, errorCases, maximumError, knownDifferences };
}
