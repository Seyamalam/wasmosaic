export function contourCases() {
  const cases = [];
  const mask = (name, rows, columns, values, extra = {}) => {
    for (const mode of [0, 1, 2, 3])
      for (const method of [1, 2])
        cases.push({
          name: `${name}/${mode}/${method}`,
          kind: "contours",
          rows,
          columns,
          values,
          mode,
          method,
          ...extra,
        });
  };
  for (let bits = 0; bits < 512; bits++)
    mask(
      `3x3-${bits}`,
      3,
      3,
      Array.from({ length: 9 }, (_, i) => (bits >> i) & 1),
    );
  const nested = Array.from({ length: 17 * 19 }, (_, i) => {
    const x = i % 19,
      y = Math.floor(i / 19);
    const distance = Math.min(x, y, 18 - x, 16 - y);
    return distance % 3 !== 0 ? 127 : 0;
  });
  mask("nested", 17, 19, nested);
  mask("offset", 17, 19, nested, { offset: { x: -31, y: 42 } });
  mask("source-roi", 17, 19, nested, { roi: true });
  mask("source-hierarchy-alias", 17, 19, nested, { alias: true });
  mask("empty", 0, 0, []);
  mask("single-row", 1, 9, [1, 1, 0, 1, 0, 1, 1, 1, 0]);
  for (let seed = 1; seed <= 32; seed++) {
    let state = seed;
    const values = Array.from({ length: 17 * 19 }, () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state >>> 28 > seed % 12 ? 255 : 0;
    });
    mask(`random-${seed}`, 17, 19, values);
  }
  const pipeline = Array.from({ length: 49 }, (_, i) => {
    const x = i % 7,
      y = Math.floor(i / 7);
    return x >= 1 && x <= 5 && y >= 1 && y <= 5 ? 255 : 0;
  });
  mask("polygon-workflow", 7, 7, pipeline, { pipeline: true });
  mask("offset-fractional", 3, 3, [0, 0, 0, 0, 1, 1, 0, 1, 1], { offset: { x: 1.7, y: -2.9 } });
  mask("offset-overflow", 3, 3, [0, 0, 0, 0, 1, 1, 0, 1, 1], {
    offset: { x: 2147483647, y: -2147483648 },
  });
  for (const [name, mode, method] of [
    ["invalid-mode", 99, 2],
    ["invalid-method", 3, 99],
  ])
    cases.push({
      name,
      kind: "contours",
      rows: 3,
      columns: 3,
      values: Array.from({ length: 9 }, () => 1),
      mode,
      method,
    });
  const curves = {
    rectangle: [0, 0, 0, 2, 0, 4, 4, 4, 4, 0],
    wave: [0, 0, 1, 0.1, 2, 0, 3, 4, 4, 0],
    triangle: [0, 0, 3, 6, 6, 0],
    line: [0, 0, 1, 0, 2, 0, 3, 0],
    duplicates: [2, 3, 2, 3, 2, 3],
    singleton: [2, 3],
    pair: [2, 3, 4, 8],
    repeatedEnd: [0, 0, 0, 4, 4, 4, 4, 0, 0, 0],
    nearLine: [0, 0, 1, 0.5, 2, 0, 3, -0.5, 4, 0],
    slopingLine: [0, 0, 1, 3, 2, 6, 3, 9],
    reversal: [0, 0, 5, 0, 1, 0],
    concave: [0, 0, 1, 4, 2, 1, 4, 4, 6, 0],
  };
  for (const [name, values] of Object.entries(curves))
    for (const depth of [4, 5])
      for (const closed of [false, true])
        for (const epsilon of [0, 0.2, 1, 10]) {
          cases.push({
            name: `approx/${name}/${depth}/${closed}/${epsilon}`,
            kind: "approx",
            values,
            depth,
            closed,
            epsilon,
          });
        }
  for (const layout of ["row", "pairs", "roi", "destination-roi", "alias", "overlap"])
    cases.push({
      name: `approx/layout/${layout}`,
      kind: "approx",
      values: curves.rectangle,
      depth: 4,
      closed: true,
      epsilon: 0.5,
      layout,
    });
  for (const epsilon of [-1, Infinity, NaN])
    cases.push({
      name: `approx/epsilon/${epsilon}`,
      kind: "approx",
      values: curves.triangle,
      depth: 4,
      closed: true,
      epsilon,
    });
  cases.push({
    name: "approx/empty",
    kind: "approx",
    values: [],
    depth: 4,
    closed: false,
    epsilon: 1,
  });
  cases.push({
    name: "approx/nonfinite",
    kind: "approx",
    values: [0, 0, NaN, 2, 3, 4],
    depth: 5,
    closed: false,
    epsilon: 1,
  });
  cases.push({
    name: "approx/depth-u8",
    kind: "approx",
    values: curves.triangle,
    depth: 0,
    closed: false,
    epsilon: 1,
  });
  for (const [name, epsilon, closed, arity] of [
    ["missing", 1, true, 3],
    ["extra", 1, true, 5],
    ["undefined", undefined, true, 4],
    ["string", "0.5", true, 4],
    ["boolean", true, "closed", 4],
    ["null", null, false, 4],
  ])
    cases.push({
      name: `approx/binding/${name}`,
      kind: "approx",
      values: curves.rectangle,
      depth: 4,
      epsilon,
      closed,
      arity,
    });
  return cases;
}

export function runContourCases(api) {
  const results = contourCases().map((entry) => {
    const handles = new Set();
    const owned = (handle) => {
      handles.add(handle);
      return handle;
    };
    const mat = (...args) => owned(api.mat(...args));
    try {
      let result;
      if (entry.kind === "contours") {
        let source = mat(entry.rows, entry.columns, 1, 0, entry.values);
        if (entry.roi) {
          const parent = mat(
            entry.rows + 2,
            entry.columns + 2,
            1,
            0,
            Array.from({ length: (entry.rows + 2) * (entry.columns + 2) }, (_, i) => {
              const x = (i % (entry.columns + 2)) - 1,
                y = Math.floor(i / (entry.columns + 2)) - 1;
              return x >= 0 && x < entry.columns && y >= 0 && y < entry.rows
                ? entry.values[y * entry.columns + x]
                : 99;
            }),
          );
          source = owned(api.roi(parent, 1, 1, entry.rows, entry.columns));
        }
        const vector = owned(api.vector());
        const hierarchy = entry.alias ? source : owned(api.empty());
        api.find(
          source,
          vector,
          hierarchy,
          entry.mode,
          entry.method,
          entry.offset ?? { x: 0, y: 0 },
        );
        const contours = [];
        for (let i = 0; i < api.size(vector); i++) {
          const contour = owned(api.get(vector, i));
          const summary = api.summary(contour);
          if (entry.pipeline) {
            const polygon = owned(api.empty());
            api.approx(contour, polygon, 0.01 * api.arcLength(contour, true), true);
            summary.polygon = api.summary(polygon);
            summary.area = api.area(polygon, false);
            summary.bounds = api.bounds(polygon);
          }
          contours.push(summary);
        }
        result = { contours, hierarchy: api.summary(hierarchy), source: api.summary(source) };
      } else {
        const count = entry.values.length / 2;
        let source = mat(
          entry.layout === "row" ? 1 : count,
          entry.layout === "row" ? count : entry.layout === "pairs" ? 2 : 1,
          entry.layout === "pairs" ? 1 : 2,
          entry.depth,
          entry.values,
        );
        let destination = owned(api.empty());
        if (entry.layout === "roi") {
          const parent = mat(
            count,
            2,
            2,
            entry.depth,
            entry.values.flatMap((value, i) => (i % 2 ? [value, 99, 99] : [value])),
          );
          source = owned(api.roi(parent, 0, 0, count, 1));
        }
        if (entry.layout === "destination-roi") {
          const parent = mat(
            8,
            3,
            2,
            entry.depth,
            Array.from({ length: 48 }, () => 99),
          );
          destination = owned(api.roi(parent, 1, 1, 4, 1));
        }
        if (entry.layout === "alias") destination = source;
        if (entry.layout === "overlap") destination = owned(api.roi(source, 0, 0, 4, 1));
        const args = [source, destination, entry.epsilon, entry.closed];
        if (entry.arity === 5) args.push(null);
        api.approx(...(entry.arity === 3 ? args.slice(0, 3) : args));
        result = api.summary(destination);
      }
      return { name: entry.name, result };
    } catch {
      return { name: entry.name, error: true };
    } finally {
      for (const handle of [...handles].toReversed()) api.dispose(handle);
    }
  });
  return [{ name: "binding-contract", result: api.contract() }, ...results];
}
