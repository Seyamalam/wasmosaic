// Original fixtures shared by the package runner and the isolated reference worker.
export function templateMatchingCases(api) {
  const results = [];
  let state = 0x5a17;
  function values(length, depth) {
    return Array.from({ length }, () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return depth === 0 ? state >>> 24 : ((state >>> 20) - 2048) / 256;
    });
  }

  function match(label, image, template, method, mask, destination = api.empty()) {
    try {
      if (mask === undefined) api.match(image, template, destination, method);
      else api.match(image, template, destination, method, mask);
      results.push({ label, ...api.summary(destination) });
    } finally {
      if (destination !== image && destination !== template && destination !== mask) {
        api.dispose(destination);
      }
    }
  }

  for (const depth of [0, 5]) {
    for (const channels of [1, 2, 3, 4]) {
      const image = api.mat(7, 9, channels, depth, values(63 * channels, depth));
      const template = api.roi(image, 2, 3, 3, 4);
      const mask = api.mat(3, 4, 1, 0, [0, 2, 255, 0, 9, 8, 7, 6, 0, 1, 3, 5]);
      const weighted = api.mat(
        3,
        4,
        channels,
        5,
        Array.from({ length: 12 * channels }, (_, i) => (i % 5) / 4),
      );
      const before = api.summary(image);
      const maskBefore = api.summary(mask);
      for (let method = 0; method < 6; method += 1) {
        const prefix = `depth=${depth} channels=${channels} method=${method}`;
        match(`${prefix} unmasked`, image, template, method);
        match(`${prefix} binary mask`, image, template, method, mask);
        match(`${prefix} weighted mask`, image, template, method, weighted);
      }
      const locationResult = api.empty();
      api.match(image, template, locationResult, 0);
      const location = api.minMaxLoc(locationResult).minLoc;
      if (location.x !== 3 || location.y !== 2) throw new Error("Planted template was not located");
      if (JSON.stringify(api.summary(image)) !== JSON.stringify(before))
        throw new Error("Source image changed");
      if (JSON.stringify(api.summary(mask)) !== JSON.stringify(maskBefore))
        throw new Error("Binary mask changed");
      for (const matrix of [image, template, mask, weighted, locationResult]) api.dispose(matrix);
    }
  }

  for (const { label, input, pattern } of [
    { label: "ordinary", input: [1, 2, 3, 4], pattern: [1, 2] },
    { label: "zero template", input: [1, 2, 3], pattern: [0, 0] },
    { label: "zero image", input: [0, 0, 0], pattern: [1, 2] },
    { label: "constant template", input: [1, 2, 3], pattern: [2, 2] },
    { label: "constant image", input: [2, 2, 2], pattern: [1, 2] },
    { label: "negative correlation", input: [1, -1, 1], pattern: [-1, 1] },
    { label: "single pixel", input: [2, -3, 4], pattern: [2] },
    { label: "swapped inputs", input: [1, 2], pattern: [1, 2, 3] },
  ]) {
    const image = api.mat(1, input.length, 1, 5, input);
    const template = api.mat(1, pattern.length, 1, 5, pattern);
    const empty = api.empty();
    for (let method = 0; method < 6; method += 1) {
      match(`${label} method=${method}`, image, template, method);
      match(`${label} empty mask method=${method}`, image, template, method, empty);
    }
    for (const matrix of [image, template, empty]) api.dispose(matrix);
  }

  for (const method of [0, 1, 2, 3, 4, 5]) {
    const image = api.mat(1, 3, 1, 0, [1, 2, 3]);
    const template = api.mat(1, 2, 1, 0, [1, 2]);
    const zero = api.mat(1, 2, 1, 0, [0, 0]);
    const ones = api.mat(1, 2, 1, 0, [1, 1]);
    match(`all-zero mask method=${method}`, image, template, method, zero);
    match(`masked zero template method=${method}`, image, zero, method, ones);
    for (const matrix of [image, template, zero, ones]) api.dispose(matrix);
  }

  for (const target of ["roi", "wrong type", "image", "template"]) {
    const parent = api.mat(3, 5, 1, 5, [99, 1, 2, 3, 99, 99, 4, 5, 6, 99, 99, 7, 8, 9, 99]);
    const image = api.roi(parent, 0, 1, 3, 3);
    const template = api.roi(parent, 1, 2, 2, 2);
    const outputParent = api.mat(3, 4, 1, 5, Array(12).fill(-1));
    const output =
      target === "image"
        ? image
        : target === "template"
          ? template
          : target === "roi"
            ? api.roi(outputParent, 1, 1, 2, 2)
            : api.mat(1, 1, 3, 0, [7, 8, 9]);
    match(`destination ${target}`, image, template, 0, undefined, output);
    results.push({ label: `destination ${target} parent`, ...api.summary(outputParent) });
    for (const matrix of [parent, image, template, outputParent]) api.dispose(matrix);
  }

  const errors = [];
  function rejects(label, callback) {
    try {
      callback();
      errors.push({ label, threw: false });
    } catch {
      errors.push({ label, threw: true });
    }
  }
  const image = api.mat(2, 3, 1, 0, [1, 2, 3, 4, 5, 6]);
  const template = api.mat(1, 2, 1, 0, [1, 2]);
  const output = api.mat(1, 1, 1, 0, [42]);
  const empty = api.empty();
  const signed = api.mat(1, 2, 1, 3, [1, 2]);
  const tall = api.mat(3, 1, 1, 0, [1, 2, 3]);
  rejects("arity zero", () => api.match());
  rejects("arity three", () => api.match(image, template, output));
  rejects("arity six", () => api.match(image, template, output, 0, empty, empty));
  rejects("undefined mask", () => api.match(image, template, output, 0, undefined));
  rejects("null mask", () => api.match(image, template, output, 0, null));
  rejects("string method", () => api.match(image, template, output, "2"));
  rejects("invalid method", () => api.match(image, template, output, 6));
  rejects("empty image", () => api.match(empty, template, output, 0));
  rejects("empty template", () => api.match(image, empty, output, 0));
  rejects("mismatched depth", () => api.match(image, signed, output, 0));
  rejects("unsupported depth", () => api.match(signed, signed, output, 0));
  rejects("crossed dimensions", () => api.match(image, tall, output, 0));
  rejects("wrong mask size", () => api.match(image, template, output, 0, image));
  rejects("wrong mask depth", () => api.match(image, template, output, 0, signed));
  rejects("masked swapped sizes", () => api.match(template, image, output, 0, image));
  api.dispose(template);
  rejects("deleted template", () => api.match(image, template, output, 0));
  for (const matrix of [image, output, empty, signed, tall]) api.dispose(matrix);
  return { results, errors, constants: api.constants, functionLength: api.functionLength };
}

export function compareTemplateMatching(actual, expected) {
  if (JSON.stringify(actual.constants) !== JSON.stringify(expected.constants))
    throw new Error("Template constants differ");
  if (actual.functionLength !== expected.functionLength) throw new Error("Function length differs");
  if (
    JSON.stringify(actual.errors) !== JSON.stringify(expected.errors) ||
    actual.errors.some((entry) => !entry.threw)
  )
    throw new Error("Template error cases differ");
  if (actual.results.length !== expected.results.length)
    throw new Error("Template case count differs");
  let maxAbsoluteError = 0;
  const knownDifferences = [];
  for (let caseIndex = 0; caseIndex < actual.results.length; caseIndex += 1) {
    const a = actual.results[caseIndex];
    const e = expected.results[caseIndex];
    for (const key of ["label", "rows", "columns", "channels", "depth"]) {
      if (a[key] !== e[key]) throw new Error(`Template ${a.label}: ${key} differs`);
    }
    if (a.values.length !== e.values.length) throw new Error(`Template ${a.label}: length differs`);
    if (a.label === "destination template") {
      if (JSON.stringify(a.values) !== JSON.stringify([64, 36, 4, 0]))
        throw new Error("Template alias snapshot contract changed");
      knownDifferences.push({ label: a.label, packageValues: a.values, referenceValues: e.values });
      continue;
    }
    for (let i = 0; i < a.values.length; i += 1) {
      const x = a.values[i];
      const y = e.values[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        if (!(Object.is(x, y) || (Number.isNaN(x) && Number.isNaN(y))))
          throw new Error(`Template ${a.label}[${i}]: ${x} != ${y}`);
        continue;
      }
      const difference = Math.abs(x - y);
      maxAbsoluteError = Math.max(maxAbsoluteError, difference);
      // Scalar F64 accumulation and the reference's F32 correlation have different rounding.
      const tolerance = 0.0001 + 0.00002 * Math.abs(y);
      if (difference > tolerance)
        throw new Error(`Template ${a.label}[${i}]: ${x} != ${y}, tolerance ${tolerance}`);
    }
  }
  return {
    cases: actual.results.length - knownDifferences.length,
    errorCases: actual.errors.length,
    maxAbsoluteError,
    knownDifferences,
  };
}
