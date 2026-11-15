/* global importScripts, cv */
/* oxlint-disable unicorn/require-post-message-target-origin, anti-slop/no-runtime-typeof */

function copyBytes(matrix) {
  return new Uint8Array(matrix.data);
}

function copyF64(matrix) {
  return Array.from(matrix.data64F);
}

function summarizeMat(matrix) {
  const elementSize = matrix.elemSize();
  const rowByteLength = matrix.cols * elementSize;
  return {
    rows: matrix.rows,
    cols: matrix.cols,
    type: matrix.type(),
    depth: matrix.depth(),
    channels: matrix.channels(),
    elementSize,
    empty: matrix.empty(),
    continuous: matrix.isContinuous(),
    bytes: Array.from({ length: matrix.rows }, (_, row) =>
      Array.from(matrix.ucharPtr(row).subarray(0, rowByteLength)),
    ).flat(),
  };
}

function summarizeTypedMat(matrix) {
  const summary = summarizeMat(matrix);
  const pointerMethod = [
    "ucharPtr",
    "charPtr",
    "ushortPtr",
    "shortPtr",
    "intPtr",
    "floatPtr",
    "doublePtr",
  ][summary.depth];
  const rowScalarLength = summary.cols * summary.channels;
  const values = Array.from({ length: summary.rows }, (_, row) =>
    Array.from(matrix[pointerMethod](row).subarray(0, rowScalarLength)),
  ).flat();
  return { ...summary, values };
}

function makeSeedMat(reference, rows, cols, type, values) {
  return reference.matFromArray(rows, cols, type, values);
}

function auditTransposeCall(callback, matrices) {
  const before = Object.fromEntries(
    matrices.map(([name, matrix]) => [name, capturePrimitive(() => summarizeMat(matrix))]),
  );
  const call = captureCall(callback);
  const after = Object.fromEntries(
    matrices.map(([name, matrix]) => [name, capturePrimitive(() => summarizeMat(matrix))]),
  );
  return { before, call, after };
}

function auditTypedMatCall(callback, matrices) {
  const summarize = ([name, matrix]) => [name, capturePrimitive(() => summarizeTypedMat(matrix))];
  const before = Object.fromEntries(matrices.map(summarize));
  const call = captureCall(callback);
  const after = Object.fromEntries(matrices.map(summarize));
  return { before, call, after };
}

function auditTransposeDestination(reference, name, createDestination) {
  const source = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const destination = createDestination();
  const result = {
    name,
    audit: auditTransposeCall(
      () => reference.transpose(source, destination),
      [
        ["source", source],
        ["destination", destination],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  return result;
}

function auditTransposeInPlace(reference, rows, cols, values) {
  const matrix = makeSeedMat(reference, rows, cols, reference.CV_8UC1, values);
  const result = auditTransposeCall(
    () => reference.transpose(matrix, matrix),
    [["matrix", matrix]],
  );
  safeDelete(matrix);
  return result;
}

function auditTransposeRoi(reference, name, sourceRect, destinationRect, sameParent = false) {
  const sourceParent = makeSeedMat(
    reference,
    4,
    5,
    reference.CV_8UC1,
    Array.from({ length: 20 }, (_, index) => index + 1),
  );
  const destinationParent = sameParent
    ? sourceParent
    : makeSeedMat(reference, 5, 5, reference.CV_8UC1, new Uint8Array(25).fill(99));
  const source = sourceParent.roi(new reference.Rect(...sourceRect));
  const destination = destinationParent.roi(new reference.Rect(...destinationRect));
  const result = {
    name,
    sameParent,
    audit: auditTransposeCall(
      () => reference.transpose(source, destination),
      [
        ["source", source],
        ["destination", destination],
        ["sourceParent", sourceParent],
        ["destinationParent", destinationParent],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  if (!sameParent) safeDelete(destinationParent);
  safeDelete(sourceParent);
  return result;
}

function auditTransposeType(reference, name, type, values) {
  if (typeof type !== "number") return { name, available: false };
  let source;
  let destination;
  try {
    source = makeSeedMat(reference, 2, 3, type, values);
    destination = new reference.Mat();
    return {
      name,
      available: true,
      audit: auditTransposeCall(
        () => reference.transpose(source, destination),
        [
          ["source", source],
          ["destination", destination],
        ],
      ),
    };
  } catch (error) {
    return {
      name,
      available: true,
      setup: captureCall(() => {
        throw error;
      }),
    };
  } finally {
    safeDelete(destination);
    safeDelete(source);
  }
}

function auditTranspose(reference) {
  const aritySource = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const arityDestination = new reference.Mat();
  const arity = {
    functionLength: capturePrimitive(() => reference.transpose.length),
    zero: captureCall(() => reference.transpose()),
    one: captureCall(() => reference.transpose(aritySource)),
    two: captureCall(() => reference.transpose(aritySource, arityDestination)),
    three: captureCall(() => reference.transpose(aritySource, arityDestination, 1)),
    destinationAfter: capturePrimitive(() => summarizeMat(arityDestination)),
  };
  safeDelete(arityDestination);
  safeDelete(aritySource);

  const emptySource = new reference.Mat();
  const emptyDestination = new reference.Mat();
  const empty = auditTransposeCall(
    () => reference.transpose(emptySource, emptyDestination),
    [
      ["source", emptySource],
      ["destination", emptyDestination],
    ],
  );
  safeDelete(emptyDestination);
  safeDelete(emptySource);

  const deletedSource = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [7]);
  const liveDestination = new reference.Mat();
  deletedSource.delete();
  const deletedSourceAudit = auditTransposeCall(
    () => reference.transpose(deletedSource, liveDestination),
    [
      ["source", deletedSource],
      ["destination", liveDestination],
    ],
  );
  safeDelete(liveDestination);

  const liveSource = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [8]);
  const deletedDestination = new reference.Mat();
  deletedDestination.delete();
  const deletedDestinationAudit = auditTransposeCall(
    () => reference.transpose(liveSource, deletedDestination),
    [
      ["source", liveSource],
      ["destination", deletedDestination],
    ],
  );
  safeDelete(liveSource);

  const types = [
    ["CV_8UC1", reference.CV_8UC1, [1, 2, 3, 4, 5, 6]],
    ["CV_8UC3", reference.CV_8UC3, Array.from({ length: 18 }, (_, index) => index + 1)],
    ["CV_8SC2", reference.CV_8SC2, [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12]],
    ["CV_16UC1", reference.CV_16UC1, [1, 256, 513, 1024, 4096, 65_535]],
    [
      "CV_16SC2",
      reference.CV_16SC2,
      [-1, 2, -300, 400, -500, 600, -700, 800, -900, 1000, -1100, 1200],
    ],
    ["CV_32SC1", reference.CV_32SC1, [-1, 2, -300_000, 400_000, -500_000, 600_000]],
    [
      "CV_32FC2",
      reference.CV_32FC2,
      [-1.5, 2.25, -3.5, 4.75, -5.5, 6.25, -7.5, 8.75, -9.5, 10.25, -11.5, 12.75],
    ],
    ["CV_64FC1", reference.CV_64FC1, [-1.5, 2.25, -3.5, 4.75, -5.5, 6.25]],
    ["CV_16FC1", reference.CV_16FC1, [1, 2, 3, 4, 5, 6]],
  ].map(([name, type, values]) => auditTransposeType(reference, name, type, values));

  return {
    arity,
    destinationReplacement: [
      auditTransposeDestination(reference, "empty", () => new reference.Mat()),
      auditTransposeDestination(reference, "correct-metadata", () =>
        makeSeedMat(reference, 3, 2, reference.CV_8UC1, new Uint8Array(6).fill(99)),
      ),
      auditTransposeDestination(reference, "wrong-shape", () =>
        makeSeedMat(reference, 1, 5, reference.CV_8UC1, new Uint8Array(5).fill(99)),
      ),
      auditTransposeDestination(reference, "wrong-type-and-channels", () =>
        makeSeedMat(reference, 3, 2, reference.CV_32FC2, new Float32Array(12).fill(99)),
      ),
    ],
    inPlace: {
      square: auditTransposeInPlace(reference, 3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
      rectangular: auditTransposeInPlace(reference, 2, 3, [1, 2, 3, 4, 5, 6]),
    },
    roi: [
      auditTransposeRoi(reference, "compatible-separate", [1, 1, 3, 2], [1, 1, 2, 3]),
      auditTransposeRoi(reference, "incompatible-shape-separate", [1, 1, 3, 2], [0, 0, 3, 2]),
      auditTransposeRoi(
        reference,
        "compatible-same-parent-non-overlap",
        [0, 0, 2, 2],
        [3, 2, 2, 2],
        true,
      ),
      auditTransposeRoi(
        reference,
        "compatible-same-parent-overlap",
        [0, 0, 3, 2],
        [1, 0, 2, 3],
        true,
      ),
    ],
    empty,
    deleted: { source: deletedSourceAudit, destination: deletedDestinationAudit },
    types,
    halfFloatConstants: {
      CV_16F: encodeValue(reference.CV_16F),
      CV_16FC1: encodeValue(reference.CV_16FC1),
    },
  };
}

function auditFlipCode(reference, label, value) {
  const source = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const destination = new reference.Mat();
  const result = {
    label,
    input: encodeValue(value),
    audit: auditTransposeCall(
      () => reference.flip(source, destination, value),
      [
        ["source", source],
        ["destination", destination],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  return result;
}

function auditFlipDestination(reference, name, createDestination) {
  const source = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const destination = createDestination();
  const result = {
    name,
    audit: auditTransposeCall(
      () => reference.flip(source, destination, 1),
      [
        ["source", source],
        ["destination", destination],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  return result;
}

function auditFlipInPlace(reference, code) {
  const matrix = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const result = {
    code,
    audit: auditTransposeCall(() => reference.flip(matrix, matrix, code), [["matrix", matrix]]),
  };
  safeDelete(matrix);
  return result;
}

function auditFlipRoi(reference, name, sourceRect, destinationRect, sameParent, code) {
  const sourceParent = makeSeedMat(
    reference,
    5,
    6,
    reference.CV_8UC1,
    Array.from({ length: 30 }, (_, index) => index + 1),
  );
  const destinationParent = sameParent
    ? sourceParent
    : makeSeedMat(reference, 5, 6, reference.CV_8UC1, new Uint8Array(30).fill(99));
  const source = sourceParent.roi(new reference.Rect(...sourceRect));
  const destination = destinationParent.roi(new reference.Rect(...destinationRect));
  const result = {
    name,
    sameParent,
    code,
    audit: auditTransposeCall(
      () => reference.flip(source, destination, code),
      [
        ["source", source],
        ["destination", destination],
        ["sourceParent", sourceParent],
        ["destinationParent", destinationParent],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  if (!sameParent) safeDelete(destinationParent);
  safeDelete(sourceParent);
  return result;
}

function auditFlipTypeRoi(reference) {
  const sourceParent = makeSeedMat(
    reference,
    5,
    6,
    reference.CV_8UC1,
    Array.from({ length: 30 }, (_, index) => index + 1),
  );
  const destinationParent = makeSeedMat(
    reference,
    5,
    6,
    reference.CV_32FC2,
    new Float32Array(60).fill(99),
  );
  const source = sourceParent.roi(new reference.Rect(1, 1, 3, 2));
  const destination = destinationParent.roi(new reference.Rect(1, 1, 3, 2));
  const result = {
    name: "incompatible-type-separate",
    sameParent: false,
    code: 1,
    audit: auditTransposeCall(
      () => reference.flip(source, destination, 1),
      [
        ["source", source],
        ["destination", destination],
        ["sourceParent", sourceParent],
        ["destinationParent", destinationParent],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  safeDelete(destinationParent);
  safeDelete(sourceParent);
  return result;
}

function auditFlipType(reference, name, type, values) {
  if (typeof type !== "number") return { name, available: false };
  let source;
  let destination;
  try {
    source = makeSeedMat(reference, 2, 3, type, values);
    destination = new reference.Mat();
    return {
      name,
      available: true,
      audit: auditTypedMatCall(
        () => reference.flip(source, destination, -1),
        [
          ["source", source],
          ["destination", destination],
        ],
      ),
    };
  } catch (error) {
    return {
      name,
      available: true,
      setup: captureCall(() => {
        throw error;
      }),
    };
  } finally {
    safeDelete(destination);
    safeDelete(source);
  }
}

function auditFlip(reference) {
  const source = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const destination = new reference.Mat();
  const arity = {
    functionLength: capturePrimitive(() => reference.flip.length),
    zero: captureCall(() => reference.flip()),
    one: captureCall(() => reference.flip(source)),
    two: captureCall(() => reference.flip(source, destination)),
    three: captureCall(() => reference.flip(source, destination, 1)),
    four: captureCall(() => reference.flip(source, destination, 1, 2)),
    destinationAfter: capturePrimitive(() => summarizeMat(destination)),
  };
  safeDelete(destination);
  safeDelete(source);

  const codes = [
    ["negative two", -2],
    ["negative one", -1],
    ["negative fraction", -1.9],
    ["negative subunit fraction", -0.9],
    ["negative zero", -0],
    ["zero", 0],
    ["positive fraction", 1.9],
    ["positive subunit fraction", 0.9],
    ["one", 1],
    ["two", 2],
    ["i32 maximum", 2_147_483_647],
    ["i32 maximum plus one", 2_147_483_648],
    ["i32 minimum", -2_147_483_648],
    ["i32 minimum minus one", -2_147_483_649],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["true", true],
    ["false", false],
    ["null", null],
    ["numeric string", "1"],
    ["explicit undefined", undefined],
  ].map(([label, value]) => auditFlipCode(reference, label, value));

  const argumentSource = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [1]);
  const argumentDestination = new reference.Mat();
  const argumentTypes = {
    nullSource: captureCall(() => reference.flip(null, argumentDestination, 1)),
    undefinedSource: captureCall(() => reference.flip(undefined, argumentDestination, 1)),
    objectSource: captureCall(() => reference.flip({}, argumentDestination, 1)),
    nullDestination: captureCall(() => reference.flip(argumentSource, null, 1)),
    undefinedDestination: captureCall(() => reference.flip(argumentSource, undefined, 1)),
    objectDestination: captureCall(() => reference.flip(argumentSource, {}, 1)),
  };
  safeDelete(argumentDestination);
  safeDelete(argumentSource);

  const emptySource = new reference.Mat();
  const emptyDestination = new reference.Mat();
  const empty = auditTransposeCall(
    () => reference.flip(emptySource, emptyDestination, 1),
    [
      ["source", emptySource],
      ["destination", emptyDestination],
    ],
  );
  safeDelete(emptyDestination);
  safeDelete(emptySource);

  const secondEmptySource = new reference.Mat();
  const fullEmptyDestination = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const emptyIntoFull = auditTransposeCall(
    () => reference.flip(secondEmptySource, fullEmptyDestination, -1),
    [
      ["source", secondEmptySource],
      ["destination", fullEmptyDestination],
    ],
  );
  safeDelete(fullEmptyDestination);
  safeDelete(secondEmptySource);

  const deletedSource = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [7]);
  const liveDestination = new reference.Mat();
  deletedSource.delete();
  const deletedSourceAudit = auditTransposeCall(
    () => reference.flip(deletedSource, liveDestination, 1),
    [
      ["source", deletedSource],
      ["destination", liveDestination],
    ],
  );
  safeDelete(liveDestination);

  const liveSource = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [8]);
  const deletedDestination = new reference.Mat();
  deletedDestination.delete();
  const deletedDestinationAudit = auditTransposeCall(
    () => reference.flip(liveSource, deletedDestination, 1),
    [
      ["source", liveSource],
      ["destination", deletedDestination],
    ],
  );
  safeDelete(liveSource);

  const types = [
    ["CV_8UC1", reference.CV_8UC1, [1, 2, 3, 4, 5, 6]],
    ["CV_8UC3", reference.CV_8UC3, Array.from({ length: 18 }, (_, index) => index + 1)],
    ["CV_8SC2", reference.CV_8SC2, [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12]],
    ["CV_16UC1", reference.CV_16UC1, [1, 256, 513, 1024, 4096, 65_535]],
    [
      "CV_16SC2",
      reference.CV_16SC2,
      [-1, 2, -300, 400, -500, 600, -700, 800, -900, 1000, -1100, 1200],
    ],
    ["CV_32SC1", reference.CV_32SC1, [-1, 2, -300_000, 400_000, -500_000, 600_000]],
    [
      "CV_32FC2",
      reference.CV_32FC2,
      [-1.5, 2.25, -3.5, 4.75, -5.5, 6.25, -7.5, 8.75, -9.5, 10.25, -11.5, 12.75],
    ],
    ["CV_64FC1", reference.CV_64FC1, [-1.5, 2.25, -3.5, 4.75, -5.5, 6.25]],
    ["CV_16FC1", reference.CV_16FC1, [1, 2, 3, 4, 5, 6]],
  ].map(([name, type, values]) => auditFlipType(reference, name, type, values));

  return {
    arity,
    argumentTypes,
    codes,
    destinationReplacement: [
      auditFlipDestination(reference, "empty", () => new reference.Mat()),
      auditFlipDestination(reference, "correct-metadata", () =>
        makeSeedMat(reference, 2, 3, reference.CV_8UC1, new Uint8Array(6).fill(99)),
      ),
      auditFlipDestination(reference, "wrong-shape", () =>
        makeSeedMat(reference, 3, 2, reference.CV_8UC1, new Uint8Array(6).fill(99)),
      ),
      auditFlipDestination(reference, "wrong-type-and-channels", () =>
        makeSeedMat(reference, 2, 3, reference.CV_32FC2, new Float32Array(12).fill(99)),
      ),
    ],
    inPlace: [-1, 0, 1].map((code) => auditFlipInPlace(reference, code)),
    roi: [
      auditFlipRoi(reference, "compatible-separate", [1, 1, 3, 2], [1, 1, 3, 2], false, 1),
      auditFlipRoi(reference, "incompatible-shape-separate", [1, 1, 3, 2], [0, 0, 2, 3], false, 1),
      auditFlipTypeRoi(reference),
      auditFlipRoi(
        reference,
        "compatible-same-parent-non-overlap",
        [0, 0, 2, 2],
        [3, 2, 2, 2],
        true,
        -1,
      ),
      auditFlipRoi(
        reference,
        "compatible-same-parent-exact-view",
        [0, 0, 3, 2],
        [0, 0, 3, 2],
        true,
        1,
      ),
      auditFlipRoi(
        reference,
        "compatible-same-parent-overlap-horizontal",
        [0, 0, 3, 2],
        [1, 0, 3, 2],
        true,
        1,
      ),
      auditFlipRoi(
        reference,
        "compatible-same-parent-overlap-vertical",
        [0, 0, 3, 3],
        [0, 1, 3, 3],
        true,
        0,
      ),
      auditFlipRoi(
        reference,
        "compatible-same-parent-overlap-both-horizontal-shift",
        [0, 0, 3, 3],
        [1, 0, 3, 3],
        true,
        -1,
      ),
      auditFlipRoi(
        reference,
        "compatible-same-parent-overlap-both-vertical-shift",
        [0, 0, 3, 3],
        [0, 1, 3, 3],
        true,
        -1,
      ),
      auditFlipRoi(
        reference,
        "compatible-same-parent-overlap-both",
        [0, 0, 3, 3],
        [1, 1, 3, 3],
        true,
        -1,
      ),
      auditFlipRoi(
        reference,
        "compatible-same-parent-overlap-both-small-diagonal",
        [0, 0, 2, 2],
        [1, 1, 2, 2],
        true,
        -1,
      ),
    ],
    empty,
    emptyIntoFull,
    deleted: { source: deletedSourceAudit, destination: deletedDestinationAudit },
    types,
    halfFloatConstants: {
      CV_16F: encodeValue(reference.CV_16F),
      CV_16FC1: encodeValue(reference.CV_16FC1),
    },
  };
}

function auditRepeatCounts(reference, axis, label, value, avoidLargeOutput = false) {
  const source = makeSeedMat(reference, 1, 2, reference.CV_8UC1, [1, 2]);
  const destination = makeSeedMat(reference, 2, 2, reference.CV_8UC1, [7, 8, 9, 10]);
  const rowRepeats = axis === "rows" ? value : avoidLargeOutput ? -1 : 1;
  const columnRepeats = axis === "columns" ? value : avoidLargeOutput ? -1 : 1;
  const result = {
    axis,
    label,
    input: encodeValue(value),
    audit: auditTransposeCall(
      () => reference.repeat(source, rowRepeats, columnRepeats, destination),
      [
        ["source", source],
        ["destination", destination],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  return result;
}

function auditRepeatDestination(reference, name, createDestination) {
  const source = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const destination = createDestination();
  const result = {
    name,
    audit: auditTransposeCall(
      () => reference.repeat(source, 2, 2, destination),
      [
        ["source", source],
        ["destination", destination],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  return result;
}

function auditRepeatRoi(
  reference,
  name,
  sourceRect,
  destinationRect,
  rowRepeats,
  columnRepeats,
  sameParent,
) {
  const sourceParent = makeSeedMat(
    reference,
    6,
    7,
    reference.CV_8UC1,
    Array.from({ length: 42 }, (_, index) => index + 1),
  );
  const destinationParent = sameParent
    ? sourceParent
    : makeSeedMat(reference, 6, 7, reference.CV_8UC1, new Uint8Array(42).fill(99));
  const source = sourceParent.roi(new reference.Rect(...sourceRect));
  const destination = destinationParent.roi(new reference.Rect(...destinationRect));
  const result = {
    name,
    sameParent,
    rowRepeats,
    columnRepeats,
    audit: auditTransposeCall(
      () => reference.repeat(source, rowRepeats, columnRepeats, destination),
      [
        ["source", source],
        ["destination", destination],
        ["sourceParent", sourceParent],
        ["destinationParent", destinationParent],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  if (!sameParent) safeDelete(destinationParent);
  safeDelete(sourceParent);
  return result;
}

function auditRepeatTypeRoi(reference) {
  const sourceParent = makeSeedMat(
    reference,
    4,
    5,
    reference.CV_8UC1,
    Array.from({ length: 20 }, (_, index) => index + 1),
  );
  const destinationParent = makeSeedMat(
    reference,
    5,
    5,
    reference.CV_32FC2,
    new Float32Array(50).fill(99),
  );
  const source = sourceParent.roi(new reference.Rect(1, 1, 2, 2));
  const destination = destinationParent.roi(new reference.Rect(1, 1, 2, 4));
  const result = {
    name: "incompatible-type-separate",
    sameParent: false,
    rowRepeats: 2,
    columnRepeats: 1,
    audit: auditTransposeCall(
      () => reference.repeat(source, 2, 1, destination),
      [
        ["source", source],
        ["destination", destination],
        ["sourceParent", sourceParent],
        ["destinationParent", destinationParent],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  safeDelete(destinationParent);
  safeDelete(sourceParent);
  return result;
}

function auditRepeatType(reference, name, type, values) {
  if (typeof type !== "number") return { name, available: false };
  let source;
  let destination;
  try {
    source = makeSeedMat(reference, 2, 3, type, values);
    destination = new reference.Mat();
    return {
      name,
      available: true,
      audit: auditTypedMatCall(
        () => reference.repeat(source, 2, 2, destination),
        [
          ["source", source],
          ["destination", destination],
        ],
      ),
    };
  } catch (error) {
    return {
      name,
      available: true,
      setup: captureCall(() => {
        throw error;
      }),
    };
  } finally {
    safeDelete(destination);
    safeDelete(source);
  }
}

function auditRepeat(reference) {
  const aritySource = makeSeedMat(reference, 1, 2, reference.CV_8UC1, [1, 2]);
  const arityDestination = new reference.Mat();
  const arity = {
    functionLength: capturePrimitive(() => reference.repeat.length),
    zero: captureCall(() => reference.repeat()),
    one: captureCall(() => reference.repeat(aritySource)),
    two: captureCall(() => reference.repeat(aritySource, 1)),
    three: captureCall(() => reference.repeat(aritySource, 1, 2)),
    four: captureCall(() => reference.repeat(aritySource, 1, 2, arityDestination)),
    five: captureCall(() => reference.repeat(aritySource, 1, 2, arityDestination, 5)),
    destinationAfter: capturePrimitive(() => summarizeMat(arityDestination)),
  };
  safeDelete(arityDestination);
  safeDelete(aritySource);

  const countInputs = [
    ["negative two", -2],
    ["negative fraction", -1.9],
    ["negative subunit fraction", -0.9],
    ["negative zero", -0],
    ["zero", 0],
    ["positive subunit fraction", 0.9],
    ["positive fraction", 1.9],
    ["one", 1],
    ["two", 2],
    ["i32 maximum", 2_147_483_647, true],
    ["i32 maximum plus one", 2_147_483_648],
    ["i32 minimum", -2_147_483_648],
    ["i32 minimum minus one", -2_147_483_649],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["true", true],
    ["false", false],
    ["null", null],
    ["numeric string", "1"],
    ["explicit undefined", undefined],
  ];
  const counts = ["rows", "columns"].flatMap((axis) =>
    countInputs.map(([label, value, avoidLargeOutput = false]) =>
      auditRepeatCounts(reference, axis, label, value, avoidLargeOutput),
    ),
  );

  const argumentSource = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [1]);
  const argumentDestination = new reference.Mat();
  const argumentTypes = {
    nullSource: captureCall(() => reference.repeat(null, 1, 1, argumentDestination)),
    undefinedSource: captureCall(() => reference.repeat(undefined, 1, 1, argumentDestination)),
    objectSource: captureCall(() => reference.repeat({}, 1, 1, argumentDestination)),
    nullDestination: captureCall(() => reference.repeat(argumentSource, 1, 1, null)),
    undefinedDestination: captureCall(() => reference.repeat(argumentSource, 1, 1, undefined)),
    objectDestination: captureCall(() => reference.repeat(argumentSource, 1, 1, {})),
  };
  safeDelete(argumentDestination);
  safeDelete(argumentSource);

  const emptySource = new reference.Mat();
  const emptyDestination = new reference.Mat();
  const empty = auditTransposeCall(
    () => reference.repeat(emptySource, 2, 3, emptyDestination),
    [
      ["source", emptySource],
      ["destination", emptyDestination],
    ],
  );
  safeDelete(emptyDestination);
  safeDelete(emptySource);

  const secondEmptySource = new reference.Mat();
  const fullEmptyDestination = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const emptyIntoFull = auditTransposeCall(
    () => reference.repeat(secondEmptySource, 2, 3, fullEmptyDestination),
    [
      ["source", secondEmptySource],
      ["destination", fullEmptyDestination],
    ],
  );
  safeDelete(fullEmptyDestination);
  safeDelete(secondEmptySource);

  const deletedSource = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [7]);
  const liveDestination = new reference.Mat();
  deletedSource.delete();
  const deletedSourceAudit = auditTransposeCall(
    () => reference.repeat(deletedSource, 1, 1, liveDestination),
    [
      ["source", deletedSource],
      ["destination", liveDestination],
    ],
  );
  safeDelete(liveDestination);

  const liveSource = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [8]);
  const deletedDestination = new reference.Mat();
  deletedDestination.delete();
  const deletedDestinationAudit = auditTransposeCall(
    () => reference.repeat(liveSource, 1, 1, deletedDestination),
    [
      ["source", liveSource],
      ["destination", deletedDestination],
    ],
  );
  safeDelete(liveSource);

  const sameHandle = makeSeedMat(reference, 2, 2, reference.CV_8UC1, [1, 2, 3, 4]);
  const inPlace = auditTransposeCall(
    () => reference.repeat(sameHandle, 1, 1, sameHandle),
    [["matrix", sameHandle]],
  );
  safeDelete(sameHandle);

  const stridedParent = makeSeedMat(
    reference,
    4,
    5,
    reference.CV_8UC1,
    Array.from({ length: 20 }, (_, index) => index + 1),
  );
  const stridedSource = stridedParent.roi(new reference.Rect(1, 1, 3, 2));
  const stridedDestination = new reference.Mat();
  const strided = auditTransposeCall(
    () => reference.repeat(stridedSource, 2, 2, stridedDestination),
    [
      ["source", stridedSource],
      ["destination", stridedDestination],
      ["sourceParent", stridedParent],
    ],
  );
  safeDelete(stridedDestination);
  safeDelete(stridedSource);
  safeDelete(stridedParent);

  const types = [
    ["CV_8UC1", reference.CV_8UC1, [1, 2, 3, 4, 5, 6]],
    ["CV_8UC3", reference.CV_8UC3, Array.from({ length: 18 }, (_, index) => index + 1)],
    ["CV_8SC2", reference.CV_8SC2, [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12]],
    ["CV_16UC1", reference.CV_16UC1, [1, 256, 513, 1024, 4096, 65_535]],
    [
      "CV_16SC2",
      reference.CV_16SC2,
      [-1, 2, -300, 400, -500, 600, -700, 800, -900, 1000, -1100, 1200],
    ],
    ["CV_32SC1", reference.CV_32SC1, [-1, 2, -300_000, 400_000, -500_000, 600_000]],
    [
      "CV_32FC2",
      reference.CV_32FC2,
      [-1.5, 2.25, -3.5, 4.75, -5.5, 6.25, -7.5, 8.75, -9.5, 10.25, -11.5, 12.75],
    ],
    ["CV_64FC1", reference.CV_64FC1, [-1.5, 2.25, -3.5, 4.75, -5.5, 6.25]],
    ["CV_16FC1", reference.CV_16FC1, [1, 2, 3, 4, 5, 6]],
  ].map(([name, type, values]) => auditRepeatType(reference, name, type, values));

  return {
    arity,
    argumentTypes,
    counts,
    destinationReplacement: [
      auditRepeatDestination(reference, "empty", () => new reference.Mat()),
      auditRepeatDestination(reference, "correct-metadata", () =>
        makeSeedMat(reference, 4, 6, reference.CV_8UC1, new Uint8Array(24).fill(99)),
      ),
      auditRepeatDestination(reference, "wrong-shape", () =>
        makeSeedMat(reference, 2, 6, reference.CV_8UC1, new Uint8Array(12).fill(99)),
      ),
      auditRepeatDestination(reference, "wrong-type-and-channels", () =>
        makeSeedMat(reference, 4, 6, reference.CV_32FC2, new Float32Array(48).fill(99)),
      ),
    ],
    inPlace,
    roi: [
      auditRepeatRoi(reference, "compatible-separate", [1, 1, 2, 2], [1, 1, 4, 2], 1, 2, false),
      auditRepeatRoi(
        reference,
        "incompatible-shape-separate",
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        1,
        2,
        false,
      ),
      auditRepeatTypeRoi(reference),
      auditRepeatRoi(reference, "same-parent-non-overlap", [0, 0, 2, 2], [3, 3, 4, 2], 1, 2, true),
      auditRepeatRoi(
        reference,
        "same-parent-overlap-horizontal",
        [0, 0, 2, 2],
        [1, 0, 4, 2],
        1,
        2,
        true,
      ),
      auditRepeatRoi(
        reference,
        "same-parent-overlap-vertical",
        [0, 0, 2, 2],
        [0, 1, 2, 4],
        2,
        1,
        true,
      ),
      auditRepeatRoi(
        reference,
        "same-parent-overlap-combined",
        [0, 0, 2, 2],
        [1, 1, 4, 4],
        2,
        2,
        true,
      ),
    ],
    empty,
    emptyIntoFull,
    deleted: { source: deletedSourceAudit, destination: deletedDestinationAudit },
    strided,
    types,
    halfFloatConstants: {
      CV_16F: encodeValue(reference.CV_16F),
      CV_16FC1: encodeValue(reference.CV_16FC1),
    },
  };
}

function auditRotateCode(reference, label, value, nativeInvalid = false) {
  const source = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const destination = makeSeedMat(reference, 3, 2, reference.CV_8UC1, [7, 8, 9, 10, 11, 12]);
  const matrices = [
    ["source", source],
    ["destination", destination],
  ];
  const before = Object.fromEntries(
    matrices.map(([name, matrix]) => [name, capturePrimitive(() => summarizeMat(matrix))]),
  );
  const call = captureCall(() => reference.rotate(source, destination, value));
  const result = {
    label,
    input: encodeValue(value),
    nativeInvalid,
    audit: nativeInvalid
      ? { before, call }
      : {
          before,
          call,
          after: Object.fromEntries(
            matrices.map(([name, matrix]) => [name, capturePrimitive(() => summarizeMat(matrix))]),
          ),
        },
  };
  safeDelete(destination);
  safeDelete(source);
  return result;
}

function auditRotateDestination(reference, name, code, createDestination) {
  const source = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const destination = createDestination();
  const result = {
    name,
    code,
    audit: auditTransposeCall(
      () => reference.rotate(source, destination, code),
      [
        ["source", source],
        ["destination", destination],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  return result;
}

function auditRotateInPlace(reference, code) {
  const matrix = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const result = {
    code,
    audit: auditTransposeCall(() => reference.rotate(matrix, matrix, code), [["matrix", matrix]]),
  };
  safeDelete(matrix);
  return result;
}

function auditRotateRoi(reference, name, sourceRect, destinationRect, sameParent, code) {
  const sourceParent = makeSeedMat(
    reference,
    6,
    7,
    reference.CV_8UC1,
    Array.from({ length: 42 }, (_, index) => index + 1),
  );
  const destinationParent = sameParent
    ? sourceParent
    : makeSeedMat(reference, 6, 7, reference.CV_8UC1, new Uint8Array(42).fill(99));
  const source = sourceParent.roi(new reference.Rect(...sourceRect));
  const destination = destinationParent.roi(new reference.Rect(...destinationRect));
  const result = {
    name,
    sameParent,
    code,
    audit: auditTransposeCall(
      () => reference.rotate(source, destination, code),
      [
        ["source", source],
        ["destination", destination],
        ["sourceParent", sourceParent],
        ["destinationParent", destinationParent],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  if (!sameParent) safeDelete(destinationParent);
  safeDelete(sourceParent);
  return result;
}

function auditRotateTypeRoi(reference) {
  const sourceParent = makeSeedMat(
    reference,
    4,
    5,
    reference.CV_8UC1,
    Array.from({ length: 20 }, (_, index) => index + 1),
  );
  const destinationParent = makeSeedMat(
    reference,
    5,
    5,
    reference.CV_32FC2,
    new Float32Array(50).fill(99),
  );
  const source = sourceParent.roi(new reference.Rect(1, 1, 3, 2));
  const destination = destinationParent.roi(new reference.Rect(1, 1, 2, 3));
  const result = {
    name: "incompatible-type-separate",
    sameParent: false,
    code: 0,
    audit: auditTransposeCall(
      () => reference.rotate(source, destination, 0),
      [
        ["source", source],
        ["destination", destination],
        ["sourceParent", sourceParent],
        ["destinationParent", destinationParent],
      ],
    ),
  };
  safeDelete(destination);
  safeDelete(source);
  safeDelete(destinationParent);
  safeDelete(sourceParent);
  return result;
}

function auditRotateType(reference, name, type, values) {
  if (typeof type !== "number") return { name, available: false };
  let source;
  let destination;
  try {
    source = makeSeedMat(reference, 2, 3, type, values);
    destination = new reference.Mat();
    return {
      name,
      available: true,
      audit: auditTypedMatCall(
        () => reference.rotate(source, destination, 2),
        [
          ["source", source],
          ["destination", destination],
        ],
      ),
    };
  } catch (error) {
    return {
      name,
      available: true,
      setup: captureCall(() => {
        throw error;
      }),
    };
  } finally {
    safeDelete(destination);
    safeDelete(source);
  }
}

function auditRotate(reference) {
  const aritySource = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const arityDestination = new reference.Mat();
  const arity = {
    functionLength: capturePrimitive(() => reference.rotate.length),
    zero: captureCall(() => reference.rotate()),
    one: captureCall(() => reference.rotate(aritySource)),
    two: captureCall(() => reference.rotate(aritySource, arityDestination)),
    three: captureCall(() => reference.rotate(aritySource, arityDestination, 0)),
    four: captureCall(() => reference.rotate(aritySource, arityDestination, 0, 1)),
    destinationAfter: capturePrimitive(() => summarizeMat(arityDestination)),
  };
  safeDelete(arityDestination);
  safeDelete(aritySource);

  const codes = [
    ["negative two", -2, true],
    ["negative one", -1, true],
    ["negative fraction", -1.9, true],
    ["negative subunit fraction", -0.9],
    ["negative zero", -0],
    ["zero", 0],
    ["positive subunit fraction", 0.9],
    ["one", 1],
    ["positive fraction", 1.9],
    ["two", 2],
    ["two fraction", 2.9],
    ["three", 3, true],
    ["i32 maximum", 2_147_483_647, true],
    ["i32 maximum plus one", 2_147_483_648],
    ["i32 minimum", -2_147_483_648, true],
    ["i32 minimum minus one", -2_147_483_649],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["true", true],
    ["false", false],
    ["null", null],
    ["numeric string", "1"],
    ["explicit undefined", undefined],
  ].map(([label, value, nativeInvalid = false]) =>
    auditRotateCode(reference, label, value, nativeInvalid),
  );

  const argumentSource = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [1]);
  const argumentDestination = new reference.Mat();
  const argumentTypes = {
    nullSource: captureCall(() => reference.rotate(null, argumentDestination, 0)),
    undefinedSource: captureCall(() => reference.rotate(undefined, argumentDestination, 0)),
    objectSource: captureCall(() => reference.rotate({}, argumentDestination, 0)),
    nullDestination: captureCall(() => reference.rotate(argumentSource, null, 0)),
    undefinedDestination: captureCall(() => reference.rotate(argumentSource, undefined, 0)),
    objectDestination: captureCall(() => reference.rotate(argumentSource, {}, 0)),
  };
  safeDelete(argumentDestination);
  safeDelete(argumentSource);

  const empty = [0, 1, 2].map((code) => {
    const source = new reference.Mat();
    const destination = new reference.Mat();
    const audit = auditTransposeCall(
      () => reference.rotate(source, destination, code),
      [
        ["source", source],
        ["destination", destination],
      ],
    );
    safeDelete(destination);
    safeDelete(source);
    return { code, audit };
  });
  const emptySource = new reference.Mat();
  const populatedDestination = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const emptyIntoFull = auditTransposeCall(
    () => reference.rotate(emptySource, populatedDestination, 1),
    [
      ["source", emptySource],
      ["destination", populatedDestination],
    ],
  );
  safeDelete(populatedDestination);
  safeDelete(emptySource);

  const deletedSource = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [7]);
  const liveDestination = new reference.Mat();
  deletedSource.delete();
  const deletedSourceAudit = auditTransposeCall(
    () => reference.rotate(deletedSource, liveDestination, 0),
    [
      ["source", deletedSource],
      ["destination", liveDestination],
    ],
  );
  safeDelete(liveDestination);

  const liveSource = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [8]);
  const deletedDestination = new reference.Mat();
  deletedDestination.delete();
  const deletedDestinationAudit = auditTransposeCall(
    () => reference.rotate(liveSource, deletedDestination, 0),
    [
      ["source", liveSource],
      ["destination", deletedDestination],
    ],
  );
  safeDelete(liveSource);

  const stridedParent = makeSeedMat(
    reference,
    4,
    5,
    reference.CV_8UC1,
    Array.from({ length: 20 }, (_, index) => index + 1),
  );
  const stridedSource = stridedParent.roi(new reference.Rect(1, 1, 3, 2));
  const stridedDestination = new reference.Mat();
  const strided = auditTransposeCall(
    () => reference.rotate(stridedSource, stridedDestination, 2),
    [
      ["source", stridedSource],
      ["destination", stridedDestination],
      ["sourceParent", stridedParent],
    ],
  );
  safeDelete(stridedDestination);
  safeDelete(stridedSource);
  safeDelete(stridedParent);

  const types = [
    ["CV_8UC1", reference.CV_8UC1, [1, 2, 3, 4, 5, 6]],
    ["CV_8UC3", reference.CV_8UC3, Array.from({ length: 18 }, (_, index) => index + 1)],
    ["CV_8SC2", reference.CV_8SC2, [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12]],
    ["CV_16UC1", reference.CV_16UC1, [1, 256, 513, 1024, 4096, 65_535]],
    [
      "CV_16SC2",
      reference.CV_16SC2,
      [-1, 2, -300, 400, -500, 600, -700, 800, -900, 1000, -1100, 1200],
    ],
    ["CV_32SC1", reference.CV_32SC1, [-1, 2, -300_000, 400_000, -500_000, 600_000]],
    [
      "CV_32FC2",
      reference.CV_32FC2,
      [-1.5, 2.25, -3.5, 4.75, -5.5, 6.25, -7.5, 8.75, -9.5, 10.25, -11.5, 12.75],
    ],
    ["CV_64FC1", reference.CV_64FC1, [-1.5, 2.25, -3.5, 4.75, -5.5, 6.25]],
    ["CV_16FC1", reference.CV_16FC1, [1, 2, 3, 4, 5, 6]],
  ].map(([name, type, values]) => auditRotateType(reference, name, type, values));

  return {
    constants: {
      ROTATE_90_CLOCKWISE: capturePrimitive(() => reference.ROTATE_90_CLOCKWISE),
      ROTATE_180: capturePrimitive(() => reference.ROTATE_180),
      ROTATE_90_COUNTERCLOCKWISE: capturePrimitive(() => reference.ROTATE_90_COUNTERCLOCKWISE),
    },
    arity,
    argumentTypes,
    codes,
    destinationReplacement: [
      auditRotateDestination(reference, "empty", 0, () => new reference.Mat()),
      auditRotateDestination(reference, "correct-metadata-clockwise", 0, () =>
        makeSeedMat(reference, 3, 2, reference.CV_8UC1, new Uint8Array(6).fill(99)),
      ),
      auditRotateDestination(reference, "correct-metadata-half-turn", 1, () =>
        makeSeedMat(reference, 2, 3, reference.CV_8UC1, new Uint8Array(6).fill(99)),
      ),
      auditRotateDestination(reference, "wrong-shape", 2, () =>
        makeSeedMat(reference, 2, 3, reference.CV_8UC1, new Uint8Array(6).fill(99)),
      ),
      auditRotateDestination(reference, "wrong-type-and-channels", 0, () =>
        makeSeedMat(reference, 3, 2, reference.CV_32FC2, new Float32Array(12).fill(99)),
      ),
    ],
    inPlace: [0, 1, 2].map((code) => auditRotateInPlace(reference, code)),
    roi: [
      auditRotateRoi(reference, "compatible-separate", [1, 1, 3, 2], [1, 1, 2, 3], false, 0),
      auditRotateRoi(
        reference,
        "incompatible-shape-separate",
        [1, 1, 3, 2],
        [1, 1, 3, 2],
        false,
        0,
      ),
      auditRotateTypeRoi(reference),
      auditRotateRoi(reference, "same-parent-non-overlap", [0, 0, 2, 2], [4, 3, 2, 2], true, 0),
      auditRotateRoi(
        reference,
        "same-parent-overlap-clockwise",
        [0, 0, 3, 2],
        [1, 0, 2, 3],
        true,
        0,
      ),
      auditRotateRoi(
        reference,
        "same-parent-overlap-half-turn",
        [0, 0, 3, 2],
        [1, 0, 3, 2],
        true,
        1,
      ),
      auditRotateRoi(
        reference,
        "same-parent-overlap-counterclockwise",
        [0, 0, 3, 2],
        [1, 1, 2, 3],
        true,
        2,
      ),
    ],
    empty,
    emptyIntoFull,
    deleted: { source: deletedSourceAudit, destination: deletedDestinationAudit },
    strided,
    types,
    halfFloatConstants: {
      CV_16F: encodeValue(reference.CV_16F),
      CV_16FC1: encodeValue(reference.CV_16FC1),
    },
  };
}

function auditCountNonZeroType(reference, name, type, values) {
  if (typeof type !== "number") return { name, available: false };
  let source;
  try {
    source = makeSeedMat(reference, 2, 3, type, values);
    return {
      name,
      available: true,
      before: capturePrimitive(() => summarizeTypedMat(source)),
      call: captureCall(() => reference.countNonZero(source)),
      after: capturePrimitive(() => summarizeTypedMat(source)),
    };
  } catch (error) {
    return {
      name,
      available: true,
      setup: captureCall(() => {
        throw error;
      }),
    };
  } finally {
    safeDelete(source);
  }
}

function auditCountNonZero(reference) {
  const source = makeSeedMat(reference, 2, 3, reference.CV_8UC1, [0, 1, 2, 0, 3, 0]);
  const arity = {
    functionLength: capturePrimitive(() => reference.countNonZero.length),
    zero: captureCall(() => reference.countNonZero()),
    one: captureCall(() => reference.countNonZero(source)),
    two: captureCall(() => reference.countNonZero(source, 1)),
    three: captureCall(() => reference.countNonZero(source, 1, 2)),
  };
  const argumentTypes = {
    null: captureCall(() => reference.countNonZero(null)),
    undefined: captureCall(() => reference.countNonZero(undefined)),
    object: captureCall(() => reference.countNonZero({})),
    number: captureCall(() => reference.countNonZero(1)),
    boolean: captureCall(() => reference.countNonZero(true)),
    string: captureCall(() => reference.countNonZero("x")),
  };
  safeDelete(source);

  const empty = new reference.Mat();
  const emptyAudit = {
    before: capturePrimitive(() => summarizeMat(empty)),
    call: captureCall(() => reference.countNonZero(empty)),
    after: capturePrimitive(() => summarizeMat(empty)),
  };
  safeDelete(empty);

  const parent = makeSeedMat(
    reference,
    4,
    5,
    reference.CV_16SC1,
    [0, 1, 0, 2, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 7, 0, 8, 0, 9],
  );
  const region = parent.roi(new reference.Rect(1, 1, 3, 2));
  const roi = {
    before: capturePrimitive(() => summarizeTypedMat(region)),
    call: captureCall(() => reference.countNonZero(region)),
    after: capturePrimitive(() => summarizeTypedMat(region)),
    parentAfter: capturePrimitive(() => summarizeTypedMat(parent)),
  };
  safeDelete(region);
  safeDelete(parent);

  const deleted = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [1]);
  deleted.delete();
  const deletedAudit = captureCall(() => reference.countNonZero(deleted));

  const types = [
    ["CV_8UC1", reference.CV_8UC1, [0, 1, 255, 0, 2, 0]],
    ["CV_8SC1", reference.CV_8SC1, [0, -1, 127, -128, 0, 2]],
    ["CV_16UC1", reference.CV_16UC1, [0, 1, 65535, 0, 2, 0]],
    ["CV_16SC1", reference.CV_16SC1, [0, -1, 32767, -32768, 0, 2]],
    ["CV_32SC1", reference.CV_32SC1, [0, -1, 2147483647, -2147483648, 0, 2]],
    [
      "CV_32FC1",
      reference.CV_32FC1,
      [0, -0, 1.401298464324817e-45, Number.NaN, Infinity, -Infinity],
    ],
    ["CV_64FC1", reference.CV_64FC1, [0, -0, Number.MIN_VALUE, Number.NaN, Infinity, -Infinity]],
    ["CV_16FC1", reference.CV_16FC1, [0, -0, 1.5, Number.NaN, Infinity, -Infinity]],
  ].map(([name, type, values]) => auditCountNonZeroType(reference, name, type, values));

  const multiChannel = [
    ["CV_8UC2", reference.CV_8UC2, [0, 1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0]],
    ["CV_64FC2", reference.CV_64FC2, [0, 1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0]],
    ["CV_8UC3", reference.CV_8UC3, [0, 1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 7, 0, 8, 0, 9, 0]],
  ].map(([name, type, values]) => auditCountNonZeroType(reference, name, type, values));

  return {
    arity,
    argumentTypes,
    empty: emptyAudit,
    roi,
    deleted: deletedAudit,
    types,
    multiChannel,
  };
}

const FLOAT_UNARY_METHODS = ["exp", "log", "sqrt"];
const FLOAT_INTEGER_TYPES = [
  ["CV_8UC1", "u8", 1, [0, 1, 2, 3, 4, 5]],
  ["CV_8SC1", "i8", 1, [-3, -2, -1, 0, 1, 2]],
  ["CV_16UC1", "u16", 1, [0, 1, 2, 3, 4, 5]],
  ["CV_16SC1", "i16", 1, [-3, -2, -1, 0, 1, 2]],
  ["CV_32SC1", "i32", 1, [-3, -2, -1, 0, 1, 2]],
];
const FLOAT_VALID_TYPES = [
  ["CV_32FC1", "f32", 1, [-1.5, 2.25, -3.5, 4.75, -5.5, 6.25]],
  ["CV_32FC3", "f32", 3, [-1, 2, 3, -4, 5, 6, -7, 8, 9, -10, 11, 12, -13, 14, 15, -16, 17, 18]],
  ["CV_64FC1", "f64", 1, [-1.5, 2.25, -3.5, 4.75, -5.5, 6.25]],
  ["CV_64FC2", "f64", 2, [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12]],
];

function referenceTypeFromName(reference, name) {
  return reference[name];
}

function auditReferenceFloatDestination(reference, method, name, createDestination) {
  const source = makeSeedMat(
    reference,
    2,
    3,
    reference.CV_32FC2,
    [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3],
  );
  const destination = createDestination();
  const audit = auditTypedMatCall(
    () => reference[method](source, destination),
    [
      ["source", source],
      ["destination", destination],
    ],
  );
  safeDelete(destination);
  safeDelete(source);
  return { name, audit };
}

function auditReferenceUnaryRoi(
  reference,
  method,
  name,
  sourceRect,
  destinationRect,
  sameParent,
  destinationType = reference.CV_32FC1,
) {
  const sourceParent = makeSeedMat(
    reference,
    4,
    7,
    reference.CV_32FC1,
    Array.from({ length: 28 }, (_, index) => (index + 1) / 10),
  );
  const destinationParent = sameParent
    ? sourceParent
    : makeSeedMat(
        reference,
        4,
        7,
        destinationType,
        Array.from({ length: 28 }, () => 99),
      );
  const source = sourceParent.roi(new reference.Rect(...sourceRect));
  const destination = destinationParent.roi(new reference.Rect(...destinationRect));
  const audit = auditTypedMatCall(
    () => reference[method](source, destination),
    [
      ["source", source],
      ["destination", destination],
      ["sourceParent", sourceParent],
      ["destinationParent", destinationParent],
    ],
  );
  safeDelete(destination);
  safeDelete(source);
  if (!sameParent) safeDelete(destinationParent);
  safeDelete(sourceParent);
  return { name, sameParent, audit };
}

function auditReferenceUnaryType(reference, method, name, type, values) {
  const source = makeSeedMat(reference, 2, 3, type, values);
  const destination = new reference.Mat();
  const audit = auditTypedMatCall(
    () => reference[method](source, destination),
    [
      ["source", source],
      ["destination", destination],
    ],
  );
  safeDelete(destination);
  safeDelete(source);
  return { name, available: true, audit };
}

function auditReferenceUnary(reference, method) {
  const source = makeSeedMat(reference, 1, 2, reference.CV_32FC1, [1, 4]);
  const destination = new reference.Mat();
  const arity = {
    functionLength: capturePrimitive(() => reference[method].length),
    zero: captureCall(() => reference[method]()),
    one: captureCall(() => reference[method](source)),
    two: captureCall(() => reference[method](source, destination)),
    three: captureCall(() => reference[method](source, destination, 1)),
    destinationAfter: capturePrimitive(() => summarizeTypedMat(destination)),
  };
  const argumentTypes = {
    source: {
      null: captureCall(() => reference[method](null, destination)),
      undefined: captureCall(() => reference[method](undefined, destination)),
      object: captureCall(() => reference[method]({}, destination)),
      number: captureCall(() => reference[method](1, destination)),
      string: captureCall(() => reference[method]("x", destination)),
    },
    destination: {
      null: captureCall(() => reference[method](source, null)),
      undefined: captureCall(() => reference[method](source, undefined)),
      object: captureCall(() => reference[method](source, {})),
      number: captureCall(() => reference[method](source, 1)),
      string: captureCall(() => reference[method](source, "x")),
    },
  };
  safeDelete(destination);
  safeDelete(source);

  const deletedSource = makeSeedMat(reference, 1, 1, reference.CV_32FC1, [1]);
  const liveDestination = new reference.Mat();
  deletedSource.delete();
  const deletedSourceAudit = auditTypedMatCall(
    () => reference[method](deletedSource, liveDestination),
    [
      ["source", deletedSource],
      ["destination", liveDestination],
    ],
  );
  safeDelete(liveDestination);
  const liveSource = makeSeedMat(reference, 1, 1, reference.CV_32FC1, [1]);
  const deletedDestination = new reference.Mat();
  deletedDestination.delete();
  const deletedDestinationAudit = auditTypedMatCall(
    () => reference[method](liveSource, deletedDestination),
    [
      ["source", liveSource],
      ["destination", deletedDestination],
    ],
  );
  safeDelete(liveSource);

  const freshEmptySource = new reference.Mat();
  const freshEmptyDestination = new reference.Mat();
  const freshEmpty = auditTypedMatCall(
    () => reference[method](freshEmptySource, freshEmptyDestination),
    [
      ["source", freshEmptySource],
      ["destination", freshEmptyDestination],
    ],
  );
  safeDelete(freshEmptyDestination);
  safeDelete(freshEmptySource);
  const freshEmptyIntoFullSource = new reference.Mat();
  const populatedDestination = makeSeedMat(reference, 1, 2, reference.CV_32FC1, [7, 8]);
  const freshEmptyIntoFull = auditTypedMatCall(
    () => reference[method](freshEmptyIntoFullSource, populatedDestination),
    [
      ["source", freshEmptyIntoFullSource],
      ["destination", populatedDestination],
    ],
  );
  safeDelete(populatedDestination);
  safeDelete(freshEmptyIntoFullSource);

  const typedEmpty = [
    ["0x0-F32C1", 0, 0, reference.CV_32FC1],
    ["0x3-F32C2", 0, 3, reference.CV_32FC2],
    ["3x0-F64C1", 3, 0, reference.CV_64FC1],
  ].map(([name, rows, columns, type]) => {
    const typedSource = new reference.Mat(rows, columns, type);
    const typedDestination = new reference.Mat();
    const audit = auditTypedMatCall(
      () => reference[method](typedSource, typedDestination),
      [
        ["source", typedSource],
        ["destination", typedDestination],
      ],
    );
    safeDelete(typedDestination);
    safeDelete(typedSource);
    return { name, audit };
  });

  const validTypes = FLOAT_VALID_TYPES.map(([name, , , values]) =>
    auditReferenceUnaryType(
      reference,
      method,
      name,
      referenceTypeFromName(reference, name),
      values,
    ),
  );
  const unsupportedIntegerTypes =
    method === "sqrt"
      ? []
      : FLOAT_INTEGER_TYPES.map(([name, , , values]) =>
          auditReferenceUnaryType(
            reference,
            method,
            name,
            referenceTypeFromName(reference, name),
            values,
          ),
        );
  const numericEdges = [
    [
      "F32",
      reference.CV_32FC1,
      [
        Number.NEGATIVE_INFINITY,
        -4,
        -1.401298464324817e-45,
        -0,
        0,
        1.401298464324817e-45,
        4,
        Number.POSITIVE_INFINITY,
        Number.NaN,
      ],
    ],
    [
      "F64",
      reference.CV_64FC1,
      [
        Number.NEGATIVE_INFINITY,
        -4,
        -Number.MIN_VALUE,
        -0,
        0,
        Number.MIN_VALUE,
        4,
        Number.POSITIVE_INFINITY,
        Number.NaN,
      ],
    ],
  ].map(([name, type, values]) => {
    const edgeSource = makeSeedMat(reference, 1, values.length, type, values);
    const edgeDestination = new reference.Mat();
    const audit = auditTypedMatCall(
      () => reference[method](edgeSource, edgeDestination),
      [
        ["source", edgeSource],
        ["destination", edgeDestination],
      ],
    );
    safeDelete(edgeDestination);
    safeDelete(edgeSource);
    return { name, audit };
  });

  return {
    method,
    arity,
    argumentTypes,
    destinationReplacement: [
      auditReferenceFloatDestination(reference, method, "empty", () => new reference.Mat()),
      auditReferenceFloatDestination(reference, method, "correct-metadata", () =>
        makeSeedMat(
          reference,
          2,
          3,
          reference.CV_32FC2,
          Array.from({ length: 12 }, () => 99),
        ),
      ),
      auditReferenceFloatDestination(reference, method, "wrong-shape", () =>
        makeSeedMat(
          reference,
          3,
          2,
          reference.CV_32FC2,
          Array.from({ length: 12 }, () => 99),
        ),
      ),
      auditReferenceFloatDestination(reference, method, "wrong-type-and-channels", () =>
        makeSeedMat(
          reference,
          2,
          3,
          reference.CV_64FC1,
          Array.from({ length: 6 }, () => 99),
        ),
      ),
    ],
    roi: [
      auditReferenceUnaryRoi(
        reference,
        method,
        "compatible-separate",
        [1, 1, 3, 2],
        [2, 1, 3, 2],
        false,
      ),
      auditReferenceUnaryRoi(
        reference,
        method,
        "incompatible-detach",
        [1, 1, 3, 2],
        [2, 1, 2, 3],
        false,
      ),
      auditReferenceUnaryRoi(
        reference,
        method,
        "wrong-type-detach",
        [1, 1, 3, 2],
        [2, 1, 3, 2],
        false,
        reference.CV_64FC1,
      ),
      auditReferenceUnaryRoi(reference, method, "exact-alias", [1, 1, 3, 2], [1, 1, 3, 2], true),
      auditReferenceUnaryRoi(reference, method, "overlap-right", [0, 1, 3, 2], [1, 1, 3, 2], true),
      auditReferenceUnaryRoi(reference, method, "overlap-down", [1, 0, 3, 2], [1, 1, 3, 2], true),
    ],
    empty: { fresh: freshEmpty, freshIntoFull: freshEmptyIntoFull, typed: typedEmpty },
    deleted: { source: deletedSourceAudit, destination: deletedDestinationAudit },
    validTypes,
    unsupportedIntegerTypes,
    numericEdges,
    excludedUnsafeIntegerTypes: method === "sqrt" ? FLOAT_INTEGER_TYPES.map(([name]) => name) : [],
  };
}

function auditReferencePowCall(reference, power, matrices, source, destination) {
  return auditTypedMatCall(() => reference.pow(source, power, destination), matrices);
}

function auditReferencePowRoi(reference, name, sourceRect, destinationRect, sameParent) {
  const sourceParent = makeSeedMat(
    reference,
    4,
    7,
    reference.CV_32FC1,
    Array.from({ length: 28 }, (_, index) => index + 1),
  );
  const destinationParent = sameParent
    ? sourceParent
    : makeSeedMat(
        reference,
        4,
        7,
        reference.CV_32FC1,
        Array.from({ length: 28 }, () => 99),
      );
  const source = sourceParent.roi(new reference.Rect(...sourceRect));
  const destination = destinationParent.roi(new reference.Rect(...destinationRect));
  const audit = auditReferencePowCall(
    reference,
    2,
    [
      ["source", source],
      ["destination", destination],
      ["sourceParent", sourceParent],
      ["destinationParent", destinationParent],
    ],
    source,
    destination,
  );
  safeDelete(destination);
  safeDelete(source);
  if (!sameParent) safeDelete(destinationParent);
  safeDelete(sourceParent);
  return { name, sameParent, audit };
}

function auditReferencePow(reference) {
  const source = makeSeedMat(reference, 1, 5, reference.CV_64FC1, [4, -4, 0, -0, 2]);
  const destination = new reference.Mat();
  const arity = {
    functionLength: capturePrimitive(() => reference.pow.length),
    zero: captureCall(() => reference.pow()),
    one: captureCall(() => reference.pow(source)),
    two: captureCall(() => reference.pow(source, 2)),
    three: captureCall(() => reference.pow(source, 2, destination)),
    four: captureCall(() => reference.pow(source, 2, destination, 1)),
    destinationAfter: capturePrimitive(() => summarizeTypedMat(destination)),
  };
  const argumentTypes = {
    source: {
      null: captureCall(() => reference.pow(null, 2, destination)),
      undefined: captureCall(() => reference.pow(undefined, 2, destination)),
      object: captureCall(() => reference.pow({}, 2, destination)),
      number: captureCall(() => reference.pow(1, 2, destination)),
      string: captureCall(() => reference.pow("x", 2, destination)),
    },
    power: [
      ["negative-zero", -0],
      ["half", 0.5],
      ["f64-precision", 1.0000000000000002],
      ["NaN", Number.NaN],
      ["positive-infinity", Number.POSITIVE_INFINITY],
      ["negative-infinity", Number.NEGATIVE_INFINITY],
      ["true", true],
      ["false", false],
      ["null", null],
      ["undefined", undefined],
      ["string", "2"],
      ["object", {}],
      ["array", []],
    ].map(([label, power]) => {
      const scalarDestination = new reference.Mat();
      const audit = auditReferencePowCall(
        reference,
        power,
        [
          ["source", source],
          ["destination", scalarDestination],
        ],
        source,
        scalarDestination,
      );
      safeDelete(scalarDestination);
      return { label, audit };
    }),
    destination: {
      null: captureCall(() => reference.pow(source, 2, null)),
      undefined: captureCall(() => reference.pow(source, 2, undefined)),
      object: captureCall(() => reference.pow(source, 2, {})),
      number: captureCall(() => reference.pow(source, 2, 1)),
      string: captureCall(() => reference.pow(source, 2, "x")),
    },
  };
  safeDelete(destination);
  safeDelete(source);

  const destinationReplacement = [
    ["empty", () => new reference.Mat()],
    [
      "correct-metadata",
      () =>
        makeSeedMat(
          reference,
          2,
          3,
          reference.CV_32FC2,
          Array.from({ length: 12 }, () => 99),
        ),
    ],
    [
      "wrong-shape",
      () =>
        makeSeedMat(
          reference,
          3,
          2,
          reference.CV_32FC2,
          Array.from({ length: 12 }, () => 99),
        ),
    ],
    [
      "wrong-type-and-channels",
      () =>
        makeSeedMat(
          reference,
          2,
          3,
          reference.CV_64FC1,
          Array.from({ length: 6 }, () => 99),
        ),
    ],
  ].map(([name, createDestination]) => {
    const replacementSource = makeSeedMat(
      reference,
      2,
      3,
      reference.CV_32FC2,
      [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12],
    );
    const replacementDestination = createDestination();
    const audit = auditReferencePowCall(
      reference,
      2,
      [
        ["source", replacementSource],
        ["destination", replacementDestination],
      ],
      replacementSource,
      replacementDestination,
    );
    safeDelete(replacementDestination);
    safeDelete(replacementSource);
    return { name, audit };
  });

  const types = [...FLOAT_INTEGER_TYPES, ...FLOAT_VALID_TYPES].map(([name, , , values]) => {
    const typedSource = makeSeedMat(
      reference,
      2,
      3,
      referenceTypeFromName(reference, name),
      values,
    );
    const typedDestination = new reference.Mat();
    const audit = auditReferencePowCall(
      reference,
      2,
      [
        ["source", typedSource],
        ["destination", typedDestination],
      ],
      typedSource,
      typedDestination,
    );
    safeDelete(typedDestination);
    safeDelete(typedSource);
    return { name, available: true, audit };
  });

  const integerPowers = FLOAT_INTEGER_TYPES.map(([name]) => {
    const signed = name.includes("8S") || name.includes("16S") || name.includes("32S");
    const values = signed ? [-3, -2, -1, 0, 1, 2, 3] : [0, 1, 2, 3, 4];
    return {
      name,
      input: values,
      powers: [0, 1, 2, 3, -1, -2].map((power) => {
        const integerSource = makeSeedMat(
          reference,
          1,
          values.length,
          referenceTypeFromName(reference, name),
          values,
        );
        const integerDestination = new reference.Mat();
        const audit = auditReferencePowCall(
          reference,
          power,
          [
            ["source", integerSource],
            ["destination", integerDestination],
          ],
          integerSource,
          integerDestination,
        );
        safeDelete(integerDestination);
        safeDelete(integerSource);
        return { power, audit };
      }),
    };
  });

  const numericEdges = [
    ["F32", reference.CV_32FC1],
    ["F64", reference.CV_64FC1],
  ].map(([name, type]) => ({
    name,
    powers: [-1, -0.5, -0, 0, 0.5, 1, 2, 3].map((power) => {
      const edgeSource = makeSeedMat(reference, 1, 9, type, [
        Number.NEGATIVE_INFINITY,
        -4,
        -0,
        0,
        4,
        Number.POSITIVE_INFINITY,
        Number.NaN,
        1,
        -1,
      ]);
      const edgeDestination = new reference.Mat();
      const audit = auditReferencePowCall(
        reference,
        power,
        [
          ["source", edgeSource],
          ["destination", edgeDestination],
        ],
        edgeSource,
        edgeDestination,
      );
      safeDelete(edgeDestination);
      safeDelete(edgeSource);
      return { power: encodeValue(power), audit };
    }),
  }));

  const emptySource = new reference.Mat();
  const emptyDestination = new reference.Mat();
  const freshEmpty = auditReferencePowCall(
    reference,
    2,
    [
      ["source", emptySource],
      ["destination", emptyDestination],
    ],
    emptySource,
    emptyDestination,
  );
  safeDelete(emptyDestination);
  safeDelete(emptySource);
  const emptyIntoFullSource = new reference.Mat();
  const fullDestination = makeSeedMat(reference, 1, 2, reference.CV_32FC1, [7, 8]);
  const freshEmptyIntoFull = auditReferencePowCall(
    reference,
    2,
    [
      ["source", emptyIntoFullSource],
      ["destination", fullDestination],
    ],
    emptyIntoFullSource,
    fullDestination,
  );
  safeDelete(fullDestination);
  safeDelete(emptyIntoFullSource);

  const deletedSource = makeSeedMat(reference, 1, 1, reference.CV_32FC1, [1]);
  const liveDestination = new reference.Mat();
  deletedSource.delete();
  const deletedSourceAudit = auditReferencePowCall(
    reference,
    2,
    [
      ["source", deletedSource],
      ["destination", liveDestination],
    ],
    deletedSource,
    liveDestination,
  );
  safeDelete(liveDestination);
  const liveSource = makeSeedMat(reference, 1, 1, reference.CV_32FC1, [1]);
  const deletedDestination = new reference.Mat();
  deletedDestination.delete();
  const deletedDestinationAudit = auditReferencePowCall(
    reference,
    2,
    [
      ["source", liveSource],
      ["destination", deletedDestination],
    ],
    liveSource,
    deletedDestination,
  );
  safeDelete(liveSource);

  return {
    arity,
    argumentTypes,
    destinationReplacement,
    roi: [
      auditReferencePowRoi(reference, "compatible-separate", [1, 1, 3, 2], [2, 1, 3, 2], false),
      auditReferencePowRoi(reference, "incompatible-detach", [1, 1, 3, 2], [2, 1, 2, 3], false),
      auditReferencePowRoi(reference, "exact-alias", [1, 1, 3, 2], [1, 1, 3, 2], true),
      auditReferencePowRoi(reference, "overlap-right", [0, 1, 3, 2], [1, 1, 3, 2], true),
      auditReferencePowRoi(reference, "overlap-down", [1, 0, 3, 2], [1, 1, 3, 2], true),
    ],
    empty: { fresh: freshEmpty, freshIntoFull: freshEmptyIntoFull },
    deleted: { source: deletedSourceAudit, destination: deletedDestinationAudit },
    types,
    integerPowers,
    numericEdges,
    excludedUnsafeIntegerPowers: ["fractional", "NaN", "positive-infinity", "negative-infinity"],
  };
}

function auditReferenceMagnitudeRoi(reference, name, aliasInput, sourceRect, destinationRect) {
  const aliasParent = makeSeedMat(
    reference,
    4,
    7,
    reference.CV_32FC1,
    Array.from({ length: 28 }, (_, index) => index + 1),
  );
  const other = makeSeedMat(reference, 2, 3, reference.CV_32FC1, [0, 0, 0, 0, 0, 0]);
  const aliasedInput = aliasParent.roi(new reference.Rect(...sourceRect));
  const destination = aliasParent.roi(new reference.Rect(...destinationRect));
  const x = aliasInput === "x" ? aliasedInput : other;
  const y = aliasInput === "y" ? aliasedInput : other;
  const audit = auditTypedMatCall(
    () => reference.magnitude(x, y, destination),
    [
      ["x", x],
      ["y", y],
      ["destination", destination],
      ["aliasParent", aliasParent],
    ],
  );
  safeDelete(destination);
  safeDelete(aliasedInput);
  safeDelete(other);
  safeDelete(aliasParent);
  return { name, aliasInput, audit };
}

function auditReferenceMagnitude(reference) {
  const x = makeSeedMat(reference, 1, 2, reference.CV_32FC1, [3, 4]);
  const y = makeSeedMat(reference, 1, 2, reference.CV_32FC1, [4, 3]);
  const destination = new reference.Mat();
  const arity = {
    functionLength: capturePrimitive(() => reference.magnitude.length),
    zero: captureCall(() => reference.magnitude()),
    one: captureCall(() => reference.magnitude(x)),
    two: captureCall(() => reference.magnitude(x, y)),
    three: captureCall(() => reference.magnitude(x, y, destination)),
    four: captureCall(() => reference.magnitude(x, y, destination, 1)),
    destinationAfter: capturePrimitive(() => summarizeTypedMat(destination)),
  };
  const invalidMat = (position, value) =>
    captureCall(() =>
      reference.magnitude(
        position === "x" ? value : x,
        position === "y" ? value : y,
        position === "destination" ? value : destination,
      ),
    );
  const argumentTypes = Object.fromEntries(
    ["x", "y", "destination"].map((position) => [
      position,
      Object.fromEntries(
        [
          ["null", null],
          ["undefined", undefined],
          ["object", {}],
          ["number", 1],
          ["string", "x"],
        ].map(([name, value]) => [name, invalidMat(position, value)]),
      ),
    ]),
  );
  safeDelete(destination);
  safeDelete(y);
  safeDelete(x);

  const destinationReplacement = [
    ["empty", () => new reference.Mat()],
    [
      "correct-metadata",
      () =>
        makeSeedMat(
          reference,
          2,
          3,
          reference.CV_32FC1,
          Array.from({ length: 6 }, () => 99),
        ),
    ],
    [
      "wrong-shape",
      () =>
        makeSeedMat(
          reference,
          3,
          2,
          reference.CV_32FC1,
          Array.from({ length: 6 }, () => 99),
        ),
    ],
    [
      "wrong-type-and-channels",
      () =>
        makeSeedMat(
          reference,
          2,
          3,
          reference.CV_64FC2,
          Array.from({ length: 12 }, () => 99),
        ),
    ],
  ].map(([name, createDestination]) => {
    const left = makeSeedMat(reference, 2, 3, reference.CV_32FC1, [1, -2, 3, -4, 5, -6]);
    const right = makeSeedMat(reference, 2, 3, reference.CV_32FC1, [6, 5, 4, 3, 2, 1]);
    const output = createDestination();
    const audit = auditTypedMatCall(
      () => reference.magnitude(left, right, output),
      [
        ["x", left],
        ["y", right],
        ["destination", output],
      ],
    );
    safeDelete(output);
    safeDelete(right);
    safeDelete(left);
    return { name, audit };
  });

  const validTypes = FLOAT_VALID_TYPES.map(([name, , , values]) => {
    const type = referenceTypeFromName(reference, name);
    const left = makeSeedMat(reference, 2, 3, type, values);
    const right = makeSeedMat(
      reference,
      2,
      3,
      type,
      values.map((value) => Math.abs(value) + 1),
    );
    const output = new reference.Mat();
    const audit = auditTypedMatCall(
      () => reference.magnitude(left, right, output),
      [
        ["x", left],
        ["y", right],
        ["destination", output],
      ],
    );
    safeDelete(output);
    safeDelete(right);
    safeDelete(left);
    return { name, available: true, audit };
  });
  const unsupportedIntegerTypes = FLOAT_INTEGER_TYPES.map(([name, , , values]) => {
    const type = referenceTypeFromName(reference, name);
    const left = makeSeedMat(reference, 2, 3, type, values);
    const right = makeSeedMat(
      reference,
      2,
      3,
      type,
      values.map((value) => Math.abs(value) + 1),
    );
    const output = new reference.Mat();
    const audit = auditTypedMatCall(
      () => reference.magnitude(left, right, output),
      [
        ["x", left],
        ["y", right],
        ["destination", output],
      ],
    );
    safeDelete(output);
    safeDelete(right);
    safeDelete(left);
    return { name, audit };
  });

  const mismatchBase = makeSeedMat(reference, 2, 2, reference.CV_32FC1, [1, 2, 3, 4]);
  const mismatches = [
    ["shape", makeSeedMat(reference, 1, 4, reference.CV_32FC1, [1, 2, 3, 4])],
    ["depth", makeSeedMat(reference, 2, 2, reference.CV_64FC1, [1, 2, 3, 4])],
    ["channels", makeSeedMat(reference, 2, 2, reference.CV_32FC2, [1, 2, 3, 4, 5, 6, 7, 8])],
    ["empty", new reference.Mat()],
  ].map(([name, right]) => {
    const output = new reference.Mat();
    const audit = auditTypedMatCall(
      () => reference.magnitude(mismatchBase, right, output),
      [
        ["x", mismatchBase],
        ["y", right],
        ["destination", output],
      ],
    );
    safeDelete(output);
    safeDelete(right);
    return { name, audit };
  });
  safeDelete(mismatchBase);

  const numericEdges = [
    [
      "F32",
      reference.CV_32FC1,
      [
        0,
        -0,
        3,
        -3,
        3.4028234663852886e38,
        3.4028234663852886e38,
        1.1754943508222875e-38,
        1.401298464324817e-45,
        Infinity,
        -Infinity,
        NaN,
        1,
        Infinity,
        NaN,
      ],
      [
        0,
        0,
        4,
        4,
        0,
        3.4028234663852886e38,
        0,
        1.401298464324817e-45,
        1,
        Infinity,
        1,
        NaN,
        NaN,
        Infinity,
      ],
    ],
    [
      "F64",
      reference.CV_64FC1,
      [
        0,
        -0,
        3,
        -3,
        Number.MAX_VALUE,
        Number.MAX_VALUE,
        Number.MIN_VALUE,
        2.2250738585072014e-308,
        Infinity,
        -Infinity,
        NaN,
        1,
        Infinity,
        NaN,
      ],
      [0, 0, 4, 4, 0, Number.MAX_VALUE, 0, Number.MIN_VALUE, 1, Infinity, 1, NaN, NaN, Infinity],
    ],
  ].map(([name, type, xValues, yValues]) => {
    const left = makeSeedMat(reference, 1, xValues.length, type, xValues);
    const right = makeSeedMat(reference, 1, yValues.length, type, yValues);
    const output = new reference.Mat();
    const audit = auditTypedMatCall(
      () => reference.magnitude(left, right, output),
      [
        ["x", left],
        ["y", right],
        ["destination", output],
      ],
    );
    safeDelete(output);
    safeDelete(right);
    safeDelete(left);
    return { name, audit };
  });

  const emptyX = new reference.Mat();
  const emptyY = new reference.Mat();
  const emptyDestination = new reference.Mat();
  const empty = auditTypedMatCall(
    () => reference.magnitude(emptyX, emptyY, emptyDestination),
    [
      ["x", emptyX],
      ["y", emptyY],
      ["destination", emptyDestination],
    ],
  );
  safeDelete(emptyDestination);
  safeDelete(emptyY);
  safeDelete(emptyX);

  const deletedX = makeSeedMat(reference, 1, 1, reference.CV_32FC1, [1]);
  const liveY = makeSeedMat(reference, 1, 1, reference.CV_32FC1, [2]);
  const liveOutput = new reference.Mat();
  deletedX.delete();
  const deletedXAudit = auditTypedMatCall(
    () => reference.magnitude(deletedX, liveY, liveOutput),
    [
      ["x", deletedX],
      ["y", liveY],
      ["destination", liveOutput],
    ],
  );
  safeDelete(liveOutput);
  safeDelete(liveY);
  const liveX = makeSeedMat(reference, 1, 1, reference.CV_32FC1, [1]);
  const deletedY = makeSeedMat(reference, 1, 1, reference.CV_32FC1, [2]);
  const secondOutput = new reference.Mat();
  deletedY.delete();
  const deletedYAudit = auditTypedMatCall(
    () => reference.magnitude(liveX, deletedY, secondOutput),
    [
      ["x", liveX],
      ["y", deletedY],
      ["destination", secondOutput],
    ],
  );
  safeDelete(secondOutput);
  safeDelete(liveX);
  const thirdX = makeSeedMat(reference, 1, 1, reference.CV_32FC1, [1]);
  const thirdY = makeSeedMat(reference, 1, 1, reference.CV_32FC1, [2]);
  const deletedOutput = new reference.Mat();
  deletedOutput.delete();
  const deletedDestinationAudit = auditTypedMatCall(
    () => reference.magnitude(thirdX, thirdY, deletedOutput),
    [
      ["x", thirdX],
      ["y", thirdY],
      ["destination", deletedOutput],
    ],
  );
  safeDelete(thirdY);
  safeDelete(thirdX);

  return {
    arity,
    argumentTypes,
    destinationReplacement,
    aliases: [
      auditReferenceMagnitudeRoi(reference, "exact-x", "x", [1, 1, 3, 2], [1, 1, 3, 2]),
      auditReferenceMagnitudeRoi(reference, "exact-y", "y", [1, 1, 3, 2], [1, 1, 3, 2]),
      auditReferenceMagnitudeRoi(reference, "x-overlap-right", "x", [0, 1, 3, 2], [1, 1, 3, 2]),
      auditReferenceMagnitudeRoi(reference, "x-overlap-down", "x", [1, 0, 3, 2], [1, 1, 3, 2]),
      auditReferenceMagnitudeRoi(reference, "y-overlap-right", "y", [0, 1, 3, 2], [1, 1, 3, 2]),
      auditReferenceMagnitudeRoi(reference, "y-overlap-down", "y", [1, 0, 3, 2], [1, 1, 3, 2]),
    ],
    empty,
    deleted: { x: deletedXAudit, y: deletedYAudit, destination: deletedDestinationAudit },
    validTypes,
    unsupportedIntegerTypes,
    mismatches,
    numericEdges,
  };
}

function auditFloatMath(reference) {
  return {
    unary: Object.fromEntries(
      FLOAT_UNARY_METHODS.map((method) => [method, auditReferenceUnary(reference, method)]),
    ),
    pow: auditReferencePow(reference),
    magnitude: auditReferenceMagnitude(reference),
    halfFloatConstants: {
      CV_16F: encodeValue(reference.CV_16F),
      CV_16FC1: encodeValue(reference.CV_16FC1),
    },
  };
}

function auditCoordinateConversions(reference) {
  const makeF32 = (values) => makeSeedMat(reference, 1, values.length, reference.CV_32FC1, values);
  const captureCart = (arguments_) => captureCall(() => reference.cartToPolar(...arguments_));
  const capturePolar = (arguments_) => captureCall(() => reference.polarToCart(...arguments_));
  const x = makeF32([1, 0, -1, 0]);
  const y = makeF32([0, 1, 0, -1]);
  const magnitude = new reference.Mat();
  const angle = new reference.Mat();
  const cartArity = {
    functionLength: capturePrimitive(() => reference.cartToPolar.length),
    zero: captureCart([]),
    one: captureCart([x]),
    two: captureCart([x, y]),
    three: captureCart([x, y, magnitude]),
    four: captureCart([x, y, magnitude, angle]),
    five: captureCart([x, y, magnitude, angle, true]),
    six: captureCart([x, y, magnitude, angle, true, 1]),
    magnitude: capturePrimitive(() => summarizeTypedMat(magnitude)),
    angle: capturePrimitive(() => summarizeTypedMat(angle)),
  };
  safeDelete(angle);
  safeDelete(magnitude);
  safeDelete(y);
  safeDelete(x);

  const flagCases = [
    ["false", false],
    ["true", true],
    ["zero", 0],
    ["one", 1],
    ["empty-string", ""],
    ["string", "x"],
    ["null", null],
    ["undefined", undefined],
    ["NaN", Number.NaN],
  ].map(([name, flag]) => {
    const flagX = makeF32([0, 0]);
    const flagY = makeF32([1, -1]);
    const flagMagnitude = new reference.Mat();
    const flagAngle = new reference.Mat();
    const audit = auditTypedMatCall(
      () => reference.cartToPolar(flagX, flagY, flagMagnitude, flagAngle, flag),
      [
        ["x", flagX],
        ["y", flagY],
        ["magnitude", flagMagnitude],
        ["angle", flagAngle],
      ],
    );
    safeDelete(flagAngle);
    safeDelete(flagMagnitude);
    safeDelete(flagY);
    safeDelete(flagX);
    return { name, audit };
  });

  const polarMagnitude = makeF32([1, 1, 1, 1]);
  const polarAngle = makeF32([0, 90, 180, 270]);
  const outputX = new reference.Mat();
  const outputY = new reference.Mat();
  const polarArity = {
    functionLength: capturePrimitive(() => reference.polarToCart.length),
    zero: capturePolar([]),
    one: capturePolar([polarMagnitude]),
    two: capturePolar([polarMagnitude, polarAngle]),
    three: capturePolar([polarMagnitude, polarAngle, outputX]),
    four: capturePolar([polarMagnitude, polarAngle, outputX, outputY]),
    five: capturePolar([polarMagnitude, polarAngle, outputX, outputY, true]),
    six: capturePolar([polarMagnitude, polarAngle, outputX, outputY, true, 1]),
    x: capturePrimitive(() => summarizeTypedMat(outputX)),
    y: capturePrimitive(() => summarizeTypedMat(outputY)),
  };
  safeDelete(outputY);
  safeDelete(outputX);
  safeDelete(polarAngle);
  safeDelete(polarMagnitude);

  const replacementX = makeF32([3, 0, -4, 0]);
  const replacementY = makeF32([4, 2, 0, -5]);
  const replacementMagnitude = makeSeedMat(
    reference,
    2,
    2,
    reference.CV_64FC2,
    [99, 99, 99, 99, 99, 99, 99, 99],
  );
  const replacementAngle = makeSeedMat(reference, 2, 2, reference.CV_8UC1, [99, 99, 99, 99]);
  const replacement = auditTypedMatCall(
    () => reference.cartToPolar(replacementX, replacementY, replacementMagnitude, replacementAngle),
    [
      ["x", replacementX],
      ["y", replacementY],
      ["magnitude", replacementMagnitude],
      ["angle", replacementAngle],
    ],
  );
  safeDelete(replacementAngle);
  safeDelete(replacementMagnitude);
  safeDelete(replacementY);
  safeDelete(replacementX);

  const emptyX = new reference.Mat();
  const emptyY = new reference.Mat();
  const emptyMagnitude = makeF32([9]);
  const emptyAngle = makeF32([9]);
  const empty = auditTypedMatCall(
    () => reference.cartToPolar(emptyX, emptyY, emptyMagnitude, emptyAngle),
    [
      ["x", emptyX],
      ["y", emptyY],
      ["magnitude", emptyMagnitude],
      ["angle", emptyAngle],
    ],
  );
  safeDelete(emptyAngle);
  safeDelete(emptyMagnitude);
  safeDelete(emptyY);
  safeDelete(emptyX);

  const cartAliasParent = makeF32([2, 3, 4, 99]);
  const cartAliasX = cartAliasParent.roi(new reference.Rect(0, 0, 3, 1));
  const cartAliasY = makeF32([0, 0, 0]);
  const cartAliasMagnitude = cartAliasParent.roi(new reference.Rect(1, 0, 3, 1));
  const cartAliasAngle = new reference.Mat();
  const cartOverlap = auditTypedMatCall(
    () => reference.cartToPolar(cartAliasX, cartAliasY, cartAliasMagnitude, cartAliasAngle),
    [
      ["parent", cartAliasParent],
      ["x", cartAliasX],
      ["y", cartAliasY],
      ["magnitude", cartAliasMagnitude],
      ["angle", cartAliasAngle],
    ],
  );
  safeDelete(cartAliasAngle);
  safeDelete(cartAliasMagnitude);
  safeDelete(cartAliasY);
  safeDelete(cartAliasX);
  safeDelete(cartAliasParent);

  const polarAliasParent = makeF32([2, 3, 4, 99]);
  const polarAliasMagnitude = polarAliasParent.roi(new reference.Rect(0, 0, 3, 1));
  const polarAliasAngle = makeF32([0, 0, 0]);
  const polarAliasX = polarAliasParent.roi(new reference.Rect(1, 0, 3, 1));
  const polarAliasY = new reference.Mat();
  const polarOverlap = auditTypedMatCall(
    () => reference.polarToCart(polarAliasMagnitude, polarAliasAngle, polarAliasX, polarAliasY),
    [
      ["parent", polarAliasParent],
      ["magnitude", polarAliasMagnitude],
      ["angle", polarAliasAngle],
      ["x", polarAliasX],
      ["y", polarAliasY],
    ],
  );
  safeDelete(polarAliasY);
  safeDelete(polarAliasX);
  safeDelete(polarAliasAngle);
  safeDelete(polarAliasMagnitude);
  safeDelete(polarAliasParent);

  const sharedOutputX = makeF32([3, 0]);
  const sharedOutputY = makeF32([4, -2]);
  const sharedOutput = new reference.Mat();
  const cartSharedOutput = auditTypedMatCall(
    () => reference.cartToPolar(sharedOutputX, sharedOutputY, sharedOutput, sharedOutput),
    [
      ["x", sharedOutputX],
      ["y", sharedOutputY],
      ["output", sharedOutput],
    ],
  );
  safeDelete(sharedOutput);
  safeDelete(sharedOutputY);
  safeDelete(sharedOutputX);

  const typeCases = [
    ["CV_32FC3", reference.CV_32FC3, [1, 2, 3, 4, 5, 6]],
    ["CV_64FC2", reference.CV_64FC2, [1, 2, 3, 4]],
    ["CV_8UC1", reference.CV_8UC1, [1, 2]],
  ].map(([name, type, values]) => {
    const typeX = makeSeedMat(reference, 1, 2, type, values);
    const typeY = makeSeedMat(reference, 1, 2, type, values);
    const typeMagnitude = new reference.Mat();
    const typeAngle = new reference.Mat();
    const audit = auditTypedMatCall(
      () => reference.cartToPolar(typeX, typeY, typeMagnitude, typeAngle),
      [
        ["x", typeX],
        ["y", typeY],
        ["magnitude", typeMagnitude],
        ["angle", typeAngle],
      ],
    );
    safeDelete(typeAngle);
    safeDelete(typeMagnitude);
    safeDelete(typeY);
    safeDelete(typeX);
    return { name, audit };
  });

  const typedEmptyCases = [
    ["zero-by-three", 0, 3],
    ["three-by-zero", 3, 0],
    ["zero-by-zero", 0, 0],
  ].map(([name, rows, cols]) => {
    const typedX = new reference.Mat(rows, cols, reference.CV_32FC1);
    const typedY = new reference.Mat(rows, cols, reference.CV_32FC1);
    const typedMagnitude = makeF32([9]);
    const typedAngle = makeF32([9]);
    const audit = auditTypedMatCall(
      () => reference.cartToPolar(typedX, typedY, typedMagnitude, typedAngle),
      [
        ["x", typedX],
        ["y", typedY],
        ["magnitude", typedMagnitude],
        ["angle", typedAngle],
      ],
    );
    safeDelete(typedAngle);
    safeDelete(typedMagnitude);
    safeDelete(typedY);
    safeDelete(typedX);
    return { name, audit };
  });

  const preciseCartX = makeSeedMat(reference, 1, 4, reference.CV_64FC1, [
    1.0000000001,
    Math.PI,
    1e200,
    1e-200,
  ]);
  const preciseCartY = makeSeedMat(reference, 1, 4, reference.CV_64FC1, [
    1.0000000002,
    Math.E,
    0,
    1e-200,
  ]);
  const preciseCartMagnitude = new reference.Mat();
  const preciseCartAngle = new reference.Mat();
  const preciseCart = auditTypedMatCall(
    () => reference.cartToPolar(preciseCartX, preciseCartY, preciseCartMagnitude, preciseCartAngle),
    [
      ["x", preciseCartX],
      ["y", preciseCartY],
      ["magnitude", preciseCartMagnitude],
      ["angle", preciseCartAngle],
    ],
  );
  safeDelete(preciseCartAngle);
  safeDelete(preciseCartMagnitude);
  safeDelete(preciseCartY);
  safeDelete(preciseCartX);

  const precisePolarMagnitude = makeSeedMat(reference, 1, 4, reference.CV_64FC1, [
    1.0000000001,
    Math.PI,
    1e200,
    1e-200,
  ]);
  const precisePolarAngle = makeSeedMat(
    reference,
    1,
    4,
    reference.CV_64FC1,
    [0.1234567890123, -0.75, 7, 1e-10],
  );
  const precisePolarX = new reference.Mat();
  const precisePolarY = new reference.Mat();
  const precisePolar = auditTypedMatCall(
    () =>
      reference.polarToCart(precisePolarMagnitude, precisePolarAngle, precisePolarX, precisePolarY),
    [
      ["magnitude", precisePolarMagnitude],
      ["angle", precisePolarAngle],
      ["x", precisePolarX],
      ["y", precisePolarY],
    ],
  );
  safeDelete(precisePolarY);
  safeDelete(precisePolarX);
  safeDelete(precisePolarAngle);
  safeDelete(precisePolarMagnitude);

  const polarReplacementMagnitude = makeF32([3, 2, 4, 5]);
  const polarReplacementAngle = makeF32([0, 90, 180, 270]);
  const polarReplacementX = makeSeedMat(
    reference,
    2,
    2,
    reference.CV_64FC2,
    [99, 99, 99, 99, 99, 99, 99, 99],
  );
  const polarReplacementY = makeSeedMat(reference, 2, 2, reference.CV_8UC1, [99, 99, 99, 99]);
  const polarReplacement = auditTypedMatCall(
    () =>
      reference.polarToCart(
        polarReplacementMagnitude,
        polarReplacementAngle,
        polarReplacementX,
        polarReplacementY,
        true,
      ),
    [
      ["magnitude", polarReplacementMagnitude],
      ["angle", polarReplacementAngle],
      ["x", polarReplacementX],
      ["y", polarReplacementY],
    ],
  );
  safeDelete(polarReplacementY);
  safeDelete(polarReplacementX);
  safeDelete(polarReplacementAngle);
  safeDelete(polarReplacementMagnitude);

  const auditCartMismatch = (name, createY) => {
    const mismatchX = makeSeedMat(reference, 2, 2, reference.CV_32FC1, [1, 2, 3, 4]);
    const mismatchY = createY();
    const mismatchMagnitude = new reference.Mat();
    const mismatchAngle = new reference.Mat();
    const audit = auditTypedMatCall(
      () => reference.cartToPolar(mismatchX, mismatchY, mismatchMagnitude, mismatchAngle),
      [
        ["x", mismatchX],
        ["y", mismatchY],
        ["magnitude", mismatchMagnitude],
        ["angle", mismatchAngle],
      ],
    );
    safeDelete(mismatchAngle);
    safeDelete(mismatchMagnitude);
    safeDelete(mismatchY);
    safeDelete(mismatchX);
    return { name, audit };
  };
  const auditPolarMismatch = (name, createAngle) => {
    const mismatchMagnitude = makeSeedMat(reference, 2, 2, reference.CV_32FC1, [1, 2, 3, 4]);
    const mismatchAngle = createAngle();
    const mismatchX = new reference.Mat();
    const mismatchY = new reference.Mat();
    const audit = auditTypedMatCall(
      () => reference.polarToCart(mismatchMagnitude, mismatchAngle, mismatchX, mismatchY),
      [
        ["magnitude", mismatchMagnitude],
        ["angle", mismatchAngle],
        ["x", mismatchX],
        ["y", mismatchY],
      ],
    );
    safeDelete(mismatchY);
    safeDelete(mismatchX);
    safeDelete(mismatchAngle);
    safeDelete(mismatchMagnitude);
    return { name, audit };
  };
  const mismatches = {
    cart: [
      auditCartMismatch("shape", () =>
        makeSeedMat(reference, 1, 4, reference.CV_32FC1, [1, 2, 3, 4]),
      ),
      auditCartMismatch("depth", () =>
        makeSeedMat(reference, 2, 2, reference.CV_64FC1, [1, 2, 3, 4]),
      ),
      auditCartMismatch("channels", () =>
        makeSeedMat(reference, 2, 2, reference.CV_32FC2, [1, 2, 3, 4, 5, 6, 7, 8]),
      ),
    ],
    polar: [
      auditPolarMismatch("shape", () =>
        makeSeedMat(reference, 1, 4, reference.CV_32FC1, [0, 1, 2, 3]),
      ),
      auditPolarMismatch("depth", () =>
        makeSeedMat(reference, 2, 2, reference.CV_64FC1, [0, 1, 2, 3]),
      ),
      auditPolarMismatch("channels", () =>
        makeSeedMat(reference, 2, 2, reference.CV_32FC2, [0, 1, 2, 3, 4, 5, 6, 7]),
      ),
    ],
  };

  const deletedCartX = makeF32([1, 2]);
  const deletedCartY = makeF32([3, 4]);
  const deletedCartMagnitude = new reference.Mat();
  const deletedCartAngle = new reference.Mat();
  deletedCartX.delete();
  const deletedCartFirstInput = auditTypedMatCall(
    () => reference.cartToPolar(deletedCartX, deletedCartY, deletedCartMagnitude, deletedCartAngle),
    [
      ["x", deletedCartX],
      ["y", deletedCartY],
      ["magnitude", deletedCartMagnitude],
      ["angle", deletedCartAngle],
    ],
  );
  safeDelete(deletedCartAngle);
  safeDelete(deletedCartMagnitude);
  safeDelete(deletedCartY);

  const deletedCartOutputX = makeF32([1, 2]);
  const deletedCartOutputY = makeF32([3, 4]);
  const deletedCartOutputMagnitude = new reference.Mat();
  const deletedCartOutputAngle = new reference.Mat();
  deletedCartOutputMagnitude.delete();
  const deletedCartOutput = auditTypedMatCall(
    () =>
      reference.cartToPolar(
        deletedCartOutputX,
        deletedCartOutputY,
        deletedCartOutputMagnitude,
        deletedCartOutputAngle,
      ),
    [
      ["x", deletedCartOutputX],
      ["y", deletedCartOutputY],
      ["magnitude", deletedCartOutputMagnitude],
      ["angle", deletedCartOutputAngle],
    ],
  );
  safeDelete(deletedCartOutputAngle);
  safeDelete(deletedCartOutputY);
  safeDelete(deletedCartOutputX);

  const deletedPolarMagnitude = makeF32([1, 2]);
  const deletedPolarAngle = makeF32([0, 1]);
  const deletedPolarX = new reference.Mat();
  const deletedPolarY = new reference.Mat();
  deletedPolarMagnitude.delete();
  const deletedPolarFirstInput = auditTypedMatCall(
    () =>
      reference.polarToCart(deletedPolarMagnitude, deletedPolarAngle, deletedPolarX, deletedPolarY),
    [
      ["magnitude", deletedPolarMagnitude],
      ["angle", deletedPolarAngle],
      ["x", deletedPolarX],
      ["y", deletedPolarY],
    ],
  );
  safeDelete(deletedPolarY);
  safeDelete(deletedPolarX);
  safeDelete(deletedPolarAngle);

  const deletedPolarOutputMagnitude = makeF32([1, 2]);
  const deletedPolarOutputAngle = makeF32([0, 1]);
  const deletedPolarOutputX = new reference.Mat();
  const deletedPolarOutputY = new reference.Mat();
  deletedPolarOutputX.delete();
  const deletedPolarOutput = auditTypedMatCall(
    () =>
      reference.polarToCart(
        deletedPolarOutputMagnitude,
        deletedPolarOutputAngle,
        deletedPolarOutputX,
        deletedPolarOutputY,
      ),
    [
      ["magnitude", deletedPolarOutputMagnitude],
      ["angle", deletedPolarOutputAngle],
      ["x", deletedPolarOutputX],
      ["y", deletedPolarOutputY],
    ],
  );
  safeDelete(deletedPolarOutputY);
  safeDelete(deletedPolarOutputAngle);
  safeDelete(deletedPolarOutputMagnitude);

  const deleted = {
    cart: { firstInput: deletedCartFirstInput, output: deletedCartOutput },
    polar: { firstInput: deletedPolarFirstInput, output: deletedPolarOutput },
  };

  const cartSecondOutputParent = makeF32([0, 1, 0, -1, 99]);
  const cartSecondOutputX = makeF32([1, 0, -1, 0]);
  const cartSecondOutputY = cartSecondOutputParent.roi(new reference.Rect(0, 0, 4, 1));
  const cartSecondOutputMagnitude = new reference.Mat();
  const cartSecondOutputAngle = cartSecondOutputParent.roi(new reference.Rect(1, 0, 4, 1));
  const cartSecondOutputOverlap = auditTypedMatCall(
    () =>
      reference.cartToPolar(
        cartSecondOutputX,
        cartSecondOutputY,
        cartSecondOutputMagnitude,
        cartSecondOutputAngle,
      ),
    [
      ["parent", cartSecondOutputParent],
      ["x", cartSecondOutputX],
      ["y", cartSecondOutputY],
      ["magnitude", cartSecondOutputMagnitude],
      ["angle", cartSecondOutputAngle],
    ],
  );
  safeDelete(cartSecondOutputAngle);
  safeDelete(cartSecondOutputMagnitude);
  safeDelete(cartSecondOutputY);
  safeDelete(cartSecondOutputX);
  safeDelete(cartSecondOutputParent);

  const polarSecondOutputParent = makeF32([0, 90, 180, 270, 99]);
  const polarSecondOutputMagnitude = makeF32([1, 1, 1, 1]);
  const polarSecondOutputAngle = polarSecondOutputParent.roi(new reference.Rect(0, 0, 4, 1));
  const polarSecondOutputX = new reference.Mat();
  const polarSecondOutputY = polarSecondOutputParent.roi(new reference.Rect(1, 0, 4, 1));
  const polarSecondOutputOverlap = auditTypedMatCall(
    () =>
      reference.polarToCart(
        polarSecondOutputMagnitude,
        polarSecondOutputAngle,
        polarSecondOutputX,
        polarSecondOutputY,
      ),
    [
      ["parent", polarSecondOutputParent],
      ["magnitude", polarSecondOutputMagnitude],
      ["angle", polarSecondOutputAngle],
      ["x", polarSecondOutputX],
      ["y", polarSecondOutputY],
    ],
  );
  safeDelete(polarSecondOutputY);
  safeDelete(polarSecondOutputX);
  safeDelete(polarSecondOutputAngle);
  safeDelete(polarSecondOutputMagnitude);
  safeDelete(polarSecondOutputParent);

  return {
    cartArity,
    polarArity,
    flagCases,
    replacement,
    polarReplacement,
    empty,
    aliases: {
      cartOverlap,
      polarOverlap,
      cartSharedOutput,
      cartSecondOutputOverlap,
      polarSecondOutputOverlap,
    },
    mismatches,
    deleted,
    typeCases,
    typedEmptyCases,
    preciseCart,
    precisePolar,
  };
}

function encodeValue(value) {
  if (value === undefined) {
    return { type: "undefined" };
  }
  if (value === null) {
    return { type: "null" };
  }
  if (typeof value === "number") {
    if (Number.isNaN(value)) {
      return { type: "nan" };
    }
    if (value === Number.POSITIVE_INFINITY) {
      return { type: "positive-infinity" };
    }
    if (value === Number.NEGATIVE_INFINITY) {
      return { type: "negative-infinity" };
    }
    if (Object.is(value, -0)) {
      return { type: "negative-zero" };
    }
    return { type: "number", value };
  }
  return { type: typeof value, value };
}

function captureCall(callback) {
  try {
    return { threw: false, returnValue: encodeValue(callback()) };
  } catch (error) {
    return {
      threw: true,
      error: {
        name: error?.name,
        constructor: error?.constructor?.name,
        message: error?.message,
        text: String(error),
        instanceofError: error instanceof Error,
      },
    };
  }
}

function captureEnumCall(callback, namespace, constants) {
  try {
    return {
      threw: false,
      returnValue: describeEnumValue(callback(), namespace, constants),
    };
  } catch (error) {
    return {
      threw: true,
      error: {
        name: error?.name,
        constructor: error?.constructor?.name,
        message: error?.message,
        text: String(error),
        instanceofError: error instanceof Error,
      },
    };
  }
}

function capturePrimitive(callback) {
  try {
    return { threw: false, value: encodeValue(callback()) };
  } catch (error) {
    return {
      threw: true,
      error: {
        name: error?.name,
        constructor: error?.constructor?.name,
        message: error?.message,
        text: String(error),
        instanceofError: error instanceof Error,
      },
    };
  }
}

function encodeFloat64(value) {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value, false);
  return {
    encoded: encodeValue(value),
    bits: Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(""),
  };
}

function summarizeRotationMatrix(matrix) {
  return {
    rows: matrix.rows,
    cols: matrix.cols,
    channels: matrix.channels(),
    depth: matrix.depth(),
    type: matrix.type(),
    total: matrix.total(),
    continuous: matrix.isContinuous(),
    values: Array.from(matrix.data64F, encodeFloat64),
  };
}

function captureRotationMatrix(callback) {
  let matrix;
  try {
    matrix = callback();
    return { threw: false, matrix: summarizeRotationMatrix(matrix) };
  } catch (error) {
    return {
      threw: true,
      error: {
        name: error?.name,
        constructor: error?.constructor?.name,
        message: error?.message,
        text: String(error),
        instanceofError: error instanceof Error,
      },
    };
  } finally {
    safeDelete(matrix);
  }
}

function captureRotationRejection(callback) {
  let matrix;
  try {
    matrix = callback();
    return { threw: false };
  } catch {
    return { threw: true };
  } finally {
    safeDelete(matrix);
  }
}

function auditRotationFieldReads(reference, center, angle = 15, scale = 0.75) {
  const reads = [];
  const proxy = new Proxy(center, {
    has(target, property) {
      reads.push(`has:${String(property)}`);
      return Reflect.has(target, property);
    },
    get(target, property) {
      reads.push(`get:${String(property)}`);
      return target[property];
    },
  });
  return {
    call: captureRotationMatrix(() => reference.getRotationMatrix2D(proxy, angle, scale)),
    reads,
  };
}

function auditRotationRejectedFields(reference, center, angle = 15, scale = 0.75) {
  const reads = [];
  const proxy =
    center !== null && (typeof center === "object" || typeof center === "function")
      ? new Proxy(center, {
          has(target, property) {
            reads.push(`has:${String(property)}`);
            return Reflect.has(target, property);
          },
          get(target, property) {
            reads.push(`get:${String(property)}`);
            return target[property];
          },
        })
      : center;
  return {
    call: captureRotationRejection(() => reference.getRotationMatrix2D(proxy, angle, scale)),
    reads,
  };
}

function auditRotationMatrix(reference) {
  const inherited = Object.create({ x: 1.25, y: -2.5 });
  const nullPrototype = Object.assign(Object.create(null), { x: 1.25, y: -2.5 });
  const arrayWithFields = [];
  arrayWithFields.x = 1.25;
  arrayWithFields.y = -2.5;
  const functionWithFields = () => reference;
  functionWithFields.x = 1.25;
  functionWithFields.y = -2.5;

  const arityCenterReads = [];
  const arityCenter = new Proxy(
    { x: 1, y: 2 },
    {
      has(target, property) {
        arityCenterReads.push(`has:${String(property)}`);
        return Reflect.has(target, property);
      },
      get(target, property) {
        arityCenterReads.push(`get:${String(property)}`);
        return target[property];
      },
    },
  );
  const arity = {
    length: reference.getRotationMatrix2D.length,
    zero: captureCall(() => reference.getRotationMatrix2D()),
    one: captureCall(() => reference.getRotationMatrix2D(arityCenter)),
    two: captureCall(() => reference.getRotationMatrix2D(arityCenter, 30)),
    exact: captureRotationMatrix(() => reference.getRotationMatrix2D(arityCenter, 30, 1)),
    four: captureCall(() => reference.getRotationMatrix2D(arityCenter, 30, 1, 0)),
    centerReads: arityCenterReads,
  };

  const invalidX = {
    get x() {
      return "1";
    },
    get y() {
      throw new Error("y must not be read after invalid x");
    },
  };
  const scalarRejections = [
    ["string", "30"],
    ["null", null],
    ["undefined", undefined],
    ["boxed", new Number(30)],
    ["object", { valueOf: () => 30 }],
    ["bigint", 30n],
    ["symbol", Symbol("30")],
  ];
  const call = (center, angle, scale) =>
    captureRotationMatrix(() => reference.getRotationMatrix2D(center, angle, scale));

  const first = reference.getRotationMatrix2D({ x: 1, y: 2 }, 90, 1);
  const second = reference.getRotationMatrix2D({ x: 1, y: 2 }, 90, 1);
  const ownership = {
    distinctWrappers: first !== second,
    firstBefore: summarizeRotationMatrix(first),
    secondBefore: summarizeRotationMatrix(second),
  };
  first.data64F[0] = 99;
  ownership.firstAfterMutation = summarizeRotationMatrix(first);
  ownership.secondAfterFirstMutation = summarizeRotationMatrix(second);
  ownership.firstRelease = captureCall(() => first.delete());
  ownership.firstAfterRelease = captureRotationRejection(() => summarizeRotationMatrix(first));
  ownership.secondAfterFirstRelease = summarizeRotationMatrix(second);
  ownership.secondRelease = captureCall(() => second.delete());

  return {
    arity,
    structural: {
      plain: call({ x: 1.25, y: -2.5 }, 33.3, 0.75),
      inherited: call(inherited, 33.3, 0.75),
      nullPrototype: call(nullPrototype, 33.3, 0.75),
      arrayWithFields: call(arrayWithFields, 33.3, 0.75),
      functionWithFields: call(functionWithFields, 33.3, 0.75),
      fieldOrder: auditRotationFieldReads(reference, { x: 1.25, y: -2.5 }),
    },
    rejectedStructural: {
      missingX: auditRotationRejectedFields(reference, { y: 2 }),
      missingY: auditRotationRejectedFields(reference, { x: 1 }),
      bareArray: auditRotationRejectedFields(reference, [1, 2]),
      null: auditRotationRejectedFields(reference, null),
      undefined: auditRotationRejectedFields(reference, undefined),
      number: auditRotationRejectedFields(reference, 1),
      invalidX: auditRotationRejectedFields(reference, invalidX),
    },
    conversions: {
      centerFloat32Narrowing: call({ x: 1.00000001, y: 16_777_217 }, 30, 2),
      centerBooleans: call({ x: true, y: false }, 30, 2),
      angleTrue: call({ x: 1, y: 2 }, true, 1),
      angleFalse: call({ x: 1, y: 2 }, false, 1),
      scaleTrue: call({ x: 1, y: 2 }, 30, true),
      scaleFalse: call({ x: 1, y: 2 }, 30, false),
      angleRejected: scalarRejections.map(([label, value]) => ({
        label,
        call: captureRotationRejection(() =>
          reference.getRotationMatrix2D({ x: 1, y: 2 }, value, 1),
        ),
      })),
      scaleRejected: scalarRejections.map(([label, value]) => ({
        label,
        call: captureRotationRejection(() =>
          reference.getRotationMatrix2D({ x: 1, y: 2 }, 30, value),
        ),
      })),
    },
    fixtures: [
      { label: "quarter turn", call: call({ x: 1, y: 2 }, 90, 1) },
      { label: "negative quarter turn", call: call({ x: 1, y: 2 }, -90, 1) },
      { label: "fractional", call: call({ x: 1.25, y: -2.5 }, 33.3, 0.75) },
    ],
    signedZero: [
      { label: "negative center", call: call({ x: -0, y: -0 }, 0, 1) },
      { label: "negative angle", call: call({ x: 1, y: 2 }, -0, 1) },
      { label: "positive zero scale", call: call({ x: 1, y: 2 }, 30, 0) },
      { label: "negative zero scale", call: call({ x: 1, y: 2 }, 30, -0) },
      { label: "all negative zero", call: call({ x: -0, y: -0 }, -0, -0) },
    ],
    nonFinite: [
      { label: "NaN angle", call: call({ x: 1, y: 2 }, Number.NaN, 1) },
      { label: "positive infinite angle", call: call({ x: 1, y: 2 }, Infinity, 1) },
      { label: "negative infinite angle", call: call({ x: 1, y: 2 }, -Infinity, 1) },
      { label: "NaN scale", call: call({ x: 1, y: 2 }, 30, Number.NaN) },
      { label: "positive infinite scale", call: call({ x: 1, y: 2 }, 30, Infinity) },
      { label: "negative infinite scale", call: call({ x: 1, y: 2 }, 30, -Infinity) },
      { label: "NaN center x", call: call({ x: Number.NaN, y: 2 }, 30, 1) },
    ],
    ownership,
  };
}

function propertyDescriptorSummary(owner, property) {
  const descriptor = Object.getOwnPropertyDescriptor(owner, property);
  if (descriptor === undefined) return { present: false };
  return {
    present: true,
    configurable: descriptor.configurable,
    enumerable: descriptor.enumerable,
    writable: descriptor.writable,
    hasGetter: descriptor.get !== undefined,
    hasSetter: descriptor.set !== undefined,
    valueType: descriptor.value === null ? "null" : typeof descriptor.value,
    functionName: typeof descriptor.value === "function" ? descriptor.value.name : undefined,
  };
}

function describeEnumValue(value, namespace, constants) {
  const isObject = (typeof value === "object" && value !== null) || typeof value === "function";
  return {
    encoded: encodeValue(isObject ? undefined : value),
    type: typeof value,
    isNull: value === null,
    value: isObject ? capturePrimitive(() => value.value) : undefined,
    objectKeys: isObject ? Object.keys(value).toSorted() : [],
    ownNames: isObject ? Object.getOwnPropertyNames(value).toSorted() : [],
    ownSymbolDescriptions: isObject
      ? Object.getOwnPropertySymbols(value)
          .map((symbol) => symbol.description ?? "")
          .toSorted()
      : [],
    valueDescriptor: isObject ? propertyDescriptorSummary(value, "value") : undefined,
    constructorDescriptor: isObject ? propertyDescriptorSummary(value, "constructor") : undefined,
    prototypeIsNamespacePrototype: isObject
      ? Object.getPrototypeOf(value) === namespace.prototype
      : false,
    instanceOfNamespace: isObject ? value instanceof namespace : false,
    constructorName: isObject ? capturePrimitive(() => value.constructor?.name) : undefined,
    objectTag: isObject ? capturePrimitive(() => Object.prototype.toString.call(value)) : undefined,
    string: capturePrimitive(() => String(value)),
    number: capturePrimitive(() => Number(value)),
    json: capturePrimitive(() => JSON.stringify(value)),
    extensible: isObject ? Object.isExtensible(value) : false,
    sealed: isObject ? Object.isSealed(value) : false,
    frozen: isObject ? Object.isFrozen(value) : false,
    matchingConstants: constants.filter(([, constant]) => value === constant).map(([name]) => name),
  };
}

function summarizeEnumNamespace(name, namespace, constantNames) {
  const constants = constantNames.map((constantName) => [constantName, namespace[constantName]]);
  const values = namespace.values;
  return {
    name,
    type: typeof namespace,
    ownKeys: Object.keys(namespace).toSorted(),
    ownNames: Object.getOwnPropertyNames(namespace).toSorted(),
    prototypeOwnNames: Object.getOwnPropertyNames(namespace.prototype).toSorted(),
    descriptors: Object.fromEntries(
      constantNames.map((constantName) => [
        constantName,
        propertyDescriptorSummary(namespace, constantName),
      ]),
    ),
    constants: Object.fromEntries(
      constants.map(([constantName, value]) => [
        constantName,
        describeEnumValue(value, namespace, constants),
      ]),
    ),
    values: {
      present: values !== undefined,
      type: typeof values,
      keys: values === undefined ? [] : Object.keys(values).toSorted(),
      identities:
        values === undefined
          ? []
          : Object.keys(values)
              .toSorted()
              .map((key) => ({
                key,
                matchingConstants: constants
                  .filter(([, constant]) => values[key] === constant)
                  .map(([constantName]) => constantName),
              })),
    },
  };
}

function auditDetectorEnumPair(configuration, allNamespaces) {
  const { constants, createDetector, getter, label, namespace, setter } = configuration;
  const captureState = (detector) =>
    captureEnumCall(() => detector[getter](), namespace, constants);
  const defaultDetector = createDetector();
  const firstDefault = defaultDetector[getter]();
  const secondDefault = defaultDetector[getter]();
  const defaults = {
    first: describeEnumValue(firstDefault, namespace, constants),
    second: describeEnumValue(secondDefault, namespace, constants),
    repeatedIdentity: firstDefault === secondDefault,
  };
  safeDelete(defaultDetector);

  const defaultConstant = constants[0][1];
  const auditGetterArity = (...arguments_) => {
    const detector = createDetector();
    const stateBefore = captureState(detector);
    const call = captureEnumCall(() => detector[getter](...arguments_), namespace, constants);
    const stateAfter = captureState(detector);
    safeDelete(detector);
    return { stateBefore, call, stateAfter };
  };
  const auditSetterArity = (...arguments_) => {
    const detector = createDetector();
    const stateBefore = captureState(detector);
    const call = captureCall(() => detector[setter](...arguments_));
    const stateAfter = captureState(detector);
    safeDelete(detector);
    return { stateBefore, call, stateAfter };
  };
  const lengthDetector = createDetector();
  const arity = {
    getterLength: lengthDetector[getter].length,
    getterExact: auditGetterArity(),
    getterExtraOne: auditGetterArity(1),
    getterExtraTwo: auditGetterArity(1, 2),
    setterLength: lengthDetector[setter].length,
    setterMissing: auditSetterArity(),
    setterExact: auditSetterArity(defaultConstant),
    setterExtraOne: auditSetterArity(defaultConstant, 1),
    setterExtraTwo: auditSetterArity(defaultConstant, 1, 2),
  };
  safeDelete(lengthDetector);

  const foreignConstants = allNamespaces.map(({ constants: values, name }) => ({
    create: () => values[0][1],
    label: `foreign ${name}.${values[0][0]}`,
  }));
  const inputCases = [
    ...constants.map(([name, value]) => ({ create: () => value, label: `canonical ${name}` })),
    ...foreignConstants,
    { create: () => ({ value: constants[0][1].value }), label: "plain valid value" },
    { create: () => ({ value: -1 }), label: "plain negative one" },
    { create: () => ({ value: 1.9 }), label: "plain fraction" },
    { create: () => ({ value: Number.NaN }), label: "plain NaN" },
    { create: () => ({ value: Number.POSITIVE_INFINITY }), label: "plain infinity" },
    { create: () => ({ value: 4_294_967_296 }), label: "plain u32 modulus" },
    { create: () => ({ value: "2" }), label: "plain numeric string value" },
    { create: () => ({ value: "opencv" }), label: "plain nonnumeric string value" },
    { create: () => ({ value: true }), label: "plain boolean value" },
    { create: () => ({ value: null }), label: "plain null value" },
    { create: () => ({ value: undefined }), label: "plain undefined value" },
    { create: () => Object.create({ value: 1 }), label: "inherited value" },
    {
      create: () => Object.assign(Object.create(null), { value: 1 }),
      label: "null prototype value",
    },
    { create: () => ({}), label: "empty object" },
    { create: () => 2, label: "number primitive" },
    { create: () => "2", label: "string primitive" },
    { create: () => true, label: "boolean primitive" },
    { create: () => null, label: "null primitive" },
    { create: () => undefined, label: "undefined primitive" },
    {
      create: () => {
        let reads = 0;
        return {
          get reads() {
            return reads;
          },
          get value() {
            reads += 1;
            throw new Error("enum value getter sentinel");
          },
        };
      },
      label: "throwing value getter",
    },
  ];

  const setterInputs = inputCases.map(({ create, label: inputLabel }) => {
    const detector = createDetector();
    const input = create();
    const inputSummary = {
      type: typeof input,
      isNull: input === null,
      ownNames:
        (typeof input === "object" && input !== null) || typeof input === "function"
          ? Object.getOwnPropertyNames(input).toSorted()
          : [],
      value:
        (typeof input === "object" && input !== null) || typeof input === "function"
          ? capturePrimitive(() => input.value)
          : undefined,
      matches: allNamespaces.flatMap(({ constants: values, name }) =>
        values
          .filter(([, constant]) => constant === input)
          .map(([constantName]) => `${name}.${constantName}`),
      ),
      readsBefore:
        typeof input === "object" && input !== null && "reads" in input
          ? capturePrimitive(() => input.reads)
          : undefined,
    };
    const call = captureCall(() => detector[setter](input));
    const state = captureState(detector);
    const readsAfter =
      typeof input === "object" && input !== null && "reads" in input
        ? capturePrimitive(() => input.reads)
        : undefined;
    safeDelete(detector);
    return { label: inputLabel, input: inputSummary, call, state, readsAfter };
  });

  const deadDetector = createDetector();
  const firstDelete = captureCall(() => deadDetector.delete());
  const lifetime = {
    firstDelete,
    getterAfterDelete: captureEnumCall(() => deadDetector[getter](), namespace, constants),
    setterAfterDelete: captureCall(() => deadDetector[setter](defaultConstant)),
    secondDelete: captureCall(() => deadDetector.delete()),
  };

  return { label, defaults, arity, setterInputs, lifetime };
}

function auditDetectorEnums(reference) {
  const namespaces = [
    {
      name: "AKAZE_DescriptorType",
      namespace: reference.AKAZE_DescriptorType,
      names: [
        "DESCRIPTOR_KAZE_UPRIGHT",
        "DESCRIPTOR_KAZE",
        "DESCRIPTOR_MLDB_UPRIGHT",
        "DESCRIPTOR_MLDB",
      ],
    },
    {
      name: "KAZE_DiffusivityType",
      namespace: reference.KAZE_DiffusivityType,
      names: ["DIFF_PM_G1", "DIFF_PM_G2", "DIFF_WEICKERT", "DIFF_CHARBONNIER"],
    },
    {
      name: "AgastFeatureDetector_DetectorType",
      namespace: reference.AgastFeatureDetector_DetectorType,
      names: ["AGAST_5_8", "AGAST_7_12d", "AGAST_7_12s", "OAST_9_16"],
    },
    {
      name: "FastFeatureDetector_DetectorType",
      namespace: reference.FastFeatureDetector_DetectorType,
      names: ["TYPE_5_8", "TYPE_7_12", "TYPE_9_16"],
    },
  ].map((entry) => ({
    constants: entry.names.map((name) => [name, entry.namespace[name]]),
    name: entry.name,
    names: entry.names,
    namespace: entry.namespace,
  }));
  const byName = Object.fromEntries(namespaces.map((entry) => [entry.name, entry]));
  const configurations = [
    [
      "AKAZE descriptor type",
      () => new reference.AKAZE(),
      "getDescriptorType",
      "setDescriptorType",
      "AKAZE_DescriptorType",
    ],
    [
      "AKAZE diffusivity",
      () => new reference.AKAZE(),
      "getDiffusivity",
      "setDiffusivity",
      "KAZE_DiffusivityType",
    ],
    [
      "KAZE diffusivity",
      () => new reference.KAZE(),
      "getDiffusivity",
      "setDiffusivity",
      "KAZE_DiffusivityType",
    ],
    [
      "AGAST detector type",
      () => new reference.AgastFeatureDetector(),
      "getType",
      "setType",
      "AgastFeatureDetector_DetectorType",
    ],
    [
      "FAST detector type",
      () => new reference.FastFeatureDetector(),
      "getType",
      "setType",
      "FastFeatureDetector_DetectorType",
    ],
  ];
  const pairs = configurations.map(([label, createDetector, getter, setter, namespaceName]) => {
    const entry = byName[namespaceName];
    return auditDetectorEnumPair(
      {
        label,
        createDetector,
        getter,
        setter,
        namespace: entry.namespace,
        constants: entry.constants,
      },
      namespaces,
    );
  });
  const akaze = new reference.AKAZE();
  const kaze = new reference.KAZE();
  const crossIdentity = {
    akazeKazeDiffusivityDefaultSame: akaze.getDiffusivity() === kaze.getDiffusivity(),
    zeroConstantsAcrossNamespaces: namespaces.map((left) =>
      namespaces.map((right) => left.constants[0][1] === right.constants[0][1]),
    ),
  };
  akaze.delete();
  kaze.delete();
  return {
    namespaces: Object.fromEntries(
      namespaces.map(({ name, namespace, names }) => [
        name,
        summarizeEnumNamespace(name, namespace, names),
      ]),
    ),
    pairs,
    crossIdentity,
  };
}

function auditOptimalDftSize(owner) {
  const i32Cases = [
    ["positive fraction", 1.9],
    ["negative fraction", -1.9],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["i32 maximum", 2_147_483_647],
    ["i32 maximum plus one", 2_147_483_648],
    ["i32 minimum", -2_147_483_648],
    ["i32 minimum minus one", -2_147_483_649],
    ["u32 maximum", 4_294_967_295],
    ["u32 modulus", 4_294_967_296],
    ["null", null],
    ["true", true],
    ["false", false],
    ["numeric string", "42"],
    ["fraction string", "-1.9"],
    ["empty string", ""],
    ["non-numeric string", "opencv"],
    ["explicit undefined", undefined],
  ];
  const boundaryCases = [
    ["negative 1024", -1024],
    ["negative two", -2],
    ["negative one", -1],
    ["zero", 0],
    ["one", 1],
    ["two", 2],
    ["three", 3],
    ["seven", 7],
    ["twenty five", 25],
    ["last rounded input", 2_125_763_999],
    ["first sentinel input", 2_125_764_000],
    ["sentinel plus one", 2_125_764_001],
    ["i32 maximum minus one", 2_147_483_646],
  ];
  const auditCases = (cases) =>
    cases.map(([label, value]) => ({
      label,
      input: encodeValue(value),
      call: captureCall(() => owner.getOptimalDFTSize(value)),
    }));

  return {
    length: owner.getOptimalDFTSize.length,
    arity: {
      missing: captureCall(() => owner.getOptimalDFTSize()),
      exact: captureCall(() => owner.getOptimalDFTSize(7)),
      extraOne: captureCall(() => owner.getOptimalDFTSize(7, 1)),
      extraTwo: captureCall(() => owner.getOptimalDFTSize(7, 1, 2)),
    },
    i32: auditCases(i32Cases),
    boundaries: auditCases(boundaryCases),
  };
}

function safeDelete(detector) {
  try {
    detector.delete();
  } catch {
    // The audit records delete failures at their call site.
  }
}

function auditNumericCall(reference, operation, args) {
  const left = makeSeedMat(reference, 1, 3, reference.CV_32FC1, [2, -4, 8]);
  const right = makeSeedMat(reference, 1, 3, reference.CV_32FC1, [4, 2, 0]);
  const destination = new reference.Mat();
  const values = { left, right, destination };
  const resolvedArgs = args.map((argument) =>
    typeof argument === "string" ? values[argument] : argument,
  );
  const call = captureCall(() => reference[operation](...resolvedArgs));
  const output = capturePrimitive(() => summarizeTypedMat(destination));
  safeDelete(destination);
  safeDelete(right);
  safeDelete(left);
  return { call, output };
}

function auditNumericContracts(reference) {
  const calls = {
    multiply: [
      [],
      ["left"],
      ["left", "right"],
      ["left", "right", "destination"],
      ["left", "right", "destination", 0.5],
      ["left", "right", "destination", 0.5, reference.CV_64F],
      ["left", "right", "destination", 0.5, reference.CV_64F, 1],
    ],
    divide: [
      [],
      ["left"],
      ["left", "right"],
      ["left", "right", "destination"],
      ["left", "right", "destination", 2],
      ["left", "right", "destination", 2, reference.CV_64F],
      [2, "right", "destination"],
      [2, "right", "destination", reference.CV_64F],
      [2, "right", "destination", reference.CV_64F, 1],
    ],
    addWeighted: [
      [],
      ["left"],
      ["left", 0.5, "right", 0.25, 1],
      ["left", 0.5, "right", 0.25, 1, "destination"],
      ["left", 0.5, "right", 0.25, 1, "destination", reference.CV_64F],
      ["left", 0.5, "right", 0.25, 1, "destination", reference.CV_64F, 1],
    ],
    convertScaleAbs: [
      [],
      ["left"],
      ["left", "destination"],
      ["left", "destination", 2],
      ["left", "destination", 2, 1],
      ["left", "destination", 2, 1, 0],
    ],
  };
  const contracts = Object.fromEntries(
    Object.entries(calls).map(([operation, argumentLists]) => [
      operation,
      {
        functionLength: capturePrimitive(() => reference[operation].length),
        calls: argumentLists.map((args) => ({
          args: args.map((argument) =>
            typeof argument === "string" ? argument : encodeValue(argument),
          ),
          ...auditNumericCall(reference, operation, args),
        })),
      },
    ]),
  );

  const dtypeInputs = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 14, -2, 255, Number.NaN, Infinity];
  const dtype = dtypeInputs.map((value) => {
    const audit = auditNumericCall(reference, "multiply", [
      "left",
      "right",
      "destination",
      0.5,
      value,
    ]);
    return { value: encodeValue(value), call: audit.call, output: audit.output };
  });

  const scaleInputs = [undefined, null, true, false, "2", Number.NaN, Infinity, -Infinity];
  const scale = scaleInputs.map((value) => {
    const audit = auditNumericCall(reference, "multiply", ["left", "right", "destination", value]);
    return { value: encodeValue(value), call: audit.call, output: audit.output };
  });

  const replacement = ["multiply", "divide", "addWeighted", "convertScaleAbs"].map((operation) => {
    const left = makeSeedMat(reference, 1, 3, reference.CV_32FC1, [2, -4, 8]);
    const right = makeSeedMat(reference, 1, 3, reference.CV_32FC1, [4, 2, 1]);
    const destination = makeSeedMat(reference, 2, 1, reference.CV_8UC2, [9, 9, 9, 9]);
    const operationArguments = {
      multiply: [left, right, destination],
      divide: [left, right, destination],
      addWeighted: [left, 0.5, right, 0.25, 1, destination],
      convertScaleAbs: [left, destination],
    }[operation];
    const audit = auditTypedMatCall(
      () => reference[operation](...operationArguments),
      [
        ["left", left],
        ["right", right],
        ["destination", destination],
      ],
    );
    safeDelete(destination);
    safeDelete(right);
    safeDelete(left);
    return { operation, audit };
  });

  const overlap = ["multiply", "divide", "addWeighted", "convertScaleAbs"].map((operation) => {
    const type = operation === "convertScaleAbs" ? reference.CV_8UC1 : reference.CV_32FC1;
    const parent = makeSeedMat(reference, 1, 5, type, [1, 2, 3, 4, 5]);
    const source = parent.roi(new reference.Rect(0, 0, 3, 1));
    const destination = parent.roi(new reference.Rect(1, 0, 3, 1));
    const other = makeSeedMat(reference, 1, 3, type, [3, 3, 3]);
    const operationArguments = {
      multiply: [source, other, destination],
      divide: [source, other, destination],
      addWeighted: [source, 3, other, 0, 0, destination],
      convertScaleAbs: [source, destination, 3],
    }[operation];
    const audit = auditTypedMatCall(
      () => reference[operation](...operationArguments),
      [
        ["parent", parent],
        ["source", source],
        ["destination", destination],
        ["other", other],
      ],
    );
    safeDelete(other);
    safeDelete(destination);
    safeDelete(source);
    safeDelete(parent);
    return { operation, audit };
  });

  const empty = ["multiply", "divide", "addWeighted", "convertScaleAbs"].flatMap((operation) =>
    [
      ["canonical", 0, 0],
      ["zero-by-three", 0, 3],
      ["three-by-zero", 3, 0],
    ].map(([name, rows, cols]) => {
      const left = new reference.Mat(rows, cols, reference.CV_32FC1);
      const right = new reference.Mat(rows, cols, reference.CV_32FC1);
      const destination = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [9]);
      const operationArguments = {
        multiply: [left, right, destination],
        divide: [left, right, destination],
        addWeighted: [left, 0.5, right, 0.5, 1, destination],
        convertScaleAbs: [left, destination],
      }[operation];
      const audit = auditTypedMatCall(
        () => reference[operation](...operationArguments),
        [
          ["left", left],
          ["right", right],
          ["destination", destination],
        ],
      );
      safeDelete(destination);
      safeDelete(right);
      safeDelete(left);
      return { operation, name, audit };
    }),
  );

  const auditMixedDepthCase = (configuration, operation, explicitDtype) => {
    const left = makeSeedMat(reference, 2, 2, configuration.leftType, configuration.leftValues);
    const right = makeSeedMat(reference, 2, 2, configuration.rightType, configuration.rightValues);
    const destination = new reference.Mat();
    const operationArguments = {
      multiply: explicitDtype
        ? [left, right, destination, 1, configuration.dtype]
        : [left, right, destination],
      divide: explicitDtype
        ? [left, right, destination, 1, configuration.dtype]
        : [left, right, destination],
      addWeighted: explicitDtype
        ? [left, 0.5, right, 2, 1, destination, configuration.dtype]
        : [left, 0.5, right, 2, 1, destination],
    }[operation];
    const audit = auditTypedMatCall(
      () => reference[operation](...operationArguments),
      [
        ["left", left],
        ["right", right],
        ["destination", destination],
      ],
    );
    safeDelete(destination);
    safeDelete(right);
    safeDelete(left);
    return { name: configuration.name, operation, explicitDtype, audit };
  };
  const mixedDepth = [
    {
      name: "U8-F32",
      leftType: reference.CV_8UC1,
      leftValues: [2, 10, 250, 5],
      rightType: reference.CV_32FC1,
      rightValues: [0.5, 4, -2, 0.25],
      dtype: reference.CV_32F,
    },
    {
      name: "I16-F64",
      leftType: reference.CV_16SC1,
      leftValues: [-300, 1000, 32_767, -32_768],
      rightType: reference.CV_64FC1,
      rightValues: [0.5, -2, 1.5, 2],
      dtype: reference.CV_64F,
    },
  ].flatMap((configuration) =>
    ["multiply", "divide", "addWeighted"].flatMap((operation) => [
      auditMixedDepthCase(configuration, operation, false),
      auditMixedDepthCase(configuration, operation, true),
    ]),
  );

  const auditI32OverflowCase = (configuration, explicitDtype) => {
    const left = makeSeedMat(reference, 2, 2, reference.CV_32SC1, configuration.leftValues);
    const right = makeSeedMat(reference, 2, 2, reference.CV_32SC1, configuration.rightValues);
    const destination = new reference.Mat();
    const operationArguments = {
      multiply: explicitDtype
        ? [left, right, destination, 1, reference.CV_32S]
        : [left, right, destination],
      divide: explicitDtype
        ? [left, right, destination, 1, reference.CV_32S]
        : [left, right, destination],
      addWeighted: explicitDtype
        ? [left, 2, right, 1, 0, destination, reference.CV_32S]
        : [left, 2, right, 1, 0, destination],
    }[configuration.operation];
    const audit = auditTypedMatCall(
      () => reference[configuration.operation](...operationArguments),
      [
        ["left", left],
        ["right", right],
        ["destination", destination],
      ],
    );
    safeDelete(destination);
    safeDelete(right);
    safeDelete(left);
    return { operation: configuration.operation, explicitDtype, audit };
  };
  const i32Overflow = [
    {
      operation: "multiply",
      leftValues: [2_147_483_647, 50_000, -2_147_483_648, -50_000],
      rightValues: [2, 50_000, -1, 50_000],
    },
    {
      operation: "divide",
      leftValues: [2_147_483_647, -2_147_483_648, 7, -7],
      rightValues: [1, -1, 2, 2],
    },
    {
      operation: "addWeighted",
      leftValues: [2_147_483_647, -2_147_483_648, 2_000_000_000, -2_000_000_000],
      rightValues: [1, -1, 2_000_000_000, -2_000_000_000],
    },
  ].flatMap((configuration) => [
    auditI32OverflowCase(configuration, false),
    auditI32OverflowCase(configuration, true),
  ]);

  return { contracts, dtype, scale, replacement, overlap, empty, mixedDepth, i32Overflow };
}

function auditAffineTransform(reference) {
  const points = {
    source: [0, 0, 1, 0, 0, 1],
    destination: [2, 3, 4, 3, 2, 6],
  };
  const run = (name, source, destination) => {
    const sourceBefore = capturePrimitive(() => summarizeTypedMat(source));
    const destinationBefore = capturePrimitive(() => summarizeTypedMat(destination));
    const call = captureRotationMatrix(() => reference.getAffineTransform(source, destination));
    const sourceAfter = capturePrimitive(() => summarizeTypedMat(source));
    const destinationAfter = capturePrimitive(() => summarizeTypedMat(destination));
    safeDelete(destination);
    safeDelete(source);
    return { name, sourceBefore, destinationBefore, call, sourceAfter, destinationAfter };
  };
  const make = (rows, cols, type, values) => makeSeedMat(reference, rows, cols, type, values);
  const aritySource = make(3, 1, reference.CV_32FC2, points.source);
  const arityDestination = make(3, 1, reference.CV_32FC2, points.destination);
  const arity = {
    length: reference.getAffineTransform.length,
    zero: captureCall(() => reference.getAffineTransform()),
    one: captureCall(() => reference.getAffineTransform(aritySource)),
    two: captureRotationMatrix(() => reference.getAffineTransform(aritySource, arityDestination)),
    three: captureCall(() => reference.getAffineTransform(aritySource, arityDestination, 1)),
  };
  safeDelete(arityDestination);
  safeDelete(aritySource);

  let destinationReads = 0;
  const deferredDestination = {
    get $$() {
      destinationReads += 1;
      return undefined;
    },
  };
  const conversionOrder = {
    call: captureCall(() => reference.getAffineTransform({}, deferredDestination)),
    destinationReads,
  };

  const layouts = [
    run(
      "F32 3x1C2",
      make(3, 1, reference.CV_32FC2, points.source),
      make(3, 1, reference.CV_32FC2, points.destination),
    ),
    run(
      "F32 1x3C2",
      make(1, 3, reference.CV_32FC2, points.source),
      make(1, 3, reference.CV_32FC2, points.destination),
    ),
    run(
      "F32 3x2C1",
      make(3, 2, reference.CV_32FC1, points.source),
      make(3, 2, reference.CV_32FC1, points.destination),
    ),
    run(
      "F64 3x1C2",
      make(3, 1, reference.CV_64FC2, points.source),
      make(3, 1, reference.CV_64FC2, points.destination),
    ),
    run(
      "U8 3x1C2",
      make(3, 1, reference.CV_8UC2, points.source),
      make(3, 1, reference.CV_8UC2, points.destination),
    ),
    run(
      "wrong shape",
      make(2, 1, reference.CV_32FC2, points.source.slice(0, 4)),
      make(2, 1, reference.CV_32FC2, points.destination.slice(0, 4)),
    ),
  ];

  const stridedSourceParent = make(
    3,
    2,
    reference.CV_32FC2,
    [99, 99, 0, 0, 99, 99, 1, 0, 99, 99, 0, 1],
  );
  const stridedDestinationParent = make(
    3,
    2,
    reference.CV_32FC2,
    [99, 99, 2, 3, 99, 99, 4, 3, 99, 99, 2, 6],
  );
  const stridedSource = stridedSourceParent.roi(new reference.Rect(1, 0, 1, 3));
  const stridedDestination = stridedDestinationParent.roi(new reference.Rect(1, 0, 1, 3));
  const strided = run("strided F32 3x1C2", stridedSource, stridedDestination);
  safeDelete(stridedDestinationParent);
  safeDelete(stridedSourceParent);

  const numeric = [
    run(
      "fractional",
      make(3, 1, reference.CV_32FC2, [0.25, -0.5, 2.5, 0.75, -1.25, 3.5]),
      make(3, 1, reference.CV_32FC2, [4.5, -2.25, 8.75, 1.5, -3.5, 9.25]),
    ),
    run(
      "collinear",
      make(3, 1, reference.CV_32FC2, [0, 0, 1, 1, 2, 2]),
      make(3, 1, reference.CV_32FC2, points.destination),
    ),
    run(
      "nonfinite source",
      make(3, 1, reference.CV_32FC2, [Number.NaN, 0, 1, 0, 0, 1]),
      make(3, 1, reference.CV_32FC2, points.destination),
    ),
    run(
      "signed zero",
      make(3, 1, reference.CV_32FC2, [-0, -0, 1, -0, -0, 1]),
      make(3, 1, reference.CV_32FC2, [-0, -0, 2, -0, -0, 3]),
    ),
  ];
  return { arity, conversionOrder, layouts, strided, numeric };
}

function auditInvertAffineTransform(reference) {
  const make = (rows, cols, type, values) => makeSeedMat(reference, rows, cols, type, values);
  const run = (name, source, destination) => {
    const audit = auditTypedMatCall(
      () => reference.invertAffineTransform(source, destination),
      [
        ["source", source],
        ["destination", destination],
      ],
    );
    safeDelete(destination);
    safeDelete(source);
    return { name, audit };
  };

  const aritySource = make(2, 3, reference.CV_64FC1, [2, 0, 4, 0, 3, -6]);
  const arityDestination = new reference.Mat();
  const arity = {
    length: reference.invertAffineTransform.length,
    zero: captureCall(() => reference.invertAffineTransform()),
    one: captureCall(() => reference.invertAffineTransform(aritySource)),
    two: auditTypedMatCall(
      () => reference.invertAffineTransform(aritySource, arityDestination),
      [
        ["source", aritySource],
        ["destination", arityDestination],
      ],
    ),
    three: captureCall(() => reference.invertAffineTransform(aritySource, arityDestination, 1)),
  };
  safeDelete(arityDestination);
  safeDelete(aritySource);

  let destinationReads = 0;
  const deferredDestination = {
    get $$() {
      destinationReads += 1;
      return undefined;
    },
  };
  const conversionOrder = {
    call: captureCall(() => reference.invertAffineTransform({}, deferredDestination)),
    destinationReads,
  };

  const layouts = [
    run("F32 to empty", make(2, 3, reference.CV_32FC1, [2, 0, 4, 0, 3, -6]), new reference.Mat()),
    run("F64 to empty", make(2, 3, reference.CV_64FC1, [2, 0, 4, 0, 3, -6]), new reference.Mat()),
    run(
      "F64 to F32 destination",
      make(2, 3, reference.CV_64FC1, [2, 0, 4, 0, 3, -6]),
      make(2, 3, reference.CV_32FC1, [9, 9, 9, 9, 9, 9]),
    ),
    run(
      "F32 to F64 destination",
      make(2, 3, reference.CV_32FC1, [2, 0, 4, 0, 3, -6]),
      make(2, 3, reference.CV_64FC1, [9, 9, 9, 9, 9, 9]),
    ),
    run("U8 source", make(2, 3, reference.CV_8UC1, [2, 0, 4, 0, 3, 6]), new reference.Mat()),
    run(
      "F64 wrong shape",
      make(3, 2, reference.CV_64FC1, [2, 0, 4, 0, 3, -6]),
      new reference.Mat(),
    ),
    run(
      "F64 multichannel",
      make(
        2,
        3,
        reference.CV_64FC2,
        Array.from({ length: 12 }, (_, index) => index + 1),
      ),
      new reference.Mat(),
    ),
  ];

  const sourceParent = make(2, 5, reference.CV_64FC1, [99, 2, 0, 4, 99, 99, 0, 3, -6, 99]);
  const stridedSource = sourceParent.roi(new reference.Rect(1, 0, 3, 2));
  const strided = run("strided F64 source", stridedSource, new reference.Mat());
  safeDelete(sourceParent);

  const destinationParent = make(
    3,
    5,
    reference.CV_64FC1,
    Array.from({ length: 15 }, () => 99),
  );
  const destinationRoi = destinationParent.roi(new reference.Rect(1, 0, 3, 2));
  const destinationRegionSource = make(2, 3, reference.CV_64FC1, [2, 0, 4, 0, 3, -6]);
  const destinationRegion = {
    name: "F64 destination ROI",
    audit: auditTypedMatCall(
      () => reference.invertAffineTransform(destinationRegionSource, destinationRoi),
      [
        ["source", destinationRegionSource],
        ["destination", destinationRoi],
        ["parent", destinationParent],
      ],
    ),
  };
  safeDelete(destinationRegionSource);
  safeDelete(destinationRoi);
  safeDelete(destinationParent);

  const alias = make(2, 3, reference.CV_64FC1, [2, 0, 4, 0, 3, -6]);
  const inPlace = {
    name: "exact in place",
    audit: auditTypedMatCall(
      () => reference.invertAffineTransform(alias, alias),
      [["matrix", alias]],
    ),
  };
  safeDelete(alias);

  const numeric = [
    run(
      "fractional",
      make(2, 3, reference.CV_64FC1, [1.25, -0.5, 3.75, 2.5, 4.25, -1.5]),
      new reference.Mat(),
    ),
    run("singular", make(2, 3, reference.CV_64FC1, [1, 2, 3, 2, 4, 6]), new reference.Mat()),
    run("NaN", make(2, 3, reference.CV_64FC1, [Number.NaN, 0, 1, 0, 1, 2]), new reference.Mat()),
    run(
      "infinity",
      make(2, 3, reference.CV_64FC1, [Number.POSITIVE_INFINITY, 0, 1, 0, 1, 2]),
      new reference.Mat(),
    ),
    run("signed zero", make(2, 3, reference.CV_64FC1, [-0, 1, -0, 1, -0, 0]), new reference.Mat()),
  ];

  return { arity, conversionOrder, layouts, strided, destinationRegion, inPlace, numeric };
}

function auditStructuringElement(reference) {
  const capture = (callback) => {
    let matrix;
    try {
      matrix = callback();
      return { threw: false, matrix: summarizeTypedMat(matrix) };
    } catch (error) {
      return {
        threw: true,
        error: {
          name: error?.name,
          constructor: error?.constructor?.name,
          message: error?.message,
          text: String(error),
          instanceofError: error instanceof Error,
        },
      };
    } finally {
      safeDelete(matrix);
    }
  };
  const arity = {
    length: reference.getStructuringElement.length,
    zero: captureCall(() => reference.getStructuringElement()),
    one: captureCall(() => reference.getStructuringElement(0)),
    two: capture(() => reference.getStructuringElement(0, { width: 3, height: 3 })),
    three: capture(() =>
      reference.getStructuringElement(1, { width: 3, height: 3 }, { x: 1, y: 1 }),
    ),
    four: captureCall(() =>
      reference.getStructuringElement(0, { width: 3, height: 3 }, { x: 1, y: 1 }, 0),
    ),
  };
  const fixtures = [
    ["rect 3x2", 0, { width: 3, height: 2 }],
    ["cross default 3x3", 1, { width: 3, height: 3 }],
    ["cross anchor 0,1", 1, { width: 3, height: 3 }, { x: 0, y: 1 }],
    ["ellipse 1x1", 2, { width: 1, height: 1 }],
    ["ellipse 2x2", 2, { width: 2, height: 2 }],
    ["ellipse 4x4", 2, { width: 4, height: 4 }],
    ["ellipse 5x3", 2, { width: 5, height: 3 }],
    ["diamond 5x5", 3, { width: 5, height: 5 }],
  ].map(([name, kind, size, anchor]) => ({
    name,
    call: capture(() =>
      anchor === undefined
        ? reference.getStructuringElement(kind, size)
        : reference.getStructuringElement(kind, size, anchor),
    ),
  }));
  const conversions = [
    ["boolean shape", true, { width: 3, height: 3 }, { x: 1, y: 1 }],
    ["numeric string shape", "1", { width: 3, height: 3 }, { x: 1, y: 1 }],
    ["boolean size", 0, { width: true, height: true }, { x: 0, y: 0 }],
    ["fractional size", 0, { width: 3.9, height: 2.9 }, { x: 0, y: 0 }],
    ["boolean anchor", 1, { width: 3, height: 3 }, { x: true, y: false }],
    ["partial default anchor", 1, { width: 3, height: 3 }, { x: -1, y: 0 }],
  ].map(([name, kind, size, anchor]) => ({
    name,
    call: capture(() => reference.getStructuringElement(kind, size, anchor)),
  }));
  const rejected = [
    ["unknown shape", 4, { width: 3, height: 3 }, { x: 1, y: 1 }],
    ["zero width", 0, { width: 0, height: 3 }, { x: 0, y: 0 }],
    ["negative width", 0, { width: -1, height: 3 }, { x: 0, y: 0 }],
    ["anchor outside", 1, { width: 3, height: 3 }, { x: 3, y: 1 }],
    ["missing size width", 0, { height: 3 }, { x: 0, y: 0 }],
    ["string size width", 0, { width: "3", height: 3 }, { x: 0, y: 0 }],
  ].map(([name, kind, size, anchor]) => ({
    name,
    call: capture(() => reference.getStructuringElement(kind, size, anchor)),
  }));
  return { arity, fixtures, conversions, rejected };
}

function auditHanningWindow(reference) {
  const make = (rows, cols, type, values) => makeSeedMat(reference, rows, cols, type, values);
  const run = (name, destination, size, type) => {
    const audit = auditTypedMatCall(
      () => reference.createHanningWindow(destination, size, type),
      [["destination", destination]],
    );
    safeDelete(destination);
    return { name, audit };
  };

  const arityDestination = new reference.Mat();
  const arity = {
    length: reference.createHanningWindow.length,
    zero: captureCall(() => reference.createHanningWindow()),
    one: captureCall(() => reference.createHanningWindow(arityDestination)),
    two: captureCall(() =>
      reference.createHanningWindow(arityDestination, { width: 4, height: 3 }),
    ),
    three: captureCall(() =>
      reference.createHanningWindow(arityDestination, { width: 4, height: 3 }, reference.CV_32F),
    ),
    four: captureCall(() =>
      reference.createHanningWindow(arityDestination, { width: 4, height: 3 }, reference.CV_32F, 1),
    ),
  };
  safeDelete(arityDestination);

  let widthReads = 0;
  let heightReads = 0;
  const deferredSize = {
    get width() {
      widthReads += 1;
      return 4;
    },
    get height() {
      heightReads += 1;
      return 3;
    },
  };
  const conversionOrder = {
    call: captureCall(() => reference.createHanningWindow({}, deferredSize, reference.CV_32F)),
    widthReads,
    heightReads,
  };

  const fixtures = [
    run("F32 asymmetric 4x3", new reference.Mat(), { width: 4, height: 3 }, reference.CV_32F),
    run("F64 asymmetric 5x4", new reference.Mat(), { width: 5, height: 4 }, reference.CV_64F),
  ];
  const destinations = [
    run(
      "compatible F32",
      make(
        3,
        4,
        reference.CV_32FC1,
        Array.from({ length: 12 }, () => 99),
      ),
      { width: 4, height: 3 },
      reference.CV_32F,
    ),
    run(
      "wrong shape",
      make(
        2,
        2,
        reference.CV_32FC1,
        Array.from({ length: 4 }, () => 99),
      ),
      { width: 4, height: 3 },
      reference.CV_32F,
    ),
    run(
      "wrong depth and channels",
      make(
        3,
        4,
        reference.CV_8UC2,
        Array.from({ length: 24 }, () => 99),
      ),
      { width: 4, height: 3 },
      reference.CV_64F,
    ),
  ];

  const roiParent = make(
    5,
    6,
    reference.CV_32FC1,
    Array.from({ length: 30 }, () => 99),
  );
  const roi = roiParent.roi(new reference.Rect(1, 1, 4, 3));
  const compatibleRoi = {
    name: "compatible F32 ROI",
    audit: auditTypedMatCall(
      () => reference.createHanningWindow(roi, { width: 4, height: 3 }, reference.CV_32F),
      [
        ["destination", roi],
        ["parent", roiParent],
      ],
    ),
  };
  safeDelete(roi);
  safeDelete(roiParent);

  const conversions = [
    run("fractional size and type", new reference.Mat(), { width: 4.9, height: 3.9 }, 5.9),
    run("boolean size", new reference.Mat(), { width: true, height: true }, reference.CV_32F),
  ];
  const rejected = [
    ["width below two", { width: 1, height: 3 }, reference.CV_32F],
    ["height below two", { width: 4, height: 1 }, reference.CV_32F],
    ["negative width", { width: -1, height: 3 }, reference.CV_32F],
    ["unsupported U8 type", { width: 4, height: 3 }, reference.CV_8U],
    ["multichannel F32 type", { width: 4, height: 3 }, reference.CV_32FC2],
    ["missing width", { height: 3 }, reference.CV_32F],
    ["string width", { width: "4", height: 3 }, reference.CV_32F],
    ["string type", { width: 4, height: 3 }, "5"],
  ].map(([name, size, type]) =>
    run(
      name,
      make(
        2,
        3,
        reference.CV_64FC1,
        Array.from({ length: 6 }, () => 99),
      ),
      size,
      type,
    ),
  );

  return { arity, conversionOrder, fixtures, destinations, compatibleRoi, conversions, rejected };
}

function encodeReductionResult(value) {
  if (typeof value === "number") return encodeFloat64(value);
  if (Array.isArray(value)) return value.map((entry) => encodeReductionResult(entry));
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, encodeReductionResult(entry)]),
    );
  }
  return encodeValue(value);
}

function captureReduction(callback) {
  try {
    return { threw: false, value: encodeReductionResult(callback()) };
  } catch (error) {
    return {
      threw: true,
      error: {
        name: error?.name,
        constructor: error?.constructor?.name,
        message: error?.message,
        text: String(error),
        instanceofError: error instanceof Error,
      },
    };
  }
}

function auditMaskedReducer(reference, method) {
  const make = (rows, columns, type, values) => makeSeedMat(reference, rows, columns, type, values);
  const run = (name, source, mask) => {
    const call = captureReduction(() =>
      mask === undefined ? reference[method](source) : reference[method](source, mask),
    );
    safeDelete(mask);
    safeDelete(source);
    return { name, call };
  };
  const valuesForDepth = {
    mean: {
      u8: [1, 2, 3, 10],
      i8: [-8, -1, 3, 10],
      u16: [1, 255, 1024, 65_535],
      i16: [-32_768, -17, 23, 32_767],
      i32: [-2_000_000_000, -3, 11, 2_000_000_000],
      f32: [-1.5, -0.25, 2.25, 9.5],
      f64: [-1.25, -0.5, 2.75, 9.5],
    },
    minMaxLoc: {
      u8: [9, 2, 7, 2, 9, 4],
      i8: [-8, 2, 7, -8, 7, 4],
      u16: [65_535, 2, 1024, 2, 65_535, 4],
      i16: [-32_768, 2, 32_767, -32_768, 32_767, 4],
      i32: [-2_000_000_000, 2, 2_000_000_000, -2_000_000_000, 2_000_000_000, 4],
      f32: [-8.5, 2.25, 7.5, -8.5, 7.5, 4],
      f64: [-8.25, 2.5, 7.75, -8.25, 7.75, 4],
    },
  }[method];
  const depthCases = [
    ["U8", reference.CV_8UC1, valuesForDepth.u8],
    ["I8", reference.CV_8SC1, valuesForDepth.i8],
    ["U16", reference.CV_16UC1, valuesForDepth.u16],
    ["I16", reference.CV_16SC1, valuesForDepth.i16],
    ["I32", reference.CV_32SC1, valuesForDepth.i32],
    ["F32", reference.CV_32FC1, valuesForDepth.f32],
    ["F64", reference.CV_64FC1, valuesForDepth.f64],
  ].map(([name, type, values]) => run(name, make(2, values.length / 2, type, values)));

  const aritySource = make(2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const arityMask = make(2, 3, reference.CV_8UC1, [1, 0, 1, 0, 1, 0]);
  const arity = {
    length: reference[method].length,
    zero: captureReduction(() => reference[method]()),
    one: captureReduction(() => reference[method](aritySource)),
    two: captureReduction(() => reference[method](aritySource, arityMask)),
    three: captureReduction(() => reference[method](aritySource, arityMask, 1)),
  };
  safeDelete(arityMask);
  safeDelete(aritySource);

  let maskReads = 0;
  const deferredMask = {
    get $$() {
      maskReads += 1;
      return undefined;
    },
  };
  const conversionOrder = {
    call: captureReduction(() => reference[method]({}, deferredMask)),
    maskReads,
  };

  const channelCases =
    method === "mean"
      ? [
          ["C1", reference.CV_64FC1, [1, 2, 3, 4, 5, 6]],
          ["C2", reference.CV_64FC2, [1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60]],
          [
            "C3",
            reference.CV_64FC3,
            [1, 10, 100, 2, 20, 200, 3, 30, 300, 4, 40, 400, 5, 50, 500, 6, 60, 600],
          ],
          [
            "C4",
            reference.CV_64FC4,
            [
              1, 10, 100, 1000, 2, 20, 200, 2000, 3, 30, 300, 3000, 4, 40, 400, 4000, 5, 50, 500,
              5000, 6, 60, 600, 6000,
            ],
          ],
        ].map(([name, type, values]) => run(name, make(2, 3, type, values)))
      : [run("C1", make(2, 3, reference.CV_64FC1, [9, -3, -3, 9, 0, 9]))];

  const rejectedChannels = [
    method === "mean"
      ? run("C5", new reference.Mat(2, 2, reference.CV_64FC4 + 8))
      : run("C2", make(2, 2, reference.CV_64FC2, [1, 10, 2, 20, 3, 30, 4, 40])),
  ];

  const maskedSourceValues =
    method === "mean"
      ? [5, 50, 500, 1, 10, 100, 9, 90, 900, -2, -20, -200, 9, 90, 900, 3, 30, 300]
      : [5, 1, 9, -2, 9, 3];
  const maskedSourceType = method === "mean" ? reference.CV_64FC3 : reference.CV_64FC1;
  const masks = [
    ["all selected", [1, 1, 1, 1, 1, 1]],
    ["selective nonzero", [0, 255, 1, 0, 0, 7]],
    ["all zero", [0, 0, 0, 0, 0, 0]],
  ].map(([name, values]) =>
    run(
      name,
      make(2, 3, maskedSourceType, maskedSourceValues),
      make(2, 3, reference.CV_8UC1, values),
    ),
  );
  const rejectedMasks = [
    run(
      "I8 depth",
      make(2, 3, reference.CV_64FC1, [5, 1, 9, -2, 9, 3]),
      make(2, 3, reference.CV_8SC1, [1, 0, 1, 0, 1, 0]),
    ),
    run(
      "two channels",
      make(2, 3, reference.CV_64FC1, [5, 1, 9, -2, 9, 3]),
      make(2, 3, reference.CV_8UC2, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
    ),
    run(
      "wrong rows",
      make(2, 3, reference.CV_64FC1, [5, 1, 9, -2, 9, 3]),
      make(1, 3, reference.CV_8UC1, [1, 0, 1]),
    ),
    run(
      "wrong columns",
      make(2, 3, reference.CV_64FC1, [5, 1, 9, -2, 9, 3]),
      make(2, 2, reference.CV_8UC1, [1, 0, 1, 0]),
    ),
  ];

  const canonicalEmpty = run("canonical", new reference.Mat());
  const typedEmpty = [
    ["zero by zero", 0, 0, reference.CV_32FC1],
    ["zero by three", 0, 3, reference.CV_32FC1],
    ["three by zero", 3, 0, reference.CV_64FC1],
  ].map(([name, rows, columns, type]) => run(name, new reference.Mat(rows, columns, type)));
  const emptyMasks = [
    run("canonical mask", make(2, 3, reference.CV_64FC1, [5, 1, 9, -2, 9, 3]), new reference.Mat()),
    run(
      "typed mask",
      make(2, 3, reference.CV_64FC1, [5, 1, 9, -2, 9, 3]),
      new reference.Mat(0, 0, reference.CV_8UC1),
    ),
  ];

  const sourceChannels = method === "mean" ? 3 : 1;
  const sourceType = method === "mean" ? reference.CV_64FC3 : reference.CV_64FC1;
  const sourceParentValues = Array.from({ length: 3 * 5 * sourceChannels }, (_, index) =>
    index % 7 === 0 ? -index : index + 0.5,
  );
  const maskParentValues = [0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 3, 0];
  const sourceParent = make(3, 5, sourceType, sourceParentValues);
  const maskParent = make(3, 5, reference.CV_8UC1, maskParentValues);
  const sourceRegion = sourceParent.roi(new reference.Rect(1, 0, 3, 3));
  const maskRegion = maskParent.roi(new reference.Rect(1, 0, 3, 3));
  const strided = {
    sourceContinuous: sourceRegion.isContinuous(),
    maskContinuous: maskRegion.isContinuous(),
    unmasked: captureReduction(() => reference[method](sourceRegion)),
    masked: captureReduction(() => reference[method](sourceRegion, maskRegion)),
  };
  safeDelete(maskRegion);
  safeDelete(sourceRegion);
  safeDelete(maskParent);
  safeDelete(sourceParent);

  const numeric =
    method === "mean"
      ? [
          run("NaN", make(1, 3, reference.CV_64FC1, [Number.NaN, 1, 2])),
          run(
            "opposite infinities",
            make(1, 2, reference.CV_64FC1, [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]),
          ),
          run("negative zeros", make(1, 3, reference.CV_64FC1, [-0, -0, -0])),
        ]
      : [
          run("NaN among finite", make(1, 4, reference.CV_64FC1, [Number.NaN, 2, -1, 4])),
          run(
            "infinities",
            make(1, 4, reference.CV_64FC1, [
              0,
              Number.POSITIVE_INFINITY,
              -1,
              Number.NEGATIVE_INFINITY,
            ]),
          ),
          run("signed zero tie", make(1, 3, reference.CV_64FC1, [-0, 0, -0])),
        ];

  const locations =
    method === "minMaxLoc"
      ? [
          run("row-major ties", make(2, 4, reference.CV_64FC1, [5, -3, -3, 9, 9, 0, -3, 9])),
          run(
            "masked ties",
            make(2, 4, reference.CV_64FC1, [5, -3, -3, 9, 9, 0, -3, 9]),
            make(2, 4, reference.CV_8UC1, [0, 0, 1, 0, 1, 0, 1, 1]),
          ),
        ]
      : [];

  const deletedSource = make(1, 2, reference.CV_8UC1, [1, 2]);
  safeDelete(deletedSource);
  const liveSource = make(1, 2, reference.CV_8UC1, [1, 2]);
  const deletedMask = make(1, 2, reference.CV_8UC1, [1, 0]);
  safeDelete(deletedMask);
  const lifetime = {
    deletedSource: captureReduction(() => reference[method](deletedSource)),
    deletedMask: captureReduction(() => reference[method](liveSource, deletedMask)),
    nonMatSources: [null, {}, new Uint8Array([1, 2])].map((value) =>
      captureReduction(() => reference[method](value)),
    ),
    nonMatMasks: [null, {}, new Uint8Array([1, 0])].map((value) =>
      captureReduction(() => reference[method](liveSource, value)),
    ),
  };
  safeDelete(liveSource);

  return {
    arity,
    conversionOrder,
    depths: depthCases,
    channels: channelCases,
    rejectedChannels,
    masks,
    rejectedMasks,
    empty: { canonical: canonicalEmpty, typed: typedEmpty, masks: emptyMasks },
    strided,
    numeric,
    locations,
    lifetime,
  };
}

function traceMatrixValues(rows, columns, channels, diagonalValues, fill = 0) {
  const values = Array.from({ length: rows * columns * channels }, () => fill);
  for (let row = 0; row < diagonalValues.length; row += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      values[(row * columns + row) * channels + channel] = diagonalValues[row][channel];
    }
  }
  return values;
}

function captureTrace(callback) {
  try {
    const value = callback();
    const array = Array.isArray(value);
    return {
      threw: false,
      scalarContract: {
        array,
        plainArray: array && Object.getPrototypeOf(value) === Array.prototype,
        length: array ? value.length : null,
        numbersOnly: array && value.every((entry) => typeof entry === "number"),
      },
      value: encodeReductionResult(value),
    };
  } catch (error) {
    return {
      threw: true,
      error: {
        name: error?.name,
        constructor: error?.constructor?.name,
        message: error?.message,
        text: String(error),
        instanceofError: error instanceof Error,
      },
    };
  }
}

function auditTrace(reference) {
  const make = (rows, columns, type, values) => makeSeedMat(reference, rows, columns, type, values);
  const run = (name, source) => {
    const call = captureTrace(() => reference.trace(source));
    safeDelete(source);
    return { name, call };
  };

  const aritySource = make(2, 2, reference.CV_64FC1, [1, 0, 0, 2]);
  const arity = {
    length: reference.trace.length,
    zero: captureTrace(() => reference.trace()),
    exact: captureTrace(() => reference.trace(aritySource)),
    extra: captureTrace(() => reference.trace(aritySource, 1)),
  };
  safeDelete(aritySource);

  const depthTypes = [
    ["U8", reference.CV_8UC1],
    ["I8", reference.CV_8SC1],
    ["U16", reference.CV_16UC1],
    ["I16", reference.CV_16SC1],
    ["I32", reference.CV_32SC1],
    ["F32", reference.CV_32FC1],
    ["F64", reference.CV_64FC1],
  ];
  const depthChannels = depthTypes.flatMap(([depth, singleChannelType]) =>
    [1, 2, 3, 4].map((channels) => {
      const first = Array.from({ length: channels }, (_, channel) => channel + 1);
      const second = Array.from({ length: channels }, (_, channel) => (channel + 1) * 10);
      const type = singleChannelType + ((channels - 1) << 3);
      return run(
        `${depth} C${channels}`,
        make(2, 3, type, traceMatrixValues(2, 3, channels, [first, second], 91)),
      );
    }),
  );

  const rejectedChannels = run("F64 C5", new reference.Mat(2, 2, reference.CV_64FC1 + (4 << 3)));

  const empty = [
    run("canonical", new reference.Mat()),
    run("zero rows F32 C2", new reference.Mat(0, 3, reference.CV_32FC2)),
    run("zero columns F64 C4", new reference.Mat(3, 0, reference.CV_64FC4)),
  ];

  const stridedParent = make(
    3,
    5,
    reference.CV_64FC3,
    Array.from({ length: 45 }, (_, index) => index + 0.25),
  );
  const stridedSource = stridedParent.roi(new reference.Rect(1, 0, 3, 3));
  const strided = {
    continuous: stridedSource.isContinuous(),
    call: captureTrace(() => reference.trace(stridedSource)),
  };
  safeDelete(stridedSource);
  safeDelete(stridedParent);

  const numeric = [
    run(
      "F32 widened accumulation",
      make(
        3,
        3,
        reference.CV_32FC1,
        traceMatrixValues(3, 3, 1, [[16_777_216], [1], [-16_777_216]]),
      ),
    ),
    run(
      "F64 fractional accumulation",
      make(3, 3, reference.CV_64FC1, traceMatrixValues(3, 3, 1, [[0.1], [0.2], [0.3]])),
    ),
    run(
      "F64 ordered accumulation",
      make(3, 3, reference.CV_64FC1, traceMatrixValues(3, 3, 1, [[2 ** 53], [1], [-(2 ** 53)]])),
    ),
    run("signed zero lanes", make(1, 1, reference.CV_64FC4, [-0, 0, -0, 0])),
    run(
      "NaN and infinities",
      make(
        2,
        2,
        reference.CV_64FC4,
        traceMatrixValues(2, 2, 4, [
          [
            Number.NaN,
            Number.POSITIVE_INFINITY,
            Number.NEGATIVE_INFINITY,
            Number.POSITIVE_INFINITY,
          ],
          [1, 2, 3, Number.NEGATIVE_INFINITY],
        ]),
      ),
    ),
  ];

  const deletedSource = make(1, 1, reference.CV_64FC1, [7]);
  safeDelete(deletedSource);
  const lifetime = {
    deletedSource: captureTrace(() => reference.trace(deletedSource)),
    nonMatSources: [undefined, null, {}, 1, new Uint8Array([1])].map((value) =>
      captureTrace(() => reference.trace(value)),
    ),
  };

  return { arity, depthChannels, rejectedChannels, empty, strided, numeric, lifetime };
}

function auditBitwiseNot(reference) {
  const makeRaw = (rows, columns, type, bytes) => {
    const matrix = new reference.Mat(rows, columns, type);
    if (bytes.length > 0) matrix.data.set(bytes);
    return matrix;
  };
  // oxlint-disable-next-line unicorn/consistent-function-scoping -- The fixture generator belongs to this audit.
  const bytes = (length, seed = 1) =>
    Uint8Array.from({ length }, (_, index) => (seed + index * 37) & 255);
  const run = (name, source, destination, mask, extraMatrices = []) => {
    const matrices = [
      ["source", source],
      ["destination", destination],
      ...(mask === undefined ? [] : [["mask", mask]]),
      ...extraMatrices,
    ];
    const audit = auditTransposeCall(
      () =>
        mask === undefined
          ? reference.bitwise_not(source, destination)
          : reference.bitwise_not(source, destination, mask),
      matrices,
    );
    for (const matrix of new Set(matrices.map(([, value]) => value))) safeDelete(matrix);
    return { name, audit };
  };
  // oxlint-disable-next-line unicorn/consistent-function-scoping -- The type encoder belongs to this audit.
  const type = (depth, channels) => depth + ((channels - 1) << 3);
  const depthCases = [
    ["U8", reference.CV_8U, 1],
    ["I8", reference.CV_8S, 1],
    ["U16", reference.CV_16U, 2],
    ["I16", reference.CV_16S, 2],
    ["I32", reference.CV_32S, 4],
    ["F32", reference.CV_32F, 4],
    ["F64", reference.CV_64F, 8],
  ];

  const aritySource = makeRaw(2, 3, reference.CV_8UC1, new Uint8Array([1, 2, 3, 4, 5, 6]));
  const arityDestination = makeRaw(
    2,
    3,
    reference.CV_8UC1,
    new Uint8Array([99, 98, 97, 96, 95, 94]),
  );
  const arityMask = makeRaw(2, 3, reference.CV_8UC1, new Uint8Array([255, 0, 1, 0, 2, 0]));
  const arity = {
    length: reference.bitwise_not.length,
    zero: captureCall(() => reference.bitwise_not()),
    one: captureCall(() => reference.bitwise_not(aritySource)),
    two: captureCall(() => reference.bitwise_not(aritySource, arityDestination)),
    three: captureCall(() => reference.bitwise_not(aritySource, arityDestination, arityMask)),
    four: captureCall(() =>
      reference.bitwise_not(aritySource, arityDestination, arityMask, arityMask),
    ),
  };
  safeDelete(arityMask);
  safeDelete(arityDestination);
  safeDelete(aritySource);

  let destinationReads = 0;
  let maskReads = 0;
  const deferredDestination = {
    get $$() {
      destinationReads += 1;
      return undefined;
    },
  };
  const deferredMask = {
    get $$() {
      maskReads += 1;
      return undefined;
    },
  };
  const orderSource = makeRaw(1, 1, reference.CV_8UC1, new Uint8Array([1]));
  const conversionOrder = {
    sourceBeforeDestination: captureCall(() =>
      reference.bitwise_not({}, deferredDestination, deferredMask),
    ),
    destinationReadsAfterInvalidSource: destinationReads,
    maskReadsAfterInvalidSource: maskReads,
    destinationBeforeMask: captureCall(() => reference.bitwise_not(orderSource, {}, deferredMask)),
    maskReadsAfterInvalidDestination: maskReads,
  };
  safeDelete(orderSource);

  const depthChannels = depthCases.flatMap(([name, depth, width], depthIndex) =>
    [1, 2, 3, 4, 5].map((channels) => {
      const byteLength = 2 * 3 * channels * width;
      return run(
        `${name} C${channels}`,
        makeRaw(2, 3, type(depth, channels), bytes(byteLength, depthIndex * 19 + channels)),
        new reference.Mat(),
      );
    }),
  );
  const specialBits = [
    run(
      "F32 special bit patterns",
      makeRaw(
        2,
        3,
        reference.CV_32FC1,
        new Uint8Array([
          0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 128, 127, 0, 0, 128, 255, 1, 0, 192, 127, 1, 0, 0, 0,
        ]),
      ),
      new reference.Mat(),
    ),
    run(
      "F64 special bit patterns",
      makeRaw(
        1,
        4,
        reference.CV_64FC1,
        new Uint8Array([
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 240, 127, 1, 0, 0, 0,
          0, 0, 248, 127,
        ]),
      ),
      new reference.Mat(),
    ),
  ];

  const destinationSourceBytes = bytes(2 * 3 * 2 * 2, 17);
  const destinationCases = [
    run("fresh", makeRaw(2, 3, reference.CV_16UC2, destinationSourceBytes), new reference.Mat()),
    run(
      "compatible",
      makeRaw(2, 3, reference.CV_16UC2, destinationSourceBytes),
      makeRaw(2, 3, reference.CV_16UC2, bytes(destinationSourceBytes.length, 99)),
    ),
    run(
      "wrong layout",
      makeRaw(2, 3, reference.CV_16UC2, destinationSourceBytes),
      makeRaw(1, 2, reference.CV_8UC1, new Uint8Array([99, 98])),
    ),
  ];

  const maskedSourceBytes = bytes(2 * 3 * 3 * 4, 23);
  const maskValues = new Uint8Array([255, 0, 1, 0, 2, 0]);
  const maskCases = [
    run(
      "U8 compatible preserves unselected",
      makeRaw(2, 3, reference.CV_32FC3, maskedSourceBytes),
      makeRaw(2, 3, reference.CV_32FC3, bytes(maskedSourceBytes.length, 99)),
      makeRaw(2, 3, reference.CV_8UC1, maskValues),
    ),
    run(
      "I8 compatible preserves unselected",
      makeRaw(2, 3, reference.CV_32FC3, maskedSourceBytes),
      makeRaw(2, 3, reference.CV_32FC3, bytes(maskedSourceBytes.length, 99)),
      makeRaw(2, 3, reference.CV_8SC1, maskValues),
    ),
    run(
      "fresh zeros unselected",
      makeRaw(2, 3, reference.CV_32FC3, maskedSourceBytes),
      new reference.Mat(),
      makeRaw(2, 3, reference.CV_8UC1, maskValues),
    ),
    run(
      "wrong layout zeros unselected",
      makeRaw(2, 3, reference.CV_32FC3, maskedSourceBytes),
      makeRaw(1, 1, reference.CV_8UC1, new Uint8Array([99])),
      makeRaw(2, 3, reference.CV_8UC1, maskValues),
    ),
  ];

  const emptyMaskCases = [
    run(
      "canonical mask",
      makeRaw(2, 3, reference.CV_8UC1, new Uint8Array([1, 2, 3, 4, 5, 6])),
      new reference.Mat(),
      new reference.Mat(),
    ),
    run(
      "typed wrong-metadata mask",
      makeRaw(2, 3, reference.CV_8UC1, new Uint8Array([1, 2, 3, 4, 5, 6])),
      new reference.Mat(),
      new reference.Mat(0, 3, reference.CV_16UC2),
    ),
  ];

  const rejectedMask = (name, mask) =>
    run(
      name,
      makeRaw(2, 3, reference.CV_8UC2, bytes(12, 31)),
      makeRaw(2, 3, reference.CV_8UC2, bytes(12, 99)),
      mask,
    );
  const rejectedMasks = [
    rejectedMask("U16 depth", makeRaw(2, 3, reference.CV_16UC1, bytes(12, 1))),
    rejectedMask("two channels", makeRaw(2, 3, reference.CV_8UC2, bytes(12, 1))),
    rejectedMask("wrong rows", makeRaw(1, 3, reference.CV_8UC1, bytes(3, 1))),
    rejectedMask("wrong columns", makeRaw(2, 2, reference.CV_8UC1, bytes(4, 1))),
  ];

  const empty = [
    run("canonical source", new reference.Mat(), new reference.Mat()),
    run("zero rows F32 C2", new reference.Mat(0, 3, reference.CV_32FC2), new reference.Mat()),
    run("zero columns F64 C4", new reference.Mat(3, 0, reference.CV_64FC4), new reference.Mat()),
  ];

  const aliases = [
    (() => {
      const matrix = makeRaw(2, 3, reference.CV_8UC1, new Uint8Array([1, 2, 3, 4, 5, 6]));
      return run("source equals destination", matrix, matrix);
    })(),
    (() => {
      const matrix = makeRaw(2, 3, reference.CV_8UC1, new Uint8Array([1, 2, 3, 4, 5, 6]));
      return run(
        "source equals destination masked",
        matrix,
        matrix,
        makeRaw(2, 3, reference.CV_8UC1, maskValues),
      );
    })(),
    (() => {
      const sourceMask = makeRaw(2, 3, reference.CV_8UC1, maskValues);
      return run("source equals mask", sourceMask, new reference.Mat(), sourceMask);
    })(),
    (() => {
      const destinationMask = makeRaw(2, 3, reference.CV_8UC1, maskValues);
      return run(
        "destination equals mask",
        makeRaw(2, 3, reference.CV_8UC1, new Uint8Array([1, 2, 3, 4, 5, 6])),
        destinationMask,
        destinationMask,
      );
    })(),
    (() => {
      const matrix = makeRaw(2, 3, reference.CV_8UC1, maskValues);
      return run("all arguments alias", matrix, matrix, matrix);
    })(),
  ];

  const sourceParent = makeRaw(3, 5, reference.CV_16UC2, bytes(3 * 5 * 4, 7));
  const destinationParent = makeRaw(3, 5, reference.CV_16UC2, bytes(3 * 5 * 4, 99));
  const maskParent = makeRaw(
    3,
    5,
    reference.CV_8UC1,
    new Uint8Array([0, 1, 0, 2, 0, 1, 0, 3, 0, 4, 0, 5, 0, 6, 0]),
  );
  const strided = run(
    "strided source destination and mask",
    sourceParent.roi(new reference.Rect(1, 0, 3, 3)),
    destinationParent.roi(new reference.Rect(1, 0, 3, 3)),
    maskParent.roi(new reference.Rect(1, 0, 3, 3)),
    [
      ["sourceParent", sourceParent],
      ["destinationParent", destinationParent],
      ["maskParent", maskParent],
    ],
  );

  const overlapCases = [
    (() => {
      const parent = makeRaw(3, 6, reference.CV_8UC1, bytes(18, 1));
      return run(
        "overlapping source and destination",
        parent.roi(new reference.Rect(0, 0, 4, 3)),
        parent.roi(new reference.Rect(1, 0, 4, 3)),
        undefined,
        [["parent", parent]],
      );
    })(),
    (() => {
      const parent = makeRaw(3, 7, reference.CV_8UC1, bytes(21, 3));
      return run(
        "overlapping source destination and mask",
        parent.roi(new reference.Rect(0, 0, 4, 3)),
        parent.roi(new reference.Rect(1, 0, 4, 3)),
        parent.roi(new reference.Rect(2, 0, 4, 3)),
        [["parent", parent]],
      );
    })(),
  ];

  const deletedSource = makeRaw(1, 1, reference.CV_8UC1, new Uint8Array([1]));
  safeDelete(deletedSource);
  const deletedDestination = new reference.Mat();
  safeDelete(deletedDestination);
  const deletedMask = makeRaw(1, 1, reference.CV_8UC1, new Uint8Array([1]));
  safeDelete(deletedMask);
  const liveSource = makeRaw(1, 1, reference.CV_8UC1, new Uint8Array([1]));
  const liveDestination = new reference.Mat();
  const lifetime = {
    deletedSource: captureCall(() => reference.bitwise_not(deletedSource, liveDestination)),
    deletedDestination: captureCall(() => reference.bitwise_not(liveSource, deletedDestination)),
    deletedMask: captureCall(() => reference.bitwise_not(liveSource, liveDestination, deletedMask)),
    nonMatSources: [undefined, null, {}, 1, new Uint8Array([1])].map((value) =>
      captureCall(() => reference.bitwise_not(value, liveDestination)),
    ),
    nonMatDestinations: [undefined, null, {}, 1, new Uint8Array([1])].map((value) =>
      captureCall(() => reference.bitwise_not(liveSource, value)),
    ),
    nonMatMasks: [undefined, null, {}, 1, new Uint8Array([1])].map((value) =>
      captureCall(() => reference.bitwise_not(liveSource, liveDestination, value)),
    ),
  };
  safeDelete(liveDestination);
  safeDelete(liveSource);

  return {
    arity,
    conversionOrder,
    depthChannels,
    specialBits,
    destinations: destinationCases,
    masks: maskCases,
    emptyMasks: emptyMaskCases,
    rejectedMasks,
    empty,
    aliases,
    strided,
    overlap: overlapCases,
    lifetime,
  };
}

function auditSetIdentity(reference) {
  const auditCase = (name, rows, cols, type, values, scalar, useDefault = false) => {
    const matrix = makeSeedMat(reference, rows, cols, type, values);
    const audit = auditTypedMatCall(
      () => (useDefault ? reference.setIdentity(matrix) : reference.setIdentity(matrix, scalar)),
      [["matrix", matrix]],
    );
    safeDelete(matrix);
    return { name, audit };
  };
  const arityMatrix = makeSeedMat(reference, 1, 1, reference.CV_8UC1, [0]);
  const arity = {
    length: reference.setIdentity.length,
    zero: captureCall(() => reference.setIdentity()),
    one: captureCall(() => reference.setIdentity(arityMatrix)),
    two: captureCall(() => reference.setIdentity(arityMatrix, [2, 0, 0, 0])),
    three: captureCall(() => reference.setIdentity(arityMatrix, [2, 0, 0, 0], 1)),
  };
  safeDelete(arityMatrix);
  let scalarReads = 0;
  const deferredScalar = {
    get length() {
      scalarReads += 1;
      return 4;
    },
    0: 1,
    1: 2,
    2: 3,
    3: 4,
  };
  const conversionOrder = {
    call: captureCall(() => reference.setIdentity({}, deferredScalar)),
    scalarReads,
  };
  const layouts = [
    auditCase(
      "default rectangular",
      2,
      3,
      reference.CV_8UC1,
      Array.from({ length: 6 }, () => 9),
      undefined,
      true,
    ),
    auditCase(
      "per-channel rectangular",
      2,
      3,
      reference.CV_16SC3,
      Array.from({ length: 18 }, () => 9),
      [11, -12, 13, 99],
    ),
    auditCase(
      "array-like scalar",
      2,
      2,
      reference.CV_32FC4,
      Array.from({ length: 16 }, () => 9),
      {
        0: 1.25,
        1: -2.5,
        2: 3.75,
        3: -4.5,
        length: 4,
      },
    ),
    auditCase(
      "boolean scalar",
      2,
      2,
      reference.CV_64FC4,
      Array.from({ length: 16 }, () => 9),
      [true, false, true, false],
    ),
    auditCase(
      "nonfinite scalar",
      2,
      2,
      reference.CV_64FC4,
      Array.from({ length: 16 }, () => 9),
      [Number.NaN, Infinity, -Infinity, -0],
    ),
  ];
  const integerSentinels = [
    ["U8", reference.CV_8UC4],
    ["I8", reference.CV_8SC4],
    ["U16", reference.CV_16UC4],
    ["I16", reference.CV_16SC4],
    ["I32", reference.CV_32SC4],
  ].map(([name, type]) =>
    auditCase(
      name,
      2,
      2,
      type,
      Array.from({ length: 16 }, () => 7),
      [Number.NaN, Infinity, 2_147_483_648, -2_147_483_649],
    ),
  );
  const canonicalEmpty = new reference.Mat();
  const canonicalEmptyAudit = auditTypedMatCall(
    () => reference.setIdentity(canonicalEmpty),
    [["matrix", canonicalEmpty]],
  );
  safeDelete(canonicalEmpty);
  const empty = [
    { name: "canonical", audit: canonicalEmptyAudit },
    auditCase("zero rows", 0, 3, reference.CV_32FC2, [], [1, 2, 3, 4]),
    auditCase("zero columns", 3, 0, reference.CV_64FC3, [], [1, 2, 3, 4]),
  ];
  const parent = makeSeedMat(
    reference,
    3,
    5,
    reference.CV_32FC2,
    Array.from({ length: 30 }, (_, index) => index + 1),
  );
  const roi = parent.roi(new reference.Rect(1, 0, 3, 3));
  const roiAudit = auditTypedMatCall(
    () => reference.setIdentity(roi, [2.5, -3.5, 0, 0]),
    [
      ["parent", parent],
      ["roi", roi],
    ],
  );
  safeDelete(roi);
  safeDelete(parent);
  return { arity, conversionOrder, layouts, integerSentinels, empty, roi: roiAudit };
}

function determinantDiagonal4(pivot) {
  return [pivot, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

function determinantNextDownF64(value) {
  const buffer = new ArrayBuffer(8);
  const view = new DataView(buffer);
  view.setFloat64(0, value, false);
  view.setBigUint64(0, view.getBigUint64(0, false) - 1n, false);
  return view.getFloat64(0, false);
}

function determinantNextDownF32(value) {
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setFloat32(0, value, false);
  view.setUint32(0, view.getUint32(0, false) - 1, false);
  return view.getFloat32(0, false);
}

function determinantHilbert(order) {
  return Array.from({ length: order * order }, (_, index) => {
    const row = Math.floor(index / order);
    const column = index % order;
    return 1 / (row + column + 1);
  });
}

function auditDeterminant(reference) {
  const createMat = (order, type, values) => makeSeedMat(reference, order, order, type, values);
  const auditMat = (name, order, type, values) => {
    const source = createMat(order, type, values);
    const audit = auditTypedMatCall(() => reference.determinant(source), [["source", source]]);
    safeDelete(source);
    return { name, audit };
  };
  const auditRejected = (name, rows, columns, type, values) => {
    const source = makeSeedMat(reference, rows, columns, type, values);
    const audit = auditTypedMatCall(() => reference.determinant(source), [["source", source]]);
    safeDelete(source);
    return { name, audit };
  };
  const aritySource = createMat(2, reference.CV_64FC1, [1, 2, 3, 4]);
  const arity = {
    length: reference.determinant.length,
    zero: captureCall(() => reference.determinant()),
    exact: captureCall(() => reference.determinant(aritySource)),
    extra: captureCall(() => reference.determinant(aritySource, 1)),
  };
  safeDelete(aritySource);

  const argumentTypes = {
    null: captureCall(() => reference.determinant(null)),
    undefined: captureCall(() => reference.determinant(undefined)),
    object: captureCall(() => reference.determinant({})),
    number: captureCall(() => reference.determinant(1)),
  };
  const deletedSource = createMat(1, reference.CV_64FC1, [7]);
  deletedSource.delete();
  const deleted = captureCall(() => reference.determinant(deletedSource));

  const continuous = [
    auditMat("F32 continuous", 2, reference.CV_32FC1, [1, 2, 3, 4]),
    auditMat("F64 continuous", 2, reference.CV_64FC1, [1, 2, 3, 4]),
  ];
  const strided = [
    ["F32 strided", reference.CV_32FC1],
    ["F64 strided", reference.CV_64FC1],
  ].map(([name, type]) => {
    const parent = makeSeedMat(reference, 2, 4, type, [99, 1, 2, 99, 99, 3, 4, 99]);
    const source = parent.roi(new reference.Rect(1, 0, 2, 2));
    const audit = auditTypedMatCall(
      () => reference.determinant(source),
      [
        ["source", source],
        ["parent", parent],
      ],
    );
    safeDelete(source);
    safeDelete(parent);
    return { name, audit };
  });

  const rejectedInputs = [
    ["U8", reference.CV_8UC1, [1, 2, 3, 4]],
    ["I8", reference.CV_8SC1, [1, 2, 3, 4]],
    ["U16", reference.CV_16UC1, [1, 2, 3, 4]],
    ["I16", reference.CV_16SC1, [1, 2, 3, 4]],
    ["I32", reference.CV_32SC1, [1, 2, 3, 4]],
  ].map(([name, type, values]) => auditRejected(name, 2, 2, type, values));
  rejectedInputs.push(
    auditRejected("F32 multichannel", 2, 2, reference.CV_32FC2, [1, 2, 3, 4, 5, 6, 7, 8]),
    auditRejected("F64 nonsquare", 2, 3, reference.CV_64FC1, [1, 2, 3, 4, 5, 6]),
  );

  const canonicalEmpty = new reference.Mat();
  const canonicalEmptyAudit = auditTypedMatCall(
    () => reference.determinant(canonicalEmpty),
    [["source", canonicalEmpty]],
  );
  safeDelete(canonicalEmpty);
  const empty = {
    canonical: { name: "canonical", audit: canonicalEmptyAudit },
    typedF32: auditRejected("typed F32", 0, 0, reference.CV_32FC1, []),
    typedF64: auditRejected("typed F64", 0, 0, reference.CV_64FC1, []),
  };

  const small = [
    auditMat("F64 1x1 negative zero", 1, reference.CV_64FC1, [-0]),
    auditMat("F64 1x1 NaN", 1, reference.CV_64FC1, [Number.NaN]),
    auditMat("F32 1x1 negative zero", 1, reference.CV_32FC1, [-0]),
    auditMat("F32 1x1 infinity", 1, reference.CV_32FC1, [Number.POSITIVE_INFINITY]),
    auditMat("F64 2x2 exact", 2, reference.CV_64FC1, [1, 2, 3, 4]),
    auditMat("F64 2x2 negative zero", 2, reference.CV_64FC1, [-0, 0, 0, 1]),
    auditMat("F64 2x2 nonfinite", 2, reference.CV_64FC1, [Number.POSITIVE_INFINITY, 0, 0, 0]),
    auditMat("F32 2x2 exact", 2, reference.CV_32FC1, [1, 2, 3, 4]),
    auditMat("F32 2x2 negative zero", 2, reference.CV_32FC1, [-0, 0, 0, 1]),
    auditMat("F32 2x2 nonfinite", 2, reference.CV_32FC1, [Number.POSITIVE_INFINITY, 0, 0, 0]),
    auditMat("F64 3x3 exact", 3, reference.CV_64FC1, [0, 2, 1, 1, 0, 3, 4, 5, 6]),
    auditMat("F64 3x3 signed zero", 3, reference.CV_64FC1, [-0, 0, 0, 0, 1, 0, 0, 0, 1]),
    auditMat("F64 3x3 nonfinite", 3, reference.CV_64FC1, [
      Number.POSITIVE_INFINITY,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1,
    ]),
    auditMat("F32 3x3 exact", 3, reference.CV_32FC1, [0, 2, 1, 1, 0, 3, 4, 5, 6]),
    auditMat("F32 3x3 signed zero", 3, reference.CV_32FC1, [-0, 0, 0, 0, 1, 0, 0, 0, 1]),
    auditMat("F32 3x3 nonfinite", 3, reference.CV_32FC1, [
      Number.POSITIVE_INFINITY,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1,
    ]),
  ];
  const widenedSmall = [
    auditMat(
      "F32 widened 2x2",
      2,
      reference.CV_32FC1,
      [16_777_216, 16_777_215, 16_777_215, 16_777_214],
    ),
    auditMat(
      "F32 widened 3x3",
      3,
      reference.CV_32FC1,
      [16_777_216, 16_777_215, 1, 16_777_215, 16_777_214, 2, 1, 2, 3],
    ),
  ];

  const f64Cutoff = 100 * Number.EPSILON;
  const f32Cutoff = 10 * 2 ** -23;
  const large = [
    auditMat("F64 cutoff exact", 4, reference.CV_64FC1, determinantDiagonal4(f64Cutoff)),
    auditMat(
      "F64 cutoff below",
      4,
      reference.CV_64FC1,
      determinantDiagonal4(determinantNextDownF64(f64Cutoff)),
    ),
    auditMat("F64 negative cutoff exact", 4, reference.CV_64FC1, determinantDiagonal4(-f64Cutoff)),
    auditMat(
      "F64 negative cutoff below",
      4,
      reference.CV_64FC1,
      determinantDiagonal4(-determinantNextDownF64(f64Cutoff)),
    ),
    auditMat("F32 cutoff exact", 4, reference.CV_32FC1, determinantDiagonal4(f32Cutoff)),
    auditMat(
      "F32 cutoff below",
      4,
      reference.CV_32FC1,
      determinantDiagonal4(determinantNextDownF32(f32Cutoff)),
    ),
    auditMat("F32 negative cutoff exact", 4, reference.CV_32FC1, determinantDiagonal4(-f32Cutoff)),
    auditMat(
      "F32 negative cutoff below",
      4,
      reference.CV_32FC1,
      determinantDiagonal4(-determinantNextDownF32(f32Cutoff)),
    ),
    auditMat(
      "F64 row swap",
      4,
      reference.CV_64FC1,
      [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3],
    ),
    auditMat(
      "F32 row swap",
      4,
      reference.CV_32FC1,
      [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3],
    ),
    auditMat(
      "F64 singular",
      4,
      reference.CV_64FC1,
      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ),
    auditMat(
      "F32 singular",
      4,
      reference.CV_32FC1,
      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ),
  ];

  const hilbert4 = determinantHilbert(4);
  const hilbert6 = determinantHilbert(6);
  const hilbert4F32 = Array.from(new Float32Array(hilbert4));
  const hilbert6F32 = Array.from(new Float32Array(hilbert6));
  const precision = [
    auditMat("F32 Hilbert4", 4, reference.CV_32FC1, hilbert4),
    auditMat("F32 Hilbert6", 6, reference.CV_32FC1, hilbert6),
    auditMat("F64 Hilbert4", 4, reference.CV_64FC1, hilbert4),
    auditMat("F64 Hilbert6", 6, reference.CV_64FC1, hilbert6),
    auditMat("F64 stored-F32 Hilbert4", 4, reference.CV_64FC1, hilbert4F32),
    auditMat("F64 stored-F32 Hilbert6", 6, reference.CV_64FC1, hilbert6F32),
  ];

  return {
    arity,
    argumentTypes,
    deleted,
    continuous,
    strided,
    rejectedInputs,
    empty,
    small,
    widenedSmall,
    large,
    precision,
  };
}

function auditContourContracts(reference) {
  const i32Rectangle = [0, 0, 4, 0, 4, 3, 0, 3];
  const f32Rectangle = [0.25, -1.5, 4.75, -1.5, 4.75, 2.25, 0.25, 2.25];
  const createMat = (rows, columns, type, values) =>
    reference.matFromArray(rows, columns, type, values);
  const captureBounds = (contour) =>
    captureCall(() => {
      const bounds = reference.boundingRect(contour);
      return [bounds.x, bounds.y, bounds.width, bounds.height];
    });
  const auditOperations = (contour) => ({
    arcOpen: captureCall(() => reference.arcLength(contour, false)),
    arcClosed: captureCall(() => reference.arcLength(contour, true)),
    area: captureCall(() => reference.contourArea(contour)),
    areaOriented: captureCall(() => reference.contourArea(contour, true)),
    bounds: captureBounds(contour),
  });
  const auditMat = (name, rows, columns, type, values) => {
    const contour = createMat(rows, columns, type, values);
    const result = { name, operations: auditOperations(contour) };
    safeDelete(contour);
    return result;
  };

  const arityContour = createMat(4, 1, reference.CV_32SC2, i32Rectangle);
  const arity = {
    lengths: {
      arcLength: reference.arcLength.length,
      contourArea: reference.contourArea.length,
      boundingRect: reference.boundingRect.length,
    },
    arcLength: {
      zero: captureCall(() => reference.arcLength()),
      one: captureCall(() => reference.arcLength(arityContour)),
      two: captureCall(() => reference.arcLength(arityContour, true)),
      three: captureCall(() => reference.arcLength(arityContour, true, 1)),
    },
    contourArea: {
      zero: captureCall(() => reference.contourArea()),
      one: captureCall(() => reference.contourArea(arityContour)),
      two: captureCall(() => reference.contourArea(arityContour, true)),
      three: captureCall(() => reference.contourArea(arityContour, true, 1)),
    },
    boundingRect: {
      zero: captureCall(() => reference.boundingRect()),
      one: captureBounds(arityContour),
      two: captureCall(() => reference.boundingRect(arityContour, 1)),
    },
  };
  safeDelete(arityContour);

  const truthyContour = createMat(4, 1, reference.CV_32SC2, i32Rectangle);
  const truthiness = [
    ["false", false],
    ["true", true],
    ["zero", 0],
    ["one", 1],
    ["empty string", ""],
    ["non-empty string", "closed"],
    ["null", null],
    ["explicit undefined", undefined],
  ].map(([label, value]) => ({
    label,
    input: encodeValue(value),
    arcLength: captureCall(() => reference.arcLength(truthyContour, value)),
    contourArea: captureCall(() => reference.contourArea(truthyContour, value)),
  }));
  safeDelete(truthyContour);

  const acceptedLayouts = [
    auditMat("I32 Nx1C2", 4, 1, reference.CV_32SC2, i32Rectangle),
    auditMat("I32 1xNC2", 1, 4, reference.CV_32SC2, i32Rectangle),
    auditMat("I32 Nx2C1", 4, 2, reference.CV_32SC1, i32Rectangle),
    auditMat("F32 Nx1C2", 4, 1, reference.CV_32FC2, f32Rectangle),
    auditMat("F32 1xNC2", 1, 4, reference.CV_32FC2, f32Rectangle),
    auditMat("F32 Nx2C1", 4, 2, reference.CV_32FC1, f32Rectangle),
  ];
  const rejectedInputs = [
    auditMat("F64 Nx1C2", 4, 1, reference.CV_64FC2, f32Rectangle),
    auditMat("U8 Nx1C2", 4, 1, reference.CV_8UC2, i32Rectangle),
    auditMat("I32 2x2C2", 2, 2, reference.CV_32SC2, i32Rectangle),
  ];

  const deletedContour = createMat(4, 1, reference.CV_32SC2, i32Rectangle);
  deletedContour.delete();
  const deleted = auditOperations(deletedContour);

  const emptyContour = new reference.Mat();
  const canonicalEmptyBoundingRect = captureBounds(emptyContour);
  safeDelete(emptyContour);

  return {
    arity,
    truthiness,
    acceptedLayouts,
    rejectedInputs,
    deleted,
    canonicalEmptyBoundingRect,
  };
}

function auditPolygonContracts(reference) {
  const i32Rectangle = [0, 0, 4, 0, 4, 3, 0, 3];
  const f32Rectangle = [0.25, -1.5, 4.75, -1.5, 4.75, 2.25, 0.25, 2.25];
  const createMat = (rows, columns, type, values) =>
    reference.matFromArray(rows, columns, type, values);
  const auditOperations = (contour, point = { x: 2, y: 1 }) => ({
    convex: captureCall(() => reference.isContourConvex(contour)),
    classify: captureCall(() => reference.pointPolygonTest(contour, point, false)),
    distance: captureCall(() => reference.pointPolygonTest(contour, point, true)),
  });
  const auditMat = (name, rows, columns, type, values, point) => {
    const contour = createMat(rows, columns, type, values);
    const operations = auditOperations(contour, point);
    safeDelete(contour);
    return { name, operations };
  };
  const auditConvexCase = (name, values) => {
    const contour = createMat(values.length / 2, 1, reference.CV_32SC2, values);
    const call = captureCall(() => reference.isContourConvex(contour));
    safeDelete(contour);
    return { name, call };
  };
  const auditQuery = (contour, label, x, y) => ({
    label,
    point: [encodeValue(x), encodeValue(y)],
    classify: captureCall(() => reference.pointPolygonTest(contour, { x, y }, false)),
    distance: captureCall(() => reference.pointPolygonTest(contour, { x, y }, true)),
  });

  const arityContour = createMat(4, 1, reference.CV_32SC2, i32Rectangle);
  const arityPoint = new reference.Point(2, 1);
  const arity = {
    lengths: {
      isContourConvex: reference.isContourConvex.length,
      pointPolygonTest: reference.pointPolygonTest.length,
    },
    isContourConvex: {
      zero: captureCall(() => reference.isContourConvex()),
      one: captureCall(() => reference.isContourConvex(arityContour)),
      two: captureCall(() => reference.isContourConvex(arityContour, 1)),
    },
    pointPolygonTest: {
      zero: captureCall(() => reference.pointPolygonTest()),
      one: captureCall(() => reference.pointPolygonTest(arityContour)),
      two: captureCall(() => reference.pointPolygonTest(arityContour, arityPoint)),
      three: captureCall(() => reference.pointPolygonTest(arityContour, arityPoint, false)),
      four: captureCall(() => reference.pointPolygonTest(arityContour, arityPoint, false, 1)),
    },
  };
  safeDelete(arityContour);

  const truthinessContour = createMat(4, 1, reference.CV_32SC2, i32Rectangle);
  const truthinessPoint = new reference.Point(2, 1.5);
  const truthiness = [
    ["false", false],
    ["true", true],
    ["zero", 0],
    ["one", 1],
    ["NaN", Number.NaN],
    ["empty string", ""],
    ["zero string", "0"],
    ["null", null],
    ["explicit undefined", undefined],
    ["object", {}],
  ].map(([label, value]) => ({
    label,
    input: encodeValue(value),
    call: captureCall(() => reference.pointPolygonTest(truthinessContour, truthinessPoint, value)),
  }));
  safeDelete(truthinessContour);

  const pointContour = createMat(4, 1, reference.CV_32SC2, i32Rectangle);
  const pointConversion = [
    ["Point2f", new reference.Point(2, 1.5)],
    ["plain", { x: 2, y: 1.5 }],
    ["array with fields", Object.assign([2, 1], { x: 2, y: 1.5 })],
    ["function with fields", Object.assign(() => undefined, { x: 2, y: 1.5 })],
    ["boolean x", { x: true, y: 1 }],
    ["numeric string", { x: "2", y: 1 }],
    ["missing y", { x: 2 }],
    ["array", [2, 1]],
    ["extra field", { x: 2, y: 1.5, z: 9 }],
  ].map(([name, point]) => ({
    name,
    call: captureCall(() => reference.pointPolygonTest(pointContour, point, true)),
  }));
  const nonFiniteQuery = {
    classify: captureCall(() =>
      reference.pointPolygonTest(pointContour, { x: Number.NaN, y: 1 }, false),
    ),
    distance: captureCall(() =>
      reference.pointPolygonTest(pointContour, { x: Number.NaN, y: 1 }, true),
    ),
  };
  safeDelete(pointContour);

  const largeContour = createMat(
    4,
    1,
    reference.CV_32SC2,
    [16_777_216, 0, 16_777_220, 0, 16_777_220, 4, 16_777_216, 4],
  );
  const float32PointBoundary = [
    16_777_216, 16_777_217, 16_777_218, 16_777_219, 16_777_220, 16_777_221,
  ].map((x) => auditQuery(largeContour, String(x), x, 2));
  safeDelete(largeContour);

  const acceptedLayouts = [
    auditMat("I32 Nx1C2", 4, 1, reference.CV_32SC2, i32Rectangle),
    auditMat("I32 1xNC2", 1, 4, reference.CV_32SC2, i32Rectangle),
    auditMat("I32 Nx2C1", 4, 2, reference.CV_32SC1, i32Rectangle),
    auditMat("F32 Nx1C2", 4, 1, reference.CV_32FC2, f32Rectangle),
    auditMat("F32 1xNC2", 1, 4, reference.CV_32FC2, f32Rectangle),
    auditMat("F32 Nx2C1", 4, 2, reference.CV_32FC1, f32Rectangle),
  ];
  const rejectedInputs = [
    auditMat("F64 Nx1C2", 4, 1, reference.CV_64FC2, f32Rectangle),
    auditMat("U8 Nx1C2", 4, 1, reference.CV_8UC2, i32Rectangle),
    auditMat("I32 2x2C2", 2, 2, reference.CV_32SC2, i32Rectangle),
  ];
  const stridedParent = createMat(
    4,
    2,
    reference.CV_32SC2,
    [99, 99, 0, 0, 99, 99, 4, 0, 99, 99, 4, 3, 99, 99, 0, 3],
  );
  const stridedContour = stridedParent.roi(new reference.Rect(1, 0, 1, 4));
  const strided = auditOperations(stridedContour);
  safeDelete(stridedContour);
  safeDelete(stridedParent);

  const canonicalEmpty = new reference.Mat();
  const empty = { canonical: auditOperations(canonicalEmpty) };
  safeDelete(canonicalEmpty);
  const typedEmpty = new reference.Mat(0, 0, reference.CV_32SC2);
  empty.typed = auditOperations(typedEmpty);
  safeDelete(typedEmpty);

  const deletedContour = createMat(4, 1, reference.CV_32SC2, i32Rectangle);
  deletedContour.delete();
  const deleted = auditOperations(deletedContour);

  const convexCases = [
    auditConvexCase("counter-clockwise rectangle", i32Rectangle),
    auditConvexCase("clockwise rectangle", [0, 0, 0, 3, 4, 3, 4, 0]),
    auditConvexCase("concave", [0, 0, 4, 0, 4, 4, 2, 2, 0, 4]),
    auditConvexCase("all collinear", [0, 0, 2, 0, 4, 0]),
    auditConvexCase("edge collinear", [0, 0, 2, 0, 4, 0, 4, 3, 0, 3]),
    auditConvexCase("adjacent duplicate", [0, 0, 4, 0, 4, 0, 4, 3, 0, 3]),
    auditConvexCase("repeated closing point", [0, 0, 4, 0, 4, 3, 0, 3, 0, 0]),
    auditConvexCase("self crossing", [0, 0, 4, 3, 0, 3, 4, 0]),
    auditConvexCase("one point", [0, 0]),
    auditConvexCase("two points", [0, 0, 4, 0]),
  ];

  const onePoint = createMat(1, 1, reference.CV_32SC2, [0, 0]);
  const twoPoints = createMat(2, 1, reference.CV_32SC2, [0, 0, 4, 0]);
  const smallContours = {
    one: [auditQuery(onePoint, "outside", 2, 1)],
    two: [auditQuery(twoPoints, "on segment", 2, 0), auditQuery(twoPoints, "off segment", 2, 1)],
  };
  safeDelete(twoPoints);
  safeDelete(onePoint);

  const rectangle = createMat(4, 1, reference.CV_32SC2, i32Rectangle);
  const clockwiseRectangle = createMat(4, 1, reference.CV_32SC2, [0, 0, 0, 3, 4, 3, 4, 0]);
  const concave = createMat(5, 1, reference.CV_32SC2, [0, 0, 4, 0, 4, 4, 2, 2, 0, 4]);
  const reverseConcave = createMat(5, 1, reference.CV_32SC2, [0, 4, 2, 2, 4, 4, 4, 0, 0, 0]);
  const classificationDistance = {
    rectangle: [
      auditQuery(rectangle, "inside", 2, 1.5),
      auditQuery(rectangle, "edge", 2, 0),
      auditQuery(rectangle, "vertex", 0, 0),
      auditQuery(rectangle, "outside", 5, 1),
      auditQuery(rectangle, "diagonal outside", -3, -4),
    ],
    clockwiseInside: auditQuery(clockwiseRectangle, "inside", 2, 1.5),
    concave: [
      auditQuery(concave, "inside", 2, 1),
      auditQuery(concave, "notch outside", 2, 3),
      auditQuery(concave, "notch boundary", 3, 3),
    ],
  };
  const rectangleBoundaryPoints = [
    ["bottom edge", 2, 0],
    ["right edge", 4, 1.5],
    ["top edge", 2, 3],
    ["left edge", 0, 1.5],
    ["bottom-left vertex", 0, 0],
    ["bottom-right vertex", 4, 0],
    ["top-right vertex", 4, 3],
    ["top-left vertex", 0, 3],
  ];
  const boundaryZeroSigns = {
    counterClockwiseRectangle: rectangleBoundaryPoints.map(([label, x, y]) =>
      auditQuery(rectangle, label, x, y),
    ),
    clockwiseRectangle: rectangleBoundaryPoints.map(([label, x, y]) =>
      auditQuery(clockwiseRectangle, label, x, y),
    ),
    concaveForward: [
      auditQuery(concave, "right notch diagonal", 3, 3),
      auditQuery(concave, "left notch diagonal", 1, 3),
      auditQuery(concave, "notch vertex", 2, 2),
    ],
    concaveReverse: [
      auditQuery(reverseConcave, "right notch diagonal", 3, 3),
      auditQuery(reverseConcave, "left notch diagonal", 1, 3),
      auditQuery(reverseConcave, "notch vertex", 2, 2),
    ],
  };
  safeDelete(reverseConcave);
  safeDelete(concave);
  safeDelete(clockwiseRectangle);
  safeDelete(rectangle);

  return {
    arity,
    truthiness,
    pointConversion,
    nonFiniteQuery,
    float32PointBoundary,
    acceptedLayouts,
    rejectedInputs,
    strided,
    empty,
    deleted,
    convexCases,
    smallContours,
    classificationDistance,
    boundaryZeroSigns,
  };
}

function auditSetterCases(createDetector, setter, getter, cases) {
  return cases.map(([label, value]) => {
    const detector = createDetector();
    const call = captureCall(() => detector[setter](value));
    const state = captureCall(() => detector[getter]());
    safeDelete(detector);
    return { label, input: encodeValue(value), call, state };
  });
}

function auditGftt(createDetector) {
  const getters = [
    "getBlockSize",
    "getDefaultName",
    "getHarrisDetector",
    "getK",
    "getMaxFeatures",
    "getMinDistance",
    "getQualityLevel",
  ];
  const setters = [
    ["setBlockSize", "getBlockSize", 7],
    ["setHarrisDetector", "getHarrisDetector", true],
    ["setK", "getK", 0.5],
    ["setMaxFeatures", "getMaxFeatures", 77],
    ["setMinDistance", "getMinDistance", 2.5],
    ["setQualityLevel", "getQualityLevel", 0.5],
  ];
  const i32Cases = [
    ["positive fraction", 1.9],
    ["negative fraction", -1.9],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["i32 maximum", 2_147_483_647],
    ["i32 maximum plus one", 2_147_483_648],
    ["i32 minimum", -2_147_483_648],
    ["i32 minimum minus one", -2_147_483_649],
    ["u32 maximum", 4_294_967_295],
    ["u32 modulus", 4_294_967_296],
    ["null", null],
    ["true", true],
    ["false", false],
    ["numeric string", "42"],
    ["fraction string", "-1.9"],
    ["empty string", ""],
    ["non-numeric string", "opencv"],
    ["explicit undefined", undefined],
  ];
  const f64Cases = [
    ["negative zero", -0],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["null", null],
    ["true", true],
    ["false", false],
    ["numeric string", "1.25"],
    ["empty string", ""],
    ["non-numeric string", "opencv"],
    ["explicit undefined", undefined],
  ];
  const booleanCases = [
    ["true", true],
    ["false", false],
    ["zero", 0],
    ["one", 1],
    ["negative one", -1],
    ["two", 2],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["null", null],
    ["empty string", ""],
    ["zero string", "0"],
    ["false string", "false"],
    ["non-empty string", "opencv"],
    ["object", {}],
    ["explicit undefined", undefined],
  ];

  const arityDetector = createDetector();
  const arity = {
    getters: getters.map((method) => ({
      method,
      length: arityDetector[method].length,
      exact: captureCall(() => arityDetector[method]()),
      extraOne: captureCall(() => arityDetector[method](1)),
      extraTwo: captureCall(() => arityDetector[method](1, 2)),
    })),
    setters: setters.map(([method, getter, value]) => ({
      method,
      length: arityDetector[method].length,
      missing: captureCall(() => arityDetector[method]()),
      exact: captureCall(() => arityDetector[method](value)),
      stateAfterExact: captureCall(() => arityDetector[getter]()),
      extraOne: captureCall(() => arityDetector[method](value, 1)),
      extraTwo: captureCall(() => arityDetector[method](value, 1, 2)),
    })),
  };
  safeDelete(arityDetector);

  const deadDetector = createDetector();
  const firstDelete = captureCall(() => deadDetector.delete());
  const postDelete = {
    getters: getters.map((method) => ({
      method,
      call: captureCall(() => deadDetector[method]()),
    })),
    setters: setters.map(([method, , value]) => ({
      method,
      call: captureCall(() => deadDetector[method](value)),
    })),
    secondDelete: captureCall(() => deadDetector.delete()),
  };

  const deleteExtraOneDetector = createDetector();
  const deleteExtraOne = captureCall(() => deleteExtraOneDetector.delete(1));
  safeDelete(deleteExtraOneDetector);
  const deleteExtraTwoDetector = createDetector();
  const deleteExtraTwo = captureCall(() => deleteExtraTwoDetector.delete(1, 2));
  safeDelete(deleteExtraTwoDetector);
  const deleteLengthDetector = createDetector();
  const deleteLength = deleteLengthDetector.delete.length;
  safeDelete(deleteLengthDetector);

  return {
    arity,
    i32: {
      blockSize: auditSetterCases(createDetector, "setBlockSize", "getBlockSize", i32Cases),
      maxFeatures: auditSetterCases(createDetector, "setMaxFeatures", "getMaxFeatures", i32Cases),
    },
    f64: {
      k: auditSetterCases(createDetector, "setK", "getK", f64Cases),
      minDistance: auditSetterCases(createDetector, "setMinDistance", "getMinDistance", f64Cases),
      qualityLevel: auditSetterCases(
        createDetector,
        "setQualityLevel",
        "getQualityLevel",
        f64Cases,
      ),
    },
    boolean: auditSetterCases(
      createDetector,
      "setHarrisDetector",
      "getHarrisDetector",
      booleanCases,
    ),
    lifetime: {
      deleteLength,
      firstDelete,
      postDelete,
      deleteExtraOne,
      deleteExtraTwo,
    },
  };
}

function auditAkazePrimitive(createDetector) {
  const getters = [
    "getDefaultName",
    "getDescriptorChannels",
    "getDescriptorSize",
    "getNOctaveLayers",
    "getNOctaves",
    "getThreshold",
  ];
  const setters = [
    ["setDescriptorChannels", 2],
    ["setDescriptorSize", 96],
    ["setNOctaveLayers", 6],
    ["setNOctaves", 5],
    ["setThreshold", 0.25],
  ];
  const i32Cases = [
    ["positive fraction", 1.9],
    ["negative fraction", -1.9],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["i32 maximum", 2_147_483_647],
    ["i32 maximum plus one", 2_147_483_648],
    ["i32 minimum", -2_147_483_648],
    ["i32 minimum minus one", -2_147_483_649],
    ["u32 maximum", 4_294_967_295],
    ["u32 modulus", 4_294_967_296],
    ["null", null],
    ["true", true],
    ["false", false],
    ["numeric string", "42"],
    ["fraction string", "-1.9"],
    ["empty string", ""],
    ["non-numeric string", "opencv"],
    ["explicit undefined", undefined],
  ];
  const f64Cases = [
    ["negative zero", -0],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["null", null],
    ["true", true],
    ["false", false],
    ["numeric string", "1.25"],
    ["empty string", ""],
    ["non-numeric string", "opencv"],
    ["explicit undefined", undefined],
  ];

  const captureState = (detector) =>
    Object.fromEntries(getters.map((method) => [method, captureCall(() => detector[method]())]));
  const defaultsDetector = createDetector();
  const defaults = captureState(defaultsDetector);
  safeDelete(defaultsDetector);

  const getterArity = getters.map((method) => {
    const detector = createDetector();
    const audit = {
      method,
      length: detector[method].length,
      exact: captureCall(() => detector[method]()),
      extraOne: {
        stateBefore: captureState(detector),
        call: captureCall(() => detector[method](1)),
        stateAfter: captureState(detector),
      },
      extraTwo: {
        stateBefore: captureState(detector),
        call: captureCall(() => detector[method](1, 2)),
        stateAfter: captureState(detector),
      },
    };
    safeDelete(detector);
    return audit;
  });
  const setterArity = setters.map(([method, value]) => {
    const lengthDetector = createDetector();
    const length = lengthDetector[method].length;
    safeDelete(lengthDetector);
    const exactDetector = createDetector();
    const stateBeforeExact = captureState(exactDetector);
    const exact = captureCall(() => exactDetector[method](value));
    const stateAfterExact = captureState(exactDetector);
    safeDelete(exactDetector);

    const auditFailure = (invoke) => {
      const detector = createDetector();
      const stateBefore = captureState(detector);
      const call = captureCall(() => invoke(detector, method, value));
      const stateAfter = captureState(detector);
      safeDelete(detector);
      return { stateBefore, call, stateAfter };
    };

    return {
      method,
      length,
      exact: { stateBefore: stateBeforeExact, call: exact, stateAfter: stateAfterExact },
      missing: auditFailure((detector, setter) => detector[setter]()),
      extraOne: auditFailure((detector, setter, argument) => detector[setter](argument, 1)),
      extraTwo: auditFailure((detector, setter, argument) => detector[setter](argument, 1, 2)),
    };
  });

  const auditCases = (setter, cases) =>
    cases.map(([label, value]) => {
      const detector = createDetector();
      const stateBefore = captureState(detector);
      const call = captureCall(() => detector[setter](value));
      const stateAfter = captureState(detector);
      safeDelete(detector);
      return { label, input: encodeValue(value), stateBefore, call, stateAfter };
    });

  const deadDetector = createDetector();
  const deleteLength = captureCall(() => deadDetector.delete.length);
  const stateBeforeDelete = captureState(deadDetector);
  const firstDelete = captureCall(() => deadDetector.delete());
  const postDelete = {
    getters: getters.map((method) => ({
      method,
      call: captureCall(() => deadDetector[method]()),
    })),
    setters: setters.map(([method, value]) => ({
      method,
      call: captureCall(() => deadDetector[method](value)),
    })),
    secondDelete: captureCall(() => deadDetector.delete()),
  };

  const auditDeleteExtra = (...arguments_) => {
    const detector = createDetector();
    const stateBefore = captureState(detector);
    const call = captureCall(() => detector.delete(...arguments_));
    const stateAfter = captureState(detector);
    safeDelete(detector);
    return { stateBefore, call, stateAfter };
  };

  return {
    defaults,
    arity: { getters: getterArity, setters: setterArity },
    i32: {
      descriptorChannels: auditCases("setDescriptorChannels", i32Cases),
      descriptorSize: auditCases("setDescriptorSize", i32Cases),
      octaveLayers: auditCases("setNOctaveLayers", i32Cases),
      octaves: auditCases("setNOctaves", i32Cases),
    },
    f64: { threshold: auditCases("setThreshold", f64Cases) },
    lifetime: {
      deleteLength,
      stateBeforeDelete,
      firstDelete,
      postDelete,
      deleteExtraOne: auditDeleteExtra(1),
      deleteExtraTwo: auditDeleteExtra(1, 2),
    },
  };
}

function auditKazePrimitive(createDetector) {
  const getters = [
    "getDefaultName",
    "getExtended",
    "getNOctaveLayers",
    "getNOctaves",
    "getThreshold",
    "getUpright",
  ];
  const setters = [
    ["setExtended", true],
    ["setNOctaveLayers", 6],
    ["setNOctaves", 5],
    ["setThreshold", 0.25],
    ["setUpright", true],
  ];
  const i32Cases = [
    ["positive fraction", 1.9],
    ["negative fraction", -1.9],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["i32 maximum", 2_147_483_647],
    ["i32 maximum plus one", 2_147_483_648],
    ["i32 minimum", -2_147_483_648],
    ["i32 minimum minus one", -2_147_483_649],
    ["u32 maximum", 4_294_967_295],
    ["u32 modulus", 4_294_967_296],
    ["null", null],
    ["true", true],
    ["false", false],
    ["numeric string", "42"],
    ["fraction string", "-1.9"],
    ["empty string", ""],
    ["non-numeric string", "opencv"],
    ["explicit undefined", undefined],
  ];
  const f64Cases = [
    ["negative zero", -0],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["null", null],
    ["true", true],
    ["false", false],
    ["numeric string", "1.25"],
    ["empty string", ""],
    ["non-numeric string", "opencv"],
    ["explicit undefined", undefined],
  ];
  const booleanCases = [
    ["true", true],
    ["false", false],
    ["zero", 0],
    ["one", 1],
    ["negative one", -1],
    ["two", 2],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["null", null],
    ["empty string", ""],
    ["zero string", "0"],
    ["false string", "false"],
    ["non-empty string", "opencv"],
    ["object", {}],
    ["explicit undefined", undefined],
  ];

  const captureState = (detector) =>
    Object.fromEntries(getters.map((method) => [method, captureCall(() => detector[method]())]));
  const defaultsDetector = createDetector();
  const defaults = captureState(defaultsDetector);
  safeDelete(defaultsDetector);

  const getterArity = getters.map((method) => {
    const detector = createDetector();
    const audit = {
      method,
      length: detector[method].length,
      exact: captureCall(() => detector[method]()),
      extraOne: {
        stateBefore: captureState(detector),
        call: captureCall(() => detector[method](1)),
        stateAfter: captureState(detector),
      },
      extraTwo: {
        stateBefore: captureState(detector),
        call: captureCall(() => detector[method](1, 2)),
        stateAfter: captureState(detector),
      },
    };
    safeDelete(detector);
    return audit;
  });
  const setterArity = setters.map(([method, value]) => {
    const lengthDetector = createDetector();
    const length = lengthDetector[method].length;
    safeDelete(lengthDetector);
    const exactDetector = createDetector();
    const stateBeforeExact = captureState(exactDetector);
    const exact = captureCall(() => exactDetector[method](value));
    const stateAfterExact = captureState(exactDetector);
    safeDelete(exactDetector);

    const auditFailure = (invoke) => {
      const detector = createDetector();
      const stateBefore = captureState(detector);
      const call = captureCall(() => invoke(detector, method, value));
      const stateAfter = captureState(detector);
      safeDelete(detector);
      return { stateBefore, call, stateAfter };
    };

    return {
      method,
      length,
      exact: { stateBefore: stateBeforeExact, call: exact, stateAfter: stateAfterExact },
      missing: auditFailure((detector, setter) => detector[setter]()),
      extraOne: auditFailure((detector, setter, argument) => detector[setter](argument, 1)),
      extraTwo: auditFailure((detector, setter, argument) => detector[setter](argument, 1, 2)),
    };
  });

  const auditCases = (setter, cases) =>
    cases.map(([label, value]) => {
      const detector = createDetector();
      const stateBefore = captureState(detector);
      const call = captureCall(() => detector[setter](value));
      const stateAfter = captureState(detector);
      safeDelete(detector);
      return { label, input: encodeValue(value), stateBefore, call, stateAfter };
    });

  const deadDetector = createDetector();
  const deleteLength = captureCall(() => deadDetector.delete.length);
  const stateBeforeDelete = captureState(deadDetector);
  const firstDelete = captureCall(() => deadDetector.delete());
  const postDelete = {
    getters: getters.map((method) => ({
      method,
      call: captureCall(() => deadDetector[method]()),
    })),
    setters: setters.map(([method, value]) => ({
      method,
      call: captureCall(() => deadDetector[method](value)),
    })),
    secondDelete: captureCall(() => deadDetector.delete()),
  };

  const auditDeleteExtra = (...arguments_) => {
    const detector = createDetector();
    const stateBefore = captureState(detector);
    const call = captureCall(() => detector.delete(...arguments_));
    const stateAfter = captureState(detector);
    safeDelete(detector);
    return { stateBefore, call, stateAfter };
  };

  return {
    defaults,
    arity: { getters: getterArity, setters: setterArity },
    boolean: {
      extended: auditCases("setExtended", booleanCases),
      upright: auditCases("setUpright", booleanCases),
    },
    i32: {
      octaveLayers: auditCases("setNOctaveLayers", i32Cases),
      octaves: auditCases("setNOctaves", i32Cases),
    },
    f64: { threshold: auditCases("setThreshold", f64Cases) },
    lifetime: {
      deleteLength,
      stateBeforeDelete,
      firstDelete,
      postDelete,
      deleteExtraOne: auditDeleteExtra(1),
      deleteExtraTwo: auditDeleteExtra(1, 2),
    },
  };
}

function auditOrb(createDetector, namespace, surface) {
  const getters = ["getDefaultName", "getFastThreshold"];
  const setters = [
    ["setEdgeThreshold", 17],
    ["setFastThreshold", 37],
    ["setFirstLevel", 2],
    ["setMaxFeatures", 1000],
    ["setNLevels", 12],
    ["setPatchSize", 45],
    ["setScaleFactor", 1.5],
    ["setScoreType", namespace.ORB_ScoreType.FAST_SCORE],
    ["setWTA_K", 4],
  ];
  const i32Cases = [
    ["positive fraction", 1.9],
    ["negative fraction", -1.9],
    ["negative zero", -0],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["i32 maximum", 2_147_483_647],
    ["i32 maximum plus one", 2_147_483_648],
    ["i32 minimum", -2_147_483_648],
    ["i32 minimum minus one", -2_147_483_649],
    ["true", true],
    ["false", false],
    ["null", null],
    ["numeric string", "42"],
    ["object", {}],
    ["explicit undefined", undefined],
  ];
  const f64Cases = [
    ["positive fraction", 1.9],
    ["negative fraction", -1.9],
    ["negative zero", -0],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["true", true],
    ["false", false],
    ["null", null],
    ["numeric string", "1.5"],
    ["object", {}],
    ["explicit undefined", undefined],
  ];

  const defaultsDetector = createDetector();
  const defaults = {
    defaultName: captureCall(() => defaultsDetector.getDefaultName()),
    fastThreshold: captureCall(() => defaultsDetector.getFastThreshold()),
  };
  safeDelete(defaultsDetector);

  const getterArity = getters.map((method) => {
    const detector = createDetector();
    const audit = {
      method,
      length: detector[method].length,
      exact: captureCall(() => detector[method]()),
      extraOne: captureCall(() => detector[method](1)),
      extraTwo: captureCall(() => detector[method](1, 2)),
    };
    safeDelete(detector);
    return audit;
  });
  const setterArity = setters.map(([method, value]) => {
    const detector = createDetector();
    const audit = {
      method,
      length: detector[method].length,
      exact: captureCall(() => detector[method](value)),
      missing: captureCall(() => detector[method]()),
      extraOne: captureCall(() => detector[method](value, 1)),
      extraTwo: captureCall(() => detector[method](value, 1, 2)),
    };
    safeDelete(detector);
    return audit;
  });

  const auditCalls = (method, cases) =>
    cases.map(([label, value]) => {
      const detector = createDetector();
      const call = captureCall(() => detector[method](value));
      safeDelete(detector);
      return { label, input: encodeValue(value), call };
    });
  const inheritedScore = Object.create({ value: 1 });
  const scoreCases = [
    ["HARRIS singleton", namespace.ORB_ScoreType.HARRIS_SCORE],
    ["FAST singleton", namespace.ORB_ScoreType.FAST_SCORE],
    ["plain fractional value", { value: 19.9 }],
    ["inherited value", inheritedScore],
    ["empty object", {}],
    ["number primitive", 1],
    ["string primitive", "1"],
    ["boolean primitive", true],
    ["null", null],
    ["explicit undefined", undefined],
  ];

  const deadDetector = createDetector();
  const deleteLength = deadDetector.delete.length;
  const firstDelete = captureCall(() => deadDetector.delete());
  const postDelete = {
    getters: getters.map((method) => ({
      method,
      call: captureCall(() => deadDetector[method]()),
    })),
    setters: setters.map(([method, value]) => ({
      method,
      call: captureCall(() => deadDetector[method](value)),
    })),
    secondDelete: captureCall(() => deadDetector.delete()),
  };
  const auditDeleteExtra = (...arguments_) => {
    const detector = createDetector();
    const call = captureCall(() => detector.delete(...arguments_));
    safeDelete(detector);
    return call;
  };

  const scoreNamespace = namespace.ORB_ScoreType;
  return {
    surface,
    defaults,
    arity: { getters: getterArity, setters: setterArity },
    i32: auditSetterCases(createDetector, "setFastThreshold", "getFastThreshold", i32Cases),
    firstLevel: auditCalls("setFirstLevel", [
      ["positive fraction", 1.9],
      ["negative fraction", -1.9],
      ["negative integer", -1],
    ]),
    f64: auditCalls("setScaleFactor", f64Cases),
    score: auditCalls("setScoreType", scoreCases),
    enums: {
      keys: Object.keys(scoreNamespace),
      harrisValue: scoreNamespace.HARRIS_SCORE.value,
      fastValue: scoreNamespace.FAST_SCORE.value,
      harrisConstructorName: scoreNamespace.HARRIS_SCORE.constructor.name,
      fastConstructorName: scoreNamespace.FAST_SCORE.constructor.name,
      harrisValueIdentity: scoreNamespace.values[0] === scoreNamespace.HARRIS_SCORE,
      fastValueIdentity: scoreNamespace.values[1] === scoreNamespace.FAST_SCORE,
      globals: [namespace.ORB_HARRIS_SCORE, namespace.ORB_FAST_SCORE],
      globalTypes: [typeof namespace.ORB_HARRIS_SCORE, typeof namespace.ORB_FAST_SCORE],
      singletonIsGlobal: [
        scoreNamespace.HARRIS_SCORE === namespace.ORB_HARRIS_SCORE,
        scoreNamespace.FAST_SCORE === namespace.ORB_FAST_SCORE,
      ],
    },
    lifetime: {
      deleteLength,
      firstDelete,
      postDelete,
      deleteExtraOne: auditDeleteExtra(1),
      deleteExtraTwo: auditDeleteExtra(1, 2),
    },
  };
}

function auditMser(createDetector, surface) {
  const getters = ["getDefaultName", "getDelta", "getMaxArea", "getMinArea", "getPass2Only"];
  const setters = [
    ["setDelta", "getDelta", 7],
    ["setMaxArea", "getMaxArea", 14_401],
    ["setMinArea", "getMinArea", 61],
    ["setPass2Only", "getPass2Only", true],
  ];
  const i32Cases = [
    ["positive fraction", 1.9],
    ["negative fraction", -1.9],
    ["negative zero", -0],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["i32 maximum", 2_147_483_647],
    ["i32 maximum plus one", 2_147_483_648],
    ["i32 minimum", -2_147_483_648],
    ["i32 minimum minus one", -2_147_483_649],
    ["u32 maximum", 4_294_967_295],
    ["u32 modulus", 4_294_967_296],
    ["null", null],
    ["true", true],
    ["false", false],
    ["numeric string", "42"],
    ["fraction string", "-1.9"],
    ["empty string", ""],
    ["non-numeric string", "opencv"],
    ["object", {}],
    ["explicit undefined", undefined],
  ];
  const booleanCases = [
    ["true", true],
    ["false", false],
    ["zero", 0],
    ["one", 1],
    ["negative one", -1],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["null", null],
    ["empty string", ""],
    ["zero string", "0"],
    ["false string", "false"],
    ["object", {}],
    ["explicit undefined", undefined],
  ];
  const captureState = (detector) =>
    Object.fromEntries(getters.map((method) => [method, captureCall(() => detector[method]())]));

  const defaultsDetector = createDetector();
  const defaults = captureState(defaultsDetector);
  safeDelete(defaultsDetector);

  const getterArity = getters.map((method) => {
    const detector = createDetector();
    const audit = {
      method,
      length: detector[method].length,
      exact: captureCall(() => detector[method]()),
      extraOne: captureCall(() => detector[method](1)),
      extraTwo: captureCall(() => detector[method](1, 2)),
    };
    safeDelete(detector);
    return audit;
  });
  const setterArity = setters.map(([method, getter, value]) => {
    const detector = createDetector();
    const audit = {
      method,
      length: detector[method].length,
      missing: captureCall(() => detector[method]()),
      exact: captureCall(() => detector[method](value)),
      stateAfterExact: captureCall(() => detector[getter]()),
      extraOne: captureCall(() => detector[method](value, 1)),
      extraTwo: captureCall(() => detector[method](value, 1, 2)),
    };
    safeDelete(detector);
    return audit;
  });

  const deadDetector = createDetector();
  const deleteLength = deadDetector.delete.length;
  const firstDelete = captureCall(() => deadDetector.delete());
  const postDelete = {
    getters: getters.map((method) => ({
      method,
      call: captureCall(() => deadDetector[method]()),
    })),
    setters: setters.map(([method, , value]) => ({
      method,
      call: captureCall(() => deadDetector[method](value)),
    })),
    secondDelete: captureCall(() => deadDetector.delete()),
  };
  const auditDeleteExtra = (...arguments_) => {
    const detector = createDetector();
    const call = captureCall(() => detector.delete(...arguments_));
    safeDelete(detector);
    return call;
  };

  return {
    surface,
    defaults,
    arity: { getters: getterArity, setters: setterArity },
    i32: {
      delta: auditSetterCases(createDetector, "setDelta", "getDelta", i32Cases),
      maxArea: auditSetterCases(createDetector, "setMaxArea", "getMaxArea", i32Cases),
      minArea: auditSetterCases(createDetector, "setMinArea", "getMinArea", i32Cases),
    },
    boolean: auditSetterCases(createDetector, "setPass2Only", "getPass2Only", booleanCases),
    lifetime: {
      deleteLength,
      firstDelete,
      postDelete,
      deleteExtraOne: auditDeleteExtra(1),
      deleteExtraTwo: auditDeleteExtra(1, 2),
    },
  };
}

function auditTonemap(factories, surface) {
  const floatCases = [
    ["negative", -2.25],
    ["negative zero", -0],
    ["zero", 0],
    ["fraction", 0.123456789],
    ["tiny", 1e-40],
    ["huge", 1e40],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["true", true],
    ["false", false],
    ["null", null],
    ["numeric string", "2.5"],
    ["empty string", ""],
    ["array", [2.5]],
    ["object", {}],
    ["explicit undefined", undefined],
  ];
  const definitions = {
    drago: {
      create: factories.drago,
      getters: ["getGamma", "getBias", "getSaturation"],
      setters: [
        ["setGamma", "getGamma", 0.25],
        ["setBias", "getBias", 0.5],
        ["setSaturation", "getSaturation", 1.5],
      ],
      maximumArguments: 3,
    },
    mantiuk: {
      create: factories.mantiuk,
      getters: ["getGamma", "getScale", "getSaturation"],
      setters: [
        ["setGamma", "getGamma", 0.25],
        ["setScale", "getScale", 0.5],
        ["setSaturation", "getSaturation", 1.5],
      ],
      maximumArguments: 3,
    },
    reinhard: {
      create: factories.reinhard,
      getters: ["getGamma", "getIntensity", "getLightAdaptation", "getColorAdaptation"],
      setters: [
        ["setGamma", "getGamma", 0.25],
        ["setIntensity", "getIntensity", -1.5],
        ["setLightAdaptation", "getLightAdaptation", 0.5],
        ["setColorAdaptation", "getColorAdaptation", 0.75],
      ],
      maximumArguments: 4,
    },
  };

  const auditFamily = ({ create, getters, setters, maximumArguments }) => {
    const captureState = (tonemap) =>
      Object.fromEntries(getters.map((method) => [method, captureCall(() => tonemap[method]())]));

    const defaultsTonemap = create();
    const defaults = captureState(defaultsTonemap);
    safeDelete(defaultsTonemap);

    const constructorArities = Array.from({ length: maximumArguments + 3 }, (_, argumentCount) => {
      let tonemap;
      const call = captureCall(() => {
        tonemap = create(...[0.25, 1.25, 2.25, 3.25, 4.25, 5.25].slice(0, argumentCount));
      });
      const state = tonemap === undefined ? null : captureState(tonemap);
      safeDelete(tonemap);
      return { argumentCount, call, state };
    });

    const getterArity = getters.map((method) => {
      const tonemap = create();
      const audit = {
        method,
        length: tonemap[method].length,
        exact: captureCall(() => tonemap[method]()),
        extraOne: captureCall(() => tonemap[method](1)),
        extraTwo: captureCall(() => tonemap[method](1, 2)),
      };
      safeDelete(tonemap);
      return audit;
    });

    const setterArity = setters.map(([method, getter, value]) => {
      const tonemap = create();
      const audit = {
        method,
        length: tonemap[method].length,
        missing: captureCall(() => tonemap[method]()),
        exact: captureCall(() => tonemap[method](value)),
        stateAfterExact: captureCall(() => tonemap[getter]()),
        extraOne: captureCall(() => tonemap[method](value, 1)),
        extraTwo: captureCall(() => tonemap[method](value, 1, 2)),
      };
      safeDelete(tonemap);
      return audit;
    });

    const state = Object.fromEntries(
      setters.map(([setter, getter]) => [
        setter,
        auditSetterCases(create, setter, getter, floatCases),
      ]),
    );

    const deadTonemap = create();
    const deleteLength = deadTonemap.delete.length;
    const firstDelete = captureCall(() => deadTonemap.delete());
    const postDelete = {
      getters: getters.map((method) => ({
        method,
        call: captureCall(() => deadTonemap[method]()),
      })),
      setters: setters.map(([method, , value]) => ({
        method,
        call: captureCall(() => deadTonemap[method](value)),
      })),
      secondDelete: captureCall(() => deadTonemap.delete()),
    };
    const auditDeleteExtra = (...arguments_) => {
      const tonemap = create();
      const call = captureCall(() => tonemap.delete(...arguments_));
      safeDelete(tonemap);
      return call;
    };

    return {
      defaults,
      constructorArities,
      arity: { getters: getterArity, setters: setterArity },
      state,
      lifetime: {
        deleteLength,
        firstDelete,
        postDelete,
        deleteExtraOne: auditDeleteExtra(1),
        deleteExtraTwo: auditDeleteExtra(1, 2),
      },
    };
  };

  return {
    surface,
    drago: auditFamily(definitions.drago),
    mantiuk: auditFamily(definitions.mantiuk),
    reinhard: auditFamily(definitions.reinhard),
  };
}

function auditReferenceTonemap(reference) {
  const drago = new reference.TonemapDrago();
  const surface = {
    constructorLengths: {
      drago: reference.TonemapDrago.length,
      mantiuk: reference.TonemapMantiuk.length,
      reinhard: reference.TonemapReinhard.length,
    },
    factoryTypes: {
      drago: typeof reference.createTonemapDrago,
      mantiuk: typeof reference.createTonemapMantiuk,
      reinhard: typeof reference.createTonemapReinhard,
    },
    dragoSigmaTypes: {
      getSigmaColor: typeof drago.getSigmaColor,
      getSigmaSpace: typeof drago.getSigmaSpace,
      setSigmaColor: typeof drago.setSigmaColor,
      setSigmaSpace: typeof drago.setSigmaSpace,
    },
    processType: typeof drago.process,
  };
  safeDelete(drago);
  return auditTonemap(
    {
      drago: (...arguments_) => new reference.TonemapDrago(...arguments_),
      mantiuk: (...arguments_) => new reference.TonemapMantiuk(...arguments_),
      reinhard: (...arguments_) => new reference.TonemapReinhard(...arguments_),
    },
    surface,
  );
}

function auditThresholdDetector(createDetector) {
  const getters = ["getDefaultName", "getNonmaxSuppression", "getThreshold"];
  const setters = [
    ["setNonmaxSuppression", "getNonmaxSuppression", false],
    ["setThreshold", "getThreshold", 37],
  ];
  const i32Cases = [
    ["positive fraction", 1.9],
    ["negative fraction", -1.9],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["i32 maximum", 2_147_483_647],
    ["i32 maximum plus one", 2_147_483_648],
    ["i32 minimum", -2_147_483_648],
    ["i32 minimum minus one", -2_147_483_649],
    ["u32 maximum", 4_294_967_295],
    ["u32 modulus", 4_294_967_296],
    ["null", null],
    ["true", true],
    ["false", false],
    ["numeric string", "42"],
    ["fraction string", "-1.9"],
    ["empty string", ""],
    ["non-numeric string", "opencv"],
    ["explicit undefined", undefined],
  ];
  const booleanCases = [
    ["true", true],
    ["false", false],
    ["zero", 0],
    ["one", 1],
    ["negative one", -1],
    ["two", 2],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["negative infinity", Number.NEGATIVE_INFINITY],
    ["null", null],
    ["empty string", ""],
    ["zero string", "0"],
    ["false string", "false"],
    ["non-empty string", "opencv"],
    ["object", {}],
    ["explicit undefined", undefined],
  ];

  const defaultsDetector = createDetector();
  const defaults = Object.fromEntries(
    getters.map((method) => [method, captureCall(() => defaultsDetector[method]())]),
  );
  safeDelete(defaultsDetector);

  const getterArity = getters.map((method) => {
    const detector = createDetector();
    const audit = {
      method,
      length: detector[method].length,
      exact: captureCall(() => detector[method]()),
      extraOne: captureCall(() => detector[method](1)),
      stateAfterExtraOne: captureCall(() => detector[method]()),
      extraTwo: captureCall(() => detector[method](1, 2)),
      stateAfterExtraTwo: captureCall(() => detector[method]()),
    };
    safeDelete(detector);
    return audit;
  });
  const setterArity = setters.map(([method, getter, value]) => {
    const lengthDetector = createDetector();
    const length = lengthDetector[method].length;
    safeDelete(lengthDetector);
    const exactDetector = createDetector();
    const exact = captureCall(() => exactDetector[method](value));
    const stateAfterExact = captureCall(() => exactDetector[getter]());
    safeDelete(exactDetector);

    const auditFailure = (invoke) => {
      const detector = createDetector();
      const stateBefore = captureCall(() => detector[getter]());
      const call = captureCall(() => invoke(detector, method, value));
      const stateAfter = captureCall(() => detector[getter]());
      safeDelete(detector);
      return { stateBefore, call, stateAfter };
    };

    return {
      method,
      length,
      exact,
      stateAfterExact,
      missing: auditFailure((detector, setter) => detector[setter]()),
      extraOne: auditFailure((detector, setter, argument) => detector[setter](argument, 1)),
      extraTwo: auditFailure((detector, setter, argument) => detector[setter](argument, 1, 2)),
    };
  });

  const deadDetector = createDetector();
  const firstDelete = captureCall(() => deadDetector.delete());
  const postDelete = {
    getters: getters.map((method) => ({
      method,
      call: captureCall(() => deadDetector[method]()),
    })),
    setters: setters.map(([method, , value]) => ({
      method,
      call: captureCall(() => deadDetector[method](value)),
    })),
    secondDelete: captureCall(() => deadDetector.delete()),
  };

  const auditDeleteExtra = (...arguments_) => {
    const detector = createDetector();
    const stateBefore = captureCall(() => detector.getThreshold());
    const call = captureCall(() => detector.delete(...arguments_));
    const stateAfter = captureCall(() => detector.getThreshold());
    safeDelete(detector);
    return { stateBefore, call, stateAfter };
  };
  const deleteLengthDetector = createDetector();
  const deleteLength = captureCall(() => deleteLengthDetector.delete.length);
  safeDelete(deleteLengthDetector);

  return {
    defaults,
    arity: { getters: getterArity, setters: setterArity },
    i32: auditSetterCases(createDetector, "setThreshold", "getThreshold", i32Cases),
    boolean: auditSetterCases(
      createDetector,
      "setNonmaxSuppression",
      "getNonmaxSuppression",
      booleanCases,
    ),
    lifetime: {
      deleteLength,
      firstDelete,
      postDelete,
      deleteExtraOne: auditDeleteExtra(1),
      deleteExtraTwo: auditDeleteExtra(1, 2),
    },
  };
}

async function loadOpenCv() {
  try {
    importScripts("/test/browser/.cache/opencv-4.13.0.js");
  } catch {
    importScripts("https://docs.opencv.org/4.13.0/opencv.js");
  }
  self.postMessage({
    progress: "OpenCV.js script loaded; initializing WASM...",
  });
  if (cv.then !== undefined) {
    return new Promise((resolve) => {
      cv.then((ready) => {
        delete ready.then;
        resolve(ready);
      });
    });
  }
  if (cv.Mat !== undefined) {
    return cv;
  }
  return new Promise((resolve) => {
    const checkReady = () => {
      if (cv.Mat !== undefined) {
        resolve(cv);
        return;
      }
      setTimeout(checkReady, 10);
    };
    checkReady();
  });
}

function auditCvtColor(reference) {
  const typeByChannels = [
    reference.CV_8UC1,
    reference.CV_8UC2,
    reference.CV_8UC3,
    reference.CV_8UC4,
  ];
  const cases = [
    ["BGR2BGRA", reference.COLOR_BGR2BGRA, 3, [1, 2, 3, 4, 5, 6], 0],
    ["BGRA2BGR", reference.COLOR_BGRA2BGR, 4, [1, 2, 3, 4, 5, 6, 7, 8], 0],
    ["BGR2RGBA", reference.COLOR_BGR2RGBA, 3, [1, 2, 3, 4, 5, 6], 0],
    ["RGBA2BGR", reference.COLOR_RGBA2BGR, 4, [1, 2, 3, 4, 5, 6, 7, 8], 0],
    ["BGR2RGB", reference.COLOR_BGR2RGB, 3, [1, 2, 3, 4, 5, 6], 0],
    ["BGRA2RGBA", reference.COLOR_BGRA2RGBA, 4, [1, 2, 3, 4, 5, 6, 7, 8], 0],
    ["BGR2GRAY", reference.COLOR_BGR2GRAY, 3, [255, 0, 0, 0, 255, 0, 0, 0, 255], 0],
    ["RGB2GRAY", reference.COLOR_RGB2GRAY, 3, [255, 0, 0, 0, 255, 0, 0, 0, 255], 0],
    ["GRAY2BGR", reference.COLOR_GRAY2BGR, 1, [1, 255], 0],
    ["GRAY2BGRA", reference.COLOR_GRAY2BGRA, 1, [1, 255], 0],
    ["BGRA2GRAY", reference.COLOR_BGRA2GRAY, 4, [255, 0, 0, 7, 0, 255, 0, 8, 0, 0, 255, 9], 0],
    ["RGBA2GRAY", reference.COLOR_RGBA2GRAY, 4, [255, 0, 0, 7, 0, 255, 0, 8, 0, 0, 255, 9], 0],
    ["RGB2RGBA-dcn3", reference.COLOR_RGB2RGBA, 3, [1, 2, 3], 3],
    ["GRAY2BGR-dcn4", reference.COLOR_GRAY2BGR, 1, [9], 4],
  ];
  const depthCases = [
    ["u16-BGR2GRAY", reference.COLOR_BGR2GRAY, 3, [2570, 5140, 7710, 1, 2, 3], 0],
    ["u16-RGB2GRAY", reference.COLOR_RGB2GRAY, 3, [65535, 1, 100, 40000, 30000, 20000], 0],
    [
      "u16-BGRA2GRAY",
      reference.COLOR_BGRA2GRAY,
      4,
      [100, 200, 300, 400, 65535, 65535, 65535, 128],
      0,
    ],
    ["u16-BGR2BGRA", reference.COLOR_BGR2BGRA, 3, [1, 2, 3], 0],
    ["u16-BGRA2RGBA", reference.COLOR_BGRA2RGBA, 4, [1, 2, 3, 4], 0],
    ["u16-GRAY2BGRA", reference.COLOR_GRAY2BGRA, 1, [42, 65535], 0],
    ["f32-BGR2GRAY", reference.COLOR_BGR2GRAY, 3, [0.1, 0.5, 0.9, 1, 1, 1], 0],
    ["f32-RGB2GRAY", reference.COLOR_RGB2GRAY, 3, [0.25, 0.5, 0.75], 0],
    ["f32-BGRA2GRAY", reference.COLOR_BGRA2GRAY, 4, [0.1, 0.2, 0.3, 0.4], 0],
    ["f32-BGR2RGBA", reference.COLOR_BGR2RGBA, 3, [0.1, 0.2, 0.3], 0],
    ["f32-GRAY2BGR", reference.COLOR_GRAY2BGR, 1, [0.5], 0],
    ["f32-BGR2BGRA-dcn3", reference.COLOR_BGR2BGRA, 3, [0.4, 0.5, 0.6], 3],
  ];
  const inPlace = reference.matFromArray(1, 2, reference.CV_8UC4, [255, 0, 0, 7, 1, 2, 3, 4]);
  reference.cvtColor(inPlace, inPlace, reference.COLOR_RGBA2GRAY);
  const inPlaceOutput = summarizeTypedMat(inPlace);
  inPlace.delete();

  const parent = reference.matFromArray(
    2,
    3,
    reference.CV_8UC4,
    Array.from({ length: 24 }, (_, index) => index + 1),
  );
  const region = parent.roi(new reference.Rect(1, 0, 2, 2));
  const regionDestination = new reference.Mat();
  reference.cvtColor(region, regionDestination, reference.COLOR_RGBA2BGRA);
  const regionOutput = summarizeTypedMat(regionDestination);
  regionDestination.delete();
  region.delete();
  parent.delete();

  return {
    constants: [
      reference.COLOR_BGR2BGRA,
      reference.COLOR_BGRA2BGR,
      reference.COLOR_BGR2RGBA,
      reference.COLOR_RGBA2BGR,
      reference.COLOR_BGR2RGB,
      reference.COLOR_BGRA2RGBA,
      reference.COLOR_BGR2GRAY,
      reference.COLOR_RGB2GRAY,
      reference.COLOR_GRAY2BGR,
      reference.COLOR_GRAY2BGRA,
      reference.COLOR_BGRA2GRAY,
      reference.COLOR_RGBA2GRAY,
    ],
    cases: cases.map(([name, code, channels, values, destinationChannels]) => {
      const source = reference.matFromArray(
        1,
        values.length / channels,
        typeByChannels[channels - 1],
        values,
      );
      const destination = new reference.Mat();
      reference.cvtColor(source, destination, code, destinationChannels);
      const output = summarizeTypedMat(destination);
      source.delete();
      destination.delete();
      return { name, output };
    }),
    depthCases: depthCases.map(([name, code, channels, values, destinationChannels]) => {
      const depthType = name.startsWith("u16-")
        ? [reference.CV_16UC1, reference.CV_16UC2, reference.CV_16UC3, reference.CV_16UC4]
        : [reference.CV_32FC1, reference.CV_32FC2, reference.CV_32FC3, reference.CV_32FC4];
      const source = reference.matFromArray(
        1,
        values.length / channels,
        depthType[channels - 1],
        values,
      );
      const destination = new reference.Mat();
      reference.cvtColor(source, destination, code, destinationChannels);
      const output = summarizeTypedMat(destination);
      source.delete();
      destination.delete();
      return { name, output };
    }),
    inPlaceOutput,
    regionOutput,
  };
}

function auditResize(reference) {
  const cases = [
    ["nearest", 2, 2, [1, 2, 3, 4], 4, 2, 0, 0, reference.INTER_NEAREST],
    ["linear-default", 2, 2, [0, 100, 150, 255], 3, 3, 0, 0, reference.INTER_LINEAR],
    [
      "area",
      4,
      4,
      [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
      2,
      2,
      0,
      0,
      reference.INTER_AREA,
    ],
    ["scale", 2, 2, [1, 2, 3, 4], 0, 0, 2, 0.5, reference.INTER_NEAREST],
  ];
  return {
    constants: [reference.INTER_NEAREST, reference.INTER_LINEAR, reference.INTER_AREA],
    cases: cases.map(
      ([name, sourceWidth, sourceHeight, values, width, height, scaleX, scaleY, interpolation]) => {
        const source = reference.matFromArray(sourceHeight, sourceWidth, reference.CV_8UC1, values);
        const destination = new reference.Mat();
        reference.resize(
          source,
          destination,
          new reference.Size(width, height),
          scaleX,
          scaleY,
          interpolation,
        );
        const output = summarizeTypedMat(destination);
        source.delete();
        destination.delete();
        return { name, output };
      },
    ),
  };
}

function auditThreshold(reference) {
  const cases = [
    ["binary", 100, 200, reference.THRESH_BINARY],
    ["binary-inverse", 100, 200, reference.THRESH_BINARY_INV],
    ["truncate", 100, 200, reference.THRESH_TRUNC],
    ["to-zero", 100, 200, reference.THRESH_TOZERO],
    ["to-zero-inverse", 100, 200, reference.THRESH_TOZERO_INV],
    ["otsu", 0, 255, reference.THRESH_BINARY | reference.THRESH_OTSU],
  ];
  return {
    constants: [
      reference.THRESH_BINARY,
      reference.THRESH_BINARY_INV,
      reference.THRESH_TRUNC,
      reference.THRESH_TOZERO,
      reference.THRESH_TOZERO_INV,
      reference.THRESH_MASK,
      reference.THRESH_OTSU,
      reference.THRESH_TRIANGLE,
    ],
    cases: cases.map(([name, threshold, maximum, type]) => {
      const values = name === "otsu" ? [10, 10, 10, 200, 200, 200] : [0, 99, 100, 101, 255];
      const source = reference.matFromArray(1, values.length, reference.CV_8UC1, values);
      const destination = new reference.Mat();
      const usedThreshold = reference.threshold(source, destination, threshold, maximum, type);
      const output = summarizeTypedMat(destination);
      source.delete();
      destination.delete();
      return { name, usedThreshold, output };
    }),
  };
}

function auditNeighborhoodFilters(reference) {
  const impulse = reference.matFromArray(1, 5, reference.CV_8UC1, [0, 0, 255, 0, 0]);
  const blurred = new reference.Mat();
  reference.GaussianBlur(
    impulse,
    blurred,
    new reference.Size(3, 1),
    0,
    0,
    reference.BORDER_CONSTANT,
  );

  const mask = reference.matFromArray(1, 5, reference.CV_8UC1, [0, 255, 255, 255, 0]);
  const kernel = reference.matFromArray(1, 3, reference.CV_8UC1, [1, 1, 1]);
  const eroded = new reference.Mat();
  const dilated = new reference.Mat();
  reference.morphologyEx(mask, eroded, reference.MORPH_ERODE, kernel);
  reference.morphologyEx(mask, dilated, reference.MORPH_DILATE, kernel);

  const ramp = reference.matFromArray(3, 3, reference.CV_8UC1, [0, 10, 20, 0, 10, 20, 0, 10, 20]);
  const gradient = new reference.Mat();
  reference.Sobel(ramp, gradient, reference.CV_16S, 1, 0, 3, 1, 0, reference.BORDER_CONSTANT);

  const step = reference.matFromArray(
    5,
    5,
    reference.CV_8UC1,
    [
      0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255,
      255, 255,
    ],
  );
  const edges = new reference.Mat();
  reference.Canny(step, edges, 50, 100, 3, false);

  const output = {
    constants: [
      reference.BORDER_CONSTANT,
      reference.BORDER_DEFAULT,
      reference.MORPH_ERODE,
      reference.MORPH_DILATE,
    ],
    blurred: summarizeTypedMat(blurred),
    eroded: summarizeTypedMat(eroded),
    dilated: summarizeTypedMat(dilated),
    gradient: summarizeTypedMat(gradient),
    edges: summarizeTypedMat(edges),
  };
  impulse.delete();
  blurred.delete();
  mask.delete();
  kernel.delete();
  eroded.delete();
  dilated.delete();
  ramp.delete();
  gradient.delete();
  step.delete();
  edges.delete();
  return output;
}

function auditFindContours(reference) {
  const source = reference.matFromArray(
    5,
    5,
    reference.CV_8UC1,
    [0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0],
  );
  const contours = new reference.MatVector();
  const hierarchy = new reference.Mat();
  reference.findContours(
    source,
    contours,
    hierarchy,
    reference.RETR_EXTERNAL,
    reference.CHAIN_APPROX_SIMPLE,
  );
  const values = Array.from({ length: contours.size() }, (_, index) => {
    const contour = contours.get(index);
    const summary = summarizeTypedMat(contour);
    contour.delete();
    return summary;
  });
  const output = {
    constants: [reference.RETR_EXTERNAL, reference.CHAIN_APPROX_SIMPLE],
    contours: values,
    hierarchy: summarizeTypedMat(hierarchy),
  };
  source.delete();
  contours.delete();
  hierarchy.delete();
  return output;
}

function auditWarpAndEqualize(reference) {
  const source = reference.matFromArray(2, 3, reference.CV_8UC1, [1, 2, 3, 4, 5, 6]);
  const transform = reference.matFromArray(2, 3, reference.CV_64FC1, [1, 0, 1, 0, 1, 0]);
  const warped = new reference.Mat();
  reference.warpAffine(
    source,
    warped,
    transform,
    new reference.Size(3, 2),
    reference.INTER_NEAREST,
    reference.BORDER_CONSTANT,
    new reference.Scalar(9, 0, 0, 0),
  );

  const histogramSource = reference.matFromArray(1, 8, reference.CV_8UC1, [0, 0, 1, 1, 2, 3, 3, 3]);
  const equalized = new reference.Mat();
  reference.equalizeHist(histogramSource, equalized);
  const output = {
    constants: [reference.WARP_INVERSE_MAP, reference.WARP_FILL_OUTLIERS],
    warped: summarizeTypedMat(warped),
    equalized: summarizeTypedMat(equalized),
  };
  source.delete();
  transform.delete();
  warped.delete();
  histogramSource.delete();
  equalized.delete();
  return output;
}

self.addEventListener("message", async ({ data: input }) => {
  try {
    const reference = await loadOpenCv();
    self.postMessage({ progress: "OpenCV.js WASM initialized; comparing outputs..." });
    const request = input?.referenceRequest;
    const sourceInput = request === undefined ? input : input.input;
    if (request === "transpose") {
      self.postMessage({ outputs: { transposeAudit: auditTranspose(reference) } });
      return;
    }
    if (request === "flip") {
      self.postMessage({ outputs: { flipAudit: auditFlip(reference) } });
      return;
    }
    if (request === "repeat") {
      self.postMessage({ outputs: { repeatAudit: auditRepeat(reference) } });
      return;
    }
    if (request === "rotate") {
      self.postMessage({ outputs: { rotateAudit: auditRotate(reference) } });
      return;
    }
    if (request === "count-non-zero") {
      self.postMessage({ outputs: { countNonZeroAudit: auditCountNonZero(reference) } });
      return;
    }
    if (request === "float-math") {
      self.postMessage({ outputs: { floatMathAudit: auditFloatMath(reference) } });
      return;
    }
    if (request === "coordinate-conversions") {
      self.postMessage({
        outputs: { coordinateAudit: auditCoordinateConversions(reference) },
      });
      return;
    }
    if (request === "numeric-contracts") {
      self.postMessage({ outputs: { numericAudit: auditNumericContracts(reference) } });
      return;
    }
    if (request === "contour-contracts") {
      self.postMessage({ outputs: { contourAudit: auditContourContracts(reference) } });
      return;
    }
    if (request === "polygon-contracts") {
      self.postMessage({ outputs: { polygonAudit: auditPolygonContracts(reference) } });
      return;
    }
    if (request === "rotation-matrix-contracts") {
      self.postMessage({ outputs: { rotationMatrixAudit: auditRotationMatrix(reference) } });
      return;
    }
    if (request === "determinant-contracts") {
      self.postMessage({ outputs: { determinantAudit: auditDeterminant(reference) } });
      return;
    }
    if (request === "set-identity-contracts") {
      self.postMessage({ outputs: { setIdentityAudit: auditSetIdentity(reference) } });
      return;
    }
    if (request === "affine-transform-contracts") {
      self.postMessage({ outputs: { affineTransformAudit: auditAffineTransform(reference) } });
      return;
    }
    if (request === "invert-affine-transform-contracts") {
      self.postMessage({
        outputs: { invertAffineTransformAudit: auditInvertAffineTransform(reference) },
      });
      return;
    }
    if (request === "structuring-element-contracts") {
      self.postMessage({
        outputs: { structuringElementAudit: auditStructuringElement(reference) },
      });
      return;
    }
    if (request === "hanning-window-contracts") {
      self.postMessage({
        outputs: { hanningWindowAudit: auditHanningWindow(reference) },
      });
      return;
    }
    if (request === "mean-contracts") {
      self.postMessage({ outputs: { meanAudit: auditMaskedReducer(reference, "mean") } });
      return;
    }
    if (request === "min-max-loc-contracts") {
      self.postMessage({
        outputs: { minMaxLocAudit: auditMaskedReducer(reference, "minMaxLoc") },
      });
      return;
    }
    if (request === "trace-contracts") {
      self.postMessage({ outputs: { traceAudit: auditTrace(reference) } });
      return;
    }
    if (request === "bitwise-not-contracts") {
      self.postMessage({ outputs: { bitwiseNotAudit: auditBitwiseNot(reference) } });
      return;
    }
    if (request === "orb-contracts") {
      self.postMessage({
        outputs: {
          orbAudit: auditOrb(() => new reference.ORB(), reference, {
            constructorLength: reference.ORB.length,
            staticCreatePresent: typeof reference.ORB.create !== "undefined",
          }),
        },
      });
      return;
    }
    if (request === "mser-contracts") {
      self.postMessage({
        outputs: {
          mserAudit: auditMser(() => new reference.MSER(), {
            constructorLength: reference.MSER.length,
            constructorArities: Array.from({ length: 11 }, (_, count) =>
              captureCall(() => {
                const detector = Reflect.construct(
                  reference.MSER,
                  [5, 60, 14_400, 0.25, 0.2, 200, 1.01, 0.003, 5, 1].slice(0, count),
                );
                detector.delete();
              }),
            ),
            staticCreatePresent: typeof reference.MSER.create !== "undefined",
          }),
        },
      });
      return;
    }
    if (request === "tonemap-contracts") {
      self.postMessage({ outputs: { tonemapAudit: auditReferenceTonemap(reference) } });
      return;
    }
    const source = reference.matFromArray(2, 3, reference.CV_8UC1, sourceInput);
    const outputs = {};
    const operations = [
      ["flip", (target) => reference.flip(source, target, 1)],
      ["transpose", (target) => reference.transpose(source, target)],
      ["rotate", (target) => reference.rotate(source, target, reference.ROTATE_90_CLOCKWISE)],
      ["repeat", (target) => reference.repeat(source, 2, 1, target)],
    ];
    for (const [name, operation] of operations) {
      const target = new reference.Mat();
      operation(target);
      outputs[name] = copyBytes(target);
      target.delete();
    }
    source.delete();

    const contour = reference.matFromArray(
      4,
      1,
      reference.CV_32SC2,
      new Int32Array([0, 0, 4, 0, 4, 3, 0, 3]),
    );
    const bounds = reference.boundingRect(contour);
    outputs.arcLength = reference.arcLength(contour, true);
    outputs.contourArea = reference.contourArea(contour, false);
    outputs.boundingRect = [bounds.x, bounds.y, bounds.width, bounds.height];
    outputs.isContourConvex = reference.isContourConvex(contour);
    outputs.pointPolygonTest = reference.pointPolygonTest(contour, new reference.Point(2, 1), true);
    contour.delete();

    const kernel = reference.getStructuringElement(
      reference.MORPH_CROSS,
      new reference.Size(3, 3),
      new reference.Point(1, 1),
    );
    outputs.getStructuringElement = copyBytes(kernel);
    kernel.delete();

    const rotation = reference.getRotationMatrix2D(new reference.Point(1, 2), 90, 1);
    outputs.getRotationMatrix2D = copyF64(rotation);
    rotation.delete();
    outputs.rotationMatrixAudit = auditRotationMatrix(reference);

    outputs.optimalDftSizes = [
      reference.getOptimalDFTSize(7),
      reference.getOptimalDFTSize(25),
      reference.getOptimalDFTSize(0),
      reference.getOptimalDFTSize(-1),
      reference.getOptimalDFTSize(2_125_763_999),
      reference.getOptimalDFTSize(2_125_764_000),
    ];
    outputs.optimalDftSizeAudit = auditOptimalDftSize(reference);

    const akaze = new reference.AKAZE();
    outputs.akazeDefaultName = akaze.getDefaultName();
    outputs.akazeDefaults = [
      akaze.getDescriptorType().value,
      akaze.getDescriptorSize(),
      akaze.getDescriptorChannels(),
      akaze.getThreshold(),
      akaze.getNOctaves(),
      akaze.getNOctaveLayers(),
      akaze.getDiffusivity().value,
    ];
    akaze.setDescriptorType(reference.AKAZE_DescriptorType.DESCRIPTOR_MLDB_UPRIGHT);
    akaze.setDescriptorSize(96);
    akaze.setDescriptorChannels(2);
    akaze.setThreshold(0.05);
    akaze.setNOctaves(5);
    akaze.setNOctaveLayers(6);
    akaze.setDiffusivity(reference.KAZE_DiffusivityType.DIFF_WEICKERT);
    outputs.akazeMutated = [
      akaze.getDescriptorType().value,
      akaze.getDescriptorSize(),
      akaze.getDescriptorChannels(),
      akaze.getThreshold(),
      akaze.getNOctaves(),
      akaze.getNOctaveLayers(),
      akaze.getDiffusivity().value,
    ];
    akaze.delete();
    const agast = new reference.AgastFeatureDetector();
    const fast = new reference.FastFeatureDetector();
    const agastType = agast.getType();
    const fastType = fast.getType();
    outputs.agastDefaultName = agast.getDefaultName();
    outputs.agastDefaults = [agast.getNonmaxSuppression(), agast.getThreshold(), agastType.value];
    agast.setNonmaxSuppression(false);
    agast.setThreshold(-1);
    agast.setType(reference.AgastFeatureDetector_DetectorType.AGAST_5_8);
    outputs.agastMutated = [
      agast.getNonmaxSuppression(),
      agast.getThreshold(),
      agast.getType().value,
    ];
    outputs.fastDefaultName = fast.getDefaultName();
    outputs.fastDefaults = [fast.getNonmaxSuppression(), fast.getThreshold(), fastType.value];
    fast.setNonmaxSuppression(false);
    fast.setThreshold(256);
    fast.setType(reference.FastFeatureDetector_DetectorType.TYPE_5_8);
    outputs.fastMutated = [fast.getNonmaxSuppression(), fast.getThreshold(), fast.getType().value];
    agast.delete();
    fast.delete();

    const kaze = new reference.KAZE();
    outputs.kazeDefaultName = kaze.getDefaultName();
    outputs.kazeDefaults = [
      kaze.getDiffusivity().value,
      kaze.getExtended(),
      kaze.getNOctaveLayers(),
      kaze.getNOctaves(),
      kaze.getThreshold(),
      kaze.getUpright(),
    ];
    kaze.setDiffusivity(reference.KAZE_DiffusivityType.DIFF_WEICKERT);
    kaze.setExtended(true);
    kaze.setNOctaveLayers(6);
    kaze.setNOctaves(5);
    kaze.setThreshold(-1);
    kaze.setUpright(true);
    outputs.kazeMutated = [
      kaze.getDiffusivity().value,
      kaze.getExtended(),
      kaze.getNOctaveLayers(),
      kaze.getNOctaves(),
      kaze.getThreshold(),
      kaze.getUpright(),
    ];
    kaze.delete();
    const gftt = new reference.GFTTDetector();
    outputs.gfttDefaultName = gftt.getDefaultName();
    outputs.gfttDefaults = [
      gftt.getBlockSize(),
      gftt.getHarrisDetector(),
      gftt.getK(),
      gftt.getMaxFeatures(),
      gftt.getMinDistance(),
      gftt.getQualityLevel(),
    ];
    gftt.setBlockSize(-1);
    gftt.setHarrisDetector(true);
    gftt.setK(-1);
    gftt.setMaxFeatures(-1);
    gftt.setMinDistance(-1);
    gftt.setQualityLevel(-1);
    outputs.gfttMutated = [
      gftt.getBlockSize(),
      gftt.getHarrisDetector(),
      gftt.getK(),
      gftt.getMaxFeatures(),
      gftt.getMinDistance(),
      gftt.getQualityLevel(),
    ];
    gftt.setK(Number.NaN);
    gftt.setMinDistance(Number.NEGATIVE_INFINITY);
    gftt.setQualityLevel(Number.POSITIVE_INFINITY);
    outputs.gfttNonFinite = [
      Number.isNaN(gftt.getK()),
      gftt.getMinDistance() === Number.NEGATIVE_INFINITY,
      gftt.getQualityLevel() === Number.POSITIVE_INFINITY,
    ];
    gftt.delete();

    outputs.gfttAudit = auditGftt(() => new reference.GFTTDetector());
    outputs.akazePrimitiveAudit = auditAkazePrimitive(() => new reference.AKAZE());
    outputs.detectorEnumAudit = auditDetectorEnums(reference);
    outputs.kazePrimitiveAudit = auditKazePrimitive(() => new reference.KAZE());
    outputs.agastPrimitiveAudit = auditThresholdDetector(
      () => new reference.AgastFeatureDetector(),
    );
    outputs.fastPrimitiveAudit = auditThresholdDetector(() => new reference.FastFeatureDetector());
    outputs.flipAudit = auditFlip(reference);
    outputs.repeatAudit = auditRepeat(reference);
    outputs.rotateAudit = auditRotate(reference);
    outputs.transposeAudit = auditTranspose(reference);
    outputs.countNonZeroAudit = auditCountNonZero(reference);
    outputs.floatMathAudit = auditFloatMath(reference);
    outputs.coordinateAudit = auditCoordinateConversions(reference);
    outputs.numericAudit = auditNumericContracts(reference);
    outputs.contourAudit = auditContourContracts(reference);
    outputs.polygonAudit = auditPolygonContracts(reference);
    outputs.determinantAudit = auditDeterminant(reference);
    outputs.setIdentityAudit = auditSetIdentity(reference);
    outputs.affineTransformAudit = auditAffineTransform(reference);
    outputs.invertAffineTransformAudit = auditInvertAffineTransform(reference);
    outputs.structuringElementAudit = auditStructuringElement(reference);
    outputs.hanningWindowAudit = auditHanningWindow(reference);
    outputs.meanAudit = auditMaskedReducer(reference, "mean");
    outputs.minMaxLocAudit = auditMaskedReducer(reference, "minMaxLoc");
    outputs.traceAudit = auditTrace(reference);
    outputs.bitwiseNotAudit = auditBitwiseNot(reference);
    outputs.cvtColorAudit = auditCvtColor(reference);
    outputs.resizeAudit = auditResize(reference);
    outputs.thresholdAudit = auditThreshold(reference);
    outputs.neighborhoodFilterAudit = auditNeighborhoodFilters(reference);
    outputs.findContoursAudit = auditFindContours(reference);
    outputs.warpAndEqualizeAudit = auditWarpAndEqualize(reference);
    outputs.orbAudit = auditOrb(() => new reference.ORB(), reference, {
      constructorLength: reference.ORB.length,
      staticCreatePresent: typeof reference.ORB.create !== "undefined",
    });
    outputs.mserAudit = auditMser(() => new reference.MSER(), {
      constructorLength: reference.MSER.length,
      staticCreatePresent: typeof reference.MSER.create !== "undefined",
    });
    outputs.tonemapAudit = auditReferenceTonemap(reference);
    self.postMessage({ outputs });
  } catch (error) {
    self.postMessage({ error: String(error) });
  }
});
