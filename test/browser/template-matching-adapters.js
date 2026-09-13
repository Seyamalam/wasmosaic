function constants(cv) {
  return [
    cv.TM_SQDIFF,
    cv.TM_SQDIFF_NORMED,
    cv.TM_CCORR,
    cv.TM_CCORR_NORMED,
    cv.TM_CCOEFF,
    cv.TM_CCOEFF_NORMED,
  ];
}

export function packageTemplateApi(cv) {
  return {
    constants: constants(cv),
    functionLength: cv.matchTemplate.length,
    mat(rows, columns, channels, depth, values) {
      if (depth === 0) return cv.matFromU8(rows, columns, channels, new Uint8Array(values));
      if (depth === 3) return cv.matFromI16(rows, columns, channels, new Int16Array(values));
      return cv.matFromF32(rows, columns, channels, new Float32Array(values));
    },
    empty: () => cv.emptyMat(),
    roi: (matrix, row, column, rows, columns) => matrix.roi(row, column, rows, columns),
    dispose: (matrix) => matrix.dispose(),
    match: (...args) => cv.matchTemplate(...args),
    minMaxLoc: (matrix) => cv.minMaxLoc(matrix),
    summary(matrix) {
      return {
        rows: matrix.rows,
        columns: matrix.columns,
        channels: matrix.channels,
        depth: matrix.depth === "f32" ? 5 : matrix.depth === "i16" ? 3 : 0,
        values: Array.from(
          matrix.depth === "f32"
            ? matrix.toFloat32Array()
            : matrix.depth === "i16"
              ? matrix.toInt16Array()
              : matrix.toUint8Array(),
        ),
      };
    },
  };
}

export function referenceTemplateApi(cv) {
  return {
    constants: constants(cv),
    functionLength: cv.matchTemplate.length,
    mat: (rows, columns, channels, depth, values) =>
      cv.matFromArray(rows, columns, depth + ((channels - 1) << 3), values),
    empty: () => new cv.Mat(),
    roi: (matrix, row, column, rows, columns) =>
      matrix.roi(new cv.Rect(column, row, columns, rows)),
    dispose: (matrix) => matrix.delete(),
    match: (...args) => cv.matchTemplate(...args),
    minMaxLoc: (matrix) => cv.minMaxLoc(matrix),
    summary(matrix) {
      const pointer =
        matrix.depth() === 5 ? "floatPtr" : matrix.depth() === 3 ? "shortPtr" : "ucharPtr";
      return {
        rows: matrix.rows,
        columns: matrix.cols,
        channels: matrix.channels(),
        depth: matrix.depth(),
        values: Array.from({ length: matrix.rows }, (_, row) =>
          Array.from(matrix[pointer](row).subarray(0, matrix.cols * matrix.channels())),
        ).flat(),
      };
    },
  };
}
