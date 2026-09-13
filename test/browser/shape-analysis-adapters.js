export function contourApi(cv, reference) {
  const depthCode = (mat) => (reference ? mat.depth() : { u8: 0, i32: 4, f32: 5 }[mat.depth]);
  return {
    contract: () => ({
      approxLength: cv.approxPolyDP.length,
      findLength: cv.findContours.length,
      modes: [cv.RETR_EXTERNAL, cv.RETR_LIST, cv.RETR_CCOMP, cv.RETR_TREE],
      methods: [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE],
    }),
    mat(rows, columns, channels, depth, values) {
      if (reference) return cv.matFromArray(rows, columns, depth + ((channels - 1) << 3), values);
      if (depth === 0) return cv.matFromU8(rows, columns, channels, new Uint8Array(values));
      if (depth === 4) return cv.matFromI32(rows, columns, channels, new Int32Array(values));
      return cv.matFromF32(rows, columns, channels, new Float32Array(values));
    },
    empty: () => (reference ? new cv.Mat() : cv.emptyMat()),
    vector: () => (reference ? new cv.MatVector() : cv.createMatVector()),
    size: (vector) => vector.size(),
    get: (vector, index) => vector.get(index),
    dispose: (mat) => (reference ? mat.delete() : mat.dispose()),
    roi: (mat, row, column, rows, columns) =>
      reference
        ? mat.roi(new cv.Rect(column, row, columns, rows))
        : mat.roi(row, column, rows, columns),
    find: (...args) => cv.findContours(...args),
    arcLength: (...args) => cv.arcLength(...args),
    area: (...args) => cv.contourArea(...args),
    bounds: (...args) => cv.boundingRect(...args),
    approx: (...args) => cv.approxPolyDP(...args),
    summary(mat) {
      const depth = depthCode(mat),
        rows = mat.rows,
        columns = reference ? mat.cols : mat.columns,
        channels = reference ? mat.channels() : mat.channels;
      const values = reference
        ? Array.from({ length: rows }, (_, row) =>
            Array.from(
              mat[depth === 4 ? "intPtr" : depth === 5 ? "floatPtr" : "ucharPtr"](row).subarray(
                0,
                columns * channels,
              ),
            ),
          ).flat()
        : Array.from(
            mat[depth === 4 ? "toInt32Array" : depth === 5 ? "toFloat32Array" : "toUint8Array"](),
          );
      return { rows, columns, channels, depth, values };
    },
  };
}
