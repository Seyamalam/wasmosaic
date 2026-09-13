export function perspectiveApi(cv, reference) {
  const names = ["u8", "i8", "u16", "i16", "i32", "f32", "f64"];
  const arrays = [
    Uint8Array,
    Int8Array,
    Uint16Array,
    Int16Array,
    Int32Array,
    Float32Array,
    Float64Array,
  ];
  const constructors = [
    "matFromU8",
    "matFromI8",
    "matFromU16",
    "matFromI16",
    "matFromI32",
    "matFromF32",
    "matFromF64",
  ];
  const readers = [
    "toUint8Array",
    "toInt8Array",
    "toUint16Array",
    "toInt16Array",
    "toInt32Array",
    "toFloat32Array",
    "toFloat64Array",
  ];
  const pointers = [
    "ucharPtr",
    "charPtr",
    "ushortPtr",
    "shortPtr",
    "intPtr",
    "floatPtr",
    "doublePtr",
  ];
  return {
    contract: () => ({
      getLength: cv.getPerspectiveTransform.length,
      methods: [
        cv.DECOMP_LU,
        cv.DECOMP_SVD,
        cv.DECOMP_EIG,
        cv.DECOMP_CHOLESKY,
        cv.DECOMP_QR,
        cv.DECOMP_NORMAL,
      ],
      warpLength: cv.warpPerspective.length,
    }),
    mat: (rows, columns, channels, depth, values) =>
      reference
        ? cv.matFromArray(rows, columns, depth + ((channels - 1) << 3), values)
        : cv[constructors[depth]](rows, columns, channels, new arrays[depth](values)),
    empty: () => (reference ? new cv.Mat() : cv.emptyMat()),
    dispose: (mat) => (reference ? mat.delete() : mat.dispose()),
    roi: (mat, row, column, rows, columns) =>
      reference
        ? mat.roi(new cv.Rect(column, row, columns, rows))
        : mat.roi(row, column, rows, columns),
    get: (...args) => cv.getPerspectiveTransform(...args),
    warp: (...args) => cv.warpPerspective(...args),
    summary(mat) {
      const rows = mat.rows,
        columns = reference ? mat.cols : mat.columns,
        channels = reference ? mat.channels() : mat.channels,
        depth = reference ? mat.depth() : names.indexOf(mat.depth);
      const values = reference
        ? Array.from({ length: rows }, (_, row) =>
            Array.from(mat[pointers[depth]](row).subarray(0, columns * channels)),
          ).flat()
        : Array.from(mat[readers[depth]]());
      return {
        rows,
        columns,
        channels,
        depth,
        values: values.map((value) => (Number.isFinite(value) ? value : String(value))),
      };
    },
  };
}
