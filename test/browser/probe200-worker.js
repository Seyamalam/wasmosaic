/* global importScripts, cv */
importScripts("/test/browser/.cache/opencv-4.13.0.js");
function dump(m) {
  const d = m.depth();
  const meth = ["ucharPtr", "charPtr", "ushortPtr", "shortPtr", "intPtr", "floatPtr", "doublePtr"][
    d
  ];
  const flat = [];
  for (let r = 0; r < m.rows; r++) flat.push(...Array.from(m[meth](r)));
  return { t: m.type(), flat };
}
function T(label, fn) {
  try {
    const r = fn();
    postMessage(label + " => " + JSON.stringify(r));
  } catch (e) {
    postMessage(label + " !! " + String(e).slice(0, 140));
  }
}
cv.then((cv) => {
  const src = cv.matFromArray(
    4,
    4,
    cv.CV_8UC1,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  );
  let hist = new cv.Mat();
  // try plain-array signatures
  T("calcHist_arrays", () => {
    cv.calcHist(src, [0], new cv.Mat(), hist, [16], [0, 16], false);
    return dump(hist);
  });
  T("calcHist_mv_arr", () => {
    const mv = new cv.MatVector();
    mv.push_back(src);
    cv.calcHist(mv, [0], new cv.Mat(), hist, [16], [0, 16], false);
    return dump(hist);
  });
  T("calcHist_f32", () => {
    const src32 = cv.matFromArray(2, 2, cv.CV_32FC1, [0.5, 1.5, 2.5, 3.5]);
    const h2 = new cv.Mat();
    cv.calcHist(src32, [0], new cv.Mat(), h2, [4], [0, 4], false);
    return dump(h2);
  });
  T("grsp_obj", () => {
    const d = new cv.Mat();
    cv.getRectSubPix(src, new cv.Size(3, 3), { x: 1.5, y: 1.5 }, d, -1);
    return dump(d);
  });
  T("grsp_pt", () => {
    const d = new cv.Mat();
    cv.getRectSubPix(src, new cv.Size(3, 3), new cv.Point(1.5, 1.5), d, -1);
    return dump(d);
  });
  T("grsp_args", () => {
    const d = new cv.Mat();
    cv.getRectSubPix(src, new cv.Size(3, 3), 1.5, 1.5, d, -1);
    return dump(d);
  });
  postMessage("DONE");
});
