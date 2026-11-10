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
function T(l, f) {
  try {
    postMessage(l + " => " + JSON.stringify(f()));
  } catch (e) {
    postMessage(l + " !! " + String(e).slice(0, 140));
  }
}
cv.then((cv) => {
  T("line_h", () => {
    const m = cv.Mat.zeros(7, 7, cv.CV_8UC1);
    cv.line(m, { x: 1, y: 2 }, { x: 5, y: 2 }, [255, 0, 0, 0], 1, 8, 0);
    return dump(m);
  });
  T("line_v", () => {
    const m = cv.Mat.zeros(7, 7, cv.CV_8UC1);
    cv.line(m, { x: 3, y: 1 }, { x: 3, y: 5 }, [200, 0, 0, 0], 1, 8, 0);
    return dump(m);
  });
  T("line_d", () => {
    const m = cv.Mat.zeros(7, 7, cv.CV_8UC1);
    cv.line(m, { x: 1, y: 1 }, { x: 5, y: 5 }, [255, 0, 0, 0], 1, 8, 0);
    return dump(m);
  });
  T("line_slope", () => {
    const m = cv.Mat.zeros(8, 8, cv.CV_8UC1);
    cv.line(m, { x: 1, y: 1 }, { x: 6, y: 4 }, [255, 0, 0, 0], 1, 8, 0);
    return dump(m);
  });
  T("line_thick", () => {
    const m = cv.Mat.zeros(9, 9, cv.CV_8UC1);
    cv.line(m, { x: 1, y: 4 }, { x: 7, y: 4 }, [255, 0, 0, 0], 2, 8, 0);
    return dump(m);
  });
  T("rect", () => {
    const m = cv.Mat.zeros(7, 7, cv.CV_8UC1);
    cv.rectangle(m, { x: 1, y: 1 }, { x: 5, y: 4 }, [255, 0, 0, 0], 1, 8, 0);
    return dump(m);
  });
  T("rect_fill", () => {
    const m = cv.Mat.zeros(7, 7, cv.CV_8UC1);
    cv.rectangle(m, { x: 1, y: 1 }, { x: 5, y: 4 }, [255, 0, 0, 0], -1, 8, 0);
    return dump(m);
  });
  T("circle", () => {
    const m = cv.Mat.zeros(9, 9, cv.CV_8UC1);
    cv.circle(m, { x: 4, y: 4 }, 3, [255, 0, 0, 0], 1, 8, 0);
    return dump(m);
  });
  T("circle_fill", () => {
    const m = cv.Mat.zeros(9, 9, cv.CV_8UC1);
    cv.circle(m, { x: 4, y: 4 }, 3, [255, 0, 0, 0], -1, 8, 0);
    return dump(m);
  });
  T("marker", () => {
    const m = cv.Mat.zeros(9, 9, cv.CV_8UC1);
    cv.drawMarker(m, { x: 4, y: 4 }, [255, 0, 0, 0], 0, 10, 1, 8);
    return dump(m);
  });
  T("marker_x", () => {
    const m = cv.Mat.zeros(9, 9, cv.CV_8UC1);
    cv.drawMarker(m, { x: 4, y: 4 }, [255, 0, 0, 0], 1, 10, 1, 8);
    return dump(m);
  });
  T("polylines", () => {
    const m = cv.Mat.zeros(7, 7, cv.CV_8UC1);
    const pts = cv.matFromArray(1, 4, cv.CV_32SC2, [1, 1, 5, 1, 5, 5, 1, 5]);
    const mv = new cv.MatVector();
    mv.push_back(pts);
    cv.polylines(m, mv, true, [255, 0, 0, 0], 1, 8, 0);
    return dump(m);
  });
  T("fillPoly", () => {
    const m = cv.Mat.zeros(7, 7, cv.CV_8UC1);
    const pts = cv.matFromArray(1, 3, cv.CV_32SC2, [1, 1, 5, 1, 3, 5]);
    const mv = new cv.MatVector();
    mv.push_back(pts);
    cv.fillPoly(m, mv, [255, 0, 0, 0], 8, 0, { x: 0, y: 0 });
    return dump(m);
  });
  T("ellipse", () => {
    const m = cv.Mat.zeros(11, 11, cv.CV_8UC1);
    cv.ellipse(m, { x: 5, y: 5 }, { width: 6, height: 4 }, 0, 0, 360, [255, 0, 0, 0], 1, 8, 0);
    return dump(m);
  });
  T("putText", () => {
    const m = cv.Mat.zeros(15, 40, cv.CV_8UC1);
    cv.putText(m, "A", { x: 1, y: 12 }, 0, 0.5, [255, 0, 0, 0], 1, 8, false);
    return dump(m);
  });
  postMessage("DONE");
});
