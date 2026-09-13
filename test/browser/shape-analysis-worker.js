/* global importScripts, cv */
/* oxlint-disable unicorn/require-post-message-target-origin */
self.addEventListener("message", async () => {
  try {
    importScripts("/test/browser/.cache/opencv-4.13.0.js");
    const reference = await new Promise((resolve) =>
      cv.then((ready) => {
        delete ready.then;
        resolve(ready);
      }),
    );
    const { contourApi } = await import("./shape-analysis-adapters.js");
    const { runContourCases } = await import("./shape-analysis-cases.js");
    self.postMessage({ result: runContourCases(contourApi(reference, true)) });
  } catch (error) {
    self.postMessage({ error: String(error) });
  }
});
