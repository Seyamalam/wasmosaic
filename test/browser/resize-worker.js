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
    const { runResizeCases } = await import("./resize-cases.js");
    const { runResizeContracts } = await import("./resize-contract-cases.js");
    self.postMessage({
      result: [...runResizeCases(reference, true), ...runResizeContracts(reference, true)],
    });
  } catch (error) {
    self.postMessage({ error: String(error) });
  }
});
