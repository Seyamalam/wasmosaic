/* global importScripts, cv */
/* oxlint-disable unicorn/require-post-message-target-origin */

async function referenceReady() {
  importScripts("/test/browser/.cache/opencv-4.13.0.js");
  return new Promise((resolve) => {
    cv.then((ready) => {
      delete ready.then;
      resolve(ready);
    });
  });
}

self.addEventListener("message", async () => {
  try {
    const reference = await referenceReady();
    const { referenceTemplateApi } = await import("./template-matching-adapters.js");
    const { templateMatchingCases } = await import("./template-matching-cases.js");
    self.postMessage({ result: templateMatchingCases(referenceTemplateApi(reference)) });
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.stack : String(error) });
  }
});
