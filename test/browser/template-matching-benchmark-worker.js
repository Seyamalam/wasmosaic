/* global importScripts, cv */
/* oxlint-disable unicorn/require-post-message-target-origin */

let api;
let initializationMs;

async function initialize(implementation) {
  const start = performance.now();
  const { packageTemplateApi, referenceTemplateApi } =
    await import("./template-matching-adapters.js");
  if (implementation === "package") {
    const backend = await import("/wasm/wasmosaic_wasm.js");
    await backend.default();
    const { createOpenCv } = await import("/dist/client.js");
    api = packageTemplateApi(createOpenCv(backend));
  } else {
    importScripts("/test/browser/.cache/opencv-4.13.0.js");
    await new Promise((resolve) => {
      cv.then((ready) => {
        api = referenceTemplateApi(ready);
        resolve();
      });
    });
  }
  initializationMs = performance.now() - start;
}

function percentile(sorted, fraction) {
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)];
}

function benchmark(config) {
  const count = config.rows * config.columns * config.channels;
  const input = config.depth === 0 ? new Uint8Array(count) : new Float32Array(count);
  let state = 0x5a17;
  for (let index = 0; index < count; index += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    input[index] = config.depth === 0 ? state >>> 24 : ((state >>> 20) - 2048) / 256;
  }
  const image = api.mat(config.rows, config.columns, config.channels, config.depth, input);
  const template = api.roi(image, 17, 23, config.templateRows, config.templateColumns);
  const result = api.empty();
  try {
    const execute = () => api.match(image, template, result, config.method);
    const firstStart = performance.now();
    execute();
    const firstCallMs = performance.now() - firstStart;
    const location = api.minMaxLoc(result).maxLoc;
    if (location.x !== 23 || location.y !== 17)
      throw new Error("Benchmark template location differs");
    for (let index = 0; index < 3; index += 1) execute();
    const samplesMs = [];
    let totalMs = 0;
    while (samplesMs.length < 5 || totalMs < 1000) {
      const start = performance.now();
      execute();
      const elapsed = performance.now() - start;
      samplesMs.push(elapsed);
      totalMs += elapsed;
    }
    const sorted = samplesMs.toSorted((left, right) => left - right);
    return {
      ...config,
      initializationMs,
      firstCallMs,
      warmupIterations: 3,
      sampleCount: samplesMs.length,
      totalMeasuredMs: totalMs,
      p50Ms: percentile(sorted, 0.5),
      p95Ms: percentile(sorted, 0.95),
      samplesMs,
      pixelBytesAcrossJsWasmPerTimedCall: 0,
      allocationCount: null,
      allocationNote:
        "Internal allocations are not instrumented; inputs and the result handle persist across calls.",
    };
  } finally {
    for (const matrix of [image, template, result]) api.dispose(matrix);
  }
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (api === undefined) await initialize(data.implementation);
    self.postMessage({ result: benchmark(data) });
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.stack : String(error) });
  }
});
