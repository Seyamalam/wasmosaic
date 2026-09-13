import { resolve, sep } from "node:path";

const root = resolve(import.meta.dir, "..");
const port = Number(process.env["WASMOSAIC_TEST_PORT"] ?? 8766);

const server = Bun.serve({
  hostname: "127.0.0.1",
  port,
  async fetch(request) {
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method not allowed", { status: 405 });
    }
    const url = new URL(request.url);
    const pathname = url.pathname === "/" ? "/test/browser/template-matching.html" : url.pathname;
    const path = resolve(root, `.${decodeURIComponent(pathname)}`);
    if (!path.startsWith(`${root}${sep}`)) return new Response("Forbidden", { status: 403 });
    const file = Bun.file(path);
    if (!(await file.exists())) return new Response("Not found", { status: 404 });
    return new Response(request.method === "HEAD" ? null : file, {
      headers: { "Cache-Control": "no-store", "Content-Type": file.type },
    });
  },
});

process.stdout.write(`Browser fixtures: ${server.url}test/browser/template-matching.html\n`);
