import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { handler as analyzeJewellery } from "./netlify/functions/analyze-jewellery.mjs";

const root = resolve("outputs");
const port = Number(process.env.PORT || 4173);
const mime = {
  ".html": "text/html; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
};

createServer(async (request, response) => {
  if (request.url?.split("?")[0] === "/api/ml/trend") {
    const base = (process.env.ML_SERVICE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
    try {
      const upstream = await fetch(`${base}/trend${new URL(request.url, "http://localhost").search}`, {
        signal: AbortSignal.timeout(30_000),
      });
      response.writeHead(upstream.status, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      response.end(await upstream.text());
    } catch {
      response.writeHead(503, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      response.end(JSON.stringify({ status: "service_offline", message: "Regression trend is unavailable until the ML service and verified rate history are configured." }));
    }
    return;
  }

  if (request.url?.split("?")[0] === "/api/ml/analyze") {
    const chunks = [];
    let size = 0;
    request.on("data", chunk => {
      size += chunk.length;
      if (size <= 6_000_000) chunks.push(chunk);
    });
    request.on("end", async () => {
      if (size > 6_000_000) {
        response.writeHead(413, { "content-type": "application/json" });
        response.end(JSON.stringify({ error: "The analysis request is too large." }));
        return;
      }
      const base = (process.env.ML_SERVICE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
      try {
        const upstream = await fetch(`${base}/analyze`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: Buffer.concat(chunks),
          signal: AbortSignal.timeout(120_000),
        });
        response.writeHead(upstream.status, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
        response.end(await upstream.text());
      } catch {
        response.writeHead(503, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
        response.end(JSON.stringify({ error: "The optional CNN/OpenCV/XGBoost service is not running. The transparent calculator is still available." }));
      }
    });
    return;
  }

  if (request.url?.split("?")[0] === "/.netlify/functions/analyze-jewellery") {
    const chunks = [];
    let size = 0;
    request.on("data", chunk => {
      size += chunk.length;
      if (size <= 6_000_000) chunks.push(chunk);
    });
    request.on("end", async () => {
      if (size > 6_000_000) {
        response.writeHead(413, { "content-type": "application/json" });
        response.end(JSON.stringify({ error: "The files are too large. Try smaller images or a smaller PDF." }));
        return;
      }
      const result = await analyzeJewellery({
        httpMethod: request.method,
        body: Buffer.concat(chunks).toString("utf8"),
      });
      response.writeHead(result.statusCode, result.headers);
      response.end(result.body);
    });
    return;
  }

  const pathname = decodeURIComponent(new URL(request.url || "/", "http://localhost").pathname);
  const requested = pathname === "/" ? "/index.html" : pathname;
  const file = resolve(root, `.${requested}`);
  if (file !== root && !file.startsWith(root + sep)) {
    response.writeHead(403);
    response.end("Forbidden");
    return;
  }
  try {
    const content = await readFile(file);
    response.writeHead(200, { "content-type": mime[extname(file)] || "application/octet-stream", "cache-control": "no-store" });
    response.end(content);
  } catch {
    response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    response.end("Not found");
  }
}).listen(port, "127.0.0.1", () => {
  console.log(`Jewellery calculator available at http://localhost:${port}`);
});
