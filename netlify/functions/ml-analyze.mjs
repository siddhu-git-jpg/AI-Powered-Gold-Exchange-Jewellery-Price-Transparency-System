export async function handler(event) {
  if (event.httpMethod !== "POST") return json(405, { error: "Use POST for ML analysis." });
  const base = process.env.ML_SERVICE_URL;
  if (!base) return json(503, { error: "The optional ML service is not configured. The calculator formula remains available." });
  if ((event.body || "").length > 5_500_000) return json(413, { error: "The analysis request is too large." });
  try {
    const upstream = await fetch(`${base.replace(/\/$/, "")}/analyze`, {
      method: "POST", headers: { "content-type": "application/json" }, body: event.body || "",
      signal: AbortSignal.timeout(120_000),
    });
    return { statusCode: upstream.status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" }, body: await upstream.text() };
  } catch {
    return json(503, { error: "The optional ML service is unavailable. The calculator formula remains available." });
  }
}

function json(statusCode, payload) {
  return { statusCode, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" }, body: JSON.stringify(payload) };
}
