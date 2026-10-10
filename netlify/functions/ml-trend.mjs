export async function handler(event) {
  if (event.httpMethod !== "GET") return json(405, { error: "Use GET for the trend forecast." });
  const base = process.env.ML_SERVICE_URL;
  if (!base) return json(503, { status: "service_offline", message: "Regression trend is unavailable until the ML service and verified rate history are configured." });
  try {
    const query = event.rawQuery ? `?${event.rawQuery}` : "";
    const upstream = await fetch(`${base.replace(/\/$/, "")}/trend${query}`, { signal: AbortSignal.timeout(30_000) });
    return { statusCode: upstream.status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" }, body: await upstream.text() };
  } catch {
    return json(503, { status: "service_offline", message: "Regression trend is currently unavailable." });
  }
}

function json(statusCode, payload) {
  return { statusCode, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" }, body: JSON.stringify(payload) };
}
