const MAX_BODY_CHARS = 5_500_000;

const responseSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "item_type",
    "metal",
    "bill_weight_g",
    "bill_purity",
    "bill_amount_inr",
    "condition_description",
    "bill_matches_photo",
    "confidence",
    "notes",
  ],
  properties: {
    item_type: { type: "string" },
    metal: { type: "string", enum: ["gold", "silver", "other", "unknown"] },
    bill_weight_g: { anyOf: [{ type: "number" }, { type: "null" }] },
    bill_purity: {
      type: "string",
      enum: ["18K", "20K", "22K", "24K", "925", "999", "unknown", "not_provided"],
    },
    bill_amount_inr: { anyOf: [{ type: "number" }, { type: "null" }] },
    condition_description: { type: "string" },
    bill_matches_photo: { type: "string", enum: ["yes", "no", "uncertain", "no_bill"] },
    confidence: { type: "string", enum: ["low", "medium", "high"] },
    notes: { type: "array", items: { type: "string" } },
  },
};

function json(statusCode, payload) {
  return {
    statusCode,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
    body: JSON.stringify(payload),
  };
}

function validDataUrl(file, allowedTypes) {
  if (!file || typeof file.dataUrl !== "string" || typeof file.name !== "string") return false;
  const match = file.dataUrl.match(/^data:([^;,]+);base64,([A-Za-z0-9+/=]+)$/);
  return Boolean(match && allowedTypes.includes(match[1]) && file.dataUrl.length < 4_200_000);
}

function outputText(response) {
  if (typeof response.output_text === "string") return response.output_text;
  for (const item of response.output || []) {
    for (const content of item.content || []) {
      if (content.type === "output_text" && typeof content.text === "string") return content.text;
    }
  }
  return "";
}

export async function handler(event) {
  if (event.httpMethod !== "POST") return json(405, { error: "Use POST for analysis." });
  if (!process.env.OPENAI_API_KEY) {
    return json(503, { error: "AI analysis is not configured yet. Add OPENAI_API_KEY to the server environment." });
  }

  const rawBody = event.isBase64Encoded
    ? Buffer.from(event.body || "", "base64").toString("utf8")
    : event.body || "";
  if (rawBody.length > MAX_BODY_CHARS) return json(413, { error: "The files are too large. Try smaller images or a smaller PDF." });

  let body;
  try {
    body = JSON.parse(rawBody);
  } catch {
    return json(400, { error: "Could not read the uploaded files. Please choose them again." });
  }

  const photo = body.photo || null;
  const bill = body.bill || null;
  const imageTypes = ["image/jpeg", "image/png", "image/webp"];
  if (!photo && !bill) return json(400, { error: "Choose a jewellery photo, a bill, or both before analyzing." });
  if (photo && !validDataUrl(photo, imageTypes)) return json(400, { error: "The jewellery photo format is unsupported or too large." });
  if (bill && !validDataUrl(bill, [...imageTypes, "application/pdf"])) {
    return json(400, { error: "The bill must be a JPG, PNG, WebP, or PDF under the upload limit." });
  }

  const content = [{
    type: "input_text",
    text: [
      "Analyze the attached old-jewellery photo and/or purchase bill for a gold exchange estimate.",
      "Read weight and purity only from the bill when clearly printed; do not infer purity or weight from a photo.",
      "Describe the item and visible condition from the photo. Do not claim a visual purity test.",
      "If both files are present, say whether the bill appears to describe the pictured item, or mark uncertain.",
      "Return a null bill weight or amount and 'unknown' purity when not clearly readable. Amount means the invoice amount, not today's exchange value.",
      "Ignore personal identifiers such as customer name, address, phone, and payment details.",
      "Keep notes short. This is an estimate aid; a jeweller must verify weight, purity, stones, and deductions.",
    ].join(" "),
  }];
  if (photo) content.push({ type: "input_image", image_url: photo.dataUrl, detail: "high" });
  if (bill) {
    if (bill.dataUrl.startsWith("data:application/pdf;")) {
      content.push({ type: "input_file", filename: bill.name.replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 100), file_data: bill.dataUrl, detail: "high" });
    } else {
      content.push({ type: "input_image", image_url: bill.dataUrl, detail: "high" });
    }
  }

  try {
    const upstream = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: process.env.OPENAI_MODEL || "gpt-6-astra",
        store: false,
        max_output_tokens: 500,
        instructions: "You extract visible information for a jewellery estimate. Be conservative and never invent values.",
        input: [{ role: "user", content }],
        text: {
          format: {
            type: "json_schema",
            name: "jewellery_bill_and_photo_analysis",
            strict: true,
            schema: responseSchema,
          },
        },
      }),
    });

    const result = await upstream.json();
    if (!upstream.ok) {
      const detail = result?.error?.message || "The AI provider could not analyze these files.";
      return json(upstream.status === 401 || upstream.status === 403 ? 502 : 502, {
        error: upstream.status === 401 || upstream.status === 403
          ? "The server API key or model access is not configured correctly."
          : detail.slice(0, 240),
      });
    }

    const text = outputText(result);
    if (!text) return json(502, { error: "The AI returned no readable analysis. Try a clearer photo or bill." });
    return json(200, { analysis: JSON.parse(text) });
  } catch (error) {
    console.error("Jewellery analysis request failed:", error?.message || "unknown error");
    return json(502, { error: "Analysis failed. Please try again with a clearer photo or bill." });
  }
}
