export async function fetchHealth() {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error("Health check failed");
  return response.json();
}

export async function fetchTopics(language) {
  const response = await fetch(`/api/topics?language=${language}`);
  if (!response.ok) throw new Error("Topic loading failed");
  return response.json();
}

export async function streamGeneration(payload, handlers) {
  const response = await fetch("/api/generate/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    throw new Error(`Generation failed with HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";
    for (const chunk of chunks) {
      const event = parseEvent(chunk);
      if (!event) continue;
      const handler = handlers[event.event];
      if (handler) handler(event.payload);
    }
  }
}

function parseEvent(chunk) {
  const lines = chunk.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const dataLine = lines.find((line) => line.startsWith("data:"));
  if (!eventLine || !dataLine) return null;
  return {
    event: eventLine.replace("event:", "").trim(),
    payload: JSON.parse(dataLine.replace("data:", "").trim() || "{}"),
  };
}

