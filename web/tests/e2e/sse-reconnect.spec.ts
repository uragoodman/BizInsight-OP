import { expect, test } from "@playwright/test";
import { writeFileSync } from "node:fs";

/**
 * Browser-level acceptance for the durable SSE reconnect contract.
 *
 * Scenario A keeps the original stream alive and starts a second request for
 * the same client turn.  This is the server's "active reconnect" branch and
 * must emit heartbeat events and one final completed event without launching a
 * second graph execution or replaying intermediate node events.
 *
 * Scenario B aborts the original stream (a real browser disconnect) and then
 * reconnects.  It records the observed contract honestly: the current server
 * cancels the in-flight execution when the client aborts, so the reconnect
 * stays in heartbeat polling and no Turn Commit is produced.  This is a
 * contract gap, not a pass.
 */

type SseEvent = Record<string, unknown> & { event?: string };

const bodyTemplate = (stamp: string) => ({
  message: "查询 2018-08-01 到 2018-08-03 的 GMV",
  use_current_explorer_context: true,
  explorer_context: {
    seller_id: "seller-1",
    metric_name: "gmv",
    start_date: "2018-08-01",
    end_date: "2018-08-03",
    granularity: "day",
  },
  memory_enabled: false,
  chat_session_id: `sse-reconnect-${stamp}`,
  chat_turn_id: `turn-${stamp}`,
});

test("active reconnect returns heartbeat plus final completed without a second execution", async ({ page }) => {
  test.setTimeout(240_000);
  const base = process.env.SSE_BACKEND_URL ?? "http://127.0.0.1:28080";
  const output = process.env.SSE_OUTPUT_A;
  const stamp = `a-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  const body = bodyTemplate(stamp);
  await page.goto(`${base}/docs`);

  const result = await page.evaluate(async ({ body }) => {
    function parseChunk(buffer: string): { events: SseEvent[]; rest: string } {
      const events: SseEvent[] = [];
      let rest = buffer;
      let boundary = rest.indexOf("\n\n");
      while (boundary >= 0) {
        const raw = rest.slice(0, boundary);
        rest = rest.slice(boundary + 2);
        for (const line of raw.split("\n")) {
          if (line.startsWith("data: ")) {
            try { events.push(JSON.parse(line.slice(6)) as SseEvent); } catch { /* ignore */ }
          }
        }
        boundary = rest.indexOf("\n\n");
      }
      return { events, rest };
    }

    async function openStream(): Promise<ReadableStreamDefaultReader<Uint8Array>> {
      const response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok || !response.body) {
        throw new Error(`stream status ${response.status}: ${await response.text()}`);
      }
      return response.body.getReader();
    }

    async function readAll(reader: ReadableStreamDefaultReader<Uint8Array>, stopOnCompleted: boolean) {
      const decoder = new TextDecoder();
      const events: SseEvent[] = [];
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseChunk(buffer);
        buffer = parsed.rest;
        for (const event of parsed.events) {
          events.push(event);
          if (stopOnCompleted && event.event === "completed") return events;
        }
      }
      return events;
    }

    const firstReader = await openStream();
    // Wait until the original stream is confirmed running before reconnecting.
    const firstReaderProbe = firstReader;
    let firstStarted = false;
    const firstPromise = (async () => {
      const events: SseEvent[] = [];
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await firstReaderProbe.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseChunk(buffer);
        buffer = parsed.rest;
        for (const event of parsed.events) {
          events.push(event);
          if (!firstStarted) firstStarted = true;
          if (event.event === "completed") return events;
        }
      }
      return events;
    })();
    while (!firstStarted) {
      await new Promise((resolve) => setTimeout(resolve, 25));
    }

    const secondReader = await openStream();
    const second = await readAll(secondReader, true);
    const first = await firstPromise;
    return { first, second };
  }, { body });

  writeFileSync(output!, JSON.stringify({ body, ...result }, null, 2), "utf8");
  const completedA = result.first.find((event) => event.event === "completed");
  const completedB = result.second.find((event) => event.event === "completed");
  const heartbeats = result.second.filter((event) => event.event === "heartbeat").length;
  const nodeEvents = result.second.filter((event) =>
    ["node_started", "node_completed", "plan"].includes(String(event.event)),
  ).length;

  expect(completedA, "original stream must commit").toBeTruthy();
  expect(completedB, "reconnect must receive the committed result").toBeTruthy();
  expect(heartbeats, "active reconnect must emit heartbeats").toBeGreaterThan(0);
  expect(nodeEvents, "reconnect must not replay intermediate node events").toBe(0);
  expect(completedB?.durable_turn_id).toBe(completedA?.durable_turn_id);
  expect(completedB?.commit_version).toBe(completedA?.commit_version);
});

test("true browser disconnect recovers the committed result on reconnect", async ({ page }) => {
  test.setTimeout(120_000);
  const base = process.env.SSE_BACKEND_URL ?? "http://127.0.0.1:28080";
  const output = process.env.SSE_OUTPUT_B;
  const stamp = `b-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  const body = bodyTemplate(stamp);
  await page.goto(`${base}/docs`);

  const result = await page.evaluate(async ({ body }) => {
    function parseChunk(buffer: string): { events: SseEvent[]; rest: string } {
      const events: SseEvent[] = [];
      let rest = buffer;
      let boundary = rest.indexOf("\n\n");
      while (boundary >= 0) {
        const raw = rest.slice(0, boundary);
        rest = rest.slice(boundary + 2);
        for (const line of raw.split("\n")) {
          if (line.startsWith("data: ")) {
            try { events.push(JSON.parse(line.slice(6)) as SseEvent); } catch { /* ignore */ }
          }
        }
        boundary = rest.indexOf("\n\n");
      }
      return { events, rest };
    }

    async function readOnce(abortAfterFirst: boolean, timeoutMs: number): Promise<{ events: SseEvent[]; timedOut: boolean }> {
      const controller = new AbortController();
      const response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        throw new Error(`stream status ${response.status}: ${await response.text()}`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const events: SseEvent[] = [];
      let buffer = "";
      let timedOut = false;
      const deadline = Date.now() + timeoutMs;
      try {
        while (true) {
          if (Date.now() > deadline) {
            timedOut = true;
            controller.abort();
            break;
          }
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parsed = parseChunk(buffer);
          buffer = parsed.rest;
          for (const event of parsed.events) {
            events.push(event);
            if (abortAfterFirst) {
              controller.abort();
              return { events, timedOut: false };
            }
          }
        }
      } catch (error) {
        if (!abortAfterFirst) throw error;
      }
      return { events, timedOut };
    }

    const first = await readOnce(true, 10_000);
    await new Promise((resolve) => setTimeout(resolve, 300));
    const reconnect = await readOnce(false, 120_000);
    return { first, reconnect };
  }, { body });

  writeFileSync(output!, JSON.stringify({ body, ...result }, null, 2), "utf8");
  const reconnectEvents = result.reconnect.events;
  const completed = reconnectEvents.find((event) => event.event === "completed");
  const heartbeats = reconnectEvents.filter((event) => event.event === "heartbeat").length;
  // The graph now continues in a worker after the browser aborts, so the
  // reconnect must observe the committed result instead of waiting forever.
  expect(result.reconnect.timedOut, "reconnect must not time out after the worker commits the Turn").toBe(false);
  expect(completed, "reconnect must receive the committed result after a true disconnect").toBeTruthy();
  expect(completed?.durable_turn_id, "completed must carry the durable turn id").toBeTruthy();
  expect(typeof completed?.commit_version, "completed must carry the commit version").toBe("number");
  expect(heartbeats, "reconnect may heartbeat while the worker runs").toBeGreaterThanOrEqual(0);
});
