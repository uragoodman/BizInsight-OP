import { afterEach, describe, expect, it, vi } from "vitest";
import { api, streamCodeAnalysisJob, type ApiAuth } from "./api";
import type { CodeAnalysisEvent, CodeAnalysisJob } from "./types";

const auth: ApiAuth = {
  actor: "phase11-user",
  analysisToken: "",
  bearerToken: "signed-token",
};

function job(status: CodeAnalysisJob["status"]): CodeAnalysisJob {
  return {
    job_id: "caj_1",
    question: "Analyze",
    status,
    file_id: "file_1",
    plan_hash: "a".repeat(64),
    plan_data: {
      question: "Analyze",
      file_id: "file_1",
      file_sha256: "b".repeat(64),
      data_contract: { file_type: "csv" },
      retention_hours: 24,
      expected_output: ["conclusion"],
    },
    expires_at: "2026-08-01T00:00:00Z",
    revision: 1,
    created_at: "2026-07-27T00:00:00Z",
    updated_at: "2026-07-27T00:01:00Z",
  };
}

function eventStream(events: CodeAnalysisEvent[]): Response {
  const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("code-analysis API", () => {
  it("parses status, heartbeat, and terminal job events", async () => {
    const completed = job("completed");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(eventStream([
      { event: "status", sequence_number: 1, job_id: "caj_1", status: "running", error_code: null },
      { event: "heartbeat", sequence_number: 2, job_id: "caj_1", status: "running" },
      { event: "completed", sequence_number: 3, job: completed },
    ])));
    const seen: string[] = [];

    const terminal = await streamCodeAnalysisJob(auth, "caj_1", (event) => seen.push(event.event));

    expect(seen).toEqual(["status", "heartbeat", "completed"]);
    expect(terminal?.status).toBe("completed");
    expect(fetch).toHaveBeenCalledWith(
      "/code-analysis/jobs/caj_1/events",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({ Authorization: "Bearer signed-token" }),
      }),
    );
  });

  it("surfaces a structured stream failure without a terminal job", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(eventStream([
      { event: "failed", sequence_number: 1, detail: "Analysis job not found" },
    ])));

    await expect(streamCodeAnalysisJob(auth, "missing", () => undefined))
      .rejects.toEqual(expect.objectContaining({ message: "Analysis job not found" }));
  });

  it("downloads an artifact through the authenticated, encoded endpoint", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("png", {
      status: 200,
      headers: { "Content-Type": "image/png" },
    })));

    const blob = await api.codeAnalysisArtifact(auth, "job/1", "artifact/1");

    expect(blob.type).toBe("image/png");
    expect(fetch).toHaveBeenCalledWith(
      "/code-analysis/jobs/job%2F1/artifacts/artifact%2F1",
      { headers: expect.objectContaining({ Authorization: "Bearer signed-token" }) },
    );
  });
});
