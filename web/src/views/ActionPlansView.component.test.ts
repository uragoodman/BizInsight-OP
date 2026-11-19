import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

describe("ActionPlansView pagination binding", () => {
  it("does not pass the click event as the load reset argument", () => {
    const source = readFileSync(new URL("./ActionPlansView.vue", import.meta.url), "utf-8");

    expect(source).toContain('@click="load()">加载更多</el-button>');
    expect(source).not.toContain('@click="load">加载更多</el-button>');
  });
});
