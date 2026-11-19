import { readFileSync, rmSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(here, "../../..");
const stateFile = path.join(here, ".e2e-account.json");

export default async function globalTeardown() {
  try {
    const account = JSON.parse(readFileSync(stateFile, "utf8")) as { tenant_id: string; user_id: string };
    const python = process.env.E2E_PYTHON ?? "python";
    const result = spawnSync(python, [path.join(here, "e2e_accounts.py"), "cleanup", account.tenant_id, account.user_id], {
      cwd: projectRoot,
      encoding: "utf8",
      env: process.env,
    });
    if (result.status !== 0) throw new Error(`E2E account cleanup failed: ${result.stderr || result.stdout}`);
  } finally {
    rmSync(stateFile, { force: true });
  }
}
