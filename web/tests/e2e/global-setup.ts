import { spawnSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(here, "../../..");
const stateFile = path.join(here, ".e2e-account.json");

export default async function globalSetup() {
  const python = process.env.E2E_PYTHON ?? "python";
  const result = spawnSync(python, [path.join(here, "e2e_accounts.py"), "seed"], {
    cwd: projectRoot,
    encoding: "utf8",
    env: process.env,
  });
  if (result.status !== 0) throw new Error(`E2E account setup failed: ${result.stderr || result.stdout}`);
  writeFileSync(stateFile, result.stdout.trim(), { encoding: "utf8", mode: 0o600 });
}
