import { existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

const [script, ...scriptArgs] = process.argv.slice(2);
if (!script) {
  console.error("Usage: node scripts/run-python.mjs <script.py> [args...]");
  process.exit(2);
}

const isWindows = process.platform === "win32";
const executableName = isWindows ? "python.exe" : "python";
const candidates = [
  process.env.BIZINSIGHT_PYTHON,
  process.env.PYTHON,
  process.env.CONDA_PREFIX
    ? path.join(process.env.CONDA_PREFIX, isWindows ? executableName : "bin/python")
    : undefined,
  path.join(os.homedir(), ".conda", "envs", "fastapi-demo", isWindows ? executableName : "bin/python"),
].filter(Boolean);

const python = candidates.find((candidate) => existsSync(candidate)) ?? "python";
const result = spawnSync(python, [script, ...scriptArgs], { stdio: "inherit" });
if (result.error) {
  console.error(`Could not start Python executable '${python}': ${result.error.message}`);
  process.exit(1);
}
process.exit(result.status ?? 1);
