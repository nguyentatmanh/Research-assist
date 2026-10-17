import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

export async function startBackend() {
  if (process.env.UI_BASE_URL)
    return { base: process.env.UI_BASE_URL, close: async () => {} };
  const dataDir = await mkdtemp(join(tmpdir(), "research-ui-"));
  const port = process.env.UI_TEST_PORT || "8765";
  const base = `http://127.0.0.1:${port}`;
  const server = spawn(
    process.env.RESEARCH_PYTHON || "python",
    [
      "-m",
      "uvicorn",
      "research_assist.api.app:create_app",
      "--factory",
      "--host",
      "127.0.0.1",
      "--port",
      port,
    ],
    {
      env: {
        ...process.env,
        RA_DATA_DIR: dataDir,
        RA_LLM_PROVIDER: "demo",
        RA_LLM_MODEL: "",
        RA_AGENT_ROUTES: "{}",
        RA_WORKFLOW_PROFILE: "balanced",
        RA_EMBEDDING_PROVIDER: "none",
        RA_CHUNK_STRATEGY: "chars",
        RA_LOCAL_DEVICE: "cpu",
        RA_RERANKER_MODEL: "",
        RA_PDF_OCR_ENABLED: "false",
        RA_ACCESS_TOKEN: "",
      },
      stdio: ["ignore", "ignore", "pipe"],
    },
  );
  let logs = "";
  let failed = null;
  server.stderr.on("data", (chunk) => {
    logs = (logs + chunk.toString()).slice(-3000);
  });
  server.on("error", (error) => {
    failed = error;
  });
  const close = async () => {
    if (server.exitCode === null) {
      const exited = new Promise((resolve) => server.once("exit", resolve));
      server.kill("SIGTERM");
      const timer = setTimeout(() => server.kill("SIGKILL"), 3000);
      await exited;
      clearTimeout(timer);
    }
    await rm(dataDir, { recursive: true, force: true });
  };
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    if (failed || server.exitCode !== null) {
      await close();
      throw new Error(`Backend startup failed: ${failed?.message || logs}`);
    }
    try {
      const response = await fetch(`${base}/api/health`);
      if (response.ok) return { base, close };
    } catch {
      /* Wait for application startup. */
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  await close();
  throw new Error(`Backend startup timed out: ${logs}`);
}
