import { statuses } from "./dom.js";

export function runPresentation(status, events = [], result = null) {
  const lastStop = [...events].reverse().find((e) => e.details?.stop_reason);
  const stopReason = result?.stop_reason || lastStop?.details.stop_reason;
  const coverageComplete = Array.isArray(result?.coverage)
    ? result.coverage.length > 0 &&
      result.coverage.every((r) => r.status === "covered")
    : result?.coverage_complete;
  const lastWriter = [...events].reverse().find((e) => e.agent === "writer");
  const findings =
    result?.reported_claim_ids?.length ??
    result?.reported_claim_count ??
    lastWriter?.details?.written_claims;
  const stopped = new Set([
    "budget_exhausted",
    "provider_error",
    "timeout",
    "no_new_evidence",
    "retrieval_limit",
    "task_blocked",
    "task_limit",
    "coordinator_step_limit",
    "agent_step_limit",
    "step_limit",
    "no_progress",
    "invalid_agent_action",
  ]);
  const partial =
    status === "completed" &&
    (stopped.has(stopReason) || coverageComplete === false);
  let label = statuses[status] || "Sẵn sàng";
  if (partial) {
    label =
      stopReason === "budget_exhausted"
        ? "Đã dừng: hết ngân sách"
        : findings > 0
          ? "Kết quả một phần"
          : "Kết thúc: chưa đủ bằng chứng";
  }
  return { label, partial, stopReason, findings };
}

export function recordStatusLabel(record) {
  return runPresentation(record.status, record.events, record.result || record)
    .label;
}
