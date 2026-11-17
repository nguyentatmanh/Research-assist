"""Share the production prompt and JSON schemas for each teachable task."""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from research_assist.agents.autonomous import INSTRUCTIONS
from research_assist.agents.prompts.fact_checker import INSTRUCTION as CHECK
from research_assist.agents.prompts.reader import INSTRUCTION as READ
from research_assist.agents.prompts.summarizer import INSTRUCTION as SUMMARIZE
from research_assist.agents.prompts.writer import INSTRUCTION as WRITE
from research_assist.application.agent_runtime import step_instruction
from research_assist.domain.models import (
    AgentDecision,
    FactCheckOutput,
    QuestionPlan,
    ReaderOutput,
    SearchCompletion,
    SynthesisOutput,
    WriterOutput,
)
from research_assist.domain.planning import PLAN_INSTRUCTION
from research_assist.experiments.data import file_hash
from research_assist.infrastructure.llm import HTTPLanguageModel, strict_schema
from research_assist.infrastructure.routing import TASK_ROLES


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    schema: type[BaseModel]


TASKS = {
    "question_plan": TaskSpec(PLAN_INSTRUCTION, QuestionPlan),
    "read_evidence": TaskSpec(READ, ReaderOutput),
    "synthesize_claims": TaskSpec(SUMMARIZE, SynthesisOutput),
    "check_claims": TaskSpec(CHECK, FactCheckOutput),
    "write_report": TaskSpec(WRITE, WriterOutput),
    "retrieval_step": TaskSpec(
        step_instruction(INSTRUCTIONS["retrieval"]), AgentDecision[SearchCompletion]
    ),
    "reader_step": TaskSpec(step_instruction(INSTRUCTIONS["reader"]), AgentDecision[ReaderOutput]),
    "summarizer_step": TaskSpec(
        step_instruction(INSTRUCTIONS["summarizer"]), AgentDecision[SynthesisOutput]
    ),
    "fact_checker_step": TaskSpec(
        step_instruction(INSTRUCTIONS["fact_checker"]), AgentDecision[FactCheckOutput]
    ),
    "writer_step": TaskSpec(step_instruction(INSTRUCTIONS["writer"]), AgentDecision[WriterOutput]),
}
ROLES = ("retrieval", "reader", "summarizer", "fact_checker", "writer")
SPLITS = ("train", "validation", "test")
LAB = Path(__file__).resolve().parents[1]


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group: str = Field(min_length=1)
    task: str
    payload: dict[str, object] = Field(min_length=1)
    expected: dict[str, object]

    def target(self) -> BaseModel:
        return TASKS[self.task].schema.model_validate(self.expected)

    def messages(self, *, completion: bool = True) -> list[dict[str, str]]:
        spec = TASKS[self.task]
        messages = [
            {
                "role": "system",
                "content": HTTPLanguageModel._system_prompt(
                    spec.instruction, strict_schema(spec.schema)
                ),
            },
            {"role": "user", "content": json.dumps(self.payload, ensure_ascii=False)},
        ]
        if completion:
            messages.append({"role": "assistant", "content": self.target().model_dump_json()})
        return messages


def load_splits(directory: Path) -> dict[str, list[Example]]:
    splits: dict[str, list[Example]] = {}
    group_owner: dict[str, str] = {}
    inputs: set[str] = set()
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = Example.model_validate_json(line)
                if row.task not in TASKS or not row.group.strip():
                    raise ValueError("Unknown agent task or blank document group")
                if group_owner.setdefault(row.group, split) != split:
                    raise ValueError(f"Document group {row.group!r} appears in multiple splits")
                key = json.dumps([row.task, row.payload], sort_keys=True, ensure_ascii=False)
                if key in inputs:
                    raise ValueError("Duplicate task and payload")
                inputs.add(key)
                row.target()
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            rows.append(row)
        if not rows:
            raise ValueError(f"{path} is empty")
        splits[split] = rows
    return splits


def for_role(rows: list[Example], role: str) -> list[Example]:
    return [row for row in rows if TASK_ROLES[row.task] == role]


def split_hashes(directory: Path) -> dict[str, str]:
    return {name: file_hash(directory / f"{name}.jsonl") for name in SPLITS}
