import pytest

from research_assist.application.agent_runtime import AgentRuntime, AgentTool
from research_assist.application.task_graph import TaskGraph
from research_assist.domain.errors import ActionRejected, BudgetExceeded
from research_assist.domain.models import AgentSession, SearchCompletion, TaskRequest


class ScriptedDecisions:
    name, model = "test", "test"

    def __init__(self, decisions):
        self.decisions, self.payloads = iter(decisions), []

    async def generate(self, **kwargs):
        self.payloads.append(kwargs["payload"])
        value = next(self.decisions)
        if isinstance(value, Exception):
            raise value
        return kwargs["schema"].model_validate(value)


async def inline_step(agent, message, work, describe=None):
    return await work


def action(kind, **kwargs):
    return {"action": kind, "reason": "Complete the assigned bounded research work.", **kwargs}


async def exercise(
    decisions, *, max_steps=6, tools=None, accept=None, child=None, session=None, bound=None
):
    llm = ScriptedDecisions(decisions)
    memory = session or AgentSession(task_id="job_1", role="retrieval")
    accepted, events = [], []

    async def accept_default(result):
        accepted.append(result)

    def request_default(request):
        return "job_2"

    await AgentRuntime(llm, max_steps).run(
        memory,
        instruction="Search the assigned sources.",
        output=SearchCompletion,
        tools=tools or [],
        context=lambda: {"question": "Explain research evidence"},
        revision=lambda: "same-version",
        accept=accept or accept_default,
        request_task=child or request_default,
        fallback=lambda: None,
        step=inline_step,
        event=lambda *args, **kwargs: events.append(kwargs),
        bound_result=bound,
    )
    return memory, llm, accepted, events


async def test_agent_observes_tool_result_before_submitting_output():
    calls = []

    async def search(args):
        calls.append(args.queries)
        return {"chunk_ids": ["ch_found"], "count": 1}

    memory, llm, result, events = await exercise(
        [
            action("use_tool", tool="search", arguments={"queries": ["specific evidence"]}),
            action("finish", result={"chunk_ids": ["ch_found"]}),
        ],
        tools=[AgentTool("search", "Search corpus", frozenset({"queries"}), search)],
    )
    assert memory.status == "completed" and memory.steps == 2
    assert calls == [["specific evidence"]] and result[0].chunk_ids == ["ch_found"]
    assert llm.payloads[1]["last_observation"] == {"chunk_ids": ["ch_found"], "count": 1}
    assert any(e.get("session", {}).get("status") == "completed" for e in events)


async def test_unavailable_tool_and_unexpected_arguments_cannot_execute():
    calls = []

    async def safe(args):
        calls.append(args)
        return {"ok": True}

    memory, llm, _, _ = await exercise(
        [
            action("use_tool", tool="approve_claims"),
            action("use_tool", tool="read", arguments={"queries": ["forbidden argument"]}),
            action("finish", result={"chunk_ids": []}),
        ],
        tools=[AgentTool("read", "Read sources", frozenset({"chunk_ids"}), safe)],
    )
    assert memory.status == "completed" and not calls
    assert "not available" in llm.payloads[1]["last_observation"]["error"]
    assert "Unexpected arguments" in llm.payloads[2]["last_observation"]["error"]


async def test_repeated_tool_call_on_same_revision_does_not_execute_twice():
    calls = []

    async def inspect(args):
        calls.append(1)
        return {"found": False}

    memory, _, _, _ = await exercise(
        [
            action("use_tool", tool="inspect"),
            action("use_tool", tool="inspect"),
            action("blocked"),
        ],
        tools=[AgentTool("inspect", "Inspect", frozenset(), inspect)],
    )
    assert memory.status == "blocked" and len(calls) == 1
    assert any(o["kind"] == "rejected" for o in memory.observations)


async def test_rejected_finish_can_be_repaired_using_feedback():
    accepted = []

    async def accept(result):
        if result.chunk_ids != ["ch_real"]:
            raise ActionRejected("Unknown source ID.")
        accepted.append(result)

    memory, llm, _, _ = await exercise(
        [
            action("finish", result={"chunk_ids": ["ch_fake"]}),
            action("finish", result={"chunk_ids": ["ch_real"]}),
        ],
        accept=accept,
    )
    assert memory.status == "completed" and len(accepted) == 1
    assert llm.payloads[1]["last_observation"]["error"] == "Unknown source ID."


@pytest.mark.parametrize("arguments", [None, {"search_mode": "lexical"}])
async def test_finish_accepts_null_or_irrelevant_search_selector(arguments):
    memory, llm, results, _ = await exercise(
        [action("finish", arguments=arguments, result={"chunk_ids": []})]
    )
    assert memory.status == "completed" and memory.steps == len(llm.payloads) == 1
    assert len(results) == 1


async def test_unused_search_selector_does_not_reject_a_read_tool():
    calls = []

    async def read(args):
        calls.extend(args.chunk_ids)
        return {"source_units": []}

    memory, _, _, _ = await exercise(
        [
            action(
                "use_tool",
                tool="read",
                arguments={
                    "search_mode": "lexical",
                    "chunk_ids": ["ch_known"],
                },
            ),
            action("finish", result={"chunk_ids": []}),
        ],
        tools=[AgentTool("read", "Read sources", frozenset({"chunk_ids"}), read)],
    )
    assert memory.status == "completed" and calls == ["ch_known"]
    assert not any(o["kind"] == "rejected" for o in memory.observations)


async def test_finish_still_rejects_nonempty_target_arguments():
    memory, llm, results, _ = await exercise(
        [
            action(
                "finish",
                arguments={"chunk_ids": ["ch_hidden"], "search_mode": "lexical"},
                result={"chunk_ids": []},
            ),
            action("blocked"),
        ]
    )
    assert memory.status == "blocked" and not results
    assert "no tool arguments" in llm.payloads[1]["last_observation"]["error"]


async def test_search_bound_handoff_does_not_spend_another_model_call():
    searched = []

    async def search(args):
        searched.append(True)
        return {"chunk_ids": ["ch_found"]}

    memory, llm, results, _ = await exercise(
        [action("use_tool", tool="search")],
        max_steps=1,
        tools=[AgentTool("search", "Search corpus", frozenset(), search)],
        bound=lambda: SearchCompletion(chunk_ids=["ch_found"]) if searched else None,
    )
    assert memory.status == "completed" and memory.steps == len(llm.payloads) == 1
    assert results[0].chunk_ids == ["ch_found"]
    assert memory.observations[-1]["kind"] == "handoff"


@pytest.mark.parametrize("task_id,allowed", [("job_1", True), ("job_foreign", False)])
async def test_tool_can_ignore_echoed_own_task_id_but_not_foreign_id(task_id, allowed):
    calls = []

    async def search(args):
        calls.append(args.queries)
        return {"chunk_ids": []}

    memory, _, _, _ = await exercise(
        [
            action(
                "use_tool",
                tool="search",
                arguments={"queries": ["evidence"], "task_id": task_id},
            ),
            action("blocked"),
        ],
        tools=[AgentTool("search", "Search corpus", frozenset({"queries"}), search)],
    )
    assert bool(calls) is allowed
    assert any(o["kind"] == "rejected" for o in memory.observations) is not allowed


async def test_agent_suspends_and_resumes_with_private_memory_and_same_step_budget():
    memory, _, _, _ = await exercise(
        [
            action(
                "request_task",
                arguments={
                    "task": {
                        "role": "reader",
                        "instruction": "Read the relevant source section.",
                    }
                },
            ),
        ]
    )
    assert memory.status == "waiting" and memory.steps == 1
    memory, llm, result, _ = await exercise(
        [
            action("finish", result={"chunk_ids": []}),
        ],
        session=memory,
        max_steps=2,
    )
    assert memory.steps == 2 and memory.status == "completed" and len(result) == 1
    assert llm.payloads[0]["memory"][0]["kind"] == "request_task"


async def test_global_budget_exception_is_never_converted_to_tool_feedback():
    async def budget(args):
        raise BudgetExceeded("No further model calls allowed.")

    with pytest.raises(BudgetExceeded):
        await exercise(
            [action("use_tool", tool="search")],
            tools=[AgentTool("search", "Search", frozenset(), budget)],
        )


def test_task_graph_enforces_dependencies_and_rolls_back_cycles():
    graph = TaskGraph({}, 4)
    parent = graph.add(
        TaskRequest(role="writer", instruction="Compose a verified report."), "orchestrator"
    )
    child = graph.add(
        TaskRequest(role="reader", instruction="Read necessary evidence."), "writer", parent.id
    )
    assert graph.ready() == [child] and parent.status == "waiting"
    before = parent.model_dump()
    with pytest.raises(ActionRejected, match="cycle"):
        graph.add(
            TaskRequest(
                role="retrieval", instruction="Find source evidence.", depends_on=[parent.id]
            ),
            "reader",
            child.id,
        )
    assert len(graph.tasks) == 2 and parent.model_dump() == before
    child.status = "completed"
    assert graph.ready() == [parent]


def test_task_graph_rejects_unknown_dependencies_and_propagates_failed_children():
    graph = TaskGraph({}, 2)
    with pytest.raises(ActionRejected, match="does not exist"):
        graph.add(
            TaskRequest(role="reader", instruction="Read the source.", depends_on=["invented"]),
            "orchestrator",
        )
    parent = graph.add(
        TaskRequest(role="writer", instruction="Compose the report."), "orchestrator"
    )
    child = graph.add(
        TaskRequest(role="reader", instruction="Read the source."), "writer", parent.id
    )
    child.status = "blocked"
    graph.settle()
    assert parent.status == "blocked" and not graph.ready()
    with pytest.raises(ActionRejected, match="task limit"):
        graph.add(
            TaskRequest(role="retrieval", instruction="Find source evidence."), "orchestrator"
        )
