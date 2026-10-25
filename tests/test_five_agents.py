"""Five active model roles, code-only coordination, and old configuration compatibility."""

from typing import get_args

from fastapi.testclient import TestClient

from research_assist.api.app import create_app
from research_assist.config import AgentRole, Settings
from research_assist.domain.models import AgentSession, RuntimeRole, RuntimeTask

ROLES = {"retrieval", "reader", "summarizer", "fact_checker", "writer"}


def test_api_and_model_routes_expose_exactly_five_agents(settings):
    assert set(get_args(AgentRole)) == ROLES
    assert set(get_args(RuntimeRole)) == ROLES
    with TestClient(create_app(settings)) as client:
        config = client.get("/api/config").json()
        specs = client.get("/api/agents").json()
    assert {r["agent"] for r in config["agent_routes"]} == ROLES
    assert {a["role"] for a in specs["agents"]} == ROLES
    assert specs["orchestrator"]["execution"] == "local"
    assert specs["orchestrator"]["agent_count"] == 5
    assert "supervisor_max_steps" not in config


def test_retired_route_and_environment_name_do_not_break_existing_deployments(monkeypatch):
    monkeypatch.setenv("RA_SUPERVISOR_MAX_STEPS", "3")
    monkeypatch.setenv("RA_AGENT_ROUTES", '{"supervisor": {"provider": "demo"}}')
    settings = Settings(_env_file=None)
    assert settings.coordinator_max_steps == 3
    assert "supervisor" not in settings.agent_routes
    monkeypatch.setenv("RA_COORDINATOR_MAX_STEPS", "5")
    assert Settings(_env_file=None).coordinator_max_steps == 5


def test_constructor_accepts_local_coordinator_limit():
    assert Settings(coordinator_max_steps=2, _env_file=None).coordinator_max_steps == 2


def test_old_supervisor_sessions_and_task_provenance_remain_readable():
    session = AgentSession.model_validate(
        {"task_id": "supervisor", "role": "supervisor", "status": "completed"}
    )
    task = RuntimeTask.model_validate(
        {
            "id": "job_1",
            "role": "retrieval",
            "requested_by": "supervisor",
            "instruction": "Search the frozen document corpus.",
        }
    )
    assert session.role == "supervisor" and task.requested_by == "supervisor"
    assert (
        RuntimeTask(
            id="job_2", role="reader", instruction="Read the retrieved source units."
        ).requested_by
        == "orchestrator"
    )
