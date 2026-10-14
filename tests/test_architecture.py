import ast
from pathlib import Path


def test_domain_and_agents_do_not_depend_on_infrastructure_or_transport() -> None:
    root = Path(__file__).parents[1] / "src" / "research_assist"
    for layer in ("domain", "agents"):
        for file in (root / layer).glob("*.py"):
            tree = ast.parse(file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    assert not name.startswith(
                        (
                            "fastapi",
                            "httpx",
                            "sqlite3",
                            "sentence_transformers",
                            "research_assist.infrastructure",
                            "research_assist.api",
                            "research_assist.application",
                        )
                    ), (file.name, name)


def test_web_does_not_interpret_model_or_document_html() -> None:
    web = Path(__file__).parents[1] / "src" / "research_assist" / "web"
    for file in web.glob("*.js"):
        text = file.read_text(encoding="utf-8")
        assert ".innerHTML" not in text and "insertAdjacentHTML(" not in text
