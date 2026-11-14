"""Bounded model-catalog check for operator-configured targets only."""

import json

import httpx

from research_assist.config import Settings


async def probe_model(route: Settings, client: httpx.AsyncClient) -> dict[str, object]:
    if route.llm_provider == "demo":
        return {"reachable": True, "model_found": True, "note": "Demo trích xuất; không chạy LLM."}
    if route.llm_provider == "gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        headers = {"x-goog-api-key": route.gemini_api_key.get_secret_value()}
    else:
        provider = route.llm_provider
        base: str = getattr(route, f"{provider}_base_url")
        key = (
            "ollama"
            if provider == "ollama"
            else getattr(route, f"{provider}_api_key").get_secret_value()
        )
        url, headers = base.rstrip("/") + "/models", {"Authorization": f"Bearer {key}"}
    try:
        async with client.stream(
            "GET", url, headers=headers, timeout=10, follow_redirects=False
        ) as response:
            if response.status_code != 200:
                return {
                    "reachable": False,
                    "model_found": None,
                    "note": (
                        f"Danh mục model trả HTTP {response.status_code}; chưa xác nhận suy luận."
                    ),
                }
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > 2_000_000:
                    raise ValueError("catalog too large")
        payload = json.loads(data)
        entries = payload.get("models", payload.get("data", []))
        models = {
            str(item.get("id", item.get("name", ""))).removeprefix("models/")
            for item in entries
            if isinstance(item, dict)
        }
        found = route.resolved_model.removeprefix("models/") in models
        return {
            "reachable": True,
            "model_found": found,
            "note": "Kết nối được danh mục model. Chưa kiểm tra sinh JSON, VRAM hoặc chất lượng."
            if found
            else "Endpoint phản hồi nhưng không thấy model đã cấu hình trong danh mục.",
        }
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return {
            "reachable": False,
            "model_found": None,
            "note": "Không đọc được danh mục model; kiểm tra endpoint và token trên server.",
        }
