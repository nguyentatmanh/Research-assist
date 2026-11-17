"""Shared authentication and access policy for the Kaggle processes."""

import os
import re
import secrets

from fastapi import HTTPException, Request

ACCESS_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{32,256}")
LOOPBACK = "127.0.0.1"
ALLOWED_HOSTS = ["localhost", LOOPBACK, "*.trycloudflare.com"]


def valid_access_token(token: str) -> bool:
    return ACCESS_TOKEN_PATTERN.fullmatch(token) is not None


def require_bearer(request: Request) -> None:
    token = os.environ.get("RA_ACCESS_TOKEN", "")
    if not valid_access_token(token):
        raise HTTPException(503, "Cấu hình RA_ACCESS_TOKEN trước khi gọi dịch vụ.")
    expected = f"Bearer {token}".encode()
    supplied = request.headers.get("authorization", "").encode()
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            401, "Mã truy cập không hợp lệ.", headers={"WWW-Authenticate": "Bearer"}
        )
