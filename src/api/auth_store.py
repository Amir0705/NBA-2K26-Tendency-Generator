"""Simple local authentication and role store for the web app."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from threading import Lock
from typing import Any

_ALLOWED_ROLES = {"admin", "viewer", "editor"}


class AuthStore:
    """JSON-backed user credentials with in-memory bearer sessions."""

    def __init__(self, users_path: str) -> None:
        self._users_path = users_path
        self._lock = Lock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._ensure_store()

    def _ensure_store(self) -> None:
        directory = os.path.dirname(self._users_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if os.path.isfile(self._users_path):
            return

        admin_user = os.environ.get("N2K_ADMIN_USERNAME", "admin").strip() or "admin"
        admin_pass = os.environ.get("N2K_ADMIN_PASSWORD", "admin123").strip() or "admin123"
        seed = {
            "users": {
                admin_user: {
                    "full_name": "Administrator",
                    "role": "admin",
                    "password_hash": self._hash_password(admin_pass),
                    "must_change_password": True,
                    "created_at": int(time.time()),
                }
            }
        }
        self._write(seed)

    def _read(self) -> dict[str, Any]:
        try:
            with open(self._users_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return {"users": {}}
            users = data.get("users", {})
            if not isinstance(users, dict):
                users = {}
            return {"users": users}
        except FileNotFoundError:
            return {"users": {}}
        except json.JSONDecodeError:
            return {"users": {}}

    def _write(self, data: dict[str, Any]) -> None:
        with open(self._users_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    @staticmethod
    def _hash_password(password: str, salt: str | None = None) -> str:
        if salt is None:
            salt = base64.b64encode(os.urandom(16)).decode("ascii")
        pw_bytes = password.encode("utf-8")
        salt_bytes = salt.encode("utf-8")
        digest = hashlib.pbkdf2_hmac("sha256", pw_bytes, salt_bytes, 120000)
        digest_b64 = base64.b64encode(digest).decode("ascii")
        return f"pbkdf2_sha256${salt}${digest_b64}"

    @staticmethod
    def _verify_password(password: str, encoded: str) -> bool:
        try:
            scheme, salt, digest = encoded.split("$", 2)
        except ValueError:
            return False
        if scheme != "pbkdf2_sha256":
            return False
        check = AuthStore._hash_password(password, salt=salt)
        return hmac.compare_digest(check, encoded)

    def _session_payload(self, username: str, role: str) -> dict[str, Any]:
        return {
            "username": username,
            "role": role,
            "issued_at": int(time.time()),
            "expires_at": int(time.time()) + 60 * 60 * 12,
        }

    def login(self, username: str, password: str) -> dict[str, Any] | None:
        uname = (username or "").strip()
        if not uname:
            return None

        with self._lock:
            data = self._read()
            user = data.get("users", {}).get(uname)
            if not isinstance(user, dict):
                return None

            if not self._verify_password(password, str(user.get("password_hash", ""))):
                return None

            token = secrets.token_urlsafe(36)
            role = str(user.get("role", "viewer"))
            self._sessions[token] = self._session_payload(uname, role)

            return {
                "token": token,
                "username": uname,
                "full_name": user.get("full_name") or uname,
                "role": role,
                "must_change_password": bool(user.get("must_change_password", False)),
            }

    def get_user_by_token(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None

        payload = self._sessions.get(token)
        if not payload:
            return None

        if int(payload.get("expires_at", 0)) < int(time.time()):
            self._sessions.pop(token, None)
            return None

        username = str(payload.get("username", ""))
        with self._lock:
            data = self._read()
            user = data.get("users", {}).get(username)

        if not isinstance(user, dict):
            self._sessions.pop(token, None)
            return None

        return {
            "username": username,
            "full_name": user.get("full_name") or username,
            "role": str(user.get("role", "viewer")),
            "must_change_password": bool(user.get("must_change_password", False)),
        }

    def list_users(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read()
            users = data.get("users", {})

        out: list[dict[str, Any]] = []
        for username, entry in users.items():
            if not isinstance(entry, dict):
                continue
            out.append(
                {
                    "username": username,
                    "full_name": entry.get("full_name") or username,
                    "role": entry.get("role", "viewer"),
                    "must_change_password": bool(entry.get("must_change_password", False)),
                }
            )
        out.sort(key=lambda u: str(u.get("username", "")).lower())
        return out

    def create_user(self, username: str, password: str, role: str, full_name: str | None = None) -> dict[str, Any]:
        uname = (username or "").strip()
        pwd = (password or "").strip()
        r = (role or "").strip().lower()

        if not uname:
            raise ValueError("username is required")
        if len(pwd) < 4:
            raise ValueError("password must be at least 4 characters")
        if r not in _ALLOWED_ROLES:
            raise ValueError("role must be one of admin, viewer, editor")

        with self._lock:
            data = self._read()
            users = data.setdefault("users", {})
            if uname in users:
                raise ValueError("username already exists")

            users[uname] = {
                "full_name": (full_name or uname).strip() or uname,
                "role": r,
                "password_hash": self._hash_password(pwd),
                "must_change_password": True,
                "created_at": int(time.time()),
            }
            self._write(data)

        return {
            "username": uname,
            "full_name": (full_name or uname).strip() or uname,
            "role": r,
            "must_change_password": True,
        }

    def change_password(self, username: str, new_password: str) -> None:
        uname = (username or "").strip()
        pwd = (new_password or "").strip()
        if not uname:
            raise ValueError("username is required")
        if len(pwd) < 4:
            raise ValueError("new password must be at least 4 characters")

        with self._lock:
            data = self._read()
            users = data.setdefault("users", {})
            user = users.get(uname)
            if not isinstance(user, dict):
                raise ValueError("user not found")

            user["password_hash"] = self._hash_password(pwd)
            user["must_change_password"] = False
            user["password_changed_at"] = int(time.time())
            self._write(data)

    def update_user(
        self,
        username: str,
        *,
        full_name: str | None = None,
        role: str | None = None,
        reset_password: str | None = None,
        must_change_password: bool | None = None,
    ) -> dict[str, Any]:
        uname = (username or "").strip()
        if not uname:
            raise ValueError("username is required")

        with self._lock:
            data = self._read()
            users = data.setdefault("users", {})
            user = users.get(uname)
            if not isinstance(user, dict):
                raise ValueError("user not found")

            if full_name is not None:
                candidate = full_name.strip()
                user["full_name"] = candidate or uname

            if role is not None:
                r = role.strip().lower()
                if r not in _ALLOWED_ROLES:
                    raise ValueError("role must be one of admin, viewer, editor")
                user["role"] = r

            if reset_password is not None:
                pwd = reset_password.strip()
                if len(pwd) < 4:
                    raise ValueError("reset_password must be at least 4 characters")
                user["password_hash"] = self._hash_password(pwd)
                user["must_change_password"] = True

            if must_change_password is not None:
                user["must_change_password"] = bool(must_change_password)

            user["updated_at"] = int(time.time())
            self._write(data)

            return {
                "username": uname,
                "full_name": user.get("full_name") or uname,
                "role": user.get("role", "viewer"),
                "must_change_password": bool(user.get("must_change_password", False)),
            }

    def delete_user(self, username: str, acting_username: str | None = None) -> None:
        uname = (username or "").strip()
        actor = (acting_username or "").strip()
        if not uname:
            raise ValueError("username is required")
        if actor and actor == uname:
            raise ValueError("you cannot delete your own account")

        with self._lock:
            data = self._read()
            users = data.setdefault("users", {})
            if uname not in users:
                raise ValueError("user not found")

            target = users.get(uname, {})
            if str(target.get("role", "")).lower() == "admin":
                admin_count = sum(
                    1
                    for entry in users.values()
                    if isinstance(entry, dict) and str(entry.get("role", "")).lower() == "admin"
                )
                if admin_count <= 1:
                    raise ValueError("cannot delete the last admin account")

            users.pop(uname, None)

            # Invalidate active sessions for deleted user.
            for token, session in list(self._sessions.items()):
                if str(session.get("username", "")) == uname:
                    self._sessions.pop(token, None)

            self._write(data)
