"""Firebase-backed auth store with role/user metadata in Firestore."""
from __future__ import annotations

import json
import os
import time
import importlib
from typing import Any

import requests

_ALLOWED_ROLES = {"admin", "viewer", "editor"}


class FirebaseAuthStore:
    """Firebase Authentication + Firestore user profile adapter."""

    def __init__(self) -> None:
        self._api_key = (os.environ.get("FIREBASE_API_KEY") or "").strip()
        self._project_id = (os.environ.get("FIREBASE_PROJECT_ID") or "").strip()
        if not self._api_key:
            raise ValueError("FIREBASE_API_KEY is required")

        self._init_firebase_admin()
        self._db = self._firestore_module().client()
        self._users_col = self._db.collection("app_users")
        self._ensure_admin_seed()

    @staticmethod
    def _username_to_email(username: str) -> str:
        uname = (username or "").strip().lower()
        if not uname:
            return ""
        if "@" in uname:
            return uname
        return f"{uname}@atd.local"

    @staticmethod
    def _email_to_username(email: str) -> str:
        value = (email or "").strip().lower()
        if value.endswith("@atd.local"):
            return value[:-10]
        return value

    def _init_firebase_admin(self) -> None:
        firebase_admin = importlib.import_module("firebase_admin")
        credentials = importlib.import_module("firebase_admin.credentials")
        try:
            firebase_admin.get_app()
            return
        except Exception:
            pass

        sa_json = (os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or "").strip()
        sa_path = (os.environ.get("FIREBASE_SERVICE_ACCOUNT_FILE") or "").strip()

        cred = None
        if sa_json:
            cred = credentials.Certificate(json.loads(sa_json))
        elif sa_path:
            cred = credentials.Certificate(sa_path)
        else:
            # Relies on GOOGLE_APPLICATION_CREDENTIALS or Render secret file mount.
            cred = credentials.ApplicationDefault()

        options: dict[str, Any] = {}
        if self._project_id:
            options["projectId"] = self._project_id

        firebase_admin.initialize_app(cred, options=options)

    def _profile_ref(self, uid: str):
        return self._users_col.document(uid)

    @staticmethod
    def _auth_module() -> Any:
        return importlib.import_module("firebase_admin.auth")

    @staticmethod
    def _firestore_module() -> Any:
        return importlib.import_module("firebase_admin.firestore")

    def _profile_for_uid(self, uid: str) -> dict[str, Any]:
        snap = self._profile_ref(uid).get()
        if not snap.exists:
            return {}
        data = snap.to_dict() or {}
        if not isinstance(data, dict):
            return {}
        return data

    def _upsert_profile(self, uid: str, data: dict[str, Any]) -> None:
        now = int(time.time())
        payload = dict(data)
        payload["updated_at"] = now
        ref = self._profile_ref(uid)
        snap = ref.get()
        if not snap.exists:
            payload["created_at"] = now
            ref.set(payload)
        else:
            ref.set(payload, merge=True)

    def _ensure_admin_seed(self) -> None:
        auth = self._auth_module()
        admin_username = (os.environ.get("N2K_ADMIN_USERNAME") or "admin").strip().lower() or "admin"
        admin_password = (os.environ.get("N2K_ADMIN_PASSWORD") or "admin123").strip() or "admin123"
        admin_full_name = (os.environ.get("N2K_ADMIN_FULL_NAME") or "Administrator").strip() or "Administrator"
        email = self._username_to_email(admin_username)

        try:
            user = auth.get_user_by_email(email)
        except auth.UserNotFoundError:
            user = auth.create_user(email=email, password=admin_password, display_name=admin_full_name)

        self._upsert_profile(
            user.uid,
            {
                "username": admin_username,
                "full_name": admin_full_name,
                "role": "admin",
                "must_change_password": True,
                "email": email,
            },
        )

    def _firebase_sign_in(self, email: str, password: str) -> dict[str, Any] | None:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self._api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        try:
            response = requests.post(url, json=payload, timeout=20)
            if response.status_code != 200:
                return None
            data = response.json()
            if not isinstance(data, dict):
                return None
            return data
        except requests.RequestException:
            return None

    def login(self, username: str, password: str) -> dict[str, Any] | None:
        uname = (username or "").strip().lower()
        if not uname:
            return None
        email = self._username_to_email(uname)
        signed = self._firebase_sign_in(email=email, password=password)
        if not signed:
            return None

        token = str(signed.get("idToken", ""))
        uid = str(signed.get("localId", ""))
        if not token or not uid:
            return None

        profile = self._profile_for_uid(uid)
        role = str(profile.get("role", "viewer")).lower()
        if role not in _ALLOWED_ROLES:
            role = "viewer"

        full_name = str(profile.get("full_name") or signed.get("displayName") or uname)
        must_change = bool(profile.get("must_change_password", False))

        if not profile:
            self._upsert_profile(
                uid,
                {
                    "username": uname,
                    "full_name": full_name,
                    "role": role,
                    "must_change_password": must_change,
                    "email": email,
                },
            )

        return {
            "token": token,
            "username": uname,
            "full_name": full_name,
            "role": role,
            "must_change_password": must_change,
        }

    def get_user_by_token(self, token: str) -> dict[str, Any] | None:
        auth = self._auth_module()
        raw = (token or "").strip()
        if not raw:
            return None
        try:
            decoded = auth.verify_id_token(raw)
        except Exception:
            return None

        uid = str(decoded.get("uid") or "")
        if not uid:
            return None

        profile = self._profile_for_uid(uid)
        email = str(decoded.get("email") or profile.get("email") or "")
        username = str(profile.get("username") or self._email_to_username(email) or uid)
        full_name = str(profile.get("full_name") or decoded.get("name") or username)
        role = str(profile.get("role", "viewer")).lower()
        if role not in _ALLOWED_ROLES:
            role = "viewer"
        must_change = bool(profile.get("must_change_password", False))

        if not profile:
            self._upsert_profile(
                uid,
                {
                    "username": username,
                    "full_name": full_name,
                    "role": role,
                    "must_change_password": must_change,
                    "email": email,
                },
            )

        return {
            "username": username,
            "full_name": full_name,
            "role": role,
            "must_change_password": must_change,
            "uid": uid,
            "email": email,
        }

    def list_users(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for doc in self._users_col.stream():
            data = doc.to_dict() or {}
            username = str(data.get("username") or "")
            if not username:
                username = doc.id
            role = str(data.get("role", "viewer")).lower()
            if role not in _ALLOWED_ROLES:
                role = "viewer"
            out.append(
                {
                    "username": username,
                    "full_name": str(data.get("full_name") or username),
                    "role": role,
                    "must_change_password": bool(data.get("must_change_password", False)),
                }
            )
        out.sort(key=lambda u: str(u.get("username", "")).lower())
        return out

    def _find_uid_by_username(self, username: str) -> str:
        auth = self._auth_module()
        uname = (username or "").strip().lower()
        if not uname:
            raise ValueError("username is required")

        docs = list(self._users_col.where("username", "==", uname).limit(1).stream())
        if docs:
            return docs[0].id

        email = self._username_to_email(uname)
        try:
            user = auth.get_user_by_email(email)
            return user.uid
        except auth.UserNotFoundError as exc:
            raise ValueError("user not found") from exc

    def create_user(self, username: str, password: str, role: str, full_name: str | None = None) -> dict[str, Any]:
        auth = self._auth_module()
        uname = (username or "").strip().lower()
        pwd = (password or "").strip()
        r = (role or "").strip().lower()
        name = (full_name or uname).strip() or uname

        if not uname:
            raise ValueError("username is required")
        if len(pwd) < 4:
            raise ValueError("password must be at least 4 characters")
        if r not in _ALLOWED_ROLES:
            raise ValueError("role must be one of admin, viewer, editor")

        email = self._username_to_email(uname)
        try:
            auth.get_user_by_email(email)
            raise ValueError("username already exists")
        except auth.UserNotFoundError:
            pass

        user = auth.create_user(email=email, password=pwd, display_name=name)
        self._upsert_profile(
            user.uid,
            {
                "username": uname,
                "full_name": name,
                "role": r,
                "must_change_password": True,
                "email": email,
            },
        )

        return {"username": uname, "full_name": name, "role": r, "must_change_password": True}

    def change_password(self, username: str, new_password: str) -> None:
        auth = self._auth_module()
        uname = (username or "").strip().lower()
        pwd = (new_password or "").strip()
        if not uname:
            raise ValueError("username is required")
        if len(pwd) < 4:
            raise ValueError("new password must be at least 4 characters")

        uid = self._find_uid_by_username(uname)
        auth.update_user(uid, password=pwd)
        self._upsert_profile(uid, {"must_change_password": False})

    def update_user(
        self,
        username: str,
        *,
        full_name: str | None = None,
        role: str | None = None,
        reset_password: str | None = None,
        must_change_password: bool | None = None,
    ) -> dict[str, Any]:
        auth = self._auth_module()
        uname = (username or "").strip().lower()
        if not uname:
            raise ValueError("username is required")

        uid = self._find_uid_by_username(uname)
        current = self._profile_for_uid(uid)

        payload: dict[str, Any] = {}
        if full_name is not None:
            candidate = full_name.strip()
            payload["full_name"] = candidate or uname
            auth.update_user(uid, display_name=payload["full_name"])

        if role is not None:
            r = role.strip().lower()
            if r not in _ALLOWED_ROLES:
                raise ValueError("role must be one of admin, viewer, editor")
            payload["role"] = r

        if reset_password is not None:
            pwd = reset_password.strip()
            if len(pwd) < 4:
                raise ValueError("reset_password must be at least 4 characters")
            auth.update_user(uid, password=pwd)
            payload["must_change_password"] = True

        if must_change_password is not None:
            payload["must_change_password"] = bool(must_change_password)

        if payload:
            self._upsert_profile(uid, payload)

        merged = dict(current)
        merged.update(payload)
        role_out = str(merged.get("role", "viewer")).lower()
        if role_out not in _ALLOWED_ROLES:
            role_out = "viewer"

        return {
            "username": uname,
            "full_name": str(merged.get("full_name") or uname),
            "role": role_out,
            "must_change_password": bool(merged.get("must_change_password", False)),
        }

    def delete_user(self, username: str, acting_username: str | None = None) -> None:
        auth = self._auth_module()
        uname = (username or "").strip().lower()
        actor = (acting_username or "").strip().lower()
        if not uname:
            raise ValueError("username is required")
        if actor and actor == uname:
            raise ValueError("you cannot delete your own account")

        uid = self._find_uid_by_username(uname)
        target = self._profile_for_uid(uid)

        if str(target.get("role", "")).lower() == "admin":
            admins = [
                d
                for d in self._users_col.where("role", "==", "admin").stream()
            ]
            if len(admins) <= 1:
                raise ValueError("cannot delete the last admin account")

        auth.delete_user(uid)
        self._profile_ref(uid).delete()
