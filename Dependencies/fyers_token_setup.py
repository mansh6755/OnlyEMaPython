"""Interactive Fyers OAuth setup that stores the access token in Dependencies/.env."""

from __future__ import annotations

import os
import secrets
import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV_FILE = Path(__file__).resolve().parent / ".env"


def _store_token(token: str) -> None:
    if ENV_FILE.exists():
        content = ENV_FILE.read_text(encoding="utf-8").splitlines()
        mode = 0o600
    else:
        content = []
        mode = 0o600
    replacement = f"FYERS_ACCESS_TOKEN={token}"
    found = False
    updated: list[str] = []
    for line in content:
        if line.startswith("FYERS_ACCESS_TOKEN="):
            updated.append(replacement)
            found = True
        else:
            updated.append(line)
    if not found:
        updated.append(replacement)
    ENV_FILE.write_text("\n".join(updated) + "\n", encoding="utf-8")
    os.chmod(ENV_FILE, mode)


def main() -> int:
    from Dependencies.fyers_api import build_auth_url, exchange_auth_code

    load_dotenv(ENV_FILE, override=True)
    app_id = os.getenv("FYERS_APP_ID", "").strip()
    secret_key = os.getenv("FYERS_SECRET_KEY", "").strip()
    redirect_uri = os.getenv("FYERS_REDIRECT_URI", "").strip()
    missing = [
        key
        for key, value in (
            ("FYERS_APP_ID", app_id),
            ("FYERS_SECRET_KEY", secret_key),
            ("FYERS_REDIRECT_URI", redirect_uri),
        )
        if not value
    ]
    if missing:
        print("Configure these keys in Dependencies/.env first: " + ", ".join(missing))
        return 2

    state = secrets.token_urlsafe(24)
    auth_url = build_auth_url(app_id, redirect_uri, state)
    print("Opening Fyers authorization in your browser.")
    print("After login, paste the complete URL from the registered redirect page.")
    print(auth_url)
    webbrowser.open(auth_url)
    callback_url = input("Redirect URL: ").strip()
    query = parse_qs(urlparse(callback_url).query)
    auth_code = query.get("auth_code", [""])[0].strip()
    returned_state = query.get("state", [""])[0].strip()
    if not auth_code or returned_state != state:
        print("Fyers callback was missing its authorization code or state did not match.")
        return 2

    try:
        token = exchange_auth_code(app_id, secret_key, auth_code)
        _store_token(token)
    except Exception:
        print("Fyers token setup failed; check the app settings and try again.")
        return 1
    print("Fyers access token saved to Dependencies/.env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
