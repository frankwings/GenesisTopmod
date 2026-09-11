"""
vlm_client.py — Standalone image-capable Gemini caller via Antigravity endpoint.

Uses the Antigravity CLI OAuth token (~/.gemini/antigravity-cli/antigravity-oauth-token)
which supports inlineData (image inputs).  The ZenithLoom Gemini node is text-only;
this module is the Phase-1 VLM interface for GenesisTopmod.

Usage
-----
    from vlm_client import VLMClient

    client = VLMClient()
    reply = client.ask_with_image("What region needs more geometry?", png_bytes)
    reply = client.ask("Explain the topology plan.")

Standalone smoke test
---------------------
    python3 vlm_client.py [path/to/image.png]
    # No image arg → generates 64×64 noise PNG.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Optional

import requests

# ── Antigravity OAuth / endpoint config ───────────────────────────────────────

# Credentials from quota-watcher constants.js
_AG_CLIENT_ID     = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
_AG_CLIENT_SECRET = "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"
_TOKEN_URI        = "https://oauth2.googleapis.com/token"

# Antigravity suite endpoint (supports inlineData + generateContent)
_AG_ENDPOINT      = "https://daily-cloudcode-pa.sandbox.googleapis.com"
_AG_USER_AGENT    = "antigravity/1.11.3 linux/amd64"

# Local token file written by Antigravity CLI login
_AG_TOKEN_FILE    = (
    Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
)

_DEFAULT_MODEL    = "gemini-2.5-flash"


class VLMClient:
    """Thin wrapper around the Antigravity generateContent endpoint.

    Supports both text-only and image+text requests (inlineData).

    Parameters
    ----------
    model : str
        Gemini model name (e.g. "gemini-2.5-flash", "gemini-2.5-pro").
    jitter_multiplier : float
        Reserved for future rate-limit jitter; currently unused.
    timeout : int
        HTTP timeout in seconds for generateContent calls.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        jitter_multiplier: float = 0.0,
        timeout: int = 90,
    ) -> None:
        self._model   = model
        self._jitter  = jitter_multiplier
        self._timeout = timeout
        # Cached auth state
        self._access_token: str      = ""
        self._token_expiry: float    = 0.0   # POSIX seconds
        self._project_id: Optional[str] = None

    # ── token management ──────────────────────────────────────────────────────

    def _load_refresh_token(self) -> str:
        if not _AG_TOKEN_FILE.exists():
            raise FileNotFoundError(
                f"Antigravity OAuth token not found: {_AG_TOKEN_FILE}\n"
                "Please log in via Antigravity CLI first."
            )
        data = json.loads(_AG_TOKEN_FILE.read_text())
        token = data.get("token", {})
        rt = token.get("refresh_token", "")
        if not rt:
            raise ValueError(
                f"No refresh_token in {_AG_TOKEN_FILE}.\n"
                "Re-login via Antigravity CLI to refresh credentials."
            )
        return rt

    def _ensure_token(self) -> str:
        """Return a valid access token, refreshing if within 5 min of expiry."""
        if self._access_token and time.time() < self._token_expiry - 300:
            return self._access_token

        refresh_token = self._load_refresh_token()
        resp = requests.post(
            _TOKEN_URI,
            data={
                "client_id":     _AG_CLIENT_ID,
                "client_secret": _AG_CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type":    "refresh_token",
            },
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(
                f"OAuth token refresh failed ({resp.status_code}): {resp.text[:300]}"
            )
        data = resp.json()
        self._access_token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600)
        return self._access_token

    def _get_project_id(self, token: str) -> str:
        """Resolve the Antigravity project ID via loadCodeAssist."""
        if self._project_id is not None:
            return self._project_id

        metadata = {
            "ideType":    "ANTIGRAVITY",
            "platform":   "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
        resp = requests.post(
            f"{_AG_ENDPOINT}/v1internal:loadCodeAssist",
            headers=self._headers(token),
            json={"metadata": metadata},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(
                f"loadCodeAssist failed ({resp.status_code}): {resp.text[:300]}"
            )
        result = resp.json()

        project = result.get("cloudaicompanionProject")
        if isinstance(project, dict):
            project = project.get("id") or project.get("projectId")

        if not project:
            # Fallback: call onboardUser with the default allowed tier
            tier_id = ""
            for tier in result.get("allowedTiers") or []:
                if tier.get("isDefault") and tier.get("id"):
                    tier_id = tier["id"]
                    break
            if not tier_id and result.get("allowedTiers"):
                tier_id = result["allowedTiers"][0].get("id", "")
            if tier_id:
                r2 = requests.post(
                    f"{_AG_ENDPOINT}/v1internal:onboardUser",
                    headers=self._headers(token),
                    json={"tierId": tier_id, "metadata": metadata},
                    timeout=30,
                )
                if r2.ok:
                    ob = r2.json()
                    project = (ob.get("response") or {}).get("cloudaicompanionProject")
                    if isinstance(project, dict):
                        project = project.get("id") or project.get("projectId")

        if not project:
            raise RuntimeError(
                f"Could not resolve Antigravity project ID. "
                f"loadCodeAssist response: {json.dumps(result)[:400]}"
            )

        self._project_id = str(project)
        return self._project_id

    def _headers(self, token: str) -> dict:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "User-Agent":    _AG_USER_AGENT,
        }

    # ── low-level generateContent call ────────────────────────────────────────

    def _generate(self, parts: list) -> str:
        """POST to :generateContent and return the first text candidate."""
        token      = self._ensure_token()
        project_id = self._get_project_id(token)

        body = {
            "model":   self._model,
            "project": project_id,
            "request": {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature":     0.2,
                    "maxOutputTokens": 4096,
                },
            },
        }

        resp = requests.post(
            f"{_AG_ENDPOINT}/v1internal:generateContent",
            headers=self._headers(token),
            json=body,
            timeout=self._timeout,
        )

        if not resp.ok:
            raise RuntimeError(
                f"generateContent failed ({resp.status_code}): {resp.text[:600]}"
            )

        result = resp.json()
        # Navigate: result["response"]["candidates"][0]["content"]["parts"][0]["text"]
        try:
            return result["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected response shape: {json.dumps(result)[:500]}"
            ) from exc

    # ── public API ────────────────────────────────────────────────────────────

    def ask_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> str:
        """Send a prompt alongside an image (inlineData) and return the reply.

        Parameters
        ----------
        prompt : str
            Text prompt to accompany the image.
        image_bytes : bytes
            Raw image bytes (PNG, JPEG, etc.).
        mime_type : str
            MIME type of *image_bytes*, e.g. "image/png".

        Returns
        -------
        str
            Model reply text.
        """
        b64_data = base64.b64encode(image_bytes).decode("ascii")
        parts = [
            {"text": prompt},
            {"inlineData": {"mimeType": mime_type, "data": b64_data}},
        ]
        return self._generate(parts)

    def ask(self, prompt: str) -> str:
        """Text-only request — for prompts that don't need an image."""
        return self._generate([{"text": prompt}])


# ── smoke test ────────────────────────────────────────────────────────────────

def _make_test_png() -> bytes:
    """Generate a 64×64 noise image as PNG bytes."""
    import io
    import numpy as np
    try:
        from PIL import Image
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Minimal valid 1×1 white PNG (no PIL needed)
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )


if __name__ == "__main__":
    import sys

    print("── VLMClient smoke test ──────────────────────────────")

    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        if not img_path.exists():
            print(f"[ERROR] File not found: {img_path}")
            sys.exit(1)
        image_bytes = img_path.read_bytes()
        print(f"Using image: {img_path}  ({len(image_bytes):,} bytes)")
    else:
        print("No image argument — generating 64×64 noise PNG...")
        image_bytes = _make_test_png()
        print(f"Test PNG: {len(image_bytes):,} bytes")

    client = VLMClient(model=_DEFAULT_MODEL)

    prompt = "Describe this image in one sentence."
    print(f"\nPrompt: {prompt!r}")
    print(f"Endpoint: {_AG_ENDPOINT}")
    print("Sending generateContent with inlineData ...")

    t0 = time.time()
    try:
        reply = client.ask_with_image(prompt, image_bytes)
        elapsed = time.time() - t0
        print(f"\n✅  SUCCESS ({elapsed:.1f}s)")
        print(f"Reply: {reply[:400]}")

        # Also test text-only path
        print("\nTesting text-only ask() ...")
        t1 = time.time()
        reply2 = client.ask("Say 'ok' in exactly one word.")
        print(f"✅  Text reply ({time.time()-t1:.1f}s): {reply2.strip()[:80]}")

    except RuntimeError as exc:
        elapsed = time.time() - t0
        msg = str(exc)
        print(f"\n❌  FAILED ({elapsed:.1f}s)")
        print(f"Error: {msg}")
        sys.exit(1)
