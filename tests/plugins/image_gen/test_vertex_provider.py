"""Tests for the Google Vertex AI image gen plugin — registration, model
routing, and request/response handling (no network, no real credentials)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import requests

import plugins.image_gen.vertex as vertex_plugin
from agent import image_gen_registry
from plugins.image_gen.vertex import (
    DEFAULT_MODEL,
    VertexImageGenProvider,
    _infer_api,
    _resolve_model,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


@pytest.fixture(autouse=True)
def _no_env_model(monkeypatch):
    monkeypatch.delenv("VERTEX_IMAGE_MODEL", raising=False)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.headers = {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


def _fake_credentials(monkeypatch, token="tok", project="proj"):
    monkeypatch.setattr(
        "agent.vertex_adapter.get_vertex_credentials",
        lambda credentials_path=None: (token, project),
    )


# ---------------------------------------------------------------------------
# Registration / catalog surface
# ---------------------------------------------------------------------------


def test_vertex_provider_registers():
    provider = VertexImageGenProvider()
    image_gen_registry.register_provider(provider)

    assert image_gen_registry.get_provider("vertex") is provider
    assert provider.display_name == "Google Vertex AI"
    assert provider.default_model() == DEFAULT_MODEL


def test_register_entry_point_wires_provider():
    class _Ctx:
        def __init__(self):
            self.registered = None

        def register_image_gen_provider(self, provider):
            self.registered = provider

    ctx = _Ctx()
    vertex_plugin.register(ctx)
    assert isinstance(ctx.registered, VertexImageGenProvider)


def test_catalog_invariants():
    models = VertexImageGenProvider().list_models()
    ids = [m["id"] for m in models]

    assert DEFAULT_MODEL in ids
    for model in models:
        assert model["id"]
        modalities = model.get("modalities", [])
        if model["id"].startswith("gemini"):
            assert "image" in modalities  # Gemini image models support editing
        if model["id"].startswith("imagen"):
            assert modalities == ["text"]  # :predict is text-to-image only


def test_capabilities_advertise_editing():
    caps = VertexImageGenProvider().capabilities()
    assert caps["modalities"] == ["text", "image"]
    assert caps["max_reference_images"] > 0


# ---------------------------------------------------------------------------
# Model / API routing
# ---------------------------------------------------------------------------


def test_infer_api_by_prefix():
    assert _infer_api("gemini-3-pro-image-preview") == "generate_content"
    assert _infer_api("gemini-3.5-flash-image") == "generate_content"
    assert _infer_api("imagen-4.0-generate-001") == "predict"
    assert _infer_api("mystery-model") is None


def test_infer_api_honors_config_override(monkeypatch):
    monkeypatch.setattr(vertex_plugin, "_load_vertex_section", lambda: {"api": "predict"})
    assert _infer_api("mystery-model") == "predict"


def test_resolve_model_ignores_stale_foreign_ids():
    # image_gen.model may hold an id from a previously-selected backend
    # (e.g. a FAL catalog id) — that must not be sent to Vertex.
    assert _resolve_model("fal-ai/flux-2/dev") == DEFAULT_MODEL
    assert _resolve_model(None) == DEFAULT_MODEL
    assert _resolve_model("gemini-3.5-flash-image") == "gemini-3.5-flash-image"
    # Chat-style ids with the publisher prefix are normalized, not rejected.
    assert _resolve_model("google/gemini-3.1-flash-image") == "gemini-3.1-flash-image"


def test_resolve_model_env_and_config_precedence(monkeypatch):
    monkeypatch.setattr(
        vertex_plugin, "_load_vertex_section", lambda: {"model": "imagen-4.0-fast-generate-001"}
    )
    assert _resolve_model(None) == "imagen-4.0-fast-generate-001"
    monkeypatch.setenv("VERTEX_IMAGE_MODEL", "gemini-2.5-flash-image")
    assert _resolve_model(None) == "gemini-2.5-flash-image"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_unavailable_without_credentials(monkeypatch):
    monkeypatch.setattr("agent.vertex_adapter.has_vertex_credentials", lambda: False)
    assert VertexImageGenProvider().is_available() is False


def test_available_with_credentials(monkeypatch):
    monkeypatch.setattr("agent.vertex_adapter.has_vertex_credentials", lambda: True)
    assert VertexImageGenProvider().is_available() is True


def test_generate_requires_credentials(monkeypatch):
    monkeypatch.setattr(
        "agent.vertex_adapter.get_vertex_credentials",
        lambda credentials_path=None: (None, None),
    )
    result = VertexImageGenProvider().generate("a red panda")
    assert result["success"] is False
    assert result["error_type"] == "auth_required"


# ---------------------------------------------------------------------------
# generateContent (Gemini image) path
# ---------------------------------------------------------------------------


def test_gemini_text_to_image_request_and_response(monkeypatch):
    _fake_credentials(monkeypatch)
    png_b64 = base64.b64encode(b"fakepng").decode("ascii")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        return _FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": png_b64}}]}}
                ],
                "usageMetadata": {"totalTokenCount": 42},
            }
        )

    monkeypatch.setattr(vertex_plugin.requests, "post", fake_post)

    result = VertexImageGenProvider().generate("a red panda", "landscape")

    assert result["success"] is True
    assert result["provider"] == "vertex"
    assert result["modality"] == "text"
    assert result["model"] == DEFAULT_MODEL
    assert result["usage"] == {"totalTokenCount": 42}

    saved = Path(result["image"])
    assert saved.is_file()
    assert saved.read_bytes() == b"fakepng"

    # Gemini 3.x previews are global-endpoint-only by default.
    assert captured["url"] == (
        "https://aiplatform.googleapis.com/v1/projects/proj/locations/global"
        f"/publishers/google/models/{DEFAULT_MODEL}:generateContent"
    )
    assert captured["headers"]["Authorization"] == "Bearer tok"
    payload = captured["payload"]
    assert payload["contents"][0]["parts"][0] == {"text": "a red panda"}
    gen_config = payload["generationConfig"]
    assert "IMAGE" in gen_config["responseModalities"]
    assert gen_config["imageConfig"]["aspectRatio"] == "16:9"


def test_gemini_edit_includes_source_image_and_skips_aspect(monkeypatch):
    _fake_credentials(monkeypatch)
    png_b64 = base64.b64encode(b"editme").decode("ascii")
    out_b64 = base64.b64encode(b"edited").decode("ascii")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _FakeResponse(
            {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": out_b64}}]}}]}
        )

    monkeypatch.setattr(vertex_plugin.requests, "post", fake_post)

    result = VertexImageGenProvider().generate(
        "make it night-time",
        "landscape",
        image_url=f"data:image/png;base64,{png_b64}",
    )

    assert result["success"] is True
    assert result["modality"] == "image"
    parts = captured["payload"]["contents"][0]["parts"]
    assert parts[0] == {"text": "make it night-time"}
    assert parts[1]["inlineData"]["data"] == png_b64
    # Output geometry follows the input image on edits.
    assert "aspectRatio" not in captured["payload"]["generationConfig"].get("imageConfig", {})


def test_edit_with_predict_only_model_reroutes_to_gemini(monkeypatch):
    _fake_credentials(monkeypatch)
    png_b64 = base64.b64encode(b"src").decode("ascii")
    out_b64 = base64.b64encode(b"out").decode("ascii")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        return _FakeResponse(
            {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": out_b64}}]}}]}
        )

    monkeypatch.setattr(vertex_plugin.requests, "post", fake_post)

    result = VertexImageGenProvider().generate(
        "restyle",
        image_url=f"data:image/png;base64,{png_b64}",
        model="imagen-4.0-generate-001",
    )

    assert result["success"] is True
    assert f"/models/{DEFAULT_MODEL}:generateContent" in captured["url"]


def test_gemini_safety_block_surfaces_reason(monkeypatch):
    _fake_credentials(monkeypatch)
    monkeypatch.setattr(
        vertex_plugin.requests,
        "post",
        lambda *a, **k: _FakeResponse({"promptFeedback": {"blockReason": "SAFETY"}}),
    )

    result = VertexImageGenProvider().generate("something disallowed")
    assert result["success"] is False
    assert result["error_type"] == "safety_blocked"
    assert "SAFETY" in result["error"]


# ---------------------------------------------------------------------------
# :predict (Imagen) path
# ---------------------------------------------------------------------------


def test_imagen_predict_request_and_response(monkeypatch):
    _fake_credentials(monkeypatch)
    png_b64 = base64.b64encode(b"imagenpng").decode("ascii")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _FakeResponse(
            {"predictions": [{"bytesBase64Encoded": png_b64, "mimeType": "image/png"}]}
        )

    monkeypatch.setattr(vertex_plugin.requests, "post", fake_post)

    result = VertexImageGenProvider().generate(
        "a lighthouse", "portrait", model="imagen-4.0-generate-001"
    )

    assert result["success"] is True
    assert result["model"] == "imagen-4.0-generate-001"
    assert Path(result["image"]).read_bytes() == b"imagenpng"

    # Imagen :predict is served regionally, not on the global endpoint.
    assert captured["url"] == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1"
        "/publishers/google/models/imagen-4.0-generate-001:predict"
    )
    payload = captured["payload"]
    assert payload["instances"] == [{"prompt": "a lighthouse"}]
    assert payload["parameters"]["aspectRatio"] == "9:16"
    assert payload["parameters"]["sampleCount"] == 1


def test_imagen_filtered_response_is_an_error(monkeypatch):
    _fake_credentials(monkeypatch)
    monkeypatch.setattr(
        vertex_plugin.requests,
        "post",
        lambda *a, **k: _FakeResponse({"predictions": [{"raiFilteredReason": "blocked by policy"}]}),
    )

    result = VertexImageGenProvider().generate("x", model="imagen-4.0-generate-001")
    assert result["success"] is False
    assert result["error_type"] == "empty_response"
    assert "blocked by policy" in result["error"]
