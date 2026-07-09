"""Tests for the Google Vertex AI (Veo) video gen plugin — registration,
model routing, parameter clamping, and the predictLongRunning/poll flow
(no network, no real credentials)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import requests

import plugins.video_gen.vertex as vertex_plugin
from agent import video_gen_registry
from plugins.video_gen.vertex import (
    DEFAULT_MODEL,
    VertexVideoGenProvider,
    _clamp_duration,
    _extract_videos,
    _resolve_model,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


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


def _fake_operation_roundtrip(monkeypatch, *, videos, op_name="projects/proj/operations/op1"):
    """Stub requests.post: first call submits, later calls poll to done."""
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "payload": json, "headers": headers})
        if url.endswith(":predictLongRunning"):
            return _FakeResponse({"name": op_name})
        return _FakeResponse({"done": True, "response": {"videos": videos}})

    monkeypatch.setattr(vertex_plugin.requests, "post", fake_post)
    return calls


# ---------------------------------------------------------------------------
# Registration / catalog surface
# ---------------------------------------------------------------------------


def test_vertex_provider_registers():
    provider = VertexVideoGenProvider()
    video_gen_registry.register_provider(provider)

    assert video_gen_registry.get_provider("vertex") is provider
    assert provider.display_name == "Google Vertex AI"
    assert provider.default_model() == DEFAULT_MODEL


def test_register_entry_point_wires_provider():
    class _Ctx:
        def __init__(self):
            self.registered = None

        def register_video_gen_provider(self, provider):
            self.registered = provider

    ctx = _Ctx()
    vertex_plugin.register(ctx)
    assert isinstance(ctx.registered, VertexVideoGenProvider)


def test_catalog_invariants():
    models = VertexVideoGenProvider().list_models()
    ids = [m["id"] for m in models]

    assert DEFAULT_MODEL in ids
    for model in models:
        # Every catalog id must route to one of the two Vertex video APIs.
        assert model["id"].startswith(("veo", "gemini"))
        assert "text" in model.get("modalities", [])


def test_capabilities_declare_veo_surface():
    caps = VertexVideoGenProvider().capabilities()
    assert caps["modalities"] == ["text", "image"]
    assert caps["supports_audio"] is True
    assert caps["supports_negative_prompt"] is True
    assert caps["max_reference_images"] == 3


# ---------------------------------------------------------------------------
# Model resolution / parameter clamping
# ---------------------------------------------------------------------------


def test_resolve_model_passthrough_and_stale_fallback():
    # Future Veo releases must pass through without a plugin update.
    assert _resolve_model("veo-4.0-generate-001", explicit=False) == "veo-4.0-generate-001"
    # Chat-style ids with the publisher prefix are normalized, not rejected.
    assert (
        _resolve_model("google/veo-3.1-fast-generate-001", explicit=False)
        == "veo-3.1-fast-generate-001"
    )
    # Gemini Omni ids route to the Interactions API rather than falling back.
    assert (
        _resolve_model("google/gemini-omni-flash-preview", explicit=False)
        == "gemini-omni-flash-preview"
    )
    # A stale video_gen.model from a previously-selected backend is ignored…
    assert _resolve_model("fal-ai/kling-video/o3", explicit=False) == DEFAULT_MODEL
    # …unless the tool call requested it explicitly.
    assert _resolve_model("some-exotic-id", explicit=True) == "some-exotic-id"
    assert _resolve_model(None, explicit=False) == DEFAULT_MODEL


def test_clamp_duration_snaps_to_veo_accepted_seconds():
    assert _clamp_duration(None) is None
    assert _clamp_duration(3) == 4
    assert _clamp_duration(5) == 6
    assert _clamp_duration(7) == 8
    assert _clamp_duration(12) == 8


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_unavailable_without_credentials(monkeypatch):
    monkeypatch.setattr("agent.vertex_adapter.has_vertex_credentials", lambda: False)
    assert VertexVideoGenProvider().is_available() is False


def test_generate_requires_credentials(monkeypatch):
    monkeypatch.setattr(
        "agent.vertex_adapter.get_vertex_credentials",
        lambda credentials_path=None: (None, None),
    )
    result = VertexVideoGenProvider().generate("a happy dog")
    assert result["success"] is False
    assert result["error_type"] == "auth_required"


# ---------------------------------------------------------------------------
# Generation flow
# ---------------------------------------------------------------------------


def test_text_to_video_request_and_response(monkeypatch):
    _fake_credentials(monkeypatch)
    vid_b64 = base64.b64encode(b"fakevideo").decode("ascii")
    calls = _fake_operation_roundtrip(
        monkeypatch, videos=[{"bytesBase64Encoded": vid_b64, "mimeType": "video/mp4"}]
    )

    result = VertexVideoGenProvider().generate(
        "a dog surfing", duration=7, aspect_ratio="16:9", resolution="1080p"
    )

    assert result["success"] is True
    assert result["provider"] == "vertex"
    assert result["modality"] == "text"
    assert result["model"] == DEFAULT_MODEL
    assert result["duration"] == 8  # 7 snaps up to 8
    saved = Path(result["video"])
    assert saved.is_file()
    assert saved.read_bytes() == b"fakevideo"

    submit = calls[0]
    assert submit["url"] == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1"
        f"/publishers/google/models/{DEFAULT_MODEL}:predictLongRunning"
    )
    assert submit["headers"]["Authorization"] == "Bearer tok"
    payload = submit["payload"]
    assert payload["instances"] == [{"prompt": "a dog surfing"}]
    params = payload["parameters"]
    assert params["aspectRatio"] == "16:9"
    assert params["durationSeconds"] == 8
    assert params["sampleCount"] == 1
    assert params["generateAudio"] is True  # Veo 3.x requires it explicitly
    assert params["resolution"] == "1080p"

    poll = calls[1]
    assert poll["url"].endswith(":fetchPredictOperation")
    assert poll["payload"] == {"operationName": "projects/proj/operations/op1"}


def test_image_to_video_and_1080p_portrait_downgrade(monkeypatch):
    _fake_credentials(monkeypatch)
    src_b64 = base64.b64encode(b"frame").decode("ascii")
    vid_b64 = base64.b64encode(b"animated").decode("ascii")
    calls = _fake_operation_roundtrip(
        monkeypatch, videos=[{"bytesBase64Encoded": vid_b64, "mimeType": "video/mp4"}]
    )

    result = VertexVideoGenProvider().generate(
        "animate this",
        image_url=f"data:image/png;base64,{src_b64}",
        aspect_ratio="9:16",
        resolution="1080p",
    )

    assert result["success"] is True
    assert result["modality"] == "image"
    payload = calls[0]["payload"]
    assert payload["instances"][0]["image"] == {
        "bytesBase64Encoded": src_b64,
        "mimeType": "image/png",
    }
    # Veo only renders 1080p in 16:9 — portrait requests fall back to 720p.
    assert payload["parameters"]["resolution"] == "720p"


def test_audio_flag_is_forwarded(monkeypatch):
    _fake_credentials(monkeypatch)
    vid_b64 = base64.b64encode(b"muted").decode("ascii")
    calls = _fake_operation_roundtrip(
        monkeypatch, videos=[{"bytesBase64Encoded": vid_b64, "mimeType": "video/mp4"}]
    )

    result = VertexVideoGenProvider().generate("silent clip", audio=False)

    assert result["success"] is True
    # Veo 3.x rejects requests without an explicit generateAudio, so the
    # flag must always be present — honoring the caller's choice.
    assert calls[0]["payload"]["parameters"]["generateAudio"] is False


# ---------------------------------------------------------------------------
# Interactions API (Gemini Omni) flow
# ---------------------------------------------------------------------------


def _omni_completed_body(vid_b64):
    """The documented Interactions response shape (GEAP docs, 2026-06)."""
    return {
        "id": "interaction-123",
        "model": "gemini-omni-flash-preview",
        "status": "completed",
        "usage": {"total_tokens": 479},
        "steps": [
            {"type": "thought", "summary": [{"type": "text", "text": "planning"}]},
            {
                "type": "model_output",
                "content": [
                    {"type": "video", "data": vid_b64, "mime_type": "video/mp4"}
                ],
            },
        ],
        "object": "interaction",
    }


def test_omni_text_to_video_request_and_response(monkeypatch):
    _fake_credentials(monkeypatch)
    vid_b64 = base64.b64encode(b"omnivideo").decode("ascii")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        return _FakeResponse(_omni_completed_body(vid_b64))

    monkeypatch.setattr(vertex_plugin.requests, "post", fake_post)

    result = VertexVideoGenProvider().generate(
        "a test clip", model="google/gemini-omni-flash-preview", aspect_ratio="9:16"
    )

    assert result["success"] is True
    assert result["provider"] == "vertex"
    assert result["modality"] == "text"
    assert result["model"] == "gemini-omni-flash-preview"
    assert result["interaction_id"] == "interaction-123"
    assert result["api"] == "interactions"
    assert Path(result["video"]).read_bytes() == b"omnivideo"

    # Interactions is only documented on the global endpoint, v1beta1.
    assert captured["url"] == (
        "https://aiplatform.googleapis.com/v1beta1/projects/proj/locations/global/interactions"
    )
    assert captured["headers"]["Authorization"] == "Bearer tok"
    payload = captured["payload"]
    assert payload["model"] == "gemini-omni-flash-preview"  # google/ prefix stripped
    assert payload["input"][0] == {"type": "text", "text": "a test clip"}
    assert payload["response_format"] == {"type": "video", "aspect_ratio": "9:16"}
    assert payload["generation_config"] == {"video_config": {"task": "text_to_video"}}


def test_omni_image_to_video_inlines_source(monkeypatch):
    _fake_credentials(monkeypatch)
    src_b64 = base64.b64encode(b"frame").decode("ascii")
    vid_b64 = base64.b64encode(b"animated").decode("ascii")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return _FakeResponse(_omni_completed_body(vid_b64))

    monkeypatch.setattr(vertex_plugin.requests, "post", fake_post)

    result = VertexVideoGenProvider().generate(
        "animate this",
        model="gemini-omni-flash-preview",
        image_url=f"data:image/png;base64,{src_b64}",
    )

    assert result["success"] is True
    assert result["modality"] == "image"
    items = captured["payload"]["input"]
    assert items[1] == {"type": "image", "data": src_b64, "mime_type": "image/png"}
    assert captured["payload"]["generation_config"] == {
        "video_config": {"task": "image_to_video"}
    }


def test_omni_completed_without_video_surfaces_model_text(monkeypatch):
    _fake_credentials(monkeypatch)
    body = {
        "id": "interaction-456",
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "I cannot generate that."}],
            }
        ],
    }
    monkeypatch.setattr(vertex_plugin.requests, "post", lambda *a, **k: _FakeResponse(body))

    result = VertexVideoGenProvider().generate("x", model="gemini-omni-flash-preview")

    assert result["success"] is False
    assert result["error_type"] == "empty_response"
    assert "I cannot generate that." in result["error"]


def test_omni_non_completed_status_is_an_error(monkeypatch):
    _fake_credentials(monkeypatch)
    body = {"id": "interaction-789", "status": "failed", "steps": []}
    monkeypatch.setattr(vertex_plugin.requests, "post", lambda *a, **k: _FakeResponse(body))

    result = VertexVideoGenProvider().generate("x", model="gemini-omni-flash-preview")

    assert result["success"] is False
    assert result["error_type"] == "interaction_failed"
    assert "interaction-789" in result["error"]


def _omni_uri_body(uri):
    """Interaction response using uri delivery (outputs over the inline cap)."""
    body = _omni_completed_body("unused")
    body["steps"][1]["content"] = [{"type": "video", "uri": uri, "mime_type": "video/mp4"}]
    return body


def test_omni_uri_delivery_downloads_with_bearer_token(monkeypatch):
    _fake_credentials(monkeypatch)
    uri = "https://storage.example.com/interactions/vid123.mp4"
    captured = {}

    monkeypatch.setattr(
        vertex_plugin.requests, "post", lambda *a, **k: _FakeResponse(_omni_uri_body(uri))
    )

    class _FakeDownload:
        status_code = 200
        content = b"bigvideo"

        def raise_for_status(self):
            pass

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeDownload()

    monkeypatch.setattr(vertex_plugin.requests, "get", fake_get)

    result = VertexVideoGenProvider().generate("x", model="gemini-omni-flash-preview")

    assert result["success"] is True
    assert Path(result["video"]).read_bytes() == b"bigvideo"
    assert captured["url"] == uri
    assert captured["headers"]["Authorization"] == "Bearer tok"


def test_omni_uri_delivery_download_failure_returns_uri(monkeypatch):
    _fake_credentials(monkeypatch)
    uri = "https://storage.example.com/interactions/vid456.mp4"

    monkeypatch.setattr(
        vertex_plugin.requests, "post", lambda *a, **k: _FakeResponse(_omni_uri_body(uri))
    )
    monkeypatch.setattr(
        vertex_plugin.requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("boom")),
    )

    result = VertexVideoGenProvider().generate("x", model="gemini-omni-flash-preview")

    # Degraded but not failed: the caller still gets a usable reference.
    assert result["success"] is True
    assert result["video"] == uri


def test_conflicting_image_and_reference_inputs(monkeypatch):
    _fake_credentials(monkeypatch)
    result = VertexVideoGenProvider().generate(
        "x",
        image_url="https://example.com/a.png",
        reference_image_urls=["https://example.com/b.png"],
    )
    assert result["success"] is False
    assert result["error_type"] == "conflicting_inputs"


def test_operation_error_is_surfaced(monkeypatch):
    _fake_credentials(monkeypatch)

    def fake_post(url, headers=None, json=None, timeout=None):
        if url.endswith(":predictLongRunning"):
            return _FakeResponse({"name": "op1"})
        return _FakeResponse({"done": True, "error": {"message": "quota exceeded"}})

    monkeypatch.setattr(vertex_plugin.requests, "post", fake_post)

    result = VertexVideoGenProvider().generate("a dog")
    assert result["success"] is False
    assert result["error_type"] == "api_error"
    assert "quota exceeded" in result["error"]


def test_rai_filtered_output_is_a_safety_error(monkeypatch):
    _fake_credentials(monkeypatch)

    def fake_post(url, headers=None, json=None, timeout=None):
        if url.endswith(":predictLongRunning"):
            return _FakeResponse({"name": "op1"})
        return _FakeResponse(
            {
                "done": True,
                "response": {
                    "raiMediaFilteredCount": 1,
                    "raiMediaFilteredReasons": ["violence"],
                },
            }
        )

    monkeypatch.setattr(vertex_plugin.requests, "post", fake_post)

    result = VertexVideoGenProvider().generate("a dog")
    assert result["success"] is False
    assert result["error_type"] == "safety_blocked"
    assert "violence" in result["error"]


def test_gcs_output_is_downloaded_locally(monkeypatch):
    _fake_credentials(monkeypatch)
    _fake_operation_roundtrip(
        monkeypatch, videos=[{"gcsUri": "gs://bucket/out/video.mp4", "mimeType": "video/mp4"}]
    )
    monkeypatch.setattr(vertex_plugin, "_download_gcs_object", lambda uri, token: b"gcsvideo")

    result = VertexVideoGenProvider().generate("a dog")

    assert result["success"] is True
    assert result["gcs_uri"] == "gs://bucket/out/video.mp4"
    assert Path(result["video"]).read_bytes() == b"gcsvideo"


def test_extract_videos_handles_generated_samples_shape():
    body = {
        "done": True,
        "response": {"generatedSamples": [{"video": {"uri": "gs://b/o.mp4"}}]},
    }
    videos = _extract_videos(body)
    assert videos[0]["gcsUri"] == "gs://b/o.mp4"
