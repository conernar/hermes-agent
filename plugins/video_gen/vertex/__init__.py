"""Google Vertex AI video generation backend (Veo + Gemini Omni).

Serves Google's video models through **Vertex AI** with OAuth2 service-account
/ Application Default Credentials — no Gemini API key required. Credential
resolution is shared with the Vertex chat provider
(``agent/vertex_adapter.py``): ``VERTEX_CREDENTIALS_PATH`` /
``GOOGLE_APPLICATION_CREDENTIALS`` in ``.env``, routing (project/region)
under ``vertex:`` in ``config.yaml``.

Routing by model id prefix — Vertex exposes two video APIs:

- ``veo-*``    → ``:predictLongRunning`` + ``:fetchPredictOperation``
  polling (regional endpoint, default us-central1). By default the finished
  video comes back inline as base64 and is saved under
  ``$HERMES_HOME/cache/videos/``; set ``video_gen.vertex.storage_uri`` to a
  ``gs://`` prefix to have Vertex write the MP4 to Cloud Storage instead
  (the plugin then downloads it with the same credentials so downstream
  consumers still get a local file).
- ``gemini-*`` → the Interactions API (``POST /v1beta1/projects/{p}/
  locations/global/interactions``), a synchronous call that returns the
  video inline in the response ``steps`` — this serves the Gemini Omni
  family (``gemini-omni-flash-preview``, text/image/reference-to-video).

  Omni exposes no structured duration/resolution/audio/seed parameters
  (the Vertex endpoint 400s on ``response_format.delivery``, so Gemini-API
  schema parity must not be assumed — probe before sending new fields).
  Output resolution is fixed (720p in the current preview). Duration-like
  constraints CAN often be steered through the prompt text: Omni interprets
  instructions semantically, so "an 8 second clip of ..." usually yields
  ~8s (verified on the live endpoint) — best-effort, unlike Veo's exact
  ``durationSeconds`` contract.

Model ids are passed through unvalidated — new Google releases work by just
setting ``video_gen.model`` (via ``hermes tools``) or calling
``video_generate`` with ``model=``. A ``google/`` publisher prefix (the
naming convention of the Vertex chat endpoint) is stripped automatically.
A configured model id that matches neither family (stale config from a
previously-selected backend) falls back to :data:`DEFAULT_MODEL` unless the
tool call requested it explicitly.

Config keys (all optional, under ``video_gen.vertex``):
    region                  endpoint location (default us-central1 — Veo is regional)
    storage_uri             gs:// output prefix (default: inline base64 response)
    person_generation       allow_adult | allow_all | dont_allow
    timeout_seconds         max wait for the operation (default 600)
    poll_interval_seconds   seconds between polls (default 10)
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    save_b64_video,
    save_bytes_video,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

# Catalog is GA-only: Google removed the Veo preview endpoints on 2026-04-02
# and the Veo 2.x / 3.0 GA endpoints on 2026-06-30. Anything newer passes
# through via config — the catalog is advisory, not a validation gate.
_MODELS: Dict[str, Dict[str, Any]] = {
    "veo-3.1-generate-001": {
        "display": "Veo 3.1",
        "speed": "~1-4min",
        "strengths": "Latest GA Veo; native audio, 1080p, image-to-video, up to 3 reference images.",
        "modalities": ["text", "image"],
    },
    "veo-3.1-fast-generate-001": {
        "display": "Veo 3.1 Fast",
        "speed": "~1-2min",
        "strengths": "Cheaper/faster Veo 3.1 tier; native audio.",
        "modalities": ["text", "image"],
    },
    "veo-3.1-lite-generate-001": {
        "display": "Veo 3.1 Lite (preview)",
        "speed": "~1-2min",
        "strengths": "Most cost-efficient Veo tier (public preview since 2026-04).",
        "modalities": ["text", "image"],
    },
    "gemini-omni-flash-preview": {
        "display": "Gemini Omni Flash (preview)",
        "speed": "~1-3min",
        "strengths": "Any-to-any Gemini video model (public preview since 2026-06); text/image/reference-to-video, conversational editing. 720p, up to 10s.",
        "modalities": ["text", "image"],
    },
}

DEFAULT_MODEL = "veo-3.1-generate-001"

VALID_ASPECT_RATIOS = {"16:9", "9:16"}
VALID_RESOLUTIONS = {"720p", "1080p"}
MAX_REFERENCE_IMAGES = 3

# Veo is documented against regional endpoints (every REST sample uses
# us-central1); the global endpoint also serves it. Override via
# ``video_gen.vertex.region`` (e.g. "global").
DEFAULT_LOCATION = "us-central1"
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_POLL_INTERVAL_SECONDS = 10

_MAX_INPUT_IMAGE_BYTES = 20 * 1024 * 1024


# ---------------------------------------------------------------------------
# Config / auth helpers
# ---------------------------------------------------------------------------


def _load_vertex_section() -> Dict[str, Any]:
    """Read ``video_gen.vertex`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("video_gen") if isinstance(cfg, dict) else None
        sub = section.get("vertex") if isinstance(section, dict) else None
        return sub if isinstance(sub, dict) else {}
    except Exception as exc:
        logger.debug("Could not load video_gen.vertex config: %s", exc)
        return {}


def _resolve_location() -> str:
    raw = _load_vertex_section().get("region")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return DEFAULT_LOCATION


def _endpoint(project_id: str, location: str, model_id: str, method: str) -> str:
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    return (
        f"https://{host}/v1/projects/{project_id}/locations/{location}"
        f"/publishers/google/models/{model_id}:{method}"
    )


def _resolve_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Return (access_token, project_id) via the shared Vertex adapter.

    Imported lazily so loading this plugin never triggers the adapter's
    google-auth lazy-install path.
    """
    try:
        from agent.vertex_adapter import get_vertex_credentials

        return get_vertex_credentials()
    except Exception as exc:
        logger.debug("Vertex credential resolution failed: %s", exc)
        return None, None


def _auth_required_response(prompt: str) -> Dict[str, Any]:
    return error_response(
        error=(
            "No Google Vertex AI credentials found. Set VERTEX_CREDENTIALS_PATH "
            "(or GOOGLE_APPLICATION_CREDENTIALS) to a service-account JSON in "
            ".env, or configure Application Default Credentials "
            "(`gcloud auth application-default login`). Project/region live "
            "under `vertex:` in config.yaml."
        ),
        error_type="auth_required",
        provider="vertex",
        prompt=prompt,
    )


def _positive_number(raw: Any, default: float) -> float:
    try:
        value = float(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return default


def _normalize_model_id(model_id: str) -> str:
    """Strip a ``google/`` publisher prefix — the endpoint path already
    carries ``/publishers/google/``, so chat-style ids like
    ``google/veo-3.1-generate-001`` would otherwise 404."""
    candidate = model_id.strip()
    if candidate.lower().startswith("google/"):
        candidate = candidate[len("google/"):]
    return candidate


def _resolve_model(model: Optional[str], *, explicit: bool) -> str:
    """Pass explicit ids through; ignore stale foreign ids from config."""
    candidate = _normalize_model_id(model or "")
    if not candidate:
        return DEFAULT_MODEL
    if explicit or candidate.lower().startswith(("veo", "gemini")):
        return candidate
    logger.debug(
        "video_gen model '%s' does not look like a Vertex video model id; using %s",
        candidate, DEFAULT_MODEL,
    )
    return DEFAULT_MODEL


def _uses_interactions_api(model_id: str) -> bool:
    """Gemini-family video models (Omni) are served by the Interactions API,
    not by Veo's ``:predictLongRunning``."""
    return model_id.lower().startswith("gemini")


def _clamp_duration(duration: Optional[int]) -> Optional[int]:
    """Clamp to the durations Veo accepts; None means let the API default.

    Veo 3.1 only takes 4, 6 or 8 seconds — odd values are rounded up so the
    user never gets less than they asked for.
    """
    if duration is None:
        return None
    value = max(4, min(8, int(duration)))
    if value % 2:
        value += 1
    return value


# ---------------------------------------------------------------------------
# Input image handling
# ---------------------------------------------------------------------------


def _mime_from_path(path: Path) -> str:
    ext = path.suffix.lstrip(".").lower()
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext in ("png", "webp", "gif"):
        return f"image/{ext}"
    return "image/png"


def _image_source_to_b64(ref: str) -> Tuple[str, str]:
    """Load a source image as ``(base64, mime_type)``.

    Accepts a data URI, an http(s) URL (downloaded and inlined), or a local
    file path (guarded read). Raises ``ValueError`` with a user-facing
    message on failure.
    """
    lower = ref.lower()

    if lower.startswith("data:"):
        header, _, payload = ref.partition(",")
        if not payload or ";base64" not in header:
            raise ValueError("data: URI image inputs must be base64-encoded")
        mime = header[5:].split(";", 1)[0].strip() or "image/png"
        return payload, mime

    if lower.startswith(("http://", "https://")):
        response = requests.get(ref, timeout=60)
        response.raise_for_status()
        if len(response.content) > _MAX_INPUT_IMAGE_BYTES:
            raise ValueError(f"Input image at {ref} exceeds 20MB")
        mime = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if not mime.startswith("image/"):
            mime = _mime_from_path(Path(ref.split("?", 1)[0]))
        return base64.b64encode(response.content).decode("ascii"), mime

    # Local file path — enforce the shared credential-read guard before
    # touching bytes (same boundary as the other media providers).
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    path = Path(ref).expanduser()
    if not path.is_file():
        raise ValueError(
            f"image input '{ref}' is neither a URL, data URI, nor an existing file"
        )
    return base64.b64encode(path.read_bytes()).decode("ascii"), _mime_from_path(path)


def _veo_image_field(source: str) -> Dict[str, Any]:
    """Build Veo's ``image`` value from a URL / data URI / gs:// URI / local path.

    Raises ``ValueError`` with a user-facing message on failure.
    """
    ref = source.strip()
    if ref.lower().startswith("gs://"):
        return {"gcsUri": ref, "mimeType": _mime_from_path(Path(ref))}
    b64, mime = _image_source_to_b64(ref)
    return {"bytesBase64Encoded": b64, "mimeType": mime}


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def _auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _http_error_detail(exc: requests.HTTPError) -> Tuple[int, str]:
    response = exc.response
    status = response.status_code if response is not None else 0
    try:
        detail = response.json().get("error", {}).get("message", "")
    except Exception:
        detail = ""
    if not detail and response is not None:
        detail = response.text[:300]
    return status, detail or str(exc)


def _submit(url: str, token: str, payload: Dict[str, Any]) -> str:
    response = requests.post(url, headers=_auth_headers(token), json=payload, timeout=60)
    response.raise_for_status()
    operation_name = response.json().get("name")
    if not operation_name:
        raise RuntimeError("Vertex predictLongRunning response did not include an operation name")
    return operation_name


def _poll(
    fetch_url: str,
    token: str,
    operation_name: str,
    *,
    timeout_seconds: float,
    poll_interval: float,
) -> Dict[str, Any]:
    """Poll ``fetchPredictOperation`` until done/timeout; returns the final body."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = requests.post(
            fetch_url,
            headers=_auth_headers(token),
            json={"operationName": operation_name},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("done"):
            return body
        if time.monotonic() >= deadline:
            return {"_timeout": True}
        time.sleep(poll_interval)


def _download_gcs_object(uri: str, token: str) -> bytes:
    """Fetch a ``gs://bucket/object`` with the same OAuth token (JSON API)."""
    without_scheme = uri[len("gs://"):]
    bucket, _, obj = without_scheme.partition("/")
    if not bucket or not obj:
        raise ValueError(f"Unparseable gs:// URI: {uri}")
    url = (
        f"https://storage.googleapis.com/storage/v1/b/{bucket}"
        f"/o/{quote(obj, safe='')}?alt=media"
    )
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    response.raise_for_status()
    return response.content


def _extract_videos(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize the two response shapes Veo operations have used."""
    response = body.get("response") if isinstance(body.get("response"), dict) else {}
    videos = response.get("videos")
    if isinstance(videos, list) and videos:
        return [v for v in videos if isinstance(v, dict)]
    samples = response.get("generatedSamples")
    out: List[Dict[str, Any]] = []
    if isinstance(samples, list):
        for sample in samples:
            video = sample.get("video") if isinstance(sample, dict) else None
            if isinstance(video, dict):
                # Older shape uses "uri" for the Cloud Storage path.
                normalized = dict(video)
                if "uri" in normalized and "gcsUri" not in normalized:
                    normalized["gcsUri"] = normalized["uri"]
                out.append(normalized)
    return out


# ---------------------------------------------------------------------------
# Interactions API (Gemini Omni)
# ---------------------------------------------------------------------------


def _interactions_endpoint(project_id: str) -> str:
    """The Interactions API is only documented on the global endpoint."""
    return (
        "https://aiplatform.googleapis.com/v1beta1"
        f"/projects/{project_id}/locations/global/interactions"
    )


# Interactions video_config.task values by input modality. "edit" exists in
# the API but needs a source-video input, which the unified video_generate
# surface intentionally does not expose.
_OMNI_TASKS = {
    "text": "text_to_video",
    "image": "image_to_video",
    "reference": "reference_to_video",
}


def _omni_media_item(source: str) -> Dict[str, Any]:
    """Build one Interactions ``input`` item for a source image.

    Mirrors the documented content item shape (``{"type", "data",
    "mime_type"}``). Raises ``ValueError`` with a user-facing message on
    failure.
    """
    ref = source.strip()
    if ref.lower().startswith("gs://"):
        raise ValueError(
            "gs:// inputs are not supported by the Interactions API — pass a "
            "local path, data URI, or https URL"
        )
    b64, mime = _image_source_to_b64(ref)
    return {"type": "image", "data": b64, "mime_type": mime}


def _extract_interaction_video(
    body: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Return ``(b64_data, uri, mime, text_detail)`` from an interaction response.

    Videos arrive inline (``data``, the default) or — for outputs over the
    inline delivery cap — as a hosted ``uri`` (``response_format.delivery``).
    Exactly one of ``b64_data`` / ``uri`` is set when a video was produced.
    ``text_detail`` collects any text the model emitted in ``model_output``
    steps — useful in error messages when it declined to produce video.
    """
    text_detail = ""
    for step in body.get("steps") or []:
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for item in step.get("content") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "video":
                mime = item.get("mime_type") or "video/mp4"
                if item.get("data"):
                    return item["data"], None, mime, text_detail
                uri = item.get("uri") or item.get("url")
                if isinstance(uri, str) and uri.strip():
                    return None, uri.strip(), mime, text_detail
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text_detail += item["text"]
    return None, None, None, text_detail


def _download_interaction_video(uri: str, token: str) -> bytes:
    """Fetch a uri-delivered interaction video with the caller's bearer token."""
    if uri.lower().startswith("gs://"):
        return _download_gcs_object(uri, token)
    response = requests.get(uri, headers=_auth_headers(token), timeout=120)
    response.raise_for_status()
    return response.content


def _generate_via_interactions(
    *,
    token: str,
    project_id: str,
    model_id: str,
    prompt: str,
    image_url: Optional[str],
    reference_image_urls: Optional[List[str]],
    aspect_ratio: str,
    section: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a video with a Gemini Omni model via the Interactions API.

    Synchronous: one POST returns the completed interaction with the video
    inline as base64 in the ``steps`` — no operation polling.
    """
    # Omni only renders 16:9 and 9:16 (720p, ≤10s in the current preview).
    aspect = (aspect_ratio or DEFAULT_ASPECT_RATIO).strip()
    if aspect not in ("16:9", "9:16"):
        aspect = DEFAULT_ASPECT_RATIO

    refs = [r.strip() for r in (reference_image_urls or []) if isinstance(r, str) and r.strip()]
    sources: List[str] = []
    if image_url and image_url.strip():
        sources.append(image_url.strip())
    sources.extend(refs)
    modality = "image" if image_url and image_url.strip() else ("reference" if refs else "text")

    input_items: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    try:
        input_items.extend(_omni_media_item(source) for source in sources)
    except ValueError as exc:
        return error_response(
            error=str(exc),
            error_type="invalid_image_url",
            provider="vertex",
            model=model_id,
            prompt=prompt,
        )
    except Exception as exc:
        return error_response(
            error=f"Could not load input image: {exc}",
            error_type="io_error",
            provider="vertex",
            model=model_id,
            prompt=prompt,
        )

    payload: Dict[str, Any] = {
        "model": model_id,
        "input": input_items,
        # Omni is any-to-any; declare video output explicitly so a prompt
        # that reads like a question still yields a clip, not prose.
        "response_format": {"type": "video", "aspect_ratio": aspect},
    }
    # Declare the task explicitly instead of letting the model infer it —
    # disambiguates "animate this frame" (image_to_video) from "use this as
    # a style reference" (reference_to_video). Verified accepted on the
    # Vertex Interactions endpoint (probe, 2026-07-09); note Vertex rejects
    # response_format.delivery, so schema parity with the Gemini API cannot
    # be assumed for new fields — probe before adding any.
    task = _OMNI_TASKS.get(modality)
    if task:
        payload["generation_config"] = {"video_config": {"task": task}}

    timeout_seconds = _positive_number(section.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    logger.info(
        "Vertex video: %s via Interactions API (location=global, aspect=%s, modality=%s)",
        model_id, aspect, modality,
    )
    started = time.monotonic()
    try:
        response = requests.post(
            _interactions_endpoint(project_id),
            headers=_auth_headers(token),
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
    except requests.HTTPError as exc:
        status, detail = _http_error_detail(exc)
        logger.error("Vertex Omni video gen failed (%d): %s", status, detail)
        return error_response(
            error=f"Vertex video generation failed ({status}): {detail}",
            error_type="api_error",
            provider="vertex",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )
    except requests.Timeout:
        return error_response(
            error=(
                f"Vertex Omni video generation timed out ({int(timeout_seconds)}s) "
                "(raise video_gen.vertex.timeout_seconds if generations routinely take longer)"
            ),
            error_type="timeout",
            provider="vertex",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )
    except requests.RequestException as exc:
        return error_response(
            error=f"Vertex connection error: {exc}",
            error_type="connection_error",
            provider="vertex",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )
    except ValueError as exc:
        return error_response(
            error=f"Vertex returned invalid JSON: {exc}",
            error_type="invalid_response",
            provider="vertex",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    interaction_id = body.get("id")
    status_text = str(body.get("status") or "").lower()
    if status_text and status_text != "completed":
        return error_response(
            error=(
                f"Vertex interaction ended with status '{status_text}'"
                + (f" (interaction id: {interaction_id})" if interaction_id else "")
            ),
            error_type=f"interaction_{status_text}",
            provider="vertex",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    b64, uri, _mime, text_detail = _extract_interaction_video(body)
    if not b64 and not uri:
        return error_response(
            error=(
                "Vertex interaction completed without video output"
                + (f" (model said: {text_detail[:300]})" if text_detail else "")
            ),
            error_type="empty_response",
            provider="vertex",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    if b64:
        try:
            video_ref = str(save_b64_video(b64, prefix="vertex_omni"))
        except Exception as exc:
            return error_response(
                error=f"Could not save video to cache: {exc}",
                error_type="io_error",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
    else:
        # uri delivery (outputs over the inline cap). Materialise locally so
        # downstream consumers (Telegram upload, previews) need no extra auth;
        # fall back to the raw URI rather than failing — mirrors the Veo
        # gs:// download fallback.
        try:
            video_ref = str(
                save_bytes_video(_download_interaction_video(uri, token), prefix="vertex_omni")
            )
        except Exception as exc:
            logger.warning(
                "Could not download interaction video %s (%s); returning the URI", uri, exc
            )
            video_ref = uri

    logger.info(
        "Vertex video: interaction %s completed in %.0fs -> %s",
        interaction_id or "<no id>", time.monotonic() - started, video_ref,
    )

    extra: Dict[str, Any] = {"location": "global", "api": "interactions"}
    if interaction_id:
        # Keep the id visible — the Interactions API supports conversational
        # follow-up edits referencing a previous interaction.
        extra["interaction_id"] = interaction_id
    if body.get("usage"):
        extra["usage"] = body["usage"]

    return success_response(
        video=video_ref,
        model=str(body.get("model") or model_id),
        prompt=prompt,
        modality=modality,
        aspect_ratio=aspect,
        duration=0,
        provider="vertex",
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class VertexVideoGenProvider(VideoGenProvider):
    """Google Vertex AI Veo backend."""

    @property
    def name(self) -> str:
        return "vertex"

    @property
    def display_name(self) -> str:
        return "Google Vertex AI"

    def is_available(self) -> bool:
        try:
            from agent.vertex_adapter import has_vertex_credentials

            return has_vertex_credentials()
        except Exception as exc:
            logger.debug("Vertex availability check failed: %s", exc)
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": model_id, **meta} for model_id, meta in _MODELS.items()]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Google Vertex AI (Veo)",
            "badge": "paid",
            "tag": (
                "Veo text/image-to-video via Google Cloud Vertex AI — "
                "service-account or ADC auth (no Gemini API key). "
                "Project under `vertex:` in config.yaml; region via "
                "`video_gen.vertex.region` (Veo is regional, default us-central1)."
            ),
            "env_vars": [
                {
                    "key": "VERTEX_CREDENTIALS_PATH",
                    "prompt": "Path to a GCP service-account JSON with Vertex AI access (blank to use ADC)",
                    "url": "https://console.cloud.google.com/iam-admin/serviceaccounts",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": sorted(VALID_ASPECT_RATIOS),
            "resolutions": sorted(VALID_RESOLUTIONS),
            "max_duration": 8,
            "min_duration": 4,
            "supports_audio": True,
            "supports_negative_prompt": True,
            "max_reference_images": MAX_REFERENCE_IMAGES,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        if not prompt:
            return error_response(
                error="prompt is required for Vertex video generation",
                error_type="missing_prompt",
                provider="vertex",
            )

        token, project_id = _resolve_credentials()
        if not token or not project_id:
            return _auth_required_response(prompt)

        model_id = _resolve_model(model, explicit=bool(kwargs.get("_model_override_explicit")))
        section = _load_vertex_section()

        if _uses_interactions_api(model_id):
            return _generate_via_interactions(
                token=token,
                project_id=project_id,
                model_id=model_id,
                prompt=prompt,
                image_url=image_url,
                reference_image_urls=reference_image_urls,
                aspect_ratio=aspect_ratio,
                section=section,
            )

        location = _resolve_location()

        aspect = (aspect_ratio or DEFAULT_ASPECT_RATIO).strip()
        if aspect not in VALID_ASPECT_RATIOS:
            aspect = DEFAULT_ASPECT_RATIO
        res = (resolution or "").strip().lower()
        if res not in VALID_RESOLUTIONS:
            res = DEFAULT_RESOLUTION
        if res == "1080p" and aspect != "16:9":
            res = "720p"  # Veo only renders 1080p in 16:9
        clamped_duration = _clamp_duration(duration)

        refs = [r.strip() for r in (reference_image_urls or []) if isinstance(r, str) and r.strip()]
        if refs and image_url and image_url.strip():
            return error_response(
                error="image_url and reference_image_urls cannot be combined on Vertex Veo",
                error_type="conflicting_inputs",
                provider="vertex",
                model=model_id,
                prompt=prompt,
            )
        if len(refs) > MAX_REFERENCE_IMAGES:
            return error_response(
                error=f"Vertex Veo supports at most {MAX_REFERENCE_IMAGES} reference images",
                error_type="too_many_references",
                provider="vertex",
                model=model_id,
                prompt=prompt,
            )

        instance: Dict[str, Any] = {"prompt": prompt}
        try:
            if image_url and image_url.strip():
                instance["image"] = _veo_image_field(image_url)
            if refs:
                instance["referenceImages"] = [
                    {"image": _veo_image_field(ref), "referenceType": "asset"} for ref in refs
                ]
        except ValueError as exc:
            return error_response(
                error=str(exc),
                error_type="invalid_image_url",
                provider="vertex",
                model=model_id,
                prompt=prompt,
            )
        except Exception as exc:
            return error_response(
                error=f"Could not load input image: {exc}",
                error_type="io_error",
                provider="vertex",
                model=model_id,
                prompt=prompt,
            )
        modality = "image" if "image" in instance else ("reference" if refs else "text")

        parameters: Dict[str, Any] = {
            "aspectRatio": aspect,
            "sampleCount": 1,
        }
        if clamped_duration is not None:
            parameters["durationSeconds"] = clamped_duration
        # Veo 3.x rejects requests without an explicit generateAudio.
        parameters["generateAudio"] = True if audio is None else bool(audio)
        parameters["resolution"] = res
        if negative_prompt and negative_prompt.strip():
            parameters["negativePrompt"] = negative_prompt.strip()
        if seed is not None:
            parameters["seed"] = int(seed)
        person_generation = section.get("person_generation")
        if isinstance(person_generation, str) and person_generation.strip():
            parameters["personGeneration"] = person_generation.strip()
        storage_uri = section.get("storage_uri")
        if isinstance(storage_uri, str) and storage_uri.strip():
            parameters["storageUri"] = storage_uri.strip()

        payload = {"instances": [instance], "parameters": parameters}
        submit_url = _endpoint(project_id, location, model_id, "predictLongRunning")
        fetch_url = _endpoint(project_id, location, model_id, "fetchPredictOperation")
        timeout_seconds = _positive_number(section.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
        poll_interval = _positive_number(
            section.get("poll_interval_seconds"), DEFAULT_POLL_INTERVAL_SECONDS
        )

        logger.info(
            "Vertex video: %s via predictLongRunning (location=%s, aspect=%s, "
            "resolution=%s, duration=%s, audio=%s, modality=%s)",
            model_id, location, aspect, res,
            f"{clamped_duration}s" if clamped_duration is not None else "model default",
            parameters["generateAudio"], modality,
        )
        started = time.monotonic()
        try:
            operation_name = _submit(submit_url, token, payload)
            body = _poll(
                fetch_url,
                token,
                operation_name,
                timeout_seconds=timeout_seconds,
                poll_interval=poll_interval,
            )
        except requests.HTTPError as exc:
            status, detail = _http_error_detail(exc)
            logger.error("Vertex video gen failed (%d): %s", status, detail)
            return error_response(
                error=f"Vertex video generation failed ({status}): {detail}",
                error_type="api_error",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error="Vertex video generation request timed out",
                error_type="timeout",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.RequestException as exc:
            return error_response(
                error=f"Vertex connection error: {exc}",
                error_type="connection_error",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except RuntimeError as exc:
            return error_response(
                error=str(exc),
                error_type="invalid_response",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if body.get("_timeout"):
            return error_response(
                error=(
                    f"Timed out waiting for Vertex Veo after {int(timeout_seconds)}s "
                    "(raise video_gen.vertex.timeout_seconds if generations routinely take longer)"
                ),
                error_type="timeout",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        op_error = body.get("error")
        if isinstance(op_error, dict) and op_error:
            return error_response(
                error=f"Vertex Veo operation failed: {op_error.get('message') or op_error}",
                error_type="api_error",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        videos = _extract_videos(body)
        if not videos:
            response_body = body.get("response") if isinstance(body.get("response"), dict) else {}
            reasons = response_body.get("raiMediaFilteredReasons")
            if reasons:
                return error_response(
                    error=f"Vertex filtered the generated video: {reasons}",
                    error_type="safety_blocked",
                    provider="vertex",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            return error_response(
                error="Vertex Veo operation completed without any video output",
                error_type="empty_response",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = videos[0]
        extra: Dict[str, Any] = {"operation": operation_name, "location": location}
        b64 = first.get("bytesBase64Encoded")
        gcs_uri = first.get("gcsUri")
        video_ref: Optional[str] = None

        if b64:
            try:
                video_ref = str(save_b64_video(b64, prefix="vertex_veo"))
            except Exception as exc:
                return error_response(
                    error=f"Could not save video to cache: {exc}",
                    error_type="io_error",
                    provider="vertex",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        elif gcs_uri:
            extra["gcs_uri"] = gcs_uri
            # Materialise the bytes locally so downstream consumers (Telegram
            # upload, browser preview) don't need GCS credentials.
            try:
                video_ref = str(
                    save_bytes_video(_download_gcs_object(gcs_uri, token), prefix="vertex_veo")
                )
            except Exception as exc:
                logger.warning("Could not download %s (%s); returning the gs:// URI", gcs_uri, exc)
                video_ref = gcs_uri

        if not video_ref:
            return error_response(
                error="Vertex Veo response contained neither video bytes nor a gs:// URI",
                error_type="empty_response",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        logger.info(
            "Vertex video: %s completed in %.0fs -> %s",
            model_id, time.monotonic() - started, video_ref,
        )

        return success_response(
            video=video_ref,
            model=model_id,
            prompt=prompt,
            modality=modality,
            aspect_ratio=aspect,
            duration=clamped_duration or 0,
            provider="vertex",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register this provider with the video gen registry."""
    ctx.register_video_gen_provider(VertexVideoGenProvider())
