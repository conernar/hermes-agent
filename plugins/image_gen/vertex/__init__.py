"""Google Vertex AI image generation backend.

Serves Google's image models through **Vertex AI** with OAuth2 service-account
/ Application Default Credentials — no Gemini API key required. Credential
resolution is shared with the Vertex chat provider
(``agent/vertex_adapter.py``): ``VERTEX_CREDENTIALS_PATH`` /
``GOOGLE_APPLICATION_CREDENTIALS`` in ``.env``, routing (project/region)
under ``vertex:`` in ``config.yaml``.

Routing by model id prefix — Vertex exposes two distinct APIs:

- ``gemini-*``  → ``:generateContent``  (text-to-image AND image editing;
  the Nano Banana family)
- ``imagen-*``  → ``:predict``          (text-to-image only)

Unknown model ids are passed through unvalidated so newly released Google
models work by just changing config — no plugin update needed. When the
prefix doesn't identify the API, set ``image_gen.vertex.api`` to
``generate_content`` or ``predict`` explicitly.

Model selection precedence (first hit wins):
1. ``model`` kwarg from the tool layer (``image_gen.model`` in config.yaml,
   maintained by ``hermes tools``) — ignored when it doesn't look like a
   Vertex model id (a stale id from a previously-selected backend).
2. ``VERTEX_IMAGE_MODEL`` env var
3. ``image_gen.vertex.model`` in config.yaml
4. :data:`DEFAULT_MODEL`

Endpoint location: ``image_gen.vertex.region`` wins when set. Otherwise
Gemini image models follow the chat adapter's region resolution (default
``global`` — the Gemini 3.x previews are global-endpoint-only), while
Imagen ``:predict`` models default to ``us-central1`` (media models are
served regionally).
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

# Catalog reflects the models Google actually serves as of 2026-07: the
# Imagen 2.x-4.x endpoints were removed on 2026-06-30 (migration target is
# the Gemini image family). The ``imagen-*`` → :predict routing below is
# kept for projects with residual/allowlisted Imagen access; anything newer
# passes through via config — the catalog is advisory, not a validation gate.
_MODELS: Dict[str, Dict[str, Any]] = {
    "gemini-3-pro-image-preview": {
        "display": "Nano Banana Pro (Gemini 3 Pro Image)",
        "speed": "~30-60s",
        "strengths": "Highest-fidelity Google image model; strong text rendering, up to 4K, image editing with multiple references.",
        "modalities": ["text", "image"],
    },
    "gemini-3.1-flash-image": {
        "display": "Gemini 3.1 Flash Image (preview)",
        "speed": "~5-15s",
        "strengths": "Fast text-to-image and editing; improved pricing/latency over 2.5 (public preview since 2026-02).",
        "modalities": ["text", "image"],
    },
    "gemini-2.5-flash-image": {
        "display": "Nano Banana (Gemini 2.5 Flash Image)",
        "speed": "~5-15s",
        "strengths": "GA fast tier; text-to-image and image editing.",
        "modalities": ["text", "image"],
    },
}

DEFAULT_MODEL = "gemini-3-pro-image-preview"

# Unified aspect names → the ratio strings both Vertex APIs accept.
_ASPECT_RATIOS = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

_MIME_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}

_MAX_INPUT_IMAGE_BYTES = 20 * 1024 * 1024

DEFAULT_TIMEOUT_SECONDS = 300


# ---------------------------------------------------------------------------
# Config / auth helpers
# ---------------------------------------------------------------------------


def _load_vertex_section() -> Dict[str, Any]:
    """Read ``image_gen.vertex`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        sub = section.get("vertex") if isinstance(section, dict) else None
        return sub if isinstance(sub, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.vertex config: %s", exc)
        return {}


def _infer_api(model_id: str) -> Optional[str]:
    """Map a model id to the Vertex API that serves it, or None when unknown.

    ``image_gen.vertex.api`` overrides the prefix inference so ids outside
    the ``gemini-``/``imagen-`` families stay usable.
    """
    override = str(_load_vertex_section().get("api") or "").strip().lower()
    if override in ("generate_content", "generatecontent"):
        return "generate_content"
    if override == "predict":
        return "predict"
    lowered = model_id.lower()
    if lowered.startswith("gemini"):
        return "generate_content"
    if lowered.startswith("imagen"):
        return "predict"
    return None


def _normalize_model_id(model_id: str) -> str:
    """Strip a ``google/`` publisher prefix — the endpoint path already
    carries ``/publishers/google/``, so chat-style ids like
    ``google/gemini-3.1-flash-image`` would otherwise 404."""
    candidate = model_id.strip()
    if candidate.lower().startswith("google/"):
        candidate = candidate[len("google/"):]
    return candidate


def _looks_like_vertex_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return lowered.startswith(("gemini", "imagen"))


def _resolve_model(model_kwarg: Optional[str]) -> str:
    """Pick the model per the module-docstring precedence."""
    candidate = _normalize_model_id(model_kwarg or "")
    if candidate and _looks_like_vertex_model(candidate):
        return candidate

    env_override = _normalize_model_id(os.environ.get("VERTEX_IMAGE_MODEL", ""))
    if env_override:
        return env_override

    cfg_model = _load_vertex_section().get("model")
    if isinstance(cfg_model, str) and cfg_model.strip():
        return _normalize_model_id(cfg_model)

    return DEFAULT_MODEL


def _resolve_location(api: str) -> str:
    """Endpoint location: explicit section key > adapter region (gemini) > regional default."""
    raw = _load_vertex_section().get("region")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if api == "generate_content":
        try:
            from agent.vertex_adapter import _resolve_region

            return _resolve_region()
        except Exception as exc:
            logger.debug("Vertex region resolution failed: %s", exc)
            return "global"
    # Imagen :predict is served regionally, not on the global endpoint.
    return "us-central1"


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


def _auth_required_response(prompt: str, aspect_ratio: str) -> Dict[str, Any]:
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
        aspect_ratio=aspect_ratio,
    )


def _timeout_seconds() -> float:
    raw = _load_vertex_section().get("timeout_seconds")
    try:
        value = float(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return DEFAULT_TIMEOUT_SECONDS


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


def _gemini_image_part(source: str) -> Dict[str, Any]:
    """Build one ``parts`` entry for a source image.

    Accepts a ``gs://`` URI (passed as fileData), a data URI, an http(s)
    URL (downloaded and inlined — Vertex fileData only accepts Cloud
    Storage URIs), or a local file path (guarded read, inlined).
    Raises ``ValueError`` with a user-facing message on failure.
    """
    ref = source.strip()
    lower = ref.lower()

    if lower.startswith("gs://"):
        return {"fileData": {"mimeType": _mime_from_path(Path(ref)), "fileUri": ref}}

    if lower.startswith("data:"):
        header, _, payload = ref.partition(",")
        if not payload or ";base64" not in header:
            raise ValueError("data: URI image inputs must be base64-encoded")
        mime = header[5:].split(";", 1)[0].strip() or "image/png"
        return {"inlineData": {"mimeType": mime, "data": payload}}

    if lower.startswith(("http://", "https://")):
        response = requests.get(ref, timeout=60)
        response.raise_for_status()
        if len(response.content) > _MAX_INPUT_IMAGE_BYTES:
            raise ValueError(f"Input image at {ref} exceeds 20MB")
        mime = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if not mime.startswith("image/"):
            mime = _mime_from_path(Path(ref.split("?", 1)[0]))
        b64 = base64.b64encode(response.content).decode("ascii")
        return {"inlineData": {"mimeType": mime, "data": b64}}

    # Local file path — enforce the shared credential-read guard before
    # touching bytes (same boundary as the other image providers).
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    path = Path(ref).expanduser()
    if not path.is_file():
        raise ValueError(
            f"image input '{ref}' is neither a URL, data URI, gs:// URI, nor an existing file"
        )
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"inlineData": {"mimeType": _mime_from_path(path), "data": b64}}


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def _post_json(url: str, token: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


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


def _extract_gemini_image(result: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str]:
    """Return (b64_data, mime, text_detail) from a generateContent response."""
    text_detail = ""
    for candidate in result.get("candidates") or []:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        for part in parts or []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                return inline["data"], mime, text_detail
            if isinstance(part.get("text"), str):
                text_detail += part["text"]
    return None, None, text_detail


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class VertexImageGenProvider(ImageGenProvider):
    """Google Vertex AI backend (Gemini image + Imagen models)."""

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
            "name": "Google Vertex AI",
            "badge": "paid",
            "tag": (
                "Nano Banana / Imagen via Google Cloud Vertex AI — "
                "service-account or ADC auth (no Gemini API key). "
                "Project/region under `vertex:` in config.yaml; model via "
                "`image_gen.vertex.model`."
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
            "max_reference_images": 6,
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        aspect = resolve_aspect_ratio(aspect_ratio)

        token, project_id = _resolve_credentials()
        if not token or not project_id:
            return _auth_required_response(prompt, aspect)

        source_images: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            source_images.append(image_url.strip())
        refs = normalize_reference_images(reference_image_urls)
        if refs:
            source_images.extend(refs)
        is_edit = bool(source_images)
        modality = "image" if is_edit else "text"

        model_id = _resolve_model(model)
        # Imagen's :predict surface is text-to-image only; edits route to the
        # default Gemini image model instead of failing (mirrors the xAI
        # provider auto-switching to its edit-capable model).
        if is_edit and _infer_api(model_id) == "predict":
            logger.debug(
                "Vertex image edit requested with predict-only model '%s'; using %s",
                model_id, DEFAULT_MODEL,
            )
            model_id = DEFAULT_MODEL

        api = _infer_api(model_id)
        if api is None:
            return error_response(
                error=(
                    f"Cannot infer which Vertex API serves model '{model_id}'. "
                    "Use a gemini-*/imagen-* id, or set image_gen.vertex.api to "
                    "'generate_content' or 'predict' in config.yaml."
                ),
                error_type="unknown_model",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        location = _resolve_location(api)
        method = "generateContent" if api == "generate_content" else "predict"
        url = _endpoint(project_id, location, model_id, method)

        if api == "generate_content":
            try:
                image_parts = [_gemini_image_part(source) for source in source_images]
            except ValueError as exc:
                return error_response(
                    error=str(exc),
                    error_type="invalid_image_url",
                    provider="vertex",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            except Exception as exc:
                return error_response(
                    error=f"Could not load source image: {exc}",
                    error_type="io_error",
                    provider="vertex",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            payload = _build_generate_content_payload(
                prompt, aspect, image_parts, is_edit=is_edit
            )
        else:
            payload = _build_predict_payload(prompt, aspect)

        logger.info(
            "Vertex image: %s via %s (location=%s, aspect=%s, modality=%s)",
            model_id, method, location, aspect, modality,
        )
        try:
            result = _post_json(url, token, payload, _timeout_seconds())
        except requests.HTTPError as exc:
            status, detail = _http_error_detail(exc)
            logger.error("Vertex image gen failed (%d): %s", status, detail)
            return error_response(
                error=f"Vertex image generation failed ({status}): {detail}",
                error_type="api_error",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error=f"Vertex image generation timed out ({int(_timeout_seconds())}s)",
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

        if api == "generate_content":
            b64, mime, text_detail = _extract_gemini_image(result)
            if not b64:
                feedback = result.get("promptFeedback") or {}
                block_reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
                if block_reason:
                    return error_response(
                        error=f"Vertex blocked the request ({block_reason}). {text_detail}".strip(),
                        error_type="safety_blocked",
                        provider="vertex",
                        model=model_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                return error_response(
                    error=(
                        "Vertex returned no image data"
                        + (f" (model said: {text_detail[:300]})" if text_detail else "")
                    ),
                    error_type="empty_response",
                    provider="vertex",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        else:
            predictions = result.get("predictions") or []
            first = predictions[0] if predictions and isinstance(predictions[0], dict) else {}
            b64 = first.get("bytesBase64Encoded")
            mime = first.get("mimeType") or "image/png"
            if not b64:
                filtered = first.get("raiFilteredReason") or "no predictions returned"
                return error_response(
                    error=f"Vertex Imagen returned no image data ({filtered})",
                    error_type="empty_response",
                    provider="vertex",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        try:
            extension = _MIME_EXTENSIONS.get((mime or "").lower(), "png")
            saved_path = save_b64_image(b64, prefix="vertex", extension=extension)
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="vertex",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        logger.info("Vertex image: %s completed -> %s", model_id, saved_path)

        extra: Dict[str, Any] = {"location": location}
        usage = result.get("usageMetadata") or result.get("usage")
        if usage:
            extra["usage"] = usage

        return success_response(
            image=str(saved_path),
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="vertex",
            modality=modality,
            extra=extra,
        )


def _build_generate_content_payload(
    prompt: str,
    aspect: str,
    image_parts: List[Dict[str, Any]],
    *,
    is_edit: bool,
) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    parts.extend(image_parts)
    generation_config: Dict[str, Any] = {"responseModalities": ["TEXT", "IMAGE"]}
    # On edits the output geometry follows the input image; only steer the
    # aspect ratio for pure text-to-image.
    image_config: Dict[str, Any] = {}
    if not is_edit:
        image_config["aspectRatio"] = _ASPECT_RATIOS.get(aspect, "16:9")
    image_size = _load_vertex_section().get("image_size")
    if isinstance(image_size, str) and image_size.strip():
        image_config["imageSize"] = image_size.strip()
    if image_config:
        generation_config["imageConfig"] = image_config
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }


def _build_predict_payload(prompt: str, aspect: str) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "sampleCount": 1,
        "aspectRatio": _ASPECT_RATIOS.get(aspect, "16:9"),
    }
    person_generation = _load_vertex_section().get("person_generation")
    if isinstance(person_generation, str) and person_generation.strip():
        parameters["personGeneration"] = person_generation.strip()
    return {
        "instances": [{"prompt": prompt}],
        "parameters": parameters,
    }


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(VertexImageGenProvider())
