"""Route a normalized plan to the matching media implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import UnsupportedMedia
from ..models import MediaDescriptor
from .audio_ops import render_audio
from .errors import MediaOperationFailure
from .inspector import inspect_media
from .video_ops import render_video
from .wav import SAMPLE_FORMAT_F32, SAMPLE_FORMAT_S16


@dataclass(frozen=True)
class RenderOutcome:
    data: bytes
    descriptor: MediaDescriptor
    operation_types: list[str]


def apply_media_operations(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
    *,
    output_sample_format: str | None = None,
) -> RenderOutcome:
    primary = inspect_media(primary_bytes)
    if secondary_bytes is not None:
        secondary = inspect_media(secondary_bytes)
        if secondary.kind != primary.kind:
            raise MediaOperationFailure(
                "MEDIA_KIND_MISMATCH",
                "Primary and secondary media must share a media kind.",
            )

    if primary.kind == "audio":
        primary_codec = str(primary.properties.get("codec", ""))
        if output_sample_format is None and primary_codec == SAMPLE_FORMAT_F32:
            # Re-checked here so a malformed persisted plan can never route a
            # float source through the integer-only legacy chain.
            raise MediaOperationFailure(
                "AUDIO_SAMPLE_FORMAT_REQUIRED",
                "Float audio sources require an explicit audio_sample_format.",
            )
        if (
            output_sample_format is not None
            and output_sample_format not in (SAMPLE_FORMAT_S16, SAMPLE_FORMAT_F32)
        ):
            raise MediaOperationFailure(
                "AUDIO_SAMPLE_FORMAT_UNSUPPORTED",
                "The requested audio delivery format is not supported.",
                details={"audio_sample_format": output_sample_format},
            )
        output = render_audio(
            primary_bytes,
            operations,
            secondary_bytes,
            output_sample_format=output_sample_format,
        )
    elif primary.kind == "video":
        if output_sample_format is not None:
            raise MediaOperationFailure(
                "AUDIO_SAMPLE_FORMAT_UNSUPPORTED",
                "audio_sample_format only applies to audio renditions.",
            )
        output = render_video(primary_bytes, operations, secondary_bytes)
    else:
        raise UnsupportedMedia("The primary media kind has no renderer.")

    descriptor = inspect_media(output)
    if descriptor.kind != primary.kind:
        raise UnsupportedMedia("The renderer changed the media kind unexpectedly.")
    return RenderOutcome(
        data=output,
        descriptor=descriptor,
        operation_types=[str(operation["type"]) for operation in operations],
    )
