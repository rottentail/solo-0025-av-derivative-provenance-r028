"""PCM audio operations with explicit clipping and length rules.

Two deterministic processing interpretations live side by side:

- The legacy integer engine operates on signed 16-bit sample values and rounds
  and clips after every gain, normalization, and mix step.  Recipes saved
  before float support always use this engine and reproduce identical bytes.
- The full-scale float engine operates on 32-bit float samples where ``1.0``
  is integer full scale.  Samples above full scale survive gain, mixing, and
  normalization without intermediate rounding or clipping; the single
  quantization happens when the chosen delivery format is encoded.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import MediaOperationFailure
from .wav import (
    SAMPLE_FORMAT_F32,
    SAMPLE_FORMAT_S16,
    SUPPORTED_SAMPLE_FORMATS,
    AudioClip,
    decode_wav,
    decode_wav_as_float,
    encode_float_clip,
    encode_wav,
    replace_channels,
)


PCM_MIN = -32768
PCM_MAX = 32767


def _clip_sample(value: float) -> int:
    if value >= PCM_MAX:
        return PCM_MAX
    if value <= PCM_MIN:
        return PCM_MIN
    return int(round(value))


def _gain_factor(db: float) -> float:
    return math.pow(10.0, float(db) / 20.0)


def trim_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    start = int(round(float(operation["start_seconds"]) * clip.sample_rate))
    duration = int(round(float(operation["duration_seconds"]) * clip.sample_rate))
    if start < 0 or start >= clip.frame_count:
        raise MediaOperationFailure(
            "AUDIO_TRIM_OUT_OF_RANGE",
            "The requested audio trim starts outside the source.",
        )
    end = min(clip.frame_count, start + max(1, duration))
    if end <= start:
        raise MediaOperationFailure(
            "AUDIO_TRIM_EMPTY",
            "The requested audio trim produces no sample frames.",
        )
    return replace_channels(clip, (values[start:end] for values in clip.channels))


def gain_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    factor = _gain_factor(float(operation["db"]))
    return replace_channels(
        clip,
        (
            [_clip_sample(value * factor) for value in values]
            for values in clip.channels
        ),
    )


def normalize_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    peak = max(
        (abs(value) for channel in clip.channels for value in channel),
        default=0,
    )
    if peak == 0:
        return clip
    target = float(operation["target_peak"])
    desired = max(1.0, target * PCM_MAX)
    factor = desired / peak
    return replace_channels(
        clip,
        (
            [_clip_sample(value * factor) for value in values]
            for values in clip.channels
        ),
    )


def mix_audio(
    primary: AudioClip,
    secondary: AudioClip,
    operation: dict[str, Any],
) -> AudioClip:
    if primary.sample_rate != secondary.sample_rate:
        raise MediaOperationFailure(
            "AUDIO_MIX_RATE_MISMATCH",
            "Mixed audio sources must share a sample rate.",
        )
    if primary.channel_count != secondary.channel_count:
        raise MediaOperationFailure(
            "AUDIO_MIX_CHANNEL_MISMATCH",
            "Mixed audio sources must share a channel count.",
        )
    frame_count = max(primary.frame_count, secondary.frame_count)
    primary_weight = float(operation["primary_weight"])
    secondary_weight = float(operation["secondary_weight"])
    secondary_factor = _gain_factor(float(operation["secondary_gain"])) * secondary_weight

    mixed: list[list[int]] = []
    for channel_index in range(primary.channel_count):
        left = primary.channels[channel_index]
        right = secondary.channels[channel_index]
        values: list[int] = []
        for frame_index in range(frame_count):
            left_value = left[frame_index] if frame_index < len(left) else 0
            right_value = right[frame_index] if frame_index < len(right) else 0
            combined = left_value * primary_weight + right_value * secondary_factor
            values.append(_clip_sample(combined))
        mixed.append(values)
    return replace_channels(primary, mixed)


def _require_float_clip(clip: AudioClip) -> None:
    if clip.sample_format != SAMPLE_FORMAT_F32:
        raise MediaOperationFailure(
            "AUDIO_SAMPLE_FORMAT_MISMATCH",
            "The float audio pipeline received an integer clip.",
        )


def gain_audio_float(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    _require_float_clip(clip)
    factor = _gain_factor(float(operation["db"]))
    return replace_channels(
        clip,
        ([value * factor for value in values] for values in clip.channels),
    )


def normalize_audio_float(
    clip: AudioClip,
    operation: dict[str, Any],
) -> AudioClip:
    _require_float_clip(clip)
    peak = max(
        (abs(value) for channel in clip.channels for value in channel),
        default=0.0,
    )
    if peak == 0.0:
        return clip
    target = float(operation["target_peak"])
    factor = target / peak
    return replace_channels(
        clip,
        ([value * factor for value in values] for values in clip.channels),
    )


def mix_audio_float(
    primary: AudioClip,
    secondary: AudioClip,
    operation: dict[str, Any],
) -> AudioClip:
    _require_float_clip(primary)
    _require_float_clip(secondary)
    if primary.sample_rate != secondary.sample_rate:
        raise MediaOperationFailure(
            "AUDIO_MIX_RATE_MISMATCH",
            "Mixed audio sources must share a sample rate.",
        )
    if primary.channel_count != secondary.channel_count:
        raise MediaOperationFailure(
            "AUDIO_MIX_CHANNEL_MISMATCH",
            "Mixed audio sources must share a channel count.",
        )
    frame_count = max(primary.frame_count, secondary.frame_count)
    primary_weight = float(operation["primary_weight"])
    secondary_weight = float(operation["secondary_weight"])
    secondary_factor = _gain_factor(float(operation["secondary_gain"])) * secondary_weight

    mixed: list[list[float]] = []
    for channel_index in range(primary.channel_count):
        left = primary.channels[channel_index]
        right = secondary.channels[channel_index]
        values: list[float] = []
        for frame_index in range(frame_count):
            left_value = left[frame_index] if frame_index < len(left) else 0.0
            right_value = right[frame_index] if frame_index < len(right) else 0.0
            values.append(
                left_value * primary_weight + right_value * secondary_factor
            )
        mixed.append(values)
    return replace_channels(primary, mixed)


def _render_float(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
    output_sample_format: str,
) -> bytes:
    clip = decode_wav_as_float(primary_bytes)
    secondary = (
        decode_wav_as_float(secondary_bytes)
        if secondary_bytes is not None
        else None
    )
    mixed = False

    for operation in operations:
        operation_type = operation["type"]
        if operation_type == "media_trim":
            clip = trim_audio(clip, operation)
        elif operation_type == "audio_gain":
            clip = gain_audio_float(clip, operation)
        elif operation_type == "audio_normalize":
            clip = normalize_audio_float(clip, operation)
        elif operation_type == "audio_mix":
            if secondary is None:
                raise MediaOperationFailure(
                    "AUDIO_MIX_SOURCE_MISSING",
                    "audio_mix requires a secondary media asset.",
                )
            if mixed:
                raise MediaOperationFailure(
                    "AUDIO_MIX_DUPLICATE",
                    "An audio plan may mix the secondary source only once.",
                )
            clip = mix_audio_float(clip, secondary, operation)
            mixed = True
        else:
            raise MediaOperationFailure(
                "UNSUPPORTED_AUDIO_OPERATION",
                f"Unsupported audio operation: {operation_type}",
            )

    return encode_float_clip(clip, output_sample_format)


def render_audio(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
    *,
    output_sample_format: str | None = None,
) -> bytes:
    if output_sample_format is None:
        clip = decode_wav(primary_bytes)
        if clip.sample_format == SAMPLE_FORMAT_F32:
            raise MediaOperationFailure(
                "AUDIO_SAMPLE_FORMAT_REQUIRED",
                "Float audio sources require an explicit audio_sample_format.",
            )
        secondary = decode_wav(secondary_bytes) if secondary_bytes is not None else None
        if secondary is not None and secondary.sample_format == SAMPLE_FORMAT_F32:
            raise MediaOperationFailure(
                "AUDIO_SAMPLE_FORMAT_REQUIRED",
                "Float audio sources require an explicit audio_sample_format.",
            )
        mixed = False

        try:
            for operation in operations:
                operation_type = operation["type"]
                if operation_type == "media_trim":
                    clip = trim_audio(clip, operation)
                elif operation_type == "audio_gain":
                    clip = gain_audio(clip, operation)
                elif operation_type == "audio_normalize":
                    clip = normalize_audio(clip, operation)
                elif operation_type == "audio_mix":
                    if secondary is None:
                        raise MediaOperationFailure(
                            "AUDIO_MIX_SOURCE_MISSING",
                            "audio_mix requires a secondary media asset.",
                        )
                    if mixed:
                        raise MediaOperationFailure(
                            "AUDIO_MIX_DUPLICATE",
                            "An audio plan may mix the secondary source only once.",
                        )
                    clip = mix_audio(clip, secondary, operation)
                    mixed = True
                else:
                    raise MediaOperationFailure(
                        "UNSUPPORTED_AUDIO_OPERATION",
                        f"Unsupported audio operation: {operation_type}",
                    )
        except (KeyError, TypeError, ValueError) as exc:
            raise MediaOperationFailure(
                "AUDIO_OPERATION_INVALID",
                "The audio operation payload is malformed.",
            ) from exc

        return encode_wav(clip)

    if output_sample_format not in SUPPORTED_SAMPLE_FORMATS:
        raise MediaOperationFailure(
            "AUDIO_SAMPLE_FORMAT_UNSUPPORTED",
            "The requested audio delivery format is not supported.",
        )
    try:
        return _render_float(
            primary_bytes,
            operations,
            secondary_bytes,
            output_sample_format,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaOperationFailure(
            "AUDIO_OPERATION_INVALID",
            "The audio operation payload is malformed.",
        ) from exc
