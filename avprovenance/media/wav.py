"""PCM WAV inspection, decoding, and deterministic encoding.

Two sample representations are supported:

- ``pcm_s16le``: signed 16-bit integer PCM (WAVE format tag 1).  The legacy
  integer processing chain keeps its original step-by-step rounding and
  clipping semantics.
- ``pcm_f32le``: 32-bit IEEE float PCM (WAVE format tag 3).  Float samples are
  interpreted in full-scale units (``1.0`` is integer full scale) and may
  exceed ``+/-1.0`` without being clipped on ingest.

The stdlib ``wave`` module only accepts integer PCM, so float files are read
and written through a small explicit RIFF/WAVE chunk implementation.
"""

from __future__ import annotations

import array
import io
import math
import struct
import sys
import wave
from dataclasses import dataclass
from typing import Iterable

from ..errors import UnsupportedMedia
from ..models import MediaDescriptor


SAMPLE_FORMAT_S16 = "pcm_s16le"
SAMPLE_FORMAT_F32 = "pcm_f32le"
SUPPORTED_SAMPLE_FORMATS = (SAMPLE_FORMAT_S16, SAMPLE_FORMAT_F32)

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003


@dataclass(frozen=True)
class AudioClip:
    sample_rate: int
    channels: tuple[list, ...]
    sample_format: str = SAMPLE_FORMAT_S16

    @property
    def frame_count(self) -> int:
        return len(self.channels[0]) if self.channels else 0

    @property
    def channel_count(self) -> int:
        return len(self.channels)


@dataclass(frozen=True)
class _WaveChunks:
    format_tag: int
    channels: int
    sample_rate: int
    byte_rate: int
    block_align: int
    bits_per_sample: int
    data: bytes


def _parse_wave_chunks(data: bytes) -> _WaveChunks:
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise UnsupportedMedia(
            "The audio payload is not a readable RIFF/WAVE file.",
            details={"prefix_hex": data[:12].hex()},
        )
    offset = 12
    fmt_values: tuple[int, int, int, int, int, int] | None = None
    payload: bytes | None = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        body_start = offset + 8
        body_end = body_start + chunk_size
        if body_end > len(data):
            raise UnsupportedMedia(
                "A WAVE chunk extends past the end of the payload.",
                details={"chunk": chunk_id.decode("ascii", "replace")},
            )
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise UnsupportedMedia(
                    "The WAVE fmt chunk is too small to describe PCM audio.",
                    details={"chunk_size": chunk_size},
                )
            fmt_values = struct.unpack_from("<HHIIHH", data, body_start)
        elif chunk_id == b"data":
            payload = data[body_start:body_end]
            break
        # RIFF chunks are word-aligned with one padding byte when odd.
        offset = body_end + (chunk_size & 1)
    if fmt_values is None:
        raise UnsupportedMedia("The WAVE payload is missing a fmt chunk.")
    if payload is None:
        raise UnsupportedMedia("The WAVE payload is missing a data chunk.")
    format_tag, channels, sample_rate, byte_rate, block_align, bits = fmt_values
    return _WaveChunks(
        format_tag=format_tag,
        channels=channels,
        sample_rate=sample_rate,
        byte_rate=byte_rate,
        block_align=block_align,
        bits_per_sample=bits,
        data=payload,
    )


def _open_reader(data: bytes) -> wave.Wave_read:
    try:
        reader = wave.open(io.BytesIO(data), "rb")
    except (EOFError, wave.Error) as exc:
        raise UnsupportedMedia(
            "The audio payload is not a readable RIFF/WAVE file.",
            details={"reason": str(exc)},
        ) from exc
    return reader


def _validate_params(reader: wave.Wave_read) -> None:
    if reader.getcomptype() != "NONE":
        raise UnsupportedMedia(
            "Compressed WAV payloads are not supported.",
            details={"compression": reader.getcomptype()},
        )
    sample_width = reader.getsampwidth()
    if sample_width != 2:
        raise UnsupportedMedia(
            "Only signed 16-bit PCM WAV is supported.",
            details={"sample_width_bytes": sample_width},
        )
    channels = reader.getnchannels()
    if channels not in (1, 2):
        raise UnsupportedMedia(
            "WAV input must contain one or two channels.",
            details={"channels": channels},
        )
    sample_rate = reader.getframerate()
    if sample_rate < 8000 or sample_rate > 192000:
        raise UnsupportedMedia(
            "WAV sample rate is outside the supported range.",
            details={"sample_rate": sample_rate},
        )
    if reader.getnframes() <= 0:
        raise UnsupportedMedia("WAV input has no sample frames.")


def _validate_float_chunks(chunks: _WaveChunks) -> None:
    if chunks.format_tag != WAVE_FORMAT_IEEE_FLOAT:
        raise UnsupportedMedia(
            "Only PCM and IEEE float WAVE payloads are supported.",
            details={"format_tag": chunks.format_tag},
        )
    if chunks.bits_per_sample != 32:
        raise UnsupportedMedia(
            "Float WAV payloads must use 32-bit samples.",
            details={"bits_per_sample": chunks.bits_per_sample},
        )
    if chunks.channels not in (1, 2):
        raise UnsupportedMedia(
            "WAV input must contain one or two channels.",
            details={"channels": chunks.channels},
        )
    if chunks.sample_rate < 8000 or chunks.sample_rate > 192000:
        raise UnsupportedMedia(
            "WAV sample rate is outside the supported range.",
            details={"sample_rate": chunks.sample_rate},
        )
    frame_bytes = chunks.channels * 4
    if chunks.block_align != frame_bytes:
        raise UnsupportedMedia(
            "The WAVE block alignment does not match float frame size.",
            details={"block_align": chunks.block_align, "expected": frame_bytes},
        )
    if not chunks.data or len(chunks.data) % frame_bytes != 0:
        raise UnsupportedMedia("WAV float sample data is not frame-aligned.")
    _assert_finite(chunks.data)


def _float_frame_count(chunks: _WaveChunks) -> int:
    return len(chunks.data) // (chunks.channels * 4)


def _assert_finite(data: bytes) -> None:
    # Check the raw little-endian bytes: array.array("f") accepts NaN values
    # without complaint, so finiteness must be verified explicitly.
    for (value,) in struct.iter_unpack("<f", data):
        if not math.isfinite(value):
            raise UnsupportedMedia("Float WAV samples must all be finite.")


def inspect_wav(data: bytes) -> MediaDescriptor:
    chunks = _parse_wave_chunks(data)
    if (
        chunks.format_tag == WAVE_FORMAT_PCM
        and chunks.bits_per_sample == 16
    ):
        reader = _open_reader(data)
        try:
            _validate_params(reader)
            sample_rate = reader.getframerate()
            frame_count = reader.getnframes()
            channels = reader.getnchannels()
        finally:
            reader.close()
        return MediaDescriptor(
            kind="audio",
            mime_type="audio/wav",
            container="wav",
            properties={
                "codec": SAMPLE_FORMAT_S16,
                "sample_rate": sample_rate,
                "channels": channels,
                "frame_count": frame_count,
                "duration_seconds": round(frame_count / sample_rate, 9),
            },
        )
    if (
        chunks.format_tag == WAVE_FORMAT_IEEE_FLOAT
        and chunks.bits_per_sample == 32
    ):
        _validate_float_chunks(chunks)
        frame_count = _float_frame_count(chunks)
        sample_rate = chunks.sample_rate
        return MediaDescriptor(
            kind="audio",
            mime_type="audio/wav",
            container="wav",
            properties={
                "codec": SAMPLE_FORMAT_F32,
                "sample_format": SAMPLE_FORMAT_F32,
                "sample_rate": sample_rate,
                "channels": chunks.channels,
                "frame_count": frame_count,
                "duration_seconds": round(frame_count / sample_rate, 9),
            },
        )
    raise UnsupportedMedia(
        "Only signed 16-bit PCM or 32-bit IEEE float WAV is supported.",
        details={
            "format_tag": chunks.format_tag,
            "bits_per_sample": chunks.bits_per_sample,
        },
    )


def decode_wav(data: bytes) -> AudioClip:
    chunks = _parse_wave_chunks(data)
    if chunks.format_tag == WAVE_FORMAT_PCM and chunks.bits_per_sample == 16:
        reader = _open_reader(data)
        try:
            _validate_params(reader)
            channels = reader.getnchannels()
            sample_rate = reader.getframerate()
            raw = reader.readframes(reader.getnframes())
        finally:
            reader.close()

        interleaved = array.array("h")
        interleaved.frombytes(raw)
        if sys.byteorder != "little":
            interleaved.byteswap()
        if len(interleaved) % channels != 0:
            raise UnsupportedMedia("WAV sample data is not frame-aligned.")
        channel_values = tuple(
            interleaved[index::channels].tolist() for index in range(channels)
        )
        return AudioClip(
            sample_rate=sample_rate,
            channels=channel_values,
            sample_format=SAMPLE_FORMAT_S16,
        )
    if chunks.format_tag == WAVE_FORMAT_IEEE_FLOAT and chunks.bits_per_sample == 32:
        _validate_float_chunks(chunks)
        interleaved = array.array("f")
        interleaved.frombytes(chunks.data)
        if sys.byteorder != "little":
            interleaved.byteswap()
        channel_count = chunks.channels
        if len(interleaved) % channel_count != 0:
            raise UnsupportedMedia("WAV sample data is not frame-aligned.")
        channel_values = tuple(
            interleaved[index::channel_count].tolist()
            for index in range(channel_count)
        )
        return AudioClip(
            sample_rate=chunks.sample_rate,
            channels=channel_values,
            sample_format=SAMPLE_FORMAT_F32,
        )
    raise UnsupportedMedia(
        "Only signed 16-bit PCM or 32-bit IEEE float WAV is supported.",
        details={
            "format_tag": chunks.format_tag,
            "bits_per_sample": chunks.bits_per_sample,
        },
    )


def decode_wav_as_float(data: bytes) -> AudioClip:
    """Decode either representation into full-scale 32-bit float channels.

    Integer samples are mapped linearly with ``32768`` as full scale; float
    samples keep their stored full-scale values verbatim, including values
    beyond ``+/-1.0``.
    """
    clip = decode_wav(data)
    if clip.sample_format == SAMPLE_FORMAT_F32:
        return clip
    channel_values = tuple(
        [value / 32768.0 for value in channel] for channel in clip.channels
    )
    return AudioClip(
        sample_rate=clip.sample_rate,
        channels=channel_values,
        sample_format=SAMPLE_FORMAT_F32,
    )


def _interleave(channels: tuple[list, ...], frame_count: int) -> list:
    channel_count = len(channels)
    result = [0] * (frame_count * channel_count)
    for channel_index, values in enumerate(channels):
        result[channel_index::channel_count] = values[:frame_count]
    return result


def _build_wave(
    *,
    format_tag: int,
    channels: int,
    sample_rate: int,
    bits_per_sample: int,
    block_align: int,
    payload: bytes,
) -> bytes:
    fmt_body = struct.pack(
        "<HHIIHH",
        format_tag,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits_per_sample,
    )
    riff_size = 4 + (8 + len(fmt_body)) + (8 + len(payload))
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", riff_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", len(fmt_body)),
            fmt_body,
            b"data",
            struct.pack("<I", len(payload)),
            payload,
        )
    )


def _encode_float_wav(clip: AudioClip) -> bytes:
    if not clip.channels:
        raise UnsupportedMedia("Cannot encode an audio clip without channels.")
    if any(len(channel) != clip.frame_count for channel in clip.channels):
        raise UnsupportedMedia("Audio channels do not share one frame count.")
    values = _interleave(clip.channels, clip.frame_count)
    payload = struct.pack(f"<{len(values)}f", *values)
    return _build_wave(
        format_tag=WAVE_FORMAT_IEEE_FLOAT,
        channels=len(clip.channels),
        sample_rate=clip.sample_rate,
        bits_per_sample=32,
        block_align=len(clip.channels) * 4,
        payload=payload,
    )


def quantize_float_to_s16(value: float) -> int:
    """Apply the legacy rounding and clipping rule to one full-scale value."""
    scaled = value * 32768.0
    if scaled >= 32767:
        return 32767
    if scaled <= -32768:
        return -32768
    return int(round(scaled))


def encode_float_clip(clip: AudioClip, sample_format: str) -> bytes:
    """Encode a full-scale float clip in the requested delivery format."""
    if sample_format == SAMPLE_FORMAT_F32:
        return _encode_float_wav(clip)
    if sample_format == SAMPLE_FORMAT_S16:
        if not clip.channels:
            raise UnsupportedMedia("Cannot encode an audio clip without channels.")
        if any(len(channel) != clip.frame_count for channel in clip.channels):
            raise UnsupportedMedia("Audio channels do not share one frame count.")
        integer_channels = tuple(
            [quantize_float_to_s16(value) for value in channel]
            for channel in clip.channels
        )
        integer_clip = AudioClip(
            sample_rate=clip.sample_rate,
            channels=integer_channels,
            sample_format=SAMPLE_FORMAT_S16,
        )
        return encode_wav(integer_clip)
    raise UnsupportedMedia(
        "Unsupported audio sample format.",
        details={"sample_format": sample_format},
    )


def encode_wav(clip: AudioClip) -> bytes:
    if clip.sample_format == SAMPLE_FORMAT_F32:
        return _encode_float_wav(clip)
    if not clip.channels:
        raise UnsupportedMedia("Cannot encode an audio clip without channels.")
    if any(len(channel) != clip.frame_count for channel in clip.channels):
        raise UnsupportedMedia("Audio channels do not share one frame count.")
    values = array.array("h", _interleave(clip.channels, clip.frame_count))
    if sys.byteorder != "little":
        values.byteswap()
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(len(clip.channels))
        writer.setsampwidth(2)
        writer.setframerate(clip.sample_rate)
        writer.writeframes(values.tobytes())
    return payload.getvalue()


def replace_channels(
    clip: AudioClip,
    channels: Iterable[list],
) -> AudioClip:
    values = tuple(list(channel) for channel in channels)
    if not values:
        raise UnsupportedMedia("Audio operations produced no channels.")
    expected = len(values[0])
    if any(len(channel) != expected for channel in values):
        raise UnsupportedMedia("Audio operation produced misaligned channels.")
    return AudioClip(
        sample_rate=clip.sample_rate,
        channels=values,
        sample_format=clip.sample_format,
    )
