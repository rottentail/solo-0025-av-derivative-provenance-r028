#!/usr/bin/env python3
"""Run bounded production workflow checks against an ephemeral HTTP server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import io
import json
import math
import struct
import sys
import tempfile
import threading
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avprovenance.http_api import ProvenanceHTTPServer, build_http_server
from avprovenance.media.wav import decode_wav


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_wav(
    *,
    frequency: float,
    duration_seconds: float = 0.6,
    sample_rate: int = 8000,
    amplitude: int = 9000,
) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    values = [
        int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
        for index in range(frame_count)
    ]
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(values)}h", *values))
    return payload.getvalue()


def make_float_wav(
    *,
    values: list[float],
    sample_rate: int = 8000,
    channels: int = 1,
) -> bytes:
    """Build a 32-bit IEEE float (WAVE format tag 3) RIFF file by hand."""
    fmt_body = struct.pack(
        "<HHIIHH",
        3,
        channels,
        sample_rate,
        sample_rate * channels * 4,
        channels * 4,
        32,
    )
    payload = struct.pack(f"<{len(values)}f", *values)
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


def make_pcm8_wav(values: list[int], *, sample_rate: int = 8000) -> bytes:
    """Build an 8-bit PCM WAV that inspection must reject."""
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(1)
        writer.setframerate(sample_rate)
        writer.writeframes(bytes(values))
    return payload.getvalue()


def make_y4m(
    *,
    width: int = 64,
    height: int = 36,
    frame_count: int = 5,
    rate_numerator: int = 12,
    rate_denominator: int = 1,
) -> bytes:
    header = (
        "YUV4MPEG2 "
        f"W{width} H{height} "
        f"F{rate_numerator}:{rate_denominator} "
        "Ip C420jpeg\n"
    ).encode("ascii")
    chroma_width = width // 2
    chroma_height = height // 2
    parts = [header]
    for frame_index in range(frame_count):
        parts.append(b"FRAME\n")
        y_plane = bytes(
            (x * 3 + y * 5 + frame_index * 11) % 256
            for y in range(height)
            for x in range(width)
        )
        u_plane = bytes(
            (96 + x * 2 + frame_index * 3) % 256
            for _y in range(chroma_height)
            for x in range(chroma_width)
        )
        v_plane = bytes(
            (144 + y * 2 + frame_index * 5) % 256
            for y in range(chroma_height)
            for _x in range(chroma_width)
        )
        parts.extend((y_plane, u_plane, v_plane))
    return b"".join(parts)


class HTTPClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        connection = http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=15,
        )
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            value = json.loads(raw.decode("utf-8"))
            return response.status, value
        finally:
            connection.close()

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        status, body = self.request(
            "POST",
            "/v1/assets",
            {
                "filename": filename,
                "data_base64": base64.b64encode(data).decode("ascii"),
            },
        )
        require_equal(status, 201, "asset upload status")
        return body

    def create_rendition(self, payload: dict[str, Any]) -> dict[str, Any]:
        status, body = self.request("POST", "/v1/renditions", payload)
        require_equal(status, 201, "rendition creation status")
        return body


@contextmanager
def running_server() -> Iterator[tuple[HTTPClient, ProvenanceHTTPServer]]:
    with tempfile.TemporaryDirectory(prefix="avprovenance-check-") as temporary:
        server = build_http_server("127.0.0.1", 0, Path(temporary))
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        thread.start()
        host, port = server.server_address[:2]
        try:
            yield HTTPClient(str(host), int(port)), server
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_ingest_media_asset(client: HTTPClient) -> dict[str, Any]:
    health_status, health = client.request("GET", "/health")
    require_equal(health_status, 200, "health status")
    require_equal(health["status"], "ok", "health payload")

    audio = make_wav(frequency=330)
    asset = client.upload("tone-source.wav", audio)
    require_equal(asset["kind"], "audio", "asset kind")
    require_equal(asset["sha256"], _sha256(audio), "asset digest")
    require_equal(asset["descriptor"]["container"], "wav", "asset container")
    require("content_key" not in asset, "public asset exposed an internal content key")

    get_status, fetched = client.request("GET", f"/v1/assets/{asset['id']}")
    require_equal(get_status, 200, "asset get status")
    require_equal(fetched["sha256"], asset["sha256"], "asset read digest")

    bad_status, bad_body = client.request(
        "POST",
        "/v1/assets",
        {
            "filename": "unknown.bin",
            "data_base64": base64.b64encode(b"not-media").decode("ascii"),
        },
    )
    require_equal(bad_status, 415, "unsupported media status")
    require_equal(
        bad_body["error"]["code"],
        "UNSUPPORTED_MEDIA",
        "unsupported media code",
    )

    float_audio = make_float_wav(values=[1.5, -1.5, 0.75, -0.25, 0.0])
    float_asset = client.upload("float-source.wav", float_audio)
    require_equal(
        float_asset["descriptor"]["properties"]["codec"],
        "pcm_f32le",
        "float asset codec",
    )
    require_equal(
        float_asset["descriptor"]["properties"]["sample_format"],
        "pcm_f32le",
        "float asset sample format",
    )
    require_equal(
        float_asset["descriptor"]["properties"]["frame_count"],
        5,
        "float asset frame count",
    )

    pcm8 = make_pcm8_wav([128, 200, 64, 32, 255])
    pcm8_status, pcm8_body = client.request(
        "POST",
        "/v1/assets",
        {
            "filename": "eight-bit.wav",
            "data_base64": base64.b64encode(pcm8).decode("ascii"),
        },
    )
    require_equal(pcm8_status, 415, "8-bit PCM rejection status")
    require_equal(
        pcm8_body["error"]["code"],
        "UNSUPPORTED_MEDIA",
        "8-bit PCM rejection code",
    )
    return {
        "workflow": "ingest-media-asset",
        "asset_id": asset["id"],
        "kind": asset["kind"],
        "digest": asset["sha256"],
        "float_asset_id": float_asset["id"],
    }


def run_execute_rendition(
    client: HTTPClient,
    server: ProvenanceHTTPServer,
) -> dict[str, Any]:
    primary_bytes = make_wav(frequency=440, amplitude=10000)
    secondary_bytes = make_wav(frequency=220, amplitude=7000)
    primary = client.upload("primary-tone.wav", primary_bytes)
    secondary = client.upload("secondary-tone.wav", secondary_bytes)

    audio_rendition = client.create_rendition(
        {
            "primary_asset_id": primary["id"],
            "secondary_asset_id": secondary["id"],
            "output_name": "mixed-tone.wav",
            "operations": [
                {
                    "type": "media_trim",
                    "start_seconds": 0.05,
                    "duration_seconds": 0.3,
                },
                {"type": "audio_gain", "db": -3},
                {
                    "type": "audio_mix",
                    "secondary_gain": -6,
                    "primary_weight": 0.75,
                    "secondary_weight": 0.4,
                },
                {"type": "audio_normalize", "target_peak": 0.8},
            ],
        }
    )
    require_equal(audio_rendition["status"], "completed", "audio rendition status")
    history = [item["to"] for item in audio_rendition["transitions"]]
    require_equal(
        history,
        ["queued", "running", "completed"],
        "audio state history",
    )
    require(
        audio_rendition["output_asset"]["descriptor"]["container"] == "wav",
        "audio output descriptor was not WAV",
    )

    video_bytes = make_y4m()
    video = client.upload("source-frames.y4m", video_bytes)
    video_rendition = client.create_rendition(
        {
            "primary_asset_id": video["id"],
            "output_name": "cropped-frames.y4m",
            "operations": [
                {
                    "type": "video_crop",
                    "x": 8,
                    "y": 6,
                    "width": 32,
                    "height": 18,
                },
                {"type": "video_luma_offset", "amount": 12},
                {"type": "video_mirror"},
            ],
        }
    )
    require_equal(video_rendition["status"], "completed", "video rendition status")
    properties = video_rendition["output_asset"]["descriptor"]["properties"]
    require_equal(properties["width"], 32, "video output width")
    require_equal(properties["height"], 18, "video output height")

    lineage_status, lineage = client.request(
        "GET",
        f"/v1/renditions/{audio_rendition['id']}/lineage",
    )
    require_equal(lineage_status, 200, "lineage status")
    require_equal(len(lineage["parents"]), 2, "lineage parent count")
    require_equal(
        lineage["edge"]["plan_digest"],
        audio_rendition["plan_digest"],
        "lineage plan digest",
    )
    invalid_status, invalid_body = client.request(
        "POST",
        "/v1/renditions",
        {
            "primary_asset_id": primary["id"],
            "output_name": "invalid.wav",
            "operations": [
                {
                    "type": "video_crop",
                    "x": 0,
                    "y": 0,
                    "width": 2,
                    "height": 2,
                }
            ],
        },
    )
    require_equal(invalid_status, 422, "invalid plan status")
    require_equal(invalid_body["error"]["code"], "PLAN_INVALID", "invalid plan code")

    # --- 32-bit float path -------------------------------------------------
    # Over-full-scale samples must survive attenuation and normalization when
    # the delivery representation stays float.
    float_values = [1.5, -1.5, 0.75, -0.25, 0.0]
    float_source = client.upload(
        "float-primary.wav",
        make_float_wav(values=float_values),
    )
    require_equal(
        float_source["descriptor"]["properties"]["codec"],
        "pcm_f32le",
        "float source codec",
    )

    float_operations = [
        {"type": "audio_gain", "db": -6.0},
        {"type": "audio_normalize", "target_peak": 0.5},
    ]
    float_rendition = client.create_rendition(
        {
            "primary_asset_id": float_source["id"],
            "output_name": "float-derived.wav",
            "audio_sample_format": "pcm_f32le",
            "operations": float_operations,
        }
    )
    require_equal(
        float_rendition["status"],
        "completed",
        "float rendition status",
    )
    require_equal(
        float_rendition["audio_sample_format"],
        "pcm_f32le",
        "persisted float sample format",
    )
    float_output = float_rendition["output_asset"]
    require_equal(
        float_output["descriptor"]["properties"]["codec"],
        "pcm_f32le",
        "float output codec",
    )
    require_equal(
        float_output["descriptor"]["properties"]["sample_format"],
        "pcm_f32le",
        "float output sample format",
    )
    float_record = server.application.repository.get_asset(float_output["id"])
    float_bytes = server.application.content_files.read(float_record.content_key)
    float_clip = decode_wav(float_bytes)
    float_peak = max(abs(v) for v in float_clip.channels[0])
    require(
        abs(float_peak - 0.5) < 1e-6,
        f"float normalize peak must be 0.5, got {float_peak}",
    )
    require(
        float_clip.channels[0][0] > 0.0
        and float_clip.channels[0][1] < 0.0,
        "over-range float samples must keep their shape after attenuation",
    )

    # A float source without an explicit delivery choice is rejected at plan
    # time rather than silently flattened into the integer chain.
    missing_choice_status, missing_choice_body = client.request(
        "POST",
        "/v1/renditions",
        {
            "primary_asset_id": float_source["id"],
            "output_name": "no-choice.wav",
            "operations": float_operations,
        },
    )
    require_equal(
        missing_choice_status,
        422,
        "float source without sample format choice status",
    )
    require_equal(
        missing_choice_body["error"]["code"],
        "PLAN_INVALID",
        "float source without sample format choice code",
    )

    # The same float plan delivered as 16-bit PCM applies one quantization at
    # the end; the descriptor must describe the actual bytes.
    int_delivery = client.create_rendition(
        {
            "primary_asset_id": float_source["id"],
            "output_name": "int-delivery.wav",
            "audio_sample_format": "pcm_s16le",
            "operations": float_operations,
        }
    )
    require_equal(
        int_delivery["output_asset"]["descriptor"]["properties"]["codec"],
        "pcm_s16le",
        "integer delivery codec",
    )
    require(
        "sample_format"
        not in int_delivery["output_asset"]["descriptor"]["properties"],
        "integer delivery descriptor must stay legacy-shaped",
    )

    # Mixed integer and float sources require and honor one delivery choice.
    integer_source = client.upload(
        "integer-aux.wav",
        make_wav(frequency=330, amplitude=12000),
    )
    mixed_rendition = client.create_rendition(
        {
            "primary_asset_id": integer_source["id"],
            "secondary_asset_id": float_source["id"],
            "output_name": "mixed-float.wav",
            "audio_sample_format": "pcm_f32le",
            "operations": [
                {
                    "type": "audio_mix",
                    "secondary_gain": 0.0,
                    "primary_weight": 1.0,
                    "secondary_weight": 1.0,
                }
            ],
        }
    )
    require_equal(
        mixed_rendition["output_asset"]["descriptor"]["properties"]["codec"],
        "pcm_f32le",
        "mixed-source float delivery codec",
    )
    require_equal(
        mixed_rendition["audio_sample_format"],
        "pcm_f32le",
        "mixed-source persisted sample format",
    )

    # The delivery choice is part of the plan digest: identical operations with
    # different representations get different digests.
    mixed_integer_delivery = client.create_rendition(
        {
            "primary_asset_id": integer_source["id"],
            "secondary_asset_id": float_source["id"],
            "output_name": "mixed-int.wav",
            "audio_sample_format": "pcm_s16le",
            "operations": [
                {
                    "type": "audio_mix",
                    "secondary_gain": 0.0,
                    "primary_weight": 1.0,
                    "secondary_weight": 1.0,
                }
            ],
        }
    )
    require(
        mixed_integer_delivery["plan_digest"]
        != mixed_rendition["plan_digest"],
        "delivery format must participate in the plan digest",
    )

    bad_format_status, bad_format_body = client.request(
        "POST",
        "/v1/renditions",
        {
            "primary_asset_id": float_source["id"],
            "output_name": "bad-format.wav",
            "audio_sample_format": "pcm_s24le",
            "operations": float_operations,
        },
    )
    require_equal(bad_format_status, 422, "unknown sample format status")
    require_equal(
        bad_format_body["error"]["code"],
        "PLAN_INVALID",
        "unknown sample format code",
    )

    return {
        "workflow": "execute-rendition",
        "audio_rendition_id": audio_rendition["id"],
        "video_rendition_id": video_rendition["id"],
        "audio_history": history,
        "invalid_plan_code": invalid_body["error"]["code"],
        "video_dimensions": [
            properties["width"],
            properties["height"],
        ],
        "float_rendition_id": float_rendition["id"],
        "float_peak": float_peak,
        "mixed_rendition_id": mixed_rendition["id"],
    }


def run_verify_reproduction(
    client: HTTPClient,
    server: ProvenanceHTTPServer,
) -> dict[str, Any]:
    video_bytes = make_y4m(frame_count=7)
    video = client.upload("verification-source.y4m", video_bytes)
    rendition = client.create_rendition(
        {
            "primary_asset_id": video["id"],
            "output_name": "verification-output.y4m",
            "operations": [
                {"type": "media_trim", "start_frame": 1, "frame_count": 4},
                {
                    "type": "video_crop",
                    "x": 4,
                    "y": 4,
                    "width": 40,
                    "height": 20,
                },
                {"type": "video_luma_offset", "amount": -9},
                {"type": "video_mirror"},
            ],
        }
    )
    check_status, check = client.request(
        "POST",
        f"/v1/renditions/{rendition['id']}/integrity-checks",
        {},
    )
    require_equal(check_status, 201, "integrity check status")
    require_equal(check["status"], "verified", "integrity result")
    require_equal(
        check["expected_digest"],
        check["reproduced_digest"],
        "integrity digest equality",
    )
    require_equal(
        check["expected_descriptor"],
        check["reproduced_descriptor"],
        "integrity descriptor equality",
    )

    lineage_status, lineage = client.request(
        "GET",
        f"/v1/renditions/{rendition['id']}/lineage",
    )
    require_equal(lineage_status, 200, "verification lineage status")
    require_equal(len(lineage["parents"]), 1, "verification parent count")

    # A float rendition must also reproduce byte-for-byte, including its
    # over-full-scale samples and float media descriptor.
    float_source = client.upload(
        "float-verify-source.wav",
        make_float_wav(values=[1.5, -1.5, 0.75, -0.25, 0.0]),
    )
    float_rendition = client.create_rendition(
        {
            "primary_asset_id": float_source["id"],
            "output_name": "float-verify-output.wav",
            "audio_sample_format": "pcm_f32le",
            "operations": [
                {"type": "audio_gain", "db": -6.0},
                {"type": "audio_normalize", "target_peak": 0.5},
            ],
        }
    )
    float_check_status, float_check = client.request(
        "POST",
        f"/v1/renditions/{float_rendition['id']}/integrity-checks",
        {},
    )
    require_equal(
        float_check_status,
        201,
        "float integrity check status",
    )
    require_equal(
        float_check["status"],
        "verified",
        "float integrity result",
    )
    require_equal(
        float_check["expected_digest"],
        float_check["reproduced_digest"],
        "float integrity digest equality",
    )
    require_equal(
        float_check["expected_descriptor"],
        float_check["reproduced_descriptor"],
        "float integrity descriptor equality",
    )
    require_equal(
        float_check["expected_descriptor"]["properties"]["codec"],
        "pcm_f32le",
        "float integrity descriptor codec",
    )

    output_asset_id = rendition["output_asset"]["id"]

    def drift_metadata(state: dict[str, Any]) -> None:
        state["assets"][output_asset_id]["sha256"] = "0" * 64

    server.application.state_file.mutate(drift_metadata)
    mismatch_status, mismatch = client.request(
        "POST",
        f"/v1/renditions/{rendition['id']}/integrity-checks",
        {},
    )
    require_equal(mismatch_status, 409, "mismatch status")
    require_equal(
        mismatch["error"]["code"],
        "INTEGRITY_MISMATCH",
        "mismatch error code",
    )
    require_equal(
        mismatch["error"]["details"]["audit"]["status"],
        "mismatch",
        "persisted mismatch status",
    )
    return {
        "workflow": "verify-reproduction",
        "rendition_id": rendition["id"],
        "integrity_check_id": check["id"],
        "digest": check["reproduced_digest"],
        "mismatch_status": mismatch["error"]["details"]["audit"]["status"],
        "float_integrity_check_id": float_check["id"],
        "float_status": float_check["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "workflow",
        choices=(
            "ingest-media-asset",
            "execute-rendition",
            "verify-reproduction",
        ),
    )
    args = parser.parse_args()

    with running_server() as (client, server):
        if args.workflow == "ingest-media-asset":
            result = run_ingest_media_asset(client)
        elif args.workflow == "execute-rendition":
            result = run_execute_rendition(client, server)
        else:
            result = run_verify_reproduction(client, server)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
