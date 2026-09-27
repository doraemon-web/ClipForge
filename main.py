import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

import requests
import yt_dlp

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware


# =========================================================
# PATHS
# =========================================================

STATIC_DIR = BASE_DIR / "static"
ASSETS_DIR = BASE_DIR / "assets"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SERVER SETTINGS
# =========================================================

APP_VERSION = "3.0.0"

# Render supplies PORT automatically.
# Locally, this falls back to 8000.
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

MAX_VIDEO_DURATION = int(
    os.getenv("MAX_VIDEO_DURATION", "7200")
)

MAX_CLIP_DURATION = int(
    os.getenv("MAX_CLIP_DURATION", "600")
)

MAX_CONCURRENT_JOBS = int(
    os.getenv("MAX_CONCURRENT_JOBS", "2")
)

RATE_LIMIT_REQUESTS = int(
    os.getenv("RATE_LIMIT_REQUESTS", "30")
)

RATE_LIMIT_WINDOW = int(
    os.getenv("RATE_LIMIT_WINDOW", "60")
)

MAX_CAPTION_DOWNLOAD_BYTES = int(
    os.getenv(
        "MAX_CAPTION_DOWNLOAD_BYTES",
        str(10 * 1024 * 1024),
    )
)

MAX_THUMBNAIL_DOWNLOAD_BYTES = int(
    os.getenv(
        "MAX_THUMBNAIL_DOWNLOAD_BYTES",
        str(20 * 1024 * 1024),
    )
)


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="ClipForge",
    version=APP_VERSION,
    docs_url="/api/docs",
    redoc_url=None,
)


# =========================================================
# SECURITY HEADERS
# =========================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = (
            "strict-origin-when-cross-origin"
        )
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )

        return response


app.add_middleware(SecurityHeadersMiddleware)


# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/assets",
    StaticFiles(directory=str(ASSETS_DIR)),
    name="assets",
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# =========================================================
# CONSTANTS
# =========================================================

YOUTUBE_HOSTS = {
    "youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}

VIDEO_QUALITIES = {
    "best",
    "2160",
    "1440",
    "1080",
    "720",
    "480",
    "360",
}

VIDEO_FORMATS = {
    "mp4",
    "mkv",
    "webm",
}

AUDIO_FORMATS = {
    "mp3",
    "m4a",
    "opus",
    "wav",
}

THUMBNAIL_FORMATS = {
    "jpg",
    "png",
    "webp",
}

CAPTION_FORMATS = {
    "srt",
    "vtt",
    "ass",
    "lrc",
}


# =========================================================
# CONCURRENCY / RATE LIMITING
# =========================================================

JOB_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

RATE_LOCK = threading.Lock()
RATE_HISTORY = defaultdict(deque)


def get_client_ip(request: Request) -> str:
    """
    Get the client IP.

    Render and other reverse proxies may provide
    X-Forwarded-For. Cloudflare's header is also supported
    for compatibility.
    """

    cloudflare_ip = request.headers.get(
        "CF-Connecting-IP"
    )

    if cloudflare_ip:
        return cloudflare_ip.strip()

    forwarded_for = request.headers.get(
        "X-Forwarded-For"
    )

    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


def check_rate_limit(request: Request):
    now = time.monotonic()
    client_ip = get_client_ip(request)

    with RATE_LOCK:
        history = RATE_HISTORY[client_ip]

        while history and (
            now - history[0] > RATE_LIMIT_WINDOW
        ):
            history.popleft()

        if len(history) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Too many requests. "
                    "Please wait a moment and try again."
                ),
            )

        history.append(now)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path.startswith("/api/"):
            check_rate_limit(request)

        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


# =========================================================
# BASIC HELPERS
# =========================================================

def is_youtube_url(url: str) -> bool:
    if not url:
        return False

    url = url.strip()

    try:
        parsed = urlparse(url)

        hostname = (parsed.hostname or "").lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        if hostname not in YOUTUBE_HOSTS:
            return False

        return parsed.scheme in {"http", "https"} and bool(
            parsed.netloc
        )

    except Exception:
        return False


def clean_filename(name: str) -> str:
    name = str(name or "")

    name = re.sub(
        r'[<>:"/\\|?*\x00-\x1F]',
        "",
        name,
    )

    name = re.sub(
        r"\s+",
        " ",
        name,
    ).strip()

    name = name.rstrip(".")

    return name[:150] or "download"


def seconds_to_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours:02d}-{minutes:02d}-{secs:02d}"

    return f"{minutes:02d}-{secs:02d}"


def format_duration(seconds) -> str:
    seconds = int(seconds or 0)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def get_video_id(url: str):
    try:
        parsed = urlparse(url)

        hostname = (parsed.hostname or "").lower()

        if hostname == "youtu.be":
            return parsed.path.strip("/").split("/")[0]

        if "youtube.com" in hostname:
            query = parse_qs(parsed.query)

            if "v" in query:
                return query["v"][0]

            parts = parsed.path.strip("/").split("/")

            if (
                len(parts) >= 2
                and parts[0] in {
                    "shorts",
                    "embed",
                    "live",
                }
            ):
                return parts[1]

    except Exception:
        pass

    return None


def remove_file(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def remove_directory(path):
    try:
        if path:
            shutil.rmtree(
                path,
                ignore_errors=True,
            )
    except Exception:
        pass


def run_process(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=MAX_VIDEO_DURATION + 600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-5000:]
            or "Process failed."
        )

    return result


async def run_job(function, *args):
    async with JOB_SEMAPHORE:
        return await asyncio.to_thread(
            function,
            *args,
        )


# =========================================================
# ERROR HANDLING
# =========================================================

def public_error(exc: Exception) -> str:
    message = str(exc).strip()

    if not message:
        return "The operation failed."

    if len(message) > 800:
        message = message[-800:]

    return message


# =========================================================
# YT-DLP
# =========================================================

def extract_info(url: str):
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": False,
        "socket_timeout": 30,
        "retries": 2,
        "fragment_retries": 2,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(
            url,
            download=False,
        )


def quality_format(height: int):
    return (
        f"bv*[height<={height}][ext=mp4]+ba[ext=m4a]/"
        f"bv*[height<={height}]+ba/"
        f"b[height<={height}][ext=mp4]/"
        f"b[height<={height}]"
    )


def get_stream_urls(info, quality):
    formats = info.get("formats", [])

    videos = []
    audios = []

    for fmt in formats:
        url = fmt.get("url")

        if not url:
            continue

        vcodec = fmt.get("vcodec")
        acodec = fmt.get("acodec")

        if vcodec and vcodec != "none":
            videos.append(fmt)

        if acodec and acodec != "none":
            audios.append(fmt)

    if not videos:
        raise RuntimeError(
            "No video stream is available."
        )

    if quality == "best":
        max_height = max(
            f.get("height") or 0
            for f in videos
        )
    else:
        max_height = int(quality)

    suitable_videos = [
        f
        for f in videos
        if (f.get("height") or 0) <= max_height
    ]

    if not suitable_videos:
        suitable_videos = videos

    suitable_videos.sort(
        key=lambda f: (
            f.get("height") or 0,
            f.get("fps") or 0,
            f.get("tbr") or 0,
        ),
        reverse=True,
    )

    video = suitable_videos[0]

    audios.sort(
        key=lambda f: (
            f.get("abr") or 0,
            f.get("tbr") or 0,
        ),
        reverse=True,
    )

    audio = audios[0] if audios else None

    return (
        video.get("url"),
        audio.get("url") if audio else None,
    )


# =========================================================
# COMMON VALIDATION
# =========================================================

async def get_checked_info(url: str):
    if not is_youtube_url(url):
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid YouTube URL.",
        )

    try:
        info = await run_job(
            extract_info,
            url,
        )

        duration = int(
            info.get("duration") or 0
        )

        if duration > MAX_VIDEO_DURATION:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"This video is longer than the "
                    f"public server limit of "
                    f"{format_duration(MAX_VIDEO_DURATION)}."
                ),
            )

        return info

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=public_error(exc),
        )


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "ClipForge",
        "version": APP_VERSION,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "yt_dlp": True,
    }


# =========================================================
# METADATA
# =========================================================

@app.get("/api/info")
async def video_info(
    url: str = Query(
        ...,
        min_length=1,
        max_length=2048,
    ),
):
    info = await get_checked_info(url)

    duration = info.get("duration") or 0

    resolutions = sorted(
        {
            f.get("height")
            for f in info.get("formats", [])
            if f.get("vcodec") != "none"
            and f.get("height")
        },
        reverse=True,
    )

    subtitles = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}

    return {
        "id": info.get("id"),
        "title": (
            info.get("title")
            or "Unknown title"
        ),
        "channel": (
            info.get("channel")
            or info.get("uploader")
            or "Unknown channel"
        ),
        "uploader": info.get("uploader"),
        "thumbnail": info.get("thumbnail"),
        "duration": duration,
        "duration_text": format_duration(duration),
        "webpage_url": (
            info.get("webpage_url")
            or url
        ),
        "view_count": info.get("view_count"),
        "upload_date": info.get("upload_date"),
        "resolutions": resolutions,
        "has_subtitles": bool(subtitles),
        "subtitle_languages": list(subtitles.keys()),
        "has_auto_captions": bool(automatic),
        "auto_caption_languages": list(
            automatic.keys()
        ),
        "is_live": bool(
            info.get("is_live")
        ),
        "age_limit": info.get("age_limit") or 0,
    }


# =========================================================
# THUMBNAIL INFO
# =========================================================

@app.get("/api/thumbnail")
async def thumbnail_info(
    url: str = Query(
        ...,
        min_length=1,
        max_length=2048,
    ),
):
    info = await get_checked_info(url)

    thumb = info.get("thumbnail")

    if not thumb:
        raise HTTPException(
            status_code=404,
            detail="Thumbnail unavailable.",
        )

    return {
        "url": thumb,
        "title": info.get("title"),
    }


# =========================================================
# SAFE HTTP DOWNLOAD
# =========================================================

def download_http_file(
    url: str,
    destination: Path,
    max_bytes: int,
    headers=None,
):
    session = requests.Session()

    request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/153.0 Safari/537.36"
        ),
    }

    if headers:
        request_headers.update(headers)

    with session.get(
        url,
        headers=request_headers,
        timeout=30,
        stream=True,
    ) as response:
        response.raise_for_status()

        content_length = response.headers.get(
            "Content-Length"
        )

        if content_length:
            try:
                if int(content_length) > max_bytes:
                    raise RuntimeError(
                        "Downloaded file is too large."
                    )
            except ValueError:
                pass

        total = 0

        with open(destination, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=64 * 1024
            ):
                if not chunk:
                    continue

                total += len(chunk)

                if total > max_bytes:
                    raise RuntimeError(
                        "Downloaded file is too large."
                    )

                file.write(chunk)


# =========================================================
# DOWNLOAD THUMBNAIL
# =========================================================

@app.get("/api/download-thumbnail")
async def download_thumbnail(
    url: str = Query(
        ...,
        min_length=1,
        max_length=2048,
    ),
    image_format: str = Query("jpg"),
):
    if image_format not in THUMBNAIL_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported thumbnail format.",
        )

    temp_dir = tempfile.mkdtemp(
        prefix="clipforge_thumbnail_"
    )

    try:
        info = await get_checked_info(url)

        thumbnail_url = info.get("thumbnail")

        if not thumbnail_url:
            raise RuntimeError(
                "Thumbnail unavailable."
            )

        source = Path(temp_dir) / "source"

        await run_job(
            download_http_file,
            thumbnail_url,
            source,
            MAX_THUMBNAIL_DOWNLOAD_BYTES,
            {
                "Referer": "https://www.youtube.com/",
            },
        )

        output = (
            Path(temp_dir)
            / f"thumbnail.{image_format}"
        )

        if image_format == "jpg":
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-q:v",
                "2",
                "-y",
                str(output),
            ]
        else:
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-y",
                str(output),
            ]

        await run_job(
            run_process,
            command,
        )

        title = clean_filename(
            info.get("title")
            or "thumbnail"
        )

        filename = (
            f"{title}_thumbnail."
            f"{image_format}"
        )

        return FileResponse(
            output,
            media_type={
                "jpg": "image/jpeg",
                "png": "image/png",
                "webp": "image/webp",
            }[image_format],
            filename=filename,
            content_disposition_type="attachment",
            background=BackgroundTask(
                remove_directory,
                temp_dir,
            ),
        )

    except HTTPException:
        remove_directory(temp_dir)
        raise

    except Exception as exc:
        remove_directory(temp_dir)

        raise HTTPException(
            status_code=500,
            detail=public_error(exc),
        )


# =========================================================
# FULL VIDEO
# =========================================================

@app.get("/api/download-video")
async def download_video(
    url: str = Query(
        ...,
        min_length=1,
        max_length=2048,
    ),
    quality: str = Query("best"),
    video_format: str = Query("mp4"),
):
    if quality not in VIDEO_QUALITIES:
        raise HTTPException(
            status_code=400,
            detail="Invalid quality.",
        )

    if video_format not in VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format.",
        )

    temp_dir = tempfile.mkdtemp(
        prefix="clipforge_video_"
    )

    try:
        info = await get_checked_info(url)

        title = clean_filename(
            info.get("title")
            or "video"
        )

        output_template = (
            Path(temp_dir)
            / "video.%(ext)s"
        )

        if quality == "best":
            selector = "bv*+ba/b"
        else:
            selector = quality_format(
                int(quality)
            )

        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": selector,
            "merge_output_format": video_format,
            "outtmpl": str(output_template),
            "retries": 2,
            "fragment_retries": 2,
            "socket_timeout": 30,
        }

        def download():
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])

        await run_job(download)

        files = [
            f
            for f in Path(temp_dir).glob("video.*")
            if f.is_file()
        ]

        if not files:
            raise RuntimeError(
                "Video download failed."
            )

        output = files[0]

        filename = (
            f"{title}.{video_format}"
        )

        return FileResponse(
            output,
            media_type={
                "mp4": "video/mp4",
                "mkv": "video/x-matroska",
                "webm": "video/webm",
            }[video_format],
            filename=filename,
            content_disposition_type="attachment",
            background=BackgroundTask(
                remove_directory,
                temp_dir,
            ),
        )

    except HTTPException:
        remove_directory(temp_dir)
        raise

    except Exception as exc:
        remove_directory(temp_dir)

        raise HTTPException(
            status_code=500,
            detail=public_error(exc),
        )


# =========================================================
# CLIP
# =========================================================

@app.get("/api/clip")
async def download_clip(
    url: str = Query(
        ...,
        min_length=1,
        max_length=2048,
    ),
    start: float = Query(0),
    end: float = Query(...),
    quality: str = Query("best"),
    video_format: str = Query("mp4"),
):
    if quality not in VIDEO_QUALITIES:
        raise HTTPException(
            status_code=400,
            detail="Invalid quality.",
        )

    if video_format not in VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format.",
        )

    if start < 0 or end <= start:
        raise HTTPException(
            status_code=400,
            detail="Invalid clip range.",
        )

    if end - start > MAX_CLIP_DURATION:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Clips are limited to "
                f"{format_duration(MAX_CLIP_DURATION)} "
                f"on the public server."
            ),
        )

    temp_dir = tempfile.mkdtemp(
        prefix="clipforge_clip_"
    )

    try:
        info = await get_checked_info(url)

        duration = info.get("duration") or 0

        if duration:
            end = min(
                end,
                duration,
            )

        if start >= end:
            raise HTTPException(
                status_code=400,
                detail="Invalid clip range.",
            )

        if end - start > MAX_CLIP_DURATION:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Clips are limited to "
                    f"{format_duration(MAX_CLIP_DURATION)}."
                ),
            )

        video_url, audio_url = await run_job(
            get_stream_urls,
            info,
            quality,
        )

        if not video_url:
            raise RuntimeError(
                "Unable to obtain video stream."
            )

        output = (
            Path(temp_dir)
            / f"clip.{video_format}"
        )

        clip_duration = end - start

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-i",
            video_url,
        ]

        if audio_url:
            command += [
                "-ss",
                str(start),
                "-i",
                audio_url,
                "-t",
                str(clip_duration),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
            ]
        else:
            command += [
                "-t",
                str(clip_duration),
                "-c",
                "copy",
            ]

        command += [
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            "-y",
            str(output),
        ]

        await run_job(
            run_process,
            command,
        )

        if not output.exists():
            raise RuntimeError(
                "FFmpeg did not create the clip."
            )

        title = clean_filename(
            info.get("title")
            or "video"
        )

        filename = (
            f"{title}_"
            f"{seconds_to_timestamp(start)}_"
            f"{seconds_to_timestamp(end)}."
            f"{video_format}"
        )

        media_type = {
            "mp4": "video/mp4",
            "mkv": "video/x-matroska",
            "webm": "video/webm",
        }[video_format]

        return FileResponse(
            output,
            media_type=media_type,
            filename=filename,
            content_disposition_type="attachment",
            background=BackgroundTask(
                remove_directory,
                temp_dir,
            ),
        )

    except HTTPException:
        remove_directory(temp_dir)
        raise

    except Exception as exc:
        remove_directory(temp_dir)

        raise HTTPException(
            status_code=500,
            detail=public_error(exc),
        )


# =========================================================
# AUDIO
# =========================================================

@app.get("/api/audio")
async def download_audio(
    url: str = Query(
        ...,
        min_length=1,
        max_length=2048,
    ),
    start: float = Query(0),
    end: float | None = Query(None),
    audio_format: str = Query("mp3"),
):
    if audio_format not in AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format.",
        )

    if start < 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio start time.",
        )

    temp_dir = tempfile.mkdtemp(
        prefix="clipforge_audio_"
    )

    try:
        info = await get_checked_info(url)

        duration = info.get("duration") or 0

        if end is None:
            end = duration

        if duration:
            end = min(
                end,
                duration,
            )

        if end <= start:
            raise HTTPException(
                status_code=400,
                detail="Invalid audio range.",
            )

        if end - start > MAX_CLIP_DURATION:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio extraction is limited to "
                    f"{format_duration(MAX_CLIP_DURATION)}."
                ),
            )

        audio_formats = [
            f
            for f in info.get("formats", [])
            if f.get("acodec") != "none"
            and f.get("url")
        ]

        audio_formats.sort(
            key=lambda f: (
                f.get("abr") or 0,
                f.get("tbr") or 0,
            ),
            reverse=True,
        )

        if not audio_formats:
            raise RuntimeError(
                "Audio stream unavailable."
            )

        audio_url = audio_formats[0]["url"]

        output = (
            Path(temp_dir)
            / f"audio.{audio_format}"
        )

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-i",
            audio_url,
            "-t",
            str(end - start),
            "-vn",
        ]

        if audio_format == "mp3":
            command += [
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
            ]

        elif audio_format == "m4a":
            command += [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]

        elif audio_format == "opus":
            command += [
                "-c:a",
                "libopus",
                "-b:a",
                "160k",
            ]

        elif audio_format == "wav":
            command += [
                "-c:a",
                "pcm_s16le",
            ]

        command += [
            "-y",
            str(output),
        ]

        await run_job(
            run_process,
            command,
        )

        title = clean_filename(
            info.get("title")
            or "audio"
        )

        filename = (
            f"{title}_"
            f"{seconds_to_timestamp(start)}_"
            f"{seconds_to_timestamp(end)}."
            f"{audio_format}"
        )

        media_types = {
            "mp3": "audio/mpeg",
            "m4a": "audio/mp4",
            "opus": "audio/ogg",
            "wav": "audio/wav",
        }

        return FileResponse(
            output,
            media_type=media_types[audio_format],
            filename=filename,
            content_disposition_type="attachment",
            background=BackgroundTask(
                remove_directory,
                temp_dir,
            ),
        )

    except HTTPException:
        remove_directory(temp_dir)
        raise

    except Exception as exc:
        remove_directory(temp_dir)

        raise HTTPException(
            status_code=500,
            detail=public_error(exc),
        )


# =========================================================
# CAPTION HELPERS
# =========================================================

def parse_timestamp(value: str) -> float:
    value = value.strip().replace(",", ".")

    parts = value.split(":")

    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])

        return (
            hours * 3600
            + minutes * 60
            + seconds
        )

    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])

        return (
            minutes * 60
            + seconds
        )

    return float(parts[0])


def format_srt_timestamp(seconds: float) -> str:
    seconds = max(0, seconds)

    milliseconds = int(
        round(
            (seconds - int(seconds)) * 1000
        )
    )

    whole = int(seconds)

    if milliseconds >= 1000:
        whole += 1
        milliseconds = 0

    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def format_vtt_timestamp(seconds: float) -> str:
    seconds = max(0, seconds)

    milliseconds = int(
        round(
            (seconds - int(seconds)) * 1000
        )
    )

    whole = int(seconds)

    if milliseconds >= 1000:
        whole += 1
        milliseconds = 0

    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{milliseconds:03d}"
    )


def strip_caption_tags(text: str) -> str:
    text = re.sub(
        r"<[^>]+>",
        "",
        text,
    )

    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )

    return text.strip()


def parse_vtt(text: str):
    cues = []

    blocks = re.split(
        r"\n\s*\n",
        text.replace("\r\n", "\n"),
    )

    for block in blocks:
        lines = [
            line.strip()
            for line in block.split("\n")
            if line.strip()
        ]

        timing_index = None

        for index, line in enumerate(lines):
            if "-->" in line:
                timing_index = index
                break

        if timing_index is None:
            continue

        timing = lines[timing_index]
        parts = timing.split("-->")

        if len(parts) != 2:
            continue

        try:
            start = parse_timestamp(
                parts[0].strip()
            )

            end_part = (
                parts[1]
                .strip()
                .split()[0]
            )

            end = parse_timestamp(
                end_part
            )

        except Exception:
            continue

        text_lines = lines[
            timing_index + 1:
        ]

        cue_text = "\n".join(
            strip_caption_tags(line)
            for line in text_lines
        ).strip()

        if cue_text:
            cues.append(
                (
                    start,
                    end,
                    cue_text,
                )
            )

    return cues


def parse_srt(text: str):
    cues = []

    blocks = re.split(
        r"\n\s*\n",
        text.replace("\r\n", "\n"),
    )

    for block in blocks:
        lines = [
            line.strip()
            for line in block.split("\n")
            if line.strip()
        ]

        timing_index = None

        for index, line in enumerate(lines):
            if "-->" in line:
                timing_index = index
                break

        if timing_index is None:
            continue

        parts = lines[timing_index].split(
            "-->"
        )

        if len(parts) != 2:
            continue

        try:
            start = parse_timestamp(
                parts[0]
            )

            end = parse_timestamp(
                parts[1]
            )

        except Exception:
            continue

        text_lines = lines[
            timing_index + 1:
        ]

        cue_text = "\n".join(
            strip_caption_tags(line)
            for line in text_lines
        ).strip()

        if cue_text:
            cues.append(
                (
                    start,
                    end,
                    cue_text,
                )
            )

    return cues


def parse_ttml(text: str):
    cues = []

    try:
        root = ET.fromstring(text)
    except Exception:
        return cues

    for element in root.iter():
        tag = element.tag

        if "}" in tag:
            tag = tag.split("}", 1)[1]

        if tag != "p":
            continue

        begin = element.attrib.get("begin")
        end = element.attrib.get("end")
        dur = element.attrib.get("dur")

        if not begin:
            continue

        try:
            start = parse_caption_time(begin)

            if end:
                finish = parse_caption_time(end)

            elif dur:
                finish = (
                    start
                    + parse_caption_time(dur)
                )

            else:
                finish = start + 2

        except Exception:
            continue

        text_value = "".join(
            element.itertext()
        ).strip()

        text_value = strip_caption_tags(
            text_value
        )

        if text_value:
            cues.append(
                (
                    start,
                    finish,
                    text_value,
                )
            )

    return cues


def parse_caption_time(value: str) -> float:
    value = value.strip()

    if value.endswith("ms"):
        return float(
            value[:-2]
        ) / 1000

    if value.endswith("s"):
        return float(
            value[:-1]
        )

    if value.endswith("m"):
        return float(
            value[:-1]
        ) * 60

    if value.endswith("h"):
        return float(
            value[:-1]
        ) * 3600

    return parse_timestamp(value)


def parse_caption_source(
    text: str,
    source_extension: str,
):
    extension = (
        source_extension
        .lower()
        .lstrip(".")
    )

    if extension == "vtt":
        return parse_vtt(text)

    if extension == "srt":
        return parse_srt(text)

    if extension in {
        "ttml",
        "xml",
    }:
        return parse_ttml(text)

    if extension in {
        "srv1",
        "srv2",
        "srv3",
    }:
        return parse_ttml(text)

    if "-->" in text:
        return parse_vtt(text)

    return parse_ttml(text)


def cues_to_srt(cues):
    output = []

    for index, (
        start,
        end,
        text,
    ) in enumerate(cues, 1):
        output.append(
            str(index)
        )

        output.append(
            f"{format_srt_timestamp(start)} --> "
            f"{format_srt_timestamp(end)}"
        )

        output.append(text)
        output.append("")

    return "\n".join(output)


def cues_to_vtt(cues):
    output = [
        "WEBVTT",
        "",
    ]

    for start, end, text in cues:
        output.append(
            f"{format_vtt_timestamp(start)} --> "
            f"{format_vtt_timestamp(end)}"
        )

        output.append(text)
        output.append("")

    return "\n".join(output)


def ass_escape(text: str) -> str:
    return (
        text
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def format_ass_timestamp(seconds: float) -> str:
    seconds = max(0, seconds)

    centiseconds = int(
        round(
            (seconds - int(seconds)) * 100
        )
    )

    whole = int(seconds)

    if centiseconds >= 100:
        whole += 1
        centiseconds = 0

    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{centiseconds:02d}"
    )


def cues_to_ass(cues):
    output = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, "
        "SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,48,&H00FFFFFF,"
        "&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,"
        "100,100,0,0,1,2,1,2,40,40,40,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text",
    ]

    for start, end, text in cues:
        output.append(
            "Dialogue: 0,"
            f"{format_ass_timestamp(start)},"
            f"{format_ass_timestamp(end)},"
            "Default,,0,0,0,,"
            f"{ass_escape(text)}"
        )

    output.append("")

    return "\n".join(output)


def cues_to_lrc(cues):
    output = []

    for start, _, text in cues:
        minutes = int(start // 60)
        seconds = start % 60

        output.append(
            f"[{minutes:02d}:{seconds:05.2f}] "
            f"{text.replace(chr(10), ' ')}"
        )

    output.append("")

    return "\n".join(output)


def get_caption_source(source_list):
    """
    Prefer VTT because it is easy to convert and
    commonly provided by YouTube.
    """

    if not source_list:
        return None

    preferred = [
        "vtt",
        "ttml",
        "srv3",
        "srv2",
        "srv1",
    ]

    for preferred_extension in preferred:
        for item in source_list:
            if (
                item.get("ext", "").lower()
                == preferred_extension
                and item.get("url")
            ):
                return item

    for item in source_list:
        if item.get("url"):
            return item

    return None


# =========================================================
# CAPTIONS
# =========================================================

@app.get("/api/captions")
async def captions(
    url: str = Query(
        ...,
        min_length=1,
        max_length=2048,
    ),
    language: str = Query(
        "en",
        min_length=1,
        max_length=50,
    ),
    caption_format: str = Query("srt"),
):
    if caption_format not in CAPTION_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported caption format.",
        )

    temp_dir = tempfile.mkdtemp(
        prefix="clipforge_caption_"
    )

    try:
        info = await get_checked_info(url)

        subtitles = info.get("subtitles") or {}
        automatic = info.get("automatic_captions") or {}

        available = list(
            dict.fromkeys(
                list(subtitles.keys())
                + list(automatic.keys())
            )
        )

        if not available:
            raise HTTPException(
                status_code=404,
                detail="No captions are available.",
            )

        selected = language

        if (
            selected not in subtitles
            and selected not in automatic
        ):
            family = selected.split("-")[0]

            matching = [
                item
                for item in available
                if item.split("-")[0] == family
            ]

            if matching:
                selected = matching[0]
            else:
                selected = available[0]

        source = (
            get_caption_source(
                subtitles.get(selected)
            )
            if selected in subtitles
            else None
        )

        if not source:
            source = get_caption_source(
                automatic.get(selected)
            )

        if not source:
            raise RuntimeError(
                "Caption source unavailable."
            )

        source_url = source.get("url")

        if not source_url:
            raise RuntimeError(
                "Caption URL unavailable."
            )

        source_ext = (
            source.get("ext")
            or "vtt"
        ).lower()

        source_file = (
            Path(temp_dir)
            / f"source.{source_ext}"
        )

        try:
            await run_job(
                download_http_file,
                source_url,
                source_file,
                MAX_CAPTION_DOWNLOAD_BYTES,
                {
                    "Referer": "https://www.youtube.com/",
                },
            )

        except requests.HTTPError as exc:
            status = (
                exc.response.status_code
                if exc.response is not None
                else None
            )

            if status == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "YouTube temporarily rate-limited "
                        "the caption request. Please wait "
                        "a little and try again."
                    ),
                )

            raise

        source_text = source_file.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        cues = parse_caption_source(
            source_text,
            source_ext,
        )

        if not cues:
            raise RuntimeError(
                "The caption track could not be parsed."
            )

        output = (
            Path(temp_dir)
            / f"caption.{caption_format}"
        )

        if caption_format == "srt":
            output.write_text(
                cues_to_srt(cues),
                encoding="utf-8",
            )

        elif caption_format == "vtt":
            output.write_text(
                cues_to_vtt(cues),
                encoding="utf-8",
            )

        elif caption_format == "ass":
            output.write_text(
                cues_to_ass(cues),
                encoding="utf-8",
            )

        elif caption_format == "lrc":
            output.write_text(
                cues_to_lrc(cues),
                encoding="utf-8",
            )

        title = clean_filename(
            info.get("title")
            or "captions"
        )

        filename = (
            f"{title}_{selected}."
            f"{caption_format}"
        )

        media_types = {
            "srt": "application/x-subrip",
            "vtt": "text/vtt",
            "ass": "text/plain",
            "lrc": "text/plain",
        }

        return FileResponse(
            output,
            media_type=media_types[
                caption_format
            ],
            filename=filename,
            content_disposition_type="attachment",
            background=BackgroundTask(
                remove_directory,
                temp_dir,
            ),
        )

    except HTTPException:
        remove_directory(temp_dir)
        raise

    except requests.HTTPError as exc:
        remove_directory(temp_dir)

        status = (
            exc.response.status_code
            if exc.response is not None
            else None
        )

        if status == 429:
            raise HTTPException(
                status_code=429,
                detail=(
                    "YouTube temporarily rate-limited "
                    "the caption request. Please try again later."
                ),
            )

        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve captions.",
        )

    except Exception as exc:
        remove_directory(temp_dir)

        raise HTTPException(
            status_code=500,
            detail=public_error(exc),
        )


# =========================================================
# FRONTEND
# =========================================================

@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend file not found.",
        )

    return FileResponse(
        index_file,
        media_type="text/html",
    )


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    import uvicorn

    print()
    print("=" * 60)
    print("  ClipForge")
    print("=" * 60)
    print()

    print(f"  Version: {APP_VERSION}")
    print()

    print("  Server:")
    print(f"  http://127.0.0.1:{PORT}/")
    print()

    print("  API:")
    print(f"  http://127.0.0.1:{PORT}/api/health")
    print()

    print("  Public protection:")
    print(
        f"  Max concurrent jobs: "
        f"{MAX_CONCURRENT_JOBS}"
    )

    print(
        f"  Max video duration: "
        f"{format_duration(MAX_VIDEO_DURATION)}"
    )

    print(
        f"  Max clip duration: "
        f"{format_duration(MAX_CLIP_DURATION)}"
    )

    print(
        f"  Rate limit: "
        f"{RATE_LIMIT_REQUESTS} requests/"
        f"{RATE_LIMIT_WINDOW}s"
    )

    print()

    print("=" * 60)
    print()

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
    )