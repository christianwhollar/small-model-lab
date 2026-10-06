"""Download the immutable, checksummed release artifact without pickle execution."""

import hashlib
import io
from pathlib import Path
import zipfile
import httpx

URL = "https://github.com/christianwhollar/small-model-lab/releases/download/v0.2.0/banking77-student-v0.2.0.zip"
SHA256 = "3c06fab31d7afb15d18bb48def6bb6ed7dd37d3ea3686213ace25ab4905536e8"


def download(destination):
    destination = Path(destination)
    if (destination / "manifest.json").exists():
        from .inference import IntentEngine

        IntentEngine(destination)
        return destination
    response = httpx.get(URL, follow_redirects=True, timeout=90)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != SHA256:
        raise ValueError("Release archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for info in archive.infolist():
            path = (destination / info.filename).resolve()
            if not path.is_relative_to(destination.resolve()) or info.file_size > 50_000_000:
                raise ValueError("Unsafe artifact path or size")
        archive.extractall(destination)
    return destination
