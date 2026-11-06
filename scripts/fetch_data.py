"""Reuse a local official release, or download the release linked by the project page."""

from pathlib import Path
import hashlib, json, os, shutil, subprocess, zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://drive.google.com/uc?id=10tQylYHdKLVDVnmVHrygLB70neHnPdoF"
ARCHIVE_SHA256 = "bbba90bdb77bbab19e74cbe9f556ea0b9330ea9d125ad8271d431aa0e37a6609"
DATA_SHA256 = "7cfb7c1dcaa6f2a55a13c4849be9e17b4c960ce4015c500ac0ddfb9d77f4ba5b"
NAME = "parkour_motion_without_run_retargetted.npz"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


archive = ROOT / "data/hiking-in-the-wild_DataModel.zip"
target = ROOT / "data/official/data&model/parkour_motion_reference" / NAME
target.parent.mkdir(parents=True, exist_ok=True)
source = None
candidates = [target, Path.home() / "hiking-in-the-wild_Data&Model/data&model/parkour_motion_reference" / NAME]
for candidate in candidates:
    if candidate.is_file() and digest(candidate) == DATA_SHA256:
        source = str(candidate)
        if candidate != target:
            shutil.copy2(candidate, target)
            shutil.copy2(candidate.parent / "parkour_motion_without_run.yaml", target.parent)
        break
if source is None:
    if not archive.exists():
        local = Path.home() / "Downloads/hiking-in-the-wild_Data&Model.zip"
        if local.is_file() and digest(local) == ARCHIVE_SHA256:
            shutil.copy2(local, archive)
        else:
            subprocess.run(
                [
                    os.environ.get("UV_BIN", "uv"),
                    "tool",
                    "run",
                    "--from",
                    "gdown",
                    "gdown",
                    URL,
                    "-O",
                    str(archive),
                ],
                check=True,
            )
    if digest(archive) != ARCHIVE_SHA256:
        raise RuntimeError("Release archive changed; verify upstream before updating the recorded checksum")
    with zipfile.ZipFile(archive) as z:
        destination = (ROOT / "data/official").resolve()
        for member in z.infolist():
            if member.filename.startswith("__MACOSX/"):
                continue
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError("Invalid archive path")
            z.extract(member, destination)
    if digest(target) != DATA_SHA256:
        raise RuntimeError("Motion checksum mismatch")
    source = URL
manifest = {
    "source_page": "https://project-instinct.github.io/hiking-in-the-wild/",
    "download_url": URL,
    "resolved_source": source,
    "motion_path": str(target),
    "motion_sha256": digest(target),
    "archive_sha256": ARCHIVE_SHA256,
    "checksum_note": "Locally recorded release snapshot, not an upstream-signed checksum.",
}
(ROOT / "data/provenance.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
