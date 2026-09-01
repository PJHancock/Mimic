"""Fetch the unmodified, pinned Menagerie Panda assets (explicit setup only)."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree as ET

REVISION = "da76818e269b82289eba39808e2fb91d679d6994"
BASE = (
    f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}"
    "/franka_emika_panda"
)
DEFAULT_DESTINATION = Path(__file__).resolve().parents[1] / "models/franka_emika_panda/upstream"


def fetch(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    def download(name: str) -> tuple[str, str]:
        path = destination / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; use a fresh destination")
        with urlopen(f"{BASE}/{name}", timeout=60) as response:
            payload = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return name, hashlib.sha256(payload).hexdigest()

    records = dict(download(name) for name in ("panda.xml", "LICENSE", "README.md"))
    root = ET.parse(destination / "panda.xml").getroot()
    names = sorted({mesh.attrib["file"] for mesh in root.findall("./asset/mesh")})
    if any(Path(name).name != name for name in names):
        raise ValueError("Unexpected mesh path in upstream XML")
    with ThreadPoolExecutor(max_workers=8) as pool:
        records.update(pool.map(download, [f"assets/{name}" for name in names]))
    manifest = {
        "repository": "google-deepmind/mujoco_menagerie",
        "revision": REVISION,
        "sha256": records,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Fetched {len(records)} files at {REVISION} into {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    fetch(parser.parse_args().destination)
