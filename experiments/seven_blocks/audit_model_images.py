"""Hash actual native tool-output images against saved bridge observations.

Only image digests and relative local paths are exported. Native reasoning,
credentials and base64 payloads are never copied into the audit artifact.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path


def audit(run, rollouts, output):
    saved = {}
    for path in sorted(run.glob("bridge*/observations/*/frame_*.png")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        saved.setdefault(digest, []).append(str(path.relative_to(run)))
    if not saved:
        raise ValueError("No saved bridge observations to compare")
    images = []
    sources = []
    for rollout in rollouts:
        sources.append({"file": rollout.name,
                        "sha256": hashlib.sha256(rollout.read_bytes()).hexdigest()})
        with rollout.open() as stream:
            for line in stream:
                row = json.loads(line)
                payload = row.get("payload", {})
                if row.get("type") != "response_item":
                    continue
                if payload.get("type") not in ("function_call_output", "custom_tool_call_output"):
                    continue
                content = payload.get("output")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "input_image":
                        continue
                    url = item.get("image_url", "")
                    if not url.startswith("data:image/") or ";base64," not in url:
                        raise ValueError("Image output is not an auditable inline image")
                    data = base64.b64decode(url.split(",", 1)[1], validate=True)
                    digest = hashlib.sha256(data).hexdigest()
                    images.append({"timestamp": row.get("timestamp"),
                                   "call_id": payload.get("call_id"),
                                   "sha256": digest,
                                   "bytes": len(data),
                                   "matching_local_files": saved.get(digest, [])})
    passed = bool(images) and all(image["matching_local_files"] for image in images)
    report = {"passed": passed,
              "scope": "Actual input_image items in native tool outputs, including repeated deliveries",
              "images_delivered_to_model": len(images),
              "unique_image_hashes_delivered": len({image["sha256"] for image in images}),
              "all_match_saved_observations": passed,
              "native_sources": sources, "images": images}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, ensure_ascii=False))
    if not passed:
        raise SystemExit("Native image delivery audit failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.rollout, args.output)
