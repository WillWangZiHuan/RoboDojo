"""Add timestamped, verbatim PUBLIC messages to existing accelerated videos.

Only native assistant messages matching exported agent_message text are admitted.
Reasoning records, private metadata, and raw tool payloads are never exported.
"""

import argparse
import bisect
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
WIDTH, HEIGHT, FPS = 1920, 1080, 30
TZ = ZoneInfo("Asia/Shanghai")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def public_messages(run, native_root):
    allowed = []
    for folder in ("model", "model_resume_001"):
        path = run / folder / "agent_events.jsonl"
        if path.exists():
            for row in load_jsonl(path):
                item = row.get("item", {})
                if item.get("type") == "agent_message":
                    allowed.append(item["text"])
    remaining = Counter(t.strip() for t in allowed)
    messages = []
    audit = json.loads((run / "model_visual_input_audit.json").read_text())
    for source in audit["native_sources"]:
        paths = list(native_root.rglob(source["file"]))
        if len(paths) != 1:
            raise ValueError(f"Expected one native source: {source['file']}")
        for row in load_jsonl(paths[0]):
            item = row.get("payload", {})
            if not (row.get("type") == "response_item"
                    and item.get("type") == "message"
                    and item.get("role") == "assistant"):
                continue
            text = "".join(c.get("text", "") for c in item.get("content", [])
                           if c.get("type") == "output_text")
            if remaining[text.strip()] <= 0:
                continue
            # Preserve the exact native public text, including Markdown/newlines.
            remaining[text.strip()] -= 1
            messages.append({"timestamp": row["timestamp"],
                "wall_unix": datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp(),
                "text": text, "phase": item.get("phase"),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    assert not any(remaining.values()), "Some public messages lack a matched timestamp"
    messages.sort(key=lambda m: m["wall_unix"])
    assert len(messages) == len(allowed)
    for i, m in enumerate(messages):
        m["number"] = i + 1
    return messages


def wrapped(text, font, max_width):
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for char in paragraph:
            if line and font.getlength(line + char) > max_width:
                lines.append(line)
                line = char
            else:
                line += char
        lines.append(line)
    assert "".join(lines) == text.replace("\n", ""), "Text was altered"
    return lines


class Layout:
    def __init__(self, name, speed, messages, requests, responses):
        self.name, self.speed = name, speed
        self.messages, self.requests, self.responses = messages, requests, responses
        self.times = [m["wall_unix"] for m in messages]
        self.req_times = [m["wall_unix"] for m in requests]
        self.fonts = {size: ImageFont.truetype(FONT, size) for size in (22, 24, 26, 28, 30, 32, 36, 44)}
        self.cached_key = self.cached = None

    def panel(self, wall, mode, message_index=None):
        mi = bisect.bisect_right(self.times, wall) - 1 if message_index is None else message_index
        ri = bisect.bisect_right(self.req_times, wall) - 1
        req = self.requests[ri] if ri >= 0 else None
        reply = self.responses.get(req["index"]) if req else None
        answered = reply is not None and reply["wall_unix"] <= wall
        key = (mi, ri, answered, mode)
        if key == self.cached_key:
            return self.cached.copy(), mi
        canvas = Image.new("RGB", (WIDTH, HEIGHT), "#fafbfc")
        d = ImageDraw.Draw(canvas)
        d.text((28, 42), f"{self.name}  /  七色方块自主展示", font=self.fonts[44], fill="#1b2936")
        d.rounded_rectangle((1010, 148, 1892, 715), 18, fill="#ffffff", outline="#dfe5eb", width=2)
        d.text((1034, 168), "模型公开说明 · 原文", font=self.fonts[30], fill="#227768")
        if mi >= 0:
            msg = self.messages[mi]
            local = datetime.fromtimestamp(msg["wall_unix"], TZ).strftime("%H:%M:%S")
            d.text((1034, 214), f"消息 {mi + 1}/{len(self.messages)}  ·  北京时间 {local}",
                   font=self.fonts[22], fill="#647585")
            for size in (32, 30, 28, 26, 24, 22):
                lines = wrapped(msg["text"], self.fonts[size], 830)
                if len(lines) * (size + 13) <= 420:
                    break
            else:
                raise ValueError("Public message does not fit without truncation")
            for j, line in enumerate(lines):
                d.text((1034, 267 + j * (size + 13)), line, font=self.fonts[size], fill="#243340")
        else:
            d.text((1034, 277), "此时尚无模型公开说明。", font=self.fonts[32], fill="#647585")
            d.text((1034, 336), "环境准备 / 只读检查", font=self.fonts[26], fill="#647585")
        d.rounded_rectangle((1010, 735, 1892, 992), 18, fill="#f0f4f8")
        d.text((1034, 752), "实际工具请求与反馈", font=self.fonts[26], fill="#32648e")
        if req and mode != "pre":
            q = req["request"]
            params = {k: v for k, v in q.items() if k not in ("description", "reason")}
            raw = f"#{req['index']}  " + json.dumps(params, ensure_ascii=False, separators=(",", ":"))
            lines = wrapped(raw, self.fonts[24], 830)
            assert len(lines) <= 4
            for j, line in enumerate(lines):
                d.text((1034, 798 + j * 31), line, font=self.fonts[24], fill="#243340")
            result, color = "等待该请求返回", "#94620c"
            if answered:
                r = reply["response"]
                result = f"状态：{r.get('status', 'running')}"
                if "executed" in r:
                    result += f"  executed={str(r['executed']).lower()}"
                if "planning_status" in r:
                    result += f"  规划：{r['planning_status']}"
                color = "#b83630" if r.get("executed") is False else "#247354"
            d.text((1034, 949), result, font=self.fonts[22], fill=color)
        else:
            d.text((1034, 807), "尚无物理动作记录。", font=self.fonts[26], fill="#647585")
        self.cached_key, self.cached = key, canvas
        return canvas.copy(), mi

    def frame(self, physical, wall, mode="main", message_index=None):
        canvas, mi = self.panel(wall, mode, message_index)
        d = ImageDraw.Draw(canvas)
        if physical is not None:
            canvas.paste(physical.resize((960, 792), Image.Resampling.LANCZOS), (28, 180))
        else:
            d.rectangle((28, 180, 988, 972), fill="#f0f4f8")
            d.text((106, 480), "录像开始前：环境尚未就绪", font=self.fonts[36], fill="#647585")
            d.text((106, 548), "仅展示当时的公开消息", font=self.fonts[30], fill="#647585")
        stamp = datetime.fromtimestamp(wall, TZ).strftime("%Y-%m-%d %H:%M:%S")
        d.text((1335, 70), stamp + " 北京时间", font=self.fonts[22], fill="#647585")
        return np.asarray(canvas), mi


def render(run, native_root, preview_only=False):
    sim = run / "simulation"
    manifest = json.loads((sim / "video_manifest.json").read_text())
    speed = manifest["view_speed"]
    source = sim / f"全过程_观看版_{speed}倍速.mp4"
    source_info = next(x for x in manifest["videos"] if x["path"] == source.name)
    assert sha(source) == source_info["sha256"], "Source changed"
    frames = source_info["frames"]
    timeline_first = next(load_jsonl(sim / "videos/frames.jsonl"))["wall_unix"]
    end = timeline_first + manifest["wall_seconds"]
    messages = public_messages(run, native_root)
    requests = json.loads((run / "tool_requests.json").read_text())
    bridge = "bridge_agent" if run.name.startswith("astra") else "bridge"
    responses = {r["index"]: r for r in load_jsonl(run / bridge / "rpc.jsonl") if r["event"] == "response"}
    name = {"astra": "GPT-6 Astra", "sol": "GPT-6 Sol", "terra": "GPT-6.1 Terra"}[run.name.split("_")[0]]
    layout = Layout(name, speed, messages, requests, responses)
    outdir = sim / "公开过程同步版"
    outdir.mkdir(exist_ok=True)
    pre = [i for i, m in enumerate(messages) if m["wall_unix"] < timeline_first]
    post = [i for i, m in enumerate(messages) if m["wall_unix"] > end]
    if preview_only:
        image = Image.open(sorted(sim.glob("frame_*_cam_head.png"))[-1]).convert("RGB")
        # Existing video includes a 48px timing strip, so pad preview identically.
        padded = Image.new("RGB", (640, 528), "#101723")
        padded.paste(image, (0, 0))
        for tag, idx in [("正文", 14 if len(messages) > 14 else 0), ("最长消息", max(range(len(messages)), key=lambda i: len(messages[i]["text"]))), ("结尾", len(messages)-1)]:
            array, _ = layout.frame(padded, messages[idx]["wall_unix"], "post" if idx in post else "main", idx)
            Image.fromarray(array).save(outdir / f"版式预览_{tag}.png")
        print(json.dumps({"run":run.name, "public_messages":len(messages), "pre":len(pre), "post":len(post), "preview":str(outdir)}, ensure_ascii=False), flush=True)
        return
    destination = outdir / f"全过程_{speed}倍速_公开过程同步.mp4"
    if destination.exists():
        raise FileExistsError(destination)
    reader = imageio_ffmpeg.read_frames(str(source))
    meta = next(reader)
    assert meta["fps"] == FPS and meta["size"] == (640, 528)
    writer = imageio_ffmpeg.write_frames(str(destination), (WIDTH, HEIGHT), fps=FPS,
        codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1,
        output_params=["-crf", "20", "-preset", "veryfast", "-threads", "2", "-movflags", "+faststart"])
    writer.send(None)
    counts = Counter()
    written = 0
    segments = []

    def emit(array, index, repeats=1):
        nonlocal written
        for _ in range(repeats):
            writer.send(array)
        written += repeats
        if index >= 0:
            counts[index] += repeats

    for idx in pre:
        start_frame = written
        arr, mi = layout.frame(None, messages[idx]["wall_unix"], "pre", idx)
        emit(arr, mi, 5 * FPS)
        segments.append({"kind":"pre_recording_public_card", "message":idx+1, "start_frame":start_frame, "frames":5*FPS})
    main_start = written
    last_image = None
    for i, raw in enumerate(reader):
        wall = min(end, timeline_first + i / FPS * speed)
        last_image = Image.frombytes("RGB", meta["size"], raw)
        array, mi = layout.frame(last_image, wall)
        emit(array, mi)
        if i in (frames // 3, frames * 2 // 3, frames - 1):
            Image.fromarray(array).save(outdir / f"实际合成帧_{i:05d}.png")
        if i % 900 == 0:
            print(json.dumps({"run":run.name, "frame":i, "total":frames}), flush=True)
    assert i + 1 == frames
    segments.append({"kind":"original_accelerated_video", "start_frame":main_start, "frames":frames,
                     "speed":speed, "source_wall_start":timeline_first, "source_wall_end":end})
    for idx in post:
        start_frame = written
        arr, mi = layout.frame(last_image, messages[idx]["wall_unix"], "post", idx)
        emit(arr, mi, 10 * FPS)
        segments.append({"kind":"post_recording_final_frame_hold", "message":idx+1, "start_frame":start_frame, "frames":10*FPS})
    writer.close()
    reader.close()
    assert set(counts) == set(range(len(messages))), "A public message never appears on screen"
    check = imageio_ffmpeg.read_frames(str(destination))
    checked_meta = next(check)
    decoded = sum(1 for _ in check)
    assert decoded == written and checked_meta["size"] == (WIDTH, HEIGHT)
    assert sha(source) == source_info["sha256"], "Original video altered"
    report = {"passed":True, "source_video":str(source), "source_sha256":source_info["sha256"],
        "output_video":str(destination), "output_sha256":sha(destination), "frames":written,
        "fps":FPS, "duration_seconds":written/FPS, "size":[WIDTH,HEIGHT], "speed":speed,
        "full_decode_passed":True, "source_frames_preserved":frames, "public_messages":len(messages),
        "all_public_messages_shown":True, "segments":segments,
        "timestamp_semantics":"Whole public-message native record timestamp; not token-by-token onset. Never present a future response before its RPC timestamp.",
        "text_policy":"Native public text preserved verbatim, including punctuation and Markdown. Only matching exported agent_message text is allowed. No hidden reasoning.",
        "pre_card_seconds":5, "post_card_seconds":10,
        "messages":[dict(m, rendered_frames=counts[i]) for i,m in enumerate(messages)]}
    (outdir / "同步时间线与验收.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "messages"},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--native-root", type=Path, default=Path("/Users/will/.codex/sessions/2026/10/28"))
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    render(args.run.resolve(), args.native_root, args.preview_only)
