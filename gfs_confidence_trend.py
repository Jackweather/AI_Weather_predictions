from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from gfs_rain_confidence import BASE_DIR


app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent))
APP_ROOT = Path(__file__).resolve().parent


def default_data_root() -> Path:
    configured_root = os.environ.get("GFS_RAIN_CONFIDENCE_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    return (BASE_DIR / "gfs_rain_confidence_output").resolve()


DATA_ROOT = default_data_root()
PLOTS_ROOT = DATA_ROOT / "plots"
LOGS_ROOT = APP_ROOT / "logs"
PNG_PATTERN = re.compile(r"f(\d{3})", re.IGNORECASE)
RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{2}z$", re.IGNORECASE)
EASTERN_TZ = ZoneInfo("America/New_York")
PRODUCTS = {
    "confidence": {
        "label": "Rain Confidence",
        "filename_prefix": "rain_confidence_f",
        "empty_message": "Preparing GFS rain confidence PNGs for the viewer.",
    },
    "trend": {
        "label": "Run-to-Run Trend",
        "filename_prefix": "rain_confidence_trend_f",
        "empty_message": "Preparing run-to-run confidence trend PNGs for the viewer.",
    },
}


def run_scripts(
    scripts: list[tuple[str, str]],
    retries: int,
    parallel: bool = False,
    max_parallel: int = 3,
) -> None:
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)

    def run_single_script(script_path: str, working_dir: str) -> None:
        script_name = Path(script_path).stem
        log_path = LOGS_ROOT / f"{script_name}.log"
        attempts = max(1, retries)

        for attempt in range(1, attempts + 1):
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\n[{datetime.now(timezone.utc).isoformat()}] attempt {attempt}\n")
                completed = subprocess.run(
                    [sys.executable, script_path],
                    cwd=working_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

            if completed.returncode == 0:
                return

        raise RuntimeError(f"Script failed after {attempts} attempts: {script_path}")

    if parallel and len(scripts) > 1:
        workers = max(1, min(max_parallel, len(scripts)))
        threads: list[threading.Thread] = []
        for script_path, working_dir in scripts:
            thread = threading.Thread(target=run_single_script, args=(script_path, working_dir), daemon=True)
            thread.start()
            threads.append(thread)

            while sum(thread_item.is_alive() for thread_item in threads) >= workers:
                for thread_item in threads:
                    thread_item.join(timeout=0.1)

        for thread in threads:
            thread.join()
        return

    for script_path, working_dir in scripts:
        run_single_script(script_path, working_dir)


def parse_run_id(run_id: str) -> datetime | None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        return None

    try:
        return datetime.strptime(run_id[:-1], "%Y%m%d_%H").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def resolve_product(product_key: str | None) -> str:
    if product_key in PRODUCTS:
        return str(product_key)
    return "confidence"


def get_product_files(run_dir: Path, product_key: str):
    product = PRODUCTS[product_key]
    return run_dir.glob(f"{product['filename_prefix']}*.png")


def format_run_label(run_id: str) -> str:
    run_time_utc = parse_run_id(run_id)
    if run_time_utc is None:
        return run_id

    run_time_eastern = run_time_utc.astimezone(EASTERN_TZ)
    return f"{run_time_utc.strftime('%HZ %b %d')} | {run_time_eastern.strftime('%a %I %p %Z')}"


def get_run_directory(run_id: str | None) -> Path | None:
    if not run_id:
        return None
    if parse_run_id(run_id) is None:
        return None

    run_dir = PLOTS_ROOT / run_id
    if not run_dir.is_dir():
        return None
    return run_dir


def list_runs(product_key: str = "confidence") -> list[dict[str, str | int | bool]]:
    run_items: list[dict[str, str | int | bool]] = []
    if not PLOTS_ROOT.is_dir():
        return run_items

    for run_dir in sorted(PLOTS_ROOT.iterdir(), key=lambda path: path.name, reverse=True):
        if not run_dir.is_dir() or parse_run_id(run_dir.name) is None:
            continue

        image_count = sum(1 for _ in get_product_files(run_dir, product_key))
        if image_count == 0:
            continue

        run_items.append(
            {
                "id": run_dir.name,
                "label": format_run_label(run_dir.name),
                "image_count": image_count,
            }
        )

    for index, item in enumerate(run_items):
        item["is_latest"] = index == 0

    return run_items


def resolve_run_id(
    requested_run_id: str | None = None,
    product_key: str = "confidence",
) -> str | None:
    run_items = list_runs(product_key)
    valid_run_ids = {str(item["id"]) for item in run_items}

    if requested_run_id in valid_run_ids:
        return requested_run_id

    if run_items:
        return str(run_items[0]["id"])

    return None


def get_run_label(run_id: str | None, product_key: str = "confidence") -> str:
    if not run_id:
        return "No saved runs"

    for run in list_runs(product_key):
        if str(run["id"]) == run_id:
            return str(run["label"])

    return run_id


def list_images(
    run_id: str | None = None,
    product_key: str = "confidence",
) -> tuple[list[dict[str, str | int]], str | None]:
    resolved_run_id = resolve_run_id(run_id, product_key)
    image_dir = get_run_directory(resolved_run_id) if resolved_run_id else None
    if image_dir is None:
        return [], resolved_run_id

    image_items: list[dict[str, str | int]] = []
    for image_path in sorted(get_product_files(image_dir, product_key), key=lambda path: path.name):
        match = PNG_PATTERN.search(image_path.stem)
        frame = int(match.group(1)) if match else -1
        stat = image_path.stat()
        image_items.append(
            {
                "filename": image_path.name,
                "frame": frame,
                "label": f"Forecast Hour F{frame:03d}" if frame >= 0 else image_path.stem,
                "url": f"/images/{resolved_run_id}/{image_path.name}?v={int(stat.st_mtime)}",
            }
        )

    image_items.sort(key=lambda item: (int(item["frame"]), str(item["filename"])))
    return image_items, resolved_run_id


@app.route("/")
def index() -> str:
    selected_product = resolve_product(request.args.get("product"))
    runs = list_runs(selected_product)
    requested_run_id = request.args.get("run")
    images, selected_run = list_images(requested_run_id, selected_product)
    return render_template(
        "index.html",
        images=images,
        runs=runs,
        products=[{"id": product_id, **product} for product_id, product in PRODUCTS.items()],
        selected_product=selected_product,
        selected_product_label=PRODUCTS[selected_product]["label"],
        selected_product_empty_message=PRODUCTS[selected_product]["empty_message"],
        selected_run=selected_run,
        selected_run_label=get_run_label(selected_run, selected_product),
        image_count=len(images),
        plots_root=str(PLOTS_ROOT),
    )



@app.route("/api/runs")
def api_runs():
    selected_product = resolve_product(request.args.get("product"))
    runs = list_runs(selected_product)
    return jsonify(
        {
            "runs": runs,
            "selected_run": resolve_run_id(request.args.get("run"), selected_product),
            "selected_product": selected_product,
        }
    )


@app.route("/api/images")
def api_images():
    selected_product = resolve_product(request.args.get("product"))
    images, resolved_run_id = list_images(request.args.get("run"), selected_product)
    return jsonify({"images": images, "run_id": resolved_run_id, "product": selected_product})


@app.route("/run-task1")
def run_task1():
    scripts = [
        ("/opt/render/project/src/gfs_rain_confidence.py", "/opt/render/project/src"),
        ("/opt/render/project/src/gfs__confidence_trend.py", "/opt/render/project/src"),
        
        
    ]
    threading.Thread(
        target=lambda: run_scripts(scripts, 3, parallel=True, max_parallel=1),
        daemon=True,
    ).start()
    return "Task started in background! Check logs folder for output.", 200


@app.route("/images/<run_id>/<path:filename>")
def serve_image(run_id: str, filename: str):
    run_dir = get_run_directory(run_id)
    if run_dir is None:
        abort(404)

    image_path = (run_dir / filename).resolve()
    try:
        image_path.relative_to(run_dir.resolve())
    except ValueError:
        abort(404)

    if not image_path.is_file():
        abort(404)

    return send_from_directory(run_dir, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
