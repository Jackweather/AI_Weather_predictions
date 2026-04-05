from __future__ import annotations

import os
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, abort, jsonify, render_template, request, send_from_directory


app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent))
APP_ROOT = Path(__file__).resolve().parent


def default_data_root() -> Path:
    configured_root = os.environ.get("GFS_RAIN_CONFIDENCE_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    linux_data_root = Path("/var/data")
    if os.name != "nt" and linux_data_root.is_dir():
        return (linux_data_root / "gfs_rain_confidence_output").resolve()

    return (APP_ROOT / "gfs_rain_confidence_output").resolve()


DATA_ROOT = default_data_root()
PRODUCT_ROOTS = {
    "baseline": DATA_ROOT / "plots",
    "ai": DATA_ROOT / "plots_ai",
}
PRODUCT_LABELS = {
    "baseline": "Baseline Ensemble",
    "ai": "AI Adaptive",
}
LOGS_ROOT = APP_ROOT / "logs"
PNG_PATTERN = re.compile(r"f(\d{3})", re.IGNORECASE)
RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{2}z$", re.IGNORECASE)
EASTERN_TZ = ZoneInfo("America/New_York")


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
                    ["python", script_path],
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


def normalize_product(product: str | None) -> str:
    if product in PRODUCT_ROOTS:
        return str(product)
    return "baseline"


def product_root(product: str | None) -> Path:
    return PRODUCT_ROOTS[normalize_product(product)]


def list_products() -> list[dict[str, str]]:
    return [
        {"id": product_id, "label": label}
        for product_id, label in PRODUCT_LABELS.items()
    ]


def parse_run_id(run_id: str) -> datetime | None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        return None

    try:
        return datetime.strptime(run_id[:-1], "%Y%m%d_%H").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def format_run_label(run_id: str) -> str:
    run_time_utc = parse_run_id(run_id)
    if run_time_utc is None:
        return run_id

    run_time_eastern = run_time_utc.astimezone(EASTERN_TZ)
    return f"{run_time_utc.strftime('%HZ %b %d')} | {run_time_eastern.strftime('%a %I %p %Z')}"


def get_run_directory(run_id: str | None, product: str | None = None) -> Path | None:
    if not run_id:
        return None
    if parse_run_id(run_id) is None:
        return None

    run_dir = product_root(product) / run_id
    if not run_dir.is_dir():
        return None
    return run_dir


def list_runs(product: str | None = None) -> list[dict[str, str | int | bool]]:
    run_items: list[dict[str, str | int | bool]] = []
    plots_root = product_root(product)
    if not plots_root.is_dir():
        return run_items

    for run_dir in sorted(plots_root.iterdir(), key=lambda path: path.name, reverse=True):
        if not run_dir.is_dir() or parse_run_id(run_dir.name) is None:
            continue

        image_count = sum(1 for _ in run_dir.glob("*.png"))
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


def resolve_run_id(requested_run_id: str | None = None, product: str | None = None) -> str | None:
    run_items = list_runs(product)
    valid_run_ids = {str(item["id"]) for item in run_items}

    if requested_run_id in valid_run_ids:
        return requested_run_id

    if run_items:
        return str(run_items[0]["id"])

    return None


def get_run_label(run_id: str | None, product: str | None = None) -> str:
    if not run_id:
        return "No saved runs"

    for run in list_runs(product):
        if str(run["id"]) == run_id:
            return str(run["label"])

    return run_id


def list_images(
    run_id: str | None = None,
    product: str | None = None,
) -> tuple[list[dict[str, str | int]], str | None]:
    normalized_product = normalize_product(product)
    resolved_run_id = resolve_run_id(run_id, normalized_product)
    image_dir = get_run_directory(resolved_run_id, normalized_product) if resolved_run_id else None
    if image_dir is None:
        return [], resolved_run_id

    image_items: list[dict[str, str | int]] = []
    for image_path in sorted(image_dir.glob("*.png"), key=lambda path: path.name):
        match = PNG_PATTERN.search(image_path.stem)
        frame = int(match.group(1)) if match else -1
        stat = image_path.stat()
        image_items.append(
            {
                "filename": image_path.name,
                "frame": frame,
                "label": f"Forecast Hour F{frame:03d}" if frame >= 0 else image_path.stem,
                "url": (
                    f"/images/{resolved_run_id}/{image_path.name}"
                    f"?product={normalized_product}&v={int(stat.st_mtime)}"
                ),
            }
        )

    image_items.sort(key=lambda item: (int(item["frame"]), str(item["filename"])))
    return image_items, resolved_run_id


@app.route("/")
def index() -> str:
    selected_product = normalize_product(request.args.get("product"))
    runs = list_runs(selected_product)
    requested_run_id = request.args.get("run")
    images, selected_run = list_images(requested_run_id, selected_product)
    return render_template(
        "index.html",
        images=images,
        runs=runs,
        products=list_products(),
        selected_product=selected_product,
        selected_product_label=PRODUCT_LABELS[selected_product],
        selected_run=selected_run,
        selected_run_label=get_run_label(selected_run, selected_product),
        image_count=len(images),
        plots_root=str(product_root(selected_product)),
    )



@app.route("/api/runs")
def api_runs():
    selected_product = normalize_product(request.args.get("product"))
    runs = list_runs(selected_product)
    return jsonify(
        {
            "runs": runs,
            "products": list_products(),
            "selected_product": selected_product,
            "selected_run": resolve_run_id(request.args.get("run"), selected_product),
        }
    )


@app.route("/api/images")
def api_images():
    selected_product = normalize_product(request.args.get("product"))
    images, resolved_run_id = list_images(request.args.get("run"), selected_product)
    return jsonify(
        {
            "images": images,
            "product": selected_product,
            "run_id": resolved_run_id,
        }
    )


def start_background_script(script_name: str):
    scripts = [
        (str(APP_ROOT / script_name), str(APP_ROOT)),
    ]
    threading.Thread(
        target=lambda: run_scripts(scripts, 3, parallel=True, max_parallel=3),
        daemon=True,
    ).start()
    return jsonify(
        {
            "ok": True,
            "message": f"Started {script_name} in background.",
        }
    )


@app.route("/run-task1")
def run_task1():
    scripts = [
        ("/opt/render/project/src/gfs_rain_confidence.py", "/opt/render/project/src"),
        ("/opt/render/project/src/gfs_rain_confidence_ai.py", "/opt/render/project/src"),

        
        
    ]
    threading.Thread(
        target=lambda: run_scripts(scripts, 2, parallel=True, max_parallel=1),
        daemon=True,
    ).start()
    return "Task started in background! Check logs folder for output.", 200


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
