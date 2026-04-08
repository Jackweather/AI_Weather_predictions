from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from gfs_rain_confidence import (
    BASE_DIR,
    CONUS_EXTENT,
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_MAX_PLOT_FORECAST_HOUR,
    DEFAULT_MAX_REQUEST_RETRIES,
    DEFAULT_REQUEST_MIN_INTERVAL,
    EASTERN,
    MAX_DOWNLOAD_FORECAST_HOUR,
    NomadsClient,
    RunCycle,
    UTC,
    build_plot_forecast_hours,
    build_run_sequence,
    calculate_confidence,
    collect_aligned_members,
    download_run,
    ensure_complete_history,
    load_prate_mmhr,
    local_grib_path,
    prune_old_run_directories,
    resolve_latest_complete_cycle,
    setup_logging,
    smooth_field,
    valid_time,
)

matplotlib.use("Agg")

VERIFICATION_COLORS = [
    "#7f0000",
    "#b30000",
    "#e34a33",
    "#fc8d59",
    "#fdbb84",
    "#fee8c8",
    "#d9f0d3",
    "#78c679",
    "#31a354",
    "#006837",
]
DEFAULT_ARCHIVE_RUN_RETENTION = 80
DEFAULT_VERIFICATION_FORECAST_HOUR = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive GFS rain confidence fields, verify them after the valid time arrives, "
            "and build verification maps and summary files that show where the model did well or poorly."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE_DIR / "gfs_rain_confidence_output",
        help="Directory used for downloaded GRIB files and generated products.",
    )
    parser.add_argument(
        "--history-runs",
        type=int,
        default=7,
        help="How many previous 6-hour runs to compare against the current run.",
    )
    parser.add_argument(
        "--retain-runs",
        type=int,
        default=8,
        help="How many cycle folders to keep in grib and plots before deleting older ones.",
    )
    parser.add_argument(
        "--retain-archive-runs",
        type=int,
        default=DEFAULT_ARCHIVE_RUN_RETENTION,
        help="How many forecast-run folders to keep in verification archive, score, and plot directories.",
    )
    parser.add_argument(
        "--cycle-lookback",
        type=int,
        default=8,
        help="How many candidate cycles to search when looking for the latest complete run.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help="Concurrent downloads per run.",
    )
    parser.add_argument(
        "--request-min-interval",
        type=float,
        default=DEFAULT_REQUEST_MIN_INTERVAL,
        help="Minimum seconds between outbound NOMADS requests across all worker threads.",
    )
    parser.add_argument(
        "--max-request-retries",
        type=int,
        default=DEFAULT_MAX_REQUEST_RETRIES,
        help="How many times to retry NOMADS requests after rate limiting or transient server errors.",
    )
    parser.add_argument(
        "--rain-threshold-mmhr",
        type=float,
        default=0.10,
        help="Rain threshold in mm/hr used for both forecast confidence and verification.",
    )
    parser.add_argument(
        "--verification-forecast-hour",
        type=int,
        default=DEFAULT_VERIFICATION_FORECAST_HOUR,
        help="Short-range forecast hour used as the verification proxy. Must be a multiple of 6.",
    )
    parser.add_argument(
        "--smooth-passes",
        type=int,
        default=2,
        help="How many passes of smoothing to apply to verification score maps.",
    )
    parser.add_argument(
        "--max-plot-forecast-hour",
        type=int,
        default=DEFAULT_MAX_PLOT_FORECAST_HOUR,
        help="Highest forecast hour to archive and later verify. Must be a multiple of 6.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="Seconds allowed for each NOMADS request.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild saved snapshots and verification results even if they already exist locally.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume required GRIB files already exist locally and only archive or verify.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.download_workers < 1:
        raise ValueError("--download-workers must be at least 1.")
    if args.request_min_interval < 0:
        raise ValueError("--request-min-interval must be zero or greater.")
    if args.max_request_retries < 1:
        raise ValueError("--max-request-retries must be at least 1.")
    if args.history_runs < 0:
        raise ValueError("--history-runs must be zero or greater.")
    if args.retain_runs < 1:
        raise ValueError("--retain-runs must be at least 1.")
    if args.retain_runs < args.history_runs + 1:
        raise ValueError("--retain-runs must be at least --history-runs + 1.")
    if args.retain_archive_runs < 1:
        raise ValueError("--retain-archive-runs must be at least 1.")
    if args.rain_threshold_mmhr < 0:
        raise ValueError("--rain-threshold-mmhr must be zero or greater.")
    if args.verification_forecast_hour < 6 or args.verification_forecast_hour % 6 != 0:
        raise ValueError("--verification-forecast-hour must be a multiple of 6 and at least 6.")
    if args.max_plot_forecast_hour < 6:
        raise ValueError("--max-plot-forecast-hour must be at least 6.")
    if args.max_plot_forecast_hour > MAX_DOWNLOAD_FORECAST_HOUR:
        raise ValueError(
            f"--max-plot-forecast-hour cannot exceed {MAX_DOWNLOAD_FORECAST_HOUR}."
        )
    if args.max_plot_forecast_hour % 6 != 0:
        raise ValueError("--max-plot-forecast-hour must be a multiple of 6.")
    if args.smooth_passes < 0:
        raise ValueError("--smooth-passes must be zero or greater.")


def archive_directory(root: Path, run_tag: str) -> Path:
    return root / "verification_archive" / run_tag


def verification_plot_directory(root: Path, run_tag: str) -> Path:
    return root / "verification_plots" / run_tag


def verification_score_directory(root: Path, run_tag: str) -> Path:
    return root / "verification_scores" / run_tag


def snapshot_npz_path(root: Path, run_cycle: RunCycle, forecast_hour: int) -> Path:
    return archive_directory(root, run_cycle.tag) / f"rain_snapshot_f{forecast_hour:03d}.npz"


def snapshot_metadata_path(root: Path, run_cycle: RunCycle, forecast_hour: int) -> Path:
    return archive_directory(root, run_cycle.tag) / f"rain_snapshot_f{forecast_hour:03d}.json"


def verification_plot_path(root: Path, run_tag: str, forecast_hour: int) -> Path:
    return verification_plot_directory(root, run_tag) / f"rain_verification_f{forecast_hour:03d}.png"


def verification_json_path(root: Path, run_tag: str, forecast_hour: int) -> Path:
    return verification_score_directory(root, run_tag) / f"rain_verification_f{forecast_hour:03d}.json"


def save_forecast_snapshot(
    root: Path,
    run_cycle: RunCycle,
    forecast_hour: int,
    confidence: np.ndarray,
    mean_rate: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    member_count: int,
    rain_threshold_mmhr: float,
) -> None:
    archive_directory(root, run_cycle.tag).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snapshot_npz_path(root, run_cycle, forecast_hour),
        confidence=confidence.astype(np.float32),
        mean_rate=mean_rate.astype(np.float32),
        lats=lats.astype(np.float32),
        lons=lons.astype(np.float32),
    )
    metadata = {
        "forecast_run_tag": run_cycle.tag,
        "forecast_hour": forecast_hour,
        "valid_time_utc": valid_time(run_cycle, forecast_hour).isoformat(),
        "member_count": member_count,
        "rain_threshold_mmhr": float(rain_threshold_mmhr),
    }
    snapshot_metadata_path(root, run_cycle, forecast_hour).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def archive_current_forecasts(
    root: Path,
    run_cycle: RunCycle,
    run_cycles: Sequence[RunCycle],
    plot_forecast_hours: Sequence[int],
    rain_threshold_mmhr: float,
    overwrite: bool,
) -> None:
    for forecast_hour in plot_forecast_hours:
        snapshot_path = snapshot_npz_path(root, run_cycle, forecast_hour)
        if snapshot_path.exists() and not overwrite:
            continue
        members, lats, lons, metadata = collect_aligned_members(root, run_cycles, forecast_hour)
        confidence, mean_rate, _, _ = calculate_confidence(members, rain_threshold_mmhr)
        save_forecast_snapshot(
            root=root,
            run_cycle=run_cycle,
            forecast_hour=forecast_hour,
            confidence=confidence,
            mean_rate=mean_rate,
            lats=lats,
            lons=lons,
            member_count=len(metadata),
            rain_threshold_mmhr=rain_threshold_mmhr,
        )
        logging.info("Archived forecast snapshot %s f%03d", run_cycle.tag, forecast_hour)


def load_snapshot(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as payload:
        return {name: payload[name] for name in payload.files}


def iter_due_snapshots(
    root: Path,
    current_run: RunCycle,
    verification_forecast_hour: int,
) -> list[dict[str, Any]]:
    archive_root = root / "verification_archive"
    if not archive_root.exists():
        return []

    ready_cutoff = current_run.init_time + dt.timedelta(hours=verification_forecast_hour)
    due_snapshots: list[dict[str, Any]] = []
    for metadata_path in sorted(archive_root.glob("*/rain_snapshot_f*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        valid_time_utc = dt.datetime.fromisoformat(metadata["valid_time_utc"])
        if valid_time_utc.tzinfo is None:
            valid_time_utc = valid_time_utc.replace(tzinfo=UTC)
        if valid_time_utc > ready_cutoff:
            continue
        due_snapshots.append(
            {
                **metadata,
                "valid_time": valid_time_utc.astimezone(UTC),
                "npz_path": metadata_path.with_suffix(".npz"),
                "verification_json_path": verification_json_path(
                    root,
                    str(metadata["forecast_run_tag"]),
                    int(metadata["forecast_hour"]),
                ),
                "verification_plot_path": verification_plot_path(
                    root,
                    str(metadata["forecast_run_tag"]),
                    int(metadata["forecast_hour"]),
                ),
            }
        )
    return due_snapshots


def verification_target(snapshot: dict[str, Any], verification_forecast_hour: int) -> tuple[RunCycle, int]:
    valid_time_utc = snapshot["valid_time"]
    verifying_cycle = RunCycle(valid_time_utc - dt.timedelta(hours=verification_forecast_hour))
    return verifying_cycle, verification_forecast_hour


def ensure_verification_inputs(
    root: Path,
    client: NomadsClient,
    due_snapshots: Sequence[dict[str, Any]],
    overwrite: bool,
    workers: int,
    verification_forecast_hour: int,
) -> None:
    required_runs: dict[str, RunCycle] = {}
    for snapshot in due_snapshots:
        verifying_cycle, _ = verification_target(snapshot, verification_forecast_hour)
        required_runs[verifying_cycle.tag] = verifying_cycle

    for verifying_cycle in required_runs.values():
        verification_path = local_grib_path(root, verifying_cycle, verification_forecast_hour)
        if verification_path.exists() and not overwrite:
            continue
        try:
            download_run(
                client=client,
                root=root,
                run_cycle=verifying_cycle,
                forecast_hours=(verification_forecast_hour,),
                overwrite=overwrite,
                workers=workers,
            )
        except Exception as exc:
            logging.warning(
                "Could not download verification input for %s f%03d: %s",
                verifying_cycle.tag,
                verification_forecast_hour,
                exc,
            )


def calculate_verification_score(
    confidence: np.ndarray,
    observed_rate: np.ndarray,
    rain_threshold_mmhr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forecast_probability = np.clip(confidence / 10.0, 0.0, 1.0)
    observed_wet = (observed_rate >= rain_threshold_mmhr).astype(np.float32)
    verification_score = 10.0 * (1.0 - np.abs(forecast_probability - observed_wet))
    return (
        np.clip(verification_score, 0.0, 10.0).astype(np.float32),
        forecast_probability.astype(np.float32),
        observed_wet,
    )


def verification_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    boundaries = np.arange(0, 11, 1)
    cmap = ListedColormap(VERIFICATION_COLORS, name="rain_verification")
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm


def build_verification_title(forecast_run_tag: str, forecast_hour: int, valid_time_utc: dt.datetime) -> str:
    valid_local = valid_time_utc.astimezone(EASTERN)
    hour_str_fmt = valid_local.strftime("%I:%M %p").lstrip("0")
    day_of_week = valid_local.strftime("%A")
    return (
        f"Rain Verification Score F{forecast_hour:03d} | Forecast Run {forecast_run_tag}\n"
        f"Verified {day_of_week}, {valid_local.strftime('%b %d, %Y')} at {hour_str_fmt} ET"
    )


def plot_verification_map(
    save_path: Path,
    forecast_run_tag: str,
    forecast_hour: int,
    valid_time_utc: dt.datetime,
    lats: np.ndarray,
    lons: np.ndarray,
    verification_score: np.ndarray,
    smooth_passes: int,
) -> None:
    plotted_score = np.clip(smooth_field(verification_score, smooth_passes), 0.0, 10.0)
    cmap, norm = verification_cmap()

    figure = plt.figure(figsize=(15, 9))
    axis = plt.axes(projection=ccrs.PlateCarree())
    axis.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    axis.coastlines(linewidth=0.7)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    axis.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)

    filled = axis.contourf(
        lons,
        lats,
        plotted_score,
        levels=np.arange(0, 11, 1),
        cmap=cmap,
        norm=norm,
        extend="neither",
        transform=ccrs.PlateCarree(),
    )

    axis.set_title(
        build_verification_title(forecast_run_tag, forecast_hour, valid_time_utc),
        fontsize=16,
        color="#334e68",
        pad=10,
        loc="left",
        fontweight="normal",
    )

    colorbar = plt.colorbar(filled, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label("Verification score (0-10)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def verify_snapshot(
    root: Path,
    snapshot: dict[str, Any],
    rain_threshold_mmhr: float,
    verification_forecast_hour: int,
    smooth_passes: int,
) -> None:
    snapshot_arrays = load_snapshot(Path(snapshot["npz_path"]))
    confidence = np.asarray(snapshot_arrays["confidence"], dtype=np.float32)
    lats = np.asarray(snapshot_arrays["lats"], dtype=np.float32)
    lons = np.asarray(snapshot_arrays["lons"], dtype=np.float32)

    verifying_cycle, verifying_hour = verification_target(snapshot, verification_forecast_hour)
    verifying_path = local_grib_path(root, verifying_cycle, verifying_hour)
    observed_rate, observed_lats, observed_lons = load_prate_mmhr(verifying_path)

    if confidence.shape != observed_rate.shape:
        raise ValueError(
            f"Grid shape mismatch for {snapshot['forecast_run_tag']} f{int(snapshot['forecast_hour']):03d}: "
            f"archived {confidence.shape} vs verification {observed_rate.shape}."
        )
    if observed_lats.shape != lats.shape or observed_lons.shape != lons.shape:
        raise ValueError(
            f"Coordinate mismatch for {snapshot['forecast_run_tag']} f{int(snapshot['forecast_hour']):03d}."
        )

    verification_score, forecast_probability, observed_wet = calculate_verification_score(
        confidence,
        observed_rate,
        rain_threshold_mmhr,
    )

    forecast_wet = forecast_probability >= 0.5
    observed_wet_bool = observed_wet >= 0.5
    hits = float(np.logical_and(forecast_wet, observed_wet_bool).sum())
    misses = float(np.logical_and(~forecast_wet, observed_wet_bool).sum())
    false_alarms = float(np.logical_and(forecast_wet, ~observed_wet_bool).sum())

    result = {
        "forecast_run_tag": str(snapshot["forecast_run_tag"]),
        "forecast_hour": int(snapshot["forecast_hour"]),
        "valid_time_utc": snapshot["valid_time"].isoformat(),
        "verification_run_tag": verifying_cycle.tag,
        "verification_forecast_hour": verifying_hour,
        "member_count": int(snapshot["member_count"]),
        "rain_threshold_mmhr": float(rain_threshold_mmhr),
        "mean_score": float(np.mean(verification_score)),
        "mean_forecast_probability": float(np.mean(forecast_probability)),
        "observed_wet_fraction": float(np.mean(observed_wet)),
        "hit_rate": safe_ratio(hits, hits + misses),
        "false_alarm_rate": safe_ratio(false_alarms, hits + false_alarms),
        "critical_success_index": safe_ratio(hits, hits + misses + false_alarms),
    }

    verification_json = Path(snapshot["verification_json_path"])
    verification_json.parent.mkdir(parents=True, exist_ok=True)
    verification_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    plot_verification_map(
        save_path=Path(snapshot["verification_plot_path"]),
        forecast_run_tag=str(snapshot["forecast_run_tag"]),
        forecast_hour=int(snapshot["forecast_hour"]),
        valid_time_utc=snapshot["valid_time"],
        lats=lats,
        lons=lons,
        verification_score=verification_score,
        smooth_passes=smooth_passes,
    )
    logging.info(
        "Verified %s f%03d against %s f%03d",
        snapshot["forecast_run_tag"],
        int(snapshot["forecast_hour"]),
        verifying_cycle.tag,
        verifying_hour,
    )


def rebuild_summary_csv(root: Path) -> None:
    score_root = root / "verification_scores"
    summary_path = root / "verification_scores" / "rain_verification_summary.csv"
    score_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "forecast_run_tag",
        "forecast_hour",
        "valid_time_utc",
        "verification_run_tag",
        "verification_forecast_hour",
        "member_count",
        "rain_threshold_mmhr",
        "mean_score",
        "mean_forecast_probability",
        "observed_wet_fraction",
        "hit_rate",
        "false_alarm_rate",
        "critical_success_index",
    ]
    rows: list[dict[str, Any]] = []
    for json_path in sorted(score_root.glob("*/rain_verification_f*.json")):
        rows.append(json.loads(json_path.read_text(encoding="utf-8")))

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> None:
    args = parse_args()
    validate_args(args)
    setup_logging(args.log_level)
    now_utc = dt.datetime.now(tz=UTC)
    plot_forecast_hours = build_plot_forecast_hours(args.max_plot_forecast_hour)
    output_root = args.output_root.resolve()
    client = NomadsClient(
        timeout=args.request_timeout,
        min_interval_seconds=args.request_min_interval,
        max_retries=args.max_request_retries,
    )

    prune_old_run_directories(output_root / "grib", args.retain_runs)
    prune_old_run_directories(output_root / "plots", args.retain_runs)
    prune_old_run_directories(output_root / "verification_archive", args.retain_archive_runs)
    prune_old_run_directories(output_root / "verification_plots", args.retain_archive_runs)
    prune_old_run_directories(output_root / "verification_scores", args.retain_archive_runs)

    current_run = resolve_latest_complete_cycle(now_utc, args.cycle_lookback, client)
    run_cycles = build_run_sequence(current_run, args.history_runs)
    ensure_complete_history(run_cycles, client)

    logging.info("Output root: %s", output_root)
    logging.info("Current run: %s", current_run.tag)
    logging.info("Comparison runs: %s", ", ".join(run_cycle.tag for run_cycle in run_cycles[1:]))

    if not args.skip_download:
        for run_cycle in run_cycles:
            download_run(
                client=client,
                root=output_root,
                run_cycle=run_cycle,
                forecast_hours=tuple(range(6, MAX_DOWNLOAD_FORECAST_HOUR + 1, 6)),
                overwrite=args.overwrite,
                workers=args.download_workers,
            )
        prune_old_run_directories(output_root / "grib", args.retain_runs)

    archive_current_forecasts(
        root=output_root,
        run_cycle=current_run,
        run_cycles=run_cycles,
        plot_forecast_hours=plot_forecast_hours,
        rain_threshold_mmhr=args.rain_threshold_mmhr,
        overwrite=args.overwrite,
    )

    due_snapshots = iter_due_snapshots(
        root=output_root,
        current_run=current_run,
        verification_forecast_hour=args.verification_forecast_hour,
    )
    if due_snapshots:
        if not args.skip_download:
            ensure_verification_inputs(
                root=output_root,
                client=client,
                due_snapshots=due_snapshots,
                overwrite=args.overwrite,
                workers=args.download_workers,
                verification_forecast_hour=args.verification_forecast_hour,
            )

        for snapshot in due_snapshots:
            verification_json = Path(snapshot["verification_json_path"])
            if verification_json.exists() and not args.overwrite:
                continue
            try:
                verify_snapshot(
                    root=output_root,
                    snapshot=snapshot,
                    rain_threshold_mmhr=args.rain_threshold_mmhr,
                    verification_forecast_hour=args.verification_forecast_hour,
                    smooth_passes=args.smooth_passes,
                )
            except Exception as exc:
                logging.warning(
                    "Skipping verification for %s f%03d: %s",
                    snapshot["forecast_run_tag"],
                    int(snapshot["forecast_hour"]),
                    exc,
                )

    rebuild_summary_csv(output_root)
    prune_old_run_directories(output_root / "verification_archive", args.retain_archive_runs)
    prune_old_run_directories(output_root / "verification_plots", args.retain_archive_runs)
    prune_old_run_directories(output_root / "verification_scores", args.retain_archive_runs)

    logging.info("Finished archiving and verifying rain forecasts for %s", current_run.tag)


if __name__ == "__main__":
    main()