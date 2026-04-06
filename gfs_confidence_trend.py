from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

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
    DEFAULT_SMOOTH_PASSES,
    DOWNLOAD_FORECAST_HOURS,
    EASTERN,
    MAX_DOWNLOAD_FORECAST_HOUR,
    UTC,
    NomadsClient,
    build_plot_forecast_hours,
    build_run_sequence,
    calculate_confidence,
    collect_aligned_members,
    download_run,
    ensure_complete_history,
    prune_old_run_directories,
    resolve_latest_complete_cycle,
    setup_logging,
    smooth_field,
    valid_time,
)

matplotlib.use("Agg")

TREND_COLORS = [
    "#7f0000",
    "#c62828",
    "#ef5350",
    "#ffcdd2",
    "#f3f4f6",
    "#d8f3dc",
    "#74c69d",
    "#2d6a4f",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build GFS run-to-run rain confidence trend maps that show where confidence "
            "increased or decreased versus the prior run for the same valid time."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE_DIR / "gfs_rain_confidence_output",
        help="Directory containing downloaded GRIB files and generated plots.",
    )
    parser.add_argument(
        "--history-runs",
        type=int,
        default=7,
        help="How many previous 6-hour runs are used when calculating each confidence field.",
    )
    parser.add_argument(
        "--retain-runs",
        type=int,
        default=8,
        help="How many cycle folders to keep in the trend plot archive.",
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
        help="Concurrent downloads per run when the trend script needs missing GRIB files.",
    )
    parser.add_argument(
        "--request-min-interval",
        type=float,
        default=DEFAULT_REQUEST_MIN_INTERVAL,
        help="Minimum seconds between NOMADS requests while probing for complete runs.",
    )
    parser.add_argument(
        "--max-request-retries",
        type=int,
        default=DEFAULT_MAX_REQUEST_RETRIES,
        help="How many times to retry NOMADS requests after transient errors.",
    )
    parser.add_argument(
        "--rain-threshold-mmhr",
        type=float,
        default=0.10,
        help="Rain threshold in mm/hr used to suppress trend shading where both runs are dry.",
    )
    parser.add_argument(
        "--smooth-passes",
        type=int,
        default=DEFAULT_SMOOTH_PASSES,
        help="How many passes of spatial smoothing to apply before plotting.",
    )
    parser.add_argument(
        "--max-plot-forecast-hour",
        type=int,
        default=DEFAULT_MAX_PLOT_FORECAST_HOUR,
        help="Highest forecast hour to render to PNGs. Must be a multiple of 6.",
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
        help="Redownload files even if they already exist locally.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume the required GRIB files already exist locally and only build plots.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.history_runs < 1:
        raise ValueError("--history-runs must be at least 1 for run-to-run trend plots.")
    if args.retain_runs < 1:
        raise ValueError("--retain-runs must be at least 1.")
    if args.retain_runs < args.history_runs + 1:
        raise ValueError("--retain-runs must be at least --history-runs + 1.")
    if args.download_workers < 1:
        raise ValueError("--download-workers must be at least 1.")
    if args.request_min_interval < 0:
        raise ValueError("--request-min-interval must be zero or greater.")
    if args.max_request_retries < 1:
        raise ValueError("--max-request-retries must be at least 1.")
    if args.max_plot_forecast_hour < 6:
        raise ValueError("--max-plot-forecast-hour must be at least 6.")
    if args.max_plot_forecast_hour > MAX_DOWNLOAD_FORECAST_HOUR - 6:
        raise ValueError(
            f"--max-plot-forecast-hour cannot exceed {MAX_DOWNLOAD_FORECAST_HOUR - 6} for trend plots."
        )
    if args.max_plot_forecast_hour % 6 != 0:
        raise ValueError("--max-plot-forecast-hour must be a multiple of 6.")
    if args.smooth_passes < 0:
        raise ValueError("--smooth-passes must be zero or greater.")


def build_plot_title(run_cycle, forecast_hour: int) -> str:
    valid_local = valid_time(run_cycle, forecast_hour).astimezone(EASTERN)
    hour_string = valid_local.strftime("%I:%M %p").lstrip("0")
    day_of_week = valid_local.strftime("%A")
    return (
        f"Run-to-Run Confidence Trend F{forecast_hour:03d}\n"
        f"Valid {day_of_week}, {valid_local.strftime('%b %d, %Y')} at {hour_string} ET"
    )


def build_plot_subtitle(run_cycle, previous_run_cycle, forecast_hour: int, member_count: int) -> str:
    previous_forecast_hour = forecast_hour + 6
    return (
        f"Current {run_cycle.tag} F{forecast_hour:03d} vs Previous {previous_run_cycle.tag} "
        f"F{previous_forecast_hour:03d} | Aligned Members {member_count}"
    )


def trend_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    boundaries = np.array([-100, -75, -50, -25, 0, 25, 50, 75, 100], dtype=np.float32)
    cmap = ListedColormap(TREND_COLORS, name="confidence_trend")
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm


def calculate_confidence_change_percent(
    current_confidence: np.ndarray,
    previous_confidence: np.ndarray,
) -> np.ndarray:
    delta_points = current_confidence - previous_confidence
    delta_percent = np.clip((delta_points / 10.0) * 100.0, -100.0, 100.0)
    return delta_percent.astype(np.float32)


def collect_comparison_fields(
    root: Path,
    run_cycle,
    forecast_hour: int,
    history_runs: int,
    rain_threshold_mmhr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    current_sequence = build_run_sequence(run_cycle, history_runs)
    previous_run_cycle = run_cycle.shifted(hours=-6)
    previous_sequence = build_run_sequence(previous_run_cycle, history_runs)
    previous_forecast_hour = forecast_hour + 6

    current_members, lats, lons, current_metadata = collect_aligned_members(
        root,
        current_sequence,
        forecast_hour,
    )
    previous_members, previous_lats, previous_lons, previous_metadata = collect_aligned_members(
        root,
        previous_sequence,
        previous_forecast_hour,
    )

    if current_members.shape[1:] != previous_members.shape[1:]:
        raise RuntimeError(
            f"Mismatched confidence grid shapes for {run_cycle.tag} f{forecast_hour:03d}: "
            f"{current_members.shape[1:]} vs {previous_members.shape[1:]}."
        )

    if lats.shape != previous_lats.shape or lons.shape != previous_lons.shape:
        raise RuntimeError(
            f"Mismatched latitude/longitude grids for {run_cycle.tag} f{forecast_hour:03d}."
        )

    current_confidence, current_mean_rate, _, _ = calculate_confidence(current_members, rain_threshold_mmhr)
    previous_confidence, previous_mean_rate, _, _ = calculate_confidence(previous_members, rain_threshold_mmhr)
    member_count = min(len(current_metadata), len(previous_metadata))
    return lats, lons, current_confidence, previous_confidence, np.maximum(current_mean_rate, previous_mean_rate), member_count


def plot_trend_map(
    save_path: Path,
    run_cycle,
    previous_run_cycle,
    forecast_hour: int,
    lats: np.ndarray,
    lons: np.ndarray,
    confidence_change_percent: np.ndarray,
    mask_rate: np.ndarray,
    member_count: int,
    rain_threshold_mmhr: float,
    smooth_passes: int,
) -> None:
    smoothed_change = np.clip(smooth_field(confidence_change_percent, smooth_passes), -100.0, 100.0)
    smoothed_mask_rate = np.maximum(smooth_field(mask_rate, smooth_passes), 0.0)
    projected_change = np.ma.masked_where(smoothed_mask_rate < rain_threshold_mmhr, smoothed_change)
    cmap, norm = trend_cmap()

    figure = plt.figure(figsize=(15, 9))
    axis = plt.axes(projection=ccrs.PlateCarree())
    axis.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    axis.coastlines(linewidth=0.7)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    axis.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)

    filled = axis.contourf(
        lons,
        lats,
        projected_change,
        levels=np.array([-100, -75, -50, -25, 0, 25, 50, 75, 100], dtype=np.float32),
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    axis.set_title(
        build_plot_title(run_cycle, forecast_hour),
        fontsize=16,
        color="#3b4a5a",
        pad=10,
        loc="left",
        fontweight="normal",
    )
    axis.text(
        0.01,
        0.01,
        build_plot_subtitle(run_cycle, previous_run_cycle, forecast_hour, member_count),
        transform=axis.transAxes,
        fontsize=10,
        color="#304253",
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    colorbar = plt.colorbar(filled, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label("Confidence change vs prior run (% of 0-10 scale)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_trend_products(
    root: Path,
    run_cycle,
    plot_forecast_hours,
    history_runs: int,
    rain_threshold_mmhr: float,
    smooth_passes: int,
) -> None:
    plot_root = root / "plots" / run_cycle.tag
    previous_run_cycle = run_cycle.shifted(hours=-6)
    for forecast_hour in plot_forecast_hours:
        lats, lons, current_confidence, previous_confidence, mask_rate, member_count = collect_comparison_fields(
            root=root,
            run_cycle=run_cycle,
            forecast_hour=forecast_hour,
            history_runs=history_runs,
            rain_threshold_mmhr=rain_threshold_mmhr,
        )
        confidence_change_percent = calculate_confidence_change_percent(
            current_confidence,
            previous_confidence,
        )
        save_path = plot_root / f"rain_confidence_trend_f{forecast_hour:03d}.png"
        plot_trend_map(
            save_path=save_path,
            run_cycle=run_cycle,
            previous_run_cycle=previous_run_cycle,
            forecast_hour=forecast_hour,
            lats=lats,
            lons=lons,
            confidence_change_percent=confidence_change_percent,
            mask_rate=mask_rate,
            member_count=member_count,
            rain_threshold_mmhr=rain_threshold_mmhr,
            smooth_passes=smooth_passes,
        )
        logging.info("Saved %s", save_path)


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

    current_run = resolve_latest_complete_cycle(now_utc, args.cycle_lookback, client)
    required_run_cycles = build_run_sequence(current_run, args.history_runs + 1)
    ensure_complete_history(required_run_cycles, client)

    if not args.skip_download:
        for run_cycle in required_run_cycles:
            download_run(
                client=client,
                root=output_root,
                run_cycle=run_cycle,
                forecast_hours=DOWNLOAD_FORECAST_HOURS,
                overwrite=args.overwrite,
                workers=args.download_workers,
            )
        prune_old_run_directories(output_root / "grib", args.retain_runs)

    build_trend_products(
        root=output_root,
        run_cycle=current_run,
        plot_forecast_hours=plot_forecast_hours,
        history_runs=args.history_runs,
        rain_threshold_mmhr=args.rain_threshold_mmhr,
        smooth_passes=max(0, args.smooth_passes),
    )
    prune_old_run_directories(output_root / "plots", args.retain_runs)

    logging.info("Finished building trend maps for %s", current_run.tag)


if __name__ == "__main__":
    main()