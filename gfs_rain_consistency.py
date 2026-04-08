from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

from gfs_rain_confidence import (
    BASE_DIR,
    CONUS_EXTENT,
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_MAX_PLOT_FORECAST_HOUR,
    DEFAULT_MAX_REQUEST_RETRIES,
    DEFAULT_REQUEST_MIN_INTERVAL,
    MAX_DOWNLOAD_FORECAST_HOUR,
    NomadsClient,
    RunCycle,
    UTC,
    build_plot_forecast_hours,
    build_run_sequence,
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

CONSISTENCY_COLOR_ANCHORS = [
    "#eef8ff",
    "#77aed3",
    "#25ebc0",
    "#03f560",
    "#bbff00",
    "#ffed4c",
    "#faa302",
    "#ff009d",
    "#fc0202",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the newest complete GFS 0.25-degree run, collect the previous 7 runs, "
            "and build 0-8 rain consistency maps that show where precipitation has stayed in place "
            "from run to run."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE_DIR / "gfs_rain_confidence_output",
        help="Directory used for downloaded GRIB files and generated plots.",
    )
    parser.add_argument(
        "--history-runs",
        type=int,
        default=7,
        help="How many previous 6-hour runs to compare against the current run. Default 7 gives 8 total runs.",
    )
    parser.add_argument(
        "--retain-runs",
        type=int,
        default=8,
        help="How many cycle folders to keep in grib and plots before deleting older ones.",
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
        help="Rain threshold in mm/hr used to count a grid point as consistently wet.",
    )
    parser.add_argument(
        "--smooth-passes",
        type=int,
        default=0,
        help="Optional smoothing passes applied before plotting. Default 0 preserves exact run counts.",
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
    if args.rain_threshold_mmhr < 0:
        raise ValueError("--rain-threshold-mmhr must be zero or greater.")
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


def calculate_consistency(members_mmhr: np.ndarray, rain_threshold_mmhr: float) -> np.ndarray:
    wet_members = members_mmhr >= rain_threshold_mmhr
    return wet_members.sum(axis=0).astype(np.float32)


def build_plot_title(run_cycle: RunCycle, forecast_hour: int) -> str:
    valid_local = valid_time(run_cycle, forecast_hour).astimezone(dt.timezone(dt.timedelta(hours=-4)))
    hour_str_fmt = valid_local.strftime("%I:%M %p").lstrip("0")
    day_of_week = valid_local.strftime("%A")
    return (
        f"Rain Run-to-Run Consistency F{forecast_hour:03d}\n"
        f"Valid {day_of_week}, {valid_local.strftime('%b %d, %Y')} at {hour_str_fmt} ET"
    )


def consistency_cmap(member_count: int) -> tuple[ListedColormap, BoundaryNorm, np.ndarray]:
    gradient = LinearSegmentedColormap.from_list(
        "rain_consistency_gradient",
        CONSISTENCY_COLOR_ANCHORS,
        N=max(2, member_count + 1),
    )
    colors = gradient(np.linspace(0.0, 1.0, member_count + 1))
    cmap = ListedColormap(colors, name=f"rain_consistency_{member_count}")
    boundaries = np.arange(-0.5, member_count + 1.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    ticks = np.arange(0, member_count + 1, 1)
    return cmap, norm, ticks


def plot_consistency_map(
    save_path: Path,
    run_cycle: RunCycle,
    forecast_hour: int,
    lats: np.ndarray,
    lons: np.ndarray,
    consistency_count: np.ndarray,
    member_count: int,
    smooth_passes: int,
) -> None:
    plotted_consistency = np.clip(
        smooth_field(consistency_count, smooth_passes),
        0.0,
        float(member_count),
    )
    cmap, norm, ticks = consistency_cmap(member_count)
    levels = np.arange(-0.5, member_count + 1.5, 1.0)

    figure = plt.figure(figsize=(15, 9))
    axis = plt.axes(projection=ccrs.PlateCarree())
    axis.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    axis.coastlines(linewidth=0.7)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    axis.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)

    filled = axis.contourf(
        lons,
        lats,
        plotted_consistency,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="neither",
        transform=ccrs.PlateCarree(),
    )

    axis.set_title(
        build_plot_title(run_cycle, forecast_hour),
        fontsize=16,
        color="#2f4858",
        pad=10,
        loc="left",
        fontweight="normal",
    )

    colorbar = plt.colorbar(filled, ax=axis, shrink=0.82, pad=0.02, ticks=ticks)
    colorbar.set_label(f"Consistent wet runs out of {member_count}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_consistency_products(
    root: Path,
    run_cycle: RunCycle,
    run_cycles: Sequence[RunCycle],
    plot_forecast_hours: Sequence[int],
    rain_threshold_mmhr: float,
    smooth_passes: int,
) -> None:
    plot_root = root / "plots" / run_cycle.tag
    for forecast_hour in plot_forecast_hours:
        members, lats, lons, metadata = collect_aligned_members(root, run_cycles, forecast_hour)
        consistency_count = calculate_consistency(members, rain_threshold_mmhr)
        save_path = plot_root / f"rain_consistency_f{forecast_hour:03d}.png"
        plot_consistency_map(
            save_path=save_path,
            run_cycle=run_cycle,
            forecast_hour=forecast_hour,
            lats=lats,
            lons=lons,
            consistency_count=consistency_count,
            member_count=len(metadata),
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

    prune_old_run_directories(output_root / "grib", args.retain_runs)
    prune_old_run_directories(output_root / "plots", args.retain_runs)

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

    build_consistency_products(
        root=output_root,
        run_cycle=current_run,
        run_cycles=run_cycles,
        plot_forecast_hours=plot_forecast_hours,
        rain_threshold_mmhr=args.rain_threshold_mmhr,
        smooth_passes=args.smooth_passes,
    )
    prune_old_run_directories(output_root / "plots", args.retain_runs)

    logging.info("Finished building rain consistency maps for %s", current_run.tag)


if __name__ == "__main__":
    main()
