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
import xarray as xr
from matplotlib import colormaps
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
    EASTERN,
    MAX_DOWNLOAD_FORECAST_HOUR,
    NomadsClient,
    RunCycle,
    UTC,
    build_coordinate_grids,
    build_run_sequence,
    download_file,
    normalize_longitudes,
    prune_old_run_directories,
    setup_logging,
    smooth_field,
    valid_time,
)

matplotlib.use("Agg")

TEMPERATURE_DOWNLOAD_FORECAST_HOURS = (0,) + tuple(range(6, MAX_DOWNLOAD_FORECAST_HOUR + 1, 6))
TEMPERATURE_MIN_F = -15.0
TEMPERATURE_MAX_F = 110.0
TEMPERATURE_INTERVAL_F = 5.0

FIELD_SPECS = {
    "t2m": {
        "backend_kwargs": (
            {"filter_by_keys": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "2t", "typeOfLevel": "heightAboveGround"}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "2t"}, "indexpath": ""},
            {"indexpath": ""},
        ),
        "selectors": (
            ("t2m",),
            ("tmp",),
            ("GRIB_shortName", "2t"),
            ("GRIB_name", "2 metre temperature"),
        ),
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the newest complete GFS 0.25-degree run, collect prior runs, "
            "and build aligned-run average 2 m temperature maps."
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
        help="How many previous 6-hour runs to compare against the current run.",
    )
    parser.add_argument(
        "--retain-runs",
        type=int,
        default=8,
        help="How many cycle folders to keep in temperature_grib and plots before deleting older ones.",
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
    if args.max_plot_forecast_hour < 0:
        raise ValueError("--max-plot-forecast-hour must be zero or greater.")
    if args.max_plot_forecast_hour > MAX_DOWNLOAD_FORECAST_HOUR:
        raise ValueError(f"--max-plot-forecast-hour cannot exceed {MAX_DOWNLOAD_FORECAST_HOUR}.")
    if args.max_plot_forecast_hour % 6 != 0:
        raise ValueError("--max-plot-forecast-hour must be a multiple of 6.")
    if args.smooth_passes < 0:
        raise ValueError("--smooth-passes must be zero or greater.")


def build_plot_forecast_hours(max_forecast_hour: int) -> tuple[int, ...]:
    return tuple(range(0, max_forecast_hour + 1, 6))


def build_temperature_url(run_cycle: RunCycle, forecast_hour: int) -> str:
    return (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        f"?dir={run_cycle.nomads_directory}"
        f"&file={run_cycle.file_name(forecast_hour)}"
        "&var_TMP=on"
        "&lev_2_m_above_ground=on"
    )


def local_grib_path(root: Path, run_cycle: RunCycle, forecast_hour: int) -> Path:
    file_name = f"temperature_{run_cycle.file_name(forecast_hour)}.grib2"
    return root / "temperature_grib" / run_cycle.tag / file_name


def missing_forecast_hours(root: Path, run_cycle: RunCycle, forecast_hours: Sequence[int], overwrite: bool) -> list[int]:
    if overwrite:
        return list(forecast_hours)
    return [hour for hour in forecast_hours if not local_grib_path(root, run_cycle, hour).exists()]


def probe_file_available(client: NomadsClient, run_cycle: RunCycle, forecast_hour: int) -> bool:
    url = build_temperature_url(run_cycle, forecast_hour)
    try:
        response = client.request("GET", url, stream=True)
    except RuntimeError as exc:
        logging.debug("Probe failed for %s f%03d: %s", run_cycle.tag, forecast_hour, exc)
        return False

    with response:
        content_type = response.headers.get("Content-Type", "").lower()
        if response.status_code != 200 or "html" in content_type:
            return False
        return True


def resolve_latest_complete_cycle(now_utc: dt.datetime, lookback_cycles: int, client: NomadsClient) -> RunCycle:
    candidate = RunCycle(now_utc.astimezone(UTC).replace(
        hour=(now_utc.astimezone(UTC).hour // 6) * 6,
        minute=0,
        second=0,
        microsecond=0,
    ))
    for offset in range(lookback_cycles + 1):
        run_cycle = candidate.shifted(hours=-6 * offset)
        if probe_file_available(client, run_cycle, MAX_DOWNLOAD_FORECAST_HOUR):
            logging.info("Selected latest complete temperature run: %s", run_cycle.tag)
            return run_cycle
        logging.info("Temperature run %s is not complete yet, falling back one cycle", run_cycle.tag)
    raise RuntimeError("Could not find a complete recent GFS temperature run within the configured lookback window.")


def ensure_complete_history(run_cycles: Sequence[RunCycle], client: NomadsClient) -> None:
    for run_cycle in run_cycles:
        if not probe_file_available(client, run_cycle, MAX_DOWNLOAD_FORECAST_HOUR):
            raise RuntimeError(
                f"Run {run_cycle.tag} does not appear complete for temperature fields on NOMADS. "
                "Reduce --history-runs or wait for older runs to become fully available."
            )


def download_run(client: NomadsClient, root: Path, run_cycle: RunCycle, forecast_hours: Sequence[int], overwrite: bool, workers: int) -> None:
    hours_to_download = missing_forecast_hours(root, run_cycle, forecast_hours, overwrite)
    if not hours_to_download:
        logging.info("Skipping %s because all %d temperature files already exist", run_cycle.tag, len(forecast_hours))
        return

    logging.info(
        "Downloading temperature fields for %s (%d missing of %d files)",
        run_cycle.tag,
        len(hours_to_download),
        len(forecast_hours),
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = []
        for forecast_hour in hours_to_download:
            url = build_temperature_url(run_cycle, forecast_hour)
            destination = local_grib_path(root, run_cycle, forecast_hour)
            futures.append(executor.submit(download_file, client, url, destination, overwrite))
        for future in as_completed(futures):
            future.result()


def select_matching_field(dataset: xr.Dataset, selectors: Sequence[tuple[str, ...]]) -> xr.DataArray:
    for selector in selectors:
        if len(selector) == 1 and selector[0] in dataset.data_vars:
            return dataset[selector[0]].squeeze(drop=True)

        if len(selector) == 2:
            attr_name, expected_value = selector
            for variable in dataset.data_vars.values():
                actual_value = variable.attrs.get(attr_name)
                if actual_value is None:
                    continue
                if str(actual_value).lower() == str(expected_value).lower():
                    return variable.squeeze(drop=True)

    if len(dataset.data_vars) == 1:
        return next(iter(dataset.data_vars.values())).squeeze(drop=True)

    raise ValueError(f"Could not locate expected field in dataset variables: {list(dataset.data_vars)}")


def load_temperature_f(grib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = FIELD_SPECS["t2m"]
    dataset: xr.Dataset | None = None
    last_error: Exception | None = None

    for backend_kwargs in spec["backend_kwargs"]:
        try:
            dataset = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs)
            data_array = select_matching_field(dataset, spec["selectors"])
            values_k = np.asarray(data_array.values, dtype=np.float32)
            if values_k.ndim != 2:
                raise ValueError(f"Expected a 2D temperature field in {grib_path.name}, got shape {values_k.shape}.")
            lats, lons = build_coordinate_grids(dataset)
            values_f = (values_k - 273.15) * 9.0 / 5.0 + 32.0
            values_f, lats, lons = normalize_longitudes(values_f.astype(np.float32), lats, lons)
            dataset.close()
            return values_f, lats, lons
        except Exception as exc:
            last_error = exc
            if dataset is not None:
                dataset.close()
                dataset = None

    raise RuntimeError(f"Unable to read 2 m temperature from {grib_path.name}: {last_error}") from last_error


def collect_aligned_members(root: Path, run_cycles: Sequence[RunCycle], current_forecast_hour: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    member_arrays: list[np.ndarray] = []
    metadata: list[tuple[str, int]] = []
    lats: np.ndarray | None = None
    lons: np.ndarray | None = None

    for offset, run_cycle in enumerate(run_cycles):
        aligned_hour = current_forecast_hour + (offset * 6)
        if aligned_hour > MAX_DOWNLOAD_FORECAST_HOUR:
            continue
        grib_path = local_grib_path(root, run_cycle, aligned_hour)
        if not grib_path.exists():
            logging.warning("Missing aligned temperature member for %s at f%03d", run_cycle.tag, aligned_hour)
            continue
        try:
            values_f, current_lats, current_lons = load_temperature_f(grib_path)
        except Exception as exc:
            logging.warning("Skipping invalid temperature member %s at f%03d: %s", run_cycle.tag, aligned_hour, exc)
            continue

        if lats is None or lons is None:
            lats = current_lats
            lons = current_lons

        member_arrays.append(values_f)
        metadata.append((run_cycle.tag, aligned_hour))

    if not member_arrays or lats is None or lons is None:
        raise RuntimeError(f"No aligned temperature members were available for forecast hour f{current_forecast_hour:03d}.")

    return np.stack(member_arrays, axis=0), lats, lons, metadata


def calculate_average_temperature(members_f: np.ndarray) -> np.ndarray:
    return members_f.mean(axis=0).astype(np.float32)


def temperature_cmap() -> tuple[ListedColormap, BoundaryNorm, np.ndarray]:
    boundaries = np.arange(TEMPERATURE_MIN_F, TEMPERATURE_MAX_F + TEMPERATURE_INTERVAL_F, TEMPERATURE_INTERVAL_F, dtype=np.float32)
    base_cmap = colormaps.get_cmap("turbo")
    colors = base_cmap(np.linspace(0.0, 1.0, len(boundaries) - 1))
    cmap = ListedColormap(colors, name="temperature_average")
    norm = BoundaryNorm(boundaries, cmap.N, clip=False)
    return cmap, norm, boundaries


def build_plot_title(run_cycle: RunCycle, forecast_hour: int) -> str:
    valid_local = valid_time(run_cycle, forecast_hour).astimezone(EASTERN)
    hour_string = valid_local.strftime("%I:%M %p").lstrip("0")
    day_of_week = valid_local.strftime("%A")
    return (
        f"Average Temperature Forecast F{forecast_hour:03d}\n"
        f"Valid {day_of_week}, {valid_local.strftime('%b %d, %Y')} at {hour_string} ET"
    )


def plot_temperature_map(save_path: Path, run_cycle: RunCycle, forecast_hour: int, lats: np.ndarray, lons: np.ndarray, average_temperature_f: np.ndarray, member_count: int, smooth_passes: int) -> None:
    smoothed_temperature_f = smooth_field(average_temperature_f, smooth_passes)
    cmap, norm, boundaries = temperature_cmap()

    figure = plt.figure(figsize=(15, 9))
    axis = plt.axes(projection=ccrs.PlateCarree())
    axis.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    axis.coastlines(linewidth=0.7)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    axis.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)

    filled = axis.contourf(
        lons,
        lats,
        smoothed_temperature_f,
        levels=boundaries,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    axis.set_title(
        build_plot_title(run_cycle, forecast_hour),
        fontsize=16,
        color="#33455f",
        pad=10,
        loc="left",
        fontweight="normal",
    )
    
    colorbar = plt.colorbar(filled, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label("Average 2 m temperature (F)")
    colorbar.set_ticks(boundaries)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_temperature_products(root: Path, run_cycle: RunCycle, run_cycles: Sequence[RunCycle], plot_forecast_hours: Sequence[int], smooth_passes: int) -> None:
    plot_root = root / "plots" / run_cycle.tag
    for forecast_hour in plot_forecast_hours:
        members_f, lats, lons, metadata = collect_aligned_members(root, run_cycles, forecast_hour)
        average_temperature_f = calculate_average_temperature(members_f)
        save_path = plot_root / f"temperature_average_f{forecast_hour:03d}.png"
        plot_temperature_map(
            save_path=save_path,
            run_cycle=run_cycle,
            forecast_hour=forecast_hour,
            lats=lats,
            lons=lons,
            average_temperature_f=average_temperature_f,
            member_count=len(metadata),
            smooth_passes=smooth_passes,
        )
        logging.info("Saved %s using %d aligned temperature members", save_path, len(metadata))


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

    prune_old_run_directories(output_root / "temperature_grib", args.retain_runs)
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
                forecast_hours=TEMPERATURE_DOWNLOAD_FORECAST_HOURS,
                overwrite=args.overwrite,
                workers=args.download_workers,
            )
        prune_old_run_directories(output_root / "temperature_grib", args.retain_runs)

    build_temperature_products(
        root=output_root,
        run_cycle=current_run,
        run_cycles=run_cycles,
        plot_forecast_hours=plot_forecast_hours,
        smooth_passes=max(0, args.smooth_passes),
    )
    prune_old_run_directories(output_root / "plots", args.retain_runs)

    logging.info("Finished building average temperature maps for %s", current_run.tag)


if __name__ == "__main__":
    main()
