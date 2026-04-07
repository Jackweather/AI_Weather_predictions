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

LIGHTNING_COLORS = [
    "#efefef",
    "#dfd7c8",
    "#f2d08a",
    "#f4b861",
    "#f39a3f",
    "#ef7f2d",
    "#e35f22",
    "#cc4519",
    "#a92b11",
    "#7a180a",
]

LIGHTNING_DOWNLOAD_FORECAST_HOURS = (0,) + tuple(range(6, MAX_DOWNLOAD_FORECAST_HOUR + 1, 6))

FIELD_SPECS = {
    "cape": {
        "backend_kwargs": (
            {"filter_by_keys": {"shortName": "cape", "typeOfLevel": "pressureFromGroundLayer"}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "cape", "typeOfLevel": "heightAboveGroundLayer"}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "cape"}, "indexpath": ""},
        ),
        "selectors": (
            ("cape",),
            ("CAPE",),
            ("GRIB_shortName", "cape"),
            ("GRIB_name", "Convective available potential energy"),
        ),
    },
    "lifted_index": {
        "backend_kwargs": (
            {"filter_by_keys": {"shortName": "4lftx", "typeOfLevel": "surface"}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "4lftx"}, "indexpath": ""},
        ),
        "selectors": (
            ("4lftx",),
            ("GRIB_shortName", "4LFTX"),
            ("GRIB_shortName", "4lftx"),
            ("GRIB_name", "Best (4-layer) lifted index"),
        ),
    },
    "dzdt": {
        "backend_kwargs": (
            {"filter_by_keys": {"shortName": "wz", "typeOfLevel": "isobaricInhPa", "level": 800}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "wz", "typeOfLevel": "isobaricInPa", "level": 80000}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "wz", "level": 800}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "wz"}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "dzdt", "typeOfLevel": "isobaricInhPa", "level": 800}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "dzdt", "typeOfLevel": "isobaricInPa", "level": 80000}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "dzdt", "level": 800}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "dzdt"}, "indexpath": ""},
        ),
        "selectors": (
            ("wz",),
            ("dzdt",),
            ("GRIB_shortName", "wz"),
            ("DZDT",),
            ("GRIB_shortName", "dzdt"),
            ("GRIB_name", "Vertical velocity (geometric)"),
            ("GRIB_name", "Geometric vertical velocity"),
        ),
    },
    "refc": {
        "backend_kwargs": (
            {"filter_by_keys": {"shortName": "refc", "typeOfLevel": "atmosphere"}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "refc", "typeOfLevel": "entireAtmosphere"}, "indexpath": ""},
            {"filter_by_keys": {"shortName": "refc"}, "indexpath": ""},
        ),
        "selectors": (
            ("refc",),
            ("REFC",),
            ("GRIB_shortName", "refc"),
            ("GRIB_name", "Composite reflectivity"),
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the newest complete GFS 0.25-degree run, collect prior runs, "
            "and build 0-10 lightning confidence maps from aligned convective ingredients."
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
        help="How many cycle folders to keep in lightning_grib and plots before deleting older ones.",
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
        raise ValueError(
            f"--max-plot-forecast-hour cannot exceed {MAX_DOWNLOAD_FORECAST_HOUR}."
        )
    if args.max_plot_forecast_hour % 6 != 0:
        raise ValueError("--max-plot-forecast-hour must be a multiple of 6.")
    if args.smooth_passes < 0:
        raise ValueError("--smooth-passes must be zero or greater.")


def build_plot_forecast_hours(max_forecast_hour: int) -> tuple[int, ...]:
    return tuple(range(0, max_forecast_hour + 1, 6))


def build_lightning_url(run_cycle: RunCycle, forecast_hour: int) -> str:
    return (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        f"?dir={run_cycle.nomads_directory}"
        f"&file={run_cycle.file_name(forecast_hour)}"
        "&var_CAPE=on"
        "&lev_255-0_mb_above_ground=on"
        "&var_4LFTX=on"
        "&lev_surface=on"
        "&var_DZDT=on"
        "&lev_800_mb=on"
        "&var_REFC=on"
        "&lev_entire_atmosphere=on"
    )


def local_grib_path(root: Path, run_cycle: RunCycle, forecast_hour: int) -> Path:
    file_name = f"lightning_{run_cycle.file_name(forecast_hour)}.grib2"
    return root / "lightning_grib" / run_cycle.tag / file_name


def missing_forecast_hours(
    root: Path,
    run_cycle: RunCycle,
    forecast_hours: Sequence[int],
    overwrite: bool,
) -> list[int]:
    if overwrite:
        return list(forecast_hours)

    return [
        forecast_hour
        for forecast_hour in forecast_hours
        if not local_grib_path(root, run_cycle, forecast_hour).exists()
    ]


def probe_file_available(client: NomadsClient, run_cycle: RunCycle, forecast_hour: int) -> bool:
    url = build_lightning_url(run_cycle, forecast_hour)
    try:
        response = client.request("GET", url, stream=True)
    except RuntimeError as exc:
        logging.debug("Probe failed for %s f%03d: %s", run_cycle.tag, forecast_hour, exc)
        return False

    with response:
        return response.status_code == 200 and "html" not in response.headers.get("Content-Type", "").lower()


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
            logging.info("Selected latest complete lightning run: %s", run_cycle.tag)
            return run_cycle
        logging.info("Lightning run %s is not complete yet, falling back one cycle", run_cycle.tag)
    raise RuntimeError("Could not find a complete recent GFS lightning run within the configured lookback window.")


def ensure_complete_history(run_cycles: Sequence[RunCycle], client: NomadsClient) -> None:
    for run_cycle in run_cycles:
        if not probe_file_available(client, run_cycle, MAX_DOWNLOAD_FORECAST_HOUR):
            raise RuntimeError(
                f"Run {run_cycle.tag} does not appear complete for lightning fields on NOMADS. "
                "Reduce --history-runs or wait for older runs to become fully available."
            )


def download_run(
    client: NomadsClient,
    root: Path,
    run_cycle: RunCycle,
    forecast_hours: Sequence[int],
    overwrite: bool,
    workers: int,
) -> None:
    hours_to_download = missing_forecast_hours(root, run_cycle, forecast_hours, overwrite)
    if not hours_to_download:
        logging.info("Skipping %s because all %d lightning files already exist", run_cycle.tag, len(forecast_hours))
        return

    logging.info(
        "Downloading lightning fields for %s (%d missing of %d files)",
        run_cycle.tag,
        len(hours_to_download),
        len(forecast_hours),
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = []
        for forecast_hour in hours_to_download:
            url = build_lightning_url(run_cycle, forecast_hour)
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


def open_field(grib_path: Path, field_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = FIELD_SPECS[field_name]
    dataset: xr.Dataset | None = None
    last_error: Exception | None = None

    for backend_kwargs in spec["backend_kwargs"]:
        try:
            dataset = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs)
            data_array = select_matching_field(dataset, spec["selectors"])
            values = np.asarray(data_array.values, dtype=np.float32)
            if values.ndim != 2:
                raise ValueError(f"Expected a 2D {field_name} field in {grib_path.name}, got shape {values.shape}.")
            lats, lons = build_coordinate_grids(dataset)
            values, lats, lons = normalize_longitudes(values, lats, lons)
            dataset.close()
            return values, lats, lons
        except Exception as exc:
            last_error = exc
            if dataset is not None:
                dataset.close()
                dataset = None

    raise RuntimeError(f"Unable to read {field_name} from {grib_path.name}: {last_error}") from last_error


def load_lightning_fields(grib_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    lats: np.ndarray | None = None
    lons: np.ndarray | None = None

    for field_name in FIELD_SPECS:
        values, current_lats, current_lons = open_field(grib_path, field_name)
        fields[field_name] = values
        if lats is None or lons is None:
            lats = current_lats
            lons = current_lons

    if lats is None or lons is None:
        raise RuntimeError(f"Could not read lightning fields from {grib_path.name}.")

    return fields, lats, lons


def calculate_member_lightning_score(fields: dict[str, np.ndarray]) -> np.ndarray:
    cape = np.maximum(fields["cape"], 0.0)
    lifted_index = fields["lifted_index"]
    dzdt = fields["dzdt"]
    reflectivity = np.maximum(fields["refc"], 0.0)

    cape_score = np.clip(cape / 3000.0, 0.0, 1.0)
    li_score = np.clip((2.0 - lifted_index) / 8.0, 0.0, 1.0)

    upward_motion = np.maximum(-dzdt, 0.0)
    downward_motion = np.maximum(dzdt, 0.0)
    motion_score = np.clip((upward_motion - 0.35 * downward_motion) / 0.08, 0.0, 1.0)

    reflectivity_score = np.clip((reflectivity - 12.0) / 38.0, 0.0, 1.0)
    convective_overlap = np.minimum(cape_score, np.maximum(li_score, reflectivity_score))

    raw_score = (
        0.33 * cape_score
        + 0.24 * li_score
        + 0.18 * motion_score
        + 0.25 * reflectivity_score
        + 0.22 * convective_overlap
    )

    suppress_dry_stable = np.where(
        (cape < 150.0) & (reflectivity < 8.0),
        0.0,
        1.0,
    )
    member_score = np.clip(10.0 * raw_score * suppress_dry_stable, 0.0, 10.0)
    return member_score.astype(np.float32)


def collect_aligned_members(
    root: Path,
    run_cycles: Sequence[RunCycle],
    current_forecast_hour: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
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
            logging.warning("Missing aligned lightning member for %s at f%03d", run_cycle.tag, aligned_hour)
            continue

        try:
            fields, current_lats, current_lons = load_lightning_fields(grib_path)
        except Exception as exc:
            logging.warning("Skipping invalid lightning member %s at f%03d: %s", run_cycle.tag, aligned_hour, exc)
            continue

        if lats is None or lons is None:
            lats = current_lats
            lons = current_lons

        member_arrays.append(calculate_member_lightning_score(fields))
        metadata.append((run_cycle.tag, aligned_hour))

    if not member_arrays or lats is None or lons is None:
        raise RuntimeError(f"No aligned lightning members were available for forecast hour f{current_forecast_hour:03d}.")

    return np.stack(member_arrays, axis=0), lats, lons, metadata


def calculate_lightning_confidence(member_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_score = member_scores.mean(axis=0)
    active_members = member_scores >= 4.0
    agreement = active_members.mean(axis=0)
    spread = member_scores.std(axis=0)
    spread_penalty = np.clip(spread / 4.0, 0.0, 1.0)

    confidence = mean_score * (0.45 + 0.55 * agreement) * (1.0 - 0.30 * spread_penalty)
    confidence = np.where(mean_score >= 1.5, confidence, 0.0)
    confidence = np.clip(confidence, 0.0, 10.0)

    return confidence.astype(np.float32), mean_score.astype(np.float32), agreement.astype(np.float32)


def lightning_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    boundaries = np.arange(0, 11, 1)
    cmap = ListedColormap(LIGHTNING_COLORS, name="lightning_confidence")
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm


def build_plot_title(run_cycle: RunCycle, forecast_hour: int) -> str:
    valid_local = valid_time(run_cycle, forecast_hour).astimezone(EASTERN)
    hour_string = valid_local.strftime("%I:%M %p").lstrip("0")
    day_of_week = valid_local.strftime("%A")
    return (
        f"Lightning Confidence Forecast F{forecast_hour:03d}\n"
        f"Valid {day_of_week}, {valid_local.strftime('%b %d, %Y')} at {hour_string} ET"
    )


def plot_lightning_map(
    save_path: Path,
    run_cycle: RunCycle,
    forecast_hour: int,
    lats: np.ndarray,
    lons: np.ndarray,
    confidence: np.ndarray,
    mean_score: np.ndarray,
    smooth_passes: int,
) -> None:
    smoothed_confidence = np.clip(smooth_field(confidence, smooth_passes), 0.0, 10.0)
    smoothed_mean_score = np.maximum(smooth_field(mean_score, smooth_passes), 0.0)
    projected_confidence = np.ma.masked_where(smoothed_mean_score < 1.5, smoothed_confidence)
    cmap, norm = lightning_cmap()

    figure = plt.figure(figsize=(15, 9))
    axis = plt.axes(projection=ccrs.PlateCarree())
    axis.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    axis.coastlines(linewidth=0.7)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    axis.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)

    filled = axis.contourf(
        lons,
        lats,
        projected_confidence,
        levels=np.arange(0, 11, 1),
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=ccrs.PlateCarree(),
    )

    axis.set_title(
        build_plot_title(run_cycle, forecast_hour),
        fontsize=16,
        color="#4b3b2e",
        pad=10,
        loc="left",
        fontweight="normal",
    )

    colorbar = plt.colorbar(filled, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label("Lightning confidence (0-10)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_lightning_products(
    root: Path,
    run_cycle: RunCycle,
    run_cycles: Sequence[RunCycle],
    plot_forecast_hours: Sequence[int],
    smooth_passes: int,
) -> None:
    plot_root = root / "plots" / run_cycle.tag
    for forecast_hour in plot_forecast_hours:
        members, lats, lons, metadata = collect_aligned_members(root, run_cycles, forecast_hour)
        confidence, mean_score, _ = calculate_lightning_confidence(members)
        save_path = plot_root / f"lightning_confidence_f{forecast_hour:03d}.png"
        plot_lightning_map(
            save_path=save_path,
            run_cycle=run_cycle,
            forecast_hour=forecast_hour,
            lats=lats,
            lons=lons,
            confidence=confidence,
            mean_score=mean_score,
            smooth_passes=smooth_passes,
        )
        logging.info("Saved %s using %d aligned lightning members", save_path, len(metadata))


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

    prune_old_run_directories(output_root / "lightning_grib", args.retain_runs)
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
                forecast_hours=LIGHTNING_DOWNLOAD_FORECAST_HOURS,
                overwrite=args.overwrite,
                workers=args.download_workers,
            )
        prune_old_run_directories(output_root / "lightning_grib", args.retain_runs)

    build_lightning_products(
        root=output_root,
        run_cycle=current_run,
        run_cycles=run_cycles,
        plot_forecast_hours=plot_forecast_hours,
        smooth_passes=max(0, args.smooth_passes),
    )
    prune_old_run_directories(output_root / "plots", args.retain_runs)

    logging.info("Finished building lightning confidence maps for %s", current_run.tag)


if __name__ == "__main__":
    main()
