from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import gzip
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from scipy.spatial import cKDTree

from gfs_rain_confidence import (
    BASE_DIR,
    RunCycle,
    UTC,
    build_coordinate_grids,
    local_grib_path,
    load_prate_mmhr,
    normalize_longitudes,
)

MRMS_LATEST_URL = (
    "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/"
    "MRMS_ReflectivityAtLowestAltitude.latest.grib2.gz"
)
RAW_MRMS_DIR_NAME = "raw_mrms"
FEEDBACK_DIR_NAME = "feedback"
AI_STATE_DIR_NAME = "ai_state"
DEFAULT_REFLECTIVITY_THRESHOLD_DBZ = 20.0
DEFAULT_VALID_TIME_TOLERANCE_MINUTES = 20
DEFAULT_REQUEST_TIMEOUT = 120
DEFAULT_RETAIN_RAW_FILES = 72


@dataclass(frozen=True)
class PendingFeedbackTarget:
    run_cycle: RunCycle
    forecast_hour: int
    valid_time: dt.datetime
    feedback_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the latest MRMS reflectivity grid, align it to pending GFS AI forecast "
            "valid times, and write observed feedback grids for online model updates."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE_DIR / "gfs_rain_confidence_output",
        help="Root directory containing grib, feedback, and ai_state folders.",
    )
    parser.add_argument(
        "--mrms-url",
        default=MRMS_LATEST_URL,
        help="MRMS latest product URL to download.",
    )
    parser.add_argument(
        "--reflectivity-threshold-dbz",
        type=float,
        default=DEFAULT_REFLECTIVITY_THRESHOLD_DBZ,
        help="Reflectivity threshold used to convert MRMS reflectivity into a wet/dry label.",
    )
    parser.add_argument(
        "--valid-time-tolerance-minutes",
        type=int,
        default=DEFAULT_VALID_TIME_TOLERANCE_MINUTES,
        help="Maximum allowed difference between MRMS observation time and forecast valid time.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Seconds allowed for the MRMS download request.",
    )
    parser.add_argument(
        "--retain-raw-files",
        type=int,
        default=DEFAULT_RETAIN_RAW_FILES,
        help="How many raw MRMS downloads to keep on disk.",
    )
    parser.add_argument(
        "--overwrite-feedback",
        action="store_true",
        help="Rewrite an existing observed_rain_fXXX.npy file if it is already present.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_run_tag(run_tag: str) -> RunCycle | None:
    try:
        return RunCycle(dt.datetime.strptime(run_tag, "%Y%m%d_%HZ").replace(tzinfo=UTC))
    except ValueError:
        return None


def pending_feedback_targets(output_root: Path, overwrite_feedback: bool) -> list[PendingFeedbackTarget]:
    samples_root = output_root / AI_STATE_DIR_NAME / "samples"
    feedback_root = output_root / FEEDBACK_DIR_NAME
    if not samples_root.exists():
        return []

    targets: list[PendingFeedbackTarget] = []
    for run_dir in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        run_cycle = parse_run_tag(run_dir.name)
        if run_cycle is None:
            continue

        for snapshot_path in sorted(run_dir.glob("f*_samples.npz")):
            forecast_token = snapshot_path.stem.split("_")[0]
            try:
                forecast_hour = int(forecast_token[1:])
            except ValueError:
                continue

            feedback_path = feedback_root / run_cycle.tag / f"observed_rain_f{forecast_hour:03d}.npy"
            if feedback_path.exists() and not overwrite_feedback:
                continue

            targets.append(
                PendingFeedbackTarget(
                    run_cycle=run_cycle,
                    forecast_hour=forecast_hour,
                    valid_time=run_cycle.init_time + dt.timedelta(hours=forecast_hour),
                    feedback_path=feedback_path,
                )
            )

    return targets


def response_observation_time(response: requests.Response) -> dt.datetime:
    last_modified = response.headers.get("Last-Modified")
    if last_modified:
        observed_time = email.utils.parsedate_to_datetime(last_modified)
        if observed_time is not None:
            if observed_time.tzinfo is None:
                observed_time = observed_time.replace(tzinfo=UTC)
            return observed_time.astimezone(UTC)

    return dt.datetime.now(tz=UTC)


def download_latest_mrms(url: str, raw_root: Path, timeout: int) -> tuple[Path, dt.datetime]:
    raw_root.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        observed_time = response_observation_time(response)
        timestamp_token = observed_time.strftime("%Y%m%d_%H%M%S")
        gz_path = raw_root / f"mrms_reflectivity_{timestamp_token}.grib2.gz"
        grib_path = raw_root / f"mrms_reflectivity_{timestamp_token}.grib2"

        if not gz_path.exists():
            with gz_path.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1_048_576):
                    if chunk:
                        file_handle.write(chunk)

        if not grib_path.exists():
            with gzip.open(gz_path, "rb") as compressed_handle, grib_path.open("wb") as grib_handle:
                shutil.copyfileobj(compressed_handle, grib_handle)

    return grib_path, observed_time


def prune_old_raw_files(raw_root: Path, retain_count: int) -> None:
    if retain_count < 1 or not raw_root.exists():
        return

    raw_files = sorted(raw_root.glob("mrms_reflectivity_*"))
    stale_files = raw_files[:-retain_count]
    for stale_file in stale_files:
        stale_file.unlink(missing_ok=True)


def select_mrms_data_array(dataset: xr.Dataset) -> xr.DataArray:
    if len(dataset.data_vars) == 1:
        return next(iter(dataset.data_vars.values())).squeeze(drop=True)

    for variable in dataset.data_vars.values():
        name = str(variable.attrs.get("GRIB_name", ""))
        short_name = str(variable.attrs.get("GRIB_shortName", ""))
        if "reflect" in name.lower() or "ref" in short_name.lower():
            return variable.squeeze(drop=True)

    raise ValueError("Could not locate a reflectivity field in the MRMS dataset.")


def load_mrms_reflectivity(grib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        data_array = select_mrms_data_array(dataset)
        values = np.asarray(data_array.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Expected a 2D MRMS field in {grib_path.name}, got shape {values.shape}.")
        lats, lons = build_coordinate_grids(dataset)
    finally:
        dataset.close()

    values, lats, lons = normalize_longitudes(values, lats, lons)
    return values, lats, lons


def regrid_reflectivity_to_gfs(
    mrms_values: np.ndarray,
    mrms_lats: np.ndarray,
    mrms_lons: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    source_points = np.column_stack((mrms_lats.reshape(-1), mrms_lons.reshape(-1)))
    source_values = mrms_values.reshape(-1)
    valid_mask = np.isfinite(source_values)
    if not np.any(valid_mask):
        raise ValueError("MRMS reflectivity grid contains no finite values.")

    tree = cKDTree(source_points[valid_mask])
    target_points = np.column_stack((target_lats.reshape(-1), target_lons.reshape(-1)))
    _, indices = tree.query(target_points, workers=-1)
    mapped_values = source_values[valid_mask][indices]
    return mapped_values.reshape(target_lats.shape).astype(np.float32)


def write_feedback_grid(
    target: PendingFeedbackTarget,
    observed_label_grid: np.ndarray,
    observation_time: dt.datetime,
    reflectivity_threshold_dbz: float,
    mrms_grib_path: Path,
) -> None:
    target.feedback_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target.feedback_path, observed_label_grid.astype(np.float32))

    metadata_path = target.feedback_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "run_tag": target.run_cycle.tag,
                "forecast_hour": target.forecast_hour,
                "valid_time_utc": target.valid_time.isoformat(),
                "observation_time_utc": observation_time.isoformat(),
                "reflectivity_threshold_dbz": reflectivity_threshold_dbz,
                "source_grib_path": str(mrms_grib_path),
                "label_type": "wet_dry_from_reflectivity",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    output_root = args.output_root.resolve()
    raw_root = output_root / FEEDBACK_DIR_NAME / RAW_MRMS_DIR_NAME
    targets = pending_feedback_targets(output_root, overwrite_feedback=args.overwrite_feedback)
    if not targets:
        logging.info("No pending AI feedback targets found under %s", output_root / AI_STATE_DIR_NAME / "samples")
        return

    mrms_grib_path, observation_time = download_latest_mrms(args.mrms_url, raw_root, args.request_timeout)
    prune_old_raw_files(raw_root, args.retain_raw_files)
    mrms_values, mrms_lats, mrms_lons = load_mrms_reflectivity(mrms_grib_path)

    tolerance = dt.timedelta(minutes=max(0, args.valid_time_tolerance_minutes))
    matched_targets = [
        target
        for target in targets
        if abs(target.valid_time - observation_time) <= tolerance
    ]
    if not matched_targets:
        logging.info(
            "No pending forecast valid times matched MRMS observation time %s within %d minutes.",
            observation_time.isoformat(),
            args.valid_time_tolerance_minutes,
        )
        return

    written_count = 0
    for target in matched_targets:
        gfs_grib_path = local_grib_path(output_root, target.run_cycle, target.forecast_hour)
        if not gfs_grib_path.exists():
            logging.warning(
                "Skipping %s f%03d because the matching GFS GRIB file is missing: %s",
                target.run_cycle.tag,
                target.forecast_hour,
                gfs_grib_path,
            )
            continue

        _, gfs_lats, gfs_lons = load_prate_mmhr(gfs_grib_path)
        mapped_reflectivity = regrid_reflectivity_to_gfs(
            mrms_values,
            mrms_lats,
            mrms_lons,
            gfs_lats,
            gfs_lons,
        )
        observed_label_grid = np.where(mapped_reflectivity >= args.reflectivity_threshold_dbz, 1.0, 0.0)
        write_feedback_grid(
            target=target,
            observed_label_grid=observed_label_grid,
            observation_time=observation_time,
            reflectivity_threshold_dbz=args.reflectivity_threshold_dbz,
            mrms_grib_path=mrms_grib_path,
        )
        written_count += 1
        logging.info(
            "Saved MRMS feedback for %s f%03d using observation time %s",
            target.run_cycle.tag,
            target.forecast_hour,
            observation_time.isoformat(),
        )

    logging.info("Wrote %d MRMS feedback grids.", written_count)


if __name__ == "__main__":
    main()