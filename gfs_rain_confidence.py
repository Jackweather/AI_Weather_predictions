from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import logging
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
import requests
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")

BASE_DIR = Path("/var/data")
BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
UTC = dt.timezone.utc
EASTERN = ZoneInfo("America/New_York")
MAX_DOWNLOAD_FORECAST_HOUR = 384
DEFAULT_MAX_PLOT_FORECAST_HOUR = 336
DEFAULT_SMOOTH_PASSES = 3
DEFAULT_DOWNLOAD_WORKERS = 2
DEFAULT_REQUEST_MIN_INTERVAL = 0.5
DEFAULT_MAX_REQUEST_RETRIES = 6
DOWNLOAD_FORECAST_HOURS = tuple(range(6, MAX_DOWNLOAD_FORECAST_HOUR + 1, 6))
CONUS_EXTENT = (-127.0, -66.0, 20.0, 54.0)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
LIGHT_CONFIDENCE_COLORS = [
    "#e0f3ff",  # very light blue
    "#9bd4ff",  # light blue
    "#4fa3ff",  # blue

    "#3bd16f",  # green
    "#7fff00",  # bright green-yellow

    "#ffe600",  # yellow
    "#ffb300",  # orange

    "#ff4d2d",  # red
    "#cc1f1a",  # dark red

    "#9b00ff",  # purple (extreme)
]

CFGRIB_BACKEND_KWARGS = (
    {
        "filter_by_keys": {"shortName": "prate", "typeOfLevel": "surface", "stepType": "instant"},
        "indexpath": "",
    },
    {
        "filter_by_keys": {"shortName": "prate", "typeOfLevel": "surface", "stepType": "avg"},
        "indexpath": "",
    },
    {"filter_by_keys": {"shortName": "prate", "typeOfLevel": "surface"}, "indexpath": ""},
    {"filter_by_keys": {"typeOfLevel": "surface", "stepType": "instant"}, "indexpath": ""},
    {"filter_by_keys": {"typeOfLevel": "surface", "stepType": "avg"}, "indexpath": ""},
    {"filter_by_keys": {"typeOfLevel": "surface"}, "indexpath": ""},
    {"indexpath": ""},
)


@dataclass(frozen=True)
class RunCycle:
    init_time: dt.datetime

    def __post_init__(self) -> None:
        if self.init_time.tzinfo is None:
            object.__setattr__(self, "init_time", self.init_time.replace(tzinfo=UTC))
        else:
            object.__setattr__(self, "init_time", self.init_time.astimezone(UTC))

    @property
    def cycle_hour(self) -> int:
        return self.init_time.hour

    @property
    def date_token(self) -> str:
        return self.init_time.strftime("%Y%m%d")

    @property
    def tag(self) -> str:
        return self.init_time.strftime("%Y%m%d_%HZ")

    @property
    def nomads_directory(self) -> str:
        return f"/gfs.{self.date_token}/{self.cycle_hour:02d}/atmos"

    def file_name(self, forecast_hour: int) -> str:
        return f"gfs.t{self.cycle_hour:02d}z.pgrb2.0p25.f{forecast_hour:03d}"

    def shifted(self, hours: int) -> "RunCycle":
        return RunCycle(self.init_time + dt.timedelta(hours=hours))


class NomadsClient:
    def __init__(
        self,
        timeout: int,
        min_interval_seconds: float,
        max_retries: int,
    ) -> None:
        self.timeout = timeout
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.max_retries = max(1, int(max_retries))
        self._last_request_started = 0.0
        self._rate_lock = threading.Lock()
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "gfs-rain-confidence/1.0 (+https://nomads.ncep.noaa.gov/)",
                    "Accept": "*/*",
                }
            )
            self._thread_local.session = session
        return session

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return

        with self._rate_lock:
            now = time.monotonic()
            wait_seconds = self.min_interval_seconds - (now - self._last_request_started)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.monotonic()
            self._last_request_started = now

    def _retry_delay(self, attempt: int, response: requests.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), self.min_interval_seconds)
                except ValueError:
                    retry_after_dt = email.utils.parsedate_to_datetime(retry_after)
                    if retry_after_dt is not None:
                        if retry_after_dt.tzinfo is None:
                            retry_after_dt = retry_after_dt.replace(tzinfo=UTC)
                        delay = (retry_after_dt.astimezone(UTC) - dt.datetime.now(tz=UTC)).total_seconds()
                        return max(delay, self.min_interval_seconds)

        base_delay = max(self.min_interval_seconds, 1.0)
        backoff = min(90.0, base_delay * (2 ** (attempt - 1)))
        jitter = random.uniform(0.0, max(0.25, self.min_interval_seconds))
        return backoff + jitter

    def request(self, method: str, url: str, *, stream: bool = False) -> requests.Response:
        last_error: Exception | None = None
        response: requests.Response | None = None

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self._get_session().request(
                    method=method,
                    url=url,
                    stream=stream,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                delay = self._retry_delay(attempt, None)
                logging.warning(
                    "Request error for %s on attempt %d/%d: %s. Retrying in %.1fs",
                    url,
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue

            if response.status_code not in RETRYABLE_STATUS_CODES:
                return response

            if attempt == self.max_retries:
                return response

            delay = self._retry_delay(attempt, response)
            logging.warning(
                "Received HTTP %s for %s on attempt %d/%d. Retrying in %.1fs",
                response.status_code,
                url,
                attempt,
                self.max_retries,
                delay,
            )
            response.close()
            time.sleep(delay)

        raise RuntimeError(f"Request failed for {url}: {last_error}") from last_error


def response_contains_grib_payload(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "")
    disposition = response.headers.get("Content-Disposition", "")
    if response.status_code != 200:
        return False
    if "html" in content_type.lower() and ".grib2" not in disposition.lower():
        return False
    return True


def assert_grib_payload(response: requests.Response, file_label: str) -> None:
    if response_contains_grib_payload(response):
        return

    content_type = response.headers.get("Content-Type", "")
    raise RuntimeError(
        f"Unexpected response while fetching {file_label}: HTTP {response.status_code}, "
        f"content type {content_type or 'unknown'}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the newest complete GFS 0.25-degree run, collect prior runs, "
            "and build 0-10 rain confidence maps from aligned precipitation forecasts."
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
        help="Rain threshold in mm/hr used to mark a grid point as wet.",
    )
    parser.add_argument(
        "--smooth-passes",
        type=int,
        default=DEFAULT_SMOOTH_PASSES,
        help="How many passes of spatial smoothing to apply before plotting. Higher values produce broader smoothing.",
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


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


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


def build_plot_forecast_hours(max_forecast_hour: int) -> tuple[int, ...]:
    return tuple(range(6, max_forecast_hour + 1, 6))


def floor_to_cycle(timestamp: dt.datetime) -> RunCycle:
    timestamp = timestamp.astimezone(UTC)
    cycle_hour = (timestamp.hour // 6) * 6
    floored = timestamp.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    return RunCycle(floored)


def build_url(run_cycle: RunCycle, forecast_hour: int) -> str:
    return requests.Request(
        method="GET",
        url=BASE_URL,
        params={
            "dir": run_cycle.nomads_directory,
            "file": run_cycle.file_name(forecast_hour),
            "var_PRATE": "on",
            "lev_surface": "on",
        },
    ).prepare().url


def local_grib_path(root: Path, run_cycle: RunCycle, forecast_hour: int) -> Path:
    return root / "grib" / run_cycle.tag / f"{run_cycle.file_name(forecast_hour)}.grib2"


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
    url = build_url(run_cycle, forecast_hour)
    try:
        response = client.request("GET", url, stream=True)
    except RuntimeError as exc:
        logging.debug("Probe failed for %s f%03d: %s", run_cycle.tag, forecast_hour, exc)
        return False

    with response:
        return response_contains_grib_payload(response)


def resolve_latest_complete_cycle(now_utc: dt.datetime, lookback_cycles: int, client: NomadsClient) -> RunCycle:
    candidate = floor_to_cycle(now_utc)
    for offset in range(lookback_cycles + 1):
        run_cycle = candidate.shifted(hours=-6 * offset)
        if probe_file_available(client, run_cycle, MAX_DOWNLOAD_FORECAST_HOUR):
            logging.info("Selected latest complete run: %s", run_cycle.tag)
            return run_cycle
        logging.info("Run %s is not complete yet, falling back one cycle", run_cycle.tag)
    raise RuntimeError("Could not find a complete recent GFS run within the configured lookback window.")


def build_run_sequence(current_run: RunCycle, history_runs: int) -> list[RunCycle]:
    return [current_run.shifted(hours=-6 * offset) for offset in range(history_runs + 1)]


def prune_old_run_directories(root: Path, keep_runs: int) -> None:
    if keep_runs < 1 or not root.exists():
        return

    run_directories = sorted(path for path in root.iterdir() if path.is_dir())
    stale_directories = run_directories[:-keep_runs]
    for stale_directory in stale_directories:
        shutil.rmtree(stale_directory, ignore_errors=False)
        logging.info("Deleted old run folder %s", stale_directory)


def download_file(
    client: NomadsClient,
    url: str,
    destination: Path,
    overwrite: bool,
) -> Path:
    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with client.request("GET", url, stream=True) as response:
            assert_grib_payload(response, destination.name)
            with temp_path.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1_048_576):
                    if chunk:
                        file_handle.write(chunk)
        temp_path.replace(destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return destination


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
        logging.info("Skipping %s because all %d files already exist", run_cycle.tag, len(forecast_hours))
        return

    logging.info(
        "Downloading %s (%d missing of %d files)",
        run_cycle.tag,
        len(hours_to_download),
        len(forecast_hours),
    )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = []
        for forecast_hour in hours_to_download:
            url = build_url(run_cycle, forecast_hour)
            destination = local_grib_path(root, run_cycle, forecast_hour)
            futures.append(executor.submit(download_file, client, url, destination, overwrite))
        for future in as_completed(futures):
            future.result()


def normalize_longitudes(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.nanmax(lons) <= 180:
        return values, lats, lons

    adjusted_lons = np.where(lons > 180, lons - 360, lons)
    sort_order = np.argsort(adjusted_lons[0, :])
    return values[:, sort_order], lats[:, sort_order], adjusted_lons[:, sort_order]


def build_coordinate_grids(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    latitude = dataset.coords.get("latitude")
    if latitude is None:
        latitude = dataset.coords.get("lat")

    longitude = dataset.coords.get("longitude")
    if longitude is None:
        longitude = dataset.coords.get("lon")

    if latitude is None or longitude is None:
        raise ValueError("GRIB dataset is missing latitude/longitude coordinates.")

    lat_values = np.asarray(latitude.values, dtype=np.float32)
    lon_values = np.asarray(longitude.values, dtype=np.float32)
    if lat_values.ndim == 1 and lon_values.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
        return lat_grid.astype(np.float32), lon_grid.astype(np.float32)
    if lat_values.ndim == 2 and lon_values.ndim == 2:
        return lat_values, lon_values

    raise ValueError(
        "Unsupported GRIB coordinate layout: expected 1D or 2D latitude/longitude arrays."
    )


def select_precipitation_rate(dataset: xr.Dataset) -> xr.DataArray:
    if "prate" in dataset.data_vars:
        return dataset["prate"].squeeze(drop=True)

    for variable in dataset.data_vars.values():
        if variable.attrs.get("GRIB_name") == "Precipitation rate":
            return variable.squeeze(drop=True)
        if variable.attrs.get("GRIB_shortName") == "prate":
            return variable.squeeze(drop=True)

    if len(dataset.data_vars) == 1:
        return next(iter(dataset.data_vars.values())).squeeze(drop=True)

    raise ValueError("Could not locate a precipitation-rate field in the GRIB dataset.")


def load_prate_mmhr(grib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset: xr.Dataset | None = None
    last_error: Exception | None = None

    for backend_kwargs in CFGRIB_BACKEND_KWARGS:
        try:
            dataset = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs)
            data_array = select_precipitation_rate(dataset)
            values = np.asarray(data_array.values, dtype=np.float32)
            if values.ndim != 2:
                raise ValueError(
                    f"Expected a 2D precipitation field in {grib_path.name}, got shape {values.shape}."
                )
            lats, lons = build_coordinate_grids(dataset)
            break
        except Exception as exc:
            last_error = exc
            if dataset is not None:
                dataset.close()
                dataset = None
    else:
        raise RuntimeError(f"Unable to read precipitation rate from {grib_path.name}: {last_error}") from last_error

    assert dataset is not None
    dataset.close()

    values = np.maximum(values, 0.0) * 3600.0
    return normalize_longitudes(values, lats, lons)


def valid_time(run_cycle: RunCycle, forecast_hour: int) -> dt.datetime:
    return run_cycle.init_time + dt.timedelta(hours=forecast_hour)


def build_plot_title(run_cycle: RunCycle, forecast_hour: int) -> str:
    valid_local = valid_time(run_cycle, forecast_hour).astimezone(EASTERN)
    hour_str_fmt = valid_local.strftime("%I:%M %p").lstrip("0")
    day_of_week = valid_local.strftime("%A")
    return (
        f"Precip Confidence Forecast F{forecast_hour:03d}\n"
        f"Valid {day_of_week}, {valid_local.strftime('%b %d, %Y')} at {hour_str_fmt} ET"
    )


def build_plot_subtitle(run_cycle: RunCycle, forecast_hour: int, member_count: int) -> str:
    return (
        f"GFS 0.25deg | Run {run_cycle.tag} | Forecast Hour F{forecast_hour:03d} | "
        f"Aligned Members {member_count}"
    )


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
            logging.warning("Missing aligned member for %s at f%03d", run_cycle.tag, aligned_hour)
            continue
        values, current_lats, current_lons = load_prate_mmhr(grib_path)
        if lats is None or lons is None:
            lats = current_lats
            lons = current_lons
        member_arrays.append(values)
        metadata.append((run_cycle.tag, aligned_hour))

    if not member_arrays or lats is None or lons is None:
        raise RuntimeError(f"No aligned members were available for forecast hour f{current_forecast_hour:03d}.")

    return np.stack(member_arrays, axis=0), lats, lons, metadata


def calculate_confidence(
    members_mmhr: np.ndarray,
    rain_threshold_mmhr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wet_members = members_mmhr >= rain_threshold_mmhr
    wet_count = wet_members.sum(axis=0)
    wet_fraction = wet_members.mean(axis=0)
    mean_rate = members_mmhr.mean(axis=0)

    wet_sum = np.sum(np.where(wet_members, members_mmhr, 0.0), axis=0)
    wet_mean_rate = np.where(wet_count > 0, wet_sum / wet_count, 0.0)

    spread = members_mmhr.std(axis=0)
    spread_ratio = np.where(wet_mean_rate > 0, spread / np.maximum(wet_mean_rate, 0.25), 1.0)
    spread_penalty = np.clip(spread_ratio / 2.0, 0.0, 1.0)

    confidence = 10.0 * wet_fraction * (1.0 - 0.55 * spread_penalty)
    confidence = np.where(wet_count > 0, confidence, 0.0)
    confidence = np.clip(confidence, 0.0, 10.0)

    return confidence.astype(np.float32), mean_rate.astype(np.float32), wet_fraction.astype(np.float32), spread.astype(np.float32)


def confidence_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    boundaries = np.arange(0, 11, 1)
    cmap = ListedColormap(LIGHT_CONFIDENCE_COLORS, name="rain_confidence")
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm


def smooth_field(field: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0:
        return field

    smoothed = np.asarray(field, dtype=np.float32)
    kernel_1d = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32)
    kernel = np.outer(kernel_1d, kernel_1d)
    kernel /= kernel.sum()
    kernel_size = kernel.shape[0]
    pad_width = kernel_size // 2

    for _ in range(passes):
        padded = np.pad(smoothed, ((pad_width, pad_width), (pad_width, pad_width)), mode="edge")
        next_field = np.zeros_like(smoothed)
        for row_offset in range(kernel_size):
            for col_offset in range(kernel_size):
                next_field += (
                    padded[
                        row_offset : row_offset + smoothed.shape[0],
                        col_offset : col_offset + smoothed.shape[1],
                    ]
                    * kernel[row_offset, col_offset]
                )
        smoothed = next_field

    return smoothed


def plot_confidence_map(
    save_path: Path,
    run_cycle: RunCycle,
    forecast_hour: int,
    lats: np.ndarray,
    lons: np.ndarray,
    confidence: np.ndarray,
    mean_rate: np.ndarray,
    member_count: int,
    rain_threshold_mmhr: float,
    smooth_passes: int,
) -> None:
    smoothed_confidence = np.clip(smooth_field(confidence, smooth_passes), 0.0, 10.0)
    smoothed_mean_rate = np.maximum(smooth_field(mean_rate, smooth_passes), 0.0)
    projected_confidence = np.ma.masked_where(
        smoothed_mean_rate < rain_threshold_mmhr,
        smoothed_confidence,
    )
    cmap, norm = confidence_cmap()

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
        color="#3b4a5a",
        pad=10,
        loc="left",
        fontweight="normal",
    )

    colorbar = plt.colorbar(filled, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label("R confidence (0-10)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_confidence_products(
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
        confidence, mean_rate, _, _ = calculate_confidence(members, rain_threshold_mmhr)
        save_path = plot_root / f"rain_confidence_f{forecast_hour:03d}.png"
        plot_confidence_map(
            save_path=save_path,
            run_cycle=run_cycle,
            forecast_hour=forecast_hour,
            lats=lats,
            lons=lons,
            confidence=confidence,
            mean_rate=mean_rate,
            member_count=len(metadata),
            rain_threshold_mmhr=rain_threshold_mmhr,
            smooth_passes=smooth_passes,
        )
        logging.info("Saved %s", save_path)


def ensure_complete_history(
    run_cycles: Iterable[RunCycle],
    client: NomadsClient,
) -> None:
    for run_cycle in run_cycles:
        if not probe_file_available(client, run_cycle, MAX_DOWNLOAD_FORECAST_HOUR):
            raise RuntimeError(
                f"Run {run_cycle.tag} does not appear complete on NOMADS. "
                "Reduce --history-runs or wait for older runs to become fully available."
            )


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

    # Enforce retention on existing data before any new work begins.
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
                forecast_hours=DOWNLOAD_FORECAST_HOURS,
                overwrite=args.overwrite,
                workers=args.download_workers,
            )
        prune_old_run_directories(output_root / "grib", args.retain_runs)

    build_confidence_products(
        root=output_root,
        run_cycle=current_run,
        run_cycles=run_cycles,
        plot_forecast_hours=plot_forecast_hours,
        rain_threshold_mmhr=args.rain_threshold_mmhr,
        smooth_passes=max(0, args.smooth_passes),
    )
    prune_old_run_directories(output_root / "plots", args.retain_runs)

    logging.info("Finished building confidence maps for %s", current_run.tag)


if __name__ == "__main__":
    main()
