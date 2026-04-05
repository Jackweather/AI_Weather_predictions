from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gfs_rain_confidence import (
    BASE_DIR,
    CONUS_EXTENT,
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_MAX_PLOT_FORECAST_HOUR,
    DEFAULT_MAX_REQUEST_RETRIES,
    DEFAULT_REQUEST_MIN_INTERVAL,
    DEFAULT_SMOOTH_PASSES,
    DOWNLOAD_FORECAST_HOURS,
    MAX_DOWNLOAD_FORECAST_HOUR,
    NomadsClient,
    RunCycle,
    UTC,
    build_plot_forecast_hours,
    build_plot_title,
    build_run_sequence,
    calculate_confidence,
    ccrs,
    cfeature,
    collect_aligned_members,
    confidence_cmap,
    download_run,
    ensure_complete_history,
    floor_to_cycle,
    plt,
    prune_old_run_directories,
    resolve_latest_complete_cycle,
    setup_logging,
    smooth_field,
    validate_args,
)

AI_PLOTS_DIR_NAME = "plots_ai"
AI_STATE_DIR_NAME = "ai_state"
FEEDBACK_DIR_NAME = "feedback"
MODEL_STATE_FILE_NAME = "online_rain_model.json"
FEEDBACK_METRICS_FILE_NAME = "feedback_metrics.csv"
TRAINING_SAMPLE_SIZE = 20000
FULL_AI_BLEND_SAMPLE_COUNT = 50000
FEATURE_NAMES = (
    "log_current_rate",
    "log_ensemble_mean",
    "log_ensemble_max",
    "log_spread",
    "wet_fraction",
    "current_minus_mean",
    "current_to_mean_ratio",
    "recent_trend",
    "lat_sin",
    "lon_sin",
    "forecast_hour_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build AI-adaptive rain confidence maps from the newest complete GFS run, "
            "the previous 7 aligned runs, and any observation feedback grids available on disk."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE_DIR / "gfs_rain_confidence_output",
        help="Directory used for downloaded GRIB files, AI state, and generated plots.",
    )
    parser.add_argument(
        "--feedback-root",
        type=Path,
        default=None,
        help="Directory containing observed feedback grids. Defaults to <output-root>/feedback.",
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
        help="Rain threshold in mm/hr used to classify observed or predicted rain.",
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
        "--training-sample-size",
        type=int,
        default=TRAINING_SAMPLE_SIZE,
        help="How many grid points to save per forecast hour for online learning updates.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.04,
        help="Learning rate used for online logistic updates when feedback is present.",
    )
    parser.add_argument(
        "--online-epochs",
        type=int,
        default=3,
        help="How many gradient passes to apply per feedback batch.",
    )
    parser.add_argument(
        "--full-ai-blend-samples",
        type=int,
        default=FULL_AI_BLEND_SAMPLE_COUNT,
        help="Samples required before the learned model reaches full influence over the baseline.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def validate_ai_args(args: argparse.Namespace) -> None:
    validate_args(args)
    if args.training_sample_size < 100:
        raise ValueError("--training-sample-size must be at least 100.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0.")
    if args.online_epochs < 1:
        raise ValueError("--online-epochs must be at least 1.")
    if args.full_ai_blend_samples < 1000:
        raise ValueError("--full-ai-blend-samples must be at least 1000.")


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class OnlineRainModel:
    feature_names: tuple[str, ...]
    weights: np.ndarray
    bias: float = 0.0
    sample_count: int = 0
    update_steps: int = 0

    @classmethod
    def create(cls) -> "OnlineRainModel":
        return cls(
            feature_names=FEATURE_NAMES,
            weights=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> "OnlineRainModel":
        if not path.exists():
            return cls.create()

        payload = json.loads(path.read_text(encoding="utf-8"))
        feature_names = tuple(payload.get("feature_names", FEATURE_NAMES))
        weights = np.asarray(payload.get("weights", []), dtype=np.float32)
        if weights.shape != (len(feature_names),):
            return cls.create()

        return cls(
            feature_names=feature_names,
            weights=weights,
            bias=float(payload.get("bias", 0.0)),
            sample_count=int(payload.get("sample_count", 0)),
            update_steps=int(payload.get("update_steps", 0)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "feature_names": list(self.feature_names),
                    "weights": self.weights.astype(float).tolist(),
                    "bias": self.bias,
                    "sample_count": self.sample_count,
                    "update_steps": self.update_steps,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights + self.bias
        return sigmoid(logits).astype(np.float32)

    def update(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        learning_rate: float,
        epochs: int,
    ) -> None:
        feature_batch = np.asarray(features, dtype=np.float32)
        label_batch = np.asarray(labels, dtype=np.float32)
        if feature_batch.size == 0 or label_batch.size == 0:
            return

        for _ in range(max(1, epochs)):
            probabilities = self.predict_proba(feature_batch)
            error = probabilities - label_batch
            gradient_weights = feature_batch.T @ error / feature_batch.shape[0]
            gradient_bias = float(error.mean())
            self.weights -= learning_rate * gradient_weights
            self.bias -= learning_rate * gradient_bias

        self.sample_count += int(feature_batch.shape[0])
        self.update_steps += int(max(1, epochs))


def parse_run_tag(run_tag: str) -> RunCycle | None:
    try:
        return RunCycle(dt.datetime.strptime(run_tag, "%Y%m%d_%HZ").replace(tzinfo=UTC))
    except ValueError:
        return None


def model_blend_weight(model: OnlineRainModel, full_ai_blend_samples: int) -> float:
    if full_ai_blend_samples <= 0:
        return 0.9
    return float(min(0.9, model.sample_count / float(full_ai_blend_samples) * 0.9))


def build_feature_matrix(
    members_mmhr: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    forecast_hour: int,
    rain_threshold_mmhr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_rate = np.asarray(members_mmhr[0], dtype=np.float32)
    ensemble_mean = np.asarray(members_mmhr.mean(axis=0), dtype=np.float32)
    ensemble_max = np.asarray(members_mmhr.max(axis=0), dtype=np.float32)
    spread = np.asarray(members_mmhr.std(axis=0), dtype=np.float32)
    wet_fraction = np.asarray((members_mmhr >= rain_threshold_mmhr).mean(axis=0), dtype=np.float32)
    current_minus_mean = np.tanh((current_rate - ensemble_mean) / 4.0).astype(np.float32)
    current_to_mean_ratio = np.tanh((current_rate / np.maximum(ensemble_mean, 0.25)) - 1.0).astype(np.float32)
    recent_trend = np.tanh((members_mmhr[0] - members_mmhr[-1]) / 4.0).astype(np.float32)
    lat_sin = np.sin(np.deg2rad(lats)).astype(np.float32)
    lon_sin = np.sin(np.deg2rad(lons)).astype(np.float32)
    forecast_hour_norm = np.full_like(current_rate, forecast_hour / MAX_DOWNLOAD_FORECAST_HOUR, dtype=np.float32)

    feature_stack = np.stack(
        [
            np.log1p(current_rate),
            np.log1p(ensemble_mean),
            np.log1p(ensemble_max),
            np.log1p(spread),
            wet_fraction,
            current_minus_mean,
            current_to_mean_ratio,
            recent_trend,
            lat_sin,
            lon_sin,
            forecast_hour_norm,
        ],
        axis=-1,
    )

    return (
        feature_stack.reshape(-1, feature_stack.shape[-1]).astype(np.float32),
        ensemble_mean,
        wet_fraction,
        spread,
    )


def save_training_snapshot(
    output_root: Path,
    run_cycle: RunCycle,
    forecast_hour: int,
    features: np.ndarray,
    prediction_grid: np.ndarray,
    sample_size: int,
) -> None:
    total_points = features.shape[0]
    if total_points == 0:
        return

    chosen_size = min(total_points, max(100, sample_size))
    seed = int(run_cycle.date_token) + (run_cycle.cycle_hour * 1000) + forecast_hour
    random_generator = np.random.default_rng(seed)
    selected_indices = random_generator.choice(total_points, size=chosen_size, replace=False)
    row_indices, col_indices = np.unravel_index(selected_indices, prediction_grid.shape)

    snapshot_root = output_root / AI_STATE_DIR_NAME / "samples" / run_cycle.tag
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_root / f"f{forecast_hour:03d}_samples.npz"
    np.savez_compressed(
        snapshot_path,
        rows=row_indices.astype(np.int32),
        cols=col_indices.astype(np.int32),
        features=features[selected_indices].astype(np.float32),
        probabilities=prediction_grid.reshape(-1)[selected_indices].astype(np.float32),
        shape=np.asarray(prediction_grid.shape, dtype=np.int32),
    )


def load_feedback_grid(feedback_root: Path, run_tag: str, forecast_hour: int) -> np.ndarray | None:
    candidate_paths = [
        feedback_root / run_tag / f"observed_rain_f{forecast_hour:03d}.npy",
        feedback_root / run_tag / f"observed_rain_f{forecast_hour:03d}.npz",
    ]
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue

        if candidate_path.suffix.lower() == ".npy":
            return np.asarray(np.load(candidate_path), dtype=np.float32)

        payload = np.load(candidate_path)
        if "observed_mmhr" in payload.files:
            return np.asarray(payload["observed_mmhr"], dtype=np.float32)
        if payload.files:
            return np.asarray(payload[payload.files[0]], dtype=np.float32)

    return None


def compute_feedback_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predicted_labels = probabilities >= 0.5
    return {
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "accuracy": float(np.mean(predicted_labels == labels)),
        "predicted_rain_rate": float(probabilities.mean()),
        "observed_rain_rate": float(labels.mean()),
    }


def append_feedback_metrics(metrics_path: Path, row: dict[str, str | int | float]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = metrics_path.exists()
    fieldnames = [
        "processed_at_utc",
        "run_tag",
        "forecast_hour",
        "sample_count",
        "brier_before",
        "brier_after",
        "accuracy_before",
        "accuracy_after",
        "observed_rain_rate",
    ]
    with metrics_path.open("a", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def process_feedback_updates(
    output_root: Path,
    feedback_root: Path,
    model: OnlineRainModel,
    rain_threshold_mmhr: float,
    learning_rate: float,
    online_epochs: int,
) -> int:
    samples_root = output_root / AI_STATE_DIR_NAME / "samples"
    metrics_path = output_root / AI_STATE_DIR_NAME / FEEDBACK_METRICS_FILE_NAME
    if not samples_root.exists():
        return 0

    processed_count = 0
    for run_dir in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        for snapshot_path in sorted(run_dir.glob("f*_samples.npz")):
            marker_path = snapshot_path.with_suffix(".done.json")
            if marker_path.exists():
                continue

            forecast_hour = int(snapshot_path.stem.split("_")[0][1:])
            observed_grid = load_feedback_grid(feedback_root, run_dir.name, forecast_hour)
            if observed_grid is None:
                continue

            sample_payload = np.load(snapshot_path)
            rows = np.asarray(sample_payload["rows"], dtype=np.int32)
            cols = np.asarray(sample_payload["cols"], dtype=np.int32)
            features = np.asarray(sample_payload["features"], dtype=np.float32)
            if observed_grid.ndim != 2:
                logging.warning("Skipping feedback for %s f%03d because the observed grid is not 2D", run_dir.name, forecast_hour)
                continue
            if rows.max(initial=-1) >= observed_grid.shape[0] or cols.max(initial=-1) >= observed_grid.shape[1]:
                logging.warning("Skipping feedback for %s f%03d because the observed grid shape does not match saved samples", run_dir.name, forecast_hour)
                continue

            labels = (observed_grid[rows, cols] >= rain_threshold_mmhr).astype(np.float32)
            probabilities_before = model.predict_proba(features)
            metrics_before = compute_feedback_metrics(probabilities_before, labels)
            model.update(features, labels, learning_rate=learning_rate, epochs=online_epochs)
            probabilities_after = model.predict_proba(features)
            metrics_after = compute_feedback_metrics(probabilities_after, labels)

            processed_at = dt.datetime.now(tz=UTC).isoformat()
            append_feedback_metrics(
                metrics_path,
                {
                    "processed_at_utc": processed_at,
                    "run_tag": run_dir.name,
                    "forecast_hour": forecast_hour,
                    "sample_count": int(features.shape[0]),
                    "brier_before": f"{metrics_before['brier']:.6f}",
                    "brier_after": f"{metrics_after['brier']:.6f}",
                    "accuracy_before": f"{metrics_before['accuracy']:.6f}",
                    "accuracy_after": f"{metrics_after['accuracy']:.6f}",
                    "observed_rain_rate": f"{metrics_before['observed_rain_rate']:.6f}",
                },
            )
            marker_path.write_text(
                json.dumps(
                    {
                        "processed_at_utc": processed_at,
                        "metrics_before": metrics_before,
                        "metrics_after": metrics_after,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            processed_count += 1
            logging.info(
                "Applied feedback for %s f%03d | brier %.4f -> %.4f | accuracy %.3f -> %.3f",
                run_dir.name,
                forecast_hour,
                metrics_before["brier"],
                metrics_after["brier"],
                metrics_before["accuracy"],
                metrics_after["accuracy"],
            )

    return processed_count


def plot_ai_confidence_map(
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
    blend_weight: float,
    learned_samples: int,
) -> None:
    smoothed_confidence = np.clip(smooth_field(confidence, smooth_passes), 0.0, 10.0)
    smoothed_mean_rate = np.maximum(smooth_field(mean_rate, smooth_passes), 0.0)
    projected_confidence = np.ma.masked_where(smoothed_mean_rate < rain_threshold_mmhr, smoothed_confidence)
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
        build_plot_title(run_cycle, forecast_hour).replace("Precip Confidence", "AI Adaptive Rain Confidence"),
        fontsize=16,
        color="#3b4a5a",
        pad=10,
        loc="left",
        fontweight="normal",
    )
    axis.text(
        0.01,
        0.02,
        (
            f"Blend {blend_weight:.0%} learned / {100 - round(blend_weight * 100):d}% baseline"
            f" | Learned samples {learned_samples:,} | Aligned members {member_count}"
        ),
        transform=axis.transAxes,
        fontsize=10,
        color="#31455c",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "boxstyle": "round,pad=0.35"},
    )

    colorbar = plt.colorbar(filled, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label("AI rain confidence (0-10)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_ai_confidence_products(
    output_root: Path,
    run_cycle: RunCycle,
    run_cycles: list[RunCycle],
    plot_forecast_hours: tuple[int, ...],
    rain_threshold_mmhr: float,
    smooth_passes: int,
    model: OnlineRainModel,
    training_sample_size: int,
    full_ai_blend_samples: int,
) -> None:
    plot_root = output_root / AI_PLOTS_DIR_NAME / run_cycle.tag
    blend_weight = model_blend_weight(model, full_ai_blend_samples)

    for forecast_hour in plot_forecast_hours:
        members, lats, lons, metadata = collect_aligned_members(output_root, run_cycles, forecast_hour)
        features, mean_rate, _, _ = build_feature_matrix(
            members,
            lats,
            lons,
            forecast_hour,
            rain_threshold_mmhr,
        )
        baseline_confidence, baseline_mean_rate, _, _ = calculate_confidence(members, rain_threshold_mmhr)
        baseline_probability = (baseline_confidence / 10.0).reshape(-1)
        learned_probability = model.predict_proba(features)
        blended_probability = (
            ((1.0 - blend_weight) * baseline_probability) + (blend_weight * learned_probability)
        ).astype(np.float32)
        confidence_grid = (blended_probability.reshape(mean_rate.shape) * 10.0).astype(np.float32)
        save_path = plot_root / f"ai_rain_confidence_f{forecast_hour:03d}.png"

        plot_ai_confidence_map(
            save_path=save_path,
            run_cycle=run_cycle,
            forecast_hour=forecast_hour,
            lats=lats,
            lons=lons,
            confidence=confidence_grid,
            mean_rate=baseline_mean_rate,
            member_count=len(metadata),
            rain_threshold_mmhr=rain_threshold_mmhr,
            smooth_passes=smooth_passes,
            blend_weight=blend_weight,
            learned_samples=model.sample_count,
        )
        save_training_snapshot(
            output_root=output_root,
            run_cycle=run_cycle,
            forecast_hour=forecast_hour,
            features=features,
            prediction_grid=blended_probability.reshape(mean_rate.shape),
            sample_size=training_sample_size,
        )
        logging.info("Saved %s", save_path)


def main() -> None:
    args = parse_args()
    validate_ai_args(args)
    setup_logging(args.log_level)

    output_root = args.output_root.resolve()
    feedback_root = (args.feedback_root or (output_root / FEEDBACK_DIR_NAME)).resolve()
    plot_forecast_hours = build_plot_forecast_hours(args.max_plot_forecast_hour)
    model_path = output_root / AI_STATE_DIR_NAME / MODEL_STATE_FILE_NAME
    model = OnlineRainModel.load(model_path)

    processed_feedback = process_feedback_updates(
        output_root=output_root,
        feedback_root=feedback_root,
        model=model,
        rain_threshold_mmhr=args.rain_threshold_mmhr,
        learning_rate=args.learning_rate,
        online_epochs=args.online_epochs,
    )
    if processed_feedback:
        model.save(model_path)

    now_utc = dt.datetime.now(tz=UTC)
    client = NomadsClient(
        timeout=args.request_timeout,
        min_interval_seconds=args.request_min_interval,
        max_retries=args.max_request_retries,
    )

    prune_old_run_directories(output_root / "grib", args.retain_runs)
    prune_old_run_directories(output_root / AI_PLOTS_DIR_NAME, args.retain_runs)

    current_run = resolve_latest_complete_cycle(now_utc, args.cycle_lookback, client)
    run_cycles = build_run_sequence(current_run, args.history_runs)
    ensure_complete_history(run_cycles, client)

    logging.info("Output root: %s", output_root)
    logging.info("Feedback root: %s", feedback_root)
    logging.info("Current run: %s", current_run.tag)
    logging.info("Model samples: %d | update steps: %d", model.sample_count, model.update_steps)

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

    build_ai_confidence_products(
        output_root=output_root,
        run_cycle=current_run,
        run_cycles=run_cycles,
        plot_forecast_hours=plot_forecast_hours,
        rain_threshold_mmhr=args.rain_threshold_mmhr,
        smooth_passes=max(0, args.smooth_passes),
        model=model,
        training_sample_size=args.training_sample_size,
        full_ai_blend_samples=args.full_ai_blend_samples,
    )
    model.save(model_path)
    prune_old_run_directories(output_root / AI_PLOTS_DIR_NAME, args.retain_runs)

    logging.info("Finished building AI confidence maps for %s", current_run.tag)


if __name__ == "__main__":
    main()