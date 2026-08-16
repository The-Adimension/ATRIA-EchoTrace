"""Command-line interface.

Each subcommand corresponds to a phase of the original notebook:

======================  =====================================================
``atria serve``         the Adapter-Human Exchange Interface (notebook L1381+)
``atria doctor``        the dataset sanity-check cells (L131-135, L371-389)
``atria ingest``        raw dataset preprocessing into the three artefacts
``atria train``         the MedGemma FineTuning section (L618-851)
``atria evaluate``      the evaluation section (L992-1169)
``atria publish-adapter``  the HuggingFace Adapter Transfer cell (L1580-1754)
======================  =====================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from pathlib import Path

from . import __version__
from .config import ADAPTERS, Settings, display_path, settings as default_settings
from .logging_setup import configure, get_logger

logger = get_logger("cli")


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #
def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    """Run the web application."""
    import uvicorn

    host = args.host or settings.host
    port = args.port or settings.port
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"

    print(f"\n  ATRIA EchoTrace {__version__}")
    print(f"  Workstation : {url}")
    print(f"  API docs    : {url}/api/docs")
    print(f"  Dataset     : {settings.dataset_dir}")
    print(f"  Outputs     : {settings.output_dir}\n")

    if not args.no_browser:
        # Delay so the browser opens after uvicorn is accepting connections.
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "atria_echotrace.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        log_config=None,  # already configured by configure()
    )
    return 0


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    """Validate the environment and dataset; print a readiness report."""
    from .data.dataset import DatasetError, DatasetRepository

    ok = True
    print(f"\nATRIA EchoTrace {__version__} — environment report\n" + "=" * 62)
    print(f"Python          : {sys.version.split()[0]}")
    print(f"Dataset dir     : {settings.dataset_dir}")
    print(f"Output dir      : {settings.output_dir}")
    print(f"Base model      : {settings.base_model_id}")
    print(f"HF token        : {'present' if settings.token_value() else 'absent'}")

    # Where each weight would actually come from — the first question anyone asks
    # when the AI tier will not start.
    weights = settings.weights_report()
    base = weights["base"]
    print("\nWeights")
    print("-" * 62)
    print(f"  base          : [{base['source']}] {base['detail']}")
    for adapter in weights["adapters"]:
        print(f"  adapter {adapter['id']:<7}: [{adapter['source']}] {adapter['detail']}")
    if not base["ready"]:
        print(f"  Place a copy in {weights['models_dir']}, or sign in with `hf auth login`.")

    # --- AI tier ---
    print("\nAI tier")
    print("-" * 62)
    try:
        import torch  # noqa: F401
        import transformers

        from .ml.runtime import describe_device

        device = describe_device(force_cpu=settings.force_cpu)
        print(f"  transformers  : {transformers.__version__}")
        print(f"  device        : {device['device']}")
        if device.get("gpu_name"):
            print(f"  gpu           : {device['gpu_name']} ({device['total_memory_gb']} GB, {device['capability']})")
        print(f"  compute dtype : {device['compute_dtype']}")
        print(f"  quantisation  : {device['quantization'] or 'none'}")
        if device.get("torch_cuda_build"):
            print(f"  torch build   : {device['torch_version']} (CUDA {device['torch_cuda_build']})")
        elif device.get("torch_version"):
            print(f"  torch build   : {device['torch_version']} (no CUDA)")
        print(f"  note          : {device['reason']}")
        # A CPU-only wheel on a GPU machine is a one-command fix, not a hardware fault.
        if device.get("remedy"):
            print(f"  FIX           : {device['remedy']}")
    except ImportError as exc:
        print(f"  not installed ({exc}).")
        print('  Install with: pip install -e ".[ai]"')

    # --- dataset ---
    print("\nDataset")
    print("-" * 62)
    try:
        report = DatasetRepository(settings.dataset_dir).validate()
    except DatasetError as exc:
        print(f"  FAIL: {exc}")
        return 1

    print(f"  frames        : {report.n_tracings} ({report.n_frames_present} PNGs present)")
    print(f"  cases         : {report.n_cases}")
    print(f"  sources       : {report.source_counts}")
    print(f"  views         : {report.view_counts}")
    print(f"  instants      : {report.instant_counts}")
    print(f"  splits        : {report.split_counts}")
    print(f"  LV vertices   : {dict(sorted(report.lv_point_counts.items()))}")
    print(f"  LA vertices   : {dict(sorted(report.la_point_counts.items()))}")
    print(f"  metadata.csv  : {'present' if report.has_metadata_csv else 'absent (optional)'}")

    if report.missing_pngs:
        ok = False
        print(f"  FAIL: {len(report.missing_pngs)} tracing(s) without a PNG: {report.missing_pngs[:5]}")
    if report.frames_without_lv:
        print(f"  WARN: {len(report.frames_without_lv)} frame(s) without an LV polygon")
    if report.incomplete_cases:
        print(f"  WARN: {len(report.incomplete_cases)} case(s) missing ED or ES: {report.incomplete_cases[:5]}")
    if report.uncalibrated_sources:
        print(
            f"  NOTE: no pixel spacing for source(s) {report.uncalibrated_sources}; "
            "physical areas (cm²) are withheld for those frames."
        )
    if report.instant_area_anomalies:
        print(
            f"  WARN: {len(report.instant_area_anomalies)} case(s) whose ES trace is at least "
            "as large as their ED trace, suggesting transposed ED/ES labels:"
        )
        for item in report.instant_area_anomalies[:5]:
            print(
                f"          {item['case_key']}: ED {item['ed_area_px']} px² vs "
                f"ES {item['es_area_px']} px²"
                + (f" (recorded EF {item['ef']:.1f}%)" if item.get("ef") else "")
            )

    print("\n" + ("PASS — ready to serve." if ok else "FAIL — see messages above."))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    """Run the reference preprocessors over a raw dataset.

    The heavy lifting belongs to the vendored scripts that produced the adapters'
    training corpus (``data/ingest/reference/``); this only dispatches to them.
    """
    from .data.ingest import IngestError
    from .data.ingest import run as ingest_run

    if args.dataset in ("camus", "echonet") and not args.source:
        print(f"ingest {args.dataset} needs --source.", file=sys.stderr)
        return 1

    try:
        if args.dataset == "camus":
            result = ingest_run.ingest_camus(
                source_dir=Path(args.source),
                output_dir=Path(args.output),
                n_points=args.points,
            )
        elif args.dataset == "echonet":
            result = ingest_run.ingest_echonet(
                source_dir=Path(args.source),
                output_dir=Path(args.output),
                n_points=args.points,
                max_videos=args.max_videos,
                # 0 means "keep the native 112x112 frames".
                target_size=args.target_size or None,
            )
        else:  # unified
            if not args.camus_processed or not args.echonet_processed:
                print(
                    "ingest unified needs --camus-processed and --echonet-processed.",
                    file=sys.stderr,
                )
                return 1
            result = ingest_run.merge_unified(
                camus_processed=Path(args.camus_processed),
                echonet_processed=Path(args.echonet_processed),
                output_dir=Path(args.output),
                n_points=args.points,
            )
    except IngestError as exc:
        print(f"Ingest failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.as_dict(), indent=2))
    print(
        f"\nWrote {result.n_frames} frames for {result.n_cases} case(s) to {result.output_dir}\n"
        f"Serve it with: atria serve --dataset-dir {result.output_dir}"
    )
    return 0


# --------------------------------------------------------------------------- #
# classify-camus / classify-echonet
# --------------------------------------------------------------------------- #
def cmd_classify(args: argparse.Namespace, settings: Settings) -> int:
    """Derive classification-task datasets from the processed corpus.

    The library (``data/classify.py``) does the work, exactly as the standalone scripts
    under ``datasets/classification_scripts/`` do, so the two can never drift.
    """
    from .data.classify import BUILDERS, TASKS, ClassificationError, report

    failed = False
    for name in (list(TASKS) if args.task == "all" else [args.task]):
        kwargs: dict[str, object] = {"dry_run": args.dry_run}
        if args.mode == "dirs":
            kwargs["link"] = args.link
        try:
            result = BUILDERS[args.mode](
                TASKS[name], args.dataset_dir, args.output_root, **kwargs
            )
        except ClassificationError as exc:
            print(f"\n{name} [{args.mode}]: FAILED — {exc}", file=sys.stderr)
            failed = True
            continue
        report(result)
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    """Fine-tune MedGemma with QLoRA."""
    from .data.dataset import DatasetRepository
    from .ml.train import run_training

    try:
        result = run_training(
            output_dir=Path(args.output_dir),
            repo=DatasetRepository(settings.dataset_dir),
            settings=settings,
            target_structure=args.structure,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            train_max_samples=args.train_samples,
            val_max_samples=args.val_samples,
        )
    except (RuntimeError, ImportError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.__dict__, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    """Score the model against ground truth (Dice / IoU / parse rate)."""
    from .data.dataset import DatasetRepository
    from .ml.engine import get_inference_engine
    from .ml.evaluate import (
        EvaluationRun,
        evaluate_frames,
        new_run_id,
        save_ranked_figures,
        save_run,
        select_frames,
    )

    import time

    repo = DatasetRepository(settings.dataset_dir)
    frames = select_frames(
        repo=repo,
        target_structure=args.structure,
        split=args.split,
        source=args.source,
        max_samples=args.max_samples,
    )
    if not frames:
        print(
            f"No frames matched split={args.split!r} source={args.source!r} "
            f"structure={args.structure!r}.",
            file=sys.stderr,
        )
        return 1

    engine = get_inference_engine(settings)
    print(f"Loading adapter {args.adapter!r}…")
    try:
        engine.load(args.adapter)
    except Exception as exc:  # noqa: BLE001
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 1

    status = engine.status()
    run = EvaluationRun(
        run_id=new_run_id(),
        split=args.split,
        source=args.source,
        target_structure=args.structure.upper(),
        adapter=status.get("adapter"),
        prompt_variant=args.prompt_variant,
        device=status.get("device"),
        max_samples=args.max_samples,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    print(f"Evaluating {len(frames)} frames on {status.get('device')}…")

    def progress(index: int, total: int, message: str) -> None:
        print(f"  [{index}/{total}] {message}", flush=True)

    evaluate_frames(
        engine=engine,
        repo=repo,
        frames=frames,
        run=run,
        target_structure=args.structure,
        prompt_variant=args.prompt_variant,
        progress=progress,
    )
    run.state = "completed"
    run.finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    settings.ensure_output_dirs()
    path = save_run(run, settings.evaluations_dir)
    # The notebook plotted its best and worst predictions after the numbers (L1149-1169);
    # a mean Dice hides the systematic failures that looking at the extremes reveals.
    figures = save_ranked_figures(run, repo, settings.evaluations_dir, count=args.figures)

    summary = run.summary
    print("\n" + "=" * 62)
    print(f"EVALUATION RESULTS — {run.target_structure} — adapter {args.adapter}")
    print("=" * 62)
    print(f"Total samples : {summary['total_samples']}")
    print(f"Parsed        : {summary['parsed']} ({summary['parse_rate_percent']}%)")
    if summary["dice"]["mean"] is not None:
        print(f"Mean Dice     : {summary['dice']['mean']} +/- {summary['dice']['std']}")
        print(f"Mean IoU      : {summary['iou']['mean']} +/- {summary['iou']['std']}")
    ranked = run.ranked()
    if ranked["best"]:
        print("\nBest:")
        for item in ranked["best"]:
            print(f"  {item['stem']}: Dice {item['dice']}")
        print("Worst:")
        for item in ranked["worst"]:
            print(f"  {item['stem']}: Dice {item['dice']}")
    print(f"\nSaved to {path}")
    return 0


# --------------------------------------------------------------------------- #
# publish-adapter
# --------------------------------------------------------------------------- #
def cmd_export_corpus(args: argparse.Namespace, settings: Settings) -> int:
    """Turn revised contours into a dataset that ``atria train`` accepts.

    Closes the ground-truth evolution loop the Notebook-readme describes: the
    clinician's revision becomes the label, and the result is the same
    frames/tracings.json/metadata.csv contract the preprocessors emit.
    """
    from .export.package import export_corpus

    try:
        summary = export_corpus(
            output_dir=Path(args.out),
            output_root=settings.revisions_dir,
            uploads_dir=settings.uploads_dir,
            dataset_dir=settings.dataset_dir,
            revision_ids=args.revision or None,
            split=args.split,
        )
    except ValueError as exc:
        print(f"Nothing exported: {exc}", file=sys.stderr)
        return 1

    print(
        f"Exported {summary['frames']} frame(s) from {len(summary['revisions'])} "
        f"revision(s) to {display_path(summary['dir'])} (split={summary['split']})."
    )
    for note in summary["skipped"]:
        print(f"  skipped {note}", file=sys.stderr)
    print(f"\nTrain on it with:\n  atria train --dataset-dir {display_path(summary['dir'])}")
    return 0


def cmd_publish(args: argparse.Namespace, settings: Settings) -> int:
    """Upload a trained adapter to the Hugging Face Hub."""
    from .ml.publish import plan_publish, publish_adapter

    try:
        plan = plan_publish(
            folder=Path(args.folder),
            repo_id=args.repo,
            gated=not args.public,
            include_all=args.all_files,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Cannot prepare upload: {exc}", file=sys.stderr)
        return 1

    print("\n" + plan.describe() + "\n")
    if not args.yes:
        print(
            "This uploads the files above to the Hugging Face Hub, creating the "
            "repository if it does not exist.\nRe-run with --yes to proceed.",
            file=sys.stderr,
        )
        return 1

    try:
        url = publish_adapter(plan, token=settings.token_value())
    except RuntimeError as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        return 1
    print(f"Done: {url}")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atria",
        description=(
            "ATRIA EchoTrace — MedGemma-driven echocardiographic contour tracing with "
            "human-in-the-loop revision. Research use only; not a medical device."
        ),
    )
    parser.add_argument("--version", action="version", version=f"atria-echotrace {__version__}")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Directory containing frames/, tracings.json (and optionally metadata.csv).",
    )
    parser.add_argument("--output-dir", type=Path, dest="global_output_dir", help="Where to write outputs.")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR.")

    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p_serve = sub.add_parser("serve", help="Run the web workstation (default command).")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--no-browser", action="store_true", help="Do not open a browser.")
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    p_serve.set_defaults(func=cmd_serve)

    p_doctor = sub.add_parser("doctor", help="Validate the environment and dataset.")
    p_doctor.set_defaults(func=cmd_doctor)

    p_ingest = sub.add_parser(
        "ingest",
        help="Run the reference preprocessors over a raw CAMUS or EchoNet dataset.",
        description=(
            "Runs the vendored scripts that produced the corpus the published adapters "
            "were fine-tuned on, so the output matches that corpus exactly. Needs the "
            'ingest extra: pip install -e ".[ingest]"'
        ),
    )
    p_ingest.add_argument(
        "dataset",
        choices=["camus", "echonet", "unified"],
        help="camus | echonet | unified (merge two processed datasets)",
    )
    p_ingest.add_argument(
        "--source",
        help="Raw dataset root. CAMUS: the CAMUS_public directory. EchoNet: the "
        "directory holding Videos/, FileList.csv and VolumeTracings.csv.",
    )
    p_ingest.add_argument("--output", required=True, help="Destination directory.")
    p_ingest.add_argument(
        "--points", type=int, default=30, help="Vertices per contour (default 30)."
    )
    p_ingest.add_argument(
        "--max-videos", type=int, default=None, help="EchoNet only: cap the video count."
    )
    p_ingest.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="EchoNet only: Lanczos-upscale frames to this square size "
        "(default 224, which reproduces the training corpus; 0 keeps native 112).",
    )
    p_ingest.add_argument(
        "--camus-processed", help="unified only: an existing camus_processed directory."
    )
    p_ingest.add_argument(
        "--echonet-processed", help="unified only: an existing echonet_processed directory."
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # classification-set derivation: same library as the standalone scripts under
    # datasets/classification_scripts/, so the two entry points cannot drift.
    from .data.classify import CLASSIFIED_ROOT, TASKS

    p_cls = sub.add_parser(
        "classify",
        help=f"Derive classification-task datasets ({'/'.join(TASKS)}).",
        description=(
            "Turn the processed corpus into classification sets. Mode 'metadata' writes "
            "a single mapping.csv linking every PNG to its class; mode 'dirs' writes one "
            "directory per class (ImageFolder layout). Labels are copied from the "
            "original dataset metadata; none are invented."
        ),
    )
    p_cls.add_argument("task", choices=[*TASKS, "all"], help="Classification task.")
    p_cls.add_argument("mode", choices=["metadata", "dirs"], help="Which product to write.")
    p_cls.add_argument("--dataset-dir", type=Path, default=None,
                       help="Processed corpus. Default: auto-detected under "
                            "datasets/processed_datasets/.")
    p_cls.add_argument("--output-root", type=Path, default=CLASSIFIED_ROOT,
                       help="Default: datasets/classified_datasets/.")
    p_cls.add_argument("--link", action="store_true",
                       help="dirs mode: hard-link frames instead of copying.")
    p_cls.add_argument("--dry-run", action="store_true",
                       help="Report the class distribution without writing.")
    p_cls.set_defaults(func=cmd_classify)

    p_train = sub.add_parser("train", help="Fine-tune MedGemma with QLoRA.")
    p_train.add_argument("--output-dir", required=True, help="Checkpoint/adapter destination.")
    p_train.add_argument("--structure", default="LV", choices=["LV", "LA"])
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--learning-rate", type=float, default=2e-4)
    p_train.add_argument("--train-samples", type=int, default=1000)
    p_train.add_argument("--val-samples", type=int, default=200)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Score predictions against ground truth.")
    p_eval.add_argument(
        "--adapter", default="camus", help=f"One of {sorted(ADAPTERS)}, a repo id, or a local path."
    )
    p_eval.add_argument("--structure", default="LV", choices=["LV", "LA"])
    p_eval.add_argument("--split", default="test", help="Dataset split; use 'all' for every frame.")
    p_eval.add_argument("--source", default=None, help="Filter by camus or echonet.")
    p_eval.add_argument("--max-samples", type=int, default=50)
    p_eval.add_argument("--prompt-variant", default=None, choices=["training", "generic"])
    p_eval.add_argument(
        "--figures",
        type=int,
        default=3,
        help="Best/worst prediction figures to render per band (notebook default 3; 0 disables).",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_corpus = sub.add_parser(
        "export-corpus",
        help="Turn revised contours into a trainable dataset (frames/ + tracings.json + metadata.csv).",
    )
    p_corpus.add_argument("--out", required=True, help="Destination directory for the corpus.")
    p_corpus.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="REVISION_ID",
        help="Export only this revision; repeatable. Default: every revision.",
    )
    p_corpus.add_argument(
        "--split", default="train", help="Split label for every exported frame (default: train)."
    )
    p_corpus.set_defaults(func=cmd_export_corpus)

    p_pub = sub.add_parser("publish-adapter", help="Upload a trained adapter to Hugging Face.")
    p_pub.add_argument("--folder", required=True, help="Checkpoint folder to upload from.")
    p_pub.add_argument("--repo", required=True, help="Target repo id, 'owner/name'.")
    p_pub.add_argument(
        "--public", action="store_true", help="Do not gate the repository (gated by default)."
    )
    p_pub.add_argument("--all-files", action="store_true", help="Upload every file in the folder.")
    p_pub.add_argument("--yes", action="store_true", help="Confirm the upload.")
    p_pub.set_defaults(func=cmd_publish)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # `atria` with no arguments serves the app, which is the click-and-run path.
    if not argv:
        argv = ["serve"]

    parser = build_parser()
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.dataset_dir:
        overrides["dataset_dir"] = args.dataset_dir
    if args.global_output_dir:
        overrides["output_dir"] = args.global_output_dir
    if args.log_level:
        overrides["log_level"] = args.log_level

    # Mirror overrides into the environment so that `serve --reload`, which runs the
    # app in a uvicorn subprocess, inherits them too.
    for key, value in overrides.items():
        os.environ[f"ATRIA_{key.upper()}"] = str(value)

    settings = default_settings.model_copy(update=overrides) if overrides else default_settings
    configure(settings.log_level)

    # The API resolves configuration through api.deps, so an override made here must
    # be visible to the routers as well.
    if overrides:
        from .api import deps

        deps.set_settings(settings)

    if args.command == "evaluate" and args.split == "all":
        args.split = None

    try:
        return int(args.func(args, settings) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
