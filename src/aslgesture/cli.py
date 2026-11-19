"""Command-line interface: ``asl-gesture <command>`` (or ``python -m aslgesture <command>``)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")  # quieten MediaPipe's C++ logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import config  # noqa: E402
from .features import FEATURE_SETS  # noqa: E402


def cmd_extract(args) -> None:
    from .landmarks import extract_dataset

    summary = extract_dataset(Path(args.raw_dir))
    config.ensure_dirs()
    (config.TABLE_DIR / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_train(args) -> None:
    from . import supervised

    summary = supervised.run(feature_sets=args.feature_sets, model_keys=args.models)
    print(summary.to_string(index=False))
    if args.export:
        key, fs = args.export
        print("Exported:", supervised.export_model(key, fs))


def cmd_export(args) -> None:
    from . import supervised

    print(json.dumps(supervised.export_model(args.model, args.feature_set), indent=2, default=str))


def cmd_cluster(args) -> None:
    from . import clustering

    print(clustering.run().to_string(index=False))


def cmd_robustness(args) -> None:
    from . import robustness

    res = robustness.run()
    pivot = res.pivot_table(index="condition", columns=["model", "feature_set"], values="accuracy", sort=False)
    print(pivot.round(3).to_string())


def cmd_demo(args) -> None:
    from .realtime import run_stream

    stats = run_stream(
        source=args.source, model_path=Path(args.model), tracking=not args.no_tracking,
        use_clahe=args.clahe, use_denoise=args.denoise, use_gamma=args.auto_gamma,
        smoothing=args.smoothing, show=not args.no_display,
        save=Path(args.save) if args.save else None, left_hand=args.left_hand,
        max_frames=args.max_frames,
    )
    stats.pop("labels", None)
    print(json.dumps(stats, indent=2))


def cmd_benchmark(args) -> None:
    import pandas as pd

    from .realtime import run_stream

    rows = []
    settings = [  # (tracking, auto-gamma, denoise)
        (False, False, False), (True, False, False), (True, True, False), (True, False, True), (True, True, True),
    ]
    for tracking, use_gamma, use_denoise in settings:
        s = run_stream(args.source, Path(args.model), tracking=tracking, use_gamma=use_gamma,
                       use_denoise=use_denoise, show=False, max_frames=args.frames)
        s.pop("labels", None)
        rows.append(s)
        print(f"tracking={tracking!s:5} gamma={use_gamma!s:5} denoise={use_denoise!s:5} -> "
              f"{s['processing_fps']} FPS, {s['latency_ms_mean']} ms/frame")
    df = pd.DataFrame(rows)
    config.ensure_dirs()
    df.to_csv(config.TABLE_DIR / "realtime_benchmark.csv", index=False)
    print(df.to_string(index=False))


def cmd_predict(args) -> None:
    from .landmarks import imread
    from .realtime import GestureRecognizer

    rec = GestureRecognizer(Path(args.model), tracking=False, smoothing=1, min_confidence=0.0)
    try:
        for p in args.images:
            img = imread(p)
            if img is None:
                print(f"{p}: could not read image")
                continue
            rec.history.clear()
            pred = rec(img)
            print(f"{p}: " + (f"{pred.label} ({pred.confidence:.0%})" if pred.label else "no hand detected"))
    finally:
        rec.close()


def cmd_make_demo_video(args) -> None:
    from .realtime import make_demo_video

    images = sorted((config.EXAMPLES_DIR / "own_images").glob(f"*_{args.take}.jpg"))
    out = make_demo_video(Path(args.out), images, frames_per_image=args.frames_per_image)
    print(f"Wrote {out} ({len(images)} source photos)")


def cmd_all(args) -> None:
    cmd_extract(argparse.Namespace(raw_dir=str(config.RAW_DIR)))
    cmd_train(argparse.Namespace(feature_sets=list(FEATURE_SETS), models=["knn", "dt", "lr"], export=None))
    cmd_cluster(args)
    cmd_robustness(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="asl-gesture", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("extract", help="run MediaPipe over data/raw/<A-J>/ and write landmarks.csv")
    s.add_argument("--raw-dir", default=str(config.RAW_DIR))
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("train", help="5-fold CV tuning + test evaluation for every model / feature set")
    s.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS),
                   choices=list(FEATURE_SETS))
    s.add_argument("--models", nargs="+", default=["knn", "dt", "lr"], choices=["knn", "dt", "lr"])
    s.add_argument("--export", nargs=2, metavar=("MODEL", "FEATURE_SET"),
                   help="also export this trained model to models/ for the demo")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("export-model", help="copy a trained model into models/ for demo/predict")
    s.add_argument("--model", default="lr", choices=["knn", "dt", "lr"])
    s.add_argument("--feature-set", default="norm", choices=list(FEATURE_SETS))
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("cluster", help="K-Means and agglomerative clustering analysis")
    s.set_defaults(func=cmd_cluster)

    s = sub.add_parser("robustness", help="end-to-end accuracy under synthetic capture conditions")
    s.set_defaults(func=cmd_robustness)

    for name, func, helptext in (("demo", cmd_demo, "real-time recognition from a webcam or video"),
                                 ("benchmark", cmd_benchmark, "measure FPS / latency on a video")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--source", default="0", help="camera index (default 0) or video file path")
        s.add_argument("--model", default=str(config.DEFAULT_MODEL))
        if name == "demo":
            s.add_argument("--no-tracking", action="store_true", help="re-detect the hand on every frame")
            s.add_argument("--clahe", action="store_true", help="CLAHE lighting equalisation before detection")
            s.add_argument("--denoise", action="store_true", help="median-filter denoising before detection")
            s.add_argument("--auto-gamma", action="store_true", help="adaptive gamma brightening for dark scenes")
            s.add_argument("--smoothing", type=int, default=5, help="frames averaged for a stable prediction")
            s.add_argument("--save", help="write the annotated video to this .mp4 path")
            s.add_argument("--no-display", action="store_true")
            s.add_argument("--left-hand", action="store_true", help="mirror frames so a left hand is recognised")
            s.add_argument("--max-frames", type=int)
        else:
            s.add_argument("--frames", type=int, default=None)
        s.set_defaults(func=func)

    s = sub.add_parser("predict", help="classify one or more image files")
    s.add_argument("images", nargs="+")
    s.add_argument("--model", default=str(config.DEFAULT_MODEL))
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("make-demo-video", help="animate examples/own_images into a camera-like clip")
    s.add_argument("--out", default=str(config.DATA_DIR / "demo_input.mp4"))
    s.add_argument("--take", default="1", help="which photo per letter to use (1, 2 or 3)")
    s.add_argument("--frames-per-image", type=int, default=40)
    s.set_defaults(func=cmd_make_demo_video)

    s = sub.add_parser("all", help="extract -> train -> cluster -> robustness")
    s.set_defaults(func=cmd_all)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
