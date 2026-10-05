import argparse
import json

from .config import load_config
from .pipeline import run_video


def main():
    parser = argparse.ArgumentParser(description="Detect, track, and count vehicles in a local video")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="A new directory for this run")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.max_frames is not None:
            config.max_frames = args.max_frames
        result = run_video(args.input, args.output, config, progress=lambda n: print(f"Processed {n} frames", flush=True))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result["counts"], indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
