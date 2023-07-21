import argparse
from pathlib import Path

from src.dataset import ITLPCampus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the data.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        print(f"Creating output directory: {out_dir}")
        out_dir.mkdir(parents=True)
    else:
        print(f"Will download in existing directory: {out_dir}")

    ITLPCampus.download_data(out_dir=out_dir)

