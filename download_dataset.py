import argparse
from pathlib import Path

from itlp_campus.dataset import ITLPCampus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the data.")
    parser.add_argument("--outdoor", action="store_true", help="Download outdoor data.")
    parser.add_argument("--indoor", action="store_true", help="Download indoor data.")
    args = parser.parse_args()

    if not args.outdoor and not args.indoor:
        print("Please specify at least one of --outdoor or --indoor (or both).")

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        print(f"Creating output directory: {out_dir}")
        out_dir.mkdir(parents=True)
    else:
        print(f"Will download in existing directory: {out_dir}")

    if args.outdoor:
        print("Downloading outdoor data...")
        ITLPCampus.download_outdoor_data(out_dir=out_dir / "itlp_campus_outdoor")
    if args.indoor:
        print("Downloading indoor data...")
        ITLPCampus.download_indoor_data(out_dir=out_dir / "itlp_campus_indoor")
    print("Done.")
