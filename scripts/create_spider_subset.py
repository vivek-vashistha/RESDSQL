import json
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Create a smaller Spider dev subset.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/spider/dev.json",
        help="Path to original Spider dev.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/spider/dev_100.json",
        help="Path to write subset dev file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of examples to keep from the start of the file",
    )
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.output)

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    subset = data[: args.limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(subset)} examples to {out}")


if __name__ == "__main__":
    main()


