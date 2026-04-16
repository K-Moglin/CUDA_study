import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="assignment_test_cases.json")
    args = parser.parse_args()

    cases = [
        {"name": "tiny", "T_local": 4, "H": 8, "I": 16, "E": 4, "shared": 1, "topk": 2, "warmup": 1, "iters": 2},
        {"name": "small", "T_local": 8, "H": 16, "I": 32, "E": 4, "shared": 1, "topk": 2, "warmup": 1, "iters": 2},
        {"name": "medium", "T_local": 16, "H": 32, "I": 64, "E": 8, "shared": 1, "topk": 2, "warmup": 1, "iters": 2},
    ]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, indent=2)
    print(f"Wrote {args.output} with {len(cases)} cases")


if __name__ == "__main__":
    main()
