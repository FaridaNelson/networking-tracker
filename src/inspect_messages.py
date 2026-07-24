from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MESSAGES_FILE = (
    PROJECT_ROOT / "data" / "raw" / "linkedin" / "messages.csv"
)


def main() -> None:
    if not MESSAGES_FILE.exists():
        raise FileNotFoundError(
            f"Could not find LinkedIn messages file: {MESSAGES_FILE}"
        )

    messages = pd.read_csv(
        MESSAGES_FILE,
        encoding="utf-8-sig",
    )

    print("Columns:")
    print(messages.columns.tolist())

    print(f"\nTotal rows: {len(messages)}")

    print("\nFirst three rows:")
    print(messages.head(3).to_string(index=False))


if __name__ == "__main__":
    main()