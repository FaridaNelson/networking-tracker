from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONNECTIONS_FILE = (
    PROJECT_ROOT / "data" / "raw" / "linkedin" / "Connections.csv"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "linkedin_connections_filtered.csv"
)

START_DATE = pd.Timestamp("2025-12-01")
END_DATE = pd.Timestamp("2026-07-24")


def main() -> None:
    if not CONNECTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find LinkedIn connections file: {CONNECTIONS_FILE}"
        )

    connections = pd.read_csv(
        CONNECTIONS_FILE,
        skiprows=3,
        encoding="utf-8-sig",
    )

    connections["Connected On"] = pd.to_datetime(
        connections["Connected On"],
        format="%d %b %Y",
        errors="coerce",
    )

    invalid_dates = connections["Connected On"].isna().sum()

    filtered = connections.loc[
        connections["Connected On"].between(
            START_DATE,
            END_DATE,
            inclusive="both",
        )
    ].copy()

    filtered["Name"] = (
        filtered["First Name"].fillna("").str.strip()
        + " "
        + filtered["Last Name"].fillna("").str.strip()
    ).str.strip()

    blank_profile_mask = (
        filtered["Name"].eq("")
        & filtered["URL"].fillna("").str.strip().eq("")
        & filtered["Email Address"].fillna("").str.strip().eq("")
    )

    blank_profiles_removed = int(blank_profile_mask.sum())

    filtered = filtered.loc[~blank_profile_mask].copy()

    filtered["Platform"] = "LinkedIn"
    filtered["Activity Type"] = "Connection"

    filtered = filtered[
        [
            "Connected On",
            "Activity Type",
            "Name",
            "Company",
            "Position",
            "Platform",
            "URL",
            "Email Address",
        ]
    ]

    filtered = filtered.sort_values(
        by="Connected On",
        ascending=True,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    filtered.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print(f"Total LinkedIn connections: {len(connections)}")
    print(f"Rows with invalid dates: {invalid_dates}")
    print(f"Blank profile rows removed: {blank_profiles_removed}")
    print(
        f"Connections from {START_DATE.date()} "
        f"through {END_DATE.date()}: {len(filtered)}"
    )
    print(f"Saved filtered data to: {OUTPUT_FILE}")

    print("\nFirst five filtered rows:")
    print(filtered.head().to_string(index=False))

    print("\nLast five filtered rows:")
    print(filtered.tail().to_string(index=False))


if __name__ == "__main__":
    main()