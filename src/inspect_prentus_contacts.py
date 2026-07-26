from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRENTUS_CONTACTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "prentus"
    / "contacts_export_2026-07-09.csv"
)


def main() -> None:
    if not PRENTUS_CONTACTS_FILE.exists():
        raise FileNotFoundError(
            "Could not find Prentus contacts export: "
            f"{PRENTUS_CONTACTS_FILE}"
        )

    contacts = pd.read_csv(
        PRENTUS_CONTACTS_FILE,
        dtype=str,
        keep_default_na=False,
    )

    print("Columns:")
    for column in contacts.columns:
        print(f"- {column}")

    print(f"\nTotal rows: {len(contacts)}")

    fields_to_inspect = [
        "connectedDate",
        "messageDate",
        "createdDate",
        "status",
        "type",
        "messages",
    ]

    for field in fields_to_inspect:
        if field not in contacts.columns:
            continue

        print(f"\nSample values for {field}:")
        values = (
            contacts[field]
            .loc[contacts[field].str.strip().ne("")]
            .drop_duplicates()
            .head(10)
        )

        if values.empty:
            print("(no nonblank values)")
        else:
            for value in values:
                print(repr(value))

    preview_columns = [
        column
        for column in [
            "firstName",
            "lastName",
            "companyName",
            "jobTitle",
            "linkedin",
            "connectedDate",
            "messageDate",
            "status",
            "type",
        ]
        if column in contacts.columns
    ]

    print("\nFirst five rows:")
    print(
        contacts[preview_columns]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()