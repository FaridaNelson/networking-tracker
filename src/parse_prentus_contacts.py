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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prentus_contacts.csv"
)

DATE_COLUMNS = [
    "connectedDate",
    "endorsedDate",
    "messageDate",
    "createdDate",
]


def clean_text_series(series: pd.Series) -> pd.Series:
    """Trim text fields and replace missing values with empty strings."""
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def parse_date_column(
    contacts: pd.DataFrame,
    column: str,
) -> None:
    """Parse a Prentus ISO timestamp column as UTC."""
    if column not in contacts.columns:
        return

    contacts[column] = pd.to_datetime(
        contacts[column],
        errors="coerce",
        utc=True,
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

    original_row_count = len(contacts)

    text_columns = [
        "companyName",
        "email",
        "firstName",
        "goal",
        "jobTitle",
        "lastName",
        "linkedin",
        "bio",
        "phone",
        "status",
        "statusTimeline",
        "type",
        "uid",
    ]

    for column in text_columns:
        if column in contacts.columns:
            contacts[column] = clean_text_series(
                contacts[column]
            )

    for column in DATE_COLUMNS:
        parse_date_column(
            contacts,
            column,
        )

    contacts["Name"] = (
        contacts["firstName"]
        + " "
        + contacts["lastName"]
    ).str.strip()

    contacts["messages"] = pd.to_numeric(
        contacts["messages"],
        errors="coerce",
    ).fillna(0).astype(int)

    blank_contact_mask = (
        contacts["Name"].eq("")
        & contacts["linkedin"].eq("")
        & contacts["email"].eq("")
    )

    blank_contacts_removed = int(
        blank_contact_mask.sum()
    )

    contacts = contacts.loc[
        ~blank_contact_mask
    ].copy()

    contacts["Outreach Date"] = (
        contacts["messageDate"]
        .combine_first(
            contacts["connectedDate"]
        )
        .combine_first(
            contacts["createdDate"]
        )
    )

    contacts["Outreach Date Source"] = ""

    contacts.loc[
        contacts["messageDate"].notna(),
        "Outreach Date Source",
    ] = "messageDate"

    contacts.loc[
        contacts["messageDate"].isna()
        & contacts["connectedDate"].notna(),
        "Outreach Date Source",
    ] = "connectedDate"

    contacts.loc[
        contacts["messageDate"].isna()
        & contacts["connectedDate"].isna()
        & contacts["createdDate"].notna(),
        "Outreach Date Source",
    ] = "createdDate"

    output_columns = [
        "Outreach Date",
        "Outreach Date Source",
        "Name",
        "firstName",
        "lastName",
        "companyName",
        "jobTitle",
        "linkedin",
        "email",
        "status",
        "type",
        "messages",
        "connectedDate",
        "messageDate",
        "endorsedDate",
        "createdDate",
        "goal",
        "bio",
        "phone",
        "statusTimeline",
        "uid",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in contacts.columns
    ]

    contacts = contacts[
        output_columns
    ].sort_values(
        by=[
            "Outreach Date",
            "Name",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    contacts.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )

    invalid_date_counts = {
        column: int(
            contacts[column].isna().sum()
        )
        for column in DATE_COLUMNS
        if column in contacts.columns
    }

    duplicate_name_rows = int(
        contacts["Name"].duplicated(
            keep=False
        ).sum()
    )

    print(
        f"Total Prentus rows: {original_row_count}"
    )
    print(
        "Blank contact rows removed: "
        f"{blank_contacts_removed}"
    )
    print(
        "Prentus contacts parsed: "
        f"{len(contacts)}"
    )
    print(
        "Rows belonging to duplicated names: "
        f"{duplicate_name_rows}"
    )

    print("\nMissing or invalid dates:")
    for column, count in invalid_date_counts.items():
        print(f"- {column}: {count}")

    print("\nStatus counts:")
    print(
        contacts["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nOutreach date source counts:")
    print(
        contacts["Outreach Date Source"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        f"\nSaved parsed contacts to: {OUTPUT_FILE}"
    )

    preview_columns = [
        "Outreach Date",
        "Outreach Date Source",
        "Name",
        "companyName",
        "jobTitle",
        "status",
        "type",
    ]

    print("\nFirst five parsed contacts:")
    print(
        contacts[preview_columns]
        .head()
        .to_string(index=False)
    )

    print("\nLast five parsed contacts:")
    print(
        contacts[preview_columns]
        .tail()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()