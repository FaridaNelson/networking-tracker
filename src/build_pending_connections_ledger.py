from pathlib import Path

import pandas as pd

from utils import clean_text, require_columns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVIDENCE_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "linkedin_pending_with_prentus_evidence.csv"
)

LEDGER_OUTPUT_FILE = (
    PROJECT_ROOT
    / "master"
    / "pending_connections_ledger.csv"
)


REQUIRED_COLUMNS = {
    "Name",
    "Relative Date",
    "Approximate Date",
    "Connection Note",
    "Prentus Evidence Found",
    "Prentus Evidence Method",
    "Exact Connection Date",
    "Exact Connection Date Source",
    "Prentus Status",
    "Prentus Type",
    "Prentus UID",
}


def load_evidence() -> pd.DataFrame:
    """
    Load the LinkedIn-first networking evidence dataset.

    Every input row represents one LinkedIn pending invitation.
    Prentus fields are supplemental evidence only.
    """
    if not EVIDENCE_INPUT_FILE.exists():
        raise FileNotFoundError(
            "Could not find the networking evidence file: "
            f"{EVIDENCE_INPUT_FILE}\n"
            "Run src/build_networking_evidence.py first."
        )

    evidence = pd.read_csv(
        EVIDENCE_INPUT_FILE,
        dtype=str,
        keep_default_na=False,
    )

    require_columns(
        evidence,
        REQUIRED_COLUMNS,
        "LinkedIn networking evidence file",
    )

    return evidence


def text_value(
    row: pd.Series,
    column: str,
) -> str:
    """
    Safely retrieve and clean an optional text field.
    """
    return clean_text(row.get(column, ""))


def parse_boolean(value: object) -> bool:
    """
    Interpret common CSV boolean representations.
    """
    return clean_text(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def choose_activity_date(
    row: pd.Series,
) -> tuple[str, str, str]:
    """
    Choose the best available date without overwriting provenance.

    Priority:
    1. Exact connection date derived from Prentus
    2. LinkedIn approximate date

    Returns:
        activity_date,
        date_precision,
        date_source
    """
    exact_date = text_value(
        row,
        "Exact Connection Date",
    )

    exact_source = text_value(
        row,
        "Exact Connection Date Source",
    )

    if exact_date:
        return (
            exact_date,
            "Exact",
            exact_source,
        )

    approximate_date = text_value(
        row,
        "Approximate Date",
    )

    if approximate_date:
        return (
            approximate_date,
            "Approximate",
            "LinkedIn relative-date calculation",
        )

    return (
        "",
        "Unknown",
        "",
    )


def build_ledger_record(
    row: pd.Series,
    record_number: int,
) -> dict[str, object]:
    """
    Build one presentation-ready pending-connection ledger record.
    """
    prentus_found = parse_boolean(
        row.get("Prentus Evidence Found", "")
    )

    activity_date, date_precision, date_source = (
        choose_activity_date(row)
    )

    if prentus_found:
        evidence_status = "LinkedIn + Prentus"
        evidence_sources = (
            "LinkedIn Pending Invitation | Prentus"
        )
    else:
        evidence_status = "LinkedIn Only"
        evidence_sources = (
            "LinkedIn Pending Invitation"
        )

    return {
        "Pending Connection ID": (
            f"PEND-{record_number:04d}"
        ),
        "Name": text_value(row, "Name"),
        "Headline": text_value(row, "Headline"),
        "Company": (
            text_value(row, "Prentus Company")
            or text_value(row, "Company")
        ),
        "LinkedIn URL": text_value(
            row,
            "Prentus LinkedIn URL",
        ),
        "Activity Type": "Pending Invitation",
        "Activity Date": activity_date,
        "Date Precision": date_precision,
        "Date Source": date_source,
        "LinkedIn Relative Date": text_value(
            row,
            "Relative Date",
        ),
        "LinkedIn Approximate Date": text_value(
            row,
            "Approximate Date",
        ),
        "Exact Connection Date": text_value(
            row,
            "Exact Connection Date",
        ),
        "Exact Connection Date Source": text_value(
            row,
            "Exact Connection Date Source",
        ),
        "Connection Note": text_value(
            row,
            "Connection Note",
        ),
        "Evidence Status": evidence_status,
        "Evidence Sources": evidence_sources,
        "Prentus Evidence Method": text_value(
            row,
            "Prentus Evidence Method",
        ),
        "Prentus Status": text_value(
            row,
            "Prentus Status",
        ),
        "Prentus Type": text_value(
            row,
            "Prentus Type",
        ),
        "Prentus UID": text_value(
            row,
            "Prentus UID",
        ),
    }


def main() -> None:
    evidence = load_evidence()

    ledger_records = [
        build_ledger_record(
            row=row,
            record_number=index,
        )
        for index, (_, row) in enumerate(
            evidence.iterrows(),
            start=1,
        )
    ]

    ledger = pd.DataFrame(ledger_records)

    if len(ledger) != len(evidence):
        raise RuntimeError(
            "Row-preservation check failed: "
            f"{len(evidence)} evidence rows became "
            f"{len(ledger)} ledger rows."
        )

    duplicate_ids = int(
        ledger["Pending Connection ID"]
        .duplicated()
        .sum()
    )

    if duplicate_ids:
        raise RuntimeError(
            "Pending connection ID validation failed: "
            f"{duplicate_ids} duplicate IDs found."
        )

    LEDGER_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ledger.to_csv(
        LEDGER_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    evidence_status_counts = (
        ledger["Evidence Status"]
        .value_counts()
    )

    date_precision_counts = (
        ledger["Date Precision"]
        .value_counts()
    )

    print(
        "Pending invitation evidence rows loaded: "
        f"{len(evidence)}"
    )

    print(
        "Pending connection ledger rows created: "
        f"{len(ledger)}"
    )

    print("\nEvidence status:")
    print(
        evidence_status_counts.to_string()
    )

    print("\nDate precision:")
    print(
        date_precision_counts.to_string()
    )

    print(
        "\nSaved pending-connections ledger to: "
        f"{LEDGER_OUTPUT_FILE}"
    )

    preview_columns = [
        "Pending Connection ID",
        "Name",
        "Activity Type",
        "Activity Date",
        "Date Precision",
        "Evidence Status",
    ]

    print("\nFirst ten pending connection records:")
    print(
        ledger[preview_columns]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()