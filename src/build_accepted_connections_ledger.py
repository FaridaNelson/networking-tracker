from pathlib import Path

import pandas as pd

from evidence import (
    build_prentus_name_index,
    find_prentus_evidence,
    load_prentus_contacts,
)
from utils import clean_text, require_columns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LINKEDIN_CONNECTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "linkedin_connections_filtered.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "master"
    / "accepted_connections_ledger.csv"
)


LINKEDIN_REQUIRED_COLUMNS = {
    "Connected On",
    "Activity Type",
    "Name",
    "Company",
    "Position",
    "Platform",
    "URL",
    "Email Address",
}


def load_linkedin_connections() -> pd.DataFrame:
    """
    Load the canonical accepted LinkedIn connections dataset.
    """
    if not LINKEDIN_CONNECTIONS_FILE.exists():
        raise FileNotFoundError(
            "Could not find LinkedIn connections: "
            f"{LINKEDIN_CONNECTIONS_FILE}"
        )

    connections = pd.read_csv(
        LINKEDIN_CONNECTIONS_FILE,
        dtype=str,
        keep_default_na=False,
    )

    require_columns(
        connections,
        LINKEDIN_REQUIRED_COLUMNS,
        "LinkedIn accepted connections file",
    )

    return connections


def parse_boolean(value: object) -> bool:
    return clean_text(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def build_ledger_record(
    connection: pd.Series,
    evidence: dict[str, object],
    record_number: int,
) -> dict[str, object]:
    """
    Build one accepted-connection ledger record.

    LinkedIn's Connected On field remains the authoritative
    accepted-connection date.
    """
    prentus_found = parse_boolean(
        evidence.get(
            "Prentus Evidence Found",
            False,
        )
    )

    if prentus_found:
        evidence_status = "LinkedIn + Prentus"
        evidence_sources = (
            "LinkedIn Connection | Prentus"
        )
    else:
        evidence_status = "LinkedIn Only"
        evidence_sources = "LinkedIn Connection"

    return {
        "Accepted Connection ID": (
            f"CONN-{record_number:04d}"
        ),
        "Connected On": clean_text(
            connection.get("Connected On", "")
        ),
        "Connected Date Source": (
            "LinkedIn Connections Export"
        ),
        "Activity Type": clean_text(
            connection.get("Activity Type", "")
        ),
        "Name": clean_text(
            connection.get("Name", "")
        ),
        "Company": clean_text(
            connection.get("Company", "")
        ),
        "Position": clean_text(
            connection.get("Position", "")
        ),
        "Platform": clean_text(
            connection.get("Platform", "")
        ),
        "LinkedIn URL": clean_text(
            connection.get("URL", "")
        ),
        "Email Address": clean_text(
            connection.get("Email Address", "")
        ),
        "Evidence Status": evidence_status,
        "Evidence Sources": evidence_sources,
        **evidence,
    }


def main() -> None:
    connections = load_linkedin_connections()
    contacts = load_prentus_contacts()

    name_index = build_prentus_name_index(
        contacts
    )

    ledger_records: list[dict[str, object]] = []

    for record_number, (_, connection) in enumerate(
        connections.iterrows(),
        start=1,
    ):
        evidence = find_prentus_evidence(
            name=connection.get("Name", ""),
            contacts=contacts,
            name_index=name_index,
        )

        ledger_records.append(
            build_ledger_record(
                connection=connection,
                evidence=evidence,
                record_number=record_number,
            )
        )

    ledger = pd.DataFrame(ledger_records)

    if len(ledger) != len(connections):
        raise RuntimeError(
            "Row-preservation check failed: "
            f"{len(connections)} LinkedIn rows became "
            f"{len(ledger)} ledger rows."
        )

    duplicate_ids = int(
        ledger[
            "Accepted Connection ID"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_ids:
        raise RuntimeError(
            "Accepted Connection ID validation failed: "
            f"{duplicate_ids} duplicate IDs found."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ledger.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    evidence_counts = (
        ledger["Evidence Status"]
        .value_counts()
    )

    print(
        "LinkedIn accepted connections loaded: "
        f"{len(connections)}"
    )

    print(
        "Accepted connection ledger rows created: "
        f"{len(ledger)}"
    )

    print("\nEvidence status:")
    print(evidence_counts.to_string())

    print(
        "\nSaved accepted-connections ledger to: "
        f"{OUTPUT_FILE}"
    )

    preview_columns = [
        "Accepted Connection ID",
        "Connected On",
        "Name",
        "Company",
        "Position",
        "Evidence Status",
    ]

    print("\nFirst ten accepted connections:")
    print(
        ledger[preview_columns]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()