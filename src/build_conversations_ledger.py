from pathlib import Path

import pandas as pd

from evidence import (
    build_prentus_name_index,
    find_prentus_evidence,
    load_prentus_contacts,
)
from utils import clean_text, require_columns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONVERSATIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "linkedin_conversations_filtered.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "master"
    / "conversations_ledger.csv"
)


REQUIRED_COLUMNS = {
    "CONVERSATION ID",
    "First_Message_Date",
    "Last_Message_Date",
    "Activity Type",
    "Contact Name",
    "Platform",
    "Messages_Sent",
    "Messages_Received",
    "Total_Messages",
    "Interaction Status",
    "Contact Profile URL",
}


def parse_integer(value: object) -> int:
    """
    Convert a numeric CSV value into an integer.
    """
    if pd.isna(value):
        return 0

    text = clean_text(value)

    if not text:
        return 0

    return int(float(text))


def build_ledger_record(
    row: pd.Series,
    record_number: int,
    prentus_contacts: pd.DataFrame,
    prentus_name_index: dict[str, list[int]],
) -> dict[str, object]:
    """
    Convert one processed LinkedIn conversation into one
    canonical conversation-ledger record.
    """
    contact_name = clean_text(
        row.get("Contact Name", "")
    )

    prentus_evidence = find_prentus_evidence(
        contact_name,
        prentus_contacts,
        prentus_name_index,
    )

    prentus_found = bool(
        prentus_evidence.get(
            "Prentus Evidence Found",
            False,
        )
    )

    evidence_status = (
        "LinkedIn + Prentus"
        if prentus_found
        else "LinkedIn Only"
    )

    return {
        "Conversation Record ID": (
            f"CONV-{record_number:04d}"
        ),
        "LinkedIn Conversation ID": clean_text(
            row.get("CONVERSATION ID", "")
        ),
        "First Message Date": clean_text(
            row.get("First_Message_Date", "")
        ),
        "Last Message Date": clean_text(
            row.get("Last_Message_Date", "")
        ),
        "Activity Type": clean_text(
            row.get("Activity Type", "")
        ),
        "Contact Name": contact_name,
        "Platform": clean_text(
            row.get("Platform", "")
        ),
        "Messages Sent": parse_integer(
            row.get("Messages_Sent", 0)
        ),
        "Messages Received": parse_integer(
            row.get("Messages_Received", 0)
        ),
        "Total Messages": parse_integer(
            row.get("Total_Messages", 0)
        ),
        "Interaction Status": clean_text(
            row.get("Interaction Status", "")
        ),
        "Contact Profile URL": clean_text(
            row.get("Contact Profile URL", "")
        ),
        "Prentus Evidence Found": prentus_found,
        "Evidence Status": evidence_status,
        "Prentus Evidence Method": clean_text(
            prentus_evidence.get(
                "Prentus Evidence Method",
                "",
            )
        ),
        "Prentus Evidence Confidence": clean_text(
            prentus_evidence.get(
                "Prentus Evidence Confidence",
                "",
            )
        ),
        "Prentus Candidate Count": (
            prentus_evidence.get(
                "Prentus Candidate Count",
                0,
            )
        ),
        "Prentus Name": clean_text(
            prentus_evidence.get(
                "Prentus Name",
                "",
            )
        ),
        "Prentus Outreach Date": clean_text(
            prentus_evidence.get(
                "Prentus Outreach Date",
                "",
            )
        ),
        "Prentus Outreach Date Source": clean_text(
            prentus_evidence.get(
                "Prentus Outreach Date Source",
                "",
            )
        ),
        "Prentus Message Date": clean_text(
            prentus_evidence.get(
                "Prentus Message Date",
                "",
            )
        ),
        "Prentus Connected Date": clean_text(
            prentus_evidence.get(
                "Prentus Connected Date",
                "",
            )
        ),
        "Exact Connection Date": clean_text(
            prentus_evidence.get(
                "Exact Connection Date",
                "",
            )
        ),
        "Exact Connection Date Source": clean_text(
            prentus_evidence.get(
                "Exact Connection Date Source",
                "",
            )
        ),
        "Prentus Record Created Date": clean_text(
            prentus_evidence.get(
                "Prentus Record Created Date",
                "",
            )
        ),
        "Prentus Endorsed Date": clean_text(
            prentus_evidence.get(
                "Prentus Endorsed Date",
                "",
            )
        ),
        "Prentus Status": clean_text(
            prentus_evidence.get(
                "Prentus Status",
                "",
            )
        ),
        "Prentus Type": clean_text(
            prentus_evidence.get(
                "Prentus Type",
                "",
            )
        ),
        "Prentus Company": clean_text(
            prentus_evidence.get(
                "Prentus Company",
                "",
            )
        ),
        "Prentus Job Title": clean_text(
            prentus_evidence.get(
                "Prentus Job Title",
                "",
            )
        ),
        "Prentus LinkedIn URL": clean_text(
            prentus_evidence.get(
                "Prentus LinkedIn URL",
                "",
            )
        ),
        "Prentus Email": clean_text(
            prentus_evidence.get(
                "Prentus Email",
                "",
            )
        ),
        "Prentus UID": clean_text(
            prentus_evidence.get(
                "Prentus UID",
                "",
            )
        ),
        "Source File": (
            "linkedin_conversations_filtered.csv"
        ),
    }


def main() -> None:
    if not CONVERSATIONS_FILE.exists():
        raise FileNotFoundError(
            "Could not find processed conversations file: "
            f"{CONVERSATIONS_FILE}"
        )

    conversations = pd.read_csv(
        CONVERSATIONS_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    require_columns(
        conversations,
        REQUIRED_COLUMNS,
        "LinkedIn conversations file",
    )

    prentus_contacts = load_prentus_contacts()

    prentus_name_index = build_prentus_name_index(
        prentus_contacts
    )

    ledger_records = [
        build_ledger_record(
            row=row,
            record_number=record_number,
            prentus_contacts=prentus_contacts,
            prentus_name_index=prentus_name_index,
        )
        for record_number, (_, row) in enumerate(
            conversations.iterrows(),
            start=1,
        )
    ]

    ledger = pd.DataFrame(ledger_records)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ledger.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    interaction_counts = (
        ledger["Interaction Status"]
        .value_counts()
    )

    evidence_counts = (
        ledger["Evidence Status"]
        .value_counts()
    )

    evidence_method_counts = (
        ledger["Prentus Evidence Method"]
        .value_counts()
    )

    print(
        "LinkedIn conversation records loaded: "
        f"{len(conversations)}"
    )

    print(
        "\nConversation ledger rows created: "
        f"{len(ledger)}"
    )

    print("\nInteraction status:")
    print(interaction_counts.to_string())

    print("\nEvidence status:")
    print(evidence_counts.to_string())

    print("\nPrentus evidence methods:")
    print(evidence_method_counts.to_string())

    print(
        "\nSaved conversations ledger to: "
        f"{OUTPUT_FILE}"
    )

    print("\nFirst ten conversation records:")
    print(
        ledger.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()