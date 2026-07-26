from pathlib import Path

import pandas as pd

from utils import clean_text, normalize_name, require_columns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LINKEDIN_PENDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "linkedin_pending_invitations.csv"
)

PRENTUS_CONTACTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prentus_contacts.csv"
)

EVIDENCE_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "linkedin_pending_with_prentus_evidence.csv"
)

REVIEW_OUTPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "prentus_evidence_review.csv"
)


LINKEDIN_REQUIRED_COLUMNS = {
    "Name",
    "Relative Date",
    "Approximate Date",
    "Connection Note",
}

PRENTUS_REQUIRED_COLUMNS = {
    "Name",
    "Outreach Date",
    "Outreach Date Source",
    "connectedDate",
    "messageDate",
    "createdDate",
    "endorsedDate",
    "status",
    "type",
}


def load_linkedin_pending() -> pd.DataFrame:
    """
    Load the canonical LinkedIn pending-invitations dataset.
    """
    if not LINKEDIN_PENDING_FILE.exists():
        raise FileNotFoundError(
            "Could not find LinkedIn pending invitations: "
            f"{LINKEDIN_PENDING_FILE}"
        )

    invitations = pd.read_csv(
        LINKEDIN_PENDING_FILE,
        dtype=str,
        keep_default_na=False,
    )

    require_columns(
        invitations,
        LINKEDIN_REQUIRED_COLUMNS,
        "LinkedIn pending invitations file",
    )

    return invitations


def load_prentus_contacts() -> pd.DataFrame:
    """
    Load the processed Prentus contacts dataset.
    """
    if not PRENTUS_CONTACTS_FILE.exists():
        raise FileNotFoundError(
            "Could not find processed Prentus contacts: "
            f"{PRENTUS_CONTACTS_FILE}"
        )

    contacts = pd.read_csv(
        PRENTUS_CONTACTS_FILE,
        dtype=str,
        keep_default_na=False,
    )

    require_columns(
        contacts,
        PRENTUS_REQUIRED_COLUMNS,
        "Prentus contacts file",
    )

    return contacts


def build_prentus_name_index(
    contacts: pd.DataFrame,
) -> dict[str, list[int]]:
    """
    Build a lookup from normalized names to Prentus row indices.

    Multiple indices are retained so duplicate names can be flagged
    for review rather than resolved automatically.
    """
    name_index: dict[str, list[int]] = {}

    for row_index, normalized_name in contacts[
        "Normalized Name"
    ].items():
        if not normalized_name:
            continue

        name_index.setdefault(
            normalized_name,
            [],
        ).append(row_index)

    return name_index


def empty_evidence_fields(
    candidate_count: int = 0,
) -> dict[str, object]:
    """
    Return blank evidence fields for a LinkedIn invitation.
    """
    return {
        "Prentus Evidence Found": False,
        "Prentus Evidence Source": "",
        "Prentus Evidence Method": "",
        "Prentus Evidence Confidence": "",
        "Prentus Candidate Count": candidate_count,
        "Prentus Name": "",
        "Prentus Outreach Date": "",
        "Prentus Outreach Date Source": "",
        "Prentus Message Date": "",
        "Prentus Connected Date": "",
        "Exact Connection Date": "",
        "Exact Connection Date Source": "",
        "Prentus Record Created Date": "",
        "Prentus Endorsed Date": "",
        "Prentus Status": "",
        "Prentus Type": "",
        "Prentus Company": "",
        "Prentus Job Title": "",
        "Prentus LinkedIn URL": "",
        "Prentus Email": "",
        "Prentus UID": "",
    }


def make_prentus_evidence_fields(
    contact: pd.Series,
) -> dict[str, object]:
    """
    Convert one Prentus contact into supporting evidence fields.

    For this project, Prentus messageDate is treated as the exact
    connection date when connectedDate is unavailable because the
    initial message was entered when the connection was added.
    """
    evidence = empty_evidence_fields(
        candidate_count=1,
    )

    connected_date = clean_text(
        contact.get("connectedDate", "")
    )

    message_date = clean_text(
        contact.get("messageDate", "")
    )

    if connected_date:
        exact_connection_date = connected_date
        exact_connection_date_source = (
            "Prentus connectedDate"
        )
    elif message_date:
        exact_connection_date = message_date
        exact_connection_date_source = (
            "Prentus messageDate"
        )
    else:
        exact_connection_date = ""
        exact_connection_date_source = ""

    evidence.update(
        {
            "Prentus Evidence Found": True,
            "Prentus Evidence Source": "Prentus",
            "Prentus Evidence Method": (
                "EXACT_NORMALIZED_NAME"
            ),
            "Prentus Evidence Confidence": "HIGH",
            "Prentus Name": clean_text(
                contact.get("Name", "")
            ),
            "Prentus Outreach Date": clean_text(
                contact.get("Outreach Date", "")
            ),
            "Prentus Outreach Date Source": clean_text(
                contact.get(
                    "Outreach Date Source",
                    "",
                )
            ),
            "Prentus Message Date": message_date,
            "Prentus Connected Date": connected_date,
            "Exact Connection Date": (
                exact_connection_date
            ),
            "Exact Connection Date Source": (
                exact_connection_date_source
            ),
            "Prentus Record Created Date": clean_text(
                contact.get("createdDate", "")
            ),
            "Prentus Endorsed Date": clean_text(
                contact.get("endorsedDate", "")
            ),
            "Prentus Status": clean_text(
                contact.get("status", "")
            ),
            "Prentus Type": clean_text(
                contact.get("type", "")
            ),
            "Prentus Company": clean_text(
                contact.get("companyName", "")
            ),
            "Prentus Job Title": clean_text(
                contact.get("jobTitle", "")
            ),
            "Prentus LinkedIn URL": clean_text(
                contact.get("linkedin", "")
            ),
            "Prentus Email": clean_text(
                contact.get("email", "")
            ),
            "Prentus UID": clean_text(
                contact.get("uid", "")
            ),
        }
    )

    return evidence


def make_review_record(
    invitation: pd.Series,
    reason: str,
    candidate_count: int,
    candidate_names: list[str] | None = None,
) -> dict[str, object]:
    """
    Create a human-review row without modifying the canonical record.
    """
    return {
        "LinkedIn Name": clean_text(
            invitation.get("Name", "")
        ),
        "LinkedIn Headline": clean_text(
            invitation.get("Headline", "")
        ),
        "LinkedIn Relative Date": clean_text(
            invitation.get("Relative Date", "")
        ),
        "LinkedIn Approximate Date": clean_text(
            invitation.get("Approximate Date", "")
        ),
        "Connection Note": clean_text(
            invitation.get("Connection Note", "")
        ),
        "Review Reason": reason,
        "Prentus Candidate Count": candidate_count,
        "Prentus Candidate Names": " | ".join(
            candidate_names or []
        ),
    }


def main() -> None:
    invitations = load_linkedin_pending()
    contacts = load_prentus_contacts()

    invitations["Normalized Name"] = (
        invitations["Name"].map(normalize_name)
    )

    contacts["Normalized Name"] = (
        contacts["Name"].map(normalize_name)
    )

    prentus_name_index = build_prentus_name_index(
        contacts
    )

    evidence_records: list[dict[str, object]] = []
    review_records: list[dict[str, object]] = []

    for _, invitation in invitations.iterrows():
        linkedin_record = invitation.to_dict()

        normalized_name = invitation[
            "Normalized Name"
        ]

        candidate_indices = prentus_name_index.get(
            normalized_name,
            [],
        )

        if not normalized_name:
            evidence_fields = empty_evidence_fields()

            review_records.append(
                make_review_record(
                    invitation=invitation,
                    reason="Blank LinkedIn name",
                    candidate_count=0,
                )
            )

        elif len(candidate_indices) == 1:
            contact = contacts.loc[
                candidate_indices[0]
            ]

            evidence_fields = (
                make_prentus_evidence_fields(
                    contact
                )
            )

        elif len(candidate_indices) > 1:
            evidence_fields = empty_evidence_fields(
                candidate_count=len(
                    candidate_indices
                )
            )

            evidence_fields[
                "Prentus Evidence Method"
            ] = "DUPLICATE_NORMALIZED_NAME"

            evidence_fields[
                "Prentus Evidence Confidence"
            ] = "NONE"

            candidate_names = [
                clean_text(
                    contacts.loc[
                        candidate_index,
                        "Name",
                    ]
                )
                for candidate_index
                in candidate_indices
            ]

            review_records.append(
                make_review_record(
                    invitation=invitation,
                    reason=(
                        "Multiple Prentus contacts share "
                        "this normalized name"
                    ),
                    candidate_count=len(
                        candidate_indices
                    ),
                    candidate_names=candidate_names,
                )
            )

        else:
            evidence_fields = empty_evidence_fields()

            evidence_fields[
                "Prentus Evidence Method"
            ] = "NO_EXACT_NAME_EVIDENCE"

            evidence_fields[
                "Prentus Evidence Confidence"
            ] = "NONE"

            review_records.append(
                make_review_record(
                    invitation=invitation,
                    reason=(
                        "No exact normalized-name "
                        "evidence in Prentus"
                    ),
                    candidate_count=0,
                )
            )

        evidence_records.append(
            {
                **linkedin_record,
                **evidence_fields,
            }
        )

    evidence = pd.DataFrame(
        evidence_records
    )

    review = pd.DataFrame(
        review_records
    )

    evidence = evidence.drop(
        columns=["Normalized Name"],
        errors="ignore",
    )

    EVIDENCE_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REVIEW_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence.to_csv(
        EVIDENCE_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    review.to_csv(
        REVIEW_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    evidence_found_count = int(
        evidence["Prentus Evidence Found"].sum()
    )

    no_evidence_count = (
        len(evidence)
        - evidence_found_count
    )

    exact_connection_date_count = (
        evidence["Exact Connection Date"]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    exact_connection_date_source_counts = (
        evidence.loc[
            evidence[
                "Exact Connection Date Source"
            ]
            .astype(str)
            .str.strip()
            .ne(""),
            "Exact Connection Date Source",
        ]
        .value_counts()
    )

    print(
        "LinkedIn pending invitations: "
        f"{len(invitations)}"
    )

    print(
        "Prentus contacts available: "
        f"{len(contacts)}"
    )

    print(
        "\nPrentus evidence found: "
        f"{evidence_found_count}"
    )

    print(
        "No Prentus evidence found: "
        f"{no_evidence_count}"
    )

    print(
        "\nRows with an exact connection date: "
        f"{exact_connection_date_count}"
    )

    print("\nExact connection date sources:")

    if exact_connection_date_source_counts.empty:
        print("None")
    else:
        print(
            exact_connection_date_source_counts
            .to_string()
        )

    print(
        "\nSaved networking evidence to: "
        f"{EVIDENCE_OUTPUT_FILE}"
    )

    print(
        "Saved review records to: "
        f"{REVIEW_OUTPUT_FILE}"
    )

    preview_columns = [
        "Name",
        "Relative Date",
        "Approximate Date",
        "Prentus Evidence Found",
        "Exact Connection Date",
        "Exact Connection Date Source",
        "Prentus Status",
    ]

    print("\nFirst ten evidence records:")
    print(
        evidence[preview_columns]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()