from pathlib import Path

import pandas as pd

from utils import clean_text, normalize_name, require_columns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRENTUS_CONTACTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prentus_contacts.csv"
)


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


def load_prentus_contacts() -> pd.DataFrame:
    """
    Load and normalize the canonical Prentus contacts dataset.
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

    contacts = contacts.copy()

    contacts["Normalized Name"] = (
        contacts["Name"].map(normalize_name)
    )

    return contacts


def build_prentus_name_index(
    contacts: pd.DataFrame,
) -> dict[str, list[int]]:
    """
    Map each normalized Prentus name to its row indices.

    Multiple indices are retained so ambiguous duplicate-name
    matches are never resolved automatically.
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


def derive_exact_connection_date(
    contact: pd.Series,
) -> tuple[str, str]:
    """
    Derive the exact connection date from Prentus.

    For this project:
    1. connectedDate is preferred when present.
    2. messageDate is used otherwise because the initial Prentus
       message was entered when the connection was added.
    3. createdDate is not treated as an exact connection date.
    """
    connected_date = clean_text(
        contact.get("connectedDate", "")
    )

    message_date = clean_text(
        contact.get("messageDate", "")
    )

    if connected_date:
        return (
            connected_date,
            "Prentus connectedDate",
        )

    if message_date:
        return (
            message_date,
            "Prentus messageDate",
        )

    return "", ""


def empty_prentus_evidence(
    candidate_count: int = 0,
) -> dict[str, object]:
    """
    Return the standard blank Prentus evidence structure.
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


def make_prentus_evidence(
    contact: pd.Series,
) -> dict[str, object]:
    """
    Convert one exact Prentus contact match into evidence fields.
    """
    exact_date, exact_date_source = (
        derive_exact_connection_date(contact)
    )

    evidence = empty_prentus_evidence(
        candidate_count=1,
    )

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
            "Prentus Message Date": clean_text(
                contact.get("messageDate", "")
            ),
            "Prentus Connected Date": clean_text(
                contact.get("connectedDate", "")
            ),
            "Exact Connection Date": exact_date,
            "Exact Connection Date Source": (
                exact_date_source
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


def find_prentus_evidence(
    name: object,
    contacts: pd.DataFrame,
    name_index: dict[str, list[int]],
) -> dict[str, object]:
    """
    Find exact normalized-name Prentus evidence for one person.

    No fuzzy matching is performed.

    Outcomes:
    - one candidate: return exact-match evidence
    - zero candidates: return blank evidence
    - multiple candidates: return ambiguous-match metadata
    """
    normalized_name = normalize_name(name)

    if not normalized_name:
        evidence = empty_prentus_evidence()

        evidence["Prentus Evidence Method"] = (
            "BLANK_SOURCE_NAME"
        )

        evidence["Prentus Evidence Confidence"] = (
            "NONE"
        )

        return evidence

    candidate_indices = name_index.get(
        normalized_name,
        [],
    )

    if len(candidate_indices) == 1:
        contact = contacts.loc[
            candidate_indices[0]
        ]

        return make_prentus_evidence(contact)

    if len(candidate_indices) > 1:
        evidence = empty_prentus_evidence(
            candidate_count=len(candidate_indices)
        )

        evidence["Prentus Evidence Method"] = (
            "DUPLICATE_NORMALIZED_NAME"
        )

        evidence["Prentus Evidence Confidence"] = (
            "NONE"
        )

        return evidence

    evidence = empty_prentus_evidence()

    evidence["Prentus Evidence Method"] = (
        "NO_EXACT_NAME_EVIDENCE"
    )

    evidence["Prentus Evidence Confidence"] = (
        "NONE"
    )

    return evidence