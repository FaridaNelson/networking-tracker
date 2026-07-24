from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PENDING_INVITATIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "linkedin"
    / "pending_invitations.txt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "linkedin_pending_invitations.csv"
)

UI_ONLY_LINES = {
    "withdraw",
    "show less",
    "show more",
}

CAPTURE_DATE = pd.Timestamp("2026-07-24")

SENT_PATTERN = re.compile(
    r"^Sent\s+(.+)$",
    flags=re.IGNORECASE,
)

PROFILE_PICTURE_PATTERN = re.compile(
    r"[’']s profile picture$",
    flags=re.IGNORECASE,
)


def clean_line(value: str) -> str:
    """Normalize whitespace without changing the text itself."""
    return re.sub(r"\s+", " ", value).strip()


def is_profile_picture_line(value: str) -> bool:
    return bool(PROFILE_PICTURE_PATTERN.search(value))


def approximate_date(relative_date: str) -> pd.Timestamp | None:
    """
    Convert LinkedIn's relative date into an approximate date.

    The original relative text is preserved because month-based dates
    are estimates rather than exact historical dates.
    """
    value = relative_date.lower().strip()

    if value == "today":
        return CAPTURE_DATE

    if value == "yesterday":
        return CAPTURE_DATE - pd.Timedelta(days=1)

    hour_match = re.fullmatch(r"(\d+)\s+hours?\s+ago", value)
    if hour_match:
        return CAPTURE_DATE

    day_match = re.fullmatch(r"(\d+)\s+days?\s+ago", value)
    if day_match:
        days = int(day_match.group(1))
        return CAPTURE_DATE - pd.Timedelta(days=days)

    week_match = re.fullmatch(r"(\d+)\s+weeks?\s+ago", value)
    if week_match:
        weeks = int(week_match.group(1))
        return CAPTURE_DATE - pd.Timedelta(weeks=weeks)

    month_match = re.fullmatch(r"(\d+)\s+months?\s+ago", value)
    if month_match:
        months = int(month_match.group(1))
        return CAPTURE_DATE - pd.DateOffset(months=months)

    return None


def find_record_start(
    lines: list[str],
    previous_sent_index: int,
    current_sent_index: int,
) -> int:
    """
    Locate the beginning of the invitation record preceding a Sent line.

    Prefer LinkedIn's '<name>'s profile picture' marker. Some profiles do
    not include that marker, so fall back to the final two lines before
    the Sent line: name followed by headline.
    """
    search_start = previous_sent_index + 1

    for index in range(current_sent_index - 1, search_start - 1, -1):
        if is_profile_picture_line(lines[index]):
            return index

    nonempty_indices = [
        index
        for index in range(search_start, current_sent_index)
        if lines[index]
        and lines[index].casefold() not in UI_ONLY_LINES
    ]

    if len(nonempty_indices) >= 2:
        return nonempty_indices[-2]

    if nonempty_indices:
        return nonempty_indices[-1]

    return current_sent_index


def parse_identity_block(
    lines: list[str],
    start_index: int,
    sent_index: int,
) -> tuple[str, str]:
    """Extract the person's name and headline."""
    identity_lines = [
        lines[index]
        for index in range(start_index, sent_index)
        if lines[index]
        and lines[index].casefold() not in UI_ONLY_LINES
        and not is_profile_picture_line(lines[index])
    ]

    if not identity_lines:
        return "", ""

    name = identity_lines[0]
    headline = " ".join(identity_lines[1:])

    return name, headline

def find_footer_start(
    lines: list[str],
    search_start: int,
) -> int:
    """
    Find the beginning of LinkedIn's page footer.

    The footer begins with the consecutive lines:
    About, Accessibility, Help Center.
    """
    footer_sequence = [
        "About",
        "Accessibility",
        "Help Center",
    ]

    for index in range(
        search_start,
        len(lines) - len(footer_sequence) + 1,
    ):
        candidate = lines[
            index:index + len(footer_sequence)
        ]

        if candidate == footer_sequence:
            return index

    return len(lines)

def main() -> None:
    if not PENDING_INVITATIONS_FILE.exists():
        raise FileNotFoundError(
            "Could not find pending invitations file: "
            f"{PENDING_INVITATIONS_FILE}"
        )

    raw_text = PENDING_INVITATIONS_FILE.read_text(
        encoding="utf-8",
    )

    lines = [
        clean_line(line)
        for line in raw_text.splitlines()
    ]

    sent_indices = [
        index
        for index, line in enumerate(lines)
        if SENT_PATTERN.fullmatch(line)
    ]

    if not sent_indices:
        raise ValueError(
            "No LinkedIn 'Sent ...' lines were detected in "
            f"{PENDING_INVITATIONS_FILE}"
        )

    content_end_index = find_footer_start(
        lines,
        sent_indices[-1] + 1,
    )

    record_starts: list[int] = []

    for position, sent_index in enumerate(sent_indices):
        previous_sent_index = (
            sent_indices[position - 1]
            if position > 0
            else -1
        )

        record_starts.append(
            find_record_start(
                lines,
                previous_sent_index,
                sent_index,
            )
        )

    records: list[dict[str, object]] = []

    for position, sent_index in enumerate(sent_indices):
        start_index = record_starts[position]

        next_start_index = (
            record_starts[position + 1]
            if position + 1 < len(record_starts)
            else content_end_index
        )

        name, headline = parse_identity_block(
            lines,
            start_index,
            sent_index,
        )

        sent_match = SENT_PATTERN.fullmatch(
            lines[sent_index]
        )

        relative_date = (
            sent_match.group(1).strip()
            if sent_match
            else ""
        )

        note_lines = [
            lines[index]
            for index in range(
                sent_index + 1,
                next_start_index,
            )
            if lines[index]
            and lines[index].casefold() not in UI_ONLY_LINES
        ]

        connection_note = " ".join(note_lines).strip()

        records.append(
            {
                "Approximate Date": approximate_date(
                    relative_date
                ),
                "Relative Date": relative_date,
                "Activity Type": "Connection Outreach",
                "Name": name,
                "Headline": headline,
                "Connection Note": connection_note,
                "Status": "Pending",
                "Platform": "LinkedIn",
                "Source": "LinkedIn Sent Invitations Page",
                "Capture Date": CAPTURE_DATE,
            }
        )

    invitations = pd.DataFrame(records)

    invitations["Name"] = (
        invitations["Name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    invitations["Headline"] = (
        invitations["Headline"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    invitations["Connection Note"] = (
        invitations["Connection Note"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    blank_name_count = int(
        invitations["Name"].eq("").sum()
    )

    duplicate_name_count = int(
        invitations["Name"].duplicated(
            keep=False
        ).sum()
    )

    invitations = invitations.sort_values(
        by=[
            "Approximate Date",
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

    invitations.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print(
        "LinkedIn Sent lines detected: "
        f"{len(sent_indices)}"
    )
    print(
        "Pending invitation records parsed: "
        f"{len(invitations)}"
    )
    print(
        "Records with blank names: "
        f"{blank_name_count}"
    )
    print(
        "Rows belonging to duplicated names: "
        f"{duplicate_name_count}"
    )
    print(
        "Records containing connection notes: "
        f"{int(invitations['Connection Note'].ne('').sum())}"
    )
    print(
        "Records without connection notes: "
        f"{int(invitations['Connection Note'].eq('').sum())}"
    )
    print(
        f"Saved parsed invitations to: {OUTPUT_FILE}"
    )

    print("\nFirst five parsed invitations:")
    print(
        invitations.head().to_string(
            index=False
        )
    )

    print("\nLast five parsed invitations:")
    print(
        invitations.tail().to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()