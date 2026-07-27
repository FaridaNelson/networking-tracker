from pathlib import Path

import pandas as pd

from utils import clean_text, require_columns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PENDING_LEDGER_FILE = (
    PROJECT_ROOT
    / "master"
    / "pending_connections_ledger.csv"
)

ACCEPTED_LEDGER_FILE = (
    PROJECT_ROOT
    / "master"
    / "accepted_connections_ledger.csv"
)

CONVERSATIONS_LEDGER_FILE = (
    PROJECT_ROOT
    / "master"
    / "conversations_ledger.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "master"
    / "networking_summary_by_week.csv"
)


PENDING_REQUIRED_COLUMNS = {
    "Activity Date",
    "Evidence Status",
}

ACCEPTED_REQUIRED_COLUMNS = {
    "Connected On",
    "Evidence Status",
}

CONVERSATIONS_REQUIRED_COLUMNS = {
    "First Message Date",
    "Messages Sent",
    "Messages Received",
    "Total Messages",
    "Interaction Status",
    "Evidence Status",
}


SUMMARY_COLUMNS = [
    "Reporting Week",
    "Week Start",
    "Week End",
    "Pending Invitations",
    "Accepted Connections",
    "Conversations Started",
    "Messages Sent",
    "Messages Received",
    "Total Messages",
    "Two-Way Conversations",
    "Outbound Only Conversations",
    "Inbound Only Conversations",
    "Pending + Prentus",
    "Accepted + Prentus",
    "Conversations + Prentus",
]


def load_csv(
    path: Path,
    required_columns: set[str],
    description: str,
) -> pd.DataFrame:
    """
    Load a canonical ledger and validate its required columns.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {description}: {path}"
        )

    dataframe = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    require_columns(
        dataframe,
        required_columns,
        description,
    )

    return dataframe.copy()


def parse_dates(
    values: pd.Series,
    source_name: str,
) -> pd.Series:
    """
    Parse mixed date and ISO timestamp values.

    Raises an error when a nonblank date cannot be parsed so
    records are never silently excluded.
    """
    cleaned_values = values.map(clean_text)

    parsed_dates = pd.to_datetime(
        cleaned_values.replace("", pd.NA),
        errors="coerce",
        format="mixed",
        utc=True,
    )

    invalid_mask = (
        cleaned_values.ne("")
        & parsed_dates.isna()
    )

    if invalid_mask.any():
        invalid_values = (
            cleaned_values[invalid_mask]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"{source_name} contains unparseable date values: "
            f"{invalid_values[:10]}"
        )

    return parsed_dates


def add_week_fields(
    dataframe: pd.DataFrame,
    date_column: str,
    source_name: str,
) -> pd.DataFrame:
    """
    Add ISO reporting-week fields.

    Weeks run Monday through Sunday.
    Reporting Week uses the ISO week-year format YYYY-Www.
    """
    result = dataframe.copy()

    parsed_dates = parse_dates(
        result[date_column],
        source_name,
    )

    result["Parsed Date"] = parsed_dates

    result = result[
        result["Parsed Date"].notna()
    ].copy()

    local_dates = (
        result["Parsed Date"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    iso_calendar = local_dates.dt.isocalendar()

    result["Reporting Week"] = (
        iso_calendar["year"].astype(str)
        + "-W"
        + iso_calendar["week"]
        .astype(str)
        .str.zfill(2)
    )

    result["Week Start"] = (
        local_dates
        - pd.to_timedelta(
            local_dates.dt.weekday,
            unit="D",
        )
    )

    result["Week End"] = (
        result["Week Start"]
        + pd.Timedelta(days=6)
    )

    result["Week Start"] = (
        result["Week Start"]
        .dt.strftime("%Y-%m-%d")
    )

    result["Week End"] = (
        result["Week End"]
        .dt.strftime("%Y-%m-%d")
    )

    return result


def parse_integer_series(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Convert a CSV column into integers.

    Blank values are treated as zero. Invalid nonblank values raise
    an error.
    """
    cleaned_values = values.map(clean_text)

    numeric_values = pd.to_numeric(
        cleaned_values.replace("", "0"),
        errors="coerce",
    )

    invalid_mask = numeric_values.isna()

    if invalid_mask.any():
        invalid_values = (
            cleaned_values[invalid_mask]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"{column_name} contains nonnumeric values: "
            f"{invalid_values[:10]}"
        )

    fractional_mask = numeric_values.mod(1).ne(0)

    if fractional_mask.any():
        invalid_values = (
            cleaned_values[fractional_mask]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"{column_name} contains non-integer values: "
            f"{invalid_values[:10]}"
        )

    return numeric_values.astype(int)


def summarize_pending(
    pending: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build weekly metrics from the pending-connections ledger.
    """
    pending = add_week_fields(
        pending,
        "Activity Date",
        "Pending connections ledger",
    )

    pending["Has Prentus Evidence"] = (
        pending["Evidence Status"]
        .map(clean_text)
        .eq("LinkedIn + Prentus")
        .astype(int)
    )

    return (
        pending.groupby(
            [
                "Reporting Week",
                "Week Start",
                "Week End",
            ],
            as_index=False,
        )
        .agg(
            **{
                "Pending Invitations": (
                    "Reporting Week",
                    "size",
                ),
                "Pending + Prentus": (
                    "Has Prentus Evidence",
                    "sum",
                ),
            }
        )
    )


def summarize_accepted(
    accepted: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build weekly metrics from the accepted-connections ledger.
    """
    accepted = add_week_fields(
        accepted,
        "Connected On",
        "Accepted connections ledger",
    )

    accepted["Has Prentus Evidence"] = (
        accepted["Evidence Status"]
        .map(clean_text)
        .eq("LinkedIn + Prentus")
        .astype(int)
    )

    return (
        accepted.groupby(
            [
                "Reporting Week",
                "Week Start",
                "Week End",
            ],
            as_index=False,
        )
        .agg(
            **{
                "Accepted Connections": (
                    "Reporting Week",
                    "size",
                ),
                "Accepted + Prentus": (
                    "Has Prentus Evidence",
                    "sum",
                ),
            }
        )
    )


def summarize_conversations(
    conversations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build weekly metrics from the conversations ledger.

    Each conversation is assigned to the ISO week of its first
    message. Message totals therefore describe conversations that
    started during that week.
    """
    conversations = add_week_fields(
        conversations,
        "First Message Date",
        "Conversations ledger",
    )

    conversations["Messages Sent"] = (
        parse_integer_series(
            conversations["Messages Sent"],
            "Messages Sent",
        )
    )

    conversations["Messages Received"] = (
        parse_integer_series(
            conversations["Messages Received"],
            "Messages Received",
        )
    )

    conversations["Total Messages"] = (
        parse_integer_series(
            conversations["Total Messages"],
            "Total Messages",
        )
    )

    mismatch_mask = (
        conversations["Messages Sent"]
        + conversations["Messages Received"]
        != conversations["Total Messages"]
    )

    if mismatch_mask.any():
        raise ValueError(
            "Conversation message totals are inconsistent. "
            f"Found {int(mismatch_mask.sum())} row(s) where "
            "Messages Sent + Messages Received does not equal "
            "Total Messages."
        )

    interaction_status = (
        conversations["Interaction Status"]
        .map(clean_text)
    )

    known_statuses = {
        "Two-Way Conversation",
        "Outbound Only",
        "Inbound Only",
    }

    unknown_status_mask = (
        interaction_status.ne("")
        & ~interaction_status.isin(known_statuses)
    )

    if unknown_status_mask.any():
        unknown_statuses = (
            interaction_status[unknown_status_mask]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Conversations ledger contains unknown interaction "
            f"statuses: {unknown_statuses}"
        )

    conversations["Is Two-Way"] = (
        interaction_status
        .eq("Two-Way Conversation")
        .astype(int)
    )

    conversations["Is Outbound Only"] = (
        interaction_status
        .eq("Outbound Only")
        .astype(int)
    )

    conversations["Is Inbound Only"] = (
        interaction_status
        .eq("Inbound Only")
        .astype(int)
    )

    conversations["Has Prentus Evidence"] = (
        conversations["Evidence Status"]
        .map(clean_text)
        .eq("LinkedIn + Prentus")
        .astype(int)
    )

    return (
        conversations.groupby(
            [
                "Reporting Week",
                "Week Start",
                "Week End",
            ],
            as_index=False,
        )
        .agg(
            **{
                "Conversations Started": (
                    "Reporting Week",
                    "size",
                ),
                "Messages Sent": (
                    "Messages Sent",
                    "sum",
                ),
                "Messages Received": (
                    "Messages Received",
                    "sum",
                ),
                "Total Messages": (
                    "Total Messages",
                    "sum",
                ),
                "Two-Way Conversations": (
                    "Is Two-Way",
                    "sum",
                ),
                "Outbound Only Conversations": (
                    "Is Outbound Only",
                    "sum",
                ),
                "Inbound Only Conversations": (
                    "Is Inbound Only",
                    "sum",
                ),
                "Conversations + Prentus": (
                    "Has Prentus Evidence",
                    "sum",
                ),
            }
        )
    )


def build_complete_week_range(
    weekly_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Build one continuous Monday-through-Sunday week sequence
    covering all source data.

    Weeks with no activity are retained with zero metrics.
    """
    week_starts: list[str] = []

    for dataframe in weekly_frames:
        if dataframe.empty:
            continue

        week_starts.extend(
            dataframe["Week Start"].tolist()
        )

    if not week_starts:
        return pd.DataFrame(
            columns=[
                "Reporting Week",
                "Week Start",
                "Week End",
            ]
        )

    first_week_start = pd.to_datetime(
        min(week_starts)
    )

    last_week_start = pd.to_datetime(
        max(week_starts)
    )

    complete_week_starts = pd.date_range(
        start=first_week_start,
        end=last_week_start,
        freq="7D",
    )

    complete = pd.DataFrame(
        {
            "Week Start": complete_week_starts,
        }
    )

    complete["Week End"] = (
        complete["Week Start"]
        + pd.Timedelta(days=6)
    )

    iso_calendar = (
        complete["Week Start"]
        .dt.isocalendar()
    )

    complete["Reporting Week"] = (
        iso_calendar["year"].astype(str)
        + "-W"
        + iso_calendar["week"]
        .astype(str)
        .str.zfill(2)
    )

    complete["Week Start"] = (
        complete["Week Start"]
        .dt.strftime("%Y-%m-%d")
    )

    complete["Week End"] = (
        complete["Week End"]
        .dt.strftime("%Y-%m-%d")
    )

    return complete[
        [
            "Reporting Week",
            "Week Start",
            "Week End",
        ]
    ]


def validate_summary(
    summary: pd.DataFrame,
    pending: pd.DataFrame,
    accepted: pd.DataFrame,
    conversations: pd.DataFrame,
) -> None:
    """
    Validate that weekly totals reconcile with the source ledgers.
    """
    expected_pending = int(
        pending["Activity Date"]
        .map(clean_text)
        .ne("")
        .sum()
    )

    expected_accepted = int(
        accepted["Connected On"]
        .map(clean_text)
        .ne("")
        .sum()
    )

    expected_conversations = int(
        conversations["First Message Date"]
        .map(clean_text)
        .ne("")
        .sum()
    )

    actual_pending = int(
        summary["Pending Invitations"].sum()
    )

    actual_accepted = int(
        summary["Accepted Connections"].sum()
    )

    actual_conversations = int(
        summary["Conversations Started"].sum()
    )

    if actual_pending != expected_pending:
        raise ValueError(
            "Pending invitation totals do not reconcile: "
            f"expected {expected_pending}, "
            f"got {actual_pending}."
        )

    if actual_accepted != expected_accepted:
        raise ValueError(
            "Accepted connection totals do not reconcile: "
            f"expected {expected_accepted}, "
            f"got {actual_accepted}."
        )

    if actual_conversations != expected_conversations:
        raise ValueError(
            "Conversation totals do not reconcile: "
            f"expected {expected_conversations}, "
            f"got {actual_conversations}."
        )

    status_total = int(
        summary[
            [
                "Two-Way Conversations",
                "Outbound Only Conversations",
                "Inbound Only Conversations",
            ]
        ].sum().sum()
    )

    if status_total != actual_conversations:
        raise ValueError(
            "Conversation interaction-status totals do not "
            "equal Conversations Started."
        )

    message_totals_match = (
        summary["Messages Sent"]
        + summary["Messages Received"]
        == summary["Total Messages"]
    )

    if not message_totals_match.all():
        raise ValueError(
            "Weekly message totals are inconsistent."
        )

    if summary["Reporting Week"].duplicated().any():
        raise ValueError(
            "Duplicate Reporting Week values found."
        )

    if summary["Week Start"].duplicated().any():
        raise ValueError(
            "Duplicate Week Start values found."
        )


def main() -> None:
    pending = load_csv(
        PENDING_LEDGER_FILE,
        PENDING_REQUIRED_COLUMNS,
        "pending connections ledger",
    )

    accepted = load_csv(
        ACCEPTED_LEDGER_FILE,
        ACCEPTED_REQUIRED_COLUMNS,
        "accepted connections ledger",
    )

    conversations = load_csv(
        CONVERSATIONS_LEDGER_FILE,
        CONVERSATIONS_REQUIRED_COLUMNS,
        "conversations ledger",
    )

    pending_weekly = summarize_pending(pending)
    accepted_weekly = summarize_accepted(accepted)
    conversations_weekly = (
        summarize_conversations(conversations)
    )

    summary = build_complete_week_range(
        [
            pending_weekly,
            accepted_weekly,
            conversations_weekly,
        ]
    )

    merge_columns = [
        "Reporting Week",
        "Week Start",
        "Week End",
    ]

    for weekly_frame in [
        pending_weekly,
        accepted_weekly,
        conversations_weekly,
    ]:
        summary = summary.merge(
            weekly_frame,
            on=merge_columns,
            how="left",
        )

    metric_columns = [
        column
        for column in SUMMARY_COLUMNS
        if column not in merge_columns
    ]

    for column in metric_columns:
        if column not in summary.columns:
            summary[column] = 0

        summary[column] = (
            summary[column]
            .fillna(0)
            .astype(int)
        )

    summary = summary[SUMMARY_COLUMNS]

    validate_summary(
        summary,
        pending,
        accepted,
        conversations,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Pending connection records loaded: "
        f"{len(pending)}"
    )

    print(
        "Accepted connection records loaded: "
        f"{len(accepted)}"
    )

    print(
        "Conversation records loaded: "
        f"{len(conversations)}"
    )

    print(
        "\nWeekly summary rows created: "
        f"{len(summary)}"
    )

    if not summary.empty:
        print(
            "Reporting range: "
            f"{summary['Reporting Week'].iloc[0]} "
            "through "
            f"{summary['Reporting Week'].iloc[-1]}"
        )

    print(
        "\nSaved weekly networking summary to: "
        f"{OUTPUT_FILE}"
    )

    preview_columns = [
        "Reporting Week",
        "Week Start",
        "Week End",
        "Pending Invitations",
        "Accepted Connections",
        "Conversations Started",
    ]

    print("\nWeekly networking summary:")
    print(
        summary[
            preview_columns
        ].to_string(index=False)
    )

    activity_columns = [
        "Reporting Week",
        "Messages Sent",
        "Messages Received",
        "Total Messages",
        "Two-Way Conversations",
        "Outbound Only Conversations",
        "Inbound Only Conversations",
    ]

    print("\nConversation activity by week:")
    print(
        summary[
            activity_columns
        ].to_string(index=False)
    )

    evidence_columns = [
        "Reporting Week",
        "Pending + Prentus",
        "Accepted + Prentus",
        "Conversations + Prentus",
    ]

    print("\nPrentus evidence by week:")
    print(
        summary[
            evidence_columns
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()