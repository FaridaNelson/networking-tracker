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
    / "networking_summary_by_month.csv"
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
    "Reporting Month",
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


def parse_reporting_month(
    values: pd.Series,
    source_name: str,
) -> pd.Series:
    """
    Convert date values into YYYY-MM reporting-month strings.

    Raises an error when a nonblank date cannot be parsed so records
    are never silently excluded from the report.
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

        preview = invalid_values[:10]

        raise ValueError(
            f"{source_name} contains unparseable date values: "
            f"{preview}"
        )

    return parsed_dates.dt.strftime("%Y-%m").fillna("")


def parse_integer_series(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Convert a CSV column into integers.

    Blank values are treated as zero. Nonblank invalid values raise
    an error rather than being silently discarded.
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

        preview = invalid_values[:10]

        raise ValueError(
            f"{column_name} contains nonnumeric values: "
            f"{preview}"
        )

    fractional_mask = (
        numeric_values.mod(1).ne(0)
    )

    if fractional_mask.any():
        invalid_values = (
            cleaned_values[fractional_mask]
            .drop_duplicates()
            .tolist()
        )

        preview = invalid_values[:10]

        raise ValueError(
            f"{column_name} contains non-integer values: "
            f"{preview}"
        )

    return numeric_values.astype(int)


def summarize_pending(
    pending: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monthly metrics from the pending-connections ledger.
    """
    pending["Reporting Month"] = parse_reporting_month(
        pending["Activity Date"],
        "Pending connections ledger",
    )

    pending = pending[
        pending["Reporting Month"].ne("")
    ].copy()

    pending["Has Prentus Evidence"] = (
        pending["Evidence Status"]
        .map(clean_text)
        .eq("LinkedIn + Prentus")
        .astype(int)
    )

    monthly = (
        pending.groupby(
            "Reporting Month",
            as_index=False,
        )
        .agg(
            **{
                "Pending Invitations": (
                    "Reporting Month",
                    "size",
                ),
                "Pending + Prentus": (
                    "Has Prentus Evidence",
                    "sum",
                ),
            }
        )
    )

    return monthly


def summarize_accepted(
    accepted: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monthly metrics from the accepted-connections ledger.
    """
    accepted["Reporting Month"] = parse_reporting_month(
        accepted["Connected On"],
        "Accepted connections ledger",
    )

    accepted = accepted[
        accepted["Reporting Month"].ne("")
    ].copy()

    accepted["Has Prentus Evidence"] = (
        accepted["Evidence Status"]
        .map(clean_text)
        .eq("LinkedIn + Prentus")
        .astype(int)
    )

    monthly = (
        accepted.groupby(
            "Reporting Month",
            as_index=False,
        )
        .agg(
            **{
                "Accepted Connections": (
                    "Reporting Month",
                    "size",
                ),
                "Accepted + Prentus": (
                    "Has Prentus Evidence",
                    "sum",
                ),
            }
        )
    )

    return monthly


def summarize_conversations(
    conversations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monthly metrics from the conversations ledger.

    Each conversation is assigned to the month of its first message.
    Its aggregate message totals therefore describe conversations
    started in that month, not necessarily messages sent during that
    calendar month.
    """
    conversations["Reporting Month"] = (
        parse_reporting_month(
            conversations["First Message Date"],
            "Conversations ledger",
        )
    )

    conversations = conversations[
        conversations["Reporting Month"].ne("")
    ].copy()

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

    message_mismatch_mask = (
        conversations["Messages Sent"]
        + conversations["Messages Received"]
        != conversations["Total Messages"]
    )

    if message_mismatch_mask.any():
        mismatch_count = int(
            message_mismatch_mask.sum()
        )

        raise ValueError(
            "Conversation message totals are inconsistent. "
            f"Found {mismatch_count} row(s) where Messages Sent "
            "+ Messages Received does not equal Total Messages."
        )

    interaction_status = (
        conversations["Interaction Status"]
        .map(clean_text)
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

    monthly = (
        conversations.groupby(
            "Reporting Month",
            as_index=False,
        )
        .agg(
            **{
                "Conversations Started": (
                    "Reporting Month",
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

    return monthly


def build_complete_month_range(
    monthly_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Build one continuous reporting-month sequence covering all data.

    Missing months are retained with zero-valued metrics.
    """
    available_months: list[str] = []

    for dataframe in monthly_frames:
        if dataframe.empty:
            continue

        available_months.extend(
            dataframe["Reporting Month"].tolist()
        )

    if not available_months:
        return pd.DataFrame(
            columns=["Reporting Month"]
        )

    first_month = min(available_months)
    last_month = max(available_months)

    month_range = pd.period_range(
        start=first_month,
        end=last_month,
        freq="M",
    )

    return pd.DataFrame(
        {
            "Reporting Month": (
                month_range.astype(str)
            )
        }
    )


def validate_summary(
    summary: pd.DataFrame,
    pending: pd.DataFrame,
    accepted: pd.DataFrame,
    conversations: pd.DataFrame,
) -> None:
    """
    Validate that monthly totals reconcile with the source ledgers.
    """
    expected_pending = (
        pending["Activity Date"]
        .map(clean_text)
        .ne("")
        .sum()
    )

    expected_accepted = (
        accepted["Connected On"]
        .map(clean_text)
        .ne("")
        .sum()
    )

    expected_conversations = (
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
            f"expected {expected_pending}, got {actual_pending}."
        )

    if actual_accepted != expected_accepted:
        raise ValueError(
            "Accepted connection totals do not reconcile: "
            f"expected {expected_accepted}, got {actual_accepted}."
        )

    if actual_conversations != expected_conversations:
        raise ValueError(
            "Conversation totals do not reconcile: "
            f"expected {expected_conversations}, "
            f"got {actual_conversations}."
        )

    conversation_status_total = int(
        summary[
            [
                "Two-Way Conversations",
                "Outbound Only Conversations",
                "Inbound Only Conversations",
            ]
        ].sum().sum()
    )

    if conversation_status_total != actual_conversations:
        raise ValueError(
            "Conversation interaction-status totals do not "
            "equal Conversations Started."
        )

    if not (
        summary["Messages Sent"]
        + summary["Messages Received"]
        == summary["Total Messages"]
    ).all():
        raise ValueError(
            "Monthly message totals are inconsistent."
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

    pending_monthly = summarize_pending(pending)
    accepted_monthly = summarize_accepted(accepted)
    conversations_monthly = (
        summarize_conversations(conversations)
    )

    summary = build_complete_month_range(
        [
            pending_monthly,
            accepted_monthly,
            conversations_monthly,
        ]
    )

    for monthly_frame in [
        pending_monthly,
        accepted_monthly,
        conversations_monthly,
    ]:
        summary = summary.merge(
            monthly_frame,
            on="Reporting Month",
            how="left",
        )

    metric_columns = [
        column
        for column in SUMMARY_COLUMNS
        if column != "Reporting Month"
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
        "\nMonthly summary rows created: "
        f"{len(summary)}"
    )

    if not summary.empty:
        print(
            "Reporting range: "
            f"{summary['Reporting Month'].iloc[0]} "
            "through "
            f"{summary['Reporting Month'].iloc[-1]}"
        )

    print(
        "\nSaved monthly networking summary to: "
        f"{OUTPUT_FILE}"
    )

    print("\nMonthly networking summary:")
    preview_columns = [
        "Reporting Month",
        "Pending Invitations",
        "Accepted Connections",
        "Conversations Started",
        "Messages Sent",
        "Messages Received",
        "Total Messages",
    ]

    print(
        summary[
            preview_columns
        ].to_string(index=False)
    )

    print("\nConversation status by month:")

    status_columns = [
        "Reporting Month",
        "Two-Way Conversations",
        "Outbound Only Conversations",
        "Inbound Only Conversations",
    ]

    print(
        summary[
            status_columns
        ].to_string(index=False)
    )

    print("\nPrentus evidence by month:")

    evidence_columns = [
        "Reporting Month",
        "Pending + Prentus",
        "Accepted + Prentus",
        "Conversations + Prentus",
    ]

    print(
        summary[
            evidence_columns
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()