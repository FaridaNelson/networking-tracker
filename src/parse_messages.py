from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MESSAGES_FILE = (
    PROJECT_ROOT / "data" / "raw" / "linkedin" / "messages.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "linkedin_conversations_filtered.csv"
)

START_DATE = pd.Timestamp("2025-12-01", tz="UTC")
END_DATE = pd.Timestamp("2026-07-24 23:59:59", tz="UTC")

MY_NAME = "Farida Nelson"

# Personal conversations intentionally excluded from the
# professional networking tracker.
EXCLUDED_CONTACTS = {
    "Tim Hill",
    "Paula Ward,Tim Hill",
}

def main() -> None:
    if not MESSAGES_FILE.exists():
        raise FileNotFoundError(
            f"Could not find LinkedIn messages file: {MESSAGES_FILE}"
        )

    messages = pd.read_csv(
        MESSAGES_FILE,
        encoding="utf-8-sig",
    )

    messages["DATE"] = pd.to_datetime(
        messages["DATE"],
        utc=True,
        errors="coerce",
    )

    invalid_dates = int(messages["DATE"].isna().sum())

    filtered = messages.loc[
        messages["DATE"].between(
            START_DATE,
            END_DATE,
            inclusive="both",
        )
    ].copy()

    filtered["Sent By Me"] = filtered["FROM"].eq(MY_NAME)

    filtered["Contact Name"] = filtered["TO"].where(
        filtered["Sent By Me"],
        filtered["FROM"],
    )

    filtered["Contact Profile URL"] = (
        filtered["RECIPIENT PROFILE URLS"].where(
            filtered["Sent By Me"],
            filtered["SENDER PROFILE URL"],
        )
    )

    filtered["Contact Name"] = (
        filtered["Contact Name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    filtered["Contact Profile URL"] = (
        filtered["Contact Profile URL"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    filtered = filtered.loc[
        filtered["Contact Name"].ne("")
        & filtered["Contact Name"].ne(MY_NAME)
    ].copy()

    filtered = filtered.loc[
        ~filtered["Contact Name"].isin(EXCLUDED_CONTACTS)
    ].copy()

    filtered["Sent Count"] = filtered["Sent By Me"].astype(int)
    filtered["Received Count"] = (~filtered["Sent By Me"]).astype(int)

    conversations = (
        filtered.groupby(
            [
                "CONVERSATION ID",
                "Contact Name",
                "Contact Profile URL",
            ],
            dropna=False,
        )
        .agg(
            First_Message_Date=("DATE", "min"),
            Last_Message_Date=("DATE", "max"),
            Messages_Sent=("Sent Count", "sum"),
            Messages_Received=("Received Count", "sum"),
            Total_Messages=("DATE", "size"),
        )
        .reset_index()
    )

    conversations["Interaction Status"] = "Inbound Only"

    conversations.loc[
        (conversations["Messages_Sent"] > 0)
        & (conversations["Messages_Received"] == 0),
        "Interaction Status",
    ] = "Outbound Only"

    conversations.loc[
        (conversations["Messages_Sent"] > 0)
        & (conversations["Messages_Received"] > 0),
        "Interaction Status",
    ] = "Two-Way Conversation"

    conversations["Platform"] = "LinkedIn"
    conversations["Activity Type"] = "Conversation"

    conversations = conversations[
        [
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
        ]
    ]

    conversations = conversations.sort_values(
        by=["First_Message_Date", "Contact Name"],
        ascending=True,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conversations.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    status_counts = conversations["Interaction Status"].value_counts()

    two_way_count = int(
        status_counts.get("Two-Way Conversation", 0)
    )

    outbound_only_count = int(
        status_counts.get("Outbound Only", 0)
    )

    inbound_only_count = int(
        status_counts.get("Inbound Only", 0)
    )

    print(f"Total LinkedIn message rows: {len(messages)}")
    print(f"Rows with invalid dates: {invalid_dates}")
    print(
        "Message rows within date range: "
        f"{len(filtered)}"
    )

    print(
        "Distinct conversation/contact records: "
        f"{len(conversations)}"
    )

    print("\nInteraction status counts:")
    print(f"Two-way conversations: {two_way_count}")
    print(f"Outbound only: {outbound_only_count}")
    print(f"Inbound only: {inbound_only_count}")

    print(f"\nSaved conversation summary to: {OUTPUT_FILE}")

    print("\nFirst five conversations:")
    print(
        conversations.head().to_string(index=False)
    )

    print("\nLast five conversations:")
    print(
        conversations.tail().to_string(index=False)
    )

if __name__ == "__main__":
    main()