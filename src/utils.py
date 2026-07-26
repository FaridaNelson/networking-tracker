import re
import unicodedata
from collections.abc import Iterable

import pandas as pd


def clean_text(value: object) -> str:
    """
    Convert a value to a clean string.

    - Missing pandas values become an empty string.
    - Leading and trailing whitespace is removed.
    - Repeated internal whitespace is collapsed.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    return re.sub(
        r"\s+",
        " ",
        text,
    )


def normalize_name(value: object) -> str:
    """
    Normalize a person's name for conservative comparisons.

    The function:
    - converts missing values to an empty string
    - removes leading and trailing whitespace
    - removes accents and other combining characters
    - ignores capitalization
    - replaces punctuation with spaces
    - collapses repeated whitespace

    It intentionally does not:
    - remove initials
    - remove credentials
    - remove suffixes
    - reorder name components
    - perform fuzzy matching
    """
    text = clean_text(value)

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.casefold()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = text.replace(
        "_",
        " ",
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    source_name: str,
) -> None:
    """
    Validate that a DataFrame contains all required columns.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    required = set(
        required_columns
    )

    missing_columns = (
        required
        - set(dataframe.columns)
    )

    if not missing_columns:
        return

    missing_list = ", ".join(
        sorted(missing_columns)
    )

    raise ValueError(
        f"{source_name} is missing required columns: "
        f"{missing_list}"
    )