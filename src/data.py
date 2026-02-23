# All data loading and cleaning functions.
# Used by train.py to build the document corpus.

import re


def clean_text(s) -> str:
    """Normalise whitespace and remove non-breaking spaces."""
    if s is None:
        return ""
    s = str(s).replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_year(y) -> str:
    """Convert a year value to a 4-digit string. Handles floats like '1912.0'."""
    try:
        return str(int(float(y)))
    except Exception:
        return ""


def take_cast(cast, n: int = 5) -> str:
    """Return the first n cast members as a comma-separated string."""
    cast = clean_text(cast)
    if not cast:
        return ""
    parts = [p.strip() for p in cast.split(",") if p.strip()]
    return ", ".join(parts[:n])


def row_to_doc(r) -> str:
    """
    Convert a movie DataFrame row into a structured text document.
    This format is used during training — inference must match it exactly.

    Example output:
        <TITLE> the dark knight <YEAR> 2008 <DIRECTOR> christopher nolan
        <CAST> christian bale, heath ledger <GENRE> action
        <PLOT> batman fights the joker...
    """
    title    = clean_text(r["Title"])
    year     = safe_year(r["Release Year"])
    origin   = clean_text(r["Origin/Ethnicity"])
    director = clean_text(r["Director"])
    genre    = clean_text(r["Genre"])
    cast     = take_cast(r["Cast"], n=5)
    plot     = clean_text(r["Plot"])

    if not title or not plot:
        return ""

    return (
        f"<TITLE> {title} "
        f"<YEAR> {year} "
        f"<ORIGIN> {origin} "
        f"<DIRECTOR> {director} "
        f"<CAST> {cast} "
        f"<GENRE> {genre} "
        f"<PLOT> {plot} "
    ).strip()


def load_docs(spark, table: str) -> list[str]:
    """Load movie table from Spark and return a list of formatted document strings."""
    df   = spark.table(table).toPandas()
    docs = [row_to_doc(r) for _, r in df.iterrows()]
    docs = [d for d in docs if d]
    print(f"Loaded {len(docs)} documents from {table}")
    return docs