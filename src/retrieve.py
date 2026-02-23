# Movie retrieval from the Spark table using keyword scoring.
# Separated from generation so the retrieval strategy can be improved
# independently (e.g. swapping keyword search for embeddings later).

import re
from pyspark.sql import functions as F

from config import MOVIE_TABLE, STOPWORDS, RETRIEVE_K, PLOT_TRUNCATE_CHARS


def retrieve_movies(question: str, spark, k: int = RETRIEVE_K):
    """
    Score every movie in the table by how many query keywords it contains,
    return the top k results as a Spark DataFrame.

    Scoring:
        - question is lowercased and split into tokens
        - stopwords and short tokens (< 3 chars) are removed
        - each token scores 1 point per field it appears in
          (Title, Plot, Genre, Director, Cast combined into one haystack)
        - results ordered by score descending

    Args:
        question:  user's natural language query
        spark:     active SparkSession
        k:         number of results to return

    Returns:
        Spark DataFrame with top k matching movies
    """
    df = spark.table(MOVIE_TABLE)
    q  = (question or "").lower().strip()

    tokens = [
        t for t in re.findall(r"[a-z0-9]+", q)
        if len(t) >= 3 and t not in STOPWORDS
    ]
    tokens = list(dict.fromkeys(tokens))   # unique, preserve order

    if not tokens:
        return df.limit(k)

    hay = F.lower(F.concat_ws(" ",
        F.coalesce(F.col("Title"),    F.lit("")),
        F.coalesce(F.col("Plot"),     F.lit("")),
        F.coalesce(F.col("Genre"),    F.lit("")),
        F.coalesce(F.col("Director"), F.lit("")),
        F.coalesce(F.col("Cast"),     F.lit("")),
    ))

    score = None
    for t in tokens:
        hit   = F.when(hay.contains(t), F.lit(1)).otherwise(F.lit(0))
        score = hit if score is None else (score + hit)

    return (
        df.withColumn("_score", score)
        .where(F.col("_score") > 0)
        .orderBy(F.col("_score").desc())
        .limit(k)
    )


def row_to_prompt(r) -> str:
    """
    Format a movie row into a structured prompt string.
    Must match the format used in training exactly — any difference
    causes the model to see token patterns it was never trained on.

    Example output:
        <TITLE> the dark knight <YEAR> 2008 <DIRECTOR> christopher nolan
        <GENRE> action <PLOT> batman fights the joker...
    """
    def ct(v) -> str:
        if v is None:
            return ""
        return re.sub(r"\s+", " ", str(v).replace("\u00a0", " ")).strip()

    # Fix titles stored as "Dark Knight, The" -> "The Dark Knight"
    title = ct(r["Title"])
    match = re.match(r"^(.*),\s*(The|A|An)$", title, re.IGNORECASE)
    if match:
        title = f"{match.group(2)} {match.group(1)}"

    return (
        f"<TITLE> {title} "
        f"<YEAR> {ct(r['Release Year'])} "
        f"<DIRECTOR> {ct(r['Director'])} "
        f"<GENRE> {ct(r['Genre'])} "
        f"<PLOT> {ct(r['Plot'])[:PLOT_TRUNCATE_CHARS]} "
    ).strip()