# src/retrieval.py
import re
from pyspark.sql import functions as F

DEFAULT_MOVIE_TABLE = "default.wiki_movie_plots_deduped"


def retrieve_movies(spark, question: str, k: int = 5, movie_table: str = DEFAULT_MOVIE_TABLE):
    """
    Simple keyword-based retrieval using Spark.
    Scores rows by how many query tokens appear across Title/Plot/Genre/Director/Cast.
    """
    q = (question or "").lower().strip()
    tokens = [t for t in re.findall(r"[a-z0-9]+", q) if len(t) >= 3]
    tokens = list(dict.fromkeys(tokens))  # unique, preserve order

    df = spark.table(movie_table)

    hay = F.lower(F.concat_ws(
        " ",
        F.coalesce(F.col("Title"), F.lit("")),
        F.coalesce(F.col("Plot"), F.lit("")),
        F.coalesce(F.col("Genre"), F.lit("")),
        F.coalesce(F.col("Director"), F.lit("")),
        F.coalesce(F.col("Cast"), F.lit("")),
    ))

    if not tokens:
        return df.limit(k)

    score = None
    for t in tokens:
        hit = F.when(hay.contains(t), F.lit(1)).otherwise(F.lit(0))
        score = hit if score is None else (score + hit)

    return (
        df.withColumn("_score", score)
          .where(F.col("_score") > 0)
          .orderBy(F.col("_score").desc())
          .limit(k)
    )


def format_context(rows, use_tags=use_tags) -> str:
    """
    Convert retrieved Spark Rows into a readable context block for the LM prompt.
    """
    blocks = []
    for r in rows:
        title = (r["Title"] or "").strip()
        year = r["Release Year"]
        genre = (r["Genre"] or "").strip()
        director = (r["Director"] or "").strip()
        cast = (r["Cast"] or "").strip()
        plot = (r["Plot"] or "").strip()

        if len(plot) > max_plot_chars:
            plot = plot[:max_plot_chars].rsplit(" ", 1)[0] + "..."

        blocks.append(
            f"<TITLE> {title}\n"
            f"<YEAR> {year}\n"
            f"<ORIGIN> {origin}\n"
            f"<DIRECTOR> {director}\n"
            f"<CAST> {cast}\n"
            f"<GENRE> {genre}\n"
            f"<PLOT> {plot}\n"
        )

    return "\n---\n".join(blocks).strip()
