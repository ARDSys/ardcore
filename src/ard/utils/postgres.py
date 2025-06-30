import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_db_connection_string() -> str:
    """Get PostgreSQL connection string from environment variables."""
    host = os.getenv("POSTGRES_HOSTNAME")
    port = os.getenv("POSTGRES_PORT")
    dbname = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    # URL encode the credentials
    encoded_password = quote_plus(password)
    encoded_user = quote_plus(user)

    return f"postgresql://{encoded_user}:{encoded_password}@{host}:{port}/{dbname}"


def get_db_connection() -> Tuple[
    psycopg2.extensions.connection, psycopg2.extensions.cursor
]:
    """Create and return a database connection and cursor."""
    conn_string = get_db_connection_string()
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    return conn, cur


def check_paper_exists(cur: psycopg2.extensions.cursor, paper_hash: str) -> bool:
    """Check if paper already exists in database."""
    query = f"SELECT COUNT(*) FROM public.papers WHERE paper_hash = '{paper_hash}'"
    cur.execute(query)
    result = cur.fetchone()
    return result[0] > 0


def insert_paper_triplets(
    conn: psycopg2.extensions.connection,
    cur: psycopg2.extensions.cursor,
    df: pd.DataFrame,
    paper_hash: str,
    storage_path: str,
) -> int:
    """
    Insert paper triplets into database.

    Args:
        conn: Database connection
        cur: Database cursor
        df: DataFrame containing triplets
        paper_hash: Paper hash identifier
        storage_path: Source file path

    Returns:
        int: Number of rows inserted
    """
    # Rename columns
    df_columns = list(df.columns)
    column_mapping = {
        df_columns[0]: "subject",
        df_columns[1]: "predicate",
        df_columns[2]: "object",
    }
    df = df.rename(columns=column_mapping)

    # Select and add required columns
    df = df[["subject", "predicate", "object"]]
    df["paper_hash"] = paper_hash
    df["source_file"] = storage_path

    # Create values string for SQL insert
    values = ",".join(
        cur.mogrify(
            "(%s,%s,%s,%s,%s)",
            (row.subject, row.predicate, row.object, row.paper_hash, row.source_file),
        ).decode("utf-8")
        for row in df.itertuples(index=False)
    )

    # Insert data
    cur.execute(f"""
        INSERT INTO papers (subject, predicate, object, paper_hash, source_file)
        VALUES {values}
    """)
    conn.commit()

    return len(df)


def execute_query(
    conn: psycopg2.extensions.connection,
    cur: psycopg2.extensions.cursor,
    query: str,
    params: Optional[Tuple] = None,
    fetch: bool = False,
) -> Optional[List[Dict]]:
    """
    Execute a generic SQL query

    Args:
        conn: Database connection
        cur: Database cursor
        query: SQL query to execute
        params: Optional tuple of parameters for the query
        fetch: Whether to fetch and return results

    Returns:
        List of dictionaries containing query results if fetch=True, None otherwise
    """
    try:
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)

        if fetch:
            return cur.fetchall()
        else:
            conn.commit()
            return None

    except Exception as e:
        conn.rollback()
        raise Exception(f"Error executing query: {str(e)}")


def insert_record(
    conn: psycopg2.extensions.connection,
    cur: psycopg2.extensions.cursor,
    table: str,
    data: Dict[str, Any],
) -> None:
    """
    Insert a record into a table

    Args:
        conn: Database connection
        cur: Database cursor
        table: Table name
        data: Dictionary of column names and values
    """
    columns = ", ".join(data.keys())
    values = ", ".join(["%s"] * len(data))
    query = f"INSERT INTO {table} ({columns}) VALUES ({values})"

    execute_query(conn, cur, query, tuple(data.values()))


def update_record(
    conn: psycopg2.extensions.connection,
    cur: psycopg2.extensions.cursor,
    table: str,
    data: Dict[str, Any],
    where_clause: str,
    where_params: Tuple,
) -> None:
    """
    Update records in a table

    Args:
        conn: Database connection
        cur: Database cursor
        table: Table name
        data: Dictionary of column names and new values
        where_clause: SQL WHERE clause
        where_params: Parameters for the WHERE clause
    """
    set_clause = ", ".join([f"{k} = %s" for k in data.keys()])
    query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

    execute_query(conn, cur, query, tuple(data.values()) + where_params)


def select_records(
    conn: psycopg2.extensions.connection,
    cur: psycopg2.extensions.cursor,
    table: str,
    columns: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
    where_params: Optional[Tuple] = None,
) -> List[Dict]:
    """
    Select records from a table

    Args:
        conn: Database connection
        cur: Database cursor
        table: Table name
        columns: List of columns to select (None for all)
        where_clause: Optional SQL WHERE clause
        where_params: Parameters for the WHERE clause

    Returns:
        List of dictionaries containing query results
    """
    cols = "*" if not columns else ", ".join(columns)
    query = f"SELECT {cols} FROM {table}"

    if where_clause:
        query += f" WHERE {where_clause}"

    return execute_query(conn, cur, query, where_params, fetch=True)
