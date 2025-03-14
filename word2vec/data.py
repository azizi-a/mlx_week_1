import word2vec.config as config
import os
import requests
import sqlalchemy


def download_text8():
    """Download the text8 dataset if it doesn't exist."""
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(config.DATA1_PATH), exist_ok=True)

    if not os.path.exists(config.DATA1_PATH):
        print("Downloading text8 dataset...")
        r = requests.get(config.DATA_URL)
        with open(config.DATA1_PATH, "wb") as f:
            f.write(r.content)
        print("Download complete!")
    else:
        print("text8 file already exists")

    # Read and return the data
    with open(config.DATA1_PATH) as f:
        return f.read()


def get_data_from_postgres(query, cache_file):
    """
    Get data from Postgres, using local cache if available
    """
    # Check if cached data exists
    if os.path.exists(cache_file):
        print("Using cached Postgres data")
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    # Create directory for cache file if it doesn't exist
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # If no cache, get from postgres and save
    data = execute_query(query)
    text_data = save_pgsql_data_as_text(data, cache_file)
    return text_data


def save_pgsql_data_as_text(data, output_file=config.DATA2_PATH):
    """
    Convert postgres query results to text and save to file
    """
    # Convert query results to single text string
    text_data = " ".join([row[0] for row in data if row[0]])

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text_data)

    return text_data


def execute_query(query):
    """
    Execute a PostgreSQL query and return the results

    Args:
        query (str): SQL query to execute

    Returns:
        list: List of tuples containing the query results
    """
    # Create SQLAlchemy engine
    engine = sqlalchemy.create_engine(config.DB_URL)

    try:
        # Execute query
        with engine.connect() as connection:
            result = connection.execute(sqlalchemy.text(query))
            return result.fetchall()
    finally:
        # Dispose of the engine
        engine.dispose()


if __name__ == "__main__":
    download_text8()
    get_data_from_postgres()
