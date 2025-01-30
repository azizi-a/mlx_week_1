import requests
import os
from config import DATA_URL, DATA1_PATH, DB_URL, DATA2_PATH
from sqlalchemy import create_engine, text

def download_text8():
    """Download the text8 dataset if it doesn't exist."""
    if not os.path.exists(DATA1_PATH):
        print("Downloading text8 dataset...")
        r = requests.get(DATA_URL)
        with open(DATA1_PATH, "wb") as f:
            f.write(r.content)
        print("Download complete!")
    else:
        print("text8 file already exists")

    # Read and return the data
    with open(DATA1_PATH) as f:
        return f.read()

def get_data_from_postgres(query, cache_file):
    """
    Get data from Postgres, using local cache if available
    """
    # Check if cached data exists
    if os.path.exists(cache_file):
        print("Using cached Postgres data")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    # If no cache, get from postgres and save
    data = execute_query(query)
    text_data = save_pgsql_data_as_text(data, cache_file)
    return text_data

def save_pgsql_data_as_text(data, output_file=DATA2_PATH):
    """
    Convert postgres query results to text and save to file
    """
    # Convert query results to single text string
    text_data = " ".join([row[0] for row in data if row[0]])
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
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
    engine = create_engine(DB_URL)
    
    try:
        # Execute query
        with engine.connect() as connection:
            result = connection.execute(text(query))
            return result.fetchall()
    finally:
        # Dispose of the engine
        engine.dispose()

if __name__ == "__main__":
    download_text8() 
    get_data_from_postgres()