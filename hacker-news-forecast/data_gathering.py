"""Script for gathering Hacker News data from PostgreSQL database."""

import pandas as pd
from sqlalchemy import create_engine
from config import DB_URL, QUERY

def gather_data():
    """Fetch story titles and scores from Hacker News database."""
    engine = create_engine(DB_URL)
    
    try:
        df = pd.read_sql(QUERY, engine)
        print(f"Retrieved {len(df)} records")
        return df
    except Exception as e:
        print(f"Error gathering data: {e}")
        return None
    finally:
        engine.dispose()

if __name__ == "__main__":
    gather_data() 