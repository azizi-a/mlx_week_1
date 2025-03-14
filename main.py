def main():
    choice = (
        input("What would you like to run? (1: word2vec, 2: hacker-news-forecast): ")
        .strip()
        .lower()
    )
    # Allow selection by number as well
    if choice == "1":
        choice = "word2vec"
    elif choice == "2":
        choice = "hacker-news-forecast"
    if choice == "word2vec":
        print("Running Word2Vec model...")
        # Import and run word2vec script
        import word2vec.main as word2vec_main

        word2vec_main.main()
    elif choice == "hacker-news-forecast":
        print("Running Hacker News Forecast model...")
        # Import and run hacker news forecast script
        import hacker_news_forecast.run as hn_forecast_main

        hn_forecast_main.run()
    else:
        print(
            "Invalid choice. Please run again and select either 'word2vec' or 'hacker-news-forecast'."
        )


if __name__ == "__main__":
    main()
