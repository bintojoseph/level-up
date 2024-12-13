# News Article Scraper

An asynchronous news article scraper that collects articles from various news websites including Reuters and TechCrunch.

## Features

- Asynchronous scraping for better performance
- Supports multiple news sources
- Extracts article content, metadata, and images
- Saves articles in both JSON and CSV formats
- Progress tracking with tqdm
- Comprehensive error handling and logging

## Supported News Sources

- Reuters (Technology, Business, World)
- TechCrunch (Home, Startups)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the scraper:
```bash
python news_scraper.py
```

The scraper will:
1. Create a `scraped_articles` directory
2. Save individual articles as JSON files
3. Create a consolidated CSV file with all articles

## Output

- Individual articles are saved as JSON files in the `scraped_articles` directory
- A consolidated CSV file `all_articles.csv` is created with all scraped articles
- Each article contains:
  - Title
  - Full text
  - Summary
  - Keywords
  - URL
  - Authors
  - Publication date
  - Top image URL
  - Scraping timestamp

## Note

Please be mindful of the websites' robots.txt files and implement appropriate delays between requests to avoid overwhelming their servers.
