import asyncio
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from newspaper import Article
from typing import List, Dict, Optional
from tqdm import tqdm
import logging
import json
import os
import nltk
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import os
from dotenv import load_dotenv
from bson import ObjectId
from json import JSONEncoder

# Custom JSON encoder to handle ObjectId
class MongoJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return JSONEncoder.default(self, o)

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsScraperAsync:
    def __init__(self):
        # Initialize MongoDB Atlas client
        mongodb_uri = os.getenv('MONGODB_URI')
        if not mongodb_uri:
            raise ValueError("MongoDB Atlas URI not found in environment variables")
            
        try:
            self.mongo_client = MongoClient(mongodb_uri)
            # Test the connection
            self.mongo_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {str(e)}")
            raise
            
        self.db = self.mongo_client['news_articles']
        self.articles_collection = self.db['articles']
        
        # Create index on url field to ensure uniqueness
        self.articles_collection.create_index('url', unique=True)
        
        # Download required NLTK data
        try:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('averaged_perceptron_tagger')
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {str(e)}")

        self.news_sources = {
            'TechCrunch': {
                'home': 'https://techcrunch.com/',
                'startups': 'https://techcrunch.com/startups/',
                'ai': 'https://techcrunch.com/artificial-intelligence/'
            },
            'The Verge': {
                'home': 'https://www.theverge.com/',
                'tech': 'https://www.theverge.com/tech',
                'ai': 'https://www.theverge.com/ai-artificial-intelligence'
            },
            'Wired': {
                'home': 'https://www.wired.com/',
                'business': 'https://www.wired.com/category/business/',
                'ai': 'https://www.wired.com/tag/artificial-intelligence/'
            },
            'ArsTechnica': {
                'home': 'https://arstechnica.com/',
                'tech': 'https://arstechnica.com/gadgets/',
                'science': 'https://arstechnica.com/science/'
            }
        }
        
        # Create directory for storing articles
        self.output_dir = 'scraped_articles'
        os.makedirs(self.output_dir, exist_ok=True)

        # Configure headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch the HTML content of a page."""
        try:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.text()
                logger.warning(f"Failed to fetch {url}, status code: {response.status}")
                return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_links(self, html: str, source: str) -> List[str]:
        """Extract article links from HTML based on the news source."""
        soup = BeautifulSoup(html, 'html.parser')
        links = []

        if source == 'TechCrunch':
            # TechCrunch specific link extraction
            articles = soup.find_all('a', href=True)
            links = [a['href'] for a in articles if 'techcrunch.com' in a.get('href', '') 
                    and any(x in a.get('href', '') for x in ['/2024/', '/2023/'])]
            
            # Remove duplicates while preserving order
            seen = set()
            links = [x for x in links if not (x in seen or seen.add(x))]
            
            logger.info(f"Found {len(links)} articles on {source}")

        return links

    async def scrape_article(self, url: str) -> Optional[Dict]:
        """Scrape a single article using newspaper3k."""
        try:
            article = Article(url)
            await asyncio.get_event_loop().run_in_executor(None, article.download)
            await asyncio.get_event_loop().run_in_executor(None, article.parse)
            await asyncio.get_event_loop().run_in_executor(None, article.nlp)

            article_data = {
                'title': article.title,
                'text': article.text,
                'summary': article.summary,
                'keywords': article.keywords,
                'url': url,
                'authors': article.authors,
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'top_image': article.top_image,
                'scraped_at': datetime.now().isoformat()
            }
            
            # Try to insert the article into MongoDB
            try:
                self.articles_collection.insert_one(article_data)
                logger.info(f"Article '{article.title}' saved to MongoDB")
            except DuplicateKeyError:
                logger.warning(f"Article with URL {url} already exists in MongoDB")
            except Exception as e:
                logger.error(f"Error saving article to MongoDB: {str(e)}")
            
            return article_data
        except Exception as e:
            logger.error(f"Error scraping article {url}: {str(e)}")
            return None

    async def scrape_source(self, source: str, category: str, session: aiohttp.ClientSession) -> List[Dict]:
        """Scrape articles from a specific source and category."""
        url = self.news_sources[source][category]
        html = await self.fetch_page(session, url)
        if not html:
            return []

        article_links = self.extract_links(html, source)
        articles = []

        # Use tqdm for progress tracking
        for link in tqdm(article_links[:10], desc=f"Scraping {source} - {category}"):
            article_data = await self.scrape_article(link)
            if article_data:
                articles.append(article_data)
                # Save individual article
                self.save_article(article_data, source, category)

        return articles

    def save_article(self, article_data: Dict, source: str, category: str) -> None:
        """Save article to a JSON file."""
        try:
            # Create directory for source if it doesn't exist
            source_dir = os.path.join(self.output_dir, source.lower())
            os.makedirs(source_dir, exist_ok=True)
            
            # Save individual article
            filename = f"{category.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(source_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, indent=2, cls=MongoJSONEncoder)
                
            logger.info(f"Saved article to {filepath}")
            
            # Update the combined articles file
            combined_file = os.path.join(self.output_dir, 'articles.json')
            articles = []
            
            if os.path.exists(combined_file):
                with open(combined_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        articles = data.get('articles', [])
                    except json.JSONDecodeError:
                        logger.warning(f"Error reading {combined_file}, starting fresh")
            
            articles.append(article_data)
            
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump({'articles': articles}, f, indent=2, cls=MongoJSONEncoder)
                
        except Exception as e:
            logger.error(f"Error saving article: {str(e)}")

    async def run(self):
        """Run the scraper for all sources and categories."""
        async with aiohttp.ClientSession() as session:
            all_articles = []
            for source, categories in self.news_sources.items():
                for category in categories:
                    logger.info(f"Scraping {source} - {category}")
                    articles = await self.scrape_source(source, category, session)
                    
                    # Add source and category information to each article
                    for article in articles:
                        article['source'] = source
                        article['category'] = category
                        article['scraped_at'] = '2024-12-12T15:59:19+05:30'
                    
                    all_articles.extend(articles)

            # Save all articles to JSON
            if all_articles:
                json_file = f"{self.output_dir}/articles.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'articles': all_articles,
                        'metadata': {
                            'total_articles': len(all_articles),
                            'sources': list(self.news_sources.keys()),
                            'scraped_at': '2024-12-12T15:59:19+05:30'
                        }
                    }, f, ensure_ascii=False, indent=2, cls=MongoJSONEncoder)
                logger.info(f"Saved {len(all_articles)} articles to {json_file}")

            # Save to CSV as well for backward compatibility
            if all_articles:
                df = pd.DataFrame(all_articles)
                df.to_csv(f"{self.output_dir}/all_articles.csv", index=False)
                logger.info(f"Saved {len(all_articles)} articles to CSV")

async def main():
    scraper = NewsScraperAsync()
    await scraper.run()

if __name__ == "__main__":
    asyncio.run(main())
