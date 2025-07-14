import os
import requests
from bs4 import BeautifulSoup
import logging
import re
import time
from typing import List, Dict, Optional, Any
from requests.adapters import HTTPAdapter, Retry
import openai

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

openai.api_key = os.getenv("OPENAI_API_KEY")

# System prompt for product filtering and relevance checking
PRODUCT_FILTER_PROMPT = """
You are an expert e-commerce product filter and formatter. Your job is to:

1. **Assess Product Relevance**: Determine if a product matches the user's search intent
2. **Filter Appropriately**: Only include products that are truly relevant to the query
3. **Format Beautifully**: Present products in a clear, organized way with proper links

**Guidelines for Relevance:**
- For medical equipment queries, focus on actual medical devices, not accessories unless specifically requested
- For price-sensitive queries, prioritize products within the specified budget
- For brand queries, include products from the requested brand
- For specific product types, ensure the product actually matches the category

**Formatting Rules:**
- Always include the product name, price, and direct link
- Use clear, readable formatting with proper spacing
- For multiple products, use numbered lists
- Include a summary of total products found
- If filtering by price, mention the price range in the response

**Example Good Response:**
"I found 5 relevant wheelchairs under 100 KWD:

1. **Al Essa Light Wheelchair** - 75.0 KWD
   [View Product](https://example.com/product1)

2. **Drive Travel Wheelchair** - 85.0 KWD  
   [View Product](https://example.com/product2)

These are the most relevant options within your budget. Would you like more details about any specific model?"

Answer with "RELEVANT" if the product should be included, or "NOT_RELEVANT" if it should be filtered out.
"""

class SimpleScraper:
    """
    Simple scraper using BeautifulSoup with OpenAI LLM assistance for filtering and formatting.
    """
    
    def __init__(self, 
                 base_url: str = "https://www.alessaonline.com",
                 user_agent: str = None,
                 timeout: int = 10,
                 max_retries: int = 3,
                 enable_llm_filtering: bool = True,
                 llm_model: str = "gpt-4o-mini"):
        """
        Initialize the scraper.
        
        Args:
            base_url: Base URL for the website
            user_agent: User agent string
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            enable_llm_filtering: Whether to use LLM filtering
            llm_model: OpenAI model to use for filtering
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.enable_llm_filtering = enable_llm_filtering
        self.llm_model = llm_model
        
        # Default user agent if none provided
        if not user_agent:
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        
        # Setup HTTP session with retries
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": user_agent})

    def build_search_url(self, query: str, page: int = 1) -> str:
        """Build search URL for alessaonline.com."""
        search_term = re.sub(r'[^a-zA-Z0-9 ]', '', query).strip().replace(' ', '+')
        url = f"{self.base_url}/default/catalogsearch/result/?q={search_term}"
        if page > 1:
            url += f"&p={page}"
        return url

    def parse_price(self, price_str: str) -> Optional[float]:
        """Extract float price from string."""
        try:
            # Remove all non-numeric characters except decimal points
            num = re.sub(r'[^0-9.]', '', price_str)
            return float(num) if num else None
        except (ValueError, TypeError):
            return None

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a page."""
        try:
            logger.info(f"Scraping product prices from: {url}")
            resp = self.session.get(url, timeout=self.timeout, verify=False)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"HTTP error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Parsing error for {url}: {e}")
            return None

    def parse_products(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse products from BeautifulSoup object."""
        products = []
        
        for product in soup.select('li.item.product.product-item'):
            name_tag = product.select_one('h2.product.name.product-item-name a')
            price_tag = product.select_one('span.price')
            
            if name_tag and price_tag:
                name = name_tag.get_text(strip=True)
                price = self.parse_price(price_tag.get_text(strip=True))
                url = name_tag.get('href', '')
                
                if price is not None:
                    products.append({
                        'name': name,
                        'price': price,
                        'url': url
                    })
        
        logger.info(f"Scraped {len(products)} products from search.")
        return products

    def is_relevant_with_llm(self, query: str, product_data: Dict[str, Any]) -> bool:
        """Check if product is relevant using OpenAI LLM with enhanced prompt."""
        if not self.enable_llm_filtering:
            return True
            
        prompt = f"""
{PRODUCT_FILTER_PROMPT}

**User Query:** "{query}"

**Product to Evaluate:**
- Name: {product_data.get('name', '')}
- Price: {product_data.get('price', '')} KWD
- URL: {product_data.get('url', '')}

Based on the user's query and the product information above, should this product be included in the results?

Answer only: RELEVANT or NOT_RELEVANT
"""
        
        try:
            # Use the new OpenAI API format
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            answer = response.choices[0].message.content.strip().upper()
            return "RELEVANT" in answer
        except Exception as e:
            logger.warning(f"LLM filtering failed: {e}")
            return True  # Fallback: include by default

    def format_products_with_llm(self, query: str, products: List[Dict[str, Any]]) -> str:
        """Format products using OpenAI LLM for better presentation."""
        if not products:
            return "No products found matching your search criteria."
        
        # Prepare product list for LLM formatting
        product_list = []
        for i, product in enumerate(products[:10], 1):  # Limit to first 10 for formatting
            product_list.append(f"{i}. {product['name']} - {product['price']} KWD - {product['url']}")
        
        products_text = "\n".join(product_list)
        
        prompt = f"""
{PRODUCT_FILTER_PROMPT}

**User Query:** "{query}"

**Products Found:**
{products_text}

**Total Products:** {len(products)}

Please format these products in a clear, organized way that includes:
1. A summary of what was found
2. Numbered list of products with names, prices, and clickable links
3. A helpful closing message

Format the response to be user-friendly and include proper markdown formatting for links.
"""
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM formatting failed: {e}")
            # Fallback to simple formatting
            return self._format_products_simple(query, products)

    def _format_products_simple(self, query: str, products: List[Dict[str, Any]]) -> str:
        """Simple fallback formatting when LLM is not available."""
        total_products = len(products)
        reply = f"I found {total_products} products matching your search for '{query}':\n\n"
        
        # Show first 5 products in a numbered list
        for i, product in enumerate(products[:5], 1):
            reply += f"{i}. **{product['name']}** - {product['price']} KWD\n"
            reply += f"   [View Product]({product['url']})\n\n"
        
        if total_products > 5:
            reply += f"... and {total_products - 5} more products available. Would you like to see more specific options or filter by price range?"
        else:
            reply += "These are all the products I found. Would you like more details about any specific item?"
        
        return reply

    def search_products(self, 
                       query: str, 
                       max_pages: int = 1,
                       max_price: Optional[float] = None,
                       use_llm_formatting: bool = True) -> Dict[str, Any]:
        """
        Search for products with enhanced filtering and formatting.
        
        Args:
            query: Search query
            max_pages: Maximum pages to search
            max_price: Maximum price filter
            use_llm_formatting: Whether to use LLM for formatting
            
        Returns:
            Dictionary with 'products' list and 'formatted_reply' string
        """
        all_products = []
        
        for page in range(1, max_pages + 1):
            url = self.build_search_url(query, page)
            soup = self.fetch_page(url)
            
            if not soup:
                break
                
            products = self.parse_products(soup)
            if not products:
                break
                
            # Apply filters
            filtered_products = []
            for product in products:
                # Price filter
                if max_price and product.get('price', 0) > max_price:
                    continue
                    
                # LLM relevance filter (if enabled)
                if self.is_relevant_with_llm(query, product):
                    filtered_products.append(product)
            
            all_products.extend(filtered_products)
            
            # Stop if we got fewer products than expected (might be last page)
            if len(products) < 10:  # Assuming 10+ products per page
                break
        
        logger.info(f"Total relevant items for '{query}': {len(all_products)}")
        
        # Format the response
        if use_llm_formatting and self.enable_llm_filtering:
            formatted_reply = self.format_products_with_llm(query, all_products)
        else:
            formatted_reply = self._format_products_simple(query, all_products)
        
        return {
            'products': all_products,
            'formatted_reply': formatted_reply
        }


# Backward compatibility - create a default scraper instance
_scraper = SimpleScraper(enable_llm_filtering=True)

def get_product_prices_from_search(query: str, max_pages: int = 1) -> Dict[str, Any]:
    """
    Search for products by query using the SimpleScraper with enhanced formatting.
    
    Args:
        query: Search query
        max_pages: Maximum pages to search
        
    Returns:
        Dictionary with 'products' list and 'formatted_reply' string
    """
    return _scraper.search_products(query, max_pages=max_pages, use_llm_formatting=True)


if __name__ == "__main__":
    # Example usage
    scraper = SimpleScraper(enable_llm_filtering=True)
    
    result = scraper.search_products(
        query="wheelchair under 100 KWD", 
        max_pages=2,
        max_price=100.0,
        use_llm_formatting=True
    )
    
    print("Formatted Response:")
    print(result['formatted_reply'])
    print(f"\nTotal products found: {len(result['products'])}")
