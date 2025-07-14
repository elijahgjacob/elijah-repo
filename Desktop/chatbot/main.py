from fastapi import FastAPI, Request, status, Query
from fastapi.responses import JSONResponse
from logging_config import logger
from scraping import get_product_prices_from_search
from llm import get_llm_response
from collections import defaultdict

app = FastAPI()

# Chat history storage (in production, use a proper database)
chat_history = defaultdict(list)
MAX_HISTORY = 10

# Define product keywords here for now
PRODUCT_KEYWORDS = [
    "product", "price", "buy", "purchase", "order", "cost", "shop", "catalog", "device", "equipment", "supply", "supplies", "sku", "item", "oxygen", "mobility", "walker", "wheelchair", "bed", "nebulizer", "concentrator", "mask", "cannula", "cpap", "bipap", "suction", "monitor", "pulse oximeter", "thermometer", "glucose", "meter", "strip", "test", "bandage", "dressing", "brace", "support", "sling", "crutch", "stethoscope", "bp monitor", "blood pressure", "inhaler", "humidifier", "compressor", "lift", "hoist", "ramp", "commode", "urinal", "bedpan", "cushion", "mattress", "pillow", "orthopedic", "prosthesis", "orthosis", "splint", "cast", "walker", "rollator", "scooter", "cart", "trolley", "disposable", "consumable", "accessory", "refill", "replacement", "brand", "brands"
]

def add_to_history(session_id: str, user_message: str, bot_reply: str):
    """Add message to chat history, maintaining max 10 messages"""
    chat_history[session_id].append({
        "user": user_message,
        "bot": bot_reply,
        "timestamp": "now"  # In production, use actual timestamps
    })
    
    # Keep only the last MAX_HISTORY messages
    if len(chat_history[session_id]) > MAX_HISTORY:
        chat_history[session_id] = chat_history[session_id][-MAX_HISTORY:]

def get_chat_context(session_id: str) -> str:
    """Get formatted chat history for context"""
    if not chat_history[session_id]:
        return ""
    
    context = "\n\nPrevious conversation:\n"
    for i, msg in enumerate(chat_history[session_id][-5:], 1):  # Last 5 messages for context
        context += f"{i}. User: {msg['user']}\n   Bot: {msg['bot']}\n"
    return context

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("text")
    session_id = data.get("session_id", "default")  # Use session_id if provided
    
    logger.info(f"Received /chat request: {query} (session: {session_id})")
    
    try:
        # Always attempt scraping first for product-related queries
        if any(word in query.lower() for word in PRODUCT_KEYWORDS):
            # Use enhanced scraping with LLM formatting
            result = get_product_prices_from_search(query)
            
            # Check if result is the new format (dict) or old format (list)
            if isinstance(result, dict):
                products = result['products']
                formatted_reply = result['formatted_reply']
            else:
                # Fallback for old format
                products = result
                formatted_reply = None
            
            if products:
                if formatted_reply:
                    # Use LLM-formatted reply
                    reply = formatted_reply
                else:
                    # Fallback to simple formatting
                    total_products = len(products)
                    reply = f"I found {total_products} products matching your search:\n\n"
                    
                    # Show first 5 products in a numbered list
                    for i, product in enumerate(products[:5], 1):
                        reply += f"{i}. **{product['name']}** - {product['price']} KWD\n"
                        reply += f"   [View Product]({product['url']})\n\n"
                    
                    if total_products > 5:
                        reply += f"... and {total_products - 5} more products available. Would you like to see more specific options or filter by price range?"
                    else:
                        reply += "These are all the products I found. Would you like more details about any specific item?"
                
                logger.info(f"Scraping reply: {reply}")
                add_to_history(session_id, query, reply)
                return {"reply": reply, "products": products, "session_id": session_id}
            else:
                logger.info("No products found from scraping, falling back to LLM...")
                # Include chat history context for LLM
                context = get_chat_context(session_id)
                enhanced_query = f"{query}\n{context}" if context else query
                reply = get_llm_response(enhanced_query)
                add_to_history(session_id, query, reply)
                return {"reply": reply, "session_id": session_id}
        else:
            # Include chat history context for LLM
            context = get_chat_context(session_id)
            enhanced_query = f"{query}\n{context}" if context else query
            reply = get_llm_response(enhanced_query)
            add_to_history(session_id, query, reply)
            return {"reply": reply, "session_id": session_id}
    except Exception as e:
        logger.error(f"Error during /chat processing: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "An internal error occurred. Please try again later.", "details": str(e)}
        )

@app.get("/scrape-prices")
async def scrape_prices(category_url: str = Query(..., description="Category or search URL to scrape")):
    logger.info(f"Received /scrape-prices request: {category_url}")
    try:
        products = get_product_prices_from_search(category_url)
        logger.info(f"Scraped {len(products)} products from {category_url}")
        return {"products": products}
    except Exception as e:
        logger.error(f"Error during /scrape-prices processing: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "An internal error occurred. Please try again later.", "details": str(e)}
        )

@app.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a specific session"""
    return {"session_id": session_id, "history": chat_history.get(session_id, [])}

@app.delete("/chat-history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a specific session"""
    if session_id in chat_history:
        del chat_history[session_id]
    return {"message": f"Chat history cleared for session {session_id}"} 