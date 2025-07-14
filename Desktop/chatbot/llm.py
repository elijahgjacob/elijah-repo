from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY
import logging

logger = logging.getLogger(__name__)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

def get_llm_response(query):
    logger.info("Calling OpenAI LLM...")
    response = llm.invoke(query)
    # Only return the message content
    if isinstance(response, dict) and 'content' in response:
        reply = response['content']
    elif hasattr(response, 'content'):
        reply = response.content
    else:
        reply = str(response)
    logger.info(f"LLM response: {reply}")
    return reply 