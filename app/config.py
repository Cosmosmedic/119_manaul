import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
GPT_MODEL = "gpt-4o-mini"

