from langchain_openai import ChatOpenAI
from groq import Groq
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model = "groq/compound",
    temperature = 0.2

)

