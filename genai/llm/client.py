import os
from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "groq/compound"


def generate_answer(prompt: str) -> str:
    """
    Call Groq LLM to generate answer.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,  # low temperature for factual answers
    )

    return response.choices[0].message.content.strip()
