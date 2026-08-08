from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()  # reads .env in the current directory

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in 3 words"}],
)
print(response.choices[0].message.content)