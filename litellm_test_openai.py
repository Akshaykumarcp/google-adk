from litellm import completion
import os
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

# azure call
response = completion(
    model = "azure/gpt-4o", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

print(response)