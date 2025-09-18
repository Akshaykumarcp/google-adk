# @title Import necessary libraries
# python root_agent_v1.py - single agent

import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

print("Libraries imported.")

# @title Define the get_weather Tool
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    print(f"--- Tool: get_weather called for city: {city} ---") # Log tool execution
    city_normalized = city.lower().replace(" ", "") # Basic normalization

    # Mock weather data
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}

# Example tool usage (optional test)
# print(get_weather("New York"))
# print(get_weather("Paris"))

# @title Define the Weather Agent
# Use one of the model constants defined earlier
# AGENT_MODEL = MODEL_GEMINI_2_0_FLASH # Starting with Gemini
AGENT_MODEL = "gpt-4o" # Starting with Gemini


weather_agent = Agent(
    name="weather_agent_v1",
    model=LiteLlm(model="azure/gpt-4o"), # Can be a string for Gemini or a LiteLlm object
    description="Provides weather information for specific cities.",
    instruction="You are a helpful weather assistant. "
                "When the user asks for the weather in a specific city, "
                "use the 'get_weather' tool to find the information. "
                "If the tool returns an error, inform the user politely. "
                "If the tool is successful, present the weather report clearly.",
    tools=[get_weather], # Pass the function directly
)

print(f"Agent '{weather_agent.name}' created using model '{AGENT_MODEL}'.")

# @title Setup Session Service and Runner

# --- Session Management ---
# Key Concept: SessionService stores conversation history & state.
# InMemorySessionService is simple, non-persistent storage for this tutorial.
# session_service = InMemorySessionService()

# Define constants for identifying the interaction context
async def setup_session_and_runner():
    # --- Session Management ---
    session_service = InMemorySessionService()

    APP_NAME = "weather_tutorial_app"
    USER_ID = "user_1"
    SESSION_ID = "session_001"

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    # --- Runner ---
    runner = Runner(
        agent=weather_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    # print(f"Runner created for agent '{runner.agent.name}'.")

    return USER_ID, SESSION_ID, runner


# Create the specific session where the conversation will happen

# --- Runner ---
# Key Concept: Runner orchestrates the agent execution loop.

# Run the async function
USER_ID, SESSION_ID, runner = asyncio.run(setup_session_and_runner())
print(f"Runner created for agent '{runner.agent.name}'.")

# @title Define Agent Interaction Function

from google.genai import types # For creating message Content/Parts

async def call_agent_async(query: str, runner, user_id, session_id):
  """Sends a query to the agent and prints the final response."""
  print(f"\n>>> User Query: {query}")

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer.
  async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
      # You can uncomment the line below to see *all* events during execution
      print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

      # Key Concept: is_final_response() marks the concluding message for the turn.
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # Handle potential errors/escalations
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          # Add more checks here if needed (e.g., specific error codes)
          print(f"<<< Agent Response: {final_response_text}")

          break # Stop processing events once the final response is found

#   print(f"<<< Agent Response: {final_response_text}")

  # @title Run the Initial Conversation

# We need an async function to await our interaction helper
async def run_conversation():
    await call_agent_async("What is the weather like in London?",
                                       runner=runner,
                                       user_id=USER_ID,
                                       session_id=SESSION_ID)

    # await call_agent_async("How about Paris?",
    #                                    runner=runner,
    #                                    user_id=USER_ID,
    #                                    session_id=SESSION_ID) # Expecting the tool's error message

    # await call_agent_async("Tell me the weather in New York",
    #                                    runner=runner,
    #                                    user_id=USER_ID,
    #                                    session_id=SESSION_ID)

# Execute the conversation using await in an async context (like Colab/Jupyter)
# await run_conversation()

# --- OR ---

# Uncomment the following lines if running as a standard Python script (.py file):
if __name__ == "__main__":
    try:
        asyncio.run(run_conversation())
    except Exception as e:
        print(f"An error occurred: {e}")