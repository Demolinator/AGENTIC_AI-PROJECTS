import os
import re
import requests
import random
from langgraph.graph import StateGraph
from langgraph.constants import START, END
from typing import TypedDict
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to query Gemini LLM
def query_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        if hasattr(response, "candidates") and len(response.candidates) > 0:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            return "I'm sorry, I couldn't generate a response at this time."
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return "An error occurred while processing your request."

# Function to get the user's location
def get_user_location() -> str:
    try:
        # Replace 'YOUR_IPINFO_API_KEY' with your actual ipinfo.io API key
        response = requests.get("https://ipinfo.io", timeout=5)
        data = response.json()
        return data.get("city", "Unknown Location")
    except Exception as e:
        print(f"Debug: Error getting location - {e}")
        return "Unknown Location"

# Function to get the weather for a given location
def get_coordinates(location: str) -> dict:
    try:
        # Simplified to use only the city name
        url = f"https://nominatim.openstreetmap.org/search?city={location}&format=json"
        headers = {"User-Agent": "YourAppName/1.0 (contact@example.com)"}  # Replace with your contact info
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data:
                latitude = float(data[0]["lat"])
                longitude = float(data[0]["lon"])
                print(f"Debug: Found coordinates - Latitude: {latitude}, Longitude: {longitude}")
                return {"latitude": latitude, "longitude": longitude}
            else:
                print(f"Debug: No coordinates found for {location}")
                return {}
        else:
            print(f"Debug: Geocoding API returned status code {response.status_code}")
            return {}
    except Exception as e:
        print(f"Debug: Error fetching coordinates - {e}")
        return {}

WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear skies",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorms",
    96: "Thunderstorms with slight hail",
    99: "Thunderstorms with heavy hail"
}


def get_weather_for_today(location: str) -> str:
    try:
        coordinates = get_coordinates(location)
        if not coordinates:
            return f"Sorry, weather information is not available for {location}."

        latitude = coordinates["latitude"]
        longitude = coordinates["longitude"]

        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            current_weather = data.get("current_weather", {})
            temperature = current_weather.get("temperature")
            weather_code = current_weather.get("weathercode", -1)

            description = WEATHER_CODE_DESCRIPTIONS.get(weather_code, "Unknown weather condition")

            if temperature is not None:
                return f"The weather in {location} is {description} with a temperature of {temperature}°C."
            else:
                return f"Weather details are incomplete for {location}. Please try again later."
        else:
            return f"Could not fetch weather details for {location}."
    except Exception as e:
        return "Could not fetch weather details at the moment. Please try again later."


# Define the state schema
class State(TypedDict):
    message: str
    greeting_response: str
    weather_response: str
    joke_response: str
    final_response: str

# Greeting Agent Node with Gemini
def greeting_agent_function(state: State) -> State:
    message = state.get("message", "").strip()
    prompt = f"The user said: '{message}'. Generate a friendly greeting response."
    gemini_response = query_gemini(prompt)

    if gemini_response:
        state["greeting_response"] = gemini_response
    else:
        greetings = [
            "Hello! How can I assist you today?",
            "Hi there! What can I do for you?",
            "Hey! Need any help?"
        ]
        state["greeting_response"] = random.choice(greetings)

    return state

# Weather Agent Node with Gemini
def weather_agent_function(state: State) -> State:
    message = state.get("message", "").strip().lower()
    if "weather" in message or "temperature" in message or "forecast" in message:
        location = get_user_location()
        fallback_weather_response = get_weather_for_today(location)

        prompt = (
            f"The user asked about the weather in {location}. The current weather is: {fallback_weather_response}.\n"
            f"Generate a friendly and concise response incorporating this information."
        )

        gemini_response = query_gemini(prompt)

        # Use Gemini response or fallback data
        state["weather_response"] = (
            gemini_response if gemini_response else fallback_weather_response
        )
    else:
        state["weather_response"] = "I can provide weather information if you ask specifically."
    return state


# Joke Agent Node with Gemini
def joke_agent_function(state: State) -> State:
    joke_keywords = ["joke", "funny", "laugh"]
    message = state.get("message", "").strip().lower()

    if any(keyword in message for keyword in joke_keywords):
        fallback_jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call fake spaghetti? An impasta!"
        ]
        fallback_joke = random.choice(fallback_jokes)

        prompt = "The user asked for a joke. Provide a lighthearted and funny joke."
        gemini_response = query_gemini(prompt)

        state["joke_response"] = gemini_response if gemini_response else fallback_joke
    else:
        state["joke_response"] = "I can tell jokes if you ask for one!"

    return state

# Front-End Orchestration Node with Gemini Integration
def front_end_agent_function(state: State) -> State:
    prompt = (
        f"User message: '{state['message']}'.\n"
        f"Greeting response: '{state['greeting_response']}'.\n"
        f"Weather response: '{state['weather_response']}'.\n"
        f"Joke response: '{state['joke_response']}'.\n"
        "Create a single, friendly, and concise reply that incorporates all these responses. Avoid repetition."
    )

    gemini_response = query_gemini(prompt)
    if gemini_response:
        state["final_response"] = gemini_response.strip()
    else:
        state["final_response"] = " ".join([
            state["greeting_response"],
            state["weather_response"],
            state["joke_response"]
        ]).strip()
    return state


# Create StateGraph with the defined state schema
greeting_graph = StateGraph(state_schema=State)

# Add nodes
greeting_graph.add_node("GreetingAgent", greeting_agent_function)
greeting_graph.add_node("WeatherAgent", weather_agent_function)
greeting_graph.add_node("JokeAgent", joke_agent_function)
greeting_graph.add_node("FrontEndAgent", front_end_agent_function)

# Define the workflow
greeting_graph.add_edge(START, "GreetingAgent")  # Process greeting first
greeting_graph.add_edge("GreetingAgent", "JokeAgent")  # Route to JokeAgent
greeting_graph.add_edge("JokeAgent", "WeatherAgent")  # Route to WeatherAgent
greeting_graph.add_edge("WeatherAgent", "FrontEndAgent")  # Pass weather to front-end
greeting_graph.add_edge("FrontEndAgent", END)  # End workflow at front-end

# Compile the graph
compiled_graph = greeting_graph.compile()

# Function to run the graph
def run_greeting_agent(input_message: str) -> str:
    # Prepare initial state
    initial_state = {
        "message": input_message,
        "greeting_response": "",
        "weather_response": "",
        "joke_response": "",
        "final_response": ""
    }
    result = compiled_graph.invoke(initial_state)
    return result["final_response"]

# Test the agents
if __name__ == "__main__":
    print("Chatbot is running. Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = run_greeting_agent(user_message)
        print(f"Chatbot: {response}")