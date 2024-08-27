import json
import re
import os
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize LangChain's OpenAI client
llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Function to clean HTML tags from text
def clean_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text) if isinstance(text, str) else text

# Function to prefilter events based on the query
def prefilter_events(events, query):
    query_lower = query.lower()
    filtered_events = [
        event for event in events
        if any(keyword.lower() in query_lower for keyword in event.get('Mots clés', '').split(','))
        or any(keyword.lower() in query_lower for keyword in event.get('Titre', '').split())
        or any(keyword.lower() in query_lower for keyword in event.get('Description', '').split())
    ]
    return filtered_events if filtered_events else events  # return all if no events match prefiltering

# Function to preprocess events and create a TF-IDF matrix
def preprocess_events(events):
    for event in events:
        event['combined_text'] = f"{event['Titre']} {event['Description']} {event.get('Mots clés', '')}"
        event['combined_text'] = clean_html_tags(event['combined_text'])
    texts = [event['combined_text'] for event in events]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix, events

# Function to get top N relevant events based on query
def get_top_relevant_events(query, vectorizer, tfidf_matrix, events, top_n=20):
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    relevant_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return [events[idx] for idx in relevant_indices]

# Function to generate a user-friendly prompt with the suggested events
def generate_prompt(query, suggested_events):
    events_text = "\n".join([
        f"ID: {event['ID']}, Title: {event['Titre']}, Date: {event['Date de début']} to {event['Date de fin']}, Location: {event['Nom du lieu']}, {event['Adresse du lieu']}, {event['Code postal']}, {event['Ville']}"
        for event in suggested_events
    ])
    prompt = f"""
    You are an assistant that helps people find events in Paris. Based on the query below, suggest the best matching event from the list.
    
    Query: {query}
    
    Events:
    {events_text}

    Suggested Event:
    """
    return prompt

# Find the best matching event
def find_event(query, events, vectorizer, tfidf_matrix, top_n=20):
    relevant_events = get_top_relevant_events(query, vectorizer, tfidf_matrix, events, top_n)
    prompt = generate_prompt(query, relevant_events)
    
    # Use LangChain to generate the response
    prompt_template = PromptTemplate(input_variables=["prompt"], template="{prompt}")
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(prompt=prompt)
    
    return response, relevant_events

# Load formatted events from the JSON file
def load_formatted_events(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Save filtered events to a JSON file
def save_filtered_events(filtered_events, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(filtered_events, file, ensure_ascii=False, indent=2)

# Main function to find relevant events based on user query and generate a prompt
def find_and_generate_prompt(user_query, events, top_n=20):
    filtered_events = prefilter_events(events, user_query)
    vectorizer, tfidf_matrix, processed_events = preprocess_events(filtered_events)
    suggested_event, relevant_events = find_event(user_query, processed_events, vectorizer, tfidf_matrix, top_n)
    
    # Save the filtered events
    save_filtered_events(relevant_events, 'filtered_events.json')
    
    return suggested_event

# Flask route to handle the API request
@app.route('/find-event', methods=['POST'])
def find_event_route():
    body = request.json
    query = body['query']
    
    # Load your events data
    formatted_events_file_path = 'data/formatted_xlsx_events.json'
    formatted_events = load_formatted_events(formatted_events_file_path)
    
    # Find the event
    suggested_event = find_and_generate_prompt(query, formatted_events)
    
    return jsonify({"suggestedEvent": suggested_event})

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000, debug=True)
