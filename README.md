### Project Documentation: DateGenius

---

#### **Overview**

**DateGenius** is a proof-of-concept project designed to enhance the event discovery experience in Paris. The project involves data cleaning, event formatting, using your large language model of choice to suggest events based on user queries. It also includes a basic Flask web server for deploying the event suggestion functionality.

---

### **Modules Overview**

#### 1. **Data Cleaning and Formatting**

- **Core Libraries:**
  - `pandas`, `json`, `re`, `BeautifulSoup`
  
- **Key Functions:**
  - **`remove_html_tags`**: Strips HTML tags.
  - **`find_and_replace_ambiguous_unicode`**: Cleans non-ASCII characters.
  - **`clean_data`**: Cleans and processes raw event data.
  - **`format_events_for_prompt`**: Structures event data for prompt generation.
  - **`convert_and_clean_excel`**: Main function for cleaning and saving Excel data to JSON.

#### 2. **Event Recommendation System**

- **Core Libraries:**
  - `json`, `re`, `os`, `TfidfVectorizer`, `cosine_similarity`, `LLMChain`, `OpenAI`

- **Key Functions:**
  - **`prefilter_events`**: Filters events based on a query.
  - **`preprocess_events`**: Prepares event data for TF-IDF vectorization.
  - **`get_top_relevant_events`**: Finds top N relevant events using cosine similarity.
  - **`generate_prompt`**: Generates a text prompt for the language model.
  - **`find_and_generate_prompt`**: Main function for finding and suggesting events.

#### 3. **Web Server Integration**

- **Core Library:**
  - `Flask`

- **Key Functionality:**
  - **`/chatbot` endpoint**: Accepts user queries and returns suggested events.

---

### **Usage**

1. **Data Cleaning:**
   - Use `convert_and_clean_excel()` to process the Excel file and save the cleaned and formatted data in JSON format.

2. **Event Suggestion:**
   - Call `find_and_generate_prompt()` with a user query to receive event suggestions.

3. **Web Server:**
   - Run the Flask app and interact with the chatbot at `http://localhost:5000/chatbot`.

---

### **Environment Setup**

- **Install Dependencies:**
  ```bash
  pip install pandas beautifulsoup4 scikit-learn langchain openai flask python-dotenv
  ```

- **Environment Variables:**
  - Add your OpenAI API key in a `.env` file:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

---

This documentation provides a high-level overview of the DateGenius project, outlining its key components and usage instructions. The project can be extended with additional features like more refined filtering a good option would be pinecone with a vector database, advanced prompt generation, and further model integration.
