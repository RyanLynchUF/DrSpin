import streamlit as st
import requests
from pathlib import Path
import PyPDF2
import io
from typing import Optional
import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure page settings
st.set_page_config(
    page_title="Dr. Spin - Positive Perspective Generator",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS to match the logo's color scheme
st.markdown("""
    <style>
    .stButton>button {
        background-color: #FF9999;
        color: black;
    }
    .stTextInput>div>div>input {
        border-color: #FF9999;
    }
    .stHeader {
        background-color: #FF9999;
    }
    </style>
    """, unsafe_allow_html=True)

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_llm_client(api_key: str, model_type: str) -> Optional[any]:
    """Initialize the selected LLM client with the provided API key."""
    try:
        if model_type == "OpenAI":
            return openai.Client(api_key=api_key)
        elif model_type == "Anthropic":
            return anthropic.Client(api_key=api_key)
        return None
    except Exception as e:
        st.error(f"Error initializing {model_type} client: {str(e)}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_llm_response(client, input_text: str, model_type: str, include_citations: bool) -> str:
    """Get response from the selected LLM with retry logic."""
    try:
        system_prompt = """You are Dr. Spin, an AI assistant specialized in finding positive perspectives 
        in negative situations. Provide thoughtful, balanced responses that acknowledge concerns while 
        highlighting opportunities and silver linings. If citations are requested, include relevant sources."""
        
        if model_type == "OpenAI":
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ]
            )
            return response.choices[0].message.content
        elif model_type == "Anthropic":
            response = client.messages.create(
                model="claude-2",
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": input_text}]
            )
            return response.content[0].text
        
    except Exception as e:
        st.error(f"Error getting response from {model_type}: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

def extract_text_from_file(uploaded_file) -> str:
    """Extract text content from uploaded files (PDF or TXT)."""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            return " ".join(page.extract_text() for page in pdf_reader.pages)
        else:  # txt file
            return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return ""

def main():
    # Header
    st.title("Dr. Spin 🎯")
    st.markdown("Transform negative perspectives into positive insights")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        model_type = st.selectbox("Select LLM", ["OpenAI", "Anthropic"])
        api_key = st.text_input("Enter API Key", type="password")
        include_citations = st.checkbox("Include citations", 
                                     help="Enable to include source citations (may use more tokens)")
        
        st.header("Negative Aspects")
        negative_aspects = st.multiselect(
            "What makes this negative?",
            ["Economic Impact", "Social Issues", "Environmental Concerns", 
             "Personal Impact", "Political Issues", "Other"]
        )
    
    # Main content area
    input_type = st.radio("Input Type", ["Text", "Website URL", "Document"])
    
    if input_type == "Text":
        user_input = st.text_area("Enter your text")
    elif input_type == "Website URL":
        url = st.text_input("Enter URL")
        if url:
            try:
                response = requests.get(url)
                user_input = response.text
            except Exception as e:
                st.error(f"Error fetching URL: {str(e)}")
                user_input = ""
    else:  # Document
        uploaded_file = st.file_uploader("Upload document", type=["pdf", "txt"])
        if uploaded_file:
            user_input = extract_text_from_file(uploaded_file)
        else:
            user_input = ""
    
    if st.button("Generate Positive Perspective"):
        if not api_key:
            st.error("Please enter an API key")
            return
        
        if not user_input:
            st.error("Please provide input text, URL, or document")
            return
        
        client = initialize_llm_client(api_key, model_type)
        if not client:
            return
        
        # Prepare prompt with context
        prompt = f"""
        Input: {user_input}
        
        Negative aspects identified: {', '.join(negative_aspects) if negative_aspects else 'None specified'}
        
        Please provide a positive perspective on this situation
        {' with citations' if include_citations else ''}.
        """
        
        with st.spinner("Generating positive perspective..."):
            response = get_llm_response(client, prompt, model_type, include_citations)
            st.session_state.chat_history.append({"input": user_input, "response": response})
            st.markdown("### Positive Perspective")
            st.write(response)
    
    # Display chat history
    if st.session_state.chat_history:
    
        st.markdown("### Previous Interactions")
        for idx, interaction in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Interaction {len(st.session_state.chat_history) - idx}"):
                st.markdown("**Input:**")
                st.write(interaction["input"])
                st.markdown("**Response:**")
                st.write(interaction["response"])

if __name__ == "__main__":
    main()

