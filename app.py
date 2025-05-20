import streamlit as st
import requests
import PyPDF2
import io
from typing import Optional
import openai
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import re
import time
from bs4 import BeautifulSoup

# Configure page settings
st.set_page_config(
    page_title="Dr. Spin - Putting a Positive Spin on Life",
    page_icon="src/img/dr-spin-favicon-64x64.png",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    /* Button styling */
    .stButton>button {
        background-color: #b8625a;
        color: white;
        border-radius: 20px;
    }
    
        
    /* Reduce heading padding */
    .block-container div[data-testid="stMarkdownContainer"] h3 {
        margin-bottom: 0;  /* Reduce bottom margin */
        padding-bottom: 0;      /* Remove bottom padding */
    }
            
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'include_citations' not in st.session_state:
    st.session_state.include_citations = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_citations' not in st.session_state:
    st.session_state.show_citations = False
if 'negative_sentiments' not in st.session_state:
    st.session_state.negative_sentiments = []
if 'refresh' not in st.session_state:
    st.session_state.refresh = False

SYSTEM_PROMPT = """You are Dr. Spin, an AI assistant specialized in finding positive perspectives in challenging news and situations. Your goal is to craft thoughtful, fact-based, and empathetic responses that validate the user's concerns while highlighting opportunities for optimism and action. You are provided an input in the following format:

{
    user_input: a free text input from the user that describes a situation they find negative. This may include a web link that describes the negative situation.
    uploaded_file: a file uploaded by the user that describes a situation they find negative.
    negative_sentiments: an optional list of negative sentiments that the user has identified.
    include_citations: a boolean that indicates if the user has requested citations in the response.  If this is TRUE, you much include citations.
}

Your task is to analyze the input and provide a response that includes:
- Validation of the user’s feelings and acknowledgment of their concerns.
- A big-picture perspective demonstrating how the situation may not be as dire as it seems, with evidence of progress or improvement in related areas.
- Actionable insights that show opportunities for the user or others to make a positive impact or navigate the situation constructively.
- Optional citations, if requested, to support your points with reliable sources.

### Response Format
Your response should be in the following JSON-compatible format. Use Markdown formatting in the `positive_perspective`. Ensure the content inside the curly braces can be parsed with `json.loads()`:

{
    "positive_perspective": "A detailed and empathetic response that provides a positive, fact-based outlook and actionable suggestions.",
    "negative_sentiments": "A REQUIRED list of 3-5 negative sentiments either identified by the user or inferred from the context.  Include this in every repsonse. IT IS REQUIRED.",
    "citations": "If the include_citations input is True, create a dictionary of citations with the format {citation_title: citation_url}. If citations are included, the positive_perspective should have direct quotations from the linked citation. If include_citations is false, this can be an empty dictionary."
}

### Guidelines for Crafting Responses
1. **Start with Empathy and Validation**: Consider what point-of-view the user must have to consider this negative.  Frame your entire response assuming this negative point-of-view.  Open your response by acknowledging the user's concerns and showing understanding of their emotions. Clearly state that the positive perspective is meant to complement their feelings, not dismiss them.
   
2. **Highlight Progress and Context**: Use a broader, fact-based view to show how the situation might reflect ongoing improvements in the world. For example, point out long-term trends, global advancements, or areas where humanity is making progress.

3. **Show Opportunities for Action**: Emphasize how the situation presents opportunities for the user or others to act, whether by contributing to solutions, finding personal growth, or supporting broader improvements. Highlight that positive change is often driven by challenges.

4. **Incorporate Evidence**: Use factual data to support your positive outlook. Use statistics where possible. If the user requests citations, include reliable sources to back up your points and include direct quotes from the citation.

5. **Anticipate Negative Sentiments**: If the user hasn’t listed specific negative sentiments, infer up to five broad concerns based on the input and address these in your response.  THIS IS REQUIRED.

6. **Close on a Hopeful Note**: End your response by reinforcing the user's agency or by leaving them with a hopeful takeaway.

7. **Provide Citations**: Provide reputable sources to support the positive perspective.  Ensure that the sources come from real website and content.  Use the format {citation_title: citation_url}.

---

By using these steps and strategies, craft a response that reflects an empathetic, fact-based, and actionable perspective while fostering an optimistic and realistic worldview. Remember: The goal is to align with the principles of *Factfulness*—providing clarity, positivity, and empowerment in the face of challenges.

        """

def initialize_llm_client(api_key: str, model_provider: str) -> Optional[any]:
    """Initialize and configure a Language Model client with the provided credentials.

    Args:
        api_key (str): Authentication key for the selected model provider
        model_provider (str): Name of the LLM provider ("Google - Gemini" or "OpenAI - ChatGPT")

    Returns:
        Optional[any]: Configured client instance or None if initialization fails
    """
    try:
        if model_provider == "Google - Gemini":
            genai.configure(api_key=api_key)
            return genai
        elif model_provider == "OpenAI - ChatGPT":
            return openai.Client(api_key=api_key)
        return None
    except Exception as e:
        st.error(f"Error initializing {model_provider} client: {str(e)}")
        return None

def start_gemini_chat(client) -> Optional[any]:

    tools = [extract_url_content]
    model = client.GenerativeModel('gemini-1.5-flash', 
                                    system_instruction=SYSTEM_PROMPT,
                                    tools=tools,
                                    generation_config={
                                                        "temperature":1.7,
                                                        "max_output_tokens":2048}
                                                        )
    chat = model.start_chat(enable_automatic_function_calling=True)
        
    return chat


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def send_gemini_message(chat, input_text: str) -> str:
    """Send a message to Google's Gemini model and get the response with retry logic.
    
    Args:
        chat (genai.GenerativeModel): An initialized Gemini chat model instance
        input_text (str): The text message to send to the model
        
    Returns:
        Optional[genai.types.GenerateContentResponse]: The model's response if successful, None if failed
        
    Raises:
        ValueError: If input_text is empty or chat is not initialized
        genai.types.GoogleGenerativeAIError: For Gemini-specific API errors
        requests.exceptions.RequestException: For network-related errors
    """
    try: 
        response = chat.send_message(input_text)
        # Validate response
        if not response or not response.text:
            raise ValueError("Empty response received from model")
            
        return response

    except (genai.types.GoogleGenerativeAIError, requests.exceptions.RequestException) as e:
        st.error(f"API or network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error when getting response from Gemini: {str(e)}")
        return None


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
def send_open_ai_chat(client, prompt) -> Optional[any]:
    """Send a chat message to OpenAI's API and handle URL content extraction if needed.

    Args:
        client (Any): An initialized OpenAI client instance
        prompt (str): The user's input prompt to process

        Optional[ChatCompletion]: The model's response if successful, None if failed

    Raises:
        ValueError: If client is not initialized or prompt is empty
        openai.OpenAIError: For OpenAI-specific API errors
        requests.exceptions.RequestException: For network-related errors
        json.JSONDecodeError: For JSON parsing errors
    """

    extract_url_content_tool = {
            "name": "extract_url_content",
            "description": "Extract the content of a URL from the input string to understand what is on the website.  Omly use this tool if there are URLs in the user's input. \
                A URL should start with https://www or end with .com, .org, .net, .io, .gov, or .edu.  If the URL is not found, do not use this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to extract content from."
                    },
                },
            }
        }

    tools = [{
        "type": "function",
        "function": extract_url_content_tool
    }]

    messages = [
        {
            "role": "system", "content": SYSTEM_PROMPT
        },
        {
            "role": "user", "content": prompt
         }
    ]

    response  = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        response_format={ "type": "json_object"},
        temperature=1,
        frequency_penalty=1.2
    )

    if response.choices[0].message.tool_calls:
        
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = tool_call.function.arguments
        url = json.loads(arguments).get('url')
        content = extract_url_content(url)
        
        function_call_result_message = [response.choices[0].message,
            {
            "role": "tool",
            "content": json.dumps({
                "url": url,
                "content": content
            }),
            "tool_call_id": response.choices[0].message.tool_calls[0].id    
            }]
        
        messages.extend(function_call_result_message)
        
        response  = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format={ "type": "json_object"},
                    temperature=1,
                    frequency_penalty=1.2
                    )

    return response


@st.fragment
def select_negative_sentiments(response):
    st.pills(
                "negative_sentiment_pills",
                options= [sentiment.capitalize() for sentiment in response['negative_sentiments']], 
                selection_mode="multi",
                key="negative_sentiments",
                label_visibility='hidden'
            )
    
@st.fragment
def refresh_with_selected_negative_sentiments():
    if st.button("🔄 Refresh with selected negative sentiments", use_container_width=True):
            st.session_state.refresh = True
            st.rerun()


def display_chat_message(role: str, content: str, avatar: str = None):
    """Display a chat message with the specified role and content."""
    with st.chat_message(role, avatar=avatar):
        st.write(content)


def extract_url_content(input_string: str) -> str:
    """ Extract the content of a URL from the input string to understand what is on the website.
     
      Args:
        input_string: The input string containing a URL.
        
      Returns:
        str: The content of the URL, truncated to 5000 characters.
    """
    url_pattern = re.compile(r"https?://[^\s]+")
    try:
        match = url_pattern.search(input_string)
        if not match:
            return ""
        url = match.group(0)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract readable content (simplified; can be improved)
        paragraphs = soup.find_all('p')
        content = "\n".join(p.get_text() for p in paragraphs)
        return content[:5000] # Truncate to 5000 characters to fit token limits
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return ""
    except Exception as e:
        print(f"Error extracting URL content: {str(e)}")
        return ""


def parse_response(response, model_provider: str) -> dict:
    """Parse the response text into a structured format."""
    try:
        message_placeholder = st.empty()
        
        if model_provider == "Google - Gemini":
            final_response = response.text
        else:  # OpenAI
            final_response = response.choices[0].message.content

        # Extract JSON if it's wrapped in markdown or has extra characters
        if final_response[0] != '{' or final_response[-1] != '}':
            # Find the first { and last }
            start_idx = final_response.find('{')
            end_idx = final_response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                final_response = final_response[start_idx:end_idx]
            
        final_response = json.loads(final_response)

        return final_response
    
    except json.JSONDecodeError:
        st.error("Failed to parse LLM response as JSON")
        return {
            'positive_perspective': response,
            'negative_sentiments': [],
            'citations': []
        }
    except Exception:
        message_placeholder.markdown(response)

def stream_response(response):
    """Basic function to simulate a stream response for LLM's response text."""
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

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
    # Sidebar
    with st.sidebar:
        st.markdown("## How to Use")
        
        st.markdown("""
        1. Select your preferred model provider.  Please be familiar with any costs of the [Gemini API](https://ai.google.dev/pricing#1_5flash-8B) or [OpenAI API](https://openai.com/api/pricing/).  At this time, Gemini offers a better free tier.
        2. Enter your API key.  You can find your API key [here](https://platform.openai.com/api-keys) for OpenAI or [here](https://ai.google.dev/gemini-api/docs/api-key) for Google Gemini.
        3. Tell Dr. Spin what's on your mind.  You can include links or upload a document.  
        4. Generate a positive spin!
        5. Refine the response by selecting which negative sentiments you'd like to address.
        """)
        
        model_provider = st.selectbox("Select Model Provider", ["Google - Gemini", "OpenAI - ChatGPT"],
                                  help="""Select the provider you'd like to use for generating the positive spin.  Current models used are:
                                  \n - Google: Gemini 1.5 Flash-8B
                                  \n - OpenAI: gpt-4o-mini""",
                                  label_visibility="visible")
        
        api_key = st.text_input("API Key", type="password")
        
        st.markdown("### About")
        st.markdown("""One of my favorite quotes is:

> _"When you are a pessimist and the bad thing happens, you live it twice.  Once when you worry about it, and the second time when it happens”_ – [Amos Tversky](https://en.wikipedia.org/wiki/Amos_Tversky)

It reinforces a principle I think can help improve anyone's life: the power of positive thinking. I don't prescribe to this mantra in the light of a preachy, self-help book, but rather look at it as a method for dealing with an objective reality. 

Often, the negative news or outcomes that can make someone feel down are either (1) an event that has already occurred (or will inevitability occur) and can't be changed or (2) is an event that may come, but can be altered through action.  In both of these situations, a negative viewpoint won't do anything except make the path forward more difficult. A positive perspective can help you see negative news in a different light, and inspire an impactful change moving forward.

Applying this frame of mind is difficult. For some reason, we (as humans) are inclined to be [more driven to negative news](https://www.nature.com/articles/s41562-023-01538-4) (and news media companies are aware of this). So, I created Dr. Spin to help me put a positive spin on life.

Grab your rose-colored glasses and let's find some silver linings!
""")
        
        st.markdown("Made by [Ryan Lynch](https://ryanlynch.me)")

    # Top navigation
    col1, col2, col3 = st.columns([5, 1, 1], vertical_alignment='center')
    with col3:
        st.link_button("📚 Blog", "https://ryanlynch.me/Technical+Projects+Dashboard/Dr.+Spin/Dr.+Spin+-+a+positive+spin+on+life+using+AI")

    # Main content
    st.image("src/img/dr-spin-logo-horizontal-slim.png", use_container_width=False)

    # Input section
    st.session_state.user_input = st.text_area("Tell Dr. Spin what's on your mind.", value=st.session_state.user_input, height=200, label_visibility='hidden',
                                max_chars=2048,
                                placeholder="Tell Dr. Spin some negative news or a problem you're facing.  You can include links or upload a document." + \
                                    " If you'd like, expand on why you view this news as negative, or let Dr. Spin determine the negative sentiment for you.")
    with st.expander("**Optional: Upload Document**", icon="📄"):
        uploaded_file = st.file_uploader("(Optional) Upload Document", type=["pdf", "txt"], 
                                        help="Upload Document", 
                                        label_visibility='hidden', 
                                        key="uploaded_file"
                                        )
    col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment='center')
    with col1:
        if st.checkbox("Include citations in positive spin?", value=st.session_state.include_citations,
            help="Enable to include source citations in the positive spin (uses more model tokens). AI citations may be made up and false!  Always double check."):
            st.session_state.include_citations = True
        else:
            st.session_state.include_citations = False

    with col2:
        generate_button = st.button("🌟 Positive Spin", use_container_width=True, 
                                    type="tertiary")
    with col3:
        if st.button("🗑️  Reset", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.negative_sentiments = []
            st.session_state.refresh = False
            st.session_state.user_input = ""
            st.session_state.include_citations = False
            st.rerun()


    if generate_button:
        if not api_key:
            st.error("Please enter an API key.")
            return
    
        if not st.session_state.user_input and not uploaded_file:
            st.error("Please provide input text or a document.")
            return
         

    if (generate_button and st.session_state.user_input) or (generate_button and uploaded_file) or (generate_button and st.session_state.chat_history) or (st.session_state.refresh):
        st.session_state.refresh = False
        with st.status("Welcome to the spin zone...", expanded=True) as status:
            client = initialize_llm_client(api_key, model_provider)
            if not client:
                st.error("Error connecting to selected model. Please check your API key and try again.")
                return
            
            status.update(label="Welcome to the spin zone...", state="running", expanded=True)

            user_input = st.session_state.user_input

            if model_provider == "Google - Gemini":
                url_pattern = re.compile(r"https?://[^\s]+")
                if url_pattern.search(user_input):
                    user_input += "\n\n Consider the following website content: " + extract_url_content(user_input)            

            if uploaded_file:
                user_input += "\n\n Consider the following document content: " + extract_text_from_file(uploaded_file)
            
            # Prepare prompt with context
            prompt = f"""
            Please provide a positive perspective on this situation.
            {{
                user_input: {st.session_state.user_input},
                uploaded_file: {uploaded_file},
                negative_sentiments: {st.session_state.negative_sentiments},
                include_citations: {st.session_state.include_citations}
            }}
            """

            if model_provider == "Google - Gemini":
                chat = start_gemini_chat(client)
                response = send_gemini_message(chat, prompt)
            else:
                response = send_open_ai_chat(client, prompt)

            if response is None:
                return

            # Parse and display response
            parsed_response = parse_response(response, model_provider)
            
            # Update negative sentiments
            st.markdown("### I understand this must make you feel...")
            select_negative_sentiments(parsed_response)
            
            # Results section
            st.markdown("### 🌟 But, let's look at the bright side...")

            # Display response as stream
            st.write_stream(stream_response(parsed_response['positive_perspective']))

            if st.session_state.include_citations:
                st.markdown("### 🔗 Citations")
                if parsed_response.get('citations'):
                    for title, url in parsed_response['citations'].items():
                        st.markdown(f"- [{title}]({url})")
                else:
                    st.markdown("Dr. Spin could not find any relevant citations for this topic.")
                
            status.update(label="Silver linings found!", state="complete", expanded=True)
            

        # Step 3: Refresh button
        refresh_with_selected_negative_sentiments()

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### Previous Interactions")
            for idx, interaction in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Interaction {len(st.session_state.chat_history) - idx}"):
                    display_chat_message("user", interaction["input"])
                    display_chat_message("assistant", interaction["response"], avatar="src/img/dr-spin-favicon-64x64.png")


        st.session_state.chat_history.append({
            "input": user_input, 
            "response": parsed_response['positive_perspective']
                })

if __name__ == "__main__":
    main()