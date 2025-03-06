import streamlit as st
from langchain.llms import OpenAI  
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Function to generate a summarized response
def generate_response(txt, api_key):
    try:
        # Instantiate the LLM model
        llm = OpenAI(temperature=0, openai_api_key=api_key)

        # Improved text splitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(txt)

        # Create documents from text
        docs = [Document(page_content=t) for t in texts]

        # Text summarization using Map-Reduce approach
        chain = load_summarize_chain(llm, chain_type='map_reduce')
        return chain.run(docs)
    
    except Exception as e:
        return f"Error: {str(e)}"

# Page title
st.set_page_config(page_title='🦜🔗 Text Summarization App')
st.title('🦜🔗 Text Summarization App')

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
with st.form('summarize_form', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password')
    submitted = st.form_submit_button('Submit')

    if submitted:
        if not openai_api_key.startswith('sk-'):
            st.error("Invalid OpenAI API key. Please enter a valid key.")
        elif not txt_input.strip():
            st.error("Please enter some text for summarization.")
        else:
            with st.spinner('Summarizing...'):
                response = generate_response(txt_input, openai_api_key)
            st.success("Summary:")
            st.info(response)

# Instructions for getting an OpenAI API key
st.subheader("Get an OpenAI API Key")
st.write("Follow these steps to get your API key:")
st.markdown("""
1. Go to [OpenAI API Keys](https://platform.openai.com/account/api-keys).
2. Click on **+ Create new secret key**.
3. Copy and **paste the key** in the input box above.
""")
