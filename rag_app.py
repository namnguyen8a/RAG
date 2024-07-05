import os
from dotenv import load_dotenv
import gradio as gr
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from groq import Groq
import time

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Constants
RAG_PROMPT = """Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên dữ liệu được cho.
Nếu dữ liệu được cho không liên quan đến câu hỏi, vui lòng trả lời "Tôi không biết"
---
Dữ liệu: {context}
---
Câu hỏi: {question}
---
Trả lời:"""

# Global variable to store the uploaded file
uploaded_file = None

# Function to process uploaded PDF
def process_pdf(file_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    loader = PyPDFLoader(file_path)
    splits = loader.load_and_split(text_splitter)
    emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents=splits, embedding=emb)
    return vectorstore

# Function to predict based on user input and history
def predict(message, history, file):
    global uploaded_file
    if file is not None:
        uploaded_file = file.name
    
    if uploaded_file is None:
        return "No PDF file uploaded or failed to process the file."
    
    vectorstore = process_pdf(uploaded_file)
    if vectorstore is None:
        return "Failed to process the PDF file."
    
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    
    docs = vectorstore.similarity_search(message)
    if not docs:
        context = "No relevant context found in the document."
    else:
        context = docs[0].page_content
    
    prompt_message = RAG_PROMPT.format(context=context, question=message)
    history_openai_format.append({"role": "user", "content": prompt_message})
    
    # Measure the start time
    start_time = time.time()
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt_message,
            }
        ],
        model="llama3-8b-8192",
    )
    
    # Measure the end time
    end_time = time.time()
    time_taken = end_time - start_time

    # Get the generated response
    partial_message = response.choices[0].message.content

    return partial_message

# Gradio interface function
def chat_interface(file, history, message):
    response = predict(message, history, file)
    history.append((message, response))
    return history, history

# Define the Gradio interface
interface = gr.Interface(
    fn=chat_interface,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.State([]),  # Single state input
        gr.Textbox(label="User Input")
    ],
    outputs=[
        gr.Chatbot(),  # Chatbot output
        gr.State([])   # Single state output
    ],
    live=False  # Set live to False to prevent immediate processing
)

# Launch the interface
interface.launch()
