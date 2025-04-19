from pathlib import Path
from dotenv import load_dotenv
from os import getenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load and split PDF
pdf_path = Path(__file__).parent / "react interview.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)

# Embedding model
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY
)

# Inject into Qdrant once, comment out after
# vector_store = QdrantVectorStore.from_documents(
#     documents=split_docs,
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embedder
# )

# Use existing collection
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

# Tool: Ask PDF
def ask_pdf(question: str):
    print("🔨 Tool Called: ask_pdf", question)
    search_result = retriever.similarity_search(query=question, k=4)
    context = "\n\n".join([doc.page_content for doc in search_result])
    return context

# Tool registry
avaiable_tools = {
    "ask_pdf": {
        "fn": ask_pdf,
        "description": "Takes a question and answers it using the uploaded PDF data."
    }
}

# System prompt
system_prompt = f"""
You are a helpful assistant who solves user queries using available tools.
You operate in a loop: plan → action → observe → output.

Rules:
- Only do one step at a time.
- Always follow the JSON format.
- Use a tool when needed.
- Wait for tool result (observe) before final answer.

JSON Output Format:
{{
    "step": "plan" | "action" | "observe" | "output",
    "content": "your explanation or result",
    "function": "tool name if step is 'action'",
    "input": "input for the tool"
}}

Available Tools:
{json.dumps({k: v['description'] for k, v in avaiable_tools.items()}, indent=4)}

Example:
User: What is useState?

1. Plan:
{{ "step": "plan", "content": "The user is asking about useState. I will search in the PDF using ask_pdf." }}

2. Action:
{{ "step": "action", "function": "ask_pdf", "input": "what is useState in React?" }}

3. Observe:
{{ "step": "observe", "content": "React's useState is a hook that lets you..." }}

4. Output:
{{ "step": "output", "content": "The useState hook lets you add state to functional components..." }}
"""

# === Start Chat ===
messages = [
    { "role": "system", "content": system_prompt }
]

while True:
    user_query = input("Ask Your Question Here > ")
    if user_query.lower() in ["exit", "quit"]:
        break

    messages.append({ "role": "user", "content": user_query })

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=messages
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({ "role": "assistant", "content": json.dumps(parsed_output) })

        step = parsed_output.get("step")

        if step == "plan":
            print(f"🧠 Plan: {parsed_output.get('content')}")
            continue

        if step == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if avaiable_tools.get(tool_name):
                result = avaiable_tools[tool_name]["fn"](tool_input)
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({ "step": "observe", "content": result })
                })
                continue

        if step == "output":
            print(f"🤖 Answer: {parsed_output.get('content')}")
            break
