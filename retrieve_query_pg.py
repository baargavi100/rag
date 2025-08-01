import os
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferMemory

# Step 1: Embeddings + LLM + PGVector
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = OllamaLLM(model="mistral")  # Better for reasoning than tinyllama

CONNECTION_STRING = "postgresql+psycopg2://postgres:baargavi123@localhost:5432/vector_db"

# Load PGVector vector store
vectorstore = PGVector(
    collection_name="pdf_vectors",
    connection_string=CONNECTION_STRING,
    embedding_function=embedding_model
)

# Step 2: Setup retriever and convert to tool
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})



retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="search_pdf",
    description="Use this tool to answer questions by searching PDF documents related to interviews."
)

# Step 3: Setup conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Step 4: Initialize Agent
agent = initialize_agent(
    tools=[retriever_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    handle_parsing_errors=True,  # Prevent tool format errors from stopping the agent
    verbose=True
)

# Step 5: Interactive chat loop
print("\n🤖 Agent ready. Ask questions about the PDFs.")
while True:
    query = input("\nYou: ")
    if query.lower() in ("exit", "quit"):
        break
    try:
        response = agent.run(query)
        print(f"\nAgent: {response}")
    except Exception as e:
        print("Error:", e)
