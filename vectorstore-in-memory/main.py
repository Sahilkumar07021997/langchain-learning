import os
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables (like OpenAI API Key)
load_dotenv()

if __name__ == '__main__':
    print("---------------------vectorstore-in-memory practice.------------------------------")

    # Path to the PDF file to load and process
    pdf_path = r"C:\DRIVE D\COURSES -DONE\langchain-learning\vectorstore-in-memory\ReAct_paper.pdf"

    # Check if the PDF exists before proceeding
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    # Load the PDF and convert it into LangChain Document objects
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split documents into chunks for better indexing and retrieval
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,         # Max characters per chunk
        chunk_overlap=30,        # Overlap to retain context
        separator="\n"           # Split based on newlines
    )
    docs = text_splitter.split_documents(documents=documents)
    print(f"Number of documents: {len(docs)}")

    # Generate embeddings using OpenAI and create a FAISS vectorstore from chunks
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

    # Save the FAISS vectorstore to disk locally (for reuse)
    vectorstore.save_local("skd_faiss_index_local")

    # Load the previously saved FAISS vectorstore (simulate production-like reuse)
    new_vectorstore = FAISS.load_local(
        "skd_faiss_index_local",
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # Only for local testing! Avoid in production.
    )

    # Initialize the LLM for answering questions (with temperature=0 for deterministic output)
    llm = OpenAI(temperature=0)

    # Load a pre-defined prompt template for RAG from LangChain Hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create a chain to combine retrieved documents into a single prompt
    combine_docs_chain = create_stuff_documents_chain(
        llm,
        retrieval_qa_chat_prompt
    )

    # Create a Retrieval-Augmented Generation chain using the vectorstore retriever
    retrieval_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    # Invoke the chain with a question to answer using retrieved context from the PDF
    res = retrieval_chain.invoke({"input": "who is the author of this paper"})

    # Print the final generated answer
    print(f"Result: {res['answer']}")
