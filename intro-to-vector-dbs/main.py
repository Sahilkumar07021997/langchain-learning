import os
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file (e.g., OPENAI API Key, Pinecone Index name)
load_dotenv()

if __name__ == '__main__':
    print("Retrieving...")

    # Initialize the OpenAI embedding model
    embeddings = OpenAIEmbeddings()

    # Initialize the OpenAI chat language model (e.g., GPT-3.5 or GPT-4)
    llm = ChatOpenAI()

    # Define the query you'd like to answer
    query = "what is pinecone in machine learning?"

    # Direct LLM call without using any external document/context
    # This simply wraps the query as a static prompt and sends it to the LLM
    chain = PromptTemplate.from_template(template=query) | llm

    # Invoke the LLM with an empty input since the prompt is static
    result = chain.invoke(input={})
    print(f"Without retrieval: {result.content}")  # Print the LLM-generated answer

    # Set up a vector store for retrieval (using Pinecone)
    # It fetches the index name from environment variables
    vector_store = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"),
        embedding=embeddings
    )

    # Load a pre-defined LangChain prompt template for retrieval-augmented QA from the LangChain Hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create a document processing chain that will “stuff” retrieved documents into the prompt
    # and send them along with the query to the LLM
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_qa_chat_prompt
    )

    # Create the full Retrieval-Augmented Generation (RAG) chain
    # This chain retrieves relevant documents from Pinecone and passes them through the combine_docs_chain
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    # Invoke the RAG chain with the query
    result = retrieval_chain.invoke(input={"input": query})

    # Print the final answer generated using retrieved context
    print(f"RAG result: {result.get('answer')}")