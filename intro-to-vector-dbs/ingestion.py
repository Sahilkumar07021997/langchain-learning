import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables like OPENAI_API_KEY and INDEX_NAME from a .env file
load_dotenv()

if __name__ == '__main__':
    print("Loading..., Splitting..., Embedding..., Storing/Ingesting....")

    # Load a text document with UTF-8 encoding
    # Note: Use double backslashes `\\` or a raw string `r""` to avoid escape sequence errors on Windows
    loader = TextLoader(r"C:\DRIVE D\COURSES -DONE\Langchain\intro-to-vector-dbs\mediumblog1.txt", encoding="utf-8")
    document = loader.load()  # Load the contents of the file into a document object

    print("Splitting the document into chunks...")
    # Create a text splitter with a max chunk size of 1000 characters and no overlap
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)  # Split the loaded document into smaller chunks
    print(f"Created {len(texts)} chunks!")

    print("Generating embeddings for each chunk...")
    # Initialize the OpenAI embedding model using the API key from environment variables
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))

    print("Storing embeddings into Pinecone vector database...")
    # Store the document chunks into Pinecone with the generated embeddings
    # The index name is fetched from the environment variable 'INDEX_NAME'
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get('INDEX_NAME'))

    print("Ingestion complete! Vector store is ready.")
