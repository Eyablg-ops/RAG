import os            # tools to interact with the operating system (here: delete a file)
import tempfile      # tools to create temporary files
from pathlib import Path   # not used in this file (you can remove it)

# TextLoader: a LangChain class that reads a .txt file and converts it to Documents
from langchain_community.document_loaders import( TextLoader, PyPDFLoader,WebBaseLoader,DirectoryLoader)

from dotenv import load_dotenv

load_dotenv()  # loads your .env keys. Not needed here, since no model is called


def load_text_file():
    # --- PART 1: create a temporary .txt file for the demo ---

    # "with ... as temp_file" opens the file and closes it automatically at the end.
    # delete=False  -> don't delete it when closed (we still need to read it)
    # suffix=".txt" -> the file name will end with .txt
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        # b"..." means "bytes": the file is written in binary mode
        # \n is a line break (so the text is on 2 lines)
        temp_file.write(b"Hello, this is a sample text file.\nThis file is used to")
        # .name is the full path of the temporary file on your Mac
        temp_file_path = temp_file.name

    # --- PART 2: load the file ---

    # try / finally: the "finally" block ALWAYS runs, even if there is an error
    try:
        loader = TextLoader(temp_file_path)  # prepare a loader for this file path
        documents = loader.load()            # read the file -> returns a LIST of Documents
        # --- PART 3: display the result methode1 ---
        print(f"Loaded {len(documents)} document(s)")
        print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")
        
        # --- PART 3: display the result  methode 2 ---
        for doc in documents:        # go through each Document in the list
            print(doc)               # prints the whole Document (text + metadata)
            print(doc.page_content)  # prints only the text
            print(doc.metadata)      # prints only the metadata
    finally:
        # --- PART 4: cleanup ---
        os.remove(temp_file_path)    # delete the temporary file

def pdf_loader(pdf_path):

    loader = PyPDFLoader(pdf_path)  # prepare a loader for this PDF file path
    documents = loader.load()        # read the file -> returns a LIST of Documents

    # --- PART 2: display the result ---
    print(f"Loaded {len(documents)} document(s)")

    for i,doc in enumerate(documents):
        print(f"Document {i+1}:")
        print(f"Content: {doc.page_content[:100]}...")  # preview first 100 chars
        print(f"Metadata: {doc.metadata}")
# Entry point: runs only when you launch this file directly
if __name__ == "__main__":
    load_text_file()
    pdf_loader("docs/langchain_demo.pdf")  # replace with your PDF path