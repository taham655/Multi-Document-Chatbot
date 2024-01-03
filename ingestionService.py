import os 
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import  HuggingFaceHub, Replicate
from langchain.document_transformers import LongContextReorder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import docx
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


#llama 13b id
#"meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"

#llama 70b id
#"meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
class IngestionService:
    def __init__(self, documents_directory, persist_directory='db'):
        self.directory_path = documents_directory
        self.persist_directory = persist_directory
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.chunk_size = 1000
        self.memory = ConversationBufferWindowMemory(
            k=2,
            memory_key='chat_history', return_messages=True
            )
        self._setup()
   
        self.llm = Replicate(
                model="meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
                model_kwargs={"temperature": 0.1, "max_length": 2000, "top_p": 1},
            )   
        
        # HuggingFaceHub(
        #         repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature": 0.1}
        #     )

        
        
    def get_pdf_text_chunks(self, file_path):
        """Extract text from a PDF file and return it in chunks of 500 characters."""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        for i in range(0, len(text), self.chunk_size):
                            chunk = text[i:i+self.chunk_size] + f" - Page {page_num + 1}, {os.path.basename(file_path)}"
                            chunks.append(chunk)
                print(f"{os.path.basename(file_path)}: {len(chunks)} chunks")
        except Exception as e:
            print(f"{os.path.basename(file_path)}: can't chunk this file - corrupt or unsupported format")

        return chunks


    def get_docx_text_chunks(self, file_path):
        """Extract text from a DOCX file and return it in chunks of 500 characters."""
        chunks = []
        doc = docx.Document(file_path)
        for page_num, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text
            if text:
                for i in range(0, len(text), self.chunk_size):
                    chunk = text[i:i+self.chunk_size] + f" - Page {page_num + 1}, {os.path.basename(file_path)}"
                    chunks.append(chunk)
        return chunks

    def load_files_and_chunk_text(self, directory):
        print("Chunking..")
        """Load all files from a directory, extract text, and return it in chunks of 500 characters."""
        self.all_chunks = []
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if file_name.lower().endswith('.pdf'):
                self.all_chunks.extend(self.get_pdf_text_chunks(file_path))
            elif file_name.lower().endswith('.docx'):
                self.all_chunks.extend(self.get_docx_text_chunks(file_path))
        return self.all_chunks


    def _setup(self):
        
        prompt_template = """
        Use the following pieces of information to answer the user's question. At the end of the asnwer specify the page number and the document name, they are already part of the context just take them out and mention them use the format (Page number(s), File name). If there are more than one page then mention all the pages like 'Page 1,2 -3' and than the source.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 

        Context: {context}
        Question: {question}
        
        Don't say things like "Sure, I'd be happy to help! Based on the information provided" or anything similar just get straight to the asnwer.
        Do not say anything else just give the asnwser in the format mentioned.
        Please don't make your own answers, if you say something that is not provided in the context i will lose my job.
        Only return the helpful answer below and nothing else. If the answer of the question is not in the context, just say "I don't know". Please do not make up an answer or try to give it using your own knowledge.
        Dont justidy your answer, just give the answer.
        You must always return the answer in the format: "<answer> (Page number(s), File name)". 
        """
        self.prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])  # Creating a prompt template object

        # Load embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        if not os.path.exists(self.persist_directory):
            texts = self.load_files_and_chunk_text(self.directory_path)
            print("Embedding..")
            vectordb = Chroma.from_texts(texts, self.embeddings , persist_directory=self.persist_directory)
            vectordb.persist()



    def retrieve_documents(self):

        access_db = Chroma(persist_directory=self.persist_directory, 
        embedding_function=self.embeddings)
        self.retriever = access_db.as_retriever(search_kwargs={"k": 5})
       
        # compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=self.retriever)

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt}
    )  
        return conversation_chain