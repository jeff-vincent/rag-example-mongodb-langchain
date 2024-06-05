import os
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class DataIngestRag:
    # This class handles ingesting unstructured text data into MongoDB such that it is searchable
    # for vector similarity to an arbitrary user input

    def __init__(self) -> None:
        self.db_name = os.environ.get('DB_NAME')
        self.index_name = os.environ.get('INDEX_NAME')
        self.collection_name = os.environ.get('COLLECTION_NAME')
        self.mongo_password = os.environ.get('MONGO_PASSWORD')
        self.mongo_user = os.environ.get('MONGO_USER')
        self.mongo_connection_string = \
            f'mongodb+srv://{self.mongo_user}:{self.mongo_password}@vectorsearch.{self.db_name}.mongodb.net/?retryWrites=true'
        self.mongo_client = MongoClient(self.mongo_connection_string)
        self.mongo_collection = self.mongo_client[self.db_name][self.collection_name]
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.chunked_text = None

    def chunk_text(self, input_text):
        self.chunked_text = self.text_splitter.split_documents(input_text)
        
    def upload_to_mongodb(self):
        MongoDBAtlasVectorSearch.from_documents(
        documents=self.chunked_text,
        embedding=OpenAIEmbeddings(),
        collection=self.mongo_collection,
        index_name=self.index_name)

class QueryRag:
    # This class accepts an arbitrary user input, a question, and searches MongoDB 
    # based on vector similarity for related documents. It then passes the user's 
    # input as the value `question` to OpenAI along with the Vector search results, 
    # which are passed as the value `context`.

    def __init__(self) -> None:
        self.index_name = os.environ.get('INDEX_NAME')
        self.db_name = os.environ.get('DB_NAME')
        self.collection_name = os.environ.get('COLLECTION_NAME')
        self.mongo_password = os.environ.get('MONGO_PASSWORD')
        self.mongo_user = os.environ.get('MONGO_USER')
        self.mongo_connection_string = \
            f'mongodb+srv://{self.mongo_user}:{self.mongo_password}@vectorsearch.abc.mongodb.net/?retryWrites=true'
        self.mongo_client = MongoClient(self.mongo_connection_string)
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.model = ChatOpenAI(openai_api_key=self.openai_api_key, model="gpt-3.5-turbo")
        self.parser = StrOutputParser()
        self.template = """
                        Answer the question based on the context below. If you can't 
                        answer the question, reply "I don't know".

                        Context: {context}

                        Question: {question}
                        """

        self.prompt = ChatPromptTemplate.from_template(self.template)

    def _create_vector_search(self):
        vector_search = MongoDBAtlasVectorSearch.from_connection_string(
            self.mongo_connection_string,
            f"{self.db_name}.{self.collection_name}",
            OpenAIEmbeddings(),
            index_name=self.index_name
        )
        return vector_search

    def _search_mongodb_for_related_text(self, question, top_k):
        vector_search = self._create_vector_search()
        return vector_search.similarity_search_with_score(query=question,k=top_k,)

    def populate_and_submit_prompt(self, question):
        related_text = self._search_mongodb_for_related_text(question)
        self.prompt.format(context=related_text, question=question)
        result = self.prompt | self.model | self.parser

        return result
