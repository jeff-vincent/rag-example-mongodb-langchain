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
        self.mongo_password = os.environ.get('MONGO_PASSWORD')
        self.mongo_user = os.environ.get('MONGO_USER')
        self.mongo_connection_string = \
            f'mongodb+srv://{self.mongo_user}:{self.mongo_password}@vectorsearch.abc.mongodb.net/?retryWrites=true'


class QueryRag:
    # This class accepts an arbitrary user input, a question, and searches MongoDB 
    # based on vector similarity for related documents. It then passes the user's 
    # input as the value `question` to OpenAI along with the Vector search results, 
    # which are passed as the value `context`.

    def __init__(self) -> None:
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.model = ChatOpenAI(openai_api_key=self.openai_api_key, model="gpt-3.5-turbo")
        self.parser = StrOutputParser()
        # TODO: context is the result of vector search
        # 
        self.template = """
                        Answer the question based on the context below. If you can't 
                        answer the question, reply "I don't know".

                        Context: {context}

                        Question: {question}
                        """

        self.prompt = ChatPromptTemplate.from_template(self.template)


    def populate_and_submit_prompt(self, question):
        # TODO: search Mongo for text chunks related to question return as `context`
        self.prompt.format(context="Mary's sister is Susana", question="Who is Mary's sister?")
        result = self.prompt | self.model | self.parser
