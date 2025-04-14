
import os

from langchain_openai import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

'''
Remember to run 'export OPENAI_API_KEY="<Your Key>"' if you see:
Did not find openai_api_key, please add an environment variable `OPENAI_API_KEY` which contains it, or pass  `openai_api_key` as a named parameter. (type=value_error)
You can follow this guide to get a key and set it up:
https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key
'''

class AITextReader:
    def __init__(self, text) -> None:
        '''
            Constructor
            Takes the article text fetched from Wikipedia as an argument and stores
            it for instance methods to use. Also loads the OpenAI API Key
        '''
        self.article_text = text
        self.api_key = os.environ.get('OPENAI_API_KEY')

    
    def create_embeddings(self):
        '''
            Creates an embedding from the stored text
        '''

        # picking OpenAI as the source to generate our embeddings
        embedding_client = OpenAIEmbeddings(openai_api_key=self.api_key)
        # to create the embeddings
        embeddings = Chroma.from_texts([self.article_text], embedding_client)

        return embeddings
    

    def get_retreiver(self):
        '''
            Get a retreiver that uses created embeddings to answer questions
        '''

        # Get the embeddings for the stored article text
        embeddings = self.create_embeddings()

        #Large Language Model to answer questions using the embeddings
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Combine Chatbot + embeddings to get a "retriever"
        retriever = RetrievalQA.from_chain_type(llm, retriever=embeddings.as_retriever())

        return retriever
    

    def get_answer(self, question):
    
        #creates the embeddings, invokes the chatbot and gets a retriever (under hood)
        retriever = self.get_retreiver()

        # Send question to the retriever and get answer
        answer = retriever({"query": question})

        if answer:
            return answer['result']


# This is an example of how to use the AITextReader class to answer
# questions about arbitrary text such as a Wikipedia article

        
# Note, may take 10 seconds or so to load an answer
if __name__ == "__main__":
    # Sample text to get an answer from
    sample_text = 'Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32. The standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested. Sections 1.10.32 and 1.10.33 from "de Finibus Bonorum et Malorum" by Cicero are also reproduced in their exact original form, accompanied by English versions from the 1914 translation by H. Rackham.'
    # Read the text with AI, you could replace this with any text
    creator = AITextReader(sample_text)
    # Get an answer to a question about the text
    answer = creator.get_answer("How old is Lorem Ipsum?")
    # Display the answer (Lorem Ipsum is over 2000 years old. It originated from a piece of classical Latin literature written by Cicero in 45 BC.)
    print(answer)
