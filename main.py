# GUI
import PySimpleGUI as sg
import os
import wikipedia

from langchain_openai import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

wikipedia.set_lang("en")

# AITextReader
class AITextReader:
    def __init__(self, text) -> None:
        '''
            Constructor
            Takes the article text we fetched from Wikipedia as an argument and stores
            it for instance methods to use. Also loads the OpenAI API Key
        '''
        # Store the article text as an instance variable
        # (which take 'self' as their first argument)
        self.article_text = text

        # Read the OpenAI key from the application environment
        # can set it by running 'export OPENAI_API_KEY="<Your Key>" in the terminal
        self.api_key = os.environ.get('OPENAI_API_KEY')

    def create_embeddings(self):
        '''
            Creates an embedding from the stored text
        '''
        # picking OpenAI as the source to generate embeddings
        embedding_client = OpenAIEmbeddings(openai_api_key=self.api_key)
        embeddings = Chroma.from_texts([self.article_text], embedding_client)
        return embeddings

    def get_retreiver(self):
        '''
            Get a retreiver that uses created embeddings to answer questions
        '''

        # Get the embeddings for the stored article text
        embeddings = self.create_embeddings()

        # Choose the Large Language Model to answer questions using the embeddings
        # In this case it is OpenAI's GPT-3.5
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Combine the Chatbot with our embeddings to get a "retriever"
        retriever = RetrievalQA.from_chain_type(llm, retriever=embeddings.as_retriever(search_kwargs={"k": 1}))
        return retriever

    def get_answer(self, question):
        ''' Takes a question about the currently stored article text and gets an answer '''

        # Get a "retriever" to ask questions about the article text
        # Under the hood this creates the embeddings, invokes the chatbot and gets a retriever
        retriever = self.get_retreiver()
        answer = retriever({"query": question})
        if answer:
            return answer['result']

# Wikipedia Fetch
class WikipediaFetch:
    def fetch(self, article):
        try:
            return wikipedia.page(article, auto_suggest=False).content
        except Exception as e:
            return wikipedia.page(article, auto_suggest=True).content

# Widgets
header_text = sg.Text('Wikipedia AI', font=('Arial Bold', 20), expand_x=True, justification='center')
article_input = sg.Input(default_text="", key='article_input', expand_x=True)
ok_button = sg.Button("Ok", key='ok_button')

# The question input and button are initially not visible
question_input = sg.Input(default_text="", key='question_input', expand_x=True, visible=False)
question_button = sg.Button("Enter Question", key='question_button', visible=False)

# Initially empty Columns for article and answer texts
article_column = sg.Column([[sg.Text("", size=(100, None), key='article_text')]], scrollable=True, size=(700, 300), expand_x=True, expand_y=True, vertical_scroll_only=True)
answer_column = sg.Column([[sg.Text("", size=(100, None), key='answer_text')]], scrollable=True, size=(700, 100), expand_x=True, expand_y=True, visible=False, vertical_scroll_only=True)

# Chat History
chat_history = {}

# Layout
window_layout = [
    [header_text],
    [article_input, ok_button],
    [article_column],
    [question_input, question_button],
    [answer_column]
]

window = sg.Window('Wikipedia AI Assistant', window_layout, size=(700, 700))
# global article text
article_text = ""

def fetch_article_content():
    user_topic = values['article_input']
    result = WikipediaFetch().fetch(user_topic)
    return result

def fetch_answer_content():
    user_question = values['question_input']
    answer_text = AITextReader(article_text).get_answer(user_question)
    return answer_text

# Event Loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    
    if event == 'ok_button':
        article_text = fetch_article_content()
        window['article_text'].update(article_text)
        question_input.update(visible=True)  # Make question input visible
        question_button.update(visible=True)  # Make question button visible
        answer_column.update(visible=False)   # Hide the answer column until a question is asked

    if event == 'question_button':
        user_question = values['question_input']
        answer_text = fetch_answer_content()
        chat_history[user_question] = answer_text
        window['answer_text'].update(answer_text)
        answer_column.update(visible=True) # Make answer column visible after providing an answer

# Display chat history
formatted_chat_history = "\n".join([f"{question}\n{answer}\n" for question, answer in chat_history.items()])
sg.popup_scrolled(formatted_chat_history, title="Chat History")
window.close()
