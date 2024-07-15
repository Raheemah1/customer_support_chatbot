from PyPDF2 import PdfReader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings



load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key = api_key)



loader = CSVLoader('appclick_FAQS.csv',source_column="prompt", encoding='ISO-8859-1')
data = loader.load()

def get_vector_store(data):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(documents=data, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Your name is Zino, the customer support admin at appclick. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just find a very pleasant way to tell them you don't have access to that info and that you will contact the real Zino as you are just a customer support bot, don't provide the wrong answer and always respond nicely to greetings and salutations\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3, google_api_key=api_key)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=api_key)

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()


    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    return response['output_text']
    

