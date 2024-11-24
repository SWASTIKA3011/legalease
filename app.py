import io
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
import os
import PyPDF2
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
import logging
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key="api_key")


essay = 'case.txt'
with open(essay, 'r') as file:
    essay = file.read()

# Function to perform summarization
def summarize_text(text):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=150)

    docs = text_splitter.create_documents([text])  

    map_prompt = """
    Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """
    Write a concise summary of the following text delimited by triple backquotes.
    Return your response in bullet points which covers the key points of the text.
    ```{text}```
    BULLET POINT SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
    #                                      verbose=True
                                    )


    output = summary_chain.run(docs)

    return output  



def perform_detailed_summarization(text): 

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], 
                                                chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.create_documents([text])
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectors = embeddings.embed_documents([x.page_content for x in docs])

    num_clusters = 8
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)


    closest_indices = []

    for i in range(num_clusters):
    
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

        closest_index = np.argmin(distances)

        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)

    llm3 = ChatOpenAI(temperature=0,
                     openai_api_key="api_key",
                     max_tokens=1000,
                     model='gpt-3.5-turbo')


    map_prompt = """
    You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
    Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
    Your response should be at least three paragraphs and fully encompass what was said in the passage.

    ```{text}```
    FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    map_chain = load_summarize_chain(llm=llm3,
                     chain_type="stuff",
                     prompt=map_prompt_template)

    selected_docs = [docs[doc] for doc in selected_indices]


    summary_list = []

    for i, doc in enumerate(selected_docs):

        chunk_summary = map_chain.run([doc])
    
        summary_list.append(chunk_summary)

        
    summaries = "\n".join(summary_list)
    summaries = Document(page_content=summaries)

    llm4 = ChatOpenAI(temperature=0,
                     openai_api_key="api_key",
                     max_tokens=1000,
                     model='gpt-3.5-turbo',
                     request_timeout=120)

    combine_prompt = """
    You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
    Your goal is to give a verbose summary of what happened in the story.
    The reader should be able to grasp what happened in the book.

    ```{text}```
    VERBOSE SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    reduce_chain = load_summarize_chain(llm=llm4,
                                 chain_type="stuff",
                                 prompt=combine_prompt_template,
#                                verbose=True
                                    )

    output = reduce_chain.run([summaries])
    
    return output



st.set_page_config(layout="wide")

# Section: About
def about_section():
    st.title("LegalEase: Simplifying Legal Documents")
    st.markdown("""LegalEase is your go-to tool for unraveling complex legal jargon and documents🌟✨.
                \nSay goodbye to long reads and hello to quick insights! 🚀""")

    st.header("Main Features:")
    st.markdown("- **Summarization 📑:** Easily generate bulleted or detailed summaries of legal documents.")
    st.markdown("- **Case Analysis 📈:** Get insights into case outcomes, arguments for and against, and more.")
    st.markdown("- **Query Documents ❓:** Ask specific questions about cases and get tailored insights.")

    st.subheader("Purpose:")
    st.write("⭐️ LegalEase aims to bridge the gap between legal documents and understanding for both legal professionals and the general public.")
    st.write("⭐️ Whether you're a lawyer needing quick summaries or a curious individual trying to comprehend legal texts, LegalEase is your solution.")


    st.info(
        """
        Connect with us on [LinkedIn](www.linkedin.com/in/swastika30) !
        """
    )

# Section: Summarizer
def summarizer_section():
    global essay
    st.title('Text Summarizer 📑')

    uploaded_file = st.file_uploader("Upload a Text File", type=['txt', 'pdf'])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension.lower() == "txt":
            essay = uploaded_file.read().decode("utf-8")

        elif file_extension.lower() == "pdf":
            pdf_stream = io.BytesIO(uploaded_file.read()) 
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
            essay = pdf_text
                
            essay = essay.replace('\t', ' ')

        else:
            st.warning("Please upload a text file (.txt) or a PDF file (.pdf)")
         

    st.text_area("Text Content", essay, height=400)

    col1, col2 = st.columns(2)

    if uploaded_file:
        # If user selects bulleted summarization
        if col1.button("Bulleted Summarization"):
            summary = summarize_text(essay)
            st.subheader("Summary")
            st.write(summary)
        
        # If user selects detailed summarization
        if col2.button("Detailed Summarization"):
            summary = perform_detailed_summarization(essay)
            st.write("Detailed Summarization")
            st.text_area(".", summary, height=800)
    


docs = None
# Section: query
def qna_section():
    global docs
    st.title("Generate Arguments/Query the Document 📈")
    uploaded_file = st.file_uploader("Upload a PDF File", type=['pdf'])

    if uploaded_file is not None:
        pdf_stream = io.BytesIO(uploaded_file.read()) 
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        pdf_text = ""

        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        docs = Document(page_content=pdf_text.replace('\t', ' '))

    if docs is not None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
        splits = text_splitter.split_documents([docs])

        if 'vectordb' in globals(): 
            vectordb.delete_collection()

        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        question = st.text_input("Enter your query here..")

        llm = ChatOpenAI(temperature=0)

        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=vectordb.as_retriever(), llm=llm
        )

        unique_docs = retriever_from_llm.get_relevant_documents(query=question)

        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        ans = llm.predict(text=PROMPT.format_prompt(context=unique_docs, question=question).text) 
        if st.button("Show Answer"):
            st.write(ans)



#Section: Predictionn 
def prediction_section():
    st.title("Predict the Outcome and Resolution Ways 📊")
    global docs
    uploaded_file = st.file_uploader("Upload a PDF File", type=['pdf'])

    if uploaded_file is not None:
        pdf_stream = io.BytesIO(uploaded_file.read()) 
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        pdf_text = ""

        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        docs = Document(page_content=pdf_text.replace('\t', ' '))

    if docs is not None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
        splits = text_splitter.split_documents([docs])

        if 'vectordb' in globals(): 
            vectordb.delete_collection()

        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        question = """Please predict the possible outcome and generate arguments both 
                        in favor and against of this case"""

        llm = ChatOpenAI(temperature=0)

        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=vectordb.as_retriever(), llm=llm
        )

        unique_docs = retriever_from_llm.get_relevant_documents(query=question)

        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        ans = llm.predict(text=PROMPT.format_prompt(context=unique_docs, question=question).text) 
        if st.button("Show Answer"):
            st.write(ans)


# Section: Feedback
def feedback_section():
    st.title("Feedback 😀")
    st.write("Your feedback is valuable to us! 🤗")

    feedback = st.text_area("Please share your feedback here:")

    submitted = st.button("Submit Feedback")

    if submitted:
        st.write("Feedback Submitted! 🥳")



page = st.sidebar.radio("Navigation", ["About 🤔", "Summarizer 📑", "Predict 📈", "Query ❓", "Feedback ⭐️"])


if page == "About 🤔":
    about_section()
elif page == "Summarizer 📑":
    summarizer_section()
elif page == "Predict 📈":
    prediction_section()
elif page == "Query ❓":
    qna_section()
elif page == "Feedback ⭐️":
    feedback_section()
