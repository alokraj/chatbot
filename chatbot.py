import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
# LangChain is an open-source framework that allows developers to build applications that use large
# language models (LLMs). LLMs are deep-learning models that can generate responses to user queries.


OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

#Upload PDF files
st.header("My first Chatbot")

with  st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # st.write(":::::::::::::::::::: TEXT:::::::::::::::::::::::::::::")
    # st.write(text)
    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(":::::::::::::::::::: CHUNKS :::::::::::::::::::::::::::::")
    # st.write(chunks)


    # generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # creating vector store - FAISS
    # FAISS (Facebook AI Similarity Search) is a library
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user question
    user_question = st.text_input("Type Your question here")

    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question, k=2)
        # FAISS does not understand whether the content is relevant to the question;
        # it simply retrieves the closest matches based on vector similarity,
        # even if the matches are not semantically meaningful. If the uploaded document has no content
        # related to the query, the "closest" match might still be quite different from the query.
        #st.write("::::::::::::::::::::::::::::::::: MATCH :::::::::::::::::::::::")
        #st.write(match)
        #st.write("Top matches:")
        #for doc in match:
            #st.write(':::::::::::::',doc.page_content)

        #define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 1,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )

        #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        with st.spinner("Processing your query..."):
            response = chain.run(input_documents=match, question=user_question)
        # Display response
        st.markdown(f"### Chatbot Response:\n{response}")
        # When the LLM (e.g., GPT-3.5-turbo) is provided with some context, even if it’s irrelevant,
        # it will try to answer the question using that context. If the context is insufficient or
        # irrelevant, the LLM may "hallucinate" (generate plausible-sounding but incorrect responses).
        #st.write(response)
