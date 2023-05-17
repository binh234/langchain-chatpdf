import os
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from pypdf import PdfReader
import streamlit as st
# from dotenv import load_dotenv


def main():
    # load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    # st.caption("Use OpenAI to parse PDF")

    # api key
    api_key = st.text_input("Enter your OpenAI API key:", type="password")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

        # upload file
        pdf = st.file_uploader("Upload your PDF", type="pdf")

        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Create embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            user_question = st.text_input("Ask a question about your PDF:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI(temperature=0.4)
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)
                st.write(response)


if __name__ == '__main__':
    main()
