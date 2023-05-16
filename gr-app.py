import os
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import gradio as gr


def get_empty_state():
    return {"knowledge_base": None}


def on_token_change(user_token):
    os.environ["OPENAI_API_KEY"] = user_token


def upload_file(file_obj):
    pdf_reader = PdfReader(file_obj.name)
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
    return file_obj.name, {"knowledge_base": knowledge_base}


def answer_question(question, state):
    knowledge_base = state["knowledge_base"]
    if knowledge_base:
        docs = knowledge_base.similarity_search(question)

        llm = OpenAI(temperature=0.4)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=question)
        return response
    else:
        return "Please upload a PDF file first"


with gr.Blocks(css="style.css") as demo:
    state = gr.State(get_empty_state())
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            # Ask your PDF 💬
            """
        )
        user_token = gr.Textbox(
            value="",
            label="OpenAI API Key",
            placeholder="OpenAI API Key",
            type="password",
            show_label=True,
        )
        file_output = gr.File()
        upload_button = gr.UploadButton("Click to Upload your PDF", file_types=[".pdf"])
        upload_button.upload(upload_file, upload_button, [file_output, state])
        user_question = gr.Textbox(value="", label="Ask a question about your PDF:")
        answer = gr.Textbox(value="", label="Answer:")
        gr.Examples(
            ["What is the main topic of the PDF?", "Who is the author of the PDF?"],
            user_question,
        )

    user_token.change(on_token_change, inputs=[user_token], outputs=[])
    user_question.submit(answer_question, [user_question, state], [answer])

demo.queue().launch()
