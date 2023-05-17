import os
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from pypdf import PdfReader
import mimetypes
import validators
import requests
import tempfile
import gradio as gr


def get_empty_state():
    return {"knowledge_base": None}


def on_token_change(user_token):
    os.environ["OPENAI_API_KEY"] = user_token


def create_knowledge_base(docs):
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_documents(chunks, embeddings)
    return knowledge_base


def upload_file(file_obj):
    # pdf_reader = PdfReader(file_obj.name)
    # text = ""
    # for page in pdf_reader.pages:
    #     text += page.extract_text()
    loader = UnstructuredFileLoader(file_obj.name, strategy="fast")
    docs = loader.load()

    knowledge_base = create_knowledge_base(docs)
    return file_obj.name, {"knowledge_base": knowledge_base}


def upload_via_url(url):
    if validators.url(url):
        r = requests.get(url)

        if r.status_code != 200:
            raise ValueError(
                "Check the url of your file; returned status code %s" % r.status_code
            )

        content_type = r.headers.get("content-type")
        file_extension = mimetypes.guess_extension(content_type)
        temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
        temp_file.write(r.content)
        file_path = temp_file.name
        loader = UnstructuredFileLoader(file_path, strategy="fast")
        docs = loader.load()
        with open(file_path, mode="rb") as f:
            pass
        knowledge_base = create_knowledge_base(docs)
        return file_path, {"knowledge_base": knowledge_base}
    else:
        raise ValueError("Please enter a valid URL")


def answer_question(question, state):
    knowledge_base = state["knowledge_base"]
    if knowledge_base:
        docs = knowledge_base.similarity_search(question)

        llm = OpenAI(temperature=0.4)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=question)
        return response
    else:
        return "Please upload a file first"


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
        gr.Markdown("**Upload your file**")
        with gr.Row(elem_id="row-flex"):
            with gr.Column(scale=3):
                file_url = gr.Textbox(
                    value="",
                    label="Upload your file",
                    placeholder="Enter a url",
                    show_label=False,
                )
            with gr.Column(scale=1, min_width=160):
                upload_button = gr.UploadButton(
                    "Browse File", file_types=[".txt", ".pdf", ".doc", ".docx"]
                )
        file_output = gr.File()
        user_question = gr.Textbox(value="", label="Ask a question about your file:")
        answer = gr.Textbox(value="", label="Answer:")
        gr.Examples(
            ["What is the main topic of the file?", "Who is the author of the file?"],
            user_question,
        )

    file_url.submit(upload_via_url, file_url, [file_output, state])
    upload_button.upload(upload_file, upload_button, [file_output, state])
    user_token.change(on_token_change, inputs=[user_token], outputs=[])
    user_question.submit(answer_question, [user_question, state], [answer])

demo.queue().launch()
