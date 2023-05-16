# langchain-chatpdf

A simple langchain PDF app

![demo](images/streamlit-demo.png)

## Development

### Install libraries

```bash
pip install -r requirements.txt
```

#### Install FAISS (Facebook AI Similarity Search)

**GPU**:

```bash
pip install faiss
```

**CPU**:

```bash
pip install faiss-cpu
```

## Run the application

**With streamlit**:

```bash
streamlit run app.py
```

**With gradio**:

```bash
pip install gradio
gradio run gr-app.py
```
