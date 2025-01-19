# Multimodal-RAG-DEMO

## Introduction
This project is a demonstration system for multimodal RAG (Retrieval-Augmented Generation), which mainly includes three functional modules: research report parsing, table content recognition tool, and single-database intelligent Q&A system. The full-database intelligent module is yet to be developed.

## Get Started

Follow these steps to get started with this project:

### 1. Clone the multimodal_RAG project:

```bash
git clone https://github.com/Le1234125/Multimodal-RAG-DEMO.git
```

### 2. Create a virtual environment:
```bash
conda create -n {your_env_name} python=XXX
``` 
(Replace `{your_env_name}` with the name of your virtual environment and `XXX` with the Python version you want to install)

### 3. Install PyTorch:
Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to select the appropriate installation command based on your system configuration

### 4. Install Dependencies

Install all the necessary Python dependencies using the following command:

```bash
pip install -r requirements.txt
```

### 5. Start Milvus

Start Milvus by navigating to the dependencies directory and running Docker Compose:

```bashk˚v
cd dependencies/milvus
docker-compose up -d
```

### 6. Start Nebula Graph

Start Nebula Graph by navigating to the corresponding directory and running the installation script:

```bash
cd dependencies/nebulaGraph
bash install.sh
```

### 7. Configure nltk_data
This section first requires you to visit the nltk_data GitHub URL, clone the files to your local machine, then keep only the packages folder. Rename this folder to nltk_data and configure it in the environment variables. After that, execute the following command to install the nltk package:

```bashk˚v
pip install nltk 
```

### 8. Download Models and Update Paths
Download the relevant models and update the model paths, output paths, and image paths to the correct ones.

### 9. Run the Code
Finally, execute the following two commands to run the system:

```bashk˚v
cd src/chatbot_web_demo
streamlit run streamlit_app.py
```

## Acknowledgements

This work is built with reference to the code of the following projects:

- [Milvus](https://github.com/milvus-io/milvus)
- [Nebula Graph](https://github.com/vesoft-inc/nebula)
- [RAGAs](https://github.com/explodinggradients/ragas)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [MiniCPM](https://github.com/OpenBMB/MiniCPM-V)
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)
- [Ollama](https://github.com/ollama/ollama)
- [BEIR](https://github.com/beir-cellar/beir)

Thanks for their awesome work!

