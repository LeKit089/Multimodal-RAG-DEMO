# multimodal_RAG

## Introduction
> TODO


## Get Started

Follow these steps to get started with this project:

### 1. Install Dependencies

First, install all the necessary Python dependencies using the following command:

```bash
pip install -r requirements.txt
```

### 2. Start Milvus

Next, start Milvus by navigating to the dependencies directory and running Docker Compose:

```bashk˚v
cd dependencies/milvus
docker-compose up -d
```

### 3. Start Nebula Graph

Finally, start Nebula Graph by navigating to the corresponding directory and running the installation script:

```bash
cd dependencies/nebulaGraph
bash install.sh
```

### 4. Run the Code
> TODO


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

