
import os
import re
import streamlit as st

from .sidebar_dev import (
    sidebar,
    # build_query_engine,
    # get_milvus_collections_list,
    build_query_engine_from_db,
    DATA_DIR
)
from .ui_dev import clear_query_history
from .summary_utils import read_summary

def build_multi_doc_query_engine():
    if "selected_docs" not in st.session_state:
        st.session_state["selected_docs"] = []
    query_engines = []
    for doc in st.session_state["selected_docs"]:
            query_engine = build_query_engine_from_db(doc)
            query_engines.append(query_engine)

    def query_multi_docs(prompt):
        responses = []
        sources = []
        for engine in query_engines:
            resp = engine.query(prompt)
            responses.append(resp.response)
            sources.extend(resp.source_nodes)
        combined_response = "\n".join(responses)
        return combined_response, sources   

    st.session_state["query_engine"] = query_multi_docs

def qa_demo_all():
    logo_path = '/home/project/data/zpl/multimodal_RAG/src/chatbot_web_demo/374920_tech-logo-png.png'

    st.header("国泰智能问答")
    st.header("GuoTai AI Q&A:book:")

    if "is_ready" not in st.session_state.keys():
        st.session_state['is_ready'] = False

    
    sidebar()
    
   

    if st.session_state['is_ready']:
        build_multi_doc_query_engine()
        selected_docs = st.session_state["selected_docs"]
        summaries = []

        for doc in selected_docs:
            current_doc_id = re.search(r'\d+', doc).group()
            current_doc = f"{current_doc_id}.pdf"
            current_doc_path = os.path.join(DATA_DIR, current_doc_id)
            summary = read_summary(current_doc_path)
            summaries.append((current_doc, summary))

        for current_doc, summary in summaries:
            st.write("当前文档：", current_doc)
            st.write(summary)
            st.markdown("---")

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "有什么能够帮到您？"}]

        for message in st.session_state.messages:
            avatar = logo_path if message["role"] == "assistant" else '🧑‍💻'
            with st.chat_message(message["role"], avatar=avatar):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar='🧑‍💻'):
                st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant", avatar=logo_path):
                with st.spinner("Thinking ... "):
                    response, sources = st.session_state['query_engine'](prompt)
                    

                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

                st.markdown("-------------------")
                for idx in range(len(sources)):
                    st.write(f"源文档 **{idx+1}**:")
                    st.write(f"{sources[idx].text}")
                    page_number = sources[idx].metadata.get('page_number', "1")
                    st.write(f"源页码: **{page_number}**")
                    st.markdown("-------------------")
    else:
        clear_query_history()
'''


import os
import re
import streamlit as st

from .sidebar_dev import (
    sidebar,
    build_query_engine,
    get_milvus_collections_list,
    DATA_DIR
)
from .ui_dev import clear_query_history
from .summary_utils import read_summary

def qa_demo():
    logo_path = '/home/gt/Chatbot_Web_Demo/assets/logo.jpg'

    st.header("国泰智能问答")
    st.header("GuoTai AI Q&A:book:")

    if "is_ready" not in st.session_state.keys():
        st.session_state['is_ready'] = False

    sidebar()

    build_query_engine()

    if st.session_state['is_ready']:
        selected_docs = st.session_state["selected_docs"]
        summaries = []

        for doc in selected_docs:
            current_doc_id = re.search(r'\d+', doc).group()
            current_doc = f"{current_doc_id}.pdf"
            current_doc_path = os.path.join(DATA_DIR, current_doc_id)
            summary = read_summary(current_doc_path)
            summaries.append((current_doc, summary))

        for current_doc, summary in summaries:
            st.write("当前文档：", current_doc)
            st.write(summary)
            st.markdown("---")

        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "有什么能够帮到您？"}]

        for message in st.session_state.messages:
            avatar = logo_path if message["role"] == "assistant" else '🧑‍💻'
            with st.chat_message(message["role"], avatar=avatar):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar='🧑‍💻'):
                st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant", avatar=logo_path):
                with st.spinner("Thinking ... "):
                    resp = st.session_state['query_engine'].query(prompt)
                    response, sources = resp.response, resp.source_nodes

                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

                st.markdown("-------------------")
                for idx in range(len(sources)):
                    st.write(f"源文档 **{idx+1}**:")
                    st.write(f"{sources[idx].text}")
                    page_number = sources[idx].metadata.get('page_number', "1")
                    st.write(f"源页码: **{page_number}**")
                    st.markdown("-------------------")
    else:
        clear_query_history()
'''