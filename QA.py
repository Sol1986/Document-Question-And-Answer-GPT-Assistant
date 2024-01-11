import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading{file}')
        loader= PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading{file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        Print('Document format not supported')
        return


    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap = 20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings=OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answers(vector_store, q, k=3): #higher k, more elaborate
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5,top_p = 0.8)

    retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs={'k' : k})
    chain  = RetrievalQA.from_chain_type(llm=llm, chain_type = "stuff", retriever = retriever)
                                                                            

    answer = chain.run(q)
    return answer

def calculate_embedding_cost(text):
    import tiktoken
    enc = tiktoken.encoding_for_model('babbage-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in text])
    return total_tokens, total_tokens / 1000 * 0.0005

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override = True)

    
    st.header('LLM Question-Answering Application')
    
    with st.sidebar:
        st.image('img.png', width=300)
        st.subheader('Upload any text or pdf file and ask questions to learn about it')
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY']= api_key

        upload_file = st.file_uploader('Upload a file PDF/Docx/txt): ', type=['pdf', 'docx', 'txt'], on_change = clear_history)
        chunk_size = st.number_input('Chunk size: ', min_value = 100, max_value=2043, value=512, on_change = clear_history)
        k = st.number_input('K Most Similar Chunks - Increase for a more elaborate answer (1-20):', min_value = 1, max_value=20, value=3)
        add_data = st.button('Submit', on_click = clear_history)

        if upload_file and add_data:
            with st.spinner('Reading, chunking, and embedding file...'):
                bytes_data = upload_file.read()
                file_name = os.path.join('./', upload_file.name)
                with open (file_name, 'wb') as f:
                    f.write(bytes_data)
                

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size =chunk_size)
                st.write(f'Chunk size : {chunk_size}')
                st.write(f'Chunks: {len(chunks)}')
               
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding Cost in USD: ${embedding_cost:.5f}')
                

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store

                st.success('File Uploaded, Chunked, And Embedded Succcessfully')

    q = st.text_input('Ask a question about the content of your file: ')
    
    if q:
     with st.spinner('Searching document for answers...'):
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            # st.write(f'k:{k}')
            answer = ask_and_get_answers(vector_store, q, k)
            st.text_area('LLM Answer: ', value=answer ) 
        

            st.divider()

            if 'history' not in st.session_state:
                st.session_state.history =   ''
            value = f'Q: {q} \n\nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label = 'Chat History', value=h, key='history', height=400)













