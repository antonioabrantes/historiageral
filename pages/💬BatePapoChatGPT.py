# Billing: https://platform.openai.com/settings/organization/billing/overview

import streamlit as st

from pathlib import Path
import sys
import time
from jobs_details import jobs_details as data

import utils
from datetime import datetime

from typing import List
import os
from dotenv import load_dotenv

#from langchain_community.document_loaders.pdf import PyPDFLoader
#from langchain_openai import OpenAIEmbeddings

import PyPDF2
import os
import pandas as pd
from elevenlabs import play, save, Voice, VoiceSettings
from elevenlabs.client import ElevenLabs
import pygame
from playsound import playsound
#from pygame import error as pygame_error
from gtts import gTTS

#from chromadb.config import Settings as ChromaSettings
#from chromadb.client import Client as ChromaClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.tracers import LangChainTracer
#from langfuse.callback import CallbackHandler

from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY2")
#openai.api_key = api_key

chave_eleven = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(
  api_key=chave_eleven  # Defaults to ELEVEN_API_KEY
)

openai_api_key = os.getenv("OPENAI_API_KEY2")
#os.environ["OPENAI_API_KEY"]=openai_api_key
gemini_api_key = os.getenv("GEMINI_API_KEY")
#os.environ["GOOGLE_API_KEY"] = gemini_api_key
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
# os.environ["ANTHROPIC_API_KEY"]=anthropic_api_key
# os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")
groq_api_key = os.getenv("GROQ_API_KEY")
#os.environ["GROQ_API_KEY"] = groq_api_key


tracer = LangChainTracer(project_name="My Project")
#langfuse_handler = CallbackHandler()
#langfuse_handler.auth_check()

def init_pygame_mixer():
    try:
        pygame.mixer.init()
        st.write("Pygame mixer inicializado com sucesso!")
    except pygame_error as e:
        st.write(f"Erro ao inicializar o mixer do Pygame: {e}")
        # Aqui você pode optar por usar uma alternativa ou lidar com o erro de outra forma
        use_pydub_alternative()

# Função alternativa usando Pydub
def use_pydub_alternative():
    try:
        from pydub import AudioSegment

        # Exemplo de uso do Pydub para carregar e salvar áudio
        audio = AudioSegment.from_file("path/to/your/audiofile.mp3")
        audio.export("output.wav", format="wav")
        st.write("Áudio processado com sucesso usando Pydub!")
    except Exception as e:
        st.write(f"Erro ao usar Pydub: {e}")

# Inicializar o mixer do Pygame
#init_pygame_mixer()


# Título da página
st.title('BatePapos 💬')

   
# Introdução do assistente virtual
st.write("A Assistente Virtual Sophia está aqui para te ajudar a tirar suas dúvidas sobre o processamento de recursos de paedidos de patente! Atualmente o assistente tem informações mais comuns já cadastradas. Vamos começar?")

# Listar todos os arquivos e diretórios no diretório atual
#arquivos = os.listdir('.')
# Filtrar para mostrar apenas arquivos (não diretórios)
#arquivos = [f for f in arquivos if os.path.isfile(f)]
#st.write("Arquivos no diretório atual:")
#for arquivo in arquivos:
#    st.write(arquivo)
   
pdf_path = "streamlit/chatbot_cgrec.pdf"  # Especifique o caminho do PDF

# Verificar se o arquivo existe no caminho especificado
if os.path.exists(pdf_path):
    loader = PyPDFLoader(pdf_path)
    st.write("Arquivo chatbot_cgrec.pdf carregado com sucesso!")
else:
    st.error(f"O arquivo {pdf_path} não foi encontrado. Verifique o caminho e tente novamente.")
    
# Abrir o arquivo PDF
with open(pdf_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    num_pages = len(reader.pages)
    
    all_text = ""
    
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text = page.extract_text()
        if text:
            all_text += text
    
    # Salvar todo o texto em um arquivo
    #with open("bitcoin.txt", "w", encoding="utf-8") as output_file:
    #    output_file.write(all_text)
    
# st.write(all_text)
text = all_text

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", use_ssl=False)  # este gpt2 é gratuito para calcular número de tokens, não precisa de api_key

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter( # divide o PDF em blocos/chunks de 512 tokens
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

#embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")

db = FAISS.from_documents(chunks, embeddings)

def create_chain(model_type, retriever):
    template="""Questão: {question} Resposta: Vamos pensar passo a passo."""
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    if model_type == "ollama": # https://python.langchain.com/v0.2/docs/integrations/chat/ollama/
        model = Chatollama(model="llama3.1", base_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    elif model_type == "openai": # https://python.langchain.com/v0.2/docs/integrations/chat/openai/
        model = OpenAI(openai_api_key=api_key, temperature=0)
    elif model_type == "openai-gpt-3.5-turbo": # https://python.langchain.com/v0.2/docs/integrations/chat/openai/
        model = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", max_tokens=256, openai_api_key=openai_api_key)
        #model = ChatOpenAI()
    elif model_type == "openai-gpt-4o-mini": # https://python.langchain.com/v0.2/docs/integrations/chat/openai/
        model = ChatOpenAI(temperature=0.0, model="gpt-4o-mini", max_tokens=256, openai_api_key=openai_api_key)
    elif model_type == "anthropic": # https://python.langchain.com/v0.2/docs/integrations/chat/anthropic/
        model = ChatAnthropíc(temperature=0.0, model="claude-3-5-sonnet-20240620", max_tokens=256, timeout=None, max_retries=2)
    elif model_type == "gemini": # https://python.langchain.com/v0.2/docs/integrations/chat/google_generative_ai/
        model = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro", max_tokens=256, timeout=None, max_retries=2)
    elif model_type == "groq": # 
        model = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    else:
        raise ValueError("Unsupported model type: {model_type}")
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | model | output_parser

# Primeira opção de chain: pela seleção de runnables
retriever=db.as_retriever()
chain1 = create_chain("openai",retriever)

template="""Questão: {question} Resposta: Vamos pensar passo a passo."""
prompt = ChatPromptTemplate.from_template(template)
model = OpenAI(openai_api_key=api_key, temperature=0)
chain1 = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Segunda opção de chain: pela chamada do llm simples
def create_model(model_type):
    if model_type == "ollama": # https://python.langchain.com/v0.2/docs/integrations/chat/ollama/
        model = Chatollama (model="llama3.1", base_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    elif model_type == "openai": # https://python.langchain.com/v0.2/docs/integrations/chat/openai/
        model = OpenAI(openai_api_key=api_key, temperature=0)
    elif model_type == "openai-gpt-3.5-turbo": # https://python.langchain.com/v0.2/docs/integrations/chat/openai/
        model = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", max_tokens=256, openai_api_key=openai_api_key)
        #model = ChatOpenAI()
    elif model_type == "openai-gpt-4o-mini": # https://python.langchain.com/v0.2/docs/integrations/chat/openai/
        model = ChatOpenAI(temperature=0.0, model="gpt-4o-mini", max_tokens=256, openai_api_key=openai_api_key)
        #model = ChatOpenAI()
    elif model_type == "anthropic": # https://python.langchain.com/v0.2/docs/integrations/chat/anthropic/
        model = ChatAnthropíc(temperature=0.0, model="claude-3-5-sonnet-20240620", max_tokens=256, timeout=None, max_retries=2)
    elif model_type == "gemini": # https://python.langchain.com/v0.2/docs/integrations/chat/google_generative_ai/
        model = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro", max_tokens=256, timeout=None, max_retries=2)
    elif model_type == "groq": # 
        model = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    else:
        raise ValueError("Unsupported model type: {model_type}")
    return model
    
#llm = OpenAI(openai_api_key=api_key, temperature=0)
llm = create_model("openai")
llm = create_model("openai-gpt-3.5-turbo")
#llm = create_model("openai-gpt-4o-mini")
chain2 = load_qa_chain(llm, chain_type="stuff")

# resposta = chain.run(input_documents=docs, question=query)    
# st.write(query)
# st.write(resposta)

# Terceira oção de chain: pelo retriever sem runnables
from langchain.chains import RetrievalQA 
retriever=db.as_retriever()
chain3 = RetrievalQA.from_chain_type(llm, retriever=retriever)

chain = chain2

# Inicializa a conversa do assistente virtual
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
# Exibe o histórico de conversa
for message in st.session_state.chat_history:
    if message['role'] == "user":
        with st.chat_message("user"):
            st.markdown(message['content'])
    else:
        with st.chat_message("assistant"):
            st.markdown(message['content'])

# Entrada do usuário
user_query = st.chat_input('Você pode falar ou digitar sua resposta aqui:')

# Processamento da entrada do usuário e resposta do assistente
if user_query is not None and user_query != '':
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.chat_history.append({'role': 'user', 'content': user_query})
    # Exibe a mensagem do usuário
    with st.chat_message("user"):
        st.markdown(user_query)
        
    sistema = """Seu nome é Sophia.
    Caso você receba uma mensagem pedindo pelo andamento ou status de um pedido de patente, responda tipo 1.
    Nos outros casos responda tipo 0.

    Formato da resposta deve ser um JSON conforme o modelo:
    { "tipo" : 1 }
    """
    response = utils.send_message(user_query, sistema, True, api_key=api_key )
    st.markdown(response)
    
    if response['tipo'] == 1:
        numero = utils.extrair_numero_pedido(user_query)
        # numero = '102023024151' PI9809089 11201300558 
        # numero= '11201300558'
        # numero= '112013005558'
        # https://www.cientistaspatentes.com.br/apiphp/patents/query/?q="mysql_query":"* FROM arquivados where numero='112013005558' and anulado=0 order by data desc"
        
        if numero is not None:
            query = '"' + "mysql_query" + '"' ":" + '"' + f" * FROM arquivados where numero='{numero}' and anulado=0 order by data desc" + '"'
            url = f"http://www.cientistaspatentes.com.br/apiphp/patents/query/?q={query}"
            data = utils.acessar_sinergias(url)

            try:
                if 'patents' in data and len(data['patents']) > 0 and 'despacho' in data['patents'][0]:
                    #despachos = [patent['despacho'] for patent in data['patents']]
                    #for despacho in despachos:
                    #    print(despacho)
                        
                    despacho = data['patents'][0]['despacho'].strip()
                    formatted_date = utils.convert_date(data['patents'][0]['data'])
                    query = '"' + "mysql_query" + '"' ":" + '"' + f" * FROM despachos WHERE despacho='{despacho}'" + '"'
                    url = f"http://www.cientistaspatentes.com.br/apiphp/patents/query/?q={query}"
                    data = utils.acessar_sinergias(url)
                    descricao = data['patents'][0]['descricao'].strip()
                    resumo = data['patents'][0]['resumo'].strip()
                    if resumo != descricao:
                        resposta = f"Última publicação: {despacho} (publicado em {formatted_date}), {resumo}. {descricao}"
                    else:
                        resposta = f"Última publicação: {despacho} (publicado em {formatted_date}), {resumo}."
                    
            except:
                resposta = f"Erro na consulta {url}"
        else:
            resposta = "Número inválido"

    else:
        # Processa a mensagem do usuário e gera a resposta
        if (chain==chain1):
            resposta = chain.invoke(user_query)
        if (chain==chain2):
            docs = db.similarity_search(user_query)
            # st.write(docs[0].page_content)
            resposta = chain.run(input_documents=docs, question=user_query, callbacks=[tracer])
        if (chain==chain3):
            resposta = chain.run(user_query)
    
        
    # Adiciona a resposta do assistente ao histórico
    st.session_state.chat_history.append({'role': 'assistant', 'content': resposta})
    
    # Exibe a resposta do assistente
    with st.chat_message("assistant"):
        st.markdown(resposta)
        
        try:
            with open('mensagens.txt', 'a') as file:
                now = datetime.now()
                formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
                file.write(formatted_now + "\n")
                file.write(user_query + "\n")
                file.write(resposta + "\n\n")
                st.markdown(formatted_now)
        except Exception as e:
            st.markdown("Ocorreu um erro ao tentar abrir ou escrever no arquivo:", str(e))
            
            
        #voice='TX3LPaxmHKxFdv7VOQHJ'
        #audio = client.generate(
        #    text='testando',
        #    voice=Voice(voice_id=voice,
        #                settings=VoiceSettings(stability=0.35,
        #                                       similarity_boost=0.4,
        #                                       style=0.55,
        #                                       use_speaker_boost=False)),
        #    model='eleven_multilingual_v2'
        #)
        filename = "./resposta.mp3"
        #save(audio=audio,filename=os.path.abspath(filename))
        filename = os.path.abspath(filename)
        #playsound(filename)
        #st.write(filename) # /mount/src/patenttools/resposta.mp3
        
        #if not os.path.exists(filename):
        #    save(audio=audio, filename=os.path.abspath(filename))
        #filename = os.path.abspath(filename)
        #pygame.mixer.music.load(filename)
        #pygame.mixer.music.play()
        #while pygame.mixer.music.get_busy():
        #    pygame.time.Clock().tick(10)
        
        #tts = gTTS(text=resposta, lang='pt', slow=False)
        #temp_audio_file = "./temp_audio.mp3"
        #tts.save(temp_audio_file)
        #pygame.mixer.init()
        #pygame.mixer.music.load(temp_audio_file)
        #pygame.mixer.music.play()
        #while pygame.mixer.music.get_busy():
        #    pygame.time.Clock().tick(10)  # Control the playback speed
        #pygame.mixer.music.stop()
        #pygame.mixer.quit()
        #os.remove(temp_audio_file)

# pages = loader.load_and_split()

# embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# from langchain_community.vectorstores.faiss import FAISS # banco de dados vetorial FAISS
# db = FAISS.from_documents(pages, embeddings)

# q = "Uma exigência não cumprida na primeira instância sofre preclusão na fase recursal ?"
# trecho_relevante = db.similarity_search(q)[0] # ele acessa o texto e localiza trecho relevante

# from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain_openai import OpenAI
# llm = OpenAI(openai_api_key=api_key)
# chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever())
# resposta = chain(q, return_only_outputs=True)

# st.write(resposta.result)
# st.write("=====")
# st.write(trecho_relevante.page_content)
# st.write("=====")
# st.write(trecho_relevante.page)
# st.write("=====")

