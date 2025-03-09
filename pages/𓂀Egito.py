import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time
from PyPDF2 import PdfReader
import pdfplumber
from urllib.request import urlopen
from bs4 import BeautifulSoup

from pathlib import Path
import hashlib

       
# https://share.streamlit.io/
load_dotenv()
# Obtém a chave da API da variável de ambiente
# no streamlit https://share.streamlit.io/ escolha o app / Settings / Secrets e guarde a chave API do Google
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Carregando as instruções do sistema para o Gemini
system_instruction = """
Seu nome é Sophia, uma assistente virtual que ajuda um examinador de patentes a analisar um documento em PDF carregado pelo usuário.
Você deve fornecer o resumo do pedido de patente enviado em formato PDF.
"""

# Inicializando o modelo Gemini (gemini-1.5-pro-latest)
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    system_instruction=system_instruction
)

def text_from_pdf_old(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def text_from_pdf(pdf):
    text = ""
    with pdfplumber.open(pdf) as pdf_reader:
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text
    
# Título da página
st.title('AnalisePatente 📄')
st.write("Envie o pedido de patente.")

# View all key:value pairs in the session state

keys_to_reset = ['patent_text', 'specific_focus', 'step', 'numero_patente', 'resumo_patente', 'titulo_patente', 'relatorio_patente', 'relatorio_patente_portugues', 'titulo_patente_portugues']
def view_session_state():
    s = []
    for k, v in st.session_state.items():
        if k in keys_to_reset:
            if isinstance(v, str):
                l = len(v)
            else:
                l = 0
            if  l > 20:
                s.append(f"{k}: {v[:20]} ...") 
            else:
                s.append(f"{k}: {v}") 
    st.write(s)

# Função para resetar o estado da sessão
def reset_session_state():
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

# Botão para resetar a aplicação
if st.button("Resetar aplicação"):
    reset_session_state()
    st.experimental_rerun()
    
view_session_state()

if 'step' not in st.session_state: 
    st.session_state.step = 0

# Upload do pedido
st.write("Por favor, faça o upload do pedido em formato PDF")
pedido = st.file_uploader("Upload do pedido:", type=['pdf'])

if pedido is not None:
    if 'patent_text' not in st.session_state:
        with st.spinner('Carregando pedido...'):
            st.session_state.patent_text = text_from_pdf(pedido)
        st.success('Pedido carregado com sucesso!')
        st.session_state.step = 1

    if st.session_state.step == 1:
        st.write("Há algum tema específico que você gostaria que eu focasse no resumo?")
        specific_focus = st.text_input("Pontos específicos para focar:", "")
        if specific_focus:
            st.session_state.specific_focus = specific_focus
            st.session_state.step = 2
            st.experimental_rerun()

    if st.session_state.step == 2:
        st.markdown(f"**Tema específico de busca:** {st.session_state.specific_focus}")
        messagem_resumo = (
            f"Olá Sophia, faça o resumo do documento em português {st.session_state.patent_text} "
            f"focando nos seguintes pontos: {st.session_state.specific_focus}. "
        )
        if st.button('Faça resumo do documento'):
            with st.spinner("Processando..."):
                st.session_state.abstract = model.generate_content(messagem_resumo).text
            st.session_state.step = 3
            st.experimental_rerun()

    if st.session_state.step == 3:
        st.markdown(f"**Tema específico de busca:** {st.session_state.specific_focus}")
        st.markdown(f"**Resumo:**\n\n{st.session_state.abstract}")

        st.write("Digite o número do documento de patente:")
        numero_patente = st.text_input("Patente:", "")
        if numero_patente:
            st.session_state.numero_patente = numero_patente
            st.session_state.step = 4
            st.experimental_rerun()

    if st.session_state.step == 4:
        st.markdown(f"**Tema específico de busca:** {st.session_state.specific_focus}")
        st.markdown(f"**Resumo:**\n\n{st.session_state.abstract}")
        st.markdown(f"**Patente:** {st.session_state.numero_patente}")
        
        if st.button('Acesse google patents para buscar esta patente'):
            st.session_state.resumo_patente = ''
            st.session_state.titulo_patente = ''
            html = urlopen('https://patents.google.com/patent/US5000000A/en?oq=US5000000')
            bs = BeautifulSoup(html.read(), 'html.parser')
            st.session_state.titulo_patente = bs.title.get_text() # <title></title>
            nameList = bs.findAll("div", {"class": "abstract"}) # <div class="abstract"></div>
            for name in nameList:
                st.session_state.resumo_patente = name.getText()
            nameList = bs.findAll("section", {"itemprop": "description"}) # <section itemprop="description" itemscope>
            for name in nameList:
                st.session_state.relatorio_patente = name.getText()
            st.session_state.step = 5
            st.experimental_rerun()  

    if st.session_state.step == 5:
        st.markdown(f"**Tema específico de busca:** {st.session_state.specific_focus}")
        st.markdown(f"**Resumo:**\n\n{st.session_state.abstract}")
        st.markdown(f"**Patente:** {st.session_state.numero_patente}")
        st.markdown(f"**Título:**\n\n{st.session_state.titulo_patente}")
        #st.markdown(f"**Resumo:**\n\n{st.session_state.resumo_patente}")
        #st.markdown(f"**relatório:**\n\n{st.session_state.relatorio_patente}")

        messagem = (
            f"Olá Sophia, traduza o título em português {st.session_state.titulo_patente} "
        )
        with st.spinner("Processando..."):
            st.session_state.titulo_patente_portugues = model.generate_content(messagem).text
        st.session_state.step = 6
        st.experimental_rerun()
            
    if st.session_state.step == 6:
        st.markdown(f"**Tema específico de busca:** {st.session_state.specific_focus}")
        st.markdown(f"**Resumo:**\n\n{st.session_state.abstract}")
        st.markdown(f"**Patente:** {st.session_state.numero_patente}")
        st.markdown(f"**Título:**\n\n{st.session_state.titulo_patente}")
        st.markdown(f"**Título:**\n\n{st.session_state.titulo_patente_portugues}")
        #st.markdown(f"**Resumo:**\n\n{st.session_state.resumo_patente}")
        #st.markdown(f"**relatório:**\n\n{st.session_state.relatorio_patente}")

        messagem = (
            f"Olá Sophia, traduza o relatório em português {st.session_state.relatorio_patente} "
        )
        with st.spinner("Processando..."):
            st.session_state.titulo_patente_portugues = model.generate_content(messagem).text
        st.session_state.step = 7
        st.experimental_rerun()

    if st.session_state.step == 7:
        st.markdown(f"**Tema específico de busca:** {st.session_state.specific_focus}")
        st.markdown(f"**Resumo:**\n\n{st.session_state.abstract}")
        st.markdown(f"**Patente:** {st.session_state.numero_patente}")
        st.markdown(f"**Título:**\n\n{st.session_state.titulo_patente}")
        st.markdown(f"**Título:**\n\n{st.session_state.titulo_patente_portugues}")
        #st.markdown(f"**Resumo:**\n\n{st.session_state.resumo_patente}")
        st.markdown(f"**relatório:**\n\n{st.session_state.relatorio_patente_portugues}")

            #initial_message_analysis = (
            #    f"Olá Sophia, aponte as diferenças do pedido com a anterioridade. "
            #    f"Aqui está o pedido: {patent_text} "
            #    f"E aqui está a anterioridade: {prior_art_text}"
            #)
            #if st.button('Faça análise dos documentos'):
            #    with st.spinner("Processando..."):
            #        ai_query_analysis = model.generate_content(initial_message_analysis)
            #        st.markdown(ai_query_analysis.text)
else:
    st.warning('Por favor, faça o upload do pedido antes de continuar.')