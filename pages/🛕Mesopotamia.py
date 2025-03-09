import streamlit as st

from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import sys
import time
from jobs_details import jobs_details as data
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# arquivo JSON https://jsoneditoronline.org/

# Configura a API para o modelo genai
# Obtém a chave da API da variável de ambiente
# no streamlit https://share.streamlit.io/ escolha o app / Settings / Secrets e guarde a chave API do Google
api_key = os.getenv("GEMINI_API_KEY")
#st.write(api_key)
genai.configure(api_key=api_key)

# Instrução do sistema para o modelo generativo
system_instruction = f"""

Seu nome é Sophia, um assistente virtual que ajuda o examinador de patentes da fase recursal a fazer seu exame de recurso de pedidos de patentes que foram indeferidos e estão na fase recursal no INPI. 

Procure ser objetivo, responda em poucos parágrafos

Voce deve de início perguntar que tipo de pergunta o examinador deseja fazer: uma pergunta genérica sobre recurso ou uma pergunta sobre como escolher um modelo de parecer. Neste último caso sugerir o modelo de parecer mais provável a ser usado e o código de despacho mais adequado.

Informação sobre as perguntas e respostas mais comuns em formato JSON: {data}

Neste arquivo JSON cada resposta tem associado um modelo de parecer e um código de despacho.

Seu trabalho é entender a pergunta do examinador e indicar a resposta em que aponta em linhas gerais como deve ser o exame de recurso feito pelo examinador. 

No caso de uma pergunta sobre a melhor escolha de um modelo de parecer, se necessário pergunte sobre as razões do indeferimento do pedido e se foi apresentado um novo quadro reivindicatório na petição de recurso e sugira qual o modelo de parecer e código de despacho a ser aplicado no parecer, com base no arquivo JSON. 

No caso de uma pergunta sobre a melhor escolha de um modelo de parecer. O exame deve ser dividido em três etapas: A primeira etapa é a verificação quanto a vícios formais no indeferimento, por exemplo, se alguma petição do processo foi desconsiderada no indeferimento. Caso haja vícios formais que causaram prejuízo ao requerente então o parecer deve ser os Modelos 1 ou 9 com Despacho 100.1 Recurso conhecido e provido para correção do vício formal. Anulado o indeferimento para retorno dos autos à Primeira Instância. 

Caso não tenha sido detectados vícios formais siga para a segunda etapa relativa a detecção de vícios de julgamento. Se foi detectado algum vício de julgamento que leva a reverter a decisão de indeferimento avalie se é o caso de aplicar o princípio de causa madura, ou seja, se todos os critérios de patenteabilidade já foram avaliados no indeferimento. Em caso positivo use o Modeloo 3 e código de despacho 100.3 Recurso conhecido e provido por vício de julgamento. Reformada a decisão recorrida e deferido o pedido. Caso contrário, não é o caso de causa madura, ou seja, há outras questões de patenteabilidade a serem avalisadas e que não foram avalisadas na primeira instância, então aplique Despacho 100.2 Recurso conhecido e provido por vício de julgamento. Anulado indeferimento para retorno dos autos à 1ª Instância para a continuação do exame técnico.

No caso de não haver vícios de julgamento ou mesmo se houve vício de julgamento, mas este não causou prejuízo ao requerente siga para a terceira etapa que é a análise sobre a possibilidade de modificações do pedido. Se o examinador de recurso entende que é possível contornar os óbices do indeferimento, verifique se existe um novo QR apresentado no recurso. Se não houve QR novo ou se houve um QR novo mas que traz elementos do relatório descritivo ou mesmo se este QR novo necessita de emendas então o examinador devefazer uma exigência técnica despacho 121 segundo os Modelos 6 ou Modelo 10. Se o QR resolve os óbices do indeferimento na condição em que foi apresentado sem necessidade de emendas então avalie se é possível aplicar o princípio da causa madura, ou seja, todas as demais questões de patenteabilidade foram observadas no indeferimento. Neste caso aplica-se o modelo 5 e o despacho 100.5 Recurso conhecido e provido devido a modificação no pedido. Anulado indeferimento para retorno dos autos à 1ª Instância para a continuação do exame técnico. Se não for o caso de aplicação de causa madura, ou seja, há questões de patenteabilidade ainda não examinadas então aplica-se o Modelo 4 e despacho 100.4 Recurso conhecido e provido devido a modificação no pedido. Reformada a decisão recorrida e deferido o pedido.

Se não há solução para os óbices apresentados no indeferimento, não cabe recurso negado (111), o examinador, neste caso, necessariamente deve fazer uma exigência técnica (121).

"""


# Título da página
st.title('BatePapo 💬')

# Introdução do assistente virtual
st.write("A Assistente Virtual Sophia está aqui para te ajudar a tirar suas dúvidas sobre o processamento de recursos de paedidos de patente! Atualmente o assistente tem informações mais comuns já cadastradas. Vamos começar?")

#model = genai.GenerativeModel("gemini-pro") # teste
#response = model.generate_content("O que é uma patente ?")
#st.write(response.text)
#sys.exit(0)

# Inicializa o modelo generativo
model = genai.GenerativeModel(
  model_name="gemini-1.5-pro-latest",
  system_instruction=system_instruction
)

# Mensagem inicial do modelo
initial_model_message = "Olá, eu sou Sophia, um assistente virtual que te ajuda a tirar suas dúvidas sobre o processamento de recursos de pedidos de patente. Como você se chama?"

# Inicializa a conversa do assistente virtual
if "chat_encontra" not in st.session_state:
    st.session_state.chat_encontra = model.start_chat(history=[{'role':'model', 'parts': [initial_model_message]}])

# Exibe o histórico de conversa
for i, message in enumerate(st.session_state.chat_encontra.history):
  if message.role == "user":
    with st.chat_message("user"):
      st.markdown(message.parts[0].text)
  else:
    with st.chat_message("assistant"):
      st.markdown(message.parts[0].text)

# Entrada do usuário
user_query = st.chat_input('Você pode falar ou digitar sua resposta aqui:')

# Processamento da entrada do usuário e resposta do assistente
if user_query is not None and user_query != '':
    with st.chat_message("user"):
      st.markdown(user_query)
    with st.chat_message("assistant"):
        ai_query = st.session_state.chat_encontra.send_message(user_query).text
        st.markdown(ai_query)
