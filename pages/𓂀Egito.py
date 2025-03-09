import streamlit as st
import pandas as pd

# Função para carregar os dados do CSV
# https://share.streamlit.io/

def load_data():
    df = pd.read_csv("enem.csv", sep=';', header=None, names=['Tema', 'Questao', 'ID', 'Origem', 'Pergunta', 'Alternativas', 'Resposta'])
    df.fillna("", inplace=True)  # Substituir NaN por strings vazias
    return df

# Configuração da página do Streamlit
st.title("Quiz de Egito")

# Carregar os dados automaticamente
df = load_data()

# Armazenar o índice da pergunta atual na sessão
if 'index' not in st.session_state:
    st.session_state.index = 0

# Exibir a pergunta atual
question = df.iloc[st.session_state.index]
st.markdown(f"<h3>{question['Pergunta']}</h3>", unsafe_allow_html=True)

# st.markdown(question['Alternativas'])
# Exibir alternativas garantindo que seja string
options = [alt.strip() for alt in str(question['Alternativas']).split('#') if alt.strip()]
if options:
    selected_option = st.radio("Escolha a alternativa correta:", options)
else:
    st.warning("Nenhuma alternativa disponível para esta pergunta.")


# Mostrar a resposta correta
if st.button("Mostrar Resposta"):
    st.success(f"Resposta correta: {question['Resposta']}")

# Navegação entre perguntas
col1, col2 = st.columns(2)
with col1:
    if st.button("Pergunta Anterior"):
        if st.session_state.index > 0:
            st.session_state.index -= 1
with col2:
    if st.button("Próxima Pergunta"):
        if st.session_state.index < len(df) - 1:
            st.session_state.index += 1
