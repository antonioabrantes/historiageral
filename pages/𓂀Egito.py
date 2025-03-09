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

options = [alt.strip() for alt in str(question['Alternativas']).split('#') if alt.strip()]
options_color = []
for option in options:
    options_color.append(f"<p style='color:black; font-weight:normal;'>{option}</p>")

# Criar uma tabela com duas colunas
html_content = "<table style='width:100%;'>"

# Adiciona cada opção como uma linha na tabela
for i, option in enumerate(options_color):
    html_content += f"""
    <tr>
        <td><input type="radio" name="alternativa" value="{i}" id="option_{i}"></td>
        <td><label for="option_{i}" style="color:green; font-weight:bold;">{option}</label></td>
    </tr>
    """

html_content += "</table>"

# Exibe a tabela no Streamlit
st.markdown(html_content, unsafe_allow_html=True)

#for i, option in enumerate(options_color):
#    st.markdown(f"""
#        <div style="display: inline-block; margin-right: 10px;">
#            <input type="radio" name="alternativa" value="{i}" id="option_{i}">
#            <label for="option_{i}" style="display: inline-block;">{option}</label>
#        </div>
#    """, unsafe_allow_html=True)
    
#if options_color:
#    selected_option = st.radio("Escolha a alternativa correta:", options_color)
#else:
#    st.warning("Nenhuma alternativa disponível para esta pergunta.")


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
