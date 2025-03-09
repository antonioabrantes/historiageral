import streamlit as st


# Adicionando título original 
# https://emojipedia.org/search?q=spy
# https://emojidb.org/pharao-emojis
st.title('História Geral 🚀')

# Adicionando descrição do projeto
st.write("Perguntas sobre história geral.")

# Adicionando as diversas ferramentas
etapas = [
    {
        "nome": "Egito",
        "icone": "𓂀",
        "descricao": "Antigo Egito.",
        "pagina": "[Egito](https://historiageral.streamlit.app/Egito)"
    },
    {
        "nome": "Mesopotamia",
        "icone": "🛕",
        "descricao": "Mesopotamia, Suméria",
        "pagina": "[Mesopotamia](https://historiageral.streamlit.app/Mesopotamia)"
    },
    {
        "nome": "GrecoRomano",
        "icone": "🏺",
        "descricao": "Grécia e Roma antigas.",
        "pagina": "[GrecoRomano](https://historiageral.streamlit.app/GrecoRomano)"
    }
]

# Adicionando as etapas como seções
for etapa in etapas:
    st.header(f"{etapa['icone']} {etapa['nome']}")
    st.write(etapa['descricao'])
    st.markdown(f"**Página da solução:** {etapa['pagina']}")
