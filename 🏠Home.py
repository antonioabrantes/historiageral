import streamlit as st


# Adicionando título original 
# https://emojipedia.org/search?q=spy
st.title('Patent Tools 🚀')

# Adicionando descrição do projeto
st.write("Ferramentas que auxiliam o exame de patentes com recursos de Inteligência Artificial (Gemini Google).")

# Adicionando as diversas ferramentas
etapas = [
    {
        "nome": "BatePapo",
        "icone": "💬",
        "descricao": "Tire suas dúvidas sobre o novo fluxo de processamento de recurso de pedidos de patente.",
        "pagina": "[BatePapo](https://patenttools.streamlit.app/BatePapo)"
    },
    {
        "nome": "AnalisePatente",
        "icone": "📄",
        "descricao": "Uma ferramenta que faz resumo de documentos e destaca as diferenças com o pedido de epatente que você está examinando.",
        "pagina": "[AnalisePatente](https://patenttools.streamlit.app/AnalisePatente)"
    },
    {
        "nome": "Estatísticas",
        "icone": "📊️",
        "descricao": "Estatísticas diversas em patentes.",
        "pagina": "[Estatisticas](https://patenttools.streamlit.app/Estatisticas)"
    }
]

# Adicionando as etapas como seções
for etapa in etapas:
    st.header(f"{etapa['icone']} {etapa['nome']}")
    st.write(etapa['descricao'])
    st.markdown(f"**Página da solução:** {etapa['pagina']}")
