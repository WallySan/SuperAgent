# SuperAgent

# ⚖️ Processador de Notas Fiscais Eletrônicas (NF-e) e Análise Regulatória com IA

## 🚀 Visão Geral do Projeto

Este projeto implementa uma solução de **RegTech** (Regulatory Technology) que automatiza a análise fiscal de Notas Fiscais Eletrônicas (XML) utilizando o poder do **Google Gemini** e técnicas de **RAG (Retrieval-Augmented Generation)** com o **FAISS**.

O sistema executa um fluxo completo:
1.  Extrai o contexto fiscal da NF-e (XML bruto) via IA.
2.  Busca leis relevantes em um vasto corpus jurídico usando Similaridade Vetorial (FAISS).
3.  Cria um relatório final detalhado com *insights* de economia e a legislação aplicável, gerando um PDF profissional.

### 💡 Fluxo de Trabalho (Pitch Deck)

| Etapa | Descrição | Tecnologia |
| :--- | :--- | :--- |
| **1. Análise Semântica e Extração** | Transforma dados XML brutos em termos de busca de alta precisão (curto e longo) baseados no contexto da NF-e. | Google Gemini 2.5 Flash |
| **2. Geração Dinâmica do Corpus** | Filtra e prepara o índice FAISS com um subconjunto de leis relevantes para a categoria fiscal identificada. | `Legislacao.py` + FAISS/Sentence Transformers |
| **3. Busca de Similaridade Vetorial** | Localiza trechos de leis relevantes no índice, usando similaridade semântica para encontrar conexões precisas. | FAISS (CPU) |
| **4. Geração de Insights de Valor** | Cruza o XML original com os trechos de lei e gera um relatório com quantificação de oportunidades de economia (em R$). | Google Gemini 2.5 Flash |
| **5. Distribuição e Documentação** | Converte o relatório Markdown em um PDF formatado e pronto para arquivamento/apresentação. | `mistune` + `weasyprint` |

## 🛠 Instalação do Projeto

Siga os passos abaixo para configurar e rodar o projeto em seu ambiente.

### 1. Pré-requisitos

* **Python:** Versão 3.8 ou superior.
* **Chave de API:** Uma chave de API válida do Google AI Studio para usar o Gemini.

### 2. Configuração do Ambiente

Crie e ative um ambiente virtual (recomendado):

```bash
# Cria o ambiente virtual
python -m venv venv

# Ativa o ambiente (Linux/macOS)
source venv/bin/activate

# Ativa o ambiente (Windows)
.\venv\Scripts\activate


# 1. Instala o PyTorch ESPECIFICAMENTE para CPU (necessário para embeddings)
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# 2. Instala as bibliotecas de NLP, FAISS e as bibliotecas do projeto (LLM, PDF, etc.)
pip install -r requirements.txt
