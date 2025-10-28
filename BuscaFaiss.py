import sys
import json
import numpy as np
import faiss
from typing import Optional, List, Dict, Any

# =====================================================================
# ETAPA 0: Importação e Configuração FAISS
# =====================================================================
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import torch # Importar apenas se for necessário para checagem de device

    # 🌟 MODELO BERT LEVE 🌟 (O mesmo usado para criar o índice)
    # ATENÇÃO: É VITAL USAR O MESMO MODELO USADO EM Legislacao.py!
    MODELO_EMBEDDING = SentenceTransformer('all-MiniLM-L6-v2') 
    
    # Define o dispositivo de execução (importante para performance)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODELO_EMBEDDING.to(device) 
    
    print(f"✅ Modelo de Embedding carregado no dispositivo: {device}")
except ImportError:
    print("❌ ERRO: Bibliotecas de embedding/FAISS não instaladas. Use: pip install torch sentence-transformers numpy faiss-cpu")
    sys.exit(1)


# =====================================================================
# ETAPA 1: Funções Auxiliares de Carregamento e Busca
# =====================================================================

def carregar_indice_e_metadados(nome_base_arquivo: str) -> Optional[Dict[str, Any]]:
    """
    Carrega o índice FAISS e os metadados JSON salvos.
    O nome_base_arquivo deve ser o termo curto, ex: 'ICMS_ST'.
    """
    # FAISS usa o nome base, e os metadados usam o nome base com sufixo
    nome_indice = f"faiss_index_{nome_base_arquivo}.faiss"
    nome_metadados = f"faiss_index_{nome_base_arquivo}_metadados.json"
    
    try:
        # 1. Carregar Índice FAISS
        index = faiss.read_index(nome_indice)
        
        # 2. Carregar Metadados
        with open(nome_metadados, 'r', encoding='utf-8') as f:
            metadados = json.load(f)
            
        print(f"✅ Índice FAISS e Metadados carregados com sucesso. Base: '{nome_base_arquivo}'.")
        return {'index': index, 'metadados': metadados}
        
    except FileNotFoundError:
        print(f"❌ Erro: Arquivos '{nome_indice}' ou '{nome_metadados}' não encontrados. Verifique se Legislacao.py foi executado com o termo '{nome_base_arquivo}'.")
        return None
    except Exception as e:
        print(f"❌ Erro ao carregar arquivos: {e}")
        return None


def buscar_faiss(query: str, index, metadados: List[Dict[str, Any]], k: int = 5):
    """
    Vetoriza uma consulta (query), busca os K vizinhos mais próximos 
    no índice FAISS e retorna os documentos mais relevantes.
    """
    # Aumentando K para 5, é um bom padrão para RAG
    print(f"\n--- Buscando no FAISS (K={k}) para a query: '{query[:50]}...' ---")
    
    # 1. Gerar o embedding da query (vetor de busca)
    query_embedding = MODELO_EMBEDDING.encode(
        [query], 
        convert_to_numpy=True
    ).astype('float32')
    
    # 2. Busca no índice (D: Distâncias, I: Índices/IDs dos vetores)
    distancias, indices = index.search(query_embedding, k)
    
    resultados_relevantes = []
    
    # 3. Mapear IDs de volta para os Metadados
    for i in range(k):
        id_sequencial = indices[0][i]
        
        if id_sequencial == -1:
            continue

        distancia = distancias[0][i]
        documento_relevante = metadados[id_sequencial]
        
        documento_relevante['rank'] = i + 1
        documento_relevante['distancia_l2'] = float(distancia) 
        
        resultados_relevantes.append(documento_relevante)
        
    print(f"✅ Busca concluída. {len(resultados_relevantes)} documentos encontrados.")
    
    # Formata a saída como uma string única para ser lida pelo script chamador
    saida_formatada = ""
    for res in resultados_relevantes:
        saida_formatada += f"--- DOCUMENTO RANK {res['rank']} ---\n"
        saida_formatada += f"URL/Fonte: {res.get('path', 'N/A')}\n"
        saida_formatada += f"Distância (Similaridade): {res['distancia_l2']:.4f}\n"
        saida_formatada += f"Conteúdo:\n{res['conteudo']}\n\n"
        
    return saida_formatada


# =====================================================================
# ETAPA 2: Execução Principal (Recebendo Argumentos)
# =====================================================================

if __name__ == "__main__":
    
    # Verifica se os argumentos necessários foram fornecidos
    if len(sys.argv) < 3:
        print("Uso: python BuscaFaiss.py <termo_curto_do_faiss> <query_de_busca_completa>")
        print("Exemplo: python BuscaFaiss.py 'ICMS_ST' 'Legislação sobre ICMS-ST de produtos alimentícios'")
        sys.exit(1)

    # Argumentos de Linha de Comando
    TERMO_CURTO_FAISS = sys.argv[1] # Ex: 'ICMS_ST' - O termo usado para criar o índice
    QUERY_DE_BUSCA = sys.argv[2] # Ex: 'Legislação sobre ICMS-ST de produtos alimentícios'

    # 1. Carregar Índice
    dados_carregados = carregar_indice_e_metadados(TERMO_CURTO_FAISS)
    
    if dados_carregados:
        index_faiss = dados_carregados['index']
        metadados_docs = dados_carregados['metadados']

        # 2. Executar a Busca Semântica
        resultados_string = buscar_faiss(
            query=QUERY_DE_BUSCA,
            index=index_faiss,
            metadados=metadados_docs,
            k=5 # Quantidade de documentos mais relevantes para retornar
        )
        
        # 3. Imprimir a String de Resultados para que o script chamador (ProcessaNFe.py) capture
        # NOTA: O script chamador irá capturar TUDO o que for impresso no stdout (print)
        print(resultados_string) 

    else:
        # Se não carregou, imprime uma mensagem vazia ou de erro
        print("Não foi possível carregar o índice. Resultado da busca vazio.", file=sys.stderr)
        sys.exit(1)
