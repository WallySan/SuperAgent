import requests
import sys
import json
import uuid
import numpy as np
from typing import Optional, List, Dict, Any

# =====================================================================
# ETAPA 0: Importação e Configuração FAISS (NOVA)
# =====================================================================
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    
    # 🌟 MODELO BERT LEVE 🌟
    MODELO_EMBEDDING = SentenceTransformer('all-MiniLM-L6-v2') 
    
    print("✅ Modelo de Embedding BERT Leve (all-MiniLM-L6-v2) e FAISS carregados.")
except ImportError:
    print("❌ ERRO: As bibliotecas 'torch', 'sentence-transformers', 'numpy' ou 'faiss-cpu' não estão instaladas.")
    print("       Execute: pip install torch sentence-transformers numpy faiss-cpu")
    sys.exit(1)


# =====================================================================
# ETAPA 1: Funções de Geração e Processamento (Com BERT Leve/MiniLM)
# (Mantidas as funções de processamento de dados)
# =====================================================================

def desserializar_json_resistente(texto_json: str) -> Optional[List[Dict[str, Any]]]:
    """
    Tenta desserializar uma string JSON. Inclui tratamento de erro (resistência).
    """
    try:
        # A resposta do SharePoint tem um formato que precisa ser corrigido antes do json.loads
        # Encontra o primeiro colchete abrindo e o último fechando para isolar o JSON
        start = texto_json.find('[')
        end = texto_json.rfind(']')
        if start != -1 and end != -1:
            texto_json = texto_json[start:end+1]
        
        dados = json.loads(texto_json)
        return dados
    except json.JSONDecodeError as e:
        print(f"❌ Erro de Desserialização JSON: Não foi possível converter a string em JSON.")
        # print(f"Trecho com problema: {texto_json[:200]}...")
        return None
    except Exception as e:
        print(f"❌ Erro inesperado ao desserializar JSON: {e}")
        return None


def extrair_resultados_recursivamente(dados: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Itera recursivamente sobre os dados, procurando por arrays 'ResultRows'
    e extrai os campos 'Path' e 'PublishingPageContentOWSHTML'.
    """
    resultados_extraidos: List[Dict[str, str]] = []

    def buscar_result_rows(item: Any):
        if isinstance(item, dict):
            for chave, valor in item.items():
                if chave == 'ResultRows' and isinstance(valor, list):
                    for row in valor:
                        path = row.get('Path')
                        conteudo = row.get('PublishingPageContentOWSHTML')
                        
                        if path and conteudo:
                            # Concatena Path e PublishingPageContentOWSHTML
                            texto_concatenado = f"PATH: {path}\nCONTEÚDO: {conteudo}"
                            
                            # ID numérica sequencial (index ID) para mapear para o FAISS
                            resultados_extraidos.append({
                                'id': len(resultados_extraidos), # NOVO: ID sequencial
                                'path': path,
                                'conteudo': texto_concatenado
                            })
                    return # Encontrou ResultRows, sai para não aprofundar desnecessariamente
                else:
                    buscar_result_rows(valor)
        
        elif isinstance(item, list):
            for elemento in item:
                buscar_result_rows(elemento)

    buscar_result_rows(dados)
    
    print(f"✅ Extração Concluída. {len(resultados_extraidos)} Resultados base encontrados.")
    return resultados_extraidos


def processar_para_faiss(resultados: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Adiciona vetor (embedding REAL usando MiniLM/BERT) a cada resultado
    e prepara os dados para o FAISS.
    """
    print("\n--- Processando Documentos para FAISS (Gerando Embeddings com MiniLM) ---")
    if not resultados:
        return {'vetores': np.array([]), 'metadados': []}
    
    # 1. Extrai a lista de textos a serem vetorizados e IDs
    textos = [item['conteudo'] for item in resultados]
    
    # 2. Geração dos Embeddings em Batch (Processamento eficiente)
    # A saída é um numpy array
    embeddings_array = MODELO_EMBEDDING.encode(textos, convert_to_numpy=True)
    
    # 3. Prepara os metadados (para mapeamento após a busca FAISS)
    metadados = []
    for item in resultados:
        metadados.append({
            'id': item['id'],
            'path': item['path'],
            'conteudo': item['conteudo'],
        })
        
    print(f"✅ {len(embeddings_array)} Embeddings gerados com sucesso (dimensão {embeddings_array.shape[1]}).")
    return {'vetores': embeddings_array, 'metadados': metadados}


def construir_e_salvar_indice_faiss(dados_processados: Dict[str, Any], nome_base_arquivo: str):
    """
    Constrói e salva o índice FAISS, e salva os metadados separadamente.
    """
    vetores = dados_processados['vetores']
    metadados = dados_processados['metadados']
    
    if vetores.size == 0:
        print("⚠️ Não há vetores para indexar no FAISS.")
        return

    dimensao = vetores.shape[1]
    num_vetores = vetores.shape[0]

    print(f"\n--- Construindo Índice FAISS (Dimensão: {dimensao}, Vetores: {num_vetores}) ---")
    
    # Criação do Índice FAISS Simples (Flat Index - busca exata por similaridade)
    # IndexFlatL2 usa a distância L2 (distância Euclidiana), que é a métrica padrão
    # quando embeddings são normalizados, como os do MiniLM.
    # Para embeddings normalizados, L2 é equivalente a Cosine Similarity.
    index = faiss.IndexFlatL2(dimensao)
    
    # Adiciona os vetores ao índice
    # É necessário garantir que os vetores sejam contíguos (np.ascontiguousarray)
    index.add(np.ascontiguousarray(vetores).astype('float32'))
    
    # --- Salvamento ---
    
    # 1. Salvar o Índice FAISS
    nome_indice = f"{nome_base_arquivo}.faiss"
    try:
        faiss.write_index(index, nome_indice)
        print(f"✅ Índice FAISS (IndexFlatL2) salvo com sucesso: {nome_indice}")
    except Exception as e:
        print(f"❌ Erro ao salvar o índice FAISS: {e}")

    # 2. Salvar os Metadados (ID sequencial + Conteúdo)
    nome_metadados = f"{nome_base_arquivo}_metadados.json"
    try:
        with open(nome_metadados, 'w', encoding='utf-8') as f:
            json.dump(metadados, f, ensure_ascii=False, indent=4)
        print(f"✅ Metadados dos documentos salvos com sucesso: {nome_metadados}")
    except Exception as e:
        print(f"❌ Erro ao salvar os metadados: {e}")
        
    print("\n--- PRÓXIMA ETAPA RAG ---")
    print("Para usar o RAG, você deve carregar o índice FAISS e os metadados, vetorizar a consulta e buscar os K vizinhos mais próximos.")


# =====================================================================
# ETAPA 2: Código Original (Requisição)
# =====================================================================

def fazer_requisicao_fazenda_sp(termo_pesquisa):
    # [A função de requisição HTTP é mantida, exceto pela remoção do digest hardcoded
    # e uma nota sobre o problema]
    url = "https://legislacao.fazenda.sp.gov.br/_vti_bin/client.svc/ProcessQuery"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "text/xml",
        # O cabeçalho 'X-RequestDigest' é dinâmico. O valor hardcoded abaixo
        # quase sempre resultará em um erro 403.
        # Para um código de produção, você precisaria de uma requisição inicial
        # para obter o valor dinâmico.
        # Por simplicidade neste exemplo de FAISS, mantemos o valor de placeholder.
        "X-RequestDigest": "0x00,27 Oct 2025 23:00:32 -0000",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Referer": "https://legislacao.fazenda.sp.gov.br/Paginas/Search.aspx?"
    }

    xml_body = f"""<Request xmlns="http://schemas.microsoft.com/sharepoint/clientquery/2009" SchemaVersion="15.0.0.0" LibraryVersion="16.0.0.0" ApplicationName="Javascript Library"><Actions><ObjectPath Id="95" ObjectPathId="94" /><SetProperty Id="96" ObjectPathId="94" Name="TimeZoneId"><Parameter Type="Number">8</Parameter></SetProperty><SetProperty Id="97" ObjectPathId="94" Name="QueryText"><Parameter Type="String">{termo_pesquisa}</Parameter></SetProperty><SetProperty Id="98" ObjectPathId="94" Name="QueryTemplate"><Parameter Type="String">{{{{searchboxquery}}}} PublishingPageLayoutOWSURLH:"PesqLegisManterAto" OR TipoOWSCHCS:"Leis Complementares Federais" OR TipoOWSCHCS:"Respostas de Consultas"</Parameter></SetProperty><SetProperty Id="99" ObjectPathId="94" Name="Culture"><Parameter Type="Number">1046</Parameter></SetProperty><SetProperty Id="100" ObjectPathId="94" Name="RowsPerPage"><Parameter Type="Number">30</Parameter></SetProperty><SetProperty Id="101" ObjectPathId="94" Name="RowLimit"><Parameter Type="Number">30</Parameter></SetProperty><SetProperty Id="102" ObjectPathId="94" Name="TotalRowsExactMinimum"><Parameter Type="Number">31</Parameter></SetProperty><SetProperty Id="103" ObjectPathId="94" Name="SourceId"><Parameter Type="Guid">{{8413cd39-2156-4e00-b54d-11efd9abdb89}}</Parameter></SetProperty><ObjectPath Id="105" ObjectPathId="104" /><Method Name="SetQueryPropertyValue" Id="106" ObjectPathId="104"><Parameters><Parameter Type="String">SourceName</Parameter><Parameter TypeId="{{b25ba502-71d7-4ae4-a701-4ca2fb1223be}}"><Property Name="BoolVal" Type="Boolean">false</Property><Property Name="IntVal" Type="Number">0</Property><Property Name="QueryPropertyValueTypeIndex" Type="Number">1</Property><Property Name="StrArray" Type="Null" /><Property Name="StrVal" Type="String">Local SharePoint Results</Property></Parameter></Parameters></Method><Method Name="SetQueryPropertyValue" Id="107" ObjectPathId="104"><Parameters><Parameter Type="String">SourceLevel</Parameter><Parameter TypeId="{{b25ba502-71d7-4ae4-a701-4ca2fb1223be}}"><Property Name="BoolVal" Type="Boolean">false</Property><Property Name="IntVal" Type="Number">0</Property><Property Name="QueryPropertyValueTypeIndex" Type="Number">1</Property><Property Name="StrArray" Type="Null" /><Property Name="StrVal" Type="String">Ssa</Property></Parameter></Parameters></Method><SetProperty Id="108" ObjectPathId="94" Name="Refiners"><Parameter Type="String">PesqLegisTipo(deephits=100000,sort=name/descending,filter=15/0/*),PesqLegisRCSubTema(deephits=100000,filter=15/0/*),PesqLegisTributo(deephits=100000,sort=name/descending,filter=15/0/*),PesqLegisDataAto(deephits=100000),PesqLegisRCTema(deephits=100000,filter=15/0/*),PesqLegisRCTributo(deephits=100000,filter=15/0/*)</Parameter></SetProperty><ObjectPath Id="110" ObjectPathId="109" /><Method Name="Add" Id="111" ObjectPathId="109"><Parameters><Parameter Type="String">Title</Parameter></Parameters></Method><Method Name="Add" Id="112" ObjectPathId="109"><Parameters><Parameter Type="String">Path</Parameter></Parameters></Method><Method Name="Add" Id="113" ObjectPathId="109"><Parameters><Parameter Type="String">Author</Parameter></Parameters></Method><Method Name="Add" Id="114" ObjectPathId="109"><Parameters><Parameter Type="String">SectionNames</Parameter></Parameters></Method><Method Name="Add" Id="115" ObjectPathId="109"><Parameters><Parameter Type="String">SiteDescription</Parameter></Parameters></Method><SetProperty Id="116" ObjectPathId="94" Name="RankingModelId"><Parameter Type="String">8f6fd0bc-06f9-43cf-bbab-08c377e083f4</Parameter></SetProperty><SetProperty Id="117" ObjectPathId="94" Name="TrimDuplicates"><Parameter Type="Boolean">false</Parameter></SetProperty><Method Name="SetQueryPropertyValue" Id="118" ObjectPathId="104"><Parameters><Parameter Type="String">ListId</Parameter><Parameter TypeId="{{b25ba502-71d7-4ae4-a701-4ca2fb1223be}}"><Property Name="BoolVal" Type="Boolean">false</Property><Property Name="IntVal" Type="Number">0</Property><Property Name="QueryPropertyValueTypeIndex" Type="Number">1</Property><Property Name="StrArray" Type="Null" /><Property Name="StrVal" Type="String">f286d0d1-5624-47da-a856-a8571296eb7f</Property></Parameter></Parameters></Method><Method Name="SetQueryPropertyValue" Id="119" ObjectPathId="104"><Parameters><Parameter Type="String">ListItemId</Parameter><Parameter TypeId="{{b25ba502-71d7-4ae4-a701-4ca2fb1223be}}"><Property Name="BoolVal" Type="Boolean">false</Property><Property Name="IntVal" Type="Number">1245670</Property><Property Name="QueryPropertyValueTypeIndex" Type="Number">2</Property><Property Name="StrArray" Type="Null" /><Property Name="StrVal" Type="Null" /></Parameter></Parameters></Method><Method Name="SetQueryPropertyValue" Id="120" ObjectPathId="104"><Parameters><Parameter Type="String">CrossGeoQuery</Parameter><Parameter TypeId="{{b25ba502-71d7-4ae4-a701-4ca2fb1223be}}"><Property Name="BoolVal" Type="Boolean">false</Property><Property Name="IntVal" Type="Number">0</Property><Property Name="QueryPropertyValueTypeIndex" Type="Number">1</Property><Property Name="StrArray" Type="Null" /><Property Name="StrVal" Type="String">false</Property></Parameter></Parameters></Method><SetProperty Id="121" ObjectPathId="94" Name="ResultsUrl"><Parameter Type="String">https://legislacao.fazenda.sp.gov.br/Paginas/Search.aspx?#k={termo_pesquisa}</Parameter></SetProperty><SetProperty Id="122" ObjectPathId="94" Name="ClientType"><Parameter Type="String">UI</Parameter></SetProperty><SetProperty Id="123" ObjectPathId="94" Name="ProcessBestBets"><Parameter Type="Boolean">false</Parameter></SetProperty><Method Name="SetQueryPropertyValue" Id="124" ObjectPathId="104"><Parameters><Parameter Type="String">QuerySession</Parameter><Parameter TypeId="{{b25ba502-71d7-4ae4-a701-4ca2fb1223be}}"><Property Name="BoolVal" Type="Boolean">false</Property><Property Name="IntVal" Type="Number">0</Property><Property Name="QueryPropertyValueTypeIndex" Type="Number">1</Property><Property Name="StrArray" Type="Null" /><Property Name="StrVal" Type="String">770d0626-c5e9-4468-806f-763ae6dca132</Property></Parameter></Parameters></Method><SetProperty Id="125" ObjectPathId="94" Name="ProcessPersonalFavorites"><Parameter Type="Boolean">false</Parameter></SetProperty><SetProperty Id="126" ObjectPathId="94" Name="SafeQueryPropertiesTemplateUrl"><Parameter Type="String">querygroup://webroot/Paginas/Search.aspx?groupname=Default</Parameter></SetProperty><SetProperty Id="127" ObjectPathId="94" Name="IgnoreSafeQueryPropertiesTemplateUrl"><Parameter Type="Boolean">false</Parameter></SetProperty><Method Name="SetQueryPropertyValue" Id="128" ObjectPathId="104"><Parameters><Parameter Type="String">QueryDateTimeCulture</Parameter><Parameter TypeId="{{b25ba502-71d7-4ae4-a701-4ca2fb1223be}}"><Property Name="BoolVal" Type="Boolean">false</Property><Property Name="IntVal" Type="Number">1046</Property><Property Name="QueryPropertyValueTypeIndex" Type="Number">2</Property><Property Name="StrArray" Type="Null" /><Property Name="StrVal" Type="Null" /></Parameter></Parameters></Method><ObjectPath Id="130" ObjectPathId="129" /><ExceptionHandlingScope Id="131"><TryScope Id="133"><Method Name="ExecuteQueries" Id="135" ObjectPathId="129"><Parameters><Parameter Type="Array"><Object Type="String">dae87cb5-3265-470f-a15e-f0162a26a113Default</Object></Parameter><Parameter Type="Array"><Object ObjectPathId="94" /></Parameter><Parameter Type="Boolean">true</Parameter></Parameters></Method></TryScope><CatchScope Id="137" /></ExceptionHandlingScope></Actions><ObjectPaths><Constructor Id="94" TypeId="{{80173281-fffd-47b6-9a49-312e06ff8428}}" /><Property Id="104" ParentId="94" Name="Properties" /><Property Id="109" ParentId="94" Name="HitHighlightedProperties" /><Constructor Id="129" TypeId="{{8d2ac302-db2f-46fe-9015-872b35f15098}}" /></ObjectPaths></Request>"""

    print(f"Fazendo requisição para o termo: '{termo_pesquisa}'...")

    try:
        response = requests.post(url, headers=headers, data=xml_body)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as err:
        print(f"❌ Erro HTTP: {err}")
        print(f"Status Code: {response.status_code}")
        print("Atenção: O erro 403 (Forbidden) é comum devido ao cabeçalho 'X-RequestDigest' expirar.")
        return None
    except requests.exceptions.RequestException as err:
        print(f"❌ Ocorreu um erro na requisição: {err}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python pesquisa_fazenda_sp.py <termo_de_pesquisa>")
        print("Exemplo: python pesquisa_fazenda_sp.py borracha")
        sys.exit(1)
    
    termo = sys.argv[1]
    
    # 1. Faz a Requisição HTTP
    resposta = fazer_requisicao_fazenda_sp(termo)

    if resposta:
        print("\n--- Resposta da Requisição ---")
        print(f"Status: {resposta.status_code}")

        # 2. Desserialização JSON Resistente
        dados_json = desserializar_json_resistente(resposta.text)

        if dados_json:
            # 3. Extração Recursiva dos Resultados (ID, Path + Conteúdo)
            resultados_base = extrair_resultados_recursivamente(dados_json)

            if resultados_base:
                # 4. Processamento: Adiciona Vetor (Embedding Real/MiniLM) e prepara para FAISS
                dados_faiss = processar_para_faiss(resultados_base)
                
                # 5. Construção e Salvamento do Índice FAISS e Metadados
                construir_e_salvar_indice_faiss(dados_faiss, f"faiss_index_{termo}")
            else:
                print("⚠️ Nenhuma linha de resultado ('ResultRows') foi encontrada ou os campos estavam incompletos.")
