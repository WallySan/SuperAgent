import sys
import os
import subprocess
import json
import time # Importar a biblioteca 'time'
from google import genai
from google.genai import types

# Bibliotecas Adicionais para Geração de PDF
import mistune # Para converter Markdown para HTML
from weasyprint import HTML, CSS # Para converter HTML para PDF

# ----------------------------------------------------------------------
# 1. Funções de Suporte
# ----------------------------------------------------------------------

def ler_conteudo_xml_bruto(caminho_arquivo):
    """
    Lê o conteúdo de um arquivo XML e o retorna como string bruta.
    """
    print("--- INÍCIO DA ETAPA: Leitura do XML Bruto ---")
    if not os.path.exists(caminho_arquivo):
        print(f"ERRO: Arquivo não encontrado em {caminho_arquivo}")
        return None
        
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo_xml = f.read()
        
        print(f"SUCESSO: XML lido. Tamanho do conteúdo: {len(conteudo_xml)} caracteres.")
        print("--- FIM DA ETAPA: Leitura do XML Bruto ---\n")
        return conteudo_xml
        
    except Exception as e:
        print(f"ERRO ao ler o arquivo XML: {e}")
        print("--- FIM DA ETAPA: Leitura do XML Bruto (FALHA) ---\n")
        return None

def extrair_termos_gemini(conteudo_xml_bruto):
    """
    ETAPA 1: Análise Semântica e Extração de Termos-Chave (Gemini)
    Envia o conteúdo XML bruto ao Gemini para extrair palavras-chave e frases curtas.
    Retorna uma tupla (termo_curto, termo_completo).
    """
    print("--- INÍCIO DA ETAPA 1: Análise Semântica (Gemini) ---")
    print("-> Chamando Gemini 2.5 Flash para extração de termos de busca...")
    try:
        client = genai.Client() 
        
        prompt = f"""
        Analise o conteúdo desta Nota Fiscal Eletrônica (XML) e execute as seguintes tarefas:
        
        1. **Descubra os produtos mais relevantes do campo xProd e crie um termo conciso para pesquisa (uma a três palavras) envolvendo nome do produto, operação para que reduza a quantidade de leis retornadas, deixando uma base mais direcionada e eficaz para encontrar possíveis divergências fiscais**

        2. **Crie uma frase curta, porém mais completa e descritiva, baseada no conteúdo/produtos da NF para uma busca de similaridade mais refinada (ex: 'Legislação sobre ICMS-ST de produtos alimentícios').**
        
        Sua resposta DEVE estar estritamente no formato JSON, sem qualquer outro texto adicional:
        {{
          "termo_curto": "termo conciso",
          "termo_completo": "frase curta e descritiva"
        }}
        
        Conteúdo do XML (Bruto):
        ---
        {conteudo_xml_bruto}... [Conteúdo Omitido]
        ---
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        
        termos = json.loads(response.text)
        
        termo_curto = termos.get('termo_curto', '').strip()
        termo_completo = termos.get('termo_completo', '').strip()

        if not termo_curto or not termo_completo:
            print(f"ERRO: Gemini não retornou os termos no formato esperado. Resposta bruta: {response.text}") 
            raise ValueError("O Gemini não retornou os termos no formato esperado.")
            
        print(f"SUCESSO: Termos extraídos.")
        print(f"   [Termo Curto (FAISS Index)]: {termo_curto}")
        print(f"   [Termo Completo (FAISS Query)]: {termo_completo}")
        
        print("--- FIM DA ETAPA 1: Análise Semântica (Gemini) ---\n")
        return termo_curto, termo_completo
        
    except Exception as e:
        print(f"ERRO na comunicação ou processamento do Gemini: {e}")
        print("--- FIM DA ETAPA 1: Análise Semântica (FALHA) ---\n")
        return None, None

def executar_script_simples(script, argumento):
    """
    ETAPA 2: Geração Dinâmica do Corpus Jurídico (Legislacao.py)
    Executa um script Python via subprocess.
    """
    print(f"--- INÍCIO DA ETAPA 2: Geração Dinâmica do Corpus Jurídico ---")
    comando = [sys.executable, script, argumento]
    print(f"-> Executando {script} para construir/filtrar o índice FAISS com argumento: {argumento}")
    try:
        resultado = subprocess.run(
            comando, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Assume-se que o script de legislação cria o índice
        print(f"SUCESSO: Índice FAISS supostamente pronto para o termo.")
        print(f"   Saída de {script} (Parcial):\n{resultado.stdout.strip()[:200]}...")
        print("--- FIM DA ETAPA 2: Geração Dinâmica do Corpus Jurídico ---\n")
        return resultado.stdout
    except subprocess.CalledProcessError as e:
        print(f"ERRO ao executar {script} (Código de saída: {e.returncode}).")
        print(f"   STDERR:\n{e.stderr.strip()}")
        print("--- FIM DA ETAPA 2: Geração Dinâmica do Corpus Jurídico (FALHA) ---\n")
        return None
    except FileNotFoundError:
        print(f"ERRO: O script {script} não foi encontrado.")
        print("--- FIM DA ETAPA 2: Geração Dinâmica do Corpus Jurídico (FALHA) ---\n")
        return None

def executar_busca_faiss(script, termo_faiss, query_completa):
    """
    ETAPA 3: Busca de Similaridade Vetorial (FAISS)
    Executa o BuscaFaiss.py com os dois argumentos.
    """
    print(f"--- INÍCIO DA ETAPA 3: Busca de Similaridade Vetorial (FAISS) ---")
    comando = [sys.executable, script, termo_faiss, query_completa]
    print(f"-> Executando {script} no índice para buscar: '{query_completa}'")
    try:
        resultado = subprocess.run(
            comando, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        resultados_busca = resultado.stdout
        if "Nenhum resultado relevante encontrado" in resultados_busca:
             print("AVISO: A busca FAISS retornou resultados limitados ou irrelevantes.")

        print(f"SUCESSO: Trechos de lei encontrados e extraídos.")
        print(f"   Saída de {script} (Parcial):\n{resultados_busca.strip()[:200]}...")
        print("--- FIM DA ETAPA 3: Busca de Similaridade Vetorial (FAISS) ---\n")
        return resultados_busca
    except subprocess.CalledProcessError as e:
        print(f"ERRO ao executar {script} (Código de saída: {e.returncode}).")
        print(f"   STDERR:\n{e.stderr.strip()}")
        print("--- FIM DA ETAPA 3: Busca de Similaridade Vetorial (FAISS) (FALHA) ---\n")
        return f"ERRO NA BUSCA FAISS:\n{e.stderr}" 
    except FileNotFoundError:
        print(f"ERRO: O script {script} não foi encontrado.")
        print("--- FIM DA ETAPA 3: Busca de Similaridade Vetorial (FAISS) (FALHA) ---\n")
        return None

def analisar_resultados_gemini(conteudo_xml_bruto, resultados_busca):
    """
    ETAPA 4: Análise e Geração de Insights de Valor (Gemini)
    Envia o XML bruto e os resultados da busca ao Gemini para análise final.
    """
    print("--- INÍCIO DA ETAPA 4: Análise e Geração de Insights (Gemini) ---")
    print("-> Enviando XML e resultados da busca para a análise final do Gemini...")
    try:
        client = genai.Client()

        prompt = f"""
        Abaixo estão o conteúdo BRUTO de uma Nota Fiscal Eletrônica (XML) e um trecho de leis relevantes encontrado (FAISS).
        
        Sua tarefa é gerar uma análise detalhada ESTRITAMENTE no formato MARKDOWN, usando títulos (#), subtítulos (##) e listas.
        
        1. **Resumo da NF-e:** Breve resumo da nota fiscal (o que ela trata).
        2. **Relevância Legal:** Indique se o trecho de lei parece ser relevante ou aplicável à NF-e com link.
        3. **Trecho de Lei Chave:** Cite o trecho da lei que você considerou mais importante ou aplicável.
        4. **Oportunidade de Economia/Benefício:** O mais importante: Cite dicas para aplicação de lei para reduzir recolhimento ou obter algum benefício legal e mensurar o quanto economiza com esta ação. Mostrar se possível em R$.
        
        Gere a análise SOMENTE em formato Markdown, começando com um título de nível 1.
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        analise_markdown = response.text
        
        print("SUCESSO: Análise fiscal completa gerada.")
        print("\n--- INÍCIO DO MARKDOWN GERADO ---")
        print(analise_markdown[:500] + "...") # Log parcial para não poluir o terminal
        print("--- FIM DO MARKDOWN GERADO ---\n")
        
        print("--- FIM DA ETAPA 4: Análise e Geração de Insights (Gemini) ---\n")
        return analise_markdown
        
    except Exception as e:
        print(f"ERRO na análise final do Gemini: {e}")
        print("--- FIM DA ETAPA 4: Análise e Geração de Insights (FALHA) ---\n")
        return "# Erro de Análise\nAnálise final não pôde ser concluída."

def gerar_pdf(conteudo_markdown, nome_arquivo_xml):
    """
    ETAPA 5: Distribuição e Documentação Profissional
    Converte o texto Markdown em HTML e depois gera um PDF formatado.
    """
    print("--- INÍCIO DA ETAPA 5: Geração de Documentação Profissional (PDF) ---")
    nome_pdf = os.path.splitext(nome_arquivo_xml)[0] + "_analise.pdf"
    
    print(f"-> Tentando gerar PDF formatado: {nome_pdf}")

    try:
        # 1. Converter Markdown para HTML
        html_body = mistune.html(conteudo_markdown)
        
        # 2. Estrutura HTML completa com CSS básico para formatação (Omitida por brevidade)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Análise Legal de NF-e</title>
            <style>
                @page {{ size: A4; margin: 2.5cm; }}
                body {{ font-family: 'Arial', sans-serif; line-height: 1.6; color: #333; }}
                h1 {{ color: #004d99; border-bottom: 2px solid #004d99; padding-bottom: 10px; }}
                h2 {{ color: #3366cc; margin-top: 25px; }}
                pre {{ background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; overflow-x: auto; }}
                strong {{ font-weight: bold; }}
                ul, ol {{ margin-left: 20px; }}
            </style>
        </head>
        <body>
            {html_body}
            <footer>
                <p style="text-align: right; font-size: 10pt;">Análise Gerada por Gemini - Documento Confidencial</p>
            </footer>
        </body>
        </html>
        """
        
        # 3. Gerar o PDF usando WeasyPrint
        HTML(string=html_content).write_pdf(nome_pdf)
        
        print(f"SUCESSO: PDF gerado com sucesso em: {nome_pdf}")
        print("--- FIM DA ETAPA 5: Geração de Documentação Profissional (PDF) ---\n")
        return True
    
    except Exception as e:
        print(f"ERRO ao gerar o PDF: {e}")
        print("--- FIM DA ETAPA 5: Geração de Documentação Profissional (FALHA) ---\n")
        return False

# ----------------------------------------------------------------------
# 2. Fluxo Principal
# ----------------------------------------------------------------------

def main():
    print("\n=======================================================")
    print("         INÍCIO DO PROCESSAMENTO DE NOTA FISCAL")
    print("=======================================================")
    if len(sys.argv) < 2:
        print("ERRO: Argumento ausente.")
        print("Uso: python ProcessaNFe.py <nome_do_arquivo_xml>")
        sys.exit(1)

    arquivo_xml = sys.argv[1]
    print(f"Arquivo de entrada: {arquivo_xml}\n")

    # --- 1. Leitura do XML BRUTO ---
    conteudo_xml_bruto = ler_conteudo_xml_bruto(arquivo_xml)
    if not conteudo_xml_bruto:
        print("Fluxo interrompido após falha na leitura do XML.")
        sys.exit(1)

    # --- 2. ETAPA 1: Extrair Termos com Gemini ---
    termo_curto, termo_completo = extrair_termos_gemini(conteudo_xml_bruto)
    if not termo_curto or not termo_completo:
        print("Fluxo interrompido após falha na extração de termos.")
        sys.exit(1)

    # --- DELAY para Rate Limit ---
    DELAY_SEGUNDOS = 3
    print(f"--- GESTÃO DE RATE LIMIT: Aguardando {DELAY_SEGUNDOS} segundos antes da próxima chamada à API (Gemini)... ---")
    time.sleep(DELAY_SEGUNDOS)
    print("--- DELAY CONCLUÍDO. ---\n")
    
    # --- 3. ETAPA 2: Rodar Legislacao.py para gerar o índice FAISS ---
    legislacao_output = executar_script_simples("Legislacao.py", termo_curto)
    if legislacao_output is None:
        print("Fluxo interrompido após falha na Geração do Corpus Jurídico.")
        sys.exit(1)
        
    # --- 4. ETAPA 3: Rodar BuscaFaiss.py para buscar no índice ---
    resultados_busca = executar_busca_faiss("BuscaFaiss.py", termo_curto, termo_completo)
    
    if resultados_busca is None or "ERRO NA BUSCA FAISS" in resultados_busca:
        print("Fluxo interrompido após falha na Busca de Similaridade Vetorial.")
        sys.exit(1)

    # --- 5. ETAPA 4: Análise Final do Gemini (Gera MARKDOWN) ---
    analise_final_markdown = analisar_resultados_gemini(conteudo_xml_bruto, resultados_busca)
    
    # --- 6. ETAPA 5: Geração do PDF ---
    if analise_final_markdown and not "Erro de Análise" in analise_final_markdown:
        gerar_pdf(analise_final_markdown, arquivo_xml)
    else:
        print("AVISO: A análise final falhou ou retornou um erro. O PDF não será gerado.")

    print("\n=======================================================")
    print("            PROCESSAMENTO CONCLUÍDO COM SUCESSO")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
