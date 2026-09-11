import os
from pathlib import Path
import pandas as pd

# Definição dinâmica de caminhos a partir da raiz do projeto (TCC/)
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CLIMA_DIR = BASE_DIR / 'data' / 'raw' / 'clima'
INTERIM_DIR = BASE_DIR / 'data' / 'interim'


def padronizar_colunas(df):
    """Mapeia dinamicamente as colunas do INMET para qualquer ano (2008-2026)."""
    col_clean = {}
    for col in df.columns:
        c = (
            str(col)
            .strip()
            .upper()
            .replace('Ç', 'C')
            .replace('Ã', 'A')
            .replace('Á', 'A')
            .replace('É', 'E')
            .replace('Í', 'I')
            .replace('Ó', 'O')
            .replace('Ú', 'U')
            .replace('Ê', 'E')
        )

        if 'DATA' in c:
            col_clean[col] = 'data'
        elif 'PRECIPITACAO' in c or 'CHUVA' in c:
            col_clean[col] = 'precipitacao_mm'
        elif 'BULBO SECO' in c or ('TEMPERATURA' in c and 'AR' in c):
            if 'MAX' not in c and 'MIN' not in c:
                col_clean[col] = 'temp_ar_c'
        elif 'TEMPERATURA MAX' in c or 'TEMP. MAX' in c or 'TEMP_MAX' in c:
            col_clean[col] = 'temp_max_c'
        elif 'TEMPERATURA MIN' in c or 'TEMP. MIN' in c or 'TEMP_MIN' in c:
            col_clean[col] = 'temp_min_c'
        elif 'UMIDADE' in c:
            if 'MAX' not in c and 'MIN' not in c:
                col_clean[col] = 'umidade_relativa'

    return df.rename(columns=col_clean)


def processar_e_consolidar_cidade(nome_pasta_cidade):
    # Busca recursivamente todos os CSVs dentro de data/raw/clima/<nome_pasta_cidade>/
    pasta_praça = RAW_CLIMA_DIR / nome_pasta_cidade
    
    if not pasta_praça.exists():
        print(f'Diretório não encontrado: {pasta_praça}')
        return None

    arquivos = list(pasta_praça.rglob('*.csv')) + list(pasta_praça.rglob('*.CSV'))

    if not arquivos:
        print(f'Nenhum arquivo CSV encontrado para: {nome_pasta_cidade}')
        return None

    print(
        f'Processando {nome_pasta_cidade} ({len(arquivos)} arquivos encontrados)...'
    )

    dfs_anuais = []
    for arq in sorted(arquivos):
        try:
            # Lê pulando as 8 primeiras linhas de metadados do INMET
            df = pd.read_csv(
                arq,
                sep=';',
                skiprows=8,
                encoding='latin-1',
                decimal=',',
                on_bad_lines='skip',
            )

            df = padronizar_colunas(df)

            if 'data' in df.columns:
                dfs_anuais.append(df)
            else:
                # Tentativa alternativa caso o cabeçalho esteja na linha 0
                df_alt = pd.read_csv(
                    arq,
                    sep=';',
                    encoding='latin-1',
                    decimal=',',
                    on_bad_lines='skip',
                )
                df_alt = padronizar_colunas(df_alt)
                if 'data' in df_alt.columns:
                    dfs_anuais.append(df_alt)
                else:
                    print(
                        f'   Aviso: Coluna de data não identificada em {arq.name}'
                    )

        except Exception as e:
            print(f'   Erro no arquivo {arq.name}: {e}')

    if not dfs_anuais:
        return None

    df_consolidado = pd.concat(dfs_anuais, ignore_index=True)

    # Aceita múltiplos formatos de data sem descartar anos
    df_consolidado['data'] = pd.to_datetime(
        df_consolidado['data'], format='mixed', dayfirst=True, errors='coerce'
    )
    df_consolidado = df_consolidado.dropna(subset=['data']).sort_values(
        by='data'
    )

    cols_desejadas = [
        'precipitacao_mm',
        'temp_ar_c',
        'temp_max_c',
        'temp_min_c',
        'umidade_relativa',
    ]
    cols_existentes = [
        c for c in cols_desejadas if c in df_consolidado.columns
    ]

    for col in cols_existentes:
        df_consolidado[col] = pd.to_numeric(
            df_consolidado[col], errors='coerce'
        )
        df_consolidado.loc[df_consolidado[col] <= -9999, col] = None

    agg_rules = {}
    if 'precipitacao_mm' in cols_existentes:
        agg_rules['precipitacao_mm'] = 'sum'
    if 'temp_ar_c' in cols_existentes:
        agg_rules['temp_ar_c'] = 'mean'
    if 'temp_max_c' in cols_existentes:
        agg_rules['temp_max_c'] = 'max'
    if 'temp_min_c' in cols_existentes:
        agg_rules['temp_min_c'] = 'min'
    if 'umidade_relativa' in cols_existentes:
        agg_rules['umidade_relativa'] = 'mean'

    df_diario = df_consolidado.groupby('data').agg(agg_rules).reset_index()
    df_diario['praca'] = nome_pasta_cidade

    data_min = df_diario['data'].min().strftime('%Y-%m-%d')
    data_max = df_diario['data'].max().strftime('%Y-%m-%d')
    print(
        f'   -> {nome_pasta_cidade}: {len(df_diario)} dias consolidados (de {data_min} a {data_max})'
    )

    return df_diario


pracas_soja_trigo = ['CruzAlta', 'PassoFundo', 'SantaRosa']
pracas_arroz = ['Uruguaiana', 'SantaVitoria', 'RioGrande']


def gerar_dataset_regional(lista_pracas, nome_arquivo_saida):
    dfs = []
    for praca in lista_pracas:
        df_p = processar_e_consolidar_cidade(praca)
        if df_p is not None:
            dfs.append(df_p)

    if dfs:
        df_regional = pd.concat(dfs, ignore_index=True)
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        caminho_final = INTERIM_DIR / nome_arquivo_saida
        df_regional.to_csv(caminho_final, index=False)
        print(f'\n Dataset regional salvo em: {caminho_final}\n')


if __name__ == '__main__':
    print('=== Processando Região de Soja e Trigo ===')
    gerar_dataset_regional(
        pracas_soja_trigo, 'dados_climaticos_soja_trigo_diarios.csv'
    )

    print('=== Processando Região do Arroz ===')
    gerar_dataset_regional(pracas_arroz, 'dados_climaticos_arroz_diarios.csv')