from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

# Configuração de cada commodity: arquivos de entrada (preço e clima) e saída.
# As variáveis macroeconômicas (dólar, IPCA e Selic) são comuns a todas.
COMMODITIES = {
    'arroz': {
        'arq_preco': 'preco_arroz_limpo.csv',
        'arq_clima': 'dados_climaticos_arroz_limpos.csv',
        'arq_saida': 'dataset_final_arroz.csv',
    },
    'trigo': {
        'arq_preco': 'preco_trigo_limpo.csv',
        'arq_clima': 'dados_climaticos_soja_trigo_limpos.csv',
        'arq_saida': 'dataset_final_trigo.csv',
    },
}

ARQS_MACRO = ['dolar_limpo.csv', 'ipca_limpo.csv', 'selic_limpo.csv']


def _carregar_csv(nome_arquivo):
    df = pd.read_csv(PROCESSED_DIR / nome_arquivo)
    df['data'] = pd.to_datetime(df['data'])
    return df


def gerar_dataset_final(commodity):
    if commodity not in COMMODITIES:
        raise ValueError(
            f'Commodity "{commodity}" não configurada. Opções: {list(COMMODITIES)}'
        )
    cfg = COMMODITIES[commodity]

    arquivos = [cfg['arq_preco'], cfg['arq_clima'], *ARQS_MACRO]
    faltando = [a for a in arquivos if not (PROCESSED_DIR / a).exists()]
    if faltando:
        raise FileNotFoundError(
            f'Arquivos processados ausentes em data/processed/: {faltando}'
        )

    print(f'##### Commodity: {commodity.upper()} #####')
    print('=== 1. Carregando Datasets Processados ===')
    df_preco = _carregar_csv(cfg['arq_preco'])
    df_clima = _carregar_csv(cfg['arq_clima'])
    df_dolar, df_ipca, df_selic = (_carregar_csv(a) for a in ARQS_MACRO)

    print('=== 2. Agregando Clima Regional (Média das 3 Praças) ===')
    df_clima_reg = (
        df_clima.groupby('data')
        .agg(
            {
                'precipitacao_mm': 'mean',
                'temp_ar_c': 'mean',
                'temp_max_c': 'mean',
                'temp_min_c': 'mean',
                'umidade_relativa': 'mean',
            }
        )
        .reset_index()
    )

    print('=== 3. Unificando Dados (Preço, Clima, Dólar, IPCA e Selic - Versão 2.1) ===')
    df_merged = pd.merge(df_preco, df_clima_reg, on='data', how='inner')
    df_merged = pd.merge(df_merged, df_dolar, on='data', how='left')
    df_merged = pd.merge(df_merged, df_ipca, on='data', how='left')
    df_merged = pd.merge(df_merged, df_selic, on='data', how='left')
    df_merged = df_merged.sort_values('data').reset_index(drop=True)

    print('=== 4. Criando Features Preditivas (v2.1 Completa) ===')

    # A. Sazonalidade Cíclica
    df_merged['mes'] = df_merged['data'].dt.month
    df_merged['dia_ano'] = df_merged['data'].dt.dayofyear

    df_merged['sin_mes'] = np.sin(2 * np.pi * df_merged['mes'] / 12)
    df_merged['cos_mes'] = np.cos(2 * np.pi * df_merged['mes'] / 12)
    df_merged['sin_dia_ano'] = np.sin(2 * np.pi * df_merged['dia_ano'] / 365.25)
    df_merged['cos_dia_ano'] = np.cos(2 * np.pi * df_merged['dia_ano'] / 365.25)

    # B. Lags e Janelas Móveis do Preço da Commodity
    for lag in [1, 7, 14, 30]:
        df_merged[f'preco_brl_lag_{lag}'] = df_merged['preco_brl'].shift(lag)

    for window in [7, 14, 30]:
        df_merged[f'preco_brl_ma_{window}'] = (
            df_merged['preco_brl'].rolling(window=window).mean()
        )
        df_merged[f'preco_brl_std_{window}'] = (
            df_merged['preco_brl'].rolling(window=window).std()
        )

    # C. Lags e Janelas Móveis do Dólar
    for lag in [1, 7, 14]:
        df_merged[f'dolar_lag_{lag}'] = df_merged['dolar_venda'].shift(lag)

    for window in [7, 30]:
        df_merged[f'dolar_ma_{window}'] = (
            df_merged['dolar_venda'].rolling(window=window).mean()
        )

    # D. Precipitação Acumulada
    for window in [7, 15, 30]:
        df_merged[f'precipitacao_acc_{window}'] = (
            df_merged['precipitacao_mm'].rolling(window=window).sum()
        )

    # E. Remoção de NaNs (causados pelos lags) e Arredondamento
    df_final = df_merged.dropna().reset_index(drop=True)

    cols_float = df_final.select_dtypes(include=['float64']).columns
    df_final[cols_float] = df_final[cols_float].round(4)

    # Exportação do dataset final
    caminho_saida = PROCESSED_DIR / cfg['arq_saida']
    df_final.to_csv(caminho_saida, index=False)

    print(f'\n Dataset Final Consolidado salvo em: {caminho_saida}')
    print(f' Dimensões do dataset: {df_final.shape} (linhas, colunas)')
    print(
        f' Período coberto: {df_final["data"].min().strftime("%Y-%m-%d")} a {df_final["data"].max().strftime("%Y-%m-%d")}'
    )
    print(f' Total de NaNs restantes: {df_final.isna().sum().sum()}\n')

    return df_final


def gerar_dataset_final_arroz():
    return gerar_dataset_final('arroz')


def gerar_dataset_final_trigo():
    return gerar_dataset_final('trigo')


if __name__ == '__main__':
    for nome in COMMODITIES:
        gerar_dataset_final(nome)
