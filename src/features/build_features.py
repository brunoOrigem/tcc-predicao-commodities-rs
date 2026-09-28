import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'


def gerar_dataset_final_arroz():
    arq_clima = PROCESSED_DIR / 'dados_climaticos_arroz_limpos.csv'
    arq_preco = PROCESSED_DIR / 'preco_arroz_limpo.csv'
    arq_dolar = PROCESSED_DIR / 'dolar_limpo.csv'
    arq_ipca = PROCESSED_DIR / 'ipca_limpo.csv'

    if not arq_clima.exists() or not arq_preco.exists() or not arq_dolar.exists() or not arq_ipca.exists():
        raise FileNotFoundError(
            'Verifique se todos os arquivos processados (clima, preço, dólar e ipca) '
            'estão presentes em data/processed/'
        )

    print('=== 1. Carregando Datasets Processados ===')
    df_clima = pd.read_csv(arq_clima)
    df_preco = pd.read_csv(arq_preco)
    df_dolar = pd.read_csv(arq_dolar)
    df_ipca = pd.read_csv(arq_ipca)

    df_clima['data'] = pd.to_datetime(df_clima['data'])
    df_preco['data'] = pd.to_datetime(df_preco['data'])
    df_dolar['data'] = pd.to_datetime(df_dolar['data'])
    df_ipca['data'] = pd.to_datetime(df_ipca['data'])

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

    print('=== 3. Unificando Dados (Preço, Clima, Dólar e IPCA - Versão 2.0) ===')
    df_merged = pd.merge(df_preco, df_clima_reg, on='data', how='inner')
    df_merged = pd.merge(df_merged, df_dolar, on='data', how='left')
    df_merged = pd.merge(df_merged, df_ipca, on='data', how='left')
    df_merged = df_merged.sort_values('data').reset_index(drop=True)

    print('=== 4. Criando Features Preditivas (v2.0 Completa) ===')

    # A. Sazonalidade Cíclica
    df_merged['mes'] = df_merged['data'].dt.month
    df_merged['dia_ano'] = df_merged['data'].dt.dayofyear

    df_merged['sin_mes'] = np.sin(2 * np.pi * df_merged['mes'] / 12)
    df_merged['cos_mes'] = np.cos(2 * np.pi * df_merged['mes'] / 12)
    df_merged['sin_dia_ano'] = np.sin(2 * np.pi * df_merged['dia_ano'] / 365.25)
    df_merged['cos_dia_ano'] = np.cos(2 * np.pi * df_merged['dia_ano'] / 365.25)

    # B. Lags e Janelas Móveis do Preço do Arroz
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
    caminho_saida = PROCESSED_DIR / 'dataset_final_arroz.csv'
    df_final.to_csv(caminho_saida, index=False)

    print(f'\n Dataset Final Consolidado salvo em: {caminho_saida}')
    print(f' Dimensões do dataset: {df_final.shape} (linhas, colunas)')
    print(
        f' Período coberto: {df_final["data"].min().strftime("%Y-%m-%d")} a {df_final["data"].max().strftime("%Y-%m-%d")}'
    )
    print(f' Total de NaNs restantes: {df_final.isna().sum().sum()}\n')


if __name__ == '__main__':
    gerar_dataset_final_arroz()