import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'


def gerar_dataset_final_arroz():
    arq_clima = PROCESSED_DIR / 'dados_climaticos_arroz_limpos.csv'
    arq_preco = PROCESSED_DIR / 'preco_arroz_limpo.csv'

    if not arq_clima.exists() or not arq_preco.exists():
        raise FileNotFoundError(
            'Verifique se os arquivos dados_climaticos_arroz_limpos.csv e preco_arroz_limpo.csv '
            'estão presentes em data/processed/'
        )

    print('=== 1. Carregando Datasets Processados ===')
    df_clima = pd.read_csv(arq_clima)
    df_preco = pd.read_csv(arq_preco)

    df_clima['data'] = pd.to_datetime(df_clima['data'])
    df_preco['data'] = pd.to_datetime(df_preco['data'])

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

    print('=== 3. Realizando Merge entre Preço e Clima ===')
    df_merged = pd.merge(df_preco, df_clima_reg, on='data', how='inner')
    df_merged = df_merged.sort_values('data').reset_index(drop=True)

    print('=== 4. Criando Features Preditivas (v1.0 Baseline) ===')

    # A. Sazonalidade Cíclica
    df_merged['mes'] = df_merged['data'].dt.month
    df_merged['dia_ano'] = df_merged['data'].dt.dayofyear

    df_merged['sin_mes'] = np.sin(2 * np.pi * df_merged['mes'] / 12)
    df_merged['cos_mes'] = np.cos(2 * np.pi * df_merged['mes'] / 12)
    df_merged['sin_dia_ano'] = np.sin(2 * np.pi * df_merged['dia_ano'] / 365.25)
    df_merged['cos_dia_ano'] = np.cos(2 * np.pi * df_merged['dia_ano'] / 365.25)

    # B. Lags e Janelas Móveis do Preço
    for lag in [1, 7, 14, 30]:
        df_merged[f'preco_brl_lag_{lag}'] = df_merged['preco_brl'].shift(lag)

    for window in [7, 14, 30]:
        df_merged[f'preco_brl_ma_{window}'] = (
            df_merged['preco_brl'].rolling(window=window).mean()
        )
        df_merged[f'preco_brl_std_{window}'] = (
            df_merged['preco_brl'].rolling(window=window).std()
        )

    # C. Precipitação Acumulada
    for window in [7, 15, 30]:
        df_merged[f'precipitacao_acc_{window}'] = (
            df_merged['precipitacao_mm'].rolling(window=window).sum()
        )

    # D. Remoção de NaNs e Arredondamento para Limpeza Visual (4 casas decimais)
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