import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def tratar_e_limpar_dataset(caminho_csv, nome_saida):
    print(f'=== Processando {caminho_csv} ===')
    df = pd.read_csv(caminho_csv)
    df['data'] = pd.to_datetime(df['data'])

    dfs_limpos = []

    for praca, group in df.groupby('praca'):
        group = group.sort_values('data').set_index('data')

        # 1. Reindexação de Datas (Garante continuidade dia a dia sem saltos)
        min_date = group.index.min()
        max_date = group.index.max()
        calendario_completo = pd.date_range(
            start=min_date, end=max_date, freq='D'
        )

        group_reindexed = group.reindex(calendario_completo)
        group_reindexed['praca'] = praca

        # 2. Trata a precipitação (dias faltantes no calendário assumem 0 mm)
        group_reindexed['precipitacao_mm'] = group_reindexed[
            'precipitacao_mm'
        ].fillna(0)

        # 3. Interpolação Linear para variáveis contínuas (Temperatura e Umidade)
        cols_continuas = [
            'temp_ar_c',
            'temp_max_c',
            'temp_min_c',
            'umidade_relativa',
        ]
        for col in cols_continuas:
            if col in group_reindexed.columns:
                group_reindexed[col] = group_reindexed[col].interpolate(
                    method='linear', limit_direction='both'
                )

        dfs_limpos.append(group_reindexed.reset_index().rename(columns={'index': 'data'}))

    df_final = pd.concat(dfs_limpos, ignore_index=True)

    # Exporta o dataset totalmente sem NaNs
    caminho_final = os.path.join('data', 'processed', nome_saida)
    df_final.to_csv(caminho_final, index=False)
    print(f' Dataset limpo e contínuo salvo em: {caminho_final}')
    print(
        f' Total de linhas: {len(df_final)} | Nulos restantes: {df_final.isna().sum().sum()}\n'
    )


# Execução para as duas regiões
tratar_e_limpar_dataset(
    'data/processed/dados_climaticos_arroz_diarios.csv',
    'dados_climaticos_arroz_limpos.csv',
)
tratar_e_limpar_dataset(
    'data/processed/dados_climaticos_soja_trigo_diarios.csv',
    'dados_climaticos_soja_trigo_limpos.csv',
)