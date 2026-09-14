import os
from pathlib import Path
import pandas as pd

# Subindo de /app/src/data/ até a raiz /app (parents[2])
BASE_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = BASE_DIR / 'data' / 'interim'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'


def tratar_e_limpar_dataset(
    caminho_csv_entrada, nome_saida, data_limite_max='2026-04-20'
):
    print(f'=== Processando {caminho_csv_entrada.name} ===')

    if not os.path.exists(caminho_csv_entrada):
        raise FileNotFoundError(
            f'Arquivo de entrada não encontrado: {caminho_csv_entrada}'
        )

    df = pd.read_csv(caminho_csv_entrada)

    # 1. Parsing de datas com dayfirst=True
    df['data'] = pd.to_datetime(
        df['data'], format='mixed', dayfirst=True, errors='coerce'
    )
    df = df.dropna(subset=['data'])

    # 2. Trava estrita na data limite máxima (20/04/2026)
    if data_limite_max:
        df = df[df['data'] <= pd.to_datetime(data_limite_max)]

    dfs_limpos = []

    for praca, group in df.groupby('praca'):
        group = group.sort_values('data').set_index('data')

        min_date = group.index.min()
        max_date = group.index.max()

        # Reindexação temporal limitada
        calendario_completo = pd.date_range(
            start=min_date, end=max_date, freq='D'
        )

        group_reindexed = group.reindex(calendario_completo)
        group_reindexed['praca'] = praca

        # Tratamento da precipitação
        group_reindexed['precipitacao_mm'] = group_reindexed[
            'precipitacao_mm'
        ].fillna(0)

        # Interpolação linear para contínuos
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

        dfs_limpos.append(
            group_reindexed.reset_index().rename(columns={'index': 'data'})
        )

    df_final = pd.concat(dfs_limpos, ignore_index=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    caminho_final = PROCESSED_DIR / nome_saida
    df_final.to_csv(caminho_final, index=False)

    print(f' Dataset limpo salvo em: {caminho_final}')
    print(
        f' Intervalo final: {df_final["data"].min().strftime("%Y-%m-%d")} a {df_final["data"].max().strftime("%Y-%m-%d")}'
    )
    print(
        f' Total de linhas: {len(df_final)} | Nulos restantes: {df_final.isna().sum().sum()}\n'
    )


if __name__ == '__main__':
    tratar_e_limpar_dataset(
        INTERIM_DIR / 'dados_climaticos_arroz_diarios.csv',
        'dados_climaticos_arroz_limpos.csv',
        data_limite_max='2026-04-20',
    )
    tratar_e_limpar_dataset(
        INTERIM_DIR / 'dados_climaticos_soja_trigo_diarios.csv',
        'dados_climaticos_soja_trigo_limpos.csv',
        data_limite_max='2026-04-20',
    )