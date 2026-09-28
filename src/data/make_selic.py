from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'


def limpar_dados_selic():
    arq_selic_raw = (
        RAW_DIR / 'macroeconomia' / 'Serie taxa de juros selic Banco Central.csv'
    )
    arq_selic_saida = PROCESSED_DIR / 'selic_limpo.csv'

    if not arq_selic_raw.exists():
        raise FileNotFoundError(f'Arquivo não encontrado em: {arq_selic_raw}')

    print('=== 1. Carregando e Limpando Série Histórica da Selic ===')
    # O arquivo utiliza separador ';' e codificação latin1
    df_raw = pd.read_csv(arq_selic_raw, encoding='latin1', sep=';')

    df_raw.columns = ['data', 'selic_ad']
    df_raw = df_raw.dropna(subset=['selic_ad'])
    df_raw = df_raw[df_raw['data'] != 'Fonte'].copy()

    # Conversão de data e valor numérico (substituindo vírgula por ponto)
    df_raw['data'] = pd.to_datetime(df_raw['data'], format='%d/%m/%Y')
    df_raw['selic_ad'] = (
        df_raw['selic_ad'].astype(str).str.replace(',', '.').astype(float)
    )

    df_raw = df_raw.sort_values('data').reset_index(drop=True)

    # Filtragem de 2008 até 20/04/2026
    df_raw = df_raw[
        (df_raw['data'] >= '2008-01-01') & (df_raw['data'] <= '2026-04-20')
    ].reset_index(drop=True)

    print('=== 2. Reindexando para Calendário Diário Contínuo ===')
    data_inicio = '2008-01-01'
    data_fim = '2026-04-20'

    grid_datas = pd.date_range(
        start=data_inicio, end=data_fim, freq='D', name='data'
    )
    df_limpo = (
        pd.DataFrame(index=grid_datas)
        .merge(df_raw, on='data', how='left')
        .sort_values('data')
    )

    # Propaga os valores (fins de semana e feriados) usando forward fill
    df_limpo['selic_ad'] = df_limpo['selic_ad'].ffill().bfill()

    # Opcional útil para ML: Calcular também a taxa anualizada equivalente (base 252 dias úteis)
    # selic_aa = ((1 + selic_ad/100)^252 - 1) * 100
    df_limpo['selic_aa'] = (
        ((1 + df_limpo['selic_ad'] / 100) ** 252) - 1
    ) * 100

    # Arredondamento e exportação
    df_limpo = df_limpo.round(4)
    df_limpo.to_csv(arq_selic_saida, index=False)

    print(f'✔ Dados da Selic limpos salvos em: {arq_selic_saida}')
    print(
        f'  Registros diários: {len(df_limpo)} | Período: {df_limpo["data"].min().strftime("%Y-%m-%d")} a {df_limpo["data"].max().strftime("%Y-%m-%d")}\n'
    )


if __name__ == '__main__':
    limpar_dados_selic()