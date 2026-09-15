from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'


def limpar_dados_dolar():
    arq_dolar_raw = (
        RAW_DIR / 'macroeconomia' / 'Serie cotacao dolar Banco Central.csv'
    )
    arq_dolar_saida = PROCESSED_DIR / 'dolar_limpo.csv'

    if not arq_dolar_raw.exists():
        raise FileNotFoundError(f'Arquivo não encontrado em: {arq_dolar_raw}')

    print('=== 1. Carregando e Limpando Cotação do Dólar ===')
    df = pd.read_csv(
        arq_dolar_raw, encoding='latin1', sep=None, engine='python'
    )

    # Seleção de colunas e filtragem de datas válidas
    col_data, col_valor = df.columns[0], df.columns[1]
    df = df[
        df[col_data].astype(str).str.contains(r'\d{2}/\d{2}/\d{4}', na=False)
    ].copy()

    # Conversão de tipos
    df['data'] = pd.to_datetime(df[col_data], format='%d/%m/%Y')
    df['dolar_venda'] = (
        df[col_valor].astype(str).str.replace(',', '.').astype(float)
    )

    # Ordenação e eliminação de duplicatas de data se houver
    df_dolar = (
        df[['data', 'dolar_venda']]
        .drop_duplicates(subset=['data'])
        .sort_values('data')
    )

    print('=== 2. Reindexando para Calendário Diário Contínuo ===')
    # Cria a grade diária completa a partir de 2008 até a última data disponível
    data_inicio = '2008-01-01'
    data_fim = df_dolar['data'].max()

    grid_datas = pd.date_range(
        start=data_inicio, end=data_fim, freq='D', name='data'
    )
    df_limpo = (
        pd.DataFrame(index=grid_datas)
        .merge(df_dolar, on='data', how='left')
        .sort_values('data')
    )

    # Preenche fins de semana e feriados com a última cotação de dia útil (Forward Fill)
    df_limpo['dolar_venda'] = df_limpo['dolar_venda'].ffill().bfill()

    # Exportação para data/processed/
    df_limpo.to_csv(arq_dolar_saida, index=False)
    print(f'✔ Dados do Dólar limpos salvos em: {arq_dolar_saida}')
    print(
        f'  Registros diários: {len(df_limpo)} | Período: {df_limpo["data"].min().strftime("%Y-%m-%d")} a {df_limpo["data"].max().strftime("%Y-%m-%d")}\n'
    )


if __name__ == '__main__':
    limpar_dados_dolar()