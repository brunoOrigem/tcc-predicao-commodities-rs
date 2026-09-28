from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'


def limpar_dados_ipca():
    arq_ipca_raw = (
        RAW_DIR / 'macroeconomia' / 'serie historica inflacao ipca.xls'
    )
    arq_ipca_saida = PROCESSED_DIR / 'ipca_limpo.csv'

    if not arq_ipca_raw.exists():
        raise FileNotFoundError(f'Arquivo não encontrado em: {arq_ipca_raw}')

    print('=== 1. Carregando e Limpando Série Histórica do IPCA ===')
    df_raw = pd.read_excel(arq_ipca_raw, sheet_name=0)

    # Seleção de linhas de dados e colunas essenciais
    df_data = df_raw.iloc[7:].copy()
    df_data = df_data.iloc[:, [0, 1, 2, 3, 7]]
    df_data.columns = ['ano', 'mes', 'ipca_indice', 'ipca_mensal', 'ipca_12m']

    # Remove linhas vazias e preenche o ano nos meses subsequentes (forward fill)
    df_data = df_data.dropna(subset=['mes']).copy()
    df_data['ano'] = df_data['ano'].ffill()

    # Mapeamento dos meses para número
    meses_map = {
        'JAN': 1,
        'FEV': 2,
        'MAR': 3,
        'ABR': 4,
        'MAI': 5,
        'JUN': 6,
        'JUL': 7,
        'AGO': 8,
        'SET': 9,
        'OUT': 10,
        'NOV': 11,
        'DEZ': 12,
    }
    df_data['mes_num'] = df_data['mes'].str.strip().str.upper().map(meses_map)
    df_data = df_data.dropna(subset=['mes_num']).copy()

    # Construção da coluna de data
    df_data['data'] = pd.to_datetime(
        df_data['ano'].astype(int).astype(str)
        + '-'
        + df_data['mes_num'].astype(int).astype(str).str.zfill(2)
        + '-01'
    )

    # Conversão de colunas numéricas
    for col in ['ipca_indice', 'ipca_mensal', 'ipca_12m']:
        df_data[col] = pd.to_numeric(df_data[col], errors='coerce')

    df_mensal = (
        df_data[['data', 'ipca_indice', 'ipca_mensal', 'ipca_12m']]
        .sort_values('data')
        .reset_index(drop=True)
    )

    # Filtragem de 2008 até abril de 2026
    df_mensal = df_mensal[
        (df_mensal['data'] >= '2008-01-01')
        & (df_mensal['data'] <= '2026-04-01')
    ].reset_index(drop=True)

    print('=== 2. Reindexando para Calendário Diário Contínuo ===')
    data_inicio = '2008-01-01'
    data_fim = '2026-04-20'  # Limite exato da base do projeto

    # Cria a grade de datas diárias
    grid_datas = pd.date_range(
        start=data_inicio, end=data_fim, freq='D', name='data'
    )
    df_limpo = (
        pd.DataFrame(index=grid_datas)
        .merge(df_mensal, on='data', how='left')
        .sort_values('data')
    )

    # Propaga o valor do IPCA mensal para todos os dias do mês (Forward Fill)
    cols_ipca = ['ipca_indice', 'ipca_mensal', 'ipca_12m']
    df_limpo[cols_ipca] = df_limpo[cols_ipca].ffill().bfill()

    # Exportação para data/processed/
    df_limpo.to_csv(arq_ipca_saida, index=False)
    print(f'✔ Dados do IPCA limpos salvos em: {arq_ipca_saida}')
    print(
        f'  Registros diários: {len(df_limpo)} | Período: {df_limpo["data"].min().strftime("%Y-%m-%d")} a {df_limpo["data"].max().strftime("%Y-%m-%d")}\n'
    )


if __name__ == '__main__':
    limpar_dados_ipca()