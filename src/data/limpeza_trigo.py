from pathlib import Path
import pandas as pd

# Definição dinâmica de caminhos (Subindo de src/data/ até a raiz TCC/)
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_COMMODITIES_DIR = BASE_DIR / 'data' / 'raw' / 'commodities'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'


def tratar_e_limpar_precos_trigo(data_inicio='2008-01-01', data_fim='2026-04-20'):
    caminho_csv = RAW_COMMODITIES_DIR / 'Serie de precos trigo CEPEA.csv'

    if not caminho_csv.exists():
        raise FileNotFoundError(
            f'Arquivo do CEPEA não encontrado em: {caminho_csv}'
        )

    print(f'=== Processando Cotações da Commodity: {caminho_csv.name} ===')

    # Leitura do CSV do CEPEA: pula as 3 linhas de metadados e lê o cabeçalho
    # (Data; À vista R$; À vista US$). O arquivo traz colunas vazias extras
    # (';;;;;;;'), então mantemos apenas as 3 primeiras.
    df = pd.read_csv(
        caminho_csv,
        sep=';',
        skiprows=3,
        encoding='latin1',
        decimal=',',
        usecols=[0, 1, 2],
    )
    df.columns = ['data', 'preco_brl', 'preco_usd']

    # Converter e validar coluna de datas
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['data']).sort_values('data')

    # Conversão das cotações para numérico
    df['preco_brl'] = pd.to_numeric(df['preco_brl'], errors='coerce')
    df['preco_usd'] = pd.to_numeric(df['preco_usd'], errors='coerce')

    # Remover eventuais datas duplicadas (mantém a última cotação do dia)
    df = df.drop_duplicates(subset='data', keep='last')

    # Filtro temporal: 2008-01-01 a 2026-04-20
    df = df[(df['data'] >= data_inicio) & (df['data'] <= data_fim)]

    # Criar calendário diário contínuo (preenchendo finais de semana/feriados
    # com o último valor disponível - ffill). O calendário começa na primeira
    # cotação disponível do período (01/01 é feriado, sem pregão), seguindo o
    # mesmo padrão da série do arroz.
    min_date = df['data'].min()
    max_date = df['data'].max()
    calendario_completo = pd.date_range(start=min_date, end=max_date, freq='D')

    df_continuo = (
        df.set_index('data')
        .reindex(calendario_completo)
        .ffill()
    )

    # Tratamento de valores ausentes remanescentes: interpolação linear
    # (cobre falhas internas que porventura não tenham sido preenchidas)
    df_continuo = df_continuo.interpolate(method='linear', limit_direction='both')

    df_continuo = df_continuo.rename_axis('data').reset_index()

    # Garantir criação do diretório de saída
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    caminho_saida = PROCESSED_DIR / 'preco_trigo_limpo.csv'
    df_continuo.to_csv(caminho_saida, index=False, date_format='%Y-%m-%d')

    print(f' Dataset de preços limpo e contínuo salvo em: {caminho_saida}')
    print(
        f' Total de dias: {len(df_continuo)} | '
        f'Período: {df_continuo["data"].min().strftime("%Y-%m-%d")} a {df_continuo["data"].max().strftime("%Y-%m-%d")} | '
        f'Nulos restantes: {df_continuo.isna().sum().sum()}\n'
    )

    return df_continuo


if __name__ == '__main__':
    tratar_e_limpar_precos_trigo()
