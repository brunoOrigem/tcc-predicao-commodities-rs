import os
from pathlib import Path
import pandas as pd
import xlrd

# Definição dinâmica de caminhos (Subindo de src/data/ até a raiz TCC/)
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_COMMODITIES_DIR = BASE_DIR / 'data' / 'raw' / 'commodities'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'


def tratar_e_limpar_precos_arroz(data_inicio='2008-01-01'):
    caminho_xls = RAW_COMMODITIES_DIR / 'Serie de precos arroz CEPEA.xls'

    if not caminho_xls.exists():
        raise FileNotFoundError(
            f'Arquivo do CEPEA não encontrado em: {caminho_xls}'
        )

    print(f'=== Processando Cotações da Commodity: {caminho_xls.name} ===')

    # Abertura da planilha do CEPEA com xlrd (compatível com formato legado .xls)
    workbook = xlrd.open_workbook(
        caminho_xls, ignore_workbook_corruption=True
    )
    sheet = workbook.sheet_by_index(0)

    # Extrai todas as linhas da planilha
    rows = [sheet.row_values(i) for i in range(sheet.nrows)]

    # Os dados iniciam na linha index 4 (pulando cabeçalhos e metadados das linhas 0-3)
    df = pd.DataFrame(rows[4:], columns=['data', 'preco_brl', 'preco_usd'])

    # Converter e validar coluna de datas
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['data']).sort_values('data')

    # Filtrar histórico a partir de 2008 para alinhar perfeitamente com os dados climáticos
    df = df[df['data'] >= data_inicio]

    # Conversão das cotações para numérico
    df['preco_brl'] = pd.to_numeric(df['preco_brl'], errors='coerce')
    df['preco_usd'] = pd.to_numeric(df['preco_usd'], errors='coerce')

    # Criar calendário diário contínuo (preenchendo finais de semana/feriados com o último valor disponível - ffill)
    min_date = df['data'].min()
    max_date = df['data'].max()
    calendario_completo = pd.date_range(start=min_date, end=max_date, freq='D')

    df_continuo = (
        df.set_index('data')
        .reindex(calendario_completo)
        .ffill()
        .reset_index()
        .rename(columns={'index': 'data'})
    )

    # Garantir criação do diretório de saída
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    caminho_saida = PROCESSED_DIR / 'preco_arroz_limpo.csv'
    df_continuo.to_csv(caminho_saida, index=False)

    print(f' Dataset de preços limpo e contínuo salvo em: {caminho_saida}')
    print(
        f' Total de dias: {len(df_continuo)} | '
        f'Período: {df_continuo["data"].min().strftime("%Y-%m-%d")} a {df_continuo["data"].max().strftime("%Y-%m-%d")} | '
        f'Nulos restantes: {df_continuo.isna().sum().sum()}\n'
    )


if __name__ == '__main__':
    tratar_e_limpar_precos_arroz()