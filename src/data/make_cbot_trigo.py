from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

# Contrato futuro contínuo do trigo SRW na Bolsa de Chicago (CBOT), via Yahoo Finance.
# Cotado em centavos de dólar por bushel (US¢/bu).
TICKER = 'ZW=F'
BUSHELS_POR_TONELADA = 36.7437   # 1 t de trigo = 36,7437 bushels (60 lb/bu)


def baixar_cbot_trigo(forcar=False):
    """Baixa o histórico diário do ZW=F para data/raw/commodities/ (uma única vez)."""
    arq_raw = RAW_DIR / 'commodities' / 'Serie trigo CBOT Yahoo ZW=F.csv'
    if arq_raw.exists() and not forcar:
        return arq_raw

    import yfinance as yf

    print(f'=== 0. Baixando {TICKER} do Yahoo Finance ===')
    df = yf.download(
        TICKER, start='2007-12-01', interval='1d', auto_adjust=False, progress=False
    )
    if df.empty:
        raise RuntimeError(f'Download de {TICKER} retornou vazio.')
    if isinstance(df.columns, pd.MultiIndex):   # yfinance >= 0.2 devolve (campo, ticker)
        df.columns = df.columns.get_level_values(0)

    arq_raw.parent.mkdir(parents=True, exist_ok=True)
    df[['Close']].rename_axis('Date').to_csv(arq_raw)
    print(f'✔ Série bruta salva em: {arq_raw}')
    return arq_raw


def limpar_dados_cbot_trigo():
    arq_raw = baixar_cbot_trigo()
    arq_saida = PROCESSED_DIR / 'cbot_trigo_limpo.csv'

    print('=== 1. Carregando e Limpando Preço do Trigo CBOT ===')
    df = pd.read_csv(arq_raw, parse_dates=['Date'])
    df = df.rename(columns={'Date': 'data'}).dropna(subset=['Close'])
    df['data'] = df['data'].dt.tz_localize(None).dt.normalize()

    # Conversão de US¢/bushel para US$/tonelada
    df['cbot_usd_t'] = (df['Close'] / 100 * BUSHELS_POR_TONELADA).round(4)
    df_cbot = df[['data', 'cbot_usd_t']].drop_duplicates(subset=['data']).sort_values('data')

    print('=== 2. Reindexando para Calendário Diário Contínuo ===')
    grid_datas = pd.date_range(
        start='2008-01-01', end=df_cbot['data'].max(), freq='D', name='data'
    )
    df_limpo = (
        pd.DataFrame(index=grid_datas)
        .merge(df_cbot, on='data', how='left')
        .sort_values('data')
    )

    # Fins de semana e feriados nos EUA recebem a última cotação conhecida (Forward Fill)
    n_vazios = df_limpo['cbot_usd_t'].isna().sum()
    df_limpo['cbot_usd_t'] = df_limpo['cbot_usd_t'].ffill()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_limpo.to_csv(arq_saida, index=False)
    print(f'✔ Dados do trigo CBOT limpos salvos em: {arq_saida}')
    print(
        f'  Registros diários: {len(df_limpo)} | dias preenchidos por ffill: {n_vazios} | '
        f'Período: {df_limpo["data"].min():%Y-%m-%d} a {df_limpo["data"].max():%Y-%m-%d}\n'
    )


if __name__ == '__main__':
    limpar_dados_cbot_trigo()
