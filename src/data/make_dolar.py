import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

def limpar_dados_dolar():
    arq_dolar_raw = RAW_DIR / 'Serie cotacao dolar Banco Central.csv'
    arq_dolar_saida = PROCESSED_DIR / 'dolar_limpo.csv'
    
    if not arq_dolar_raw.exists():
        raise FileNotFoundError(f'Arquivo não encontrado em: {arq_dolar_raw}')
        
    print('=== 1. Carregando e Limpando Cotação do Dólar ===')
    # Leitura com suporte a caracteres latinos
    df = pd.read_csv(arq_dolar_raw, encoding='latin1', sep=None, engine='python')
    
    # Seleção de colunas e filtragem de datas válidas
    col_data, col_valor = df.columns[0], df.columns[1]
    df = df[df[col_data].astype(str).str.contains(r'\d{2}/\d{2}/\d{4}', na=False)].copy()
    
    # Tratamento de tipos
    df['data'] = pd.to_datetime(df[col_data], format='%d/%m/%Y')
    df['dolar_venda'] = df[col_valor].astype(str).str.replace(',', '.').astype(float)
    
    # Ordenação e seleção final
    df_limpo = df[['data', 'dolar_venda']].sort_values('data').reset_index(drop=True)
    
    # Exportação para data/processed/
    df_limpo.to_csv(arq_dolar_saida, index=False)
    print(f'✔ Dados do Dólar limpos salvos em: {arq_dolar_saida}')
    print(f'  Registros: {len(df_limpo)} | Período: {df_limpo["data"].min().strftime("%Y-%m-%d")} a {df_limpo["data"].max().strftime("%Y-%m-%d")}\n')

if __name__ == '__main__':
    limpar_dados_dolar()