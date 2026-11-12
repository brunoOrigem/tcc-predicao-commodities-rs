"""Pesos temporais da atenção aditiva (Bahdanau) sobre a LSTM do arroz.

Gera models/outputs/arroz_lstm_attention_pesos.png: o peso médio que a camada de
atenção atribui a cada um dos 30 dias da janela de entrada.

Experimento descartado como modelo de previsão (o modelo oficial é a LSTM com 36
variáveis em models/lstm_arroz.ipynb); mantido apenas para a discussão do mecanismo
de atenção na monografia.

Pipeline (o mesmo do experimento original, para reproduzir os mesmos pesos):
    1. Carrega data/processed/dataset_final_arroz.csv (36 variáveis candidatas).
    2. Treino até 31/01/2026; teste = 30 dias seguintes.
    3. Random Forest (só no treino) escolhe as TOP_K variáveis mais importantes para o preço de t+1.
    4. MinMaxScaler (fit só no treino) + janelas de LOOKBACK = 30 dias.
    5. LSTM(64, return_sequences=True) -> Atenção aditiva -> contexto -> Dense(1).
    6. Projeção recursiva de 30 dias; em cada passo, registra os pesos de atenção.
    7. Plota a média dos pesos ao longo dos 30 passos.

Uso (da raiz do projeto):
    python src/models/plotar_pesos_atencao.py
"""

import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parents[2]
ARQ_DADOS = BASE_DIR / "data" / "processed" / "dataset_final_arroz.csv"
PASTA_SAIDA = BASE_DIR / "models" / "outputs"
ARQ_FIGURA = PASTA_SAIDA / "arroz_lstm_attention_pesos.png"

ALVO = "preco_brl"
FIM_TREINO = "2026-01-31"
INICIO_TESTE = "2026-02-01"
HORIZONTE = 30
LOOKBACK = 30
TOP_K = 15
SEMENTE = 42

# Variáveis derivadas do preço, que precisam ser recalculadas na recursão
LAGS_PRECO = [1, 7, 14, 30]
JANELAS_PRECO = [7, 14, 30]


def fixar_sementes(semente=SEMENTE):
    random.seed(semente)
    np.random.seed(semente)
    tf.random.set_seed(semente)


# --------------------------------------------------------------------------- #
# Dados e seleção de variáveis
# --------------------------------------------------------------------------- #
def carregar_dados():
    dados = pd.read_csv(ARQ_DADOS, parse_dates=["data"], index_col="data").asfreq("D")
    assert dados.isna().sum().sum() == 0, "Dataset com valores ausentes"
    candidatas = [c for c in dados.columns if c != ALVO]
    assert len(candidatas) == 36, f"Esperava 36 variáveis, encontrei {len(candidatas)}"
    return dados, candidatas


def selecionar_variaveis(treino, candidatas, top_k=TOP_K):
    """Importância (Random Forest) das variáveis de t para prever o preço de t+1; só treino."""
    X = treino[candidatas].iloc[:-1]
    y = treino[ALVO].shift(-1).iloc[:-1]
    rf = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=SEMENTE,
    ).fit(X, y)
    importancias = pd.Series(rf.feature_importances_, index=candidatas).sort_values(ascending=False)
    return importancias, list(importancias.index[:top_k])


def criar_janelas(X, y, lookback):
    Xs = np.array([X[i - lookback:i] for i in range(lookback, len(X))])
    ys = np.array([y[i] for i in range(lookback, len(X))])
    return Xs, ys


# --------------------------------------------------------------------------- #
# Modelo com atenção
# --------------------------------------------------------------------------- #
@keras.utils.register_keras_serializable(package="tcc")
class AtencaoAditiva(layers.Layer):
    """Atenção aditiva (Bahdanau) sobre os estados ocultos da LSTM.

    score_t = v^T tanh(W h_t + b)    alpha = softmax(score)    contexto = sum_t alpha_t h_t
    """

    def __init__(self, unidades=32, **kwargs):
        super().__init__(**kwargs)
        self.unidades = unidades

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name="W", shape=(d, self.unidades), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(self.unidades,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(self.unidades, 1), initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, h):
        score = tf.matmul(tf.tanh(tf.matmul(h, self.W) + self.b), self.v)   # (batch, T, 1)
        alpha = tf.nn.softmax(score, axis=1)                                 # pesos ao longo do tempo
        contexto = tf.reduce_sum(alpha * h, axis=1)                          # (batch, d)
        return contexto, tf.squeeze(alpha, -1)

    def get_config(self):
        return {**super().get_config(), "unidades": self.unidades}


def modelo_atencao(n_vars):
    entrada = layers.Input(shape=(LOOKBACK, n_vars), name="janela")
    h = layers.LSTM(64, return_sequences=True, name="lstm")(entrada)
    h = layers.Dropout(0.2)(h)
    contexto, alpha = AtencaoAditiva(32, name="atencao")(h)
    saida = layers.Dense(1, name="preco")(contexto)

    modelo = keras.Model(entrada, saida, name="lstm_atencao_arroz")
    modelo_alpha = keras.Model(entrada, alpha, name="pesos_atencao")   # compartilha os pesos
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return modelo, modelo_alpha


def treinar(modelo, X_tr, y_tr):
    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    hist = modelo.fit(
        X_tr, y_tr, epochs=200, batch_size=32, validation_split=0.1,
        shuffle=True, callbacks=[parada], verbose=0,
    )
    melhor = int(np.argmin(hist.history["val_loss"]))
    print(f"   épocas: {len(hist.history['loss'])} | melhor: {melhor + 1} "
          f"| val_loss = {hist.history['val_loss'][melhor]:.6f}")


# --------------------------------------------------------------------------- #
# Pesos de atenção na projeção recursiva
# --------------------------------------------------------------------------- #
def atualizar_derivadas(df, data):
    """Recalcula, na linha `data`, as variáveis que dependem do preço (já atualizado em df)."""
    p = df[ALVO]
    pos = df.index.get_loc(data)
    df.at[data, "preco_usd"] = round(p.iloc[pos] / df.at[data, "dolar_venda"], 4)
    for lag in LAGS_PRECO:
        df.at[data, f"preco_brl_lag_{lag}"] = p.iloc[pos - lag]
    for w in JANELAS_PRECO:
        janela = p.iloc[pos - w + 1:pos + 1]
        df.at[data, f"preco_brl_ma_{w}"] = round(janela.mean(), 4)
        df.at[data, f"preco_brl_std_{w}"] = round(janela.std(), 4)


def pesos_recursivos(modelo, modelo_alpha, dados, features, escala_X, escala_y, datas_teste):
    """Projeta 30 dias recursivamente e devolve os pesos de atenção de cada passo (30, LOOKBACK)."""
    df = dados.loc[:datas_teste[-1]].copy()
    alphas = []
    for d in datas_teste:
        janela = df.loc[d - pd.Timedelta(days=LOOKBACK):d - pd.Timedelta(days=1), features]
        x = escala_X.transform(janela)[np.newaxis]
        alphas.append(modelo_alpha.predict(x, verbose=0)[0])
        df.at[d, ALVO] = float(escala_y.inverse_transform(modelo.predict(x, verbose=0))[0, 0])
        atualizar_derivadas(df, d)
    return np.array(alphas)


def grafico_atencao(alphas, caminho):
    media = alphas.mean(axis=0)
    defasagem = np.arange(LOOKBACK, 0, -1)   # posição 0 da janela = 30 dias atrás
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(defasagem, media, color="#eb6834", width=0.8)
    ax.axhline(1 / LOOKBACK, color="gray", ls=":", lw=1, label="Peso uniforme (1/30)")
    ax.invert_xaxis()
    ax.set_xlabel("Dias antes da previsão")
    ax.set_ylabel("Peso médio de atenção")
    ax.set_title("Relevância temporal aprendida pela atenção (média dos 30 passos recursivos)")
    ax.grid(alpha=0.3, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(caminho, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def main():
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    tf.config.experimental.enable_op_determinism()

    dados, candidatas = carregar_dados()
    treino = dados.loc[:FIM_TREINO]
    datas_teste = dados.loc[INICIO_TESTE:].index[:HORIZONTE]

    print(f"=== Seleção de variáveis (Random Forest, top {TOP_K}) ===")
    importancias, selecionadas = selecionar_variaveis(treino, candidatas)
    for i, (nome, imp) in enumerate(importancias.iloc[:TOP_K].items(), 1):
        print(f"  {i:2d}. {nome:<22s} {imp:.5f}")

    print("=== Treinando LSTM + Atenção ===")
    fixar_sementes()
    escala_X = MinMaxScaler().fit(treino[selecionadas])
    escala_y = MinMaxScaler().fit(treino[[ALVO]])
    X_tr, y_tr = criar_janelas(escala_X.transform(treino[selecionadas]),
                               escala_y.transform(treino[[ALVO]]).ravel(), LOOKBACK)
    modelo, modelo_alpha = modelo_atencao(len(selecionadas))
    treinar(modelo, X_tr, y_tr)

    alphas = pesos_recursivos(modelo, modelo_alpha, dados, selecionadas, escala_X, escala_y, datas_teste)
    grafico_atencao(alphas, ARQ_FIGURA)
    print(f"Gráfico salvo em {ARQ_FIGURA}")


if __name__ == "__main__":
    main()
