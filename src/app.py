import streamlit as st
import pandas as pd
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,root_mean_squared_error
import time
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import yfinance as yf

import pandas_ta as ta
import matplotlib.pyplot as plt
# Función para descargar datos OHLCV desde Binance
def get_binance_ohlcv(symbol, interval="1d", days=4000):
    """
    Descarga velas OHLCV desde Binance.
    symbol: par de trading (ej: BTCUSDT)
    interval: intervalo de vela (ej: 1h, 1d, 15m)
    days: días de datos hacia atrás
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Binance limita 1000 velas por request → dividir en chunks
    limit = 1000
    ms_interval = 60 * 60 * 1000  # 1h en milisegundos
    if interval == "1d":
        ms_interval = 24 * 60 * 60 * 1000
    
    end_time = int(time.time() * 1000)  # ahora en ms
    start_time = end_time - days * 24 * 60 * 60 * 1000
    
    all_data = []
    
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": limit
        }
        resp = requests.get(base_url, params=params)
        data = resp.json()
        
        if not data:
            break
        
        all_data.extend(data)
        
        # avanzar el start_time al último timestamp + intervalo
        last_open_time = data[-1][0]
        start_time = last_open_time + ms_interval
        
        time.sleep(0.2)  # para no sobrecargar la API
    
    # convertir a DataFrame
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    # limpiar tipos de datos
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    return df[["open_time", "open", "high", "low", "close", "volume","number_of_trades"]]

def crear_features_modelo(data, short_window=50, long_window=200, rsi_period=14, bollinger_window=20):
    """
    Calcula un conjunto de características numéricas (features) para un modelo de ML
    basado en estrategias de análisis técnico.
    
    Args:
        data (pd.DataFrame): DataFrame con datos OHLCV.
        
    Returns:
        pd.DataFrame: DataFrame con las nuevas características calculadas.
    """
    df = data.copy()
    
    # --- Feature 1: Retornos y volumen ---
    df['retorno_diario'] = df["close"].pct_change()
    df['cambio_volumen'] = df['volume'].pct_change()
    
    # --- Features de la Estrategia de Cruce de Medias Móviles ---
    sma_corta = df["close"].rolling(window=short_window).mean()
    sma_larga = df["close"].rolling(window=long_window).mean()
    
    df['distancia_smas'] = sma_corta - sma_larga
    # Evitamos dividir por cero si la SMA larga es 0 en algún punto inicial
    df['ratio_smas'] = sma_corta / sma_larga.replace(0, np.nan)
    
    # --- Features de la Estrategia de Bollinger + RSI ---
    # Calcular RSI
    df.ta.rsi(length=rsi_period, append=True)
    df.rename(columns={f'RSI_{rsi_period}': 'valor_rsi'}, inplace=True)
    
    # Calcular Bandas de Bollinger
    sma_bb = df["close"].rolling(window=bollinger_window).mean()
    std_bb = df["close"].rolling(window=bollinger_window).std()
    banda_sup = sma_bb + (std_bb * 2)
    banda_inf = sma_bb - (std_bb * 2)
    
    # Feature: Posición normalizada del precio dentro de las bandas
    # Un valor > 0 está por encima de la media, < 0 por debajo.
    df['posicion_en_bandas'] = (df["close"] - sma_bb) / std_bb.replace(0, np.nan)
    
    # Feature: Ancho de las bandas (mide volatilidad)
    df['ancho_bandas'] = (banda_sup - banda_inf) / sma_bb.replace(0, np.nan)
    

    
    # Eliminar filas con NaN generadas por los cálculos de ventanas móviles
    df.dropna(inplace=True)
    
    return df

def compute_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi 

#bandas de bollinger
def compute_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band



#estrategia de bandas y RSI


def estrategia_bb_rsi(data, window=20, num_std=2, rsi_period=14, rsi_overbought=70, rsi_oversold=30):

    df = data.copy()
    
    # 1. Calcular Bandas de Bollinger
    df['SMA'] = df['close'].rolling(window=window).mean()
    df['STD'] = df['close'].rolling(window=window).std()
    df['Banda_Superior'] = df['SMA'] + (df['STD'] * num_std)
    df['Banda_Inferior'] = df['SMA'] - (df['STD'] * num_std) 
    df.ta.rsi(length=rsi_period, append=True)
    rsi_col_name = f'RSI_{rsi_period}' 
    condicion_compra = (df['close'] <= df['Banda_Inferior']) & (df[rsi_col_name] < rsi_oversold)
    
    condicion_venta = (df['close'] >= df['Banda_Superior']) & (df[rsi_col_name] > rsi_overbought)
    
    
    df['Señal_ByR'] = np.nan
    df.loc[condicion_compra, 'Señal_ByR'] = 1  
    df.loc[condicion_venta, 'Señal_ByR'] = 0   
    return df


#estrategia de cruce de medias móviles (Golden/Death Cross)

def estrategia_cruce_medias(data, short_window=50, long_window=200):
    """
    Replica la estrategia de cruce de medias móviles (Golden/Death Cross).
    
    Args:
        data (pd.DataFrame): DataFrame con los datos de precios (debe tener 'close').
        short_window (int): Periodo para la media móvil corta.
        long_window (int): Periodo para la media móvil larga.
        
    Returns:
        pd.DataFrame: El DataFrame original con las medias móviles y la columna 'Señal'.
    """
    # Se crea una copia para no modificar el df original
    df = data.copy()
    
    # 1. Calcular las Medias Móviles Simples (SMA)
    df['SMA_Corta'] = df['close'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_Larga'] = df['close'].rolling(window=long_window, min_periods=1).mean()
    
    # 2. Definir las condiciones de cruce
    # Se compara la posición de hoy con la de ayer (.shift(1)) para encontrar el punto exacto del cruce.
    
    # Golden Cross (Cruce Alcista): La SMA corta cruza por encima de la SMA larga.
    condicion_alza = (df['SMA_Corta'] > df['SMA_Larga']) & (df['SMA_Corta'].shift(1) <= df['SMA_Larga'].shift(1))
    
    # Death Cross (Cruce Bajista): La SMA corta cruza por debajo de la SMA larga.
    condicion_baja = (df['SMA_Corta'] < df['SMA_Larga']) & (df['SMA_Corta'].shift(1) >= df['SMA_Larga'].shift(1))
    
    # 3. Crear la columna de Señal
    # Se inicializa con NaN (sin señal)
    df['Señal'] = np.nan 
    # Se asigna 1 en los puntos de cruce alcista
    df.loc[condicion_alza, 'Señal'] = 1
    # Se asigna 0 en los puntos de cruce bajista
    df.loc[condicion_baja, 'Señal'] = 0
    
    return df

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Portada Cripto",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("<h1 style='text-align: center;'>Ciprto Ball 🔮</h1>", unsafe_allow_html=True)
# --- Barra Lateral (Sidebar) ---
st.sidebar.title('Selección de Activos')

# Lista de criptomonedas que puedes modificar
lista_criptos = [
    'BTCUSDT',
    'ETHUSDT',
    'ADAUSDT',
    'SOLUSDT',

]

# Creamos el menú desplegable en la barra lateral
opcion_seleccionada = st.sidebar.selectbox(
    'Elige una criptomoneda:',
    lista_criptos
)

# --- Contenido Principal de la Página ---
st.title(f'Análisis de {opcion_seleccionada}')
df = get_binance_ohlcv(str(opcion_seleccionada), interval="1d", days=4000)


df=crear_features_modelo(df)
#porcentaje de cambio de 1,7,14,30 dias
df['pct_change_1d'] = df['close'].pct_change(periods=1) * 100
df['pct_change_7d'] = df['close'].pct_change(periods=7) * 100
df['pct_change_14d'] = df['close'].pct_change(periods=14) * 100
df['pct_change_30d'] = df['close'].pct_change(periods=30) * 100
#calcular RSI
df['RSI'] = compute_rsi(df)
df['Banda_Media'], df['Banda_Superior'], df['Banda_Inferior'] = compute_bollinger_bands(df)
#medias moviles de 9,26,50,200
df['SMA_9'] = df['close'].rolling(window=9).mean()
df['SMA_26'] = df['close'].rolling(window=26).mean()
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['SMA_200'] = df['close'].rolling(window=200).mean()
df.dropna(inplace=True)
df = estrategia_bb_rsi(df)
df = estrategia_cruce_medias(df, short_window=50, long_window=200)
df.reset_index(drop=True, inplace=True)
horizonte = 1
df['Target'] = (df['close'].shift(-horizonte) > df['close']).astype(int)



# Extraer el ticker del nombre (ej: de 'Bitcoin (BTC-USD)' a 'BTC-USD')


# --- Función para cargar datos (con caché para velocidad) ---


# Cargar los datos de la cripto seleccionada


if df is not None:

    n_filas = len(df["close"])
    restricciones = ['open_time', 'open',  'close', 'volume',
            'retorno_diario', 
            'distancia_smas', 'ratio_smas', 'valor_rsi', 
            'ancho_bandas', 'pct_change_1d', 'pct_change_7d', 'pct_change_14d',
            'pct_change_30d', 'SMA_9', 'SMA_26', 'SMA_50', 'SMA_200', 'Target',
            'SMA', 'STD', 
            'SMA_Corta', 'SMA_Larga'] 
    medir = 'Target'

    entreno = df[0:n_filas-1]
    X_entreno = entreno.drop(restricciones, axis=1)
    y = entreno[medir]
    X_train, X_test, y_train, y_test = train_test_split(X_entreno, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42, n_estimators=30, max_depth=1, min_samples_split=2, min_samples_leaf=3)
    model.fit(X_train, y_train)
        #train y test 
    if st.button("Acirtos de los ultimos 100 dias"):
                
        #train y test 
        n_filas=len(df["close"])
        corr_direccion=0


        y_pred_guardadas=[]
        y_real_guardadas=[]
        restricciones=['open_time', 'open',  'close', 'volume',
            'retorno_diario', 
            'distancia_smas', 'ratio_smas', 'valor_rsi', 
            'ancho_bandas', 'pct_change_1d', 'pct_change_7d', 'pct_change_14d',
            'pct_change_30d', 'SMA_9', 'SMA_26', 'SMA_50', 'SMA_200', 'Target',
            'SMA', 'STD', 
            'SMA_Corta', 'SMA_Larga']#'number_of_trades'

        medir='Target'
        for i in range(100):
            print(i)
            #vamos a separar los datos de entreno y los outsider
            entreno=df[0+2+i:n_filas-101+i]
            outsider=df[n_filas-101+i+1:n_filas-101+i+2]
            #print(outsider)
            outsider.reset_index(drop=True,inplace=True)
            entreno.reset_index(drop=True,inplace=True)
            #ahora ponemos el y el x
            X_entreno=entreno.drop(restricciones,axis=1)
            y=entreno[medir]
            X_train, X_test, y_train, y_test = train_test_split(X_entreno, y, test_size=0.2, random_state=42)
            #model=RandomForestRegressor( n_estimators=100, random_state=42)
            model=RandomForestClassifier( random_state=42,n_estimators= 30, max_depth= 1, min_samples_split= 2, min_samples_leaf= 3)

            model.fit(X_train,y_train)
            X_outsider=outsider.drop(restricciones,axis=1)
            y_outsider=outsider[medir]#es el real
            predicciones_outsider=model.predict(X_outsider)
            y_pred_guardadas.append(predicciones_outsider[0])
            y_real_guardadas.append(y_outsider[0])


            if predicciones_outsider[0]==y_outsider[0]:
                corr_direccion+=1
        aciertos=corr_direccion/100
        st.write("Obtuvo un acierto del: "+str(round(aciertos*100,0))+"%") 
        

    if st.button("predecir precio"):
        
        pred = model.predict(df.drop(restricciones, axis=1).tail(1))
        if pred[0] == 1:
            st.write("El modelo predice que el precio subirá mañana 🚀🚀🚀")
        else:
            st.write("El modelo predice que el precio bajará mañana 📉📉📉")
 
    # Puedes poner tu código de predicción aquí

        

    # --- Mostrar tabla con datos recientes ---
    st.subheader(f'Gráfico Histórico de {opcion_seleccionada}')
    st.line_chart(df['close'])