import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#for the app
st.set_page_config(page_title="LSTM Forecasting App", layout="wide")
st.title("📈 General Purpose LSTM Forecasting App")

#main codes LSTM
def train_and_forecast(df, target_col, feature_cols, n_past, n_future, future_macros):
    """
    Trains an LSTM model and returns the forecast and training history.
    """
    # 1. Preparing the data
    all_cols = [target_col] + feature_cols
    df_model = df[all_cols].copy()
    df_model.dropna(inplace=True)

    # 2. Augment data with future macros
    last_date = df_model.index.max()
    future_date = last_date + pd.DateOffset(months=1)
    future_row = pd.DataFrame([np.nan, *future_macros], index=all_cols, columns=[future_date]).T
    future_row.index.name = 'date'
    df_augmented = pd.concat([df_model, future_row])

    # 3. scaling the data
    scaler = StandardScaler()
    scaler = scaler.fit(df_augmented.iloc[:-1])
    df_scaled = scaler.transform(df_augmented)
    df_scaled_df = pd.DataFrame(df_scaled, index=df_augmented.index, columns=all_cols)
    df_scaled_df[target_col].fillna(method='ffill', inplace=True)
    df_scaled = df_scaled_df.values

    # 4.  sequences
    training_data = df_scaled[:-1]
    X_train, y_train = [], []
    for i in range(n_past, len(training_data) - n_future + 1):
        X_train.append(training_data[i - n_past:i, :])
        y_train.append(training_data[i + n_future - 1:i + n_future, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # 5. the model
#this tuning seems to have better accuracy, but will come back after i build this projects part 2 in SQL
    early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=16,
                        validation_split=0.1,
                        callbacks=[early_stopper],
                        verbose=0) # Set to 0 to keep the app interface clean

    # 6. prediction
    prediction_input = np.array([df_scaled[-n_past:]])
    forecast_scaled = model.predict(prediction_input)
    forecast_padded = np.zeros((forecast_scaled.shape[0], len(all_cols)))
    forecast_padded[:, 0] = forecast_scaled.flatten()
    final_forecast = scaler.inverse_transform(forecast_padded)[:, 0]

    return final_forecast[0], history, future_date

# --- Sidebar for Inputs ---
st.sidebar.header('⚙️ User Inputs')

uploaded_file = st.sidebar.file_uploader("Upload your data file (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file:
    
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        df = pd.read_csv(uploaded_file)
    
    # 'date' column exists and is set as index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
    else:
        st.error("Error: Your file must contain a 'date' column.")
        st.stop()
    
    # Column Selection
    st.sidebar.subheader("Select Columns")
    target_col = st.sidebar.selectbox('🎯 Select Target Column to Forecast', df.columns)
    available_features = [col for col in df.columns if col != target_col]
    feature_cols = st.sidebar.multiselect('🧠 Select Feature Columns (Macros)', available_features, default=available_features[:3])

    if not feature_cols:
        st.warning("Please select at least one feature column.")
        st.stop()

    # Model Parameters
    st.sidebar.subheader("Model Parameters")
    n_past = st.sidebar.slider('Lookback Window (Months)', min_value=6, max_value=60, value=24, step=1)
    n_future = 1 # Keeping this fixed for nowcasting as per the logic

    # Future Macro Inputs
    st.sidebar.subheader("🔮 Enter new Macro Values")
    future_macros_input = []
    for col in feature_cols:
        val = st.sidebar.number_input(f'Enter new value for {col}', format="%.4f")
        future_macros_input.append(val)

    # Panel for Outputs
    st.subheader("Data Preview")
    st.dataframe(df.head())

    if st.button('🚀 Generate Forecast'):
        with st.spinner('Training model... This may take a moment.'):
            # Run the forecast
            forecast_value, history, future_date = train_and_forecast(df, target_col, feature_cols, n_past, n_future, future_macros_input)
            
            # Display forecast
            st.success(f"✅ Forecast Complete!")
            st.metric(
                label=f"Forecasted {target_col} for {future_date.strftime('%Y-%m-%d')}",
                value=f"{forecast_value:.4f} (or {forecast_value*100:.2f}%)"
            )

            # Display Plots
            st.subheader("📊 Model Performance")
            
            # Loss Curve
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_title('Model Loss Over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)

            # Forecast vs Actual Plot
            st.subheader("📈 Forecast vs. Actual Data")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(df.index, df[target_col], label='Actual Historical Data')
            # Create a new series for the forecast point
            forecast_series = pd.Series([forecast_value], index=[future_date])
            ax2.plot(forecast_series.index, forecast_series.values, 'ro-', label='Forecasted Value')
            ax2.set_title(f'{target_col} Forecast')
            ax2.set_ylabel(target_col)
            ax2.legend()
            st.pyplot(fig2)

else:
    st.info("Awaiting for a file to be uploaded.")