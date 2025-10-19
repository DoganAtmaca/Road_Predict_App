import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Modeli yükleme
try:
    model = pickle.load(open("road_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Hata: 'road_model.pkl' dosyası bulunamadı.")
    st.stop()

# Datayı yükleme
try:
    data = pd.read_csv("train.csv")
except FileNotFoundError:
    st.error("Hata: 'train.csv' dosyası bulunamadı.")
    st.stop()

# 1. Feature Engineering (Veri dönüşümleri)
data['Speed_limit_flag'] = (data['speed_limit'] > 50).astype(int)
data['curvature_limit_flag'] = (data['curvature'] > 0.5).astype(int)
data['num_reported_accidents'] = data['num_reported_accidents']**3
data['accident_num_lanes'] = data['num_reported_accidents'] * data['num_lanes']
data['curvature_speed'] = data['curvature'] * data['speed_limit']
data['curvature_accidents'] = data['curvature'] * data['num_reported_accidents']
sözlük = {True:1,
         False:0}
data['road_signs_present'] = data['road_signs_present'].map(sözlük)
data['public_road'] = data['public_road'].map(sözlük)
data['holiday'] = data['holiday'].map(sözlük)
data['school_season'] = data['school_season'].map(sözlük)


sample_data = data.drop(['id', 'accident_risk'], axis=1)


sample_data_dum = pd.get_dummies(sample_data)
all_columns = sample_data_dum.columns



scaler = MinMaxScaler()
scaler.fit(sample_data_dum)


# 3. Streamlit Uygulama Düzeni ve Session State
st.set_page_config(page_title="Road Risk Comparison", layout="wide")
st.title("🚦 Which way is more risky?")

# Session State ile yolların kalıcı olmasını sağla
if 'road1' not in st.session_state:
    st.session_state.sample = sample_data.sample(2).reset_index(drop=True)
    st.session_state.road1 = st.session_state.sample.iloc[0]
    st.session_state.road2 = st.session_state.sample.iloc[1]

road1, road2 = st.session_state.road1, st.session_state.road2
show_columns = ['num_lanes', 'road_type', 'curvature', 'lighting', 'public_road', 'weather']

col1, col2 = st.columns(2)
with col1:
    st.subheader("🛣️ Road 1")
    st.table(road1[show_columns].to_frame()) # .to_frame() ekledik, Series'in başlıkları kaybolmasın diye
with col2:
    st.subheader("🛣️ Road 2")
    st.table(road2[show_columns].to_frame())

left, right = st.columns(2)
with left:
    choose1 = st.button("Road 1 is riskier", key="btn1")
with right:
    choose2 = st.button("Road 2 is riskier", key="btn2")


# 4. Tahmin ve Sonuç Hesaplama Fonksiyonu
def get_prediction_data(road_series, all_columns, scaler, model):
    # Tek satırlık Series'i DataFrame'e dönüştür
    road_df = pd.DataFrame([road_series])

    # Dummy değişkenleri oluştur (Eğitimdeki parametrelerin aynısı kullanılmalı)
    road_dum = pd.get_dummies(road_df)

    # KRİTİK DÜZELTME: Sütunları eğitim verisi sütunları ile hizala
    road_dum_aligned = road_dum.reindex(columns=all_columns, fill_value=0)
    
    # NaN değerlerini 0 ile doldur ve float tipine zorla
    road_dum_aligned = road_dum_aligned.fillna(0).astype(float)

    # Ölçeklendirmeyi uygula (numpy array döner)
    road_scaled = scaler.transform(road_dum_aligned)
    
    # Modelin beklediği DataFrame formatına geri dön
    X_pred = pd.DataFrame(road_scaled, columns=all_columns)
    
    # Tahmini yap 
    prediction = model.predict(X_pred)[0]
    return prediction


# 5. Butonlara basınca tahmin yapalım
if choose1 or choose2:
    try:
        # 1️⃣ Her iki yol için model tahmini yap
        road1_pred = get_prediction_data(st.session_state.road1, all_columns, scaler, model)
        road2_pred = get_prediction_data(st.session_state.road2, all_columns, scaler, model)
        
        # 2️⃣ Daha riskli yolu belirle
        correct_choice = "road1" if road1_pred > road2_pred else "road2"

        # Tahmin skorlarını göster
        st.markdown(f"---")
        st.markdown(f"**Model Prediction Score:** Road 1: `{road1_pred:.4f}` | Road 2: `{road2_pred:.4f}`")

        # 3️⃣ Sonuç
        user_chose_correctly = (choose1 and correct_choice == "road1") or (choose2 and correct_choice == "road2")
        
        if user_chose_correctly:
            st.success(f"✅ Doğru! {'Road 1' if correct_choice == 'road1' else 'Road 2'} daha riskli.")
            st.balloons()
        else:
            st.error(f"❌ Wrong Choice! Correct Answer: {'Road 1' if correct_choice == 'road1' else 'Road 2'}. Tekrar deneyin.")
        
    except Exception as e:
        st.error(f"Tahmin sırasında beklenmedik bir hata oluştu: {e}")
        st.warning("Lütfen `road_model.pkl` dosyasının eğitimindeki veri ön işleme adımlarının (özellikle dummy değişken parametrelerinin) bu kod ile tam olarak aynı olduğunu kontrol edin.")

# Yeni yollar seçeneği
# Yeni yollar butonu callback
def new_roads():
    sample = sample_data.sample(2).reset_index(drop=True)
    st.session_state.road1 = sample.iloc[0]
    st.session_state.road2 = sample.iloc[1]

# Buton tıklandığında state güncellenecek, sayfa otomatik güncellenecek
st.button("Comparing New Roads", on_click=new_roads)
