#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANLI MPPT DEMO — Tamamen PC/Yazılım Ortamında Çalışan Gerçek-Zamanlı Gösterim
================================================================================
Bu uygulama, fiziksel donanım gerektirmeden, MPPT algoritmalarının (P&O ve INC)
GERÇEK ZAMANLI hava verisiyle (Open-Meteo API, ücretsiz/anahtarsız) nasıl
çalıştığını canlı bir web panelinde gösterir. Sim-3/4'te doğrulanan algoritma
mantığı (yön hafızası + sınır-yansıtma + hysteresis + Newton-Raphson IV modeli)
birebir kullanılır — böylece demo, önceki simülasyon bulgularınızla tutarlıdır.

ÇALIŞTIRMA:
  1) pip install streamlit requests numpy pandas plotly pvlib --break-system-packages
     (pvlib zaten kurulu olmalı, önceki simülasyonlardan)
  2) streamlit run canli_mppt_demo.py
  3) Tarayıcıda otomatik açılan sayfada "Canlıyı Başlat" düğmesine basın

NOT: Bu dosya bu ortamda (ağ erişimi olmayan sanal alan) test edilememiştir --
sizin makinenizde (internet erişimi olan) çalıştırıp doğrulamanız gerekir.

ÇOK ÖNEMLİ -- ÇALIŞTIRMA ŞEKLİ:
  Bu bir Streamlit uygulamasıdır, NORMAL bir Python betiği DEĞİLDİR.
  YANLIŞ:  python canli_mppt_demo.py   (veya dosyaya çift tıklamak)
           -> Streamlit'in beklediği "çalışma zamanı ortamı" olmadığı için
              anında hata verip pencere kapanır.
  DOĞRU:   streamlit run canli_mppt_demo.py
           -> Yerel bir web sunucusu başlatır, tarayıcıda otomatik açılır,
              komut penceresi sunucu çalıştığı sürece AÇIK kalır (kapatmak
              için Ctrl+C).
================================================================================
"""

import sys
import traceback

def _kapanmadan_once_hata_goster(exc_type, exc_value, exc_tb):
    print("\n" + "=" * 70)
    print("HATA OLUSTU - program sonlandi. Ayrinti asagida:")
    print("=" * 70)
    traceback.print_exception(exc_type, exc_value, exc_tb)
    print("\nNOT: Bu dosyayi 'python canli_mppt_demo.py' ile degil,")
    print("     'streamlit run canli_mppt_demo.py' ile calistirmalisiniz.")
    try:
        input("\n[Pencereyi kapatmak icin Enter'a basin] ")
    except Exception:
        pass

sys.excepthook = _kapanmadan_once_hata_goster

try:
    import streamlit as st
except ModuleNotFoundError:
    print("\n" + "=" * 70)
    print("HATA: 'streamlit' kütüphanesi kurulu değil.")
    print("Çözüm: pip install streamlit requests numpy pandas plotly")
    print("=" * 70)
    input("\n[Pencereyi kapatmak icin Enter'a basin] ")
    sys.exit(1)

import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pvlib.pvsystem import retrieve_sam, calcparams_cec, singlediode

# =============================================================================
# SABİT SİSTEM/MODÜL PARAMETRELERİ (önceki simülasyonlarla tutarlı)
# =============================================================================
CEC_MODULE_NAME = "A10Green_Technology_A10J_S72_175"
NUM_SERIES_MODULES = 13
HYSTERESIS_FRACTION = 0.001
VAR_STEP_GAIN = 0.5
MAX_V_STEP = 5.0

st.set_page_config(page_title="Canlı MPPT Demo", layout="wide")


@st.cache_resource
def load_module():
    cec_modules = retrieve_sam("CECMod")
    if CEC_MODULE_NAME not in cec_modules.columns:
        candidates = [c for c in cec_modules.columns if "A10" in c]
        st.error(f"Modül bulunamadı. Adaylar: {candidates[:10]}")
        st.stop()
    m = cec_modules[CEC_MODULE_NAME]
    return {
        "I_mp_ref": float(m.get("I_mp_ref")), "V_mp_ref": float(m.get("V_mp_ref")),
        "I_sc_ref": float(m.get("I_sc_ref")), "V_oc_ref": float(m.get("V_oc_ref")),
        "alpha_sc": float(m.get("alpha_sc", 0.0)), "a_ref": float(m.get("a_ref", 1.0)),
        "I_L_ref": float(m.get("I_L_ref", m.get("I_sc_ref"))),
        "I_o_ref": float(m.get("I_o_ref", 1e-10)),
        "R_sh_ref": float(m.get("R_sh_ref", 300.0)), "R_s": float(m.get("R_s", 0.5)),
        "Adjust": float(m.get("Adjust", 8.0)),
    }


@st.cache_data(ttl=600)
def fetch_live_weather(lat, lon):
    """Open-Meteo'dan (ücretsiz, anahtarsız) canlı ışınım (GHI) ve sıcaklık çeker."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=shortwave_radiation,temperature_2m"
        "&hourly=shortwave_radiation,temperature_2m"
        "&forecast_days=1&timezone=auto"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    ghi_now = data["current"]["shortwave_radiation"]
    temp_now = data["current"]["temperature_2m"]
    hourly_ghi = data["hourly"]["shortwave_radiation"]
    hourly_temp = data["hourly"]["temperature_2m"]
    hourly_time = data["hourly"]["time"]
    return ghi_now, temp_now, pd.DataFrame(
        {"time": hourly_time, "G": hourly_ghi, "T": hourly_temp}
    )


# =============================================================================
# 4 AŞAMA TANIMI (Sim-1 -> Sim-4 ile birebir tutarlı)
# =============================================================================
STAGE_CONFIGS = {
    "Sim-1: Referans Durum": {
        "iv_model": "crude", "step_mode": "fixed", "fixed_step": 2.0,
        "aciklama": "Kaba 3-nokta IV modeli, sabit 2.0V adım",
    },
    "Sim-2: Parametrik Optimizasyon": {
        "iv_model": "crude", "step_mode": "fixed", "fixed_step": 1.0,
        "aciklama": "Kaba 3-nokta IV modeli, sabit 1.0V adım (küçültülmüş)",
    },
    "Sim-3: Algoritmik İyileştirme": {
        "iv_model": "newton", "step_mode": "fixed", "fixed_step": 1.0,
        "aciklama": "Newton-Raphson tam IV modeli, sabit 1.0V adım",
    },
    "Sim-4: Dinamik Adım": {
        "iv_model": "newton", "step_mode": "variable", "fixed_step": None,
        "aciklama": "Newton-Raphson tam IV modeli, |dP/dV| ile orantılı değişken adım",
    },
}


def build_iv_curve(mod, poa_irradiance, temp_cell, iv_model="newton", n_points=250):
    """IV eğrisini seçilen aşamaya göre üretir:
       - 'crude'  : Sim-1/2'deki kaba 3-nokta (Isc, Vmp, Voc) doğrusal model
       - 'newton' : Sim-3/4'teki Newton-Raphson tam diyot eğrisi
    """
    diode_params = calcparams_cec(
        max(poa_irradiance, 1e-3), temp_cell, mod["alpha_sc"], mod["a_ref"],
        mod["I_L_ref"], mod["I_o_ref"], mod["R_sh_ref"], mod["R_s"], Adjust=mod["Adjust"],
    )
    sd = singlediode(*diode_params)
    v_oc = max(float(sd["v_oc"]), 0.05)
    v_mp, i_mp = float(sd["v_mp"]), float(sd["i_mp"])

    if iv_model == "crude":
        i_sc = float(sd["i_sc"])
        v_mesh = np.array([0.0, v_mp, v_oc])
        i_mesh = np.array([i_sc, i_mp, 0.0])
    else:
        il, io, rs, rsh, a = diode_params
        v_mesh = np.linspace(0.0, v_oc, n_points)
        i_guess = np.full_like(v_mesh, il)
        for _ in range(20):
            expr = np.minimum((v_mesh + i_guess * rs) / a, 700.0)
            exp_term = np.exp(expr)
            f = il - io * (exp_term - 1) - (v_mesh + i_guess * rs) / rsh - i_guess
            df_val = -io * (rs / a) * exp_term - (rs / rsh) - 1.0
            i_next = i_guess - f / df_val
            if np.max(np.abs(i_next - i_guess)) < 1e-6:
                i_guess = i_next
                break
            i_guess = i_next
        i_mesh = np.maximum(0.0, i_guess)

    p_mesh = v_mesh * i_mesh
    return v_mesh, i_mesh, p_mesh, v_oc, v_mp, i_mp


def interp_i(v_mesh, i_mesh, v_query):
    return float(np.interp(np.clip(v_query, v_mesh[0], v_mesh[-1]), v_mesh, i_mesh))


class MPPTState:
    """Tek bir algoritmanın (P&O veya INC) çalışma zamanı durumu."""
    def __init__(self, v_start_sys):
        self.v_sys = v_start_sys
        self.direction = 1.0
        self.p_prev = None


def po_step(state, v_mesh, i_mesh, vmax_sys, hysteresis_module, step_mode="variable", fixed_step=1.0):
    vmin_sys = 0.0
    v_mod = np.clip(state.v_sys, vmin_sys, vmax_sys) / NUM_SERIES_MODULES

    if step_mode == "variable":
        i_now = interp_i(v_mesh, i_mesh, v_mod)
        eps = max(1e-3, 1e-4 * max(1.0, v_mod))
        dIdV = (interp_i(v_mesh, i_mesh, v_mod + eps) - interp_i(v_mesh, i_mesh, max(v_mod - eps, 0))) / (2 * eps)
        dPdV = i_now + v_mod * dIdV
        step_mod = float(np.clip(abs(dPdV) * VAR_STEP_GAIN, 0.05, MAX_V_STEP))
    else:
        step_mod = fixed_step
    step_sys = step_mod * NUM_SERIES_MODULES

    v_desired = state.v_sys + state.direction * step_sys
    v_trial = float(np.clip(v_desired, vmin_sys, vmax_sys))
    hit_boundary = (v_desired != v_trial)

    v_mod_trial = v_trial / NUM_SERIES_MODULES
    i_trial = interp_i(v_mesh, i_mesh, v_mod_trial)
    p_trial = v_mod_trial * i_trial

    if state.p_prev is None:
        state.p_prev = p_trial
    dP = p_trial - state.p_prev
    if hit_boundary or dP < -hysteresis_module:
        state.direction *= -1.0

    state.v_sys = v_trial
    state.p_prev = p_trial
    return v_mod_trial, i_trial, p_trial


def inc_step(state, v_mesh, i_mesh, vmax_sys, hysteresis_module, step_mode="variable", fixed_step=1.0):
    vmin_sys = 0.0
    v_mod = np.clip(state.v_sys, vmin_sys, vmax_sys) / NUM_SERIES_MODULES
    i_now = interp_i(v_mesh, i_mesh, v_mod)
    eps = max(1e-3, 1e-4 * max(1.0, v_mod))
    i_plus = interp_i(v_mesh, i_mesh, min(v_mod + eps, v_mesh[-1]))
    i_minus = interp_i(v_mesh, i_mesh, max(v_mod - eps, 0))
    dIdV = (i_plus - i_minus) / (2 * eps)
    dPdV = i_now + v_mod * dIdV

    if step_mode == "variable":
        step_mod = min(abs(dPdV) * VAR_STEP_GAIN, MAX_V_STEP)
    else:
        step_mod = fixed_step
    step_sys = step_mod * NUM_SERIES_MODULES

    if abs(dPdV) > hysteresis_module / max(v_mod, 1e-3):
        state.v_sys = float(np.clip(
            state.v_sys + (step_sys if dPdV > 0 else -step_sys), vmin_sys, vmax_sys))
    v_mod_now = state.v_sys / NUM_SERIES_MODULES
    i_final = interp_i(v_mesh, i_mesh, v_mod_now)
    return v_mod_now, i_final, v_mod_now * i_final


# =============================================================================
# SİM-5: DONANIM DURUM MAKİNESİ (ayrı, bağımsız döngü)
# =============================================================================
DUTY_LAG_ALPHA = 0.02  # Sim-5'te doğrulanan kararlı duty-cycle gecikme katsayısı

SYSTEM_COMPONENTS = [
    {"Bileşen": "PV Panel", "Öneri/Değer": "A10Green A10J-S72 175W x 13 seri (veya benzer)",
     "Kısıt/Not": "Voc/Isc panel türüne göre değişir; string Voc'u DC bara başlangıcından düşük olmamalı"},
    {"Bileşen": "Boost Dönüştürücü", "Öneri/Değer": "Bu kodda L=2mH kullanıldı (gerçek donanımda tipik olarak L≈100-500µH tercih edilir)",
     "Kısıt/Not": "Küçük endüktans → yüksek akım dalgalanması; büyük endüktans → yavaş tepki. Model, ayrık anahtarlama frekansını değil, ORTALANMIŞ (averaged) bir devre davranışını temsil eder"},
    {"Bileşen": "MOSFET + Sürücü", "Öneri/Değer": "Gerçek donanımda ör. IRF540N + IR2110 (bu demoda ayrıca modellenmedi)",
     "Kısıt/Not": "Anahtarlama kayıpları, ölü zaman (dead-time) gerçek donanımda ek verim kaybı yaratır; bu basitleştirilmiş modelde yok"},
    {"Bileşen": "Akım Sensörü", "Öneri/Değer": "Gerçek donanımda ör. ACS712 (Hall-effect)",
     "Kısıt/Not": "Bu demoda ±0.02A (20mA) çözünürlüklü ADC gürültüsü olarak modellendi"},
    {"Bileşen": "Gerilim Ölçümü", "Öneri/Değer": "Dirençli bölücü + op-amp buffer",
     "Kısıt/Not": "Bu demoda ±0.05V çözünürlüklü ADC gürültüsü olarak modellendi"},
    {"Bileşen": "Mikrodenetleyici", "Öneri/Değer": "Gerçek donanımda ör. STM32F4/F7 veya TI C2000 (bu demoda spesifik bir MCU modellenmedi)",
     "Kısıt/Not": "Örnekleme hızı ve PWM çözünürlüğü gerçek zamanlı performansı belirler"},
    {"Bileşen": "DC Bara Kapasitesi", "Öneri/Değer": "Bu kodda C=500µF kullanıldı",
     "Kısıt/Not": "Küçük kapasite → gerilim dalgalanması; büyük kapasite → yavaş geçiş tepkisi"},
    {"Bileşen": "Yük / İnvertör", "Öneri/Değer": "Şebeke-bağlı invertör (aktif gerilim regülasyonlu)",
     "Kısıt/Not": "Bu demoda invertör, DC-barayı sabit bir hedef gerilimde tutan aktif bir kontrolcü olarak modellendi (Model A)"},
]

KNOWN_CONSTRAINTS = [
    "ADC ölçüm gürültüsü (gerilim ±0.05V, akım ±0.02A çözünürlük varsayıldı)",
    "Duty-cycle kontrol döngüsünün fiziksel gecikmesi (anlık değil, yarı-statik hedefe kademeli yaklaşma)",
    "Endüktör/kapasitör geçiş dinamiği (ortalanmış devre modeli, ayrık anahtarlama değil)",
    "Gerçek donanımda PSO gibi hesaplama-ağır algoritmaların mikrodenetleyicide çalıştırılması pratik değildir (bu demoda ayrıca test edilmedi)",
    "DC bara başlangıç geriliminin panel Voc'undan güvenli şekilde yüksek olması gerekliliği (boost topolojisi kısıtı)",
    "Görev döngüsü (duty cycle) fiziksel olarak %95 ile sınırlandı (gerçek anahtarlamalı kaynaklarda tam %100 doygunluk mümkün değildir)",
]


KP_LOAD = 0.2  # A/V -- invertörün DC-barayı hedef gerilimde tutmak için ne kadar
               # agresif akım çektiğini belirleyen oransal kazanç


class DCDC_Boost_StateMachine:
    """Sim-5'te doğrulanan, kararlı (duty-lag kontrollü) boost dönüştürücü modeli.

    YÜK MODELİ A (aktif gerilim regülasyonu): Eskiden DC-bara, SABİT bir direnç
    (R_load) üzerinden pasif olarak deşarj oluyordu -- bu, ışınım düştükçe
    DC-bara geriliminin de orantılı olarak düşmesine yol açıyordu (V=√(P×R)),
    düşük ışınımda gerilimin panelin işletim noktasının (Vmp) altına inme riski
    taşıyordu. Artık DC-bara, gerçek bir şebeke-bağlı İNVERTÖRÜN yaptığı gibi,
    AKTİF bir gerilim regülasyon döngüsüyle sabit bir HEDEF gerilimde tutulmaya
    çalışılıyor -- invertör, bara hedefin üzerine çıktıkça daha fazla akım
    çekiyor (KP_LOAD ile orantılı), hedefin altındaysa hiç çekmiyor (güç geri
    besleyemez). Bu, ışınım ne olursa olsun DC-baranın hedefe yakın kalmasını
    sağlar -- gerçek invertörlerin davranışına çok daha sadık bir model.
    """
    def __init__(self, v_pv_init, v_dc_init, v_dc_target):
        self.L = 2e-3
        self.C_in = 20e-6
        self.C_out = 500e-6
        self.v_dc_target = v_dc_target
        self.adc_v_res = 0.05
        self.adc_i_res = 0.02
        self.v_pv = v_pv_init
        self.i_L = 0.5
        self.v_dc = v_dc_init
        self.duty = float(np.clip(1.0 - v_pv_init / max(v_dc_init, 1.0), 0.0, 0.95))
        self.dt = 1e-4

    def apply_sensor_noise(self, true_value, resolution):
        noise = np.random.normal(0, resolution / 3.0)
        measured = true_value + noise
        return np.round(measured / resolution) * resolution

    def run_micro_step(self, v_ref, i_pv_of_v):
        duty_target = np.clip(1.0 - v_ref / max(self.v_dc, 1.0), 0.0, 0.95)
        self.duty += (duty_target - self.duty) * DUTY_LAG_ALPHA
        self.duty = float(np.clip(self.duty, 0.0, 0.95))

        i_pv_actual = float(i_pv_of_v(self.v_pv))
        dv_pv_dt = (i_pv_actual - self.i_L) / self.C_in
        di_L_dt = (self.v_pv - (1.0 - self.duty) * self.v_dc) / self.L

        # AKTİF YÜK (Model A): invertör, DC-barayı hedef gerilimde tutmak için
        # çektiği akımı gerilim hatasıyla orantılı ayarlar. Hedefin altındayken
        # hiç akım çekmez (güç geri besleyemez) -- bu yüzden düşük ışınımda bile
        # bara, pasif direnç modelindeki gibi orantılı olarak çökmez.
        i_load = max(0.0, KP_LOAD * (self.v_dc - self.v_dc_target))
        dv_dc_dt = ((1.0 - self.duty) * self.i_L - i_load) / self.C_out

        self.v_pv = max(0.0, self.v_pv + dv_pv_dt * self.dt)
        self.i_L = max(0.0, self.i_L + di_L_dt * self.dt)
        self.v_dc = max(1.0, self.v_dc + dv_dc_dt * self.dt)

    def measure(self):
        v_meas = self.apply_sensor_noise(self.v_pv, self.adc_v_res)
        i_meas = self.apply_sensor_noise(float(self.i_L), self.adc_i_res)
        return v_meas, i_meas, self.v_dc

    def run_micro_steps_and_measure(self, v_ref, i_pv_of_v, n_steps, n_avg=10):
        n_avg = min(n_avg, n_steps)
        v_s, i_s, vdc_s = [], [], []
        for k in range(n_steps):
            self.run_micro_step(v_ref, i_pv_of_v)
            if k >= n_steps - n_avg:
                v_m, i_m, vdc_m = self.measure()
                v_s.append(v_m); i_s.append(i_m); vdc_s.append(vdc_m)
        return float(np.mean(v_s)), float(np.mean(i_s)), float(np.mean(vdc_s))


# =============================================================================
# ARAYÜZ
# =============================================================================
st.title("Canlı MPPT Demo — P&O vs INC")
st.caption(
    "Sim-3/4'te doğrulanan algoritma mantığı, Open-Meteo'dan çekilen canlı "
    "ışınım (G) ve sıcaklık (T) verisiyle gerçek zamanlı çalıştırılır."
)
st.info(
    "Bu demo, tek bir sabit G/T koşulunda algoritmanın MPP'ye yakınsama "
    "yeteneğini gösterir. Yıllık (4282 saatlik, dinamik koşuldaki) "
    "sonuçlarla birebir kıyaslanmamalıdır."
)

tab1, tab2 = st.tabs([
    "🔬 Algoritma Karşılaştırması (Sim-1 → Sim-4)",
    "⚙️ Donanım Durum Makinesi (Sim-5)",
])

with st.sidebar:
    st.header("Ayarlar")
    st.subheader("Simülasyon aşaması")
    stage_choice = st.selectbox(
        "Hangi aşamanın algoritma mantığı çalıştırılsın?",
        list(STAGE_CONFIGS.keys()),
        index=3,  # varsayılan: Sim-4 (en gelişmiş)
    )
    st.caption(f"📋 {STAGE_CONFIGS[stage_choice]['aciklama']}")
    st.markdown("---")
    lat = st.number_input("Enlem", value=38.4237, format="%.4f")
    lon = st.number_input("Boylam", value=27.1428, format="%.4f")
    update_interval = st.slider("Güncelleme aralığı (sn)", 1, 10, 2)
    n_iterations = st.slider("Toplam adım sayısı", 20, 500, 100)
    st.markdown("---")
    st.subheader("Veri kaynağı")
    data_source = st.radio(
        "G/T verisi nereden alınsın?",
        ["Canlı API verisi", "Manuel değerler"],
        index=1,
    )
    manual_g, manual_t = None, None
    stc_mode = False
    if data_source.startswith("Manuel"):
        manual_g = st.slider("Işınım G (W/m²)", 0, 1200, 1000)
        manual_t = st.slider("Sıcaklık (°C)", -10, 50, 25)
        stc_mode = st.checkbox(
            "Bu değeri doğrudan HÜCRE sıcaklığı olarak kullan (STC/laboratuvar modu)",
            value=True,
            help="İşaretliyse T, ışınıma bağlı ek ısınma payı OLMADAN doğrudan "
                 "hücre sıcaklığı olarak kullanılır -- G=1000, T=25 ile gerçek "
                 "STC (Standart Test Koşulları) elde edilir. İşaretli değilse T "
                 "ortam/hava sıcaklığı sayılır ve üzerine ışınıma bağlı bir "
                 "ısınma payı eklenir (gerçekçi saha koşulu yaklaşıklaması).",
        )
    st.markdown("---")
    st.subheader("Senaryo enjeksiyonu")
    inject_cloud = st.checkbox("Bulut geçişi simüle et (ışınımı periyodik düşür)")
    inject_shading = st.checkbox("Kısmi gölgeleme simüle et (ışınımı %40'a sabitle)")
    run_button = st.button("▶ Canlıyı Başlat", type="primary")


mod = load_module()

with tab1:
    st.caption(
        "Fotovoltaik sistemlerde sistemin performansının, değişken çevresel "
        "koşullarda MPPT algoritmaları ile aşamalı analizini içerir."
    )
    if run_button:
        if data_source.startswith("Manuel"):
            ghi_now, temp_now = float(manual_g), float(manual_t)
            st.success(f"Manuel değerler kullanılıyor: G={ghi_now:.1f} W/m², T={temp_now:.1f} °C")
        else:
            try:
                ghi_now, temp_now, hourly_df = fetch_live_weather(lat, lon)
            except Exception as e:
                st.error(f"Hava verisi çekilemedi: {e}")
                st.stop()
            st.success(f"Canlı veri alındı: G={ghi_now:.1f} W/m², T={temp_now:.1f} °C")
            if ghi_now < 5.0:
                st.warning(
                    "Işınım neredeyse sıfır (muhtemelen şu an gece/alacakaranlık). "
                    "Anlamlı bir güç eğrisi görmek için sol menüden 'Manuel değerler' "
                    "seçeneğine geçebilirsiniz."
                )

        po_state = MPPTState(v_start_sys=mod["V_oc_ref"] * 0.8 * NUM_SERIES_MODULES)
        inc_state = MPPTState(v_start_sys=mod["V_oc_ref"] * 0.8 * NUM_SERIES_MODULES)
        module_p_stc = mod["I_mp_ref"] * mod["V_mp_ref"]
        hysteresis_module = HYSTERESIS_FRACTION * module_p_stc

        history = {"t": [], "P_PO": [], "P_IC": [], "P_TRUE": [], "G": []}

        chart_placeholder = st.empty()
        metric_placeholder = st.empty()
        iv_placeholder = st.empty()

        for k in range(n_iterations):
            g_effective = ghi_now
            if inject_shading:
                g_effective = ghi_now * 0.4
            elif inject_cloud:
                g_effective = ghi_now * (0.3 + 0.7 * abs(np.sin(k / 8.0)))

            if stc_mode:
                # STC/laboratuvar modu: T doğrudan hücre sıcaklığı olarak kullanılır,
                # ışınıma bağlı ek ısınma payı UYGULANMAZ (gerçek STC tanımına uygun).
                temp_cell = temp_now
            else:
                # Saha yaklaşıklaması: T ortam sıcaklığı sayılır, ışınıma bağlı
                # ek ısınma payı eklenir (Sim-5'teki basit NOCT benzeri model).
                temp_cell = temp_now + g_effective * 0.03
            stage_cfg = STAGE_CONFIGS[stage_choice]
            v_mesh, i_mesh, p_mesh, v_oc, v_mp, i_mp = build_iv_curve(
                mod, g_effective, temp_cell, iv_model=stage_cfg["iv_model"])
            vmax_sys = v_oc * NUM_SERIES_MODULES
            p_true_module = v_mp * i_mp

            _, _, p_po_module = po_step(po_state, v_mesh, i_mesh, vmax_sys, hysteresis_module,
                                         step_mode=stage_cfg["step_mode"],
                                         fixed_step=stage_cfg["fixed_step"] or 1.0)
            _, _, p_ic_module = inc_step(inc_state, v_mesh, i_mesh, vmax_sys, hysteresis_module,
                                          step_mode=stage_cfg["step_mode"],
                                          fixed_step=stage_cfg["fixed_step"] or 1.0)

            history["t"].append(k)
            history["P_PO"].append(p_po_module * NUM_SERIES_MODULES)
            history["P_IC"].append(p_ic_module * NUM_SERIES_MODULES)
            history["P_TRUE"].append(p_true_module * NUM_SERIES_MODULES)
            history["G"].append(g_effective)

            with chart_placeholder.container():
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history["P_PO"], name="P&O", line=dict(color="#1f77b4")))
                fig.add_trace(go.Scatter(y=history["P_IC"], name="INC", line=dict(color="#ff7f0e")))
                fig.add_trace(go.Scatter(y=history["P_TRUE"], name="P_TRUE (ideal)",
                                          line=dict(color="black", dash="dash")))
                fig.update_layout(title=f"Canlı güç üretimi (dizi seviyesi, W) — {stage_choice}",
                                   xaxis_title="Adım", yaxis_title="Güç (W)", height=380)
                st.plotly_chart(fig, use_container_width=True, key=f"power_chart_{k}")

            with metric_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Anlık G (W/m²)", f"{g_effective:.0f}")
                c2.metric("P&O verimlilik", f"{100*p_po_module/max(p_true_module,1e-6):.1f}%")
                c3.metric("INC verimlilik", f"{100*p_ic_module/max(p_true_module,1e-6):.1f}%")
                c4.metric("Hücre sıcaklığı", f"{temp_cell:.1f} °C")

            with iv_placeholder.container():
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=v_mesh, y=p_mesh, name="P-V eğrisi", line=dict(color="#888")))
                fig2.add_trace(go.Scatter(x=[po_state.v_sys/NUM_SERIES_MODULES], y=[p_po_module],
                                           mode="markers", name="P&O konumu",
                                           marker=dict(color="#1f77b4", size=12)))
                fig2.add_trace(go.Scatter(x=[inc_state.v_sys/NUM_SERIES_MODULES], y=[p_ic_module],
                                           mode="markers", name="INC konumu",
                                           marker=dict(color="#ff7f0e", size=12)))
                fig2.update_layout(title="Anlık P-V eğrisi ve algoritma konumları (modül seviyesi)",
                                    xaxis_title="Gerilim (V)", yaxis_title="Güç (W)", height=380)
                st.plotly_chart(fig2, use_container_width=True, key=f"iv_chart_{k}")

            time.sleep(update_interval)

        st.info("Demo tamamlandı. Parametreleri değiştirip tekrar başlatabilirsiniz.")
    else:
        st.info("Sol menüden ayarları yapıp **Canlıyı Başlat** düğmesine basın.")


with tab2:
    st.header("Sim-5: Donanım Durum Makinesi")
    st.caption(
        "MPPT algoritmalarının, gerçekçi donanım fiziği ve ölçüm gürültüsü "
        "altındaki performansını simüle eder."
    )
    st.caption(
        "Bu sekme Sim-1→4'ten TAMAMEN BAĞIMSIZ çalışır. Burada P&O/INC kararları "
        "artık anlık değil, gerçek bir yükseltici (boost) dönüştürücünün "
        "endüktör/kapasitör diferansiyel denklemleri ve ADC ölçüm gürültüsü "
        "üzerinden fiziksel olarak simüle edilir."
    )

    st.subheader("Sistem Bileşenleri ve Gereksinimleri")
    st.table(pd.DataFrame(SYSTEM_COMPONENTS))
    st.caption("MOSFET/Mikrodenetleyici/Akım Sensörü fiziksel donanım için geçerli olup, demoda ayrıca modellenmedi.")

    st.subheader("Bilinen Kısıtlar")
    for c in KNOWN_CONSTRAINTS:
        st.markdown(f"- {c}")

    st.markdown("---")
    st.subheader("Canlı Donanım Simülasyonu")

    st.subheader("Veri kaynağı")
    data_source5 = st.radio(
        "G/T verisi nereden alınsın?", ["Canlı API verisi", "Manuel değerler"],
        index=1, key="datasrc5",
    )
    if data_source5.startswith("Manuel"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            g5 = st.slider("Işınım G (W/m²)", 0, 1200, 800, key="g5")
        with col_b:
            t5 = st.slider("Hücre sıcaklığı (°C)", -10, 50, 25, key="t5")
        with col_c:
            n5_steps = st.slider("Karar adımı sayısı", 10, 150, 40, key="n5steps")
        st.caption("💡 Tam STC (Standart Test Koşulları) için G=1000, T=25 girin. "
                   "Manuel modda T her zaman doğrudan hücre sıcaklığı olarak kullanılır.")
    else:
        n5_steps = st.slider("Karar adımı sayısı", 10, 150, 40, key="n5steps_api")

    n5_micro = st.slider(
        "Karar adımı başına mikro-adım (fiziksel çözünürlük)", 10, 100, 50, key="n5micro",
        help="Her MPPT kararı arasında devrenin fiziksel olarak ayarlanması için "
             "kaç adet dt=1e-4s'lik mikro-adım koşturulacağı. Sim-5'teki gibi 50 önerilir.",
    )
    run5 = st.button("⚙️ Donanım Simülasyonunu Başlat", type="primary", key="run5")

    if run5:
        if data_source5.startswith("Manuel"):
            g5_use, t5_use = float(g5), float(t5)
            stc_style = True  # manuel modda T doğrudan hücre sıcaklığı sayılır
        else:
            try:
                g5_use, t5_use, _ = fetch_live_weather(lat, lon)
            except Exception as e:
                st.error(f"Hava verisi çekilemedi: {e}")
                st.stop()
            st.success(f"Canlı veri alındı: G={g5_use:.1f} W/m², T={t5_use:.1f} °C")
            stc_style = False  # API modunda T ortam sıcaklığı sayılır, ısınma payı eklenir
            if g5_use < 5.0:
                st.warning("Işınım neredeyse sıfır (gece). Anlamlı sonuç için 'Manuel değerler' kullanın.")

        mod5 = load_module()
        module_p_stc5 = mod5["I_mp_ref"] * mod5["V_mp_ref"]
        hysteresis5 = HYSTERESIS_FRACTION * module_p_stc5

        # DC bara: string Voc'unun güvenli şekilde üzerinde başlar (Sim-5'teki kritik düzeltme)
        # ve İNVERTÖR bu hedefi SABİT tutmaya çalışır (Model A -- gerçek şebeke-bağlı
        # invertörlerin davranışı: hedef, tasarım aşamasında bir kez belirlenir,
        # anlık ışınıma göre DEĞİŞMEZ).
        string_voc5 = mod5["V_oc_ref"] * NUM_SERIES_MODULES
        v_dc_nominal = string_voc5 * 1.15
        st.info(f"String Voc: {string_voc5:.1f} V  →  DC bara HEDEF gerilimi: {v_dc_nominal:.1f} V "
                f"(Voc'un %115'i, invertör bu hedefi ışınımdan bağımsız olarak korumaya çalışır)")

        v_start_sys = mod5["V_oc_ref"] * 0.8 * NUM_SERIES_MODULES
        state_po5 = DCDC_Boost_StateMachine(v_start_sys, v_dc_nominal, v_dc_nominal)
        state_ic5 = DCDC_Boost_StateMachine(v_start_sys, v_dc_nominal, v_dc_nominal)
        v_ref_po5, v_ref_ic5 = v_start_sys, v_start_sys
        direction_po5 = 1.0
        p_prev_po5 = None

        hist5 = {"t": [], "P_PO": [], "P_IC": [], "P_TRUE": [], "VDC_PO": [], "VDC_IC": []}
        chart5_ph = st.empty()
        metric5_ph = st.empty()
        vdc5_ph = st.empty()

        for k in range(n5_steps):
            if stc_style:
                temp_cell5 = t5_use  # STC/laboratuvar tarzı: doğrudan hücre sıcaklığı
            else:
                temp_cell5 = t5_use + g5_use * 0.03  # saha yaklaşıklaması: ortam + ışınıma bağlı ısınma
            v_mesh5, i_mesh5, p_mesh5, v_oc5, v_mp5, i_mp5 = build_iv_curve(
                mod5, g5_use, temp_cell5, iv_model="newton")
            vmax_sys5 = v_oc5 * NUM_SERIES_MODULES
            p_true5 = v_mp5 * i_mp5

            def i_pv_of_v_sys5(v_sys, _vmesh=v_mesh5, _imesh=i_mesh5):
                v_mod = np.clip(v_sys, 0.0, vmax_sys5) / NUM_SERIES_MODULES
                return interp_i(_vmesh, _imesh, v_mod)

            # --- P&O (yön hafızalı + sınır-yansıtma + hysteresis, gerçek devre üzerinden) ---
            v_mod_po5 = np.clip(v_ref_po5, 0.0, vmax_sys5) / NUM_SERIES_MODULES
            i_now_po5 = interp_i(v_mesh5, i_mesh5, v_mod_po5)
            eps5 = max(1e-3, 1e-4 * max(1.0, v_mod_po5))
            dIdV5 = (interp_i(v_mesh5, i_mesh5, v_mod_po5 + eps5) -
                     interp_i(v_mesh5, i_mesh5, max(v_mod_po5 - eps5, 0))) / (2 * eps5)
            dPdV5 = i_now_po5 + v_mod_po5 * dIdV5
            step_po5 = float(np.clip(abs(dPdV5) * VAR_STEP_GAIN, 0.05, MAX_V_STEP)) * NUM_SERIES_MODULES

            v_desired5 = v_ref_po5 + direction_po5 * step_po5
            v_ref_po5 = float(np.clip(v_desired5, 0.0, vmax_sys5))
            hit_boundary5 = (v_desired5 != v_ref_po5)

            v_meas_po5, i_meas_po5, vdc_po5 = state_po5.run_micro_steps_and_measure(
                v_ref_po5, i_pv_of_v_sys5, n5_micro, n_avg=10)
            # DÜZELTME: Güç metriği artık ENDÜKTÖR AKIMI (i_meas_po5) yerine,
            # ölçülen gerilimdeki GERÇEK PANEL AKIMINDAN (IV eğrisinden doğrudan
            # okunur) hesaplanıyor. i_meas_po5, giriş kapasitesinin geçici enerji
            # alışverişi nedeniyle panel akımından sapabiliyordu -- bu da P_TRUE'yu
            # yapay olarak aşan (>%100) ölçümlere yol açıyordu. Artık güç, tanım
            # gereği IV eğrisinin üzerinde kaldığı için P_TRUE'yu aşamaz.
            i_panel_at_vmeas_po5 = i_pv_of_v_sys5(v_meas_po5)
            p_meas_po5 = max(0.0, (v_meas_po5 / NUM_SERIES_MODULES) * i_panel_at_vmeas_po5)
            if p_prev_po5 is None:
                p_prev_po5 = p_meas_po5
            if hit_boundary5 or (p_meas_po5 - p_prev_po5) < -hysteresis5:
                direction_po5 *= -1.0
            p_prev_po5 = p_meas_po5

            # --- INC (değişken adım + hysteresis, gerçek devre üzerinden) ---
            v_mod_ic5 = np.clip(v_ref_ic5, 0.0, vmax_sys5) / NUM_SERIES_MODULES
            i_now_ic5 = interp_i(v_mesh5, i_mesh5, v_mod_ic5)
            i_plus5 = interp_i(v_mesh5, i_mesh5, min(v_mod_ic5 + eps5, v_mesh5[-1]))
            i_minus5 = interp_i(v_mesh5, i_mesh5, max(v_mod_ic5 - eps5, 0))
            dIdV_ic5 = (i_plus5 - i_minus5) / (2 * eps5)
            dPdV_ic5 = i_now_ic5 + v_mod_ic5 * dIdV_ic5
            step_ic5 = min(abs(dPdV_ic5) * VAR_STEP_GAIN, MAX_V_STEP) * NUM_SERIES_MODULES
            if abs(dPdV_ic5) > hysteresis5 / max(v_mod_ic5, 1e-3):
                v_ref_ic5 = float(np.clip(
                    v_ref_ic5 + (step_ic5 if dPdV_ic5 > 0 else -step_ic5), 0.0, vmax_sys5))

            v_meas_ic5, i_meas_ic5, vdc_ic5 = state_ic5.run_micro_steps_and_measure(
                v_ref_ic5, i_pv_of_v_sys5, n5_micro, n_avg=10)
            # Aynı düzeltme INC için de uygulanıyor.
            i_panel_at_vmeas_ic5 = i_pv_of_v_sys5(v_meas_ic5)
            p_meas_ic5 = max(0.0, (v_meas_ic5 / NUM_SERIES_MODULES) * i_panel_at_vmeas_ic5)

            hist5["t"].append(k)
            hist5["P_PO"].append(p_meas_po5 * NUM_SERIES_MODULES)
            hist5["P_IC"].append(p_meas_ic5 * NUM_SERIES_MODULES)
            hist5["P_TRUE"].append(p_true5 * NUM_SERIES_MODULES)
            hist5["VDC_PO"].append(vdc_po5)
            hist5["VDC_IC"].append(vdc_ic5)

            with chart5_ph.container():
                f5 = go.Figure()
                f5.add_trace(go.Scatter(y=hist5["P_PO"], name="P&O (donanım)", line=dict(color="#1f77b4")))
                f5.add_trace(go.Scatter(y=hist5["P_IC"], name="INC (donanım)", line=dict(color="#ff7f0e")))
                f5.add_trace(go.Scatter(y=hist5["P_TRUE"], name="P_TRUE (ideal)",
                                         line=dict(color="black", dash="dash")))
                f5.update_layout(title="Sim-5: Donanım-gerçekçi güç üretimi (dizi seviyesi, W)",
                                  xaxis_title="Karar adımı", yaxis_title="Güç (W)", height=380)
                st.plotly_chart(f5, use_container_width=True, key=f"hw_power_{k}")

            v_mp_string5 = v_mp5 * NUM_SERIES_MODULES  # o anki koşulda GERÇEK, CANLI hesaplanan Vmp

            with metric5_ph.container():
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("P&O verimlilik", f"{100*p_meas_po5/max(p_true5,1e-6):.1f}%")
                m2.metric("INC verimlilik", f"{100*p_meas_ic5/max(p_true5,1e-6):.1f}%")
                m3.metric("DC Bara (P&O)", f"{vdc_po5:.1f} V")
                m4.metric("DC Bara (INC)", f"{vdc_ic5:.1f} V")
                m5.metric("Panel Vmp (canlı, o anki G/T)", f"{v_mp_string5:.1f} V")

            if vdc_po5 < v_mp_string5 or vdc_ic5 < v_mp_string5:
                st.error(
                    f"⚠️ DC-bara ({min(vdc_po5, vdc_ic5):.1f} V), panelin o anki gerçek "
                    f"işletim gerilimini ({v_mp_string5:.1f} V) altına indi — bu, boost "
                    f"topolojisi için fiziksel olarak geçersiz bir durumdur."
                )

            with vdc5_ph.container():
                f6 = go.Figure()
                f6.add_trace(go.Scatter(y=hist5["VDC_PO"], name="DC Bara — P&O", line=dict(color="#2ca02c")))
                f6.add_trace(go.Scatter(y=hist5["VDC_IC"], name="DC Bara — INC", line=dict(color="#d62728")))
                f6.update_layout(title="İnvertör Girişi DC-Link Gerilimi (V)",
                                  xaxis_title="Karar adımı", yaxis_title="Gerilim (V)", height=300)
                st.plotly_chart(f6, use_container_width=True, key=f"hw_vdc_{k}")

            time.sleep(max(0.1, update_interval / 4.0))

        st.info("Donanım simülasyonu tamamlandı.")
    else:
        st.info("Yukarıdan G/T ve adım sayısını ayarlayıp **Donanım Simülasyonunu Başlat** düğmesine basın.")