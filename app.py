import os
import base64
import sqlite3
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from database import init_db, get_all
from inference import process_video, CONF_THRESHOLD

DB_PATH = os.path.expanduser('~/pothole_app/pothole_data.db')

# ── Konfigurasi halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistem Deteksi Lubang Jalan",
    page_icon="🕳️",
    layout="wide",
)

init_db()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🕳️ Sistem Deteksi Lubang Jalan")
st.caption("Deformable DETR")
st.divider()

# ── Tab ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📹 Upload Video", "🗺️ Peta Lokasi"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    rows = get_all()
    if not rows:
        st.info("Belum ada data. Silakan upload dan proses video terlebih dahulu.")
    else:
        df = pd.DataFrame(rows, columns=['ID', 'Latitude', 'Longitude', 'Confidence', 'Image Path'])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Lubang Terdeteksi", len(df))
        col2.metric("Rata-rata Confidence",
                    f"{df['Confidence'].mean():.2%}" if len(df) > 0 else "0%")
        col3.metric("Lokasi Unik", len(df))

        st.subheader("Data Lubang Jalan")
        st.dataframe(
            df[['ID', 'Latitude', 'Longitude', 'Confidence']],
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Upload & Proses Video
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload Video Dashcam")
    st.write("Format: `.mp4` atau `.avi` · Resolusi minimal 720p · Maks 10 menit")

    uploaded = st.file_uploader("Pilih atau seret file video", type=['mp4', 'avi'])
    conf = st.slider("Confidence Threshold", 0.1, 0.9, CONF_THRESHOLD, 0.05,
                     help="Semakin tinggi = lebih ketat, semakin rendah = lebih sensitif")

    if uploaded:
        st.video(uploaded)

        if st.button("🚀 Mulai Deteksi", type="primary"):
            tmp_path = f"/tmp/{uploaded.name}"
            with open(tmp_path, 'wb') as f:
                f.write(uploaded.getbuffer())

            with st.spinner("⏳ Memproses video... (bisa beberapa menit)"):
                try:
                    stats = process_video(tmp_path, conf)
                    st.success("✅ Proses selesai!")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Frame Diproses", stats['total_frames'])
                    col2.metric("Pothole Terdeteksi", stats['detections'])
                    col3.metric("Data Baru", stats['inserted'])
                    col4.metric("GPS Gagal Dibaca", stats['no_gps'])
                    st.info("Buka tab **Peta Lokasi** untuk melihat hasil.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Peta Lokasi
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    rows = get_all()
    if not rows:
        st.info("Belum ada data untuk ditampilkan di peta.")
    else:
        df = pd.DataFrame(rows, columns=['ID', 'Latitude', 'Longitude', 'Confidence', 'Image Path'])

        # Default center: Tangerang
        center_lat = df['Latitude'].mean()
        center_lon = df['Longitude'].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=15,
                       tiles='OpenStreetMap')

        for _, row in df.iterrows():
            popup_html = f"""
            <div style='font-family:Arial; min-width:200px'>
                <b>🕳️ Lubang Jalan #{int(row['ID'])}</b><br>
                <hr style='margin:4px 0'>
                📍 <b>Lat:</b> {row['Latitude']:.6f}<br>
                📍 <b>Lon:</b> {row['Longitude']:.6f}<br>
                🎯 <b>Confidence:</b> {row['Confidence']:.2%}<br>
            </div>
            """
            if row['Image Path'] and os.path.exists(row['Image Path']):
                with open(row['Image Path'], 'rb') as img_f:
                    img_b64 = base64.b64encode(img_f.read()).decode()
                filename = os.path.basename(row['Image Path'])
                static_url = f"http://localhost:8501/app/static/evidence/{filename}"
                popup_html += (
                    f"<a href='{static_url}' target='_blank'>"
                    f"<img src='data:image/jpeg;base64,{img_b64}'"
                    f" width='240'"
                    f" style='margin-top:8px;border-radius:4px;cursor:pointer'"
                    f" title='Klik untuk lihat full'>"
                    f"</a>"
                )
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"Pothole #{int(row['ID'])} ({row['Confidence']:.0%})",
                icon=folium.Icon(color='red', icon='warning-sign', prefix='glyphicon'),
            ).add_to(m)

        st_folium(m, width=None, height=550, use_container_width=True, returned_objects=[])

        st.caption(f"Total {len(df)} lokasi lubang jalan terdeteksi.")
