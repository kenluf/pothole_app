# 🕳️ Sistem Deteksi Lubang Jalan

Sistem deteksi lubang jalan otomatis menggunakan Deformable DETR dari video dashcam dengan ekstraksi GPS real-time.

## 🚀 Fitur

- **Deteksi Otomatis**: Menggunakan Deformable DETR untuk mendeteksi lubang jalan
- **GPS Integration**: Ekstraksi koordinat GPS dari overlay kamera dashcam
- **Deduplikasi**: Sistem Haversine untuk menghindari duplikasi data berdasarkan jarak
- **Dashboard Interaktif**: Visualisasi data dengan Streamlit
- **Peta Lokasi**: Integrasi Folium untuk menampilkan lokasi lubang jalan
- **Evidence Storage**: Penyimpanan gambar bukti deteksi

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Model**: Deformable DETR (PyTorch)
- **OCR**: EasyOCR untuk ekstraksi GPS
- **Database**: SQLite
- **Mapping**: Folium + Streamlit-Folium
- **Computer Vision**: OpenCV

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended untuk inference)
- Model checkpoint Deformable DETR

## 🚀 Quick Start

1. **Clone repository**:
   ```bash
   git clone https://github.com/your-username/pothole-detection.git
   cd pothole-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup model checkpoint**:
   - Download checkpoint Deformable DETR
   - Place di path yang sesuai dengan `inference.py`

4. **Run aplikasi**:
   ```bash
   streamlit run app.py
   ```

## 📁 Struktur Project

```
pothole_app/
├── app.py                 # Main Streamlit application
├── database.py            # SQLite database operations
├── inference.py           # AI inference pipeline
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── static/               # Static files (optional)
└── evidence/             # Generated evidence images (runtime)
```

## 🔧 Configuration

### Model Settings
- **Confidence Threshold**: Default 0.4 (dapat diubah di UI)
- **Frame Sampling**: 1 frame per detik
- **GPS ROI**: Pojok kanan bawah video

### Database
- **Deduplication**: 3 meter radius
- **Update Logic**: Higher confidence score akan update data existing

## 📊 Dashboard Features

1. **Dashboard**: Metrics total deteksi, rata-rata confidence
2. **Upload Video**: Interface upload video dashcam (.mp4/.avi)
3. **Peta Lokasi**: Visualisasi interaktif dengan popup evidence

## 🔒 Security Notes

- Repository ini menggunakan **private repository** untuk keamanan
- Model checkpoint tidak di-include dalam repo
- Database files di-exclude dari version control

## 📝 License

This project is proprietary. All rights reserved.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📞 Support

Untuk pertanyaan atau support, silakan buat issue di repository ini.