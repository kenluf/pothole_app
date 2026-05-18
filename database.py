import sqlite3 
import os #cek path, buat folder, dll
from math import radians, sin, cos, sqrt, atan2 #fungsi untuk menghitung jarak antara dua titik GPS menggunakan rumus Haversine

DB_PATH = os.path.join(os.path.dirname(__file__), 'pothole_data.db')


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) #pastikan folder untuk database ada, jika tidak buat folder tersebut
    conn = sqlite3.connect(DB_PATH) #koneksi ke database SQLite
    conn.execute('''
        CREATE TABLE IF NOT EXISTS pothole_data (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude         REAL,
            longitude        REAL,
            confidence_score REAL,
            image_path       TEXT
        )
    ''')
    conn.commit()
    conn.close()


def haversine(lat1, lon1, lat2, lon2):
    """Jarak dalam meter antara dua koordinat GPS."""
    R = 6371000 #jari-jari bumi dalam meter
    phi1, phi2 = radians(lat1), radians(lat2) #konversi derajat ke radian
    dphi    = radians(lat2 - lat1) #selisih latitude dalam radian
    dlambda = radians(lon2 - lon1) #selisih longitude dalam radian
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2 
    return R * 2 * atan2(sqrt(a), sqrt(1 - a)) #hitung jarak menggunakan rumus Haversine dan kembalikan dalam meter


def insert_or_update(lat, lon, confidence, image_path, threshold_meters=3.0): 
    """
    Cek duplikat dengan Haversine.
    - Jarak < 3m & confidence lebih tinggi → UPDATE
    - Jarak < 3m & confidence lebih rendah → skip (duplicate)
    - Jarak >= 3m → INSERT baru
    Returns: 'inserted' | 'updated' | 'duplicate'
    """
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute('SELECT id, latitude, longitude, confidence_score FROM pothole_data') #ambil semua data lubang jalan yang sudah ada di database untuk dibandingkan dengan data baru yang akan dimasukkan
    rows = cur.fetchall() #ambil semua baris hasil query sekaligus

    for row_id, r_lat, r_lon, r_conf in rows: #loop untuk cek setiap data lubang jalan yang sudah ada di database hitung
        dist = haversine(lat, lon, r_lat, r_lon)
        if dist < threshold_meters: #kalau jaraknya kurang dari threshold (3 meter), cek confidence score
            if confidence > r_conf: #kalau confidence score baru lebih tinggi, update data lama dengan data baru
                cur.execute(
                    'UPDATE pothole_data SET confidence_score=?, image_path=? WHERE id=?',
                    (confidence, image_path, row_id)
                )
                conn.commit()
                conn.close()
                return 'updated'
            conn.close()
            return 'duplicate' #kalau confidence score baru lebih rendah, anggap sebagai duplikat dan skip tanpa menyimpan ke database

    cur.execute(
        'INSERT INTO pothole_data (latitude, longitude, confidence_score, image_path) VALUES (?,?,?,?)',
        (lat, lon, confidence, image_path) #simpan data baru ke database jika tidak ada data lama yang berjarak kurang dari threshold
    )
    conn.commit()
    conn.close()
    return 'inserted'


def get_all(): #fungsi untuk mengambil semua data lubang jalan dari database, digunakan untuk ditampilkan di dashboard
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute('SELECT * FROM pothole_data')
    rows = cur.fetchall()
    conn.close()
    return rows
