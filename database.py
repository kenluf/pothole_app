import sqlite3
import os
from math import radians, sin, cos, sqrt, atan2

DB_PATH = os.path.join(os.path.dirname(__file__), 'pothole_data.db')


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi    = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


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
    cur.execute('SELECT id, latitude, longitude, confidence_score FROM pothole_data')
    rows = cur.fetchall()

    for row_id, r_lat, r_lon, r_conf in rows:
        dist = haversine(lat, lon, r_lat, r_lon)
        if dist < threshold_meters:
            if confidence > r_conf:
                cur.execute(
                    'UPDATE pothole_data SET confidence_score=?, image_path=? WHERE id=?',
                    (confidence, image_path, row_id)
                )
                conn.commit()
                conn.close()
                return 'updated'
            conn.close()
            return 'duplicate'

    cur.execute(
        'INSERT INTO pothole_data (latitude, longitude, confidence_score, image_path) VALUES (?,?,?,?)',
        (lat, lon, confidence, image_path)
    )
    conn.commit()
    conn.close()
    return 'inserted'


def get_all():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute('SELECT * FROM pothole_data')
    rows = cur.fetchall()
    conn.close()
    return rows
