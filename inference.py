import os #cek path, buat folder, dll
import sys #modifikasi path python untuk impor modul dari folder lain
import re #regex untuk ekstrak GPS dari teks OCR
import cv2 #OpenCV baca video, gambar, dll
import torch #PyTorch untuk model, tensor, dll
import numpy as np #Operasi array untuk manipulasi data
from PIL import Image #manipulasi gambar (resize, konversi, dll)
import torchvision.transforms.functional as TF #transformasi gambar (to_tensor, normalize, dll)
from types import SimpleNamespace #buat objek sederhana untuk menyimpan argumen model

# ── Path ke repo Deformable-DETR ─────────────────────────────────────────────
DETR_PATH = os.path.join(os.path.dirname(__file__), 'deformable_detr')
sys.path.insert(0, DETR_PATH)

from models import build_model
from database import init_db, insert_or_update

# ── Konfigurasi ───────────────────────────────────────────────────────────────
CHECKPOINT    = os.path.join(os.path.dirname(__file__), 'deformable_detr', 'checkpoint.pth')
CHECKPOINT_URL = os.getenv(
    'CHECKPOINT_URL',
    'https://storage.googleapis.com/checkpoint_detr/checkpoint.pth'
)
FILTERED_PATH = os.path.join(os.path.dirname(__file__), 'deformable_detr', 'r50_filtered.pth')
FILTERED_URL = os.getenv(
    'R50_FILTERED_URL',
    'https://storage.googleapis.com/checkpoint_detr/r50_filtered.pth'
)
EVIDENCE_DIR  = os.path.join(os.path.dirname(__file__), 'static', 'evidence')
CONF_THRESHOLD = 0.5

# ImageNet normalization (karena model dilatih dengan backbone pretrained ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Regex GPS: "E106.5991,S6.1581" atau "N/S" dan "E/W"
GPS_REGEX = re.compile(r'[Ee]([\d.]+)[,\s]+[Ss]([\d.]+)')

# EasyOCR reader (lazy init, hanya dibuat sekali)
_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
    return _ocr_reader

# Download model files jika belum ada
def _download_file(url, dst_path):
    import urllib.request

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    print(f"Downloading model file from: {url}")
    with urllib.request.urlopen(url) as response:
        with open(dst_path, 'wb') as out_file:
            while True:
                chunk = response.read(32768)
                if not chunk:
                    break
                out_file.write(chunk)
    print(f"Saved model file to: {dst_path}")


def _ensure_model_files():
    if not os.path.exists(CHECKPOINT):
        _download_file(CHECKPOINT_URL, CHECKPOINT)
    if not os.path.exists(FILTERED_PATH):
        _download_file(FILTERED_URL, FILTERED_PATH)


# ── Build & load model ────────────────────────────────────────────────────────
def build_deformable_detr():
    args = SimpleNamespace( #simplenamespace untuk menyimpan argumen model, mirip dengan argparse.Namespace tapi lebih sederhana untuk penggunaan streamlit
        backbone='resnet50',           # Backbone CNN yang digunakan
        dilation=False,                # Tidak pakai dilated convolution
        position_embedding='sine',     # Jenis positional encoding
        position_embedding_scale=6.283185307179586,  # 2π
        num_feature_levels=4,          # Jumlah skala fitur multi-level
        enc_layers=6,                  # Jumlah layer encoder Transformer
        dec_layers=6,                  # Jumlah layer decoder Transformer
        dim_feedforward=1024,          # Dimensi layer feedforward
        hidden_dim=256,                # Dimensi embedding tersembunyi
        dropout=0.1,                   # Dropout 10% untuk regularisasi
        nheads=8,                      # Jumlah attention head
        num_queries=300,               # Jumlah kandidat objek yang dicek
        dec_n_points=4,                # Jumlah sampling point decoder
        enc_n_points=4,                # Jumlah sampling point encoder
        with_box_refine=False,         # Tidak pakai iterative box refinement
        two_stage=False,               # Tidak pakai two-stage detection
        masks=False,                   # Tidak pakai segmentasi mask
        aux_loss=False,                # Tidak pakai auxiliary loss
        dataset_file='coco',           # Format dataset (COCO)
        # loss coef (dibutuhkan build_model tapi tidak dipakai saat inference)
        cls_loss_coef=2, bbox_loss_coef=5, giou_loss_coef=2,
        mask_loss_coef=1, dice_loss_coef=1,
        focal_alpha=0.25, set_cost_class=2, set_cost_bbox=5, set_cost_giou=2,
        lr_backbone=0,        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    model, _, _ = build_model(args)
    return model, args.device


def load_model(checkpoint_path=CHECKPOINT):
    _ensure_model_files()
    print(f"Loading model dari: {checkpoint_path}")
    model, device = build_deformable_detr() 
    ckpt = torch.load(checkpoint_path, map_location='cpu') #load ke CPU dulu
    model.load_state_dict(ckpt['model'], strict=False) #fungsi untuk memuat model, strict=False karena mungkin ada mismatch kecil antara arsitektur model dan checkpoint (misal nama layer berbeda)
    model.to(device) #pindahkan model ke GPU jika tersedia, atau tetap di CPU jika tidak ada GPU
    model.eval() #set model ke mode evaluasi (non-training) untuk menonaktifkan dropout & batchnorm agar output konsisten
    print(f"Model siap di device: {device}")
    return model, device


# ── Pre & post processing ─────────────────────────────────────────────────────
def preprocess(frame_bgr):
    """BGR frame → normalized tensor, return juga ukuran asli (H, W)."""
    h, w = frame_bgr.shape[:2] #ambil tinggi dan lebar dari frame input
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #konversi dari format BGR (OpenCV) ke RGB (PIL)
    pil_img = Image.fromarray(img_rgb) 

    # Resize: short side ≤ 800, long side ≤ 1333 (sama seperti training)
    scale = min(800 / min(h, w), 1333 / max(h, w)) #hitung faktor skala untuk memastikan ukuran gambar sesuai dengan batasan yang ditentukan (short side ≤ 800, long side ≤ 1333)
    new_w, new_h = int(w * scale), int(h * scale) #hitung ukuran baru berdasarkan faktor skala
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR) #resize gambar menggunakan interpolasi bilinear untuk menjaga kualitas

    tensor = TF.to_tensor(pil_img) #tensor untuk memproses data, konversi gambar PIL ke tensor PyTorch (C, H, W) dengan nilai piksel di [0, 1]
    tensor = TF.normalize(tensor, MEAN, STD)    #normalisasi dengan nilai imagenet
    return tensor.unsqueeze(0), (h, w) #tambahkan dimensi batch (1, C, H, W) dan kembalikan ukuran asli untuk postprocessing


def postprocess(outputs, orig_hw, threshold):
    """
    outputs['pred_logits']: [1, 300, 2]
    outputs['pred_boxes']:  [1, 300, 4]  cx,cy,w,h ternormalisasi
    Return: (scores, boxes_px) list
    """
    h, w = orig_hw
    prob  = outputs['pred_logits'].sigmoid()[0]   # [300, 2]
    boxes = outputs['pred_boxes'][0].cpu()         # [300, 4]

    # Kelas 1 = pothole
    scores = prob[:, 1].cpu()
    keep   = scores > threshold

    scores = scores[keep].tolist()
    boxes  = boxes[keep]

    if len(scores) == 0:
        return [], []

    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = ((cx - bw / 2) * w).clamp(0, w - 1).int().tolist()
    y1 = ((cy - bh / 2) * h).clamp(0, h - 1).int().tolist()
    x2 = ((cx + bw / 2) * w).clamp(0, w - 1).int().tolist()
    y2 = ((cy + bh / 2) * h).clamp(0, h - 1).int().tolist()
    boxes_px = list(zip(x1, y1, x2, y2))

    return scores, boxes_px


# ── GPS extraction ─────────────────────────────────────────────────────────────
def extract_gps(frame_bgr):
    """
    Crop pojok kanan bawah → EasyOCR → Regex.
    Format 70mai: "34km/h E106.5991,S6.1581"
    Return: (latitude, longitude) atau (None, None) jika gagal.
    """
    h, w = frame_bgr.shape[:2] #ambil tinggi dan lebar dari frame input
    roi   = frame_bgr[int(h * 0.75):h, int(w * 0.35):w] #crop pojok kanan bawah 75% tinggi dan 65% lebar

    reader  = _get_ocr_reader() #jalankan OCR dengan EasyOCR
    results = reader.readtext(roi, detail=0) #jalankan OCR pada ROI, detail=0 untuk hanya mendapatkan teks tanpa koordinat atau confidence
    text    = ' '.join(results) #gabungkan semua hasil OCR menjadi satu string

    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)  # fix OCR spaces di angka desimal, misal "E106 .5991" → "E106.5991"
    match = GPS_REGEX.search(text)
    if match:
        lon = float(match.group(1))
        lat = -float(match.group(2))   # S = South = negatif
        return lat, lon
    return None, None


# ── Visualisasi ───────────────────────────────────────────────────────────────
def draw_boxes(frame, scores, boxes):
    for score, (x1, y1, x2, y2) in zip(scores, boxes): #iterasi setiap deteksi, gambar kotak dan label pada frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) #gambar kotak hijau dengan ketebalan 2
        label = f'Pothole {score:.2f}' #label dengan format "Pothole 0.85" (misal)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), #taruh label di atas kotak, pastikan tidak keluar frame
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2) #jenis font, warna dan ukuran teks
    return frame


# ── Pipeline utama ────────────────────────────────────────────────────────────
def process_video(video_path, conf_threshold=CONF_THRESHOLD, checkpoint=CHECKPOINT):
    init_db()
    os.makedirs(EVIDENCE_DIR, exist_ok=True) #buat folder untuk menyimpan foto evidence

    model, device = load_model(checkpoint) #load model deteksi lubang jalan dari checkpoint

    cap = cv2.VideoCapture(video_path) #buka file video dengan OpenCV
    if not cap.isOpened():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}") 

    fps            = cap.get(cv2.CAP_PROP_FPS) #ambil frame per detik dari video 
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #ambil total jumlah frame dalam video
    frame_interval = max(int(fps), 1)   # ambil 1 frame per detik

    print(f"\nVideo  : {video_path}")
    print(f"FPS    : {fps:.1f}  |  Total frame: {total_frames}")
    print(f"Ekstrak: setiap {frame_interval} frame (1 fps)")
    print(f"Conf   : {conf_threshold}\n")

    stats = {
        'total_frames' : 0,
        'detections'   : 0,
        'no_gps'       : 0,
        'inserted'     : 0,
        'updated'      : 0,
        'duplicate'    : 0,
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read() #baca satu frame dari video, ret = True jika berhasil, false jika sudah akhir video
        if not ret:
            break #jika sudah tidak ada frame lagi, keluar dari loop

        if frame_idx % frame_interval == 0: #proses hanya setiap frame_interval (misal setiap 30 frame untuk 1 fps)
            sec = frame_idx // frame_interval
            stats['total_frames'] += 1

            # Deteksi
            tensor, orig_hw = preprocess(frame) #ubah frame ke tensor siap untuk model
            tensor = tensor.to(device) #pindahkan tensor ke gpu
            with torch.no_grad(): #matikan kalkulasi autograd untuk efisiensi 
                outputs = model(tensor) #jalankan model untuk mendapatkan output prediksi

            scores, boxes = postprocess(outputs, orig_hw, conf_threshold) 

            if scores:
                stats['detections'] += 1
                best_score = max(scores)

                # Baca GPS
                lat, lon = extract_gps(frame)

                if lat is None:
                    stats['no_gps'] += 1 #jika GPS tidak terbaca, catat statistik dan lanjutkan tanpa menyimpan ke DB atau folder evidence
                    print(f"[{sec:4d}s] 🟡 Pothole {best_score:.2f} | GPS tidak terbaca")
                else:
                    # Simpan evidence image
                    vis_frame = draw_boxes(frame.copy(), scores, boxes) #buat salinan frame untuk visualisasi, gambar kotak deteksi pada salinan tersebut
                    img_name  = f"pothole_{sec:05d}s_{best_score:.2f}.jpg"
                    img_path  = os.path.join(EVIDENCE_DIR, img_name)
                    cv2.imwrite(img_path, vis_frame) #simpan foto dengan kotak deteksi sebagai bukti

                    # Simpan ke DB
                    result = insert_or_update(lat, lon, best_score, img_path) #simpan data ke database
                    stats[result] += 1 #update statistik berdasarkan hasil operasi database (inserted, updated, duplicate)  
                    print(f"[{sec:4d}s] 🔴 Pothole {best_score:.2f} | "
                          f"GPS ({lat:.4f}, {lon:.4f}) | DB: {result}")

        frame_idx += 1

    cap.release() #tutup video setelah selesai

    print("\n═══════════════════════════════")
    print("           SELESAI             ")
    print("═══════════════════════════════")
    print(f"Frame diproses : {stats['total_frames']}")
    print(f"Ada pothole    : {stats['detections']}")
    print(f"GPS gagal      : {stats['no_gps']}")
    print(f"Disimpan baru  : {stats['inserted']}")
    print(f"Diupdate       : {stats['updated']}")
    print(f"Duplikat (skip): {stats['duplicate']}")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse #untuk parsing argumen dari command line, seperti path video, confidence threshold, dan checkpoint model
    parser = argparse.ArgumentParser(description='Deteksi lubang jalan dari video dashcam')
    parser.add_argument('video',  help='Path ke file video (.mp4 / .avi)')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD,
                        help=f'Confidence threshold (default: {CONF_THRESHOLD})')
    parser.add_argument('--checkpoint', default=CHECKPOINT,
                        help='Path ke model checkpoint')
    args = parser.parse_args()

    process_video(args.video, args.conf, args.checkpoint)
