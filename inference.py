"""
Pipeline inferensi Deformable DETR untuk deteksi lubang jalan dari video dashcam.

Alur:
  Video MP4 → ekstrak 1 frame/detik → Deformable DETR detect
  → jika ada pothole → EasyOCR baca GPS pojok kanan bawah
  → Haversine dedup → SQLite → simpan foto evidence
"""

import os
import sys
import re
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from types import SimpleNamespace

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

# ImageNet normalization
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
    args = SimpleNamespace(
        backbone='resnet50',
        dilation=False,
        position_embedding='sine',
        position_embedding_scale=6.283185307179586,
        num_feature_levels=4,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=1024,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=300,
        dec_n_points=4,
        enc_n_points=4,
        with_box_refine=False,
        two_stage=False,
        masks=False,
        aux_loss=False,
        dataset_file='coco',
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
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()
    print(f"Model siap di device: {device}")
    return model, device


# ── Pre & post processing ─────────────────────────────────────────────────────
def preprocess(frame_bgr):
    """BGR frame → normalized tensor, return juga ukuran asli (H, W)."""
    h, w = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Resize: short side ≤ 800, long side ≤ 1333 (sama seperti training)
    scale = min(800 / min(h, w), 1333 / max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

    tensor = TF.to_tensor(pil_img)
    tensor = TF.normalize(tensor, MEAN, STD)
    return tensor.unsqueeze(0), (h, w)


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
    h, w = frame_bgr.shape[:2]
    roi   = frame_bgr[int(h * 0.75):h, int(w * 0.35):w]

    reader  = _get_ocr_reader()
    results = reader.readtext(roi, detail=0)
    text    = ' '.join(results)

    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)  # fix OCR spaces
    match = GPS_REGEX.search(text)
    if match:
        lon = float(match.group(1))
        lat = -float(match.group(2))   # S = South = negatif
        return lat, lon
    return None, None


# ── Visualisasi ───────────────────────────────────────────────────────────────
def draw_boxes(frame, scores, boxes):
    for score, (x1, y1, x2, y2) in zip(scores, boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'Pothole {score:.2f}'
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return frame


# ── Pipeline utama ────────────────────────────────────────────────────────────
def process_video(video_path, conf_threshold=CONF_THRESHOLD, checkpoint=CHECKPOINT):
    init_db()
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    model, device = load_model(checkpoint)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

    fps            = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            sec = frame_idx // frame_interval
            stats['total_frames'] += 1

            # Deteksi
            tensor, orig_hw = preprocess(frame)
            tensor = tensor.to(device)
            with torch.no_grad():
                outputs = model(tensor)

            scores, boxes = postprocess(outputs, orig_hw, conf_threshold)

            if scores:
                stats['detections'] += 1
                best_score = max(scores)

                # Baca GPS
                lat, lon = extract_gps(frame)

                if lat is None:
                    stats['no_gps'] += 1
                    print(f"[{sec:4d}s] 🟡 Pothole {best_score:.2f} | GPS tidak terbaca")
                else:
                    # Simpan evidence image
                    vis_frame = draw_boxes(frame.copy(), scores, boxes)
                    img_name  = f"pothole_{sec:05d}s_{best_score:.2f}.jpg"
                    img_path  = os.path.join(EVIDENCE_DIR, img_name)
                    cv2.imwrite(img_path, vis_frame)

                    # Simpan ke DB
                    result = insert_or_update(lat, lon, best_score, img_path)
                    stats[result] += 1
                    print(f"[{sec:4d}s] 🔴 Pothole {best_score:.2f} | "
                          f"GPS ({lat:.4f}, {lon:.4f}) | DB: {result}")

        frame_idx += 1

    cap.release()

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
    import argparse
    parser = argparse.ArgumentParser(description='Deteksi lubang jalan dari video dashcam')
    parser.add_argument('video',  help='Path ke file video (.mp4 / .avi)')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD,
                        help=f'Confidence threshold (default: {CONF_THRESHOLD})')
    parser.add_argument('--checkpoint', default=CHECKPOINT,
                        help='Path ke model checkpoint')
    args = parser.parse_args()

    process_video(args.video, args.conf, args.checkpoint)
