import time as _time_module

# print をオーバーライドして経過時間（ミリ秒）を先頭に付与する
_print_start = _time_module.monotonic()
_builtin_print = print
def print(*args, **kwargs):
    elapsed_ms = (_time_module.monotonic() - _print_start) * 1000
    _builtin_print(f"[{elapsed_ms:8.1f}ms]", *args, **kwargs)

print("import time 完了")

from flask import Flask, Response, render_template
print("import flask 完了")

from flask_socketio import SocketIO
print("import flask_socketio 完了")

from picamera2 import Picamera2
print("import picamera2 完了")

import cv2
print("import cv2 完了")

import threading
import random
import numpy as np
print("import threading / random / numpy 完了")

from skimage.measure import CircleModel, LineModelND, ransac
print("import skimage 完了")

from collections import deque
print("import collections 完了")

import time  # 既存コードの time.sleep / time.time 互換エイリアス

print("Flask / SocketIO 初期化 開始")
app = Flask(__name__)
socketio = SocketIO(app)
print("Flask / SocketIO 初期化 完了")

# Raspberry Pi CSIカメラの初期化
print("Picamera2 インスタンス生成 開始")
picam2 = Picamera2()
print("Picamera2 インスタンス生成 完了")

print("カメラ設定 開始")
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 360), "format": "BGR888"}
)
picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False,
    "ExposureTime": 10000,
    "AnalogueGain": 4
})
picam2.configure(camera_config)
print("カメラ設定 完了")

print("カメラ起動 開始")
picam2.start()
print("カメラ起動 完了")

# =========================
# HSV分類パラメータ
# =========================
RED_S_TH = 90
YELLOW_S_TH = 90

# =========================
# 検出設定
# =========================
ENABLE_CIRCLE_DETECTION = True
ENABLE_LINE_DETECTION = True

print("OpenCV カーネル / CLAHE 初期化 開始")
kernel = np.ones((3, 3), np.uint8)

# CLAHE初期化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
print("OpenCV カーネル / CLAHE 初期化 完了")

# 状態共有
state = {
    "distance": 0,
    "angle": 0,
    "maze_pos": (0, 0)
}

# ランダムなセンサ値を生成して WebSocket で送信し続けるバックグラウンドスレッド関数
# 引数: なし
# 処理: distance / angle / maze_pos をランダム更新し socketio.emit で配信（0.1秒周期）
# 戻り値: なし（無限ループ）
def camera_loop():
    while True:
        state["distance"] = random.randint(10, 100)
        state["angle"] = random.randint(-45, 45)
        state["maze_pos"] = (random.randint(0, 9), random.randint(0, 9))

        socketio.emit("state", state)
        time.sleep(0.1)

# HSV色空間を用いてフレームを黄色・赤・灰色に分類する
# 引数:
#   frame_bgr: BGR形式の入力フレーム (numpy ndarray, shape=(H,W,3))
# 処理: ガウシアンブラー → HSV変換 → 色マスク生成 → モルフォロジー処理 → 合成画像生成
# 戻り値:
#   result      : 分類済み画像 (BGR, 灰=128/赤=(0,0,255)/黄=(0,255,255))
#   mask_yellow : 黄色ピクセルの bool マスク
#   mask_red    : 赤ピクセルの bool マスク
def classify_frame(frame_bgr):

    frame_blur = cv2.GaussianBlur(frame_bgr, (5,5), 0)

    # =========================
    # 1. HSVで色検出
    # =========================
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 黄色
    mask_yellow = (
        (H > 18) & (H < 38) &
        (S > 100) &
        (V > 80)
    )

    # 赤（2領域）
    mask_red = (
        ((H < 10) | (H > 170)) &
        (S > 100) &
        (V > 80)
    )

    # =========================
    # 2. モルフォロジー
    # =========================
    kernel = np.ones((5,5), np.uint8)

    def clean(mask):
        m = mask.astype(np.uint8)*255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        return m.astype(bool)

    mask_yellow = clean(mask_yellow)
    mask_red = clean(mask_red)

    # =========================
    # 3. 合成
    # =========================
    result = np.full_like(frame_bgr, (128,128,128))

    # 色を適用
    result[mask_red] = (0,0,255)
    result[mask_yellow] = (0,255,255)

    return result, mask_yellow, mask_red

# 黄色ピクセルの割合が閾値以上かどうかでボールの有無を判定する
# 引数:
#   mask_yellow       : 黄色ピクセルの bool マスク (numpy ndarray)
#   threshold_percent : 判定閾値（フレーム全体に占める黄色ピクセルの割合 %、デフォルト 1.0）
# 処理: np.count_nonzero でゼロでない要素数を数え、全画素数 × 閾値と比較
# 戻り値: 黄色ピクセルが閾値以上なら True、未満なら False
def has_yellow_ball(mask_yellow, threshold_percent=1.0):
    return np.count_nonzero(mask_yellow) >= mask_yellow.size * (threshold_percent / 100.0)

# 黄色マスクから最大輪郭を抽出し RANSAC で円を当てはめて中心・直径を返す
# 引数:
#   mask_yellow: 黄色ピクセルの bool マスク (numpy ndarray, shape=(H,W))
# 処理: bool → uint8 変換 → 輪郭検出 → 最大輪郭を 100 点以下にサンプリング → RANSAC 円フィット
# 戻り値:
#   (cx, cy, diameter): 検出成功時。円中心座標と直径 (float のタプル)
#   None              : 検出失敗時
def detect_circle(mask_yellow):
    mask_yellow_uint8 = mask_yellow.astype(np.uint8) * 255
    contours_yellow, _ = cv2.findContours(mask_yellow_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_yellow:
        largest_contour = max(contours_yellow, key=cv2.contourArea)
        if len(largest_contour) >= 3:
            points = largest_contour.reshape(-1, 2)
            # 点が多い場合はサンプリング
            if len(points) > 100:
                step = len(points) // 100
                points = points[::step]
            try:
                model, inliers = ransac(points, CircleModel, min_samples=3,
                                       residual_threshold=3, max_trials=20)
                cx, cy = model.center
                r = model.radius
                diameter = r * 2.0
                return (cx, cy, diameter)  # (cx, cy, diameter)を返す
            except:
                return None
    return None

# 分類画像から「上が灰色・下が赤」の水平エッジを検出し Hough 変換で直線を返す
# 引数:
#   classified: classify_frame() が返す分類済み BGR 画像 (numpy ndarray, shape=(H,W,3))
# 処理: 隣接行の色比較でエッジマスク生成 → HoughLinesP → 50px 以上の直線を抽出 → 最下部を選択
# 戻り値:
#   (slope, intercept): 検出成功時。傾きと y 切片 (float のタプル)。垂直線の場合は (inf, x座標)
#   None              : 検出失敗時
def detect_line(classified):
    h, w = classified.shape[:2]
    
    # 垂直方向で上が灰色(128,128,128)、下が赤(0,0,255)のエッジを検出
    top_row = classified[:-1]
    bottom_row = classified[1:]
    
    # 上が灰色
    is_gray = (top_row[:,:,0] == 128) & (top_row[:,:,1] == 128) & (top_row[:,:,2] == 128)
    # 下が赤
    is_red = (bottom_row[:,:,0] == 0) & (bottom_row[:,:,1] == 0) & (bottom_row[:,:,2] == 255)
    # エッジマスク作成
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[1:] = (is_gray & is_red).astype(np.uint8) * 255
    
    if np.sum(edge_mask) > 0:
        # Hough直線検出
        lines = cv2.HoughLinesP(edge_mask, rho=1, theta=np.pi/180, threshold=10,
                                minLineLength=20, maxLineGap=5)
        
        if lines is not None and len(lines) > 0:
            # 一定の値より長い直線のみフィルタ（50ピクセル以上）
            MIN_LINE_LENGTH = 50
            long_lines = [line for line in lines 
                         if np.hypot(line[0][2]-line[0][0], line[0][3]-line[0][1]) >= MIN_LINE_LENGTH]
            
            if long_lines:
                # 最も画面上で下にある直線を選択（y座標の最大値）
                bottom_line = max(long_lines, key=lambda line: max(line[0][1], line[0][3]))
                x1, y1, x2, y2 = bottom_line[0]
                
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    return (slope, intercept)  # (傾き, 切片)を返す
                else:
                    return (float('inf'), x1)
    return None

# カメラ生フレームに前処理（色変換・ハイライト除去・CLAHE）を施す
# 引数:
#   frame: picamera2 が返す RGB888 形式の生フレーム (numpy ndarray, shape=(H,W,3))
# 処理: RGB→BGR 変換 → HSV で V 値を 200 にクリップしてハイライト除去 → LAB で CLAHE 適用
# 戻り値:
#   frame_bgr     : RGB→BGR 変換のみのフレーム（表示用原画像）
#   frame_enhanced: ハイライト除去 + CLAHE 適用済みフレーム（分類処理用）
def preprocess_frame(frame):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ハイライト除去（V値を200以下に制限）
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v, 0, 200)
    frame_processed = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    # CLAHE適用
    lab = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    frame_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return frame_bgr, frame_enhanced


# 検出バッファが閾値に達したとき中央値付近の平均でスムージングし、バッファをリセットする
# 引数:
#   circle_buffer      : 円検出結果を蓄積する deque (要素: (cx, cy, diameter))
#   line_buffer        : 直線検出結果を蓄積する deque (要素: (slope, intercept))
#   buffer_target_count: フラッシュを起動する蓄積数の閾値 (int)
#   smoothed_circle    : 前回のスムージング済み円結果（更新されない場合にそのまま返す）
#   smoothed_line      : 前回のスムージング済み直線結果（更新されない場合にそのまま返す）
# 処理: いずれかのバッファが閾値に達したら中央付近 4 件の平均を計算してバッファをクリア
# 戻り値:
#   (smoothed_circle, smoothed_line): 更新後のスムージング結果タプル
def flush_buffers(circle_buffer, line_buffer, buffer_target_count, smoothed_circle, smoothed_line):
    if not (len(circle_buffer) >= buffer_target_count or len(line_buffer) >= buffer_target_count):
        return smoothed_circle, smoothed_line

    min_detections = buffer_target_count // 2

    if len(circle_buffer) >= min_detections:
        sorted_circles = sorted(circle_buffer, key=lambda x: x[0])
        mid = len(sorted_circles) // 2
        middle_circles = sorted_circles[max(0, mid - 2):mid + 2]
        smoothed_circle = (
            np.mean([c[0] for c in middle_circles]),
            np.mean([c[1] for c in middle_circles]),
            np.mean([c[2] for c in middle_circles]),
        )
    else:
        smoothed_circle = None

    if len(line_buffer) >= min_detections:
        sorted_lines = sorted(line_buffer, key=lambda x: x[0])
        mid = len(sorted_lines) // 2
        middle_lines = sorted_lines[max(0, mid - 2):mid + 2]
        finite_lines = [l for l in middle_lines if l[0] != float('inf')]
        if finite_lines:
            smoothed_line = (
                np.mean([l[0] for l in finite_lines]),
                np.mean([l[1] for l in finite_lines]),
            )
        else:
            smoothed_line = (float('inf'), np.mean([l[1] for l in middle_lines]))
    else:
        smoothed_line = None

    circle_buffer.clear()
    line_buffer.clear()
    return smoothed_circle, smoothed_line


# 分類画像にスムージング済みの円・直線を緑色で描画したオーバーレイ画像を返す
# 引数:
#   classified     : classify_frame() が返す分類済み BGR 画像 (numpy ndarray, shape=(H,W,3))
#   smoothed_circle: スムージング済み円 (cx, cy, diameter) のタプル、または None
#   smoothed_line  : スムージング済み直線 (slope, intercept) のタプル、または None
# 処理: classified をコピーし、円は cv2.circle、直線は画像端まで延長して cv2.line で描画
# 戻り値: 描画済みの BGR 画像 (numpy ndarray, shape=(H,W,3))
def draw_overlay(classified, smoothed_circle, smoothed_line):
    overlay = classified.copy()
    h_img, w_img = overlay.shape[:2]

    if ENABLE_CIRCLE_DETECTION and smoothed_circle is not None:
        cx, cy, diameter = smoothed_circle
        radius = int(diameter / 2.0)
        cv2.circle(overlay, (int(cx), int(cy)), radius, (0, 255, 0), 2)
        cv2.circle(overlay, (int(cx), int(cy)), 3, (0, 255, 0), -1)

    if ENABLE_LINE_DETECTION and smoothed_line is not None:
        slope, intercept = smoothed_line
        if slope != float('inf'):
            y_at_0 = int(intercept)
            y_at_w = int(slope * (w_img - 1) + intercept)
            x_at_0 = int(-intercept / slope) if slope != 0 else -1
            x_at_h = int((h_img - 1 - intercept) / slope) if slope != 0 else -1

            candidates = []
            if 0 <= y_at_0 < h_img:
                candidates.append((0, y_at_0))
            if 0 <= y_at_w < h_img:
                candidates.append((w_img - 1, y_at_w))
            if 0 <= x_at_0 < w_img:
                candidates.append((x_at_0, 0))
            if 0 <= x_at_h < w_img:
                candidates.append((x_at_h, h_img - 1))

            if len(candidates) >= 2:
                cv2.line(overlay, candidates[0], candidates[-1], (0, 255, 0), 2)
        else:
            x_at = int(intercept)
            if 0 <= x_at < w_img:
                cv2.line(overlay, (x_at, 0), (x_at, h_img - 1), (0, 255, 0), 2)

    return overlay


# MJPEG ストリーミング用ジェネレータ。フレーム取得から前処理・検出・描画・エンコードを行う
# 引数: なし
# 処理:
#   1. picam2 からフレーム取得
#   2. preprocess_frame で前処理
#   3. classify_frame で色分類
#   4. detect_circle / detect_line で検出してバッファに蓄積
#   5. flush_buffers でスムージング
#   6. draw_overlay でオーバーレイ描画
#   7. 原画像とオーバーレイを横結合して JPEG エンコードし multipart で yield
# 戻り値: multipart/x-mixed-replace 形式の JPEG バイト列を無限に yield するジェネレータ
def gen():
    fps = 0
    prev_time = time.time()
    frame_count = 0

    circle_buffer = deque(maxlen=30)
    line_buffer = deque(maxlen=30)
    buffer_target_count = 10
    smoothed_circle = None  # (cx, cy, diameter)
    smoothed_line = None    # (slope, intercept)

    while True:
        frame = picam2.capture_array()
        frame_bgr, frame_enhanced = preprocess_frame(frame)

        classified, mask_yellow, mask_red = classify_frame(frame_enhanced)

        """
        if ENABLE_CIRCLE_DETECTION:
            circle_data = detect_circle(mask_yellow)
            if circle_data is not None:
                circle_buffer.append(circle_data)

        if ENABLE_LINE_DETECTION:
            line_data = detect_line(classified)
            if line_data is not None:
                line_buffer.append(line_data)

        smoothed_circle, smoothed_line = flush_buffers(
            circle_buffer, line_buffer, buffer_target_count,
            smoothed_circle, smoothed_line
        )

        overlay = draw_overlay(classified, smoothed_circle, smoothed_line)

        # FPS計算
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        # FPS表示
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined = np.hstack([frame_bgr, overlay])
        """

        # FPS計算
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        # FPS表示
        cv2.putText(classified, f"FPS: {fps:.1f}, Ball: {has_yellow_ball(mask_yellow, 5)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        combined = classified #np.hstack([frame_bgr, classified])
        _, jpeg = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("カメラループスレッド 起動")
    threading.Thread(target=camera_loop, daemon=True).start()
    print("Flask サーバー 起動")
    socketio.run(app, host='0.0.0.0', port=5000)