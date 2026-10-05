from paddleocr import PaddleOCR 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 
import cv2 as cv 
import time 
import threading 
from PIL import Image, ImageDraw, ImageFont 
import numpy as np
import torch
 
# Switch to PP-OCRv5 mobile models 
ocr = PaddleOCR( 
    text_detection_model_name="PP-OCRv5_mobile_det", 
    text_recognition_model_name="PP-OCRv5_mobile_rec", 
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=True, 
    engine="transformers", 
)  
 
# Tokenizer for the translator 
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M") 
model = AutoModelForSeq2SeqLM.from_pretrained( 
    "facebook/nllb-200-distilled-600M", 
)
model.to("cuda")
model.eval() 
model.generation_config.max_length = None 
torch.set_num_threads(4)
tokenizer.src_lang = "eng_Latn" 
 
# Video Capture  
cap = cv.VideoCapture(0) 
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Font
# Just use the generic windows one can be changed later
FONT_PATH = "C:/Windows/Fonts/msyh.ttc" 
fonts = {}
 
# Storage
cap_frame = None 
cap_time = 0
live_frame = None 
ocr_result = None 
overlay = ([], [], None)
 
# Locks 
frame_lock = threading.Lock() 
ocr_lock = threading.Lock() 
overlay_lock = threading.Lock()

# Running
running = threading.Event()
running.set()

# Shrinks a frame to a small grayscale image for comparing scenes
def small_gray(frame): 
    return cv.cvtColor(cv.resize(frame, (80, 45)), cv.COLOR_BGR2GRAY)
 
# Reads the current frame of the video 
def video_capture(cap, interval): 
    global cap_frame, cap_time, live_frame
    print("Start video capture") 

    # Stops program if camera is not opened
    if not cap.isOpened(): 
        print("Cannot open camera") 
        running.clear() 
        return 
 
    last_time = 0 
 
    while running.is_set(): 
        # Capture frame-by-frame 
        ret, frame = cap.read() 
 
        # If frame is read correctly ret is True 
        if not ret: 
            print("Can't receive frame (stream end?). Exiting ...") 
            running.clear()
            return 

        # Only set the captured frame once every interval seconds
        now = time.time() 
        with frame_lock:
            live_frame = frame
            if now - last_time >= interval: 
                cap_frame = frame.copy() 
                cap_time = time.perf_counter()
                last_time = now
 
# Extracts text from an image 
def ocr_predict(ocr): 
    global cap_frame, cap_time, ocr_result
    print("Start ocr prediction") 
 
    last_frame = None 
    prev_small = None
 
    while running.is_set(): 
        # Getting the frame 
        with frame_lock: 
            frame = cap_frame 
            captured = cap_time

        # Waiting for a new frame
        if frame is None or frame is last_frame: 
            time.sleep(0.05) 
            continue 
 
        last_frame = frame
        small = small_gray(frame)

        # Skip ocr if the frame is the same
        if prev_small is not None and cv.absdiff(small, prev_small).mean() < 3: 
            continue 

        prev_small = small 

        t0 = time.perf_counter() 
        result = ocr.predict(frame) 
        ocr_ms = (time.perf_counter() - t0) * 1000 
 
        with ocr_lock: 
            ocr_result = (result, small, captured, ocr_ms)     
 
 
# Translates text 
def translate(tokenizer, model, batch_size): 
    global ocr_result, overlay
    print("Start translation") 
 
    last_result = None 
 
    while running.is_set(): 
        with ocr_lock: 
            result = ocr_result 

        # Waiting for new texts
        if result is None or result is last_result: 
            time.sleep(0.1) 
            continue 
 
        last_result = result 
        result, small, captured, ocr_ms = result
        texts = list(result[0]["rec_texts"]) 

        t0 = time.perf_counter() 
        # Translating 
        # Batches translation for efficiency
        translated = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(model.device)
            out = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"),
                max_length=50,
            )
            translated.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        texts = translated
        translate_ms = (time.perf_counter() - t0) * 1000
 
        # Saving the result 
        with overlay_lock:
            overlay = (result[0]["rec_boxes"], texts, small)

        total_ms = (time.perf_counter() - captured) * 1000 
        wait_ms = total_ms - ocr_ms - translate_ms 
        print(f"lines={len(texts)} | ocr={ocr_ms:.0f} ms | translate={translate_ms:.0f} ms | wait={wait_ms:.0f} ms | total={total_ms:.0f} ms")

# Helper function for getting the font based on size
def get_font(size): 
    if size not in fonts: 
        fonts[size] = ImageFont.truetype(FONT_PATH, size) 
    return fonts[size] 

# Builds a layer of text and a mask of the background rectangles
def build_layer(width, height, boxes, texts): 
    layer = Image.new("RGB", (width, height)) 
    mask = Image.new("L", (width, height)) 
    layer_draw = ImageDraw.Draw(layer) 
    mask_draw = ImageDraw.Draw(mask) 
 
    for (x1, y1, x2, y2), text in zip(boxes, texts): 
        font = get_font(max(14, int((y2 - y1) * 0.8))) 
        left, top, right, bottom = layer_draw.textbbox((x1, y1), text, font=font) 
        mask_draw.rectangle( 
            (min(x1, left) - 4, min(y1, top) - 4, max(x2, right) + 4, max(y2, bottom) + 4), 
            fill=255, 
        ) 
        layer_draw.text((x1, y1), text, font=font, fill=(255, 255, 255)) 
 
    return cv.cvtColor(np.asarray(layer), cv.COLOR_RGB2BGR), np.asarray(mask) > 0

# Applies the translated overlay to a frame when it still matches the scene
def apply_overlay(frame, state, threshold=8): 
    with overlay_lock: 
        current = overlay 
 
    if current is not state["last_overlay"]: 
        state["last_overlay"] = current 
        state["layer"], state["mask"] = build_layer(frame.shape[1], frame.shape[0], current[0], current[1]) 
 
    small = current[2] 
    if small is not None and cv.absdiff(small_gray(frame), small).mean() <= threshold: 
        frame[state["mask"]] = state["layer"][state["mask"]] 
 
    return frame
 

# Running everything
threads = [
    threading.Thread(target=video_capture, args=(cap, 1), daemon=True),
    threading.Thread(target=ocr_predict, args=(ocr,), daemon=True),
    threading.Thread(target=translate, args=(tokenizer, model, 16), daemon=True),
]

print("Start execution")
for t in threads:
    t.start()

last_shown = None 
last_overlay = None 
layer = None 
mask = None 
visible = False 
 
overlay_state = {"last_overlay": None, "layer": None, "mask": None} 
last_shown = None 

try: 
    while running.is_set(): 
        with frame_lock: 
            frame = live_frame 
 
        if frame is not None and frame is not last_shown: 
            last_shown = frame 
            cv.imshow("Translation", apply_overlay(frame, overlay_state))
 
        if cv.waitKey(1) & 0xFF == ord("q"): 
            break 
 
        if not all(t.is_alive() for t in threads): 
            break 
except KeyboardInterrupt: 
    pass
finally:
    running.clear()
    for t in threads:
        t.join(timeout=5)

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()