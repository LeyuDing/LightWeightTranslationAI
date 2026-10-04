from paddleocr import PaddleOCR 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 
import cv2 as cv 
import time 
import threading 
from concurrent.futures import ThreadPoolExecutor 
from PIL import Image, ImageDraw, ImageFont 
import numpy as np
 
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
tokenizer.src_lang = "eng_Latn" 
 
# Video Capture  
cap = cv.VideoCapture(0) 

# Font
# Just use the generic windows one can be changed later
FONT_PATH = "C:/Windows/Fonts/msyh.ttc" 
fonts = {}
 
# Storage
cap_frame = None 
live_frame = None 
ocr_result = None 
overlay = ([], [])
 
# Locks 
frame_lock = threading.Lock() 
ocr_lock = threading.Lock() 
overlay_lock = threading.Lock()

# Running
running = threading.Event()
running.set()
 
# Reads the current frame of the video 
def video_capture(cap, interval): 
    global cap_frame, live_frame, running 
    print("Start video capture") 

    # Stops program if camera is not running
    if not cap.isOpened(): 
        print("Cannot open camera") 
        running = False 
        return 
 
    last_time = 0 
 
    while running: 
        # Capture frame-by-frame 
        ret, frame = cap.read() 
 
        # If frame is read correctly ret is True 
        if not ret: 
            print("Can't receive frame (stream end?). Exiting ...") 
            running = False 
            return 

        with frame_lock:
            live_frame = frame

        # Only set the captured frame once every interval seconds
        now = time.time() 
        if now - last_time >= interval: 
            with frame_lock:  
                cap_frame = frame.copy() 
            last_time = now 
 
# Extracts text from an image 
def ocr_predict(ocr): 
    global cap_frame, ocr_result, running 
    print("Start ocr prediction") 
 
    last_frame = None 
 
    while running: 
        # Getting the frame 
        with frame_lock: 
            frame = cap_frame 

        # Waiting for a new frame
        if frame is None or frame is last_frame: 
            time.sleep(0.05) 
            continue 
 
        last_frame = frame 
 
        result = ocr.predict(frame) 
 
        with ocr_lock: 
            ocr_result = result     
 
 
# Translates text 
def translate(tokenizer, model, batch_size): 
    global ocr_result, overlay, running
    print("Start translation") 
 
    last_result = None 
 
    while running: 
        with ocr_lock: 
            result = ocr_result 

        # Waiting for new texts
        if result is None or result is last_result: 
            time.sleep(0.1) 
            continue 
 
        last_result = result 
        texts = list(result[0]["rec_texts"]) 
 
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
 
        result[0]["rec_texts"] = texts 
 
        # Saving the result 
        with overlay_lock:
            overlay = (result[0]["rec_boxes"], texts)
 
        time.sleep(0.1) 

# Helper function for getting the font based on size
def get_font(size): 
    if size not in fonts: 
        fonts[size] = ImageFont.truetype(FONT_PATH, size) 
    return fonts[size] 

# Draw a rectangular background with the text on top
def draw_overlay(frame, boxes, texts): 
    # Converting captured image to RGB for pillow
    img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)) 
    draw = ImageDraw.Draw(img) 

    # Iterate for each text
    for (x1, y1, x2, y2), text in zip(boxes, texts): 
        # Draw the box based on the font size
        font = get_font(max(14, int((y2 - y1) * 0.8))) 
        left, top, right, bottom = draw.textbbox((x1, y1), text, font=font) 
        draw.rectangle( 
            (min(x1, left) - 4, min(y1, top) - 4, max(x2, right) + 4, max(y2, bottom) + 4), 
            fill=(0, 0, 0), 
        ) 

        draw.text((x1, y1), text, font=font, fill=(255, 255, 255)) 

    # Need to convert back to BGR for OpenCV
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
 

# Running everything
threads = [
    threading.Thread(target=video_capture, args=(cap, 1), daemon=True),
    threading.Thread(target=ocr_predict, args=(ocr,), daemon=True),
    threading.Thread(target=translate, args=(tokenizer, model, 16), daemon=True),
]

print("Start execution")
for t in threads:
    t.start()

try:
    while running:
        with frame_lock:
            frame = live_frame

        if frame is not None:
            with overlay_lock:
                boxes, texts = overlay

            cv.imshow("Translation", draw_overlay(frame, boxes, texts))

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        if not all(t.is_alive() for t in threads):
            break
except KeyboardInterrupt:
    pass
finally:
    running = False
    for t in threads:
        t.join(timeout=5)

    ## When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()