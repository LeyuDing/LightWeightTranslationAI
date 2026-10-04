from paddleocr import PaddleOCR 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 
import cv2 as cv 
import time 
import threading 
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION 
import os # Only needed for testing purposes 
import torch

os.makedirs("output", exist_ok=True) 
 
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
 
# Storage Variables 
cap_frame = None 
ocr_result = None 
 
# Locks 
frame_lock = threading.Lock() 
ocr_lock = threading.Lock() 
 
running = threading.Event()
running.set()
 
# Reads the current frame of the video 
def video_capture(cap, interval): 
    global cap_frame, running 
    print("Start video capture") 
 
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
 
        if frame is None or frame is last_frame: 
            time.sleep(0.05) 
            continue 
 
        last_frame = frame 
 
        result = ocr.predict(frame) 
 
        with ocr_lock: 
            ocr_result = result     
 
 
# Translates text 
def translate(tokenizer, model): 
    global ocr_result, running 
    print("Start translation") 
 
    last_result = None 

    cache = {} 
 
    while running: 
        with ocr_lock: 
            result = ocr_result 
 
        if result is None or result is last_result: 
            time.sleep(0.1) 
            continue 
 
        last_result = result 
        texts = list(result[0]["rec_texts"]) 
 
        # Translating 
        missing = [t for t in dict.fromkeys(texts) if t not in cache] 

        for start in range(0, len(missing), 16): 
            chunk = missing[start:start + 16] 
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(model.device) 
            with torch.inference_mode(): 
                out = model.generate( 
                    **inputs, 
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"), 
                    num_beams=1, 
                    do_sample=False, 
                    max_new_tokens=50, 
                    ) 
            cache.update(zip(chunk, tokenizer.batch_decode(out, skip_special_tokens=True))) 

        result[0]["rec_texts"] = [cache[t] for t in texts]
 
        # Saving the result 
        for res in result:   
            res.save_to_img(os.path.join("output", "ocr_result.jpg"))   
            res.save_to_json(os.path.join("output", "ocr_result.json")) 
 
        time.sleep(0.1) 
 
 
threads = [
    threading.Thread(target=video_capture, args=(cap, 1), daemon=True),
    threading.Thread(target=ocr_predict, args=(ocr,), daemon=True),
    threading.Thread(target=translate, args=(tokenizer, model), daemon=True),
]

print("Start execution")
for t in threads:
    t.start()

try:
    while running.is_set():
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
finally:
    running.clear()
    for t in threads:
        t.join(timeout=5)
    cap.release()
    cv.destroyAllWindows()