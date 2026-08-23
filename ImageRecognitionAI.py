from paddleocr import PaddleOCR
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import cv2 as cv
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import os # Only needed for testing purposes

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

running = True

# Reads the current frame of the video
def video_capture(cap, interval):
    global cap_frame, running
    print("Start video capture")

    if not cap.isOpened():
        print("Cannot open camera")
        running = False
        return

    while running:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            running = False
            return

        if time.time() - last_time >= interval:
            with frame_lock: 
                cap_frame = frame.copy()

        last_time = time.time()

# Extracts text from an image
def ocr_predict(ocr):
    global cap_frame, ocr_result, running
    print("Start ocr prediction")

    while running:
        # Getting the frame
        with frame_lock:
            if cap_frame == None:
                continue

            frame = cap_frame.copy()

        result = ocr.predict(cap_frame)

        with ocr_lock:
            ocr_result = result    


# Translates text
def translate(tokenizer, model):
    global ocr_result, running
    print("Start translation")

    while running:
        with ocr_lock:
            if ocr_result == None:
                continue

            texts = ocr_result[0]["rec_texts"]

            # Translating
            for i, text in enumerate(texts):
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                translated_tokens = model.generate(
                    **inputs, 
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"), 
                    max_length=50,
                    )

                # Translating the texts 
                texts[i] = tokenizer.batch_decode(
                    translated_tokens, 
                    skip_special_tokens=True)[0]

            # Saving the result
            for res in ocr_result:  
                res.save_to_img(os.path.join("output", "ocr_result.jpg"))  
                res.save_to_json(os.path.join("output", "ocr_result.json"))

        time.sleep(0.1)


# Running everything
with ThreadPoolExecutor(max_workers=3) as executor:
    print("Start execution")
    capture_future = executor.submit(video_capture, cap, 1)
    predict_future = executor.submit(ocr_predict, ocr)
    translate_future = executor.submit(translate, tokenizer, model)

        
## When everything done, release the capture
cap.release()
cv.destroyAllWindows()


