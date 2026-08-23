from paddleocr import PaddleOCR
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import cv2 as cv
import time
import threading
import os # Only needed for testing purposes

# Make sure PaddlePaddle is installed and on version 3.2.0
# Uses PP-OCRv6 models by default
# ocr = PaddleOCR(
#     use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
#     use_doc_unwarping=False, # Disables text image rectification model via this parameter
#     use_textline_orientation=False, # Disables text line orientation classification model via this parameter
# )
# ocr = PaddleOCR(lang="en") # Uses English model by specifying language parameter
# ocr = PaddleOCR(ocr_version="PP-OCRv5") # Switches to PP-OCRv5 version via ocr_version parameter
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # Switches to PP-OCRv4 version via ocr_version parameter
# ocr = PaddleOCR(device="gpu") # Enables GPU acceleration for model inference via device parameter

# Set Up
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

# Reads the current frame of the video
def video_capture(cap, interval):
    last_time = time.time()

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return

    if time.time() - last_time >= interval:
        cap_frame = frame

# Extracts text from an image
def ocr_predict(ocr):
    while True:
        ocr_result = ocr.predict(cap_frame)


# Translates text
def translate(tokenizer, model):
    while True:
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

## When everything done, release the capture
cap.release()
cv.destroyAllWindows()


