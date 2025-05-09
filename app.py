import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

model = YOLO('best.pt')

st.title("Number Plate Detection App 🚗")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

col1, col2 = st.columns([1, 2])  

if uploaded_file is not None:

    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    tfile.write(uploaded_file.read())
    tfile.close()

    
    results = model.predict(source=tfile.name, save=False, conf=0.5)

    with col2:
        
        result_img = results[0].plot()
        result_pil = Image.fromarray(result_img[..., ::-1])  
        st.image(result_pil, caption='Detected Number Plate', use_container_width=True)

        img_cv = cv2.imread(tfile.name)

        if len(results[0].boxes) > 0:
            st.subheader("Detected Number Plate Coordinates:")
            for i, box in enumerate(results[0].boxes):
                xyxy = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                st.write(f"Class: {model.names[cls_id]} | Confidence: {conf:.2f}")
                st.write(f"Bounding Box: {xyxy}")

                x1, y1, x2, y2 = map(int, xyxy)
                cropped_plate = img_cv[y1:y2, x1:x2]

                if cropped_plate.size != 0:
                    cropped_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)

                    st.image(cropped_rgb, caption=f'Cropped Number Plate {i+1}')
                    extracted_text = pytesseract.image_to_string(cropped_rgb, config='--psm 7')
                    st.subheader(f"Extracted Number Plate Text {i+1}:")
                    st.write(extracted_text.strip())
                else:
                    st.warning(f"Could not crop number plate {i+1}. Check bounding box coordinates.")
        else:
            st.warning("No number plate detected.")
