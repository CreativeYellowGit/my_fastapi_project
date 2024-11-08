# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
import os, base64, uuid, time
import cv2
import numpy as np
from processing import apply_processing_steps
from utils import str_to_bool

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/process/")
async def process_image(
    file: UploadFile = File(...), 
    grayscale_enabled: str = Form("true"), 
    clahe_enabled: str = Form("true"),
    noise_removal_enabled: str = Form("true"), 
    binarize_enabled: str = Form("true")
    ):
    print('시작------------------------------------------------------------------------')
    allowed_extensions = {"png", "jpg", "jpeg"}
    filename = file.filename
    extension = filename.split(".")[-1].lower()
    
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=".png, .jpg, .jpeg 만 가능합니다.")
    
    grayscale_enabled = str_to_bool(grayscale_enabled)
    clahe_enabled = str_to_bool(clahe_enabled)
    noise_removal_enabled = str_to_bool(noise_removal_enabled)
    binarize_enabled = str_to_bool(binarize_enabled)

    # 파일을 읽어서 OpenCV 이미지로 변환
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 전체 처리 시간 측정 시작
    total_start_time = time.time()
    
    # 전처리 단계별로 옵션에 따라 조건부 적용
    image, prefix_name = apply_processing_steps(image, grayscale_enabled, clahe_enabled, noise_removal_enabled, binarize_enabled)

    # 전체 처리 시간 측정 시작
    process_total_time = (time.time() - total_start_time) * 1000
    print(f'전처리 적용 전체 처리 시간: {process_total_time:.2f} ms')

    # 전처리된 이미지를 입력 파일의 확장자 형식으로 인코딩 및 저장
    _, img_encoded = cv2.imencode(f'.{extension}', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "..", "changed_images")
    os.makedirs(output_dir, exist_ok=True)  # 폴더가 없으면 생성

    output_filename = os.path.join(output_dir, f"{prefix_name}{str(uuid.uuid4())}.{extension}")

    with open(output_filename, "wb") as f:
        f.write(img_encoded)

    # 전체 처리 시간 출력
    total_time = (time.time() - total_start_time) * 1000
    print(f'저장까지 완성한 전체 처리 시간: {total_time:.2f} ms')

    print('끝@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    return {
        "filename": file.filename,
        "content": img_base64,
        "message": f"Processed image saved as {output_filename}",
        "total_processing_time_ms": f"{total_time:.2f} ms"
    }
