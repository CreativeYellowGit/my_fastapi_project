# processing.py
import cv2
import numpy as np
import time

# 1. 그레이스케일 변환 함수
def grayscale(image: np.ndarray) -> np.ndarray:
   start_time = time.time()
   print('1 그레이스케일 변환 함수')
   result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   print(f'그레이스케일 변환 시간: {(time.time() - start_time) * 1000:.2f} ms')
   return result

# 2. 히스토그램 평활화 (CLAHE) 적용 함수
def apply_clahe(image: np.ndarray) -> np.ndarray:
   start_time = time.time()
   print('2 히스토그램 평활화 (CLAHE) 적용 함수')
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
   result = clahe.apply(image)
   print(f'히스토그램 평활화 시간: {(time.time() - start_time) * 1000:.2f} ms')
   return result

# 3. 노이즈 제거 및 격자 제거 함수
def remove_noise_and_grid(image: np.ndarray) -> np.ndarray:
   start_time = time.time()
   print('3 노이즈 제거 및 격자 제거 함수')

   blurred = cv2.GaussianBlur(image, (5, 5), 0)
   kernel = np.ones((3, 3), np.uint8)
   grid_removed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

   print(f'노이즈 및 격자 제거 시간: {(time.time() - start_time) * 1000:.2f} ms')
   return grid_removed

# 4. 이진화 함수
def binarize(image: np.ndarray) -> np.ndarray:
   start_time = time.time()
   print('4 이진화 함수')
   _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   print(f'이진화 시간: {(time.time() - start_time) * 1000:.2f} ms')
   return binary

# 전처리 단계별로 조건부 적용 함수
def apply_processing_steps(
   image: np.ndarray, 
   grayscale_enabled: bool, 
   clahe_enabled: bool,
   noise_removal_enabled: bool, 
   binarize_enabled: bool) -> (np.ndarray, str): # type: ignore
   
   prefix_name = "test_two_"

   if grayscale_enabled:
      image = grayscale(image)
      prefix_name += "1_"
   if clahe_enabled:
      image = apply_clahe(image)
      prefix_name += "2_"
   if noise_removal_enabled:
      image = remove_noise_and_grid(image)  # 노이즈 및 격자 제거 함수 호출
      prefix_name += "3_"
   if binarize_enabled:
      image = binarize(image)
      prefix_name += "4_"

   return image, prefix_name
