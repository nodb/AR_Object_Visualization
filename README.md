# AR_Object_Visualization 📷

## ✍ 소개
컴퓨터비전 카메라 캘리브레이션 및 3D 하트 AR 프로그램 입니다.

## 🗂️ 사용 기술
- Python
- OpenCV

## 💻 기능
- 작동 순서 : 캘리브레이션 모드 진입 → 캘리브레이션 측정 완료(콘솔 출력) → 왜곡 보정 및 3D 하트 AR 모드 진입 → 왜곡 보정 여부 지정

### 1. 캘리브레이션 모드
클리브레이션 측정 방법
1. 원하는 화면에서 Space bar를 클릭 후 Enter 클릭(이미지 선택)
2. 1번을 최소 3번 반복
3. Esc를 클릭하여 캘리브레이션 측정(이미지 선택) 종료

화면 기능
- Nselect : 캘리브레이션(정밀도 측정)을 위해 캡쳐한 이미지 개수

### 2. 콘솔 출력
카메라 보정 결과
- 선택된 이미지 수
- RMS 오차
- 카메라 행렬 (K)
- 왜곡 계수 (k1, k2, p1, p2, k3, ...)

### 3. 왜곡 보정 및 3D 하트 AR 모드
왜곡 보정 방법
1. Tab을 클릭하여 왜곡 보정 적용/미적용(토글)
2. ESC를 클릭하여 프로그램 종료

화면 기능
- X, Y, Z : 카메라 위치 출력
- Original : 보정 미적용
- Distortion Correction : 보정 적용


## 📸 렌즈 왜곡 보정 결과 데모
- 카메라 : 갤럭시S22U 0.6배율
  
![image](https://github.com/nodb/AR_Object_Visualization/assets/71473708/61c0ced4-b29b-4a4d-b89d-de597650dcf0)


## 시연
- [동영상 시연 1](https://youtu.be/oAAc2vTq_d4)

<img src="https://github.com/nodb/AR_Object_Visualization/assets/71473708/1b7e7856-1444-474d-a90b-bed31f087d43" width="400">
<img src="https://github.com/nodb/AR_Object_Visualization/assets/71473708/8e57405b-2c9a-4fef-970b-7cbd7a14e528" width="400">

- [동영상 시연 2](https://youtu.be/jz6TujvT_6I)

<img src="https://github.com/nodb/AR_Object_Visualization/assets/71473708/97cec2c5-1416-465c-a260-33d7370880f5" width="400">
<img src="https://github.com/nodb/AR_Object_Visualization/assets/71473708/e7096265-35c1-4a37-8755-ff0a2bdf1956" width="400">
