# 2022 의료머신러닝 기말 레포트

### 캡슐 내시경 학습용 이미지 데이터셋을 활용한 캡슐 내시경 병변 검출 AI 개발

![image](https://user-images.githubusercontent.com/107654296/174196780-eb4ea782-1094-4a7e-9175-6651d7bbd06e.png)


- 데이터 : 의료 현장에서 촬영되는 캡슐 내시경 이미지 (출처 : 한국지능정보사회진흥원, 양산부산대학교병원)
- 데이터셋 크기
  -	Train	:	62,622	
  -	Test : 20,874
- 클래스정보
  - 01	Ulcer (궤양)
  - 02	Mass (종괴)
  - 04	Lymph (림프부종)
  - 05	Bleeding (출혈)
- 샘플 이미지 및 데이터 분포

  ![image](https://user-images.githubusercontent.com/107654296/174198039-f4ebdcfa-354b-4565-b3ac-760d660ada73.png)


- 모델 : YOLOv5 : https://github.com/ultralytics/yolov5
- 학습결과 : https://github.com/jaeseonYS/2022MedicalMachineLearning/releases

  ![image](https://user-images.githubusercontent.com/107654296/174196663-3094d351-83b6-4c3c-812a-26f51cdc8bee.png)

- mAP(IoU:0.5) = 0.8535
