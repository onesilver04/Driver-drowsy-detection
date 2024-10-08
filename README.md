<div align="center">
<h2> ⭐ CODE-IT 2024 Summer Project ⭐</h2>
🚨 운전자 졸음 감지 및 탑승자 위협 감지 프로젝트 🚗
</div>

## 목차
  - [개요](#개요) 
  - [Project 구조](#Project-구조)
  - [Requirements](#Requirements)
  - [Help](#Help)

## 개요
- 프로젝트 이름: 운전자 졸음 감지 및 탑승자 위협 감지 프로젝트
- 프로젝트 지속기간: 2024.07~2024.08
- 👪 팀원&역할분담
>   |한은정|opencv, YOLOv3 laod|
>
>  |유은서|mediapipe load|

***

## Project 구조

```
├── data
│   └── coco.names
├── download
|   |   alarm.wav
|   |   haarcascade_frontalface_alt.xml
|   |   haarcascade_hand.xml
|   |   pallete
│   |   reveil_auto.wav
│   └── yolov3.cfg
├── darknet.py
├── main.py
└── util.py

```

## Requirements
* Anaconda 가상환경 활성화
  * Install google mediapipe:
    ```shell
    pip install torch mediapipe opencv dlib
    ```
* directory
  * download 폴더의 모든 파일들과 yolov3.weights, shape_predictor_68_face_landmarks.dat은 main.py 같은 디렉토리에 다운로드

## Help
>mediapipe를 통해 손을 더 정확히 인식할 수 있습니다.
>
>download
> - yolov3.weights, shape_predictor_68_face_landmarks.dat 파일은 용량 문제로 깃에 첨부하지 못했습니다.
> - github 'YOLO_v3_tutorial_from_scratch' 에 관련 파일이 모두 존재! 필요하다면 여기서 다운로드 가능


## 교수님 피드백
- 한 가지 기능에 좀 더 집중적으로 파고 들었다면 좋았을 듯
- 졸업작품 시에 이용하면 좋을 듯
