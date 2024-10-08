import time
import torch # pytorch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 # opencv
from util import *
import argparse
import os
from darknet import Darknet
import pickle as pkl
import random
import dlib # 얼굴 인식 및 랜드마크 검출
import winsound # 경고음
import mediapipe as mp # 손 인식


# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
try:
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
    print("MediaPipe Hands 초기화 성공")
except Exception as e:
    print(f"MediaPipe Hands 초기화 실패: {e}")

# MediaPipe pose 초기화
mp_pose = mp.solutions.pose
try:
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print("MediaPipe Pose 초기화 성공")
except Exception as e:
    print(f"MediaPipe Pose 초기화 실패: {e}")
mp_draw = mp.solutions.drawing_utils


# 운전자 얼굴 정보/탑승자 손 정보를 저장할 변수
driver_face = None
passenger_hand = None

# 졸음 인식 설정
# 눈과 입의 랜드마크 인덱스 정의
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))
MOUTH = list(range(48, 68))

frame_width = 640
frame_height = 480

title_name = 'Drowsiness and Object Detection'

# 얼굴 검출을 위한 Haar Cascade 경로 설정
face_cascade_name = './haarcascade_frontalface_alt.xml'  # -- 본인 환경에 맞게 변경할 것(상대 경로)
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
# 얼굴 랜드마크 검출을 위한 dlib 예측 모델 파일 경로 설정
# 얼굴 랜드마크 예측 모델
predictor_file = './shape_predictor_68_face_landmarks.dat'  # -- 본인 환경에 맞게 변경할 것(상대 경로)
predictor = dlib.shape_predictor(predictor_file)

status = 'Awake'
number_closed = 0
min_EAR = 0.25
closed_limit = 10  # -- 눈 감김이 10번 이상일 경우 졸음으로 간주
yawn_count = 0  # 하품 횟수 초기화
yawn_limit = 3  # 하품이 3번 감지되면 알람
show_frame = None
sign = None
color = (0, 255, 0)
last_alarm_time = 0
alarm_interval = 10  # 알람 사이의 최소 시간 간격 (초)

# 캘리브레이션 관련 변수
calibration_frames = 20 # 프레임 수 줄일수록 캘리브레이션 시간 줄어듦
calibration_counter = 0
total_EAR = 0
calibrated = False

# EAR (Eye Aspect Ratio) 계산 함수
# 눈의 랜드마크 간의 거리 계산
def getEAR(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

# MAR (Mouth Aspect Ratio) 계산 함수
# 입의 랜드마크 간의 거리 계산
def getMAR(points):
    A = np.linalg.norm(points[13] - points[19])
    B = np.linalg.norm(points[14] - points[18])
    C = np.linalg.norm(points[15] - points[17])
    D = np.linalg.norm(points[12] - points[16])
    return (A + B + C) / (2.0 * D)

# 졸음 감지
def detect_drowsiness(image):
    global driver_face, last_alarm_time, number_closed, yawn_count, color, show_frame, sign, status
    global calibration_counter, total_EAR, calibrated, min_EAR
    global current_hand_speed, attack_warning_count
    global last_hand_position, last_hand_time
    global current_foot_speed, foot_attack_warning_count
    global last_foot_position, last_foot_time

    # 이미지를 그레이스케일로 변환하고 히스토그램 평활화
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    
    # 얼굴 인식 실패 시 처리
    if len(faces) == 0:
        cv2.putText(show_frame, "No face detected", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return
    
    # 왼쪽 맨 앞의 얼굴을 운전자로 가정
    faces = sorted(faces, key=lambda x: (-x[0], -x[1]))
    driver_face = faces[0]
    
    for i, (x, y, w, h) in enumerate(faces):
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        points = np.matrix([[p.x, p.y] for p in predictor(image, rect).parts()])

        if i == 0: # 운전자
            cv2.putText(show_frame, "Driver", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 졸음 감지 로직 적용
            show_parts = points[EYES]
            right_eye_EAR = getEAR(points[RIGHT_EYE])
            left_eye_EAR = getEAR(points[LEFT_EYE])
            mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2 

            if not calibrated:
                # 캘리브레이션 단계
                total_EAR += mean_eye_EAR
                calibration_counter += 1
                if calibration_counter >= calibration_frames:
                    avg_EAR = total_EAR / calibration_counter
                    min_EAR = avg_EAR * 0.85  # 평균 EAR의 70%를 임계값으로 설정
                    calibrated = True
                    print(f"Calibration completed. min_EAR set to: {min_EAR:.3f}")
                color = (255, 255, 0)
                status = 'Calibrating'
            else:
                if mean_eye_EAR > min_EAR:
                    color = (0, 255, 0)
                    status = 'Awake'
                    number_closed -= 1
                    if number_closed < 0:
                        number_closed = 0
                else:
                    color = (0, 0, 255)
                    status = 'Sleep'
                    number_closed += 1
                            
                sign = 'Sleep count : ' + str(number_closed) + ' / ' + str(closed_limit)

            # 하품 인식
            mouth_points = points[MOUTH]
            mouth_MAR = getMAR(mouth_points)
            min_MAR = 0.6  # 하품으로 간주할 최소 MAR 값
            
            # 하품 인식 조건: 입이 벌어지고 눈이 감긴 상태
            if mouth_MAR > min_MAR and mean_eye_EAR < min_EAR:
                yawn_count += 1
                if yawn_count >= yawn_limit:
                    current_time = time.time()
                    if current_time - last_alarm_time > alarm_interval:
                        print("Yawning detected")
                        winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                        last_alarm_time = current_time
                    yawn_count = 0  # 하품 횟수 초기화

            # 졸음 확정시 알람 설정
            if number_closed > closed_limit:
                current_time = time.time()
                if current_time - last_alarm_time > alarm_interval:
                    print("Alarm condition met")
                    winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                    last_alarm_time = current_time
                number_closed = 0  # 눈 감김 횟수 초기화

        else: # 동승자
            cv2.putText(show_frame, "Passenger", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    # 화면에 상태, 손 속도, 하품 횟수 표시
    cv2.putText(show_frame, status, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
    if attack_warning_count > 0:
        cv2.putText(show_frame, "DANGER", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    if calibrated:
        cv2.putText(show_frame, sign, (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(show_frame, f"Yawn count: {yawn_count} / {yawn_limit}", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 인수 파싱 함수
def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default="416", type=str)

    return parser.parse_args()

# 이미지 전처리 함수
def prep_image(img, inp_dim):
    orig_im = img.copy()
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

args = arg_parse()
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("./data/coco.names")  # YOLO 클래스 이름 파일 경로(상대 경로)

# 신경망 설정
print("Loading.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Success")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# GPU가 사용 가능하면 모델을 GPU에 올립니다
if CUDA:
    model.cuda()

# 모델을 평가 모드로 설정
model.eval()

# 경계 상자 색상 로드
colors = pkl.load(open("pallete", "rb"))

# 웹캠 열기
cap = cv2.VideoCapture(0)
time.sleep(2.0)
if not cap.isOpened():
    print('Could not open video')
    exit(0)

def write(x, results):
    c1 = tuple(map(int, x[1:3]))
    c2 = tuple(map(int, x[3:5]))
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


# 손 감지를 위한 Haar Cascade 로드
hand_cascade = cv2.CascadeClassifier('./haarcascade_hand.xml')

# 손 위치 추적을 위한 변수
last_hand_position = None
last_hand_time = None
hand_speed_threshold = 500 # 손 속도 임계값(픽셀/초)
attack_warning_count = 0
attack_warning_limit = 2  # 공격으로 간주할 연속 감지 횟수
current_hand_speed = 0

# 발 위치 추적을 위한 변수
foot_attack_warning_count = 0
foot_attack_warning_limit = 2  # 공격으로 간주할 연속 감지 횟수
current_foot_speed = 0
last_foot_position = None
last_foot_time = None
foot_speed_threshold = 500  # 발 속도 임계값(픽셀/초)

# 손 감지 및 움직임 추적 함수
def detect_hand_movement(frame):
    global driver_face, last_alarm_time, current_hand_speed, attack_warning_count
    global last_hand_position, last_hand_time

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_time = time.time()

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                hand_center = (
                    int((x_min + x_max) / 2 * frame.shape[1]),
                    int((y_min + y_max) / 2 * frame.shape[0])
                )

                # 모든 손을 그리기
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 손에 번호 매기기 (한 번만)
                hand_label = f"PassengerHand{idx+1}"
                cv2.putText(frame, hand_label, 
                            (int(x_min * frame.shape[1]), int(y_min * frame.shape[0]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                # 손의 속도와 방향 계산
                if last_hand_position is not None and last_hand_time is not None:
                    time_diff = current_time - last_hand_time
                    distance = np.sqrt((hand_center[0] - last_hand_position[0])**2 + 
                                       (hand_center[1] - last_hand_position[1])**2)
                    speed = distance / time_diff if time_diff > 0 else 0
                    current_hand_speed = speed

                    # 손의 이동 방향 계산
                    direction = (hand_center[0] - last_hand_position[0], 
                                 hand_center[1] - last_hand_position[1])

                    # 운전자 얼굴 위치 확인 및 위험 감지
                    if driver_face is not None:
                        dx, dy, dw, dh = driver_face
                        driver_center = (dx + dw // 2, dy + dh // 2)

                        # 손이 운전자 방향으로 움직이는지 확인
                        to_driver = (driver_center[0] - hand_center[0], 
                                     driver_center[1] - hand_center[1])
                        
                        # 내적을 사용하여 방향 유사성 확인
                        direction_norm = np.linalg.norm(direction)
                        to_driver_norm = np.linalg.norm(to_driver)
                        if direction_norm != 0 and to_driver_norm != 0:
                            direction_similarity = (direction[0] * to_driver[0] + direction[1] * to_driver[1]) / (direction_norm * to_driver_norm)

                            # 속도가 임계값을 초과하고, 운전자 방향으로 움직이는 경우
                            if speed > hand_speed_threshold and direction_similarity > 0.7:
                                attack_warning_count += 1
                                print(f"Warning count: {attack_warning_count}")  # 디버깅용
                                if attack_warning_count >= attack_warning_limit:
                                    current_time = time.time()
                                    if current_time - last_alarm_time > alarm_interval:
                                        print(f"Warning: Potential threat from {hand_label}! Speed: {speed:.2f} pixels/second")
                                        winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                                        last_alarm_time = current_time
                                    attack_warning_count = 0
                            else:
                                attack_warning_count = max(0, attack_warning_count - 1)

                last_hand_position = hand_center
                last_hand_time = current_time

            print(f"Detected {len(results.multi_hand_landmarks)} passenger hands")
        else:
            print("No hands detected")
            # 손이 감지되지 않은 경우, 초기화
            last_hand_position = None
            last_hand_time = None

    except Exception as e:
        print(f"Error in detect_hand_movement: {e}")

    return frame


def detect_foot_movement(frame):
    global driver_face, last_alarm_time, current_foot_speed, foot_attack_warning_count
    global last_foot_position, last_foot_time

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    current_time = time.time()

    if results.pose_landmarks:
        # 오른쪽 발목 (landmark 28)과 왼쪽 발목 (landmark 27) 사용
        right_ankle = results.pose_landmarks.landmark[28]
        left_ankle = results.pose_landmarks.landmark[27]

        # 두 발의 중심점 계산
        foot_center = (
            int((right_ankle.x + left_ankle.x) / 2 * frame.shape[1]),
            int((right_ankle.y + left_ankle.y) / 2 * frame.shape[0])
        )

        # 발 위치 표시
        cv2.circle(frame, foot_center, 5, (0, 255, 255), -1)
        cv2.putText(frame, "Feet", (foot_center[0] - 10, foot_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 발의 속도와 방향 계산
        if last_foot_position is not None and last_foot_time is not None:
            time_diff = current_time - last_foot_time
            distance = np.sqrt((foot_center[0] - last_foot_position[0])**2 + 
                               (foot_center[1] - last_foot_position[1])**2)
            speed = distance / time_diff if time_diff > 0 else 0
            current_foot_speed = speed

            # 발의 이동 방향 계산
            direction = (foot_center[0] - last_foot_position[0], 
                         foot_center[1] - last_foot_position[1])

            # 운전자 얼굴 위치 확인 및 위험 감지
            if driver_face is not None:
                dx, dy, dw, dh = driver_face
                driver_center = (dx + dw // 2, dy + dh // 2)

                # 발이 운전자 방향으로 움직이는지 확인
                to_driver = (driver_center[0] - foot_center[0], 
                             driver_center[1] - foot_center[1])
                
                # 내적을 사용하여 방향 유사성 확인
                direction_norm = np.linalg.norm(direction)
                to_driver_norm = np.linalg.norm(to_driver)
                if direction_norm != 0 and to_driver_norm != 0:
                    direction_similarity = (direction[0] * to_driver[0] + direction[1] * to_driver[1]) / (direction_norm * to_driver_norm)

                    # 속도가 임계값을 초과하고, 운전자 방향으로 움직이는 경우
                    if speed > foot_speed_threshold and direction_similarity > 0.7:  # 0.7은 약 45도 이내의 각도
                        foot_attack_warning_count += 1
                        if foot_attack_warning_count >= foot_attack_warning_limit:
                            if current_time - last_alarm_time > alarm_interval:
                                print(f"Warning: Potential threat from passenger's feet! Speed: {speed:.2f} pixels/second")
                                winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                                last_alarm_time = current_time
                            foot_attack_warning_count = 0
                    else:
                        foot_attack_warning_count = max(0, foot_attack_warning_count - 1)

        last_foot_position = foot_center
        last_foot_time = current_time

        print(f"Detected passenger's feet")
    else:
        print("No feet detected")
        # 발이 감지되지 않은 경우, 초기화
        last_foot_position = None
        last_foot_time = None
        current_foot_speed = 0

    return frame

# 객체 감지 결과 처리 함수
def process_detected_objects(output, frame):
    global driver_face, last_alarm_time

    for obj in output:
        cls = int(obj[-1])
        if classes[cls] in ['remote', 'cell phone']:
            x1, y1, x2, y2 = obj[1:5]
            if driver_face is not None:
                dx, dy, dw, dh = driver_face
                if (x1 > dx and x1 < dx+dw) or (x2 > dx and x2 < dx+dw):
                    current_time = time.time()
                    if current_time - last_alarm_time > alarm_interval:
                        print("Smart phone detected near driver! Please focus.")
                        winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                        last_alarm_time = current_time

    list(map(lambda x: write(x, frame), output))

# 메인 루프
frame_count = 0
while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break

        frame_count += 1
        print(f"\nProcessing frame {frame_count}")
        print(f"Frame shape: {frame.shape}")
        
        show_frame = frame.copy()
        detect_drowsiness(show_frame)
        show_frame = detect_hand_movement(show_frame)
        show_frame = detect_foot_movement(show_frame)

        # 매 프레임마다 연산(YOLO, 얼굴감지, 손감지) 수행 -> 실시간 처리에 어려움 존재하므로, 성능 모니터링하고 필요할 경우 추가
        # if time.time() - last_object_detection_time > object_detection_interval:
        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        # YOLO 모델을 사용하여 객체 탐지
        with torch.no_grad():
            output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

        if isinstance(output, int):
            cv2.imshow(title_name, show_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  # 객체가 탐지되지 않은 경우 다음 프레임으로
        
        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        # 스마트폰 탐지 시 알람 설정(운전자 근처에 있을 때만)
        for obj in output:
            cls = int (obj[-1])
            if classes[cls] in ['remote', 'cell phone']:
                x1, y1, x2, y2 = obj[1:5]
                if driver_face is not None:
                    dx, dy, dw, dh = driver_face
                    # 스마트폰이 운전자 얼굴 근처에 있는지 확인
                    if (x1 > dx and x1 < dx+dw) or (x2 > dx and x2 < dx+dw):
                        current_time = time.time()
                        if current_time - last_alarm_time > alarm_interval:
                            print("smart phone detected near driver! plz focus.")
                            winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                            last_alarm_time = current_time

        list(map(lambda x: write(x, orig_im), output))

        show_frame = detect_hand_movement(show_frame)
        show_frame = detect_foot_movement(show_frame)

        # 화면에 결과 표시
        # 화면에 상태, 손 속도, 발 속도, 하품 횟수 표시
        cv2.putText(show_frame, status, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        if attack_warning_count > 0 or foot_attack_warning_count > 0:
            cv2.putText(show_frame, "DANGER", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        cv2.putText(show_frame, f"Hand Speed: {current_hand_speed:.2f} px/s", (10, frame_height - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(show_frame, f"Foot Speed: {current_foot_speed:.2f} px/s", (10, frame_height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if calibrated:
            cv2.putText(show_frame, sign, (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(show_frame, f"Yawn count: {yawn_count} / {yawn_limit}", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    except Exception as e:
        print(f"오류 발생: {e}")
        break

    cv2.imshow(title_name, show_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()