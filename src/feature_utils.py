import cv2
import mediapipe as mp
import numpy as np


# MediaPipe landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
MOUTH_LEFT_IDX = 78
MOUTH_RIGHT_IDX = 308

NOSE_TIP_IDX = 1
CHIN_IDX = 152
LEFT_FACE_IDX = 234
RIGHT_FACE_IDX = 454

FEATURE_NAMES = [
    "left_ear",
    "right_ear",
    "avg_ear",
    "mouth_open_ratio",
    "eye_asymmetry",
    "face_height_width_ratio",
    "nose_center_x",
]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_point(face_landmarks, idx, image_w, image_h):
    lm = face_landmarks.landmark[idx]
    return np.array([lm.x * image_w, lm.y * image_h], dtype=np.float32)


def eye_aspect_ratio(eye_points):
    """
    eye_points order:
    [p1, p2, p3, p4, p5, p6]
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1, p2, p3, p4, p5, p6 = eye_points
    horizontal = euclidean(p1, p4)
    if horizontal < 1e-6:
        return 0.0

    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)


def mouth_open_ratio(upper_lip, lower_lip, mouth_left, mouth_right):
    mouth_width = euclidean(mouth_left, mouth_right)
    if mouth_width < 1e-6:
        return 0.0

    mouth_open = euclidean(upper_lip, lower_lip)
    return float(mouth_open / mouth_width)


def extract_fatigue_features_from_landmarks(face_landmarks, image_w, image_h):
    left_eye = [get_point(face_landmarks, idx, image_w, image_h) for idx in LEFT_EYE_IDX]
    right_eye = [get_point(face_landmarks, idx, image_w, image_h) for idx in RIGHT_EYE_IDX]

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    eye_asymmetry = abs(left_ear - right_ear)

    upper_lip = get_point(face_landmarks, UPPER_LIP_IDX, image_w, image_h)
    lower_lip = get_point(face_landmarks, LOWER_LIP_IDX, image_w, image_h)
    mouth_left = get_point(face_landmarks, MOUTH_LEFT_IDX, image_w, image_h)
    mouth_right = get_point(face_landmarks, MOUTH_RIGHT_IDX, image_w, image_h)
    mar = mouth_open_ratio(upper_lip, lower_lip, mouth_left, mouth_right)

    nose_tip = get_point(face_landmarks, NOSE_TIP_IDX, image_w, image_h)
    chin = get_point(face_landmarks, CHIN_IDX, image_w, image_h)
    left_face = get_point(face_landmarks, LEFT_FACE_IDX, image_w, image_h)
    right_face = get_point(face_landmarks, RIGHT_FACE_IDX, image_w, image_h)

    face_height = euclidean(nose_tip, chin)
    face_width = euclidean(left_face, right_face)
    face_height_width_ratio = float(face_height / face_width) if face_width > 1e-6 else 0.0

    nose_center_x = float(nose_tip[0] / image_w)

    feature_vector = np.array(
        [
            left_ear,
            right_ear,
            avg_ear,
            mar,
            eye_asymmetry,
            face_height_width_ratio,
            nose_center_x,
        ],
        dtype=np.float32,
    )

    return feature_vector


def create_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_features_from_frame(frame_bgr, face_mesh):
    image_h, image_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    return extract_fatigue_features_from_landmarks(face_landmarks, image_w, image_h)


def extract_sequence_features_from_video(video_path, frame_stride=2, max_frames=None):
    sequence_features = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_idx = 0

    with create_face_mesh() as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                features = extract_features_from_frame(frame, face_mesh)
                if features is not None:
                    sequence_features.append(features)

                if max_frames is not None and len(sequence_features) >= max_frames:
                    break

            frame_idx += 1

    cap.release()

    if len(sequence_features) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    return np.stack(sequence_features).astype(np.float32)
