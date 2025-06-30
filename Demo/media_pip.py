import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque

# ——————————————————————————————————————————————
# 1. Rotation & angle utilities (from your preprocessing script)
# ——————————————————————————————————————————————
import math

def unit_vector(v):
    return v / np.linalg.norm(v)

def angle_between(v1, v2):
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0.0
    v1_u, v2_u = unit_vector(v1), unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotation_matrix(axis, theta):
    if np.abs(axis).sum() < 1e-6 or abs(theta) < 1e-6:
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)],
        [2*(bc-ad),   aa+cc-bb-dd, 2*(cd+ab)],
        [2*(bd+ac),   2*(cd-ab),   aa+dd-bb-cc]
    ])

# ——————————————————————————————————————————————
# 2. NTU→MediaPipe mapping & class names
# ——————————————————————————————————————————————
# Index: NTU25 Name ---- MediaPipe33 Index


# Corrected mapping from MediaPipe to NTU-RGB+D
ntu2mp = [
    24,  # 1. Base of the spine -> right_hip (approx.)
    12,  # 2. Middle of the spine -> right_shoulder (approx. mid-spine)
    12,  # 3. Neck -> right_shoulder (no direct neck joint, shoulder is a proxy)
    0,   # 4. Head -> nose
    11,  # 5. Left shoulder -> left_shoulder
    13,  # 6. Left elbow -> left_elbow
    15,  # 7. Left wrist -> left_wrist
    19,  # 8. Left hand -> left_index (hand endpoint)
    12,  # 9. Right shoulder -> right_shoulder
    14,  # 10. Right elbow -> right_elbow
    16,  # 11. Right wrist -> right_wrist
    20,  # 12. Right hand -> right_index (hand endpoint)
    23,  # 13. Left hip -> left_hip
    25,  # 14. Left knee -> left_knee
    27,  # 15. Left ankle -> left_ankle
    31,  # 16. Left foot -> left_heel
    24,  # 17. Right hip -> right_hip
    26,  # 18. Right knee -> right_knee
    28,  # 19. Right ankle -> right_ankle
    32,  # 20. Right foot -> right_heel
    11,  # 21. Spine -> left_shoulder (another spine approximation)
    17,  # 22. Tip of the left hand -> left_pinky
    21,  # 23. Left thumb -> left_thumb
    18,  # 24. Tip of the right hand -> right_pinky
    22,  # 25. Right thumb -> right_thumb
]

CLASS_NAMES = [
    'drink water','eat meal/snack','brushing teeth','brushing hair','drop','pickup',
    'throw','sitting down','standing up (from sitting position)','clapping','reading',
    'writing','tear up paper','wear jacket','take off jacket','wear a shoe',
    'take off a shoe','wear on glasses','take off glasses','put on a hat/cap',
    'take off a hat/cap','cheer up','hand waving','kicking something',
    'reach into pocket','hopping (one foot jumping)','jump up',
    'make a phone call/answer phone','playing with phone/tablet',
    'typing on a keyboard','pointing to something with finger','taking a selfie',
    'check time (from watch)','rub two hands together','nod head/bow','shake head',
    'wipe face','salute','put the palms together','cross hands in front (say stop)',
    'sneeze/cough','staggering','falling','touch head (headache)',
    'touch chest (chest pain)','touch back (back pain)','touch neck (neck pain)',
    'nausea or vomiting condition','use a fan (with hand or paper)/feeling warm',
    'punching/slapping other person','kicking other person','pushing other person',
    'pat on back of other person','point finger at the other person','hugging other person',
    'giving something to other person','touch other person\'s pocket','handshaking',
    'walking towards each other','walking apart from each other'
]

# ——————————————————————————————————————————————
# 3. Real-time normalization function
# ——————————————————————————————————————————————
def normalize_window(win: np.ndarray) -> np.ndarray:
    """
    win: (T, 25, 3) raw landmarks
    returns: normalized, scaled, and axis-aligned landmarks, same shape
    """
    T, V, _ = win.shape
    # center on joint 1 (SpineMid)
    center = win[:, 1, :].copy()            # (T,3)
    win = win - center[:, None, :]
    # scale by average torso length (joints 0 & 1)
    bottom = win[:, 0, :]
    top    = win[:, 1, :]
    torso_len = np.linalg.norm(top - bottom, axis=1).mean()
    win = win / (torso_len + 1e-6)
    # first rotation: hip–spine → z axis
    vec = top[0] - bottom[0]
    axis = np.cross(vec, [0,0,1])
    theta = angle_between(vec, [0,0,1])
    Rz = rotation_matrix(axis, theta)
    # second rotation: shoulder–shoulder → x axis
    r_sh, l_sh = win[0, 8], win[0, 4]
    axis2 = np.cross(r_sh - l_sh, [1,0,0])
    th2   = angle_between(r_sh - l_sh, [1,0,0])
    Rx    = rotation_matrix(axis2, th2)
    # apply both rotations
    for t in range(T):
        for v in range(V):
            win[t,v] = Rx.dot(Rz.dot(win[t,v]))
    return win

# ——————————————————————————————————————————————
# 4. Load your trained PoseLSTM model
# ——————————————————————————————————————————————
from lstm_model import PoseLSTM  # ensure this module is on your PYTHONPATH
import imageio                   # ← new

gif_path = 'output.gif'
fps = 10
writer = imageio.get_writer(gif_path, mode='I', duration=20)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = PoseLSTM(in_dim=75, hidden_dim=128, num_layers=2,
                  num_classes=len(CLASS_NAMES), dropout=0.3)
model.load_state_dict(torch.load('checkpoints/pose_lstm.pth', map_location=device))
model.to(device).eval()

# ——————————————————————————————————————————————
# 5. Setup MediaPipe GPU-accelerated Pose   
# ——————————————————————————————————————————————
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ——————————————————————————————————————————————
# 6. Prepare sliding buffer
# ——————————————————————————————————————————————
T = 64
buffer = deque(maxlen=T)

# ——————————————————————————————————————————————
# 7. Video capture & inference loop
# ——————————————————————————————————————————————
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("walk.mp4")  # replace with your video file or webcam index

if not cap.isOpened():
    raise IOError("Cannot open webcam")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = pose.process(img)

        img.flags.writeable = True
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                out,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,128,255), thickness=2)
            )

            lm = results.pose_landmarks.landmark
            raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            buffer.append(raw)

            if len(buffer) == T:
                window33 = np.stack(buffer, axis=0)
                window25 = window33[:, ntu2mp, :]
                norm_win = normalize_window(window25)

                x = np.transpose(norm_win, (2,0,1))[None,...]
                x_tensor = torch.from_numpy(x).float().to(device)
                with torch.no_grad():
                    logits = model(x_tensor)
                    probs  = torch.softmax(logits, dim=1)[0]
                    idx    = probs.argmax().item()
                    conf   = probs[idx].item()
                    label  = CLASS_NAMES[idx]

                cv2.putText(out, f'{label} ({conf:.2f})',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)

        # write frame to GIF (convert BGR→RGB)
        rgb_frame = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

        cv2.imshow('Action Recognition Demo', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # clean up everything
    writer.close()          # ← finalize the GIF
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

print(f"Saved action-recognition animation to {gif_path}")
