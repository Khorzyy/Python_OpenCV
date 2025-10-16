import cv2 as cv
import mediapipe as mp
import json
import numpy as np
from function import position_data, calculate_distance, draw_line, overlay_image


def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as file:
        return json.load(file)


def limit_value(val: int, min_val: int, max_val: int) -> int:
    return max(min(val, max_val), min_val)


def initialize_camera(config: dict) -> cv.VideoCapture:
    cap = cv.VideoCapture(config["camera"]["device_id"])
    cap.set(cv.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    if not cap.isOpened():
        raise RuntimeError("Failed to open the webcam.")
    return cap


def load_images(config: dict) -> tuple:
    inner_circle = cv.imread(config["overlay"]["inner_circle_path"], -1)
    outer_circle = cv.imread(config["overlay"]["outer_circle_path"], -1)
    if inner_circle is None or outer_circle is None:
        raise FileNotFoundError("Failed to load one or more overlay images.")
    return inner_circle, outer_circle


def draw_flex_text_box(frame, text, top_left, bottom_right, font_scale=0.7, glow_color=(0, 255, 255)):
    overlay = frame.copy()
    font = cv.FONT_HERSHEY_DUPLEX
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Ensure coordinates are ordered
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    width = x2 - x1
    height = y2 - y1

    # Determine orientation mode
    vertical_mode = width < height * 0.6  # if width is much smaller → vertical mode

    # Draw transparent holographic rectangle
    alpha = 0.3
    cv.rectangle(overlay, (x1, y1), (x2, y2), (50, 255, 200), -1)
    frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Glowing border
    for thickness in range(1, 4):
        cv.rectangle(frame, (x1, y1), (x2, y2), glow_color, thickness)

    # --- TEXT DRAWING ---
    if not vertical_mode:
        # 🧩 Normal horizontal word-wrapping
        words = text.split(" ")
        lines, current_line = [], ""
        for word in words:
            test = (current_line + " " + word).strip()
            size = cv.getTextSize(test, font, font_scale, 1)[0]
            if size[0] > width - 20 and current_line != "":
                lines.append(current_line)
                current_line = word
            else:
                current_line = test
        lines.append(current_line)

        line_height = int(cv.getTextSize(
            "Tg", font, font_scale, 1)[0][1] * 1.5)
        y_offset = y1 + 30
        for line in lines:
            if y_offset < y2 - 10:
                cv.putText(frame, line, (x1 + 10, y_offset),
                           font, font_scale, (255, 255, 255), 1, cv.LINE_AA)
            y_offset += line_height

        # Line count below
        cv.putText(frame, f"{len(lines)} lines", (x1, y2 + 25),
                   font, font_scale, (0, 255, 0), 1, cv.LINE_AA)

    else:
        # 🧱 Vertical letter stacking mode
        # ignore spaces for vertical stacking
        chars = list(text.replace(" ", ""))
        char_height = int(cv.getTextSize("T", font, font_scale, 1)[0][1] * 1.2)
        y_offset = y1 + 30

        for c in chars:
            if y_offset > y2 - 10:
                break
            cv.putText(frame, c, (x1 + width // 3, y_offset),
                       font, font_scale, (255, 255, 255), 1, cv.LINE_AA)
            y_offset += char_height

        # Vertical count display
        cv.putText(frame, f"{len(chars)} chars", (x1, y2 + 25),
                   font, font_scale, (0, 255, 0), 1, cv.LINE_AA)

    return frame


def process_frame(frame, hands, config, deg):
    h, w, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
        # Get both hands
        lm1 = [(int(lm.x * w), int(lm.y * h))
               for lm in results.multi_hand_landmarks[0].landmark]
        lm2 = [(int(lm.x * w), int(lm.y * h))
               for lm in results.multi_hand_landmarks[1].landmark]

        index_tip_1 = lm1[8]  # left hand
        index_tip_2 = lm2[8]  # right hand

        # 🟩 Draw the holographic flex box between index fingers
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        frame = draw_flex_text_box(frame, text, index_tip_1, index_tip_2)

    elif results.multi_hand_landmarks:
        # fallback if only one hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        lm_list = [(int(lm.x * w), int(lm.y * h))
                   for lm in hand_landmarks.landmark]
        (wrist, thumb_tip, index_mcp, index_tip,
         middle_mcp, middle_tip, ring_tip, pinky_tip) = position_data(lm_list)

        thumb_vec = np.array(thumb_tip) - np.array(wrist)
        index_vec = np.array(index_tip) - np.array(wrist)
        dot_product = np.dot(thumb_vec, index_vec)
        magnitude = np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec)

        if magnitude != 0:
            cos_theta = np.clip(dot_product / magnitude, -1.0, 1.0)
            angleRad = np.arccos(cos_theta)
            angleDeg = np.degrees(angleRad)
        else:
            angleDeg = 0

        # 🩵 Draw degree result as fallback
        frame = draw_flex_text_box(
            frame,
            f"Thumb-Index angle: {angleDeg:.1f} degrees",
            (index_tip[0] - 150, index_tip[1] - 80),
            (index_tip[0] + 150, index_tip[1] + 80)
        )

    return frame, deg


def main():
    config = load_config()
    cap = initialize_camera(config)
    hands = mp.solutions.hands.Hands()
    deg = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame.")
                break

            frame = cv.flip(frame, 1)
            frame, deg = process_frame(
                frame, hands, config, deg)

            cv.imshow("Doctor Strange Flex Box", frame)
            if cv.waitKey(1) == ord(config["keybindings"]["quit_key"]):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
