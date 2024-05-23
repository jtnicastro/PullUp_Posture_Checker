import cv2
import math
import mediapipe as mp
import numpy as np

# ========== CALCULATIONS ========== #

def findDistance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findVector(x1, x2):
    """Calculate the vector from point x1 to point x2."""
    return x2 - x1

def findAngle(x1, y1, x2, y2):
    """Calculate the angle between two vectors."""
    theta = math.acos((x1 * x2 + y1 * y2) / (math.sqrt(x1**2 + y1**2) * math.sqrt(x2**2 + y2**2)))
    return theta * (180 / math.pi)

def inRange(a, b, rel_tol=0.02, abs_tol=0.0):
    """Determine if two values are within a certain range of each other."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def checkShoulder(shldr_height, array):
    """Check if the shoulder height is within an acceptable range."""
    return inRange(shldr_height, np.min(array), rel_tol=0.1)

def percentElevated(shldr_height, array):
    """Calculate the percentage the shoulder is elevated."""
    return ((shldr_height - np.min(array)) / np.min(array)) * 100

def isIncreasing(arm_ang_array):
    """Determine if the arm angle is increasing."""
    if len(arm_ang_array) > 5 and 15 < arm_ang_array[-1] < 170:
        return arm_ang_array[-5] < arm_ang_array[-1]
    return False

# ========== INITIALIZE ========== #

# Initialize variables
full_range = True
reps = 0
prev_reps = 0

# Define colors
colors = {
    'green': (110, 220, 0),
    'blue': (255, 127, 0),
    'red': (50, 50, 255),
    'white': (248, 248, 255),
    'black': (0, 0, 0),
    'purple': (139, 61, 72),
    'yellow': (255, 255, 0)
}

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize colors for lines
arm_color = colors['green']
body_color = colors['blue']
r_shldr_color = colors['green']
l_shldr_color = colors['green']

# Create arrays for storing shoulder heights and arm angles
r_height_arr = np.array([])
l_height_arr = np.array([])
r_angle_arr = np.array([])
l_angle_arr = np.array([])

# Create arrays for timestamps
uneven_arms_time = np.array([])
r_shldr_time = np.array([])
l_shldr_time = np.array([])
full_range_time = np.array([])

# Initialize Mediapipe pose class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ========== MAIN MODULE ========== #
if __name__ == "__main__":
    # Input video to be captured
    file_name = 'test.mp4'
    cap = cv2.VideoCapture(file_name)
    
    # Acquire video data
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Framerate of video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)  # Size of video frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Write output as MP4

    # Writes output video with determined parameters
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

# ========================= PROCESS IMAGE ============================================================================ #

    # Processing captured frames
    while cap.isOpened():
        success, image = cap.read()
        # Outputs error message if frame was not captured
        if not success:
            print("Empty Camera Frame")
            break
        
        cframe = cap.get(cv2.CAP_PROP_POS_FRAMES)
        tframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        time = cframe / fps
        
        # Obtain actual values instead of normalized coordinates
        h, w = image.shape[:2]

        # Convert BGR image to RGB before processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with Mediapipe
        result = pose.process(image)

        # Convert image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.pose_landmarks:
            # Shortcut for pose landmarks
            lm = result.pose_landmarks.landmark
            lmPose = mp_pose.PoseLandmark

            # ========== OBTAIN LANDMARK COORDINATES ========== #
            def get_coords(landmark):
                return int(lm[landmark].x * w), int(lm[landmark].y * h)
            
            r_shldr_x, r_shldr_y = get_coords(lmPose.RIGHT_SHOULDER)
            l_shldr_x, l_shldr_y = get_coords(lmPose.LEFT_SHOULDER)
            r_elbow_x, r_elbow_y = get_coords(lmPose.RIGHT_ELBOW)
            l_elbow_x, l_elbow_y = get_coords(lmPose.LEFT_ELBOW)
            r_wrist_x, r_wrist_y = get_coords(lmPose.RIGHT_WRIST)
            l_wrist_x, l_wrist_y = get_coords(lmPose.LEFT_WRIST)
            r_hip_x, r_hip_y = get_coords(lmPose.RIGHT_HIP)
            l_hip_x, l_hip_y = get_coords(lmPose.LEFT_HIP)

            # ========== CALCULATE DISTANCE ========== #
            def update_height_array(shldr_height, arr):
                return np.append(arr, shldr_height)

            hip_dist = findDistance(r_hip_x, r_hip_y, l_hip_x, l_hip_y)
            shldr_dist = findDistance(r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y)
            r_shldr_height = findDistance(r_shldr_x, r_shldr_y, r_hip_x, r_hip_y)
            l_shldr_height = findDistance(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y)
            
            r_height_arr = update_height_array(r_shldr_height, r_height_arr)
            l_height_arr = update_height_array(l_shldr_height, l_height_arr)

            # ========== CALCULATE VECTORS AND ANGLES ========== #
            def get_vectors_and_angles(elbow_x, elbow_y, shldr_x, shldr_y, wrist_x, wrist_y):
                up_arm_vect_x = findVector(elbow_x, shldr_x)
                up_arm_vect_y = findVector(elbow_y, shldr_y)
                lw_arm_vect_x = findVector(elbow_x, wrist_x)
                lw_arm_vect_y = findVector(elbow_y, wrist_y)
                arm_angle = findAngle(lw_arm_vect_x, lw_arm_vect_y, up_arm_vect_x, up_arm_vect_y)
                return arm_angle

            r_arm_angle = get_vectors_and_angles(r_elbow_x, r_elbow_y, r_shldr_x, r_shldr_y, r_wrist_x, r_wrist_y)
            l_arm_angle = get_vectors_and_angles(l_elbow_x, l_elbow_y, l_shldr_x, l_shldr_y, l_wrist_x, l_wrist_y)
            
            r_angle_arr = np.append(r_angle_arr, r_arm_angle)
            l_angle_arr = np.append(l_angle_arr, l_arm_angle)

            # ========== DISPLAY TEXT ========== #
            def display_text(img, text, coords, clr, thick):
                cv2.putText(img, text, coords, font, 1, clr, thick)
            
            display_text(image, str(int(r_arm_angle)), (r_elbow_x + 30, r_elbow_y + 20), colors['black'], 5)
            display_text(image, 'o', (r_elbow_x + 90, r_elbow_y + 6), colors['black'], 5)
            display_text(image, str(int(r_arm_angle)), (r_elbow_x + 30, r_elbow_y + 20), arm_color, 2)
            display_text(image, 'o', (r_elbow_x + 90, r_elbow_y + 6), arm_color, 2)
            display_text(image, str(int(l_arm_angle)), (l_elbow_x - 100, l_elbow_y + 20), colors['black'], 5)
            display_text(image, 'o', (l_elbow_x - 40, l_elbow_y + 6), colors['black'], 5)
            display_text(image, str(int(l_arm_angle)), (l_elbow_x - 100, l_elbow_y + 20), arm_color, 2)
            display_text(image, 'o', (l_elbow_x - 40, l_elbow_y + 6), arm_color, 2)

            # ========== CALCULATE MIDWAY POINTS ========== #
            lw_spine = (int(l_hip_x + hip_dist / 2), int((l_hip_y + r_hip_y) / 2))
            up_spine = (int(l_shldr_x + shldr_dist / 2), int((l_shldr_y + r_shldr_y) / 2))
            chin_x = int(l_shldr_x + shldr_dist / 2)
            chin_y = int(((l_shldr_y + r_shldr_y) / 2) - (shldr_dist / 2.5))
            bar = int((r_wrist_y + l_wrist_y) / 2) - 25

            # ========== DRAW LINES ========== #
            def draw_line(img, pt1, pt2, clr, thick=4):
                cv2.line(img, pt1, pt2, clr, thick)

            draw_line(image, (r_wrist_x, r_wrist_y), (r_elbow_x, r_elbow_y), body_color)
            draw_line(image, (l_wrist_x, l_wrist_y), (l_elbow_x, l_elbow_y), body_color)
            draw_line(image, (r_elbow_x, r_elbow_y), (r_shldr_x, r_shldr_y), body_color)
            draw_line(image, (l_elbow_x, l_elbow_y), (l_shldr_x, l_shldr_y), body_color)
            draw_line(image, (r_shldr_x, r_shldr_y), (r_hip_x, r_hip_y), body_color)
            draw_line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), body_color)
            draw_line(image, (r_shldr_x, r_shldr_y), (l_shldr_x, l_shldr_y), body_color)
            draw_line(image, (r_hip_x, r_hip_y), (l_hip_x, l_hip_y), body_color)
            draw_line(image, lw_spine, up_spine, body_color)

            # ========== DRAW MARKERS ========== #
            def draw_marker(img, coord, clr, rad):
                cv2.circle(img, coord, rad, clr, -1)
            
            for coord in [(l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), (l_elbow_x, l_elbow_y), (r_elbow_x, r_elbow_y), 
                          (l_wrist_x, l_wrist_y), (r_wrist_x, r_wrist_y), (l_hip_x, l_hip_y), (r_hip_x, r_hip_y), 
                          (chin_x, chin_y)]:
                draw_marker(image, coord, colors['yellow'], 7 if 'hip' in locals() else 12)

# ========================= POSTURE VERIFICATION ===================================================================== #

            # ========== VERIFY ARM ANGLE ========== #
            if inRange(r_arm_angle, l_arm_angle, rel_tol=0.24, abs_tol=9):
                arm_color = colors['green']
            else:
                display_text(image, 'UNEVEN ARM ANGLES', (92, 770), colors['black'], 5)
                display_text(image, 'UNEVEN ARM ANGLES', (92, 770), colors['red'], 4)
                for coord in [(l_elbow_x, l_elbow_y), (r_elbow_x, r_elbow_y)]:
                    draw_marker(image, coord, arm_color, 20)
                arm_color = colors['red']
                uneven_arms_time = np.append(uneven_arms_time, time)
                np.save('Uneven Arm Angle Times', uneven_arms_time)

            # ========== CHECK SHOULDER ELEVATION ========== #
            if r_arm_angle < 155 and not checkShoulder(r_shldr_height, r_height_arr):
                r_shldr_color = colors['red']
                percent = int(percentElevated(r_shldr_height, r_height_arr))
                display_text(image, f'RIGHT SHOULDER ELEVATED BY {percent}%', (40, 840), colors['black'], 4)
                display_text(image, f'RIGHT SHOULDER ELEVATED BY {percent}%', (40, 840), colors['red'], 2)
                draw_marker(image, (r_shldr_x, r_shldr_y), r_shldr_color, 25)
                r_shldr_time = np.append(r_shldr_time, time)
                np.save('Right Shoulder Elevation Times', r_shldr_time)
            else:
                r_shldr_color = colors['green']
            
            if l_arm_angle < 155 and not checkShoulder(l_shldr_height, l_height_arr):
                l_shldr_color = colors['red']
                percent = int(percentElevated(l_shldr_height, l_height_arr))
                display_text(image, f'LEFT SHOULDER ELEVATED BY {percent}%', (40, 890), colors['black'], 4)
                display_text(image, f'LEFT SHOULDER ELEVATED BY {percent}%', (40, 890), colors['red'], 2)
                draw_marker(image, (l_shldr_x, l_shldr_y), l_shldr_color, 25)
                l_shldr_time = np.append(l_shldr_time, time)
                np.save('Left Shoulder Elevation Times', l_shldr_time)
            else:
                l_shldr_color = colors['green']

            # ========== CHECK FULL RANGE OF MOTION ========== #
            if isIncreasing(r_angle_arr) and isIncreasing(l_angle_arr):
                body_color = colors['blue']
                full_range = r_angle_arr[-1] >= 115 and l_angle_arr[-1] >= 115
            if not any([isIncreasing(r_angle_arr), isIncreasing(l_angle_arr), full_range]):
                display_text(image, 'NOT FULL RANGE OF MOTION', (25, 960), colors['black'], 5)
                display_text(image, 'NOT FULL RANGE OF MOTION', (25, 960), colors['red'], 3)
                body_color = colors['red']
                full_range_time = np.append(full_range_time, time)
                np.save('Not Full Range of Motion Times', full_range_time)

            # ========== COUNTS REPETITIONS ========== #
            if full_range and chin_y <= bar:
                if reps == prev_reps:
                    reps += 1
            if chin_y > bar:
                prev_reps = reps

            display_text(image, f'REPS: {int(reps)}', (200, 60), colors['black'], 6)
            display_text(image, f'REPS: {int(reps)}', (200, 60), colors['purple'], 4)

# ========================= WRITE OUTPUT VIDEO ======================================================================= #
            # Write output frames
            video_output.write(image)
            
            # Customize display window and display output
            cv2.namedWindow('Pull Up Posture Checker', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Pull Up Posture Checker', (w, h))
            cv2.moveWindow('Pull Up Posture Checker', 150, 75)
            cv2.imshow('Pull Up Posture Checker', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Closes display window
    cap.release()
    video_output.release()
    cv2.destroyAllWindows()
