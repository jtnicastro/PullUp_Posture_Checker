# ========================= IMPORT =================================================================================== #
import cv2
import math as m
import mediapipe as mp
import numpy as np

# ========== CALCULATIONS ========== #

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate vector
def findVector(x1, x2):
    vect = x2 - x1
    return vect

# Calculate angle
def findAngle(x1, y1, x2, y2):
    theta = m.acos((x1 * x2 + y1 * y2) / (m.sqrt((x1**2)+(y1**2))
                *m.sqrt(((x2)**2)+((y2)**2))))
    degree = int(180 / m.pi) * theta
    return degree

# Calculate closeness
def inRange(a, b, rel_tol=.02, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# Check if shoulders are in range
# adds all shoulder heights to array and compares current height with minimum height
def checkShoulder(shldr_height, array):
    current_min = np.min(array)
    return inRange(shldr_height, current_min, rel_tol=0.1)

# Calculate percent elevated
def percentElevated(shldr_height, array):
    percent = ((shldr_height - np.min(array))/np.min(array)*100)
    return percent

# Determine if arm angle is increasing
''' iterates arm angle array and determines arms are increasing in angle when the current arm
    angle is greater than the arm angle five values ago. Five was selected to resolve minor 
    inconsistencies between each value of the array. Function is only performed when arm angles
    are between 15 and 170 degrees to improve accuracy.'''
def isIncreasing(arm_ang_array):
    while arm_ang_array.size > 5 and 15 < arm_ang_array[-1] < 170:
        if arm_ang_array[-5] < arm_ang_array[-1]:
            return True
        else:
            return False

# ========================= INITIALIZE =============================================================================== #

# Initialize variables
full_range = True
reps = 0
prev_reps = 0

# Colors
green = (110, 220, 0)
blue = (255, 127, 0)
red = (50, 50, 255)
white = (248, 248, 255)
black = (0, 0, 0)
purple = (139, 61, 72)

# Font
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors for Lines
arm_color = green
body_color = blue
r_shldr_color = green
l_shldr_color = green

# Create Array for Shoulder Height
r_height_arr = np.array([])
l_height_arr = np.array([])

# Creates Array for Arm Angles
r_angle_arr = np.array([])
l_angle_arr = np.array([])

# Creates Array for timestamps
uneven_arms_time = np.array([])
r_shldr_time = np.array([])
l_shldr_time = np.array([])
full_range_time = np.array([])

# Initialize mediapipe pose class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ========== MAIN MODULE ========== #
if __name__ == "__main__":
    # Input video to be captured
    file_name = 'test.mp4'
    cap = cv2.VideoCapture(file_name)
    # Aquire video data
    fps = int(cap.get(cv2.CAP_PROP_FPS))                                                       # framerate of video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)                                                               # size of video frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                                   # write output as MP4

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
        fps = cap.get(cv2.CAP_PROP_FPS)
        time = cframe / fps
        # Used to obtain actual values instead of normalized x and y coordinates
        h, w = image.shape[:2]
        # Convert BGR image to RGB before processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process image
        result = pose.process(image)
        # Convert image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Used to shorten SDK public class syntax
        lm = result.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        # ========== OBTAIN LANDMARK COORDINATES ========== #
        ''' Obtains the x and y coordinate for each desired landmark and mulitplies coordinates with 
            frames height and width to get normlaized coordinates'''
        # right shoulder
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        # left shoulder
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        # right elbow
        r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
        r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)
        # left elbow
        l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
        l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
        # right wrist
        r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
        r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
        # left wrist
        l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
        l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
        # right hip
        r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
        # left hip
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        # ========== CALCULATE DISTANCE ========== #
        # between hips
        hip_dist = findDistance(r_hip_x, r_hip_y, l_hip_x, l_hip_y)
        # between shoulders
        shldr_dist = findDistance(r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y)
        # between right shoulder and right hip
        r_shldr_height = findDistance(r_shldr_x, r_shldr_y, r_hip_x, r_hip_y)
        # between left shoulder and left hip
        l_shldr_height = findDistance(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y)
        # adds shoulder height to array
        r_height_arr = np.append(r_height_arr, r_shldr_height)
        l_height_arr = np.append(l_height_arr, l_shldr_height)

        # ========== CALCULATE VECTORS ========== #
        # vector from shoulder to elbow
        r_up_arm_vect_x = findVector(r_elbow_x, r_shldr_x)
        r_up_arm_vect_y = findVector(r_elbow_y, r_shldr_y)
        l_up_arm_vect_x = findVector(l_elbow_x, l_shldr_x)
        l_up_arm_vect_y = findVector(l_elbow_y, l_shldr_y)
        # vector from elbow to wrist
        r_lw_arm_vect_x = findVector(r_elbow_x, r_wrist_x)
        r_lw_arm_vect_y = findVector(r_elbow_y, r_wrist_y)
        l_lw_arm_vect_x = findVector(l_elbow_x, l_wrist_x)
        l_lw_arm_vect_y = findVector(l_elbow_y, l_wrist_y)

        # ========== CALCULATE ANGLES ========== #
        # right arm
        r_arm_angle = findAngle(r_lw_arm_vect_x, r_lw_arm_vect_y, r_up_arm_vect_x, r_up_arm_vect_y)
        # left arm
        l_arm_angle = findAngle(l_lw_arm_vect_x, l_lw_arm_vect_y, l_up_arm_vect_x, l_up_arm_vect_y)
        # adds arm angles to array
        r_angle_arr = np.append(r_angle_arr, r_arm_angle)
        l_angle_arr = np.append(l_angle_arr, l_arm_angle)

        # ========== DISPLAY TEXT ========== #
        # text was placed twice with thicker black font underneath to outline text
        cv2.putText(image, str(int(r_arm_angle)), (r_elbow_x + 30, r_elbow_y + 20), font, 1, black, 5)
        cv2.putText(image, 'o', (r_elbow_x + 90, r_elbow_y + 6), font, .6, black, 5)
        cv2.putText(image, str(int(r_arm_angle)), (r_elbow_x + 30, r_elbow_y + 20), font, 1, arm_color, 2)
        cv2.putText(image, 'o', (r_elbow_x + 90, r_elbow_y + 6), font, .6, arm_color, 2)
        cv2.putText(image, str(int(l_arm_angle)), (l_elbow_x - 100, l_elbow_y + 20), font, 1, black, 5)
        cv2.putText(image, 'o', (l_elbow_x - 40, l_elbow_y + 6), font, .6, black, 5)
        cv2.putText(image, str(int(l_arm_angle)), (l_elbow_x - 100, l_elbow_y + 20), font, 1, arm_color, 2)
        cv2.putText(image, 'o', (l_elbow_x - 40, l_elbow_y + 6), font, .6, arm_color, 2)

        # ========== CALCULATE MIDWAY POINTS ========== #
        # Points that were calculated based off distance from a landmark
        lw_spine = int(l_hip_x) + int(hip_dist/2), int((int(l_hip_y)+int(r_hip_y))/2)
        up_spine = int(l_shldr_x) + int(shldr_dist/2), int((int(l_shldr_y)+int(r_shldr_y))/2)
        chin_x = int(l_shldr_x) + int(shldr_dist/2)
        chin_y = int(((int(l_shldr_y)+int(r_shldr_y))/2)-(int(shldr_dist/2.5)))
        bar = int((int(r_wrist_y)+int(l_wrist_y))/2)-25

        # ========== DRAW LINES ========== #
        # lower arm
        cv2.line(image, (r_wrist_x, r_wrist_y), (r_elbow_x, r_elbow_y), body_color, 4)
        cv2.line(image, (l_wrist_x, l_wrist_y), (l_elbow_x, l_elbow_y), body_color, 4)
        # upper arm
        cv2.line(image, (r_elbow_x, r_elbow_y), (r_shldr_x, r_shldr_y), body_color, 4)
        cv2.line(image, (l_elbow_x, l_elbow_y), (l_shldr_x, l_shldr_y), body_color, 4)
        # back
        cv2.line(image, (r_shldr_x, r_shldr_y), (r_hip_x, r_hip_y), body_color, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), body_color, 4)
        # shoulder line
        cv2.line(image, (r_shldr_x, r_shldr_y), (l_shldr_x, l_shldr_y), body_color, 4)
        # hip line
        cv2.line(image, (r_hip_x, r_hip_y), (l_hip_x, l_hip_y), body_color, 4)
        # neck line
        cv2.line(image, (up_spine), (chin_x, chin_y), body_color, 4)

        # ========== DRAW MARKERS ========== #
        # shoulders
        cv2.circle(image, (l_shldr_x, l_shldr_y), 12, l_shldr_color, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 12, r_shldr_color, -1)
        # elbows
        cv2.circle(image, (l_elbow_x, l_elbow_y), 10, arm_color, -1)
        cv2.circle(image, (r_elbow_x, r_elbow_y), 10, arm_color, -1)
        # wrists
        cv2.circle(image, (l_wrist_x, l_wrist_y), 10, arm_color, -1)
        cv2.circle(image, (r_wrist_x, r_wrist_y), 10, arm_color, -1)
        # hips
        cv2.circle(image, (l_hip_x, l_hip_y), 7, white, -1)
        cv2.circle(image, (r_hip_x, r_hip_y), 7, white, -1)
        # chin
        cv2.circle(image, (chin_x, chin_y), 7, white, -1)

# ========================= POSTURE VERIFICATION ===================================================================== #

        # ========== VERIFY ARM ANGLE ========== #
        ''' Checks to make sure the angle of the right arm and angle of left arm are about equal.
            Relative tolerance is set to 18% to allow for some error and absolute tolerance was 
            set to 9 to ensure smaller angles close in value do not get classified as uneven '''
        if inRange(r_arm_angle, l_arm_angle, rel_tol=.24, abs_tol=9):
            arm_color = green
        else:
            cv2.putText(image, 'UNEVEN ARM ANGLES', (92, 770), font, 1.3, black, 5)
            cv2.putText(image, 'UNEVEN ARM ANGLES', (92, 770), font, 1.3, red, 4)
            cv2.circle(image, (l_elbow_x, l_elbow_y), 20, arm_color, 3)
            cv2.circle(image, (r_elbow_x, r_elbow_y), 20, arm_color, 3)
            arm_color = red
            uneven_arms_time = np.append(uneven_arms_time, time)                            # adds frame time to array
            np.save('Uneven Arm Angle Times', uneven_arms_time)                             # saves array as data on CPU

        # ========== CHECK SHOULDER ELEVATION ========== #
        ''' Checks to make sure both scapulars stay retracted throughout the exercise.
            Shoulders become elevated when scapulars are not retracted so calculating the distance
            between shoulders and hips can help spot this. This is tested only when arm angle is below
             ~155 degrees because scapular retraction is not necessary at the bottom of a repetition '''
        # creates array of all shoulder heights and finds the current minimum value
        # if current shoulder height us not close to minimum shoulder height user is notified
        if r_arm_angle < 155 and checkShoulder(r_shldr_height, r_height_arr) != True:
            r_shldr_color = red
            cv2.putText(image, 'RIGHT SHOULDER ELEVATED BY ' + str(int(percentElevated(r_shldr_height, r_height_arr))) +
                        '%', (40, 840), font, 1, black, 4)
            cv2.putText(image, 'RIGHT SHOULDER ELEVATED BY ' + str(int(percentElevated(r_shldr_height, r_height_arr))) +
                        '%', (40, 840), font, 1, red, 2)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 25, r_shldr_color, 3)
            r_shldr_time = np.append(r_shldr_time, time)
            np.save('Right Shoulder Elevation Times', r_shldr_time)
        else:
            r_shldr_color = green
        if l_arm_angle < 155 and checkShoulder(l_shldr_height, l_height_arr) != True:
            l_shldr_color = red
            cv2.putText(image, 'LEFT SHOULDER ELEVATED BY ' + str(int(percentElevated(l_shldr_height, l_height_arr))) +
                        '%', (40, 890), font, 1, black, 4)
            cv2.putText(image, 'LEFT SHOULDER ELEVATED BY ' + str(int(percentElevated(l_shldr_height, l_height_arr))) +
                        '%', (40, 890), font, 1, red, 2)
            cv2.circle(image, (l_shldr_x, l_shldr_y), 25, l_shldr_color, 3)
            l_shldr_time = np.append(l_shldr_time, time)
            np.save('Left Shoulder Elevation Times', l_shldr_time)
        else:
            l_shldr_color = green

        # ========== CHECK FULL RANGE OF MOTION ========== #
        ''' Verifies that full range of motion is occuring. If angle between each arm is increasing then
            user is in downward motion from the bar. If user is going down and exceeds an angle of 115 
            degrees then full range of motion is satisfied. If user begins to travel in upward motion
            (angle of arms decreases) before exceeding 115 degrees, the repetition is labled not full 
            range of motion.'''
        # each arm angle is added to array and the most previous value is indexed and compared to threshold value
        if isIncreasing(r_angle_arr) and isIncreasing(l_angle_arr):
            body_color = blue
            if r_angle_arr[-1] >= 115 and l_angle_arr[-1] >= 115:
                full_range = True
            else:
                full_range = False
        if not any([isIncreasing(r_angle_arr), isIncreasing(l_angle_arr), full_range]):
            cv2.putText(image, 'NOT FULL RANGE OF MOTION', (25, 960), font, 1.2, black, 5)
            cv2.putText(image, 'NOT FULL RANGE OF MOTION', (25, 960), font, 1.2, red, 3)
            body_color = red
            full_range_time = np.append(full_range_time, time)
            np.save('Not Full Range of Motion Times', full_range_time)

        # ========== COUNTS REPETITIONS ========== #
        ''' Counts the number of repitions that occur by tracking when the chin goes above the bar. If the 
            user did not obtain full range of motion of that repetition it is not counted.'''
        # creates prev_reps to ensure only one repetition is added when chin goes above the bar
        if full_range and chin_y <= bar:
            if reps == prev_reps:
                reps += 1
        if chin_y > bar:
            prev_reps = reps
        cv2.putText(image, 'REPS: ' + str(int(reps)), (200, 60), font, 1.5, black, 6)
        cv2.putText(image, 'REPS: ' + str(int(reps)), (200, 60), font, 1.5, purple, 4)

# ========================= WRITE OUTPUT VIDEO ======================================================================= #
        # Write output frames
        video_output.write(image)
        # Customize display window and display output
        cv2.namedWindow('Pull Up Posture Checker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pull Up Posture Checker', (w-220,h))
        cv2.moveWindow('Pull Up Posture Checker', 150,-75)
        cv2.imshow('Pull Up Posture Checker', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
# Closes display window
cap.release()
cv2.destroyAllWindows()