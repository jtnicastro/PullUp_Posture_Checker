# Pull Up Posture Checker
This project is a Python script that uses OpenCV and MediaPipe to process a video of a person performing pull ups. Users can upload a video of themselves performing pull ups and the program detects shoulder elevations and arm angles, provides real-time feedback on muscular symmetry, and counts repetitions.

## Features
+ **Real-Time Feedback**: Displays angles of arm lifts and notifies if arms are uneven or if shoulders are elevated.
+ **Repetition Counting**: Counts the number of full-range arm lift repetitions.
+ **Output Video**: Annotates the input video with arm angles, shoulder elevation percentages, and repetition count.

## Installation
1. **Clone the Repository:**
    ```python
    git clone https://github.com/jtnicastro/PullUp_Posture_Checker.git
    ```

2. **Navigate to the Directory:**
    ```python
    cd PullUp_Posture_Checker
    ```

3. **Install the Required Packages:**
    ```python
    pip install -r requirements.txt
    ```

## Usage
1. **Place Your Video:**

    Ensure your input video file is named **'test.mp4'** and is placed in the project directory.

   
2. **Run the Script:**
    ```python
    python script.py
    ```

3. **Output:** 

    The script processes the video, provides real-time feedback, and saves an annotated output video as **'output.mp4'**.

## Functions
+ findDistance(x1, y1, x2, y2): Calculates the Euclidean distance between two points.
+ findVector(x1, x2): Calculates the vector from point x1 to point x2.
+ findAngle(x1, y1, x2, y2): Calculates the angle between two vectors.
+ inRange(a, b, rel_tol, abs_tol): Checks if two values are within a specified range.
+ checkShoulder(shldr_height, array): Checks if the shoulder height is within an acceptable range.
+ percentElevated(shldr_height, array): Calculates the percentage by which the shoulder is elevated.
+ isIncreasing(arm_ang_array): Determines if the arm angle is increasing.

## Output Information
+ Angles: Displayed near the elbows in the video.
+ Percent Elevation: Displayed if a shoulder is elevated.
+ Repetition Count: Displayed on the screen.
+ Alerts: Displays warnings for uneven arm lifts and shoulder elevations.

## Example Outputs


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.tldrlegal.com/license/mit-license) file for details.

## Acknowledgments
+ [Mediapipe](https://pypi.org/project/mediapipe/)
+ [OpenCV](https://opencv.org/)




This project is inspired by the need to provide real-time feedback and monitoring for arm lifts, particularly focusing on shoulder elevation and arm angles to ensure correct form and prevent injury.
