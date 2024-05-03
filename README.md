# eyetracker
This project is currently WIP

This project uses opencv and a mediapipe face mesh to read the eyes and pupils. 

# How It Works
- The script captures video frames from the webcam.
- Utilizes MediaPipe to detect facial landmarks, focusing on eye and iris positions.
- Calculates eye aspect ratio to determine blinks.
- Moves the cursor or clicks based on the detected eye position and blink actions.

# Credits
Much of the current code was from this tutorial by AiPhile: https://www.youtube.com/watch?v=-jFobb6ARc4

This project is mostly for me to play with basic opencv and ai principals. Hopefully, my goal is to replace the computer mouse so that I can browse the internet with only my eyes (in the persuit of ultimate laziness).
