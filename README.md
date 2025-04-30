# Driver Drowsiness & Yawn Detection System 🚗😴

A real-time computer vision system that monitors a driver’s eye closure and yawning using a webcam. If signs of fatigue or inattention are detected, audio alerts and a simulated hazard light are triggered to ensure driver safety.

## 🔧 Features

- 👁️ Detects eye closure using Eye Aspect Ratio (EAR)
- 😮 Detects yawning using mouth opening distance
- 🔊 Voice alerts using `pyttsx3`
- 🚨 Simulated hazard light if eyes remain closed for a long duration
- 🧠 Real-time face landmark tracking with MediaPipe
- 🖥️ Built with OpenCV and Python

## 🛠️ Tech Stack

- Python 3.11
- OpenCV
- MediaPipe
- pyttsx3
- NumPy
- SciPy
