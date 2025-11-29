# maclrn-facial

Requirements

Run on python 3.12

To Install:
1. pip install arg parse cv2 numpy joblib time sklearn imblearn
   
To Train:
1. Run training_notebook.ipynb althrough out
2. If scaler.joblib and svm_emotion_model.joblib appears in the same directory, proceed to next step

To Run:
1. Open terminal in same directory
2. python realtime_detect.py --model svm_emotion_model.joblib --scaler scaler.joblib --camera 0
