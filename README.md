# face_prediction_office

Made for fun at the office. 12 faces with names were the traning set, which is not uploaded.
This is run with a command line on the raspberry Pi, where the video camera starts filming, and guesses the name of the person on camera if probability reaches a certain level.
Three consecutive frames with the same prediction (and high proba) is required to make the assumption. An mp3 file is played saying the name of the indiviual.  

The process is two-step; first face detection is run on each frame. If a face is found, its bounding box is sent to the face embedding model which fetches the 128-dimension facenet embedding vector of that face. Prediction is done by the predictive algoritm which has already trained such vectors against the 12 individuals in the training set.
