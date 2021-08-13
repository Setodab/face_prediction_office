# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])


# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
#recognizer = ExtraTreesClassifier(n_estimators = 300, max_depth=10, min_samples_split=2, verbose=2,min_samples_leaf=1, class_weight='balanced')
recognizer = ExtraTreesClassifier(n_estimators = 200, max_depth=5, min_samples_split=2, verbose=2,min_samples_leaf=1, class_weight='balanced')
#recognizer = SVC(C=1,kernel='linear',degree=3,class_weight='balanced',verbose=2,probability=True)
#recognizer = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto', leaf_size=30)

print(recognizer)
recognizer.fit(data["embeddings"], labels)

print(len(data["embeddings"]))

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()