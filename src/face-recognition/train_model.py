# Owner: Joseph Wong
# Last updated: 3/21/20
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Load embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

# Encode name labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# Train the model using SVC
print("[INFO] training model...")
recognizer = SVC(C=1, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Write recognizer as .pickle
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# Write encoder as .pickle
f = open("output/label_encoder.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
