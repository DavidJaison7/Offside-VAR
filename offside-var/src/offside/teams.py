import cv2
import numpy as np
from sklearn.cluster import KMeans


def torso_color(frame, xyxy):
    """Mean LAB colour of the upper-body region of a player box (mostly jersey)."""
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    h, w = y2 - y1, x2 - x1
    crop = frame[y1 + int(0.2 * h): y1 + int(0.5 * h), x1 + int(0.25 * w): x2 - int(0.25 * w)]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0)


class TeamClassifier:
    """Two-cluster KMeans on jersey colour. Referees/goalkeepers get handled in a later step."""

    def __init__(self):
        self.km = KMeans(n_clusters=2, n_init=10, random_state=0)

    def fit(self, colors):
        self.km.fit(np.array(colors))
        return self

    def predict(self, frame, xyxy_array):
        out = []
        for box in xyxy_array:
            c = torso_color(frame, box)
            out.append(0 if c is None else int(self.km.predict([c])[0]))
        return np.array(out, dtype=int)
