
class oneTrackPoint:
    def __init__(self, centroid, score, bbox):
        self.centroid= centroid
        self.score = score
        self.bbox = bbox

class Single_Track:
    def __init__(self, centroid,score, bbox):
        self.Track= list()
        self.Track.append(oneTrackPoint(centroid,score, bbox))

    def addPointToTrack(self,centroid,score, bbox):
        self.Track.append(oneTrackPoint(centroid, score, bbox))
