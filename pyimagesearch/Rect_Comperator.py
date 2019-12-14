import numpy as np
import statistics
import cv2
class RectInfo:

    def __init__(self, rect,xy,score):
        self.xy= xy
        self.score= score
        self.Area=( rect[2]- rect[0])* ( rect[3]- rect[1])
        self.rect =rect

    def dist(self, pt):
        dist = np.sqrt( (self.xy[0]-pt[0])*(self.xy[0]-pt[0]) + (self.xy[1]-pt[1])*(self.xy[1]-pt[1]) )
        return dist
    def getRect(self):
        return self.rect

class rectComperator:

    def __init__(self, radius=100, minSampleSize=3):
        self.RectList=[]
        self.radius= radius
        self.minSampleSize = minSampleSize
    def add(self, rect,score):
        cX = int((rect[0] + rect[2]) / 2.0)
        cY = int((rect[1] + rect[3]) / 2.0)
        centroid = (cX,cY)
        newEle = RectInfo(rect,centroid,score)
        self.RectList.append(newEle)

    def getRectsInRadius(self,center):
        NeighbourRects=[]
        for rect in  self.RectList:
            if rect.dist(center) < self.radius:
                NeighbourRects.append(rect)
        return   NeighbourRects

    def compareArea(self,in_rect,frame):
        center = (in_rect[0]+in_rect[2])/2,(in_rect[1]+in_rect[3])/2
        neighbourRects= self.getRectsInRadius(center)
        if len(neighbourRects) < self.minSampleSize:
            return False, 1,1
        l =[]
        for ele in  neighbourRects:
            rect = ele.getRect()
            #cv2.rectangle(frame,(rect[0],rect[1]),(rect[2], rect[3]),(0, 255, 255), 2, 1)
            l.append(ele.Area)
        mean_Val = statistics.mean(l)
        #std = statistics.pvariance(l)
        std=1
        return True ,mean_Val, std
