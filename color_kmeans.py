import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", required = True, help = "Path to the train images folder")
ap.add_argument("-v", "--test", required = True, help = "Path to the test images folder")
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
args = vars(ap.parse_args())
#train
directory = args["train"]
centroids = []
labels = []
for subdir in os.listdir(directory):
	dominant_color = []
	for file in os.listdir(directory+subdir):
		print(directory+subdir+"/"+file)
		image = cv2.imread(directory+subdir+"/"+file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.reshape((image.shape[0] * image.shape[1], 3))

		clt = KMeans(n_clusters = args["clusters"])
		clt.fit(image)

		if(clt.cluster_centers_[0][0]<230):
			dominant_color.append(clt.cluster_centers_[0])
		else:
			dominant_color.append(clt.cluster_centers_[1])
	np_colors = np.asarray(dominant_color)
	length = np_colors.shape[0]
	centroid = [sum(np_colors[:,0])/length,sum(np_colors[:,1])/length, sum(np_colors[:,2])/length]
	labels.append(subdir)
	centroids.append(centroid)
	print(centroids)
	
#test
valid_test=0
all_test=0
for file in os.listdir(args["test"]):
	all_test+=1
	print(file)
	image = cv2.imread(args["test"]+file)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.reshape((image.shape[0] * image.shape[1], 3))

	clt = KMeans(n_clusters = args["clusters"])
	clt.fit(image)
	dominant_color = []
	if(clt.cluster_centers_[0][0]<230):
		dominant_color = (clt.cluster_centers_[0])
	else:
		dominant_color = (clt.cluster_centers_[1])

	min_dst = sys.maxsize
	best_label = ""
	for idx, cnt in enumerate(centroids):
		dist = (cnt[0]-dominant_color[0])**2 + (cnt[1]-dominant_color[1])**2 + (cnt[2]-dominant_color[2])**2
		if dist<min_dst:
			min_dst = dist
			best_label = labels[idx]
	if best_label==file[:2]:
		valid_test+=1
print("Trafność: ")		
print(valid_test/all_test)