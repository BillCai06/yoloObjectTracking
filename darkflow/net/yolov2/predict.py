import numpy as np
import math
import cv2
import os
import json

import serial # if you have not already done so
ser = serial.Serial('/dev/ttyACM0', 9600)
ser.baudrate = 9600 

#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	
	resultsForJSON = []
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, mess, (left, top - 12),
			0, 1e-3 * h, colors[max_indx],thick//3)
		print( "label", mess ,"y", (left+top)/2, "x",right+bot/2)
		#ser.write(("label"+str(mess)+"x"+str((left+top)/2)+"y"+str(right+bot/2)).encode) 
		#ser.write(int((left+top)/2)+int((right+bot)/2))
		y = (left+top)/2
		x = (right+bot)/2
		xmin =213.0
		xmid =426.0
		xmax =640.0

		if mess == "person" :
		#if x <= 213 && y <= 160:
			if  y <=160.0:
				if x <= xmin:
					ser.write(b'0')
				elif x <= xmid:
					ser.write(b'1')
				else :
					ser.write(b'2')
			elif y <= 320.0:
				if x <= xmin:
					ser.write(b'3')
				elif x <= xmid:
					ser.write(b'4')
				else :
					ser.write(b'5')

			else:
				if x <= xmin:
					ser.write(b'6')
				elif x <= xmid:
					ser.write(b'7')
				else :
					ser.write(b'8')
		
		
			#data = ser.read(2) 
			#print(data)
			

	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)
