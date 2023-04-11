import argparse
import cv2
import tensorflow

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='check again')
args = vars(ap.parse_args())
model = tensorflow.keras.models.load_model('best_model.h5')

img = cv2.imread(args['input'])

cv2.imshow(winname='cat', mat = img)
cv2.waitKey(10000)