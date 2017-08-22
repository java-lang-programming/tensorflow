# -*- coding: utf-8 -*-
#
# created_at : 2017/08/19
# 
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_PATH = "/root/scikit/tonsorflow/"

def main():
  image = tf.read_file("/root/scikit/tonsorflow/sample/tensorflow/image/images/cat.jpeg")
  #　ファイルを読み込む
  decoded_image = tf.image.decode_jpeg(image, channels=3)
  croped_image = tf.random_crop(decoded_image, [100, 150, 3])

  # Launch the graph in a session.
  with tf.Session() as session:
    for i in range(3):
      result = session.run(croped_image)
      plt.imshow(result)
      plt.savefig(OUTPUT_PATH + 'croped_image' + str(i) + ".png")

if __name__ == "__main__":
  main()