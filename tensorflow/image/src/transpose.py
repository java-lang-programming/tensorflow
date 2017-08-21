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
  transposed_image = tf.image.transpose_image(decoded_image)

  session = tf.Session()
  with session.as_default():
    ouput = transposed_image.eval()

  plt.imshow(ouput)
  plt.savefig(OUTPUT_PATH + 'transposed_image.png')
if __name__ == "__main__":
  main()