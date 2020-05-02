import tensorflow as tf
#import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()
graph_def_file = "./ckpt_best/mobilefacenet_model.pbfrozen_model.pb"
input_arrays = ["image_inputs"]
output_arrays = ["embeddings"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)