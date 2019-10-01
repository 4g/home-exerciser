sizes=[161, 193, 257, 289, 321, 353, 385, 417, 449, 481, 513, 801]
modelnames=['ResNet50','mobilenet_v1_100','mobilenet_v1_101','mobilenet_v1_75','mobilenet_v1_50']

for modelname in modelnames:
    for size in sizes:
        url=f"https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_{modelname}_{size}x{size}_multi_kpt_stripped.tflite"
        print (url)