{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Copy of Untitled6.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/kaushik3012/3d-pose-warping/blob/master/Copy_of_Untitled6.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Gf751gkkz20e"
      },
      "source": [
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "import pickle\n",
        "import matplotlib.pyplot as plt\n",
        "from matplotlib import image \n",
        "import glob\n",
        "import os\n",
        "from PIL import Image\n",
        "from numpy import asarray\n",
        "import PIL\n",
        "import pathlib\n",
        "import tensorflow_datasets as tfds\n",
        "# import tensorflow.keras.datasets.cifar10 as cf"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eoeMPzxY0SDk"
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/gdrive')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Q2xeUWl80cRW"
      },
      "source": [
        "!unzip '/content/gdrive/MyDrive/In-shop Clothes Retrieval Benchmark/Img/img.zip'"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "A3RAmRBy0oZX"
      },
      "source": [
        "infile = open('/content/gdrive/MyDrive/poses_fashion3d.pkl','rb')\n",
        "poses = pickle.load(infile)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ffvugJvx4-39"
      },
      "source": [
        "poses['MEN']['Denim']"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "m30ug4q66jVw"
      },
      "source": [
        "poses['MEN']['Denim']['id_00007216']['01']['7_additional']"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "S1YSFr2x0lwU"
      },
      "source": [
        " \n",
        "#Function For Converting image into numpy array\n",
        " \n",
        "def img_to_tensor (path):\n",
        "  # load the image\n",
        "  image = Image.open(path)\n",
        "  # convert image to numpy array\n",
        "  data = asarray(image)\n",
        "  # print(type(data))\n",
        "  # # summarize shape\n",
        "  # print(data.shape)\n",
        " \n",
        "  # # create Pillow image\n",
        "  # image2 = Image.fromarray(data)\n",
        "  # print(type(image2))\n",
        " \n",
        "  # # summarize image details\n",
        "  # print(image2.mode)\n",
        "  # print(image2.size)\n",
        "  data.reshape(256,256,3)\n",
        "  return data"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VTgfiPxK21_N"
      },
      "source": [
        "img_to_tensor('img/MEN/Denim/id_00000080/01_1_front.jpg')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TqWhlIah5c0B"
      },
      "source": [
        "data = {}\n",
        "for n in poses:\n",
        "  for i in poses[n]:\n",
        "  #data_men[i] = {} \n",
        "    for j in poses[n][i]:\n",
        "    #data_men[i][j]={}\n",
        "      for k in poses[n][i][j]:\n",
        "      #data_men[i][j][k]={}\n",
        "        for l in poses[n][i][j][k]:\n",
        "          #data_men[i][j][k][l]={}\n",
        "          #for m in poses[n][i][j][k][l]:\n",
        "          #/img/MEN/Denim/id_00000182/01_1_front.jpg\n",
        "          path = 'img/'+n+'/'+i+'/'+j+'/'+k+'_'+l+'.jpg'\n",
        "          \n",
        "          x = img_to_tensor(path)\n",
        "          data.update({path : x})\n",
        "\n",
        "#x = img_to_tensor(path)\n",
        "#data_men[i][j][k][l]=x"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6s88x-0sRG_M"
      },
      "source": [
        "print(data[\"img/MEN/Denim/id_00000080/01_7_additional.jpg\"])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_77-a0WM_5Ia"
      },
      "source": [
        "!mkdir Dataset"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fkp7CYQM_r3v"
      },
      "source": [
        "!cd Dataset"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YH9jPI5pJuTR",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 231
        },
        "outputId": "cc753ddf-e2d1-48fe-9f32-7a12e5e37f72"
      },
      "source": [
        "tstImg2=np.round(np.array(Image.open('img/MEN/Denim/id_00000080/01_1_front.jpg')).convert('RGB').resize((224,224)),dtype=np.float32)\n",
        "\n",
        "tf.reshape(tstImg2, shape=[-1, 224, 224, 3])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "TypeError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-27-ebea3259fb96>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mtstImg2\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mround\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mImage\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'img/MEN/Denim/id_00000080/01_1_front.jpg'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mresize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m224\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m224\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat32\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtstImg2\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mshape\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m224\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m224\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m3\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m<__array_function__ internals>\u001b[0m in \u001b[0;36mround_\u001b[0;34m(*args, **kwargs)\u001b[0m\n",
            "\u001b[0;31mTypeError\u001b[0m: _around_dispatcher() got an unexpected keyword argument 'dtype'"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XRwFrQvBLBqx"
      },
      "source": [
        "def my_func(arg):\n",
        "  arg = tf.convert_to_tensor(arg, dtype=tf.float32)\n",
        "  return arg\n",
        "\n",
        "  "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PUQ05_HSLwxF"
      },
      "source": [
        "tensor2=tf.io.decode_image(\n",
        "    '/content/img/MEN/Denim/id_00000080/01_1_front.jpg'\n",
        ")"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Q68Xdmi8Nd48"
      },
      "source": [
        "# img=Image.open('/content/img/MEN/Denim/id_00000080/01_1_front.jpg')\n",
        "# array = tf.keras.preprocessing.image.img_to_array(img)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XiqG_ngCOaY_"
      },
      "source": [
        "# print(array)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9WehVfI4PTsR"
      },
      "source": [
        "data = image.imread('/content/img/MEN/Denim/id_00000080/01_1_front.jpg')\n",
        "plt.imshow(data)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bg1ERIWvQMQ7"
      },
      "source": [
        "# len(data_pose)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yuBqileYZMhP"
      },
      "source": [
        "joint_order=['neck', 'nose', 'lsho', 'lelb', 'lwri', 'lhip', 'lkne', 'lank', 'rsho', 'relb', 'rwri', 'rhip', 'rkne', 'rank', 'leye', 'lear', 'reye', 'rear', 'pelv']"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "K1TS_hwvfNqu"
      },
      "source": [
        "def give_name_to_keypoints(array, joint_order):\n",
        "    #array = array.T\n",
        "    res = {}\n",
        "    for i, name in enumerate(joint_order):\n",
        "        res[name] = array[i]\n",
        "    return res"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "M5rK5hhhFwS3"
      },
      "source": [
        "for i, name in enumerate(joint_order):\n",
        "  print(i,name)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VjjW4GgJhxyC"
      },
      "source": [
        "path=\"img/MEN/Denim/id_00000080/01_7_additional.jpg\"\n",
        "# print(data_pose.get(path))\n",
        "print(data.get(path))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iHqy84icfQZC"
      },
      "source": [
        "data_with_joints={}\n",
        "for path,image in data.items():\n",
        "  array=data.get(path)\n",
        "  data_with_joints[path]=give_name_to_keypoints(array, joint_order)\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fhelmibBGCx1"
      },
      "source": [
        "data_with_joints[\"img/MEN/Denim/id_00000080/01_7_additional.jpg\"]['lsho']"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SYTkjnkbogH0"
      },
      "source": [
        "img_men = tf.keras.preprocessing.image_dataset_from_directory(\n",
        "    'img/MEN',\n",
        "    image_size=(256, 256),\n",
        "    labels = 'inferred'\n",
        ")"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dGpFQfer43Aj"
      },
      "source": [
        "type(img_men)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "UdD-2XXK1835"
      },
      "source": [
        "img_men_training = tf.keras.preprocessing.image_dataset_from_directory(\n",
        "    'img/MEN',\n",
        "    validation_split=0.2,\n",
        "    subset=\"training\",\n",
        "    seed=123,\n",
        "    image_size=(256, 256),\n",
        "    labels = 'inferred'\n",
        ")"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YsijlFpiTAZ3"
      },
      "source": [
        "type(img_men_training)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KaQhVQ-K4GLz"
      },
      "source": [
        "# DATASET_SIZE = 7838\n",
        "# train_size = int(0.8 * DATASET_SIZE)\n",
        "# val_size = int(0.10 * DATASET_SIZE)\n",
        "# test_size = int(0.10 * DATASET_SIZE)\n",
        "\n",
        "# dataset = tf.data.TFRecordDataset(img_men_training)\n",
        "# # dataset = tf.data.Dataset(img_men_training)\n",
        "# dataset = dataset.shuffle()\n",
        "# train_dataset =  dataset.take(train_size)\n",
        "# test_dataset  =  dataset.skip(train_size)\n",
        "# val_dataset   =  dataset.skip(val_size)\n",
        "# test_dataset  =  test_dataset.take(test_size)"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}