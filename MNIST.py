{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "MNIST.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyMEe4SynjmaaXL7C262enKh",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
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
        "<a href=\"https://colab.research.google.com/github/vinaybajjuri58/Multi-image-classification-model/blob/master/MNIST.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kCnQumvLyCRL",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from keras import layers\n",
        "from keras.datasets import mnist "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "l4kQBJO2yUjI",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kZNsNfYQyfCU",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "4a17a2c8-e731-465b-f216-2af3461f22f6"
      },
      "source": [
        "(train_images,train_label),(test_images,test_label)=mnist.load_data()\n"
      ],
      "execution_count": 60,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(10000, 28, 28)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 60
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FjYVtI8Pz7vn",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from keras import models\n",
        "from keras import layers"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aZR2VDSIyxUP",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "train_images=train_images.reshape((60000,28*28))\n",
        "test_images=test_images.reshape((10000,28*28))\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TLshL86o0LIe",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "model=models.Sequential()\n",
        "model.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))\n",
        "model.add(layers.Dense(256,activation='relu'))\n",
        "model.add(layers.Dense(10,activation='softmax'))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FduVPEk20n99",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "model.compile(\n",
        "    loss='categorical_crossentropy',\n",
        "    optimizer='adam',\n",
        "    metrics=['accuracy']\n",
        ")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3gl6kLRw1OZS",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "train_images=train_images.astype('float32')/255\n",
        "test_images=test_images.astype('float32')/255"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "e6K8zq9H1jHR",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from keras.utils import to_categorical\n",
        "train_label=to_categorical(train_label)\n",
        "test_label=to_categorical(test_label)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "weXtCfpr14Qg",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "outputId": "82078da2-46ad-4f37-a07a-99b97a1a4c7a"
      },
      "source": [
        "model.fit(train_images,train_label,epochs=5,batch_size=128)"
      ],
      "execution_count": 66,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/5\n",
            "60000/60000 [==============================] - 7s 115us/step - loss: 0.2331 - acc: 0.9308\n",
            "Epoch 2/5\n",
            "60000/60000 [==============================] - 7s 111us/step - loss: 0.0822 - acc: 0.9741\n",
            "Epoch 3/5\n",
            "60000/60000 [==============================] - 7s 109us/step - loss: 0.0527 - acc: 0.9833\n",
            "Epoch 4/5\n",
            "60000/60000 [==============================] - 7s 110us/step - loss: 0.0356 - acc: 0.9883\n",
            "Epoch 5/5\n",
            "60000/60000 [==============================] - 7s 109us/step - loss: 0.0269 - acc: 0.9917\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7ffac922e780>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 66
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hyKJSydI4ppN",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "6f3ae69e-2d19-4991-a7a4-cddc7ec5ccef"
      },
      "source": [
        "test_loss,test_accuracy=model.evaluate(test_images,test_label)"
      ],
      "execution_count": 67,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "10000/10000 [==============================] - 1s 64us/step\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Qn4sBqbv5ZB-",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "1b35eff3-d3d2-4027-a993-734c660dccb0"
      },
      "source": [
        "print('test_accuracy'+str(test_accuracy))\n",
        "print('test_loss'+str(test_loss))"
      ],
      "execution_count": 69,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "test_accuracy0.9803\n",
            "test_loss0.06552968098119599\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XjI31u9-5pTU",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}