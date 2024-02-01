Getting started
===============

Installation
------------

The Video Emotion Recognition(VIEMR) module can be downloaded either from pip or github, with the respective commands.

* :code:`pip install VideoEmotionRecognition`
* :code:`git+https://github.com/Deflyer/Multimodal-Emotion-Recognition-from-Videos.git`

To use, it is also needed to install some other modules. Most of them are listed in the file requirements.txt, so running
pip install -r requirements.txt and later using the following commands should be enough to get everything you need to run.

.. code:: python3

    git clone https://github.com/m3hrdadfi/soxan
    mv soxan/*

In case you are using colab, there is an example with all the modules being installed in the link 
https://colab.research.google.com/drive/1UqzA6bDgtZWGji652a0UjT7ZtDh3Xyi5?authuser=1#scrollTo=d3QJ6iSd-o3D

First steps
-----------

After installing the module with all of it's dependencies, import the viemr module.

.. code:: python3

    from VideoEmotionRecognition import viemr

Then create an instance of the class VideoEmotionRecognition passing the path to the video you want to be analyzed as a parameter.

.. code:: python3

    example = viemr.VideoEmotionRecognition("video.mp4")

From now on, use :code:`example.emotion_recognition(modality="transcript")`
selecting a modality among transcript, audio, video and multimodal to run the classification.

The results can then be seen in the form of a dataframe with

.. code:: python3

    example.get_labels(modality="transcript")

or in a heatmap with

.. code:: python3

    example.get_heatmap(modality="transcript")

For more details and extra uses, please check the documentations session.
