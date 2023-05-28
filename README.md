# **Youtube Videos Transcription with Faster Whisper**

[![notebook shield](https://img.shields.io/static/v1?label=&message=Notebook&color=blue&style=for-the-badge&logo=googlecolab&link=https://colab.research.google.com/github/ArthurFDLR/whisper-youtube/blob/main/whisper_youtube.ipynb)](https://colab.research.google.com/github/lewangdev/whisper-youtube/blob/main/faster_whisper_youtube.ipynb)
[![repository shield](https://img.shields.io/static/v1?label=&message=Repository&color=blue&style=for-the-badge&logo=github&link=https://github.com/lewangdev/faster_whisper_youtube)](https://github.com/lewangdev/faster_whisper_youtube)

[faster-whisper](https://github.com/guillaumekln/faster-whisper) is a reimplementation of OpenAI's Whisper model using CTranslate2, which is a fast inference engine for Transformer models.

This implementation is up to 4 times faster than openai/whisper for the same accuracy while using less memory. The efficiency can be further improved with 8-bit quantization on both CPU and GPU.

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.

This Notebook will guide you through the transcription of a Youtube video using Faster Whisper. You'll be able to explore most inference parameters or use the Notebook as-is to store the transcript and video audio in your Google Drive.

# **Check GPU type** 🕵️

The type of GPU you get assigned in your Colab session defined the speed at which the video will be transcribed.
The higher the number of floating point operations per second (FLOPS), the faster the transcription.
But even the least powerful GPU available in Colab is able to run any Whisper model.
Make sure you've selected `GPU` as hardware accelerator for the Notebook (Runtime &rarr; Change runtime type &rarr; Hardware accelerator).

| GPU  | GPU RAM | FP32 teraFLOPS |   Availability   |
| :--: | :-----: | :------------: | :--------------: |
|  T4  |  16 GB  |      8.1       |       Free       |
| P100 |  16 GB  |      10.6      |    Colab Pro     |
| V100 |  16 GB  |      15.7      | Colab Pro (Rare) |

---

**Factory reset your Notebook's runtime if you want to get assigned a new GPU.**

```
    GPU 0: Tesla T4 (UUID: GPU-9ba4ce04-e020-44f9-8fc3-337ba5bb5496)
    Sun Oct  2 16:49:51 2022
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   36C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
```

# **Install libraries** 🏗️

This cell will take a little while to download several libraries, including Whisper.

---

```
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting git+https://github.com/openai/whisper.git
      Cloning https://github.com/openai/whisper.git to /tmp/pip-req-build-c3voj3wy
      Running command git clone -q https://github.com/openai/whisper.git /tmp/pip-req-build-c3voj3wy
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from whisper==1.0) (1.21.6)
    Requirement already satisfied: torch in /usr/local/lib/python3.7/dist-packages (from whisper==1.0) (1.12.1+cu113)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from whisper==1.0) (4.64.1)
    Requirement already satisfied: more-itertools in /usr/local/lib/python3.7/dist-packages (from whisper==1.0) (8.14.0)
    Requirement already satisfied: transformers>=4.19.0 in /usr/local/lib/python3.7/dist-packages (from whisper==1.0) (4.22.2)
    Requirement already satisfied: ffmpeg-python==0.2.0 in /usr/local/lib/python3.7/dist-packages (from whisper==1.0) (0.2.0)
    Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from ffmpeg-python==0.2.0->whisper==1.0) (0.16.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.9.0 in /usr/local/lib/python3.7/dist-packages (from transformers>=4.19.0->whisper==1.0) (0.10.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.7/dist-packages (from transformers>=4.19.0->whisper==1.0) (6.0)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from transformers>=4.19.0->whisper==1.0) (4.12.0)
    Requirement already satisfied: tokenizers!=0.11.3,<0.13,>=0.11.1 in /usr/local/lib/python3.7/dist-packages (from transformers>=4.19.0->whisper==1.0) (0.12.1)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers>=4.19.0->whisper==1.0) (2.23.0)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers>=4.19.0->whisper==1.0) (2022.6.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from transformers>=4.19.0->whisper==1.0) (21.3)
    Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers>=4.19.0->whisper==1.0) (3.8.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0,>=0.9.0->transformers>=4.19.0->whisper==1.0) (4.1.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->transformers>=4.19.0->whisper==1.0) (3.0.9)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->transformers>=4.19.0->whisper==1.0) (3.8.1)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers>=4.19.0->whisper==1.0) (2022.6.15)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers>=4.19.0->whisper==1.0) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers>=4.19.0->whisper==1.0) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers>=4.19.0->whisper==1.0) (1.24.3)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pytube in /usr/local/lib/python3.7/dist-packages (12.1.0)


    Using device: cuda:0
```

# **Optional:** Save images in Google Drive 💾

Enter a Google Drive path and run this cell if you want to store the results inside Google Drive.

---

`drive_path = "Colab Notebooks/Whisper Youtube"`

---

**Run this cell again if you change your Google Drive path.**

# **Model selection** 🧠

As of the first public release, there are 4 pre-trained options to play with:

|   Size   | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
| :------: | :--------: | :----------------: | :----------------: | :-----------: | :------------: |
|  medium  |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large-v2 |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |

---

`Model = 'large-v2'`

---

**Run this cell again if you change the model.**

```
    100%|█████████████████████████████████████| 2.87G/2.87G [01:14<00:00, 41.5MiB/s]
```

**large model is selected.**

# **Video selection** 📺

Enter the URL of the Youtube video you want to transcribe, whether you want to save the audio file in your Google Drive, and run the cell.

---

`URL = "https://youtu.be/dQw4w9WgXcQ"`

`store_audio = True`

---

**Run this cell again if you change the video.**

# **Run the model** 🚀

Run this cell to execute the transcription of the video. This can take a while and is very much based on the length of the video and the number of parameters of the model selected above.

```
Detected language 'en' with probability 0.99951171875

[00:00:00 000 --> 00:00:05 000]  So, anyone who's been paying attention for the last few months
[00:00:05 000 --> 00:00:09 000]  has been seeing headlines like this, especially in education.
[00:00:09 000 --> 00:00:11 000]  The thesis has been,
[00:00:11 000 --> 00:00:15 000]  students are going to be using chat GPT and other forms of AI
[00:00:15 000 --> 00:00:18 000]  to cheat, do their assignments, they're not going to learn,
[00:00:18 000 --> 00:00:22 000]  and it's going to completely undermine education as we know it.
[00:00:22 000 --> 00:00:25 000]  Now, what I'm going to argue today is not only are there ways
[00:00:25 000 --> 00:00:27 000]  to mitigate all of that,
[00:00:27 000 --> 00:00:30 000]  if we put the right guardrails, we do the right things, we can mitigate it,
[00:00:30 000 --> 00:00:33 000]  but I think we're at the cusp of using AI
[00:00:33 000 --> 00:00:37 000]  for probably the biggest positive transformation
[00:00:37 000 --> 00:00:40 000]  that education has ever seen.
[00:00:40 000 --> 00:00:42 000]  And the way we're going to do that
[00:00:42 000 --> 00:00:45 000]  is by giving every student on the planet
[00:00:45 000 --> 00:00:48 000]  an artificially intelligent but amazing personal tutor,
[00:00:48 000 --> 00:00:51 000]  and we're going to give every teacher on the planet
[00:00:51 000 --> 00:00:55 000]  an amazing artificially intelligent teaching assistant.
[00:00:55 000 --> 00:00:58 000]  And just to appreciate how big of a deal it would be
[00:00:58 000 --> 00:01:01 000]  to give everyone a personal tutor,
[00:01:01 000 --> 00:01:04 000]  I show you this clip
[00:01:04 000 --> 00:01:07 000]  from Benjamin Bloom's 1984 two-sigma study,
[00:01:07 000 --> 00:01:10 000]  or he called it the two-sigma problem.
[00:01:10 000 --> 00:01:12 000]  The two-sigma comes from two standard deviations,
[00:01:12 000 --> 00:01:14 000]  sigma is a symbol for standard deviation,
[00:01:14 000 --> 00:01:17 000]  and he had good data that showed that, look,
[00:01:17 000 --> 00:01:20 000]  a normal distribution, that's the one that you see in the traditional bell curve
...
[00:15:30 000 --> 00:15:32 000]  Thank you.
Transcript file created: /content/drive/My Drive/Colab Notebooks/Faster Whisper Youtube/hJP5GqnTrNo.srt
```

**Transcript file created: /content/drive/My Drive/Colab Notebooks/Faster Whisper Youtube/hJP5GqnTrNo.srt**
