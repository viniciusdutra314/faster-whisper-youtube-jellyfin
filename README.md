# **Faster Whisper + YouTube + Jellyfin**

### Transcribe any YouTube video and transfer it to your Jellyfin!
I love learning new languages. One problem I always stumble across is finding interesting/advanced content with subtitles. One solution to that is using a well optimized reimplementation of [OpenAI Whisper](https://github.com/openai/whisper) called [Faster Whisper](https://github.com/SYSTRAN/faster-whisper), which can transcribe videos using AI.

Go to [this Google Colab](https://colab.research.google.com/github/viniciusdutra314/faster-whisper-youtube-jellyfin/blob/main/faster_whisper_youtube.ipynb) and enjoy!

<a href="https://colab.research.google.com/github/viniciusdutra314/faster-whisper-youtube-jellyfin/blob/main/faster_whisper_youtube.ipynb">
<img src="google_colab_logo.png" width=200>
</a>

#### Faster Whisper speed on Google Colab:
Using the free T4 GPU, **I can generally transcribe a video in 10% of its duration using the largest model!** Here's a list of videos, their duration, and the execution time of the code.
*(I will write this table in the future)*

#### Jellyfin support:
The videos are saved along with their .info.json, which makes them compatible with the amazing [Jellyfin YouTube Metadata Plugin](https://github.com/ankenyr/jellyfin-youtube-metadata-plugin). Now, with the videos inside your Jellyfin, you can study on whatever device you want! Your smartphone, TV, laptop, you name it.

### Credits:

This is a fork from [lewangdev](https://github.com/lewangdev/faster-whisper-youtube), which itself is a fork of [ArthurFDLR](https://github.com/ArthurFDLR/whisper-youtube),  i just added a few lines to make it compatible with jellyfin. Thank you guys for making this simple yet extremely useful Google Colab notebook!