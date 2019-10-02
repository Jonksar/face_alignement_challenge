# Face alignment challenge

**There is a grand prize of 500€ for the first place and additional smaller prizes for others!**

## Goal

> The goal of the challenge is to make a good-looking montage of images from videos
> from a database of celebrity videos to match an unknown target video!
> We will also look through best scoring videos and add points for creativity!

Good-looking face-alignment has been defined as a cost function that 
incentivizes keeping visual differences low between consecutive frames
and faces aligned between target and produced outputs.

For creativity, do image warping, have videos of only a single celebrity, add a beat, do face swaps...
let your creativity flow and make us laugh! :)


## Running the code

You will need Python 3.6 or later.

### Setup

In order to get started, clone current github repository for solution interface
and set up Python development environment:

```
# TODO: correct repo url
git clone git@github.com:Jonksar/face_alignement_challenge.git
cd face_alignement_challenge

# You can set up an virtual environment here
python3 -m venv venv
. venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Download the data

```
# Download only 10% of the data, to get started faster ~1GB
wget https://veriff-face-alignment-challenge.s3-eu-west-1.amazonaws.com/youtube_faces_with_keypoints_small.zip
wget https://veriff-face-alignment-challenge.s3-eu-west-1.amazonaws.com/youtube_faces_with_keypoints_small.csv

# Download the remaining data ~10GB
wget https://veriff-face-alignment-challenge.s3-eu-west-1.amazonaws.com/youtube_faces_with_keypoints_big.csv
wget https://veriff-face-alignment-challenge.s3-eu-west-1.amazonaws.com/youtube_faces_with_keypoints_large.csv

# alternatively use aws cli:
aws s3 cp s3://veriff-face-alignment-challenge/FILENAME .

Unzip ZIP files in the root directory of the repository.
```

### Using command line interface:

You can get running with:
```
# Help about the command line interface
python cli.py --help

# Build file index, takes about 20s
python cli.py index 

# Process a video, matching it against the index.
python cli.py process-video -v PATH_TO_VIDEO
```

We also provide a baseline model that you can try by adding `--baseline` flag after `cli.py`:

    python cli.py --baseline index 
    python cli.py --baseline process-video -v PATH_TO_VIDEO


## Participating

In `processor.py`, you can find a Processor class that is abstraction for your solution.
Read through the comments and documentation in that class and add your own solution.

You can also find a baseline approach on solving the problem as BaselineProcessor,
that does OK in face-alignment, but does not work well for frame difference part of the cost function.
It is OK to use baseline approach as a starting point and improve upon it.


## Submitting
TODO 

Challenge to be used in PyCon Estonia 2019
