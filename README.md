# Face alignment challenge

## There is a grand prize of 500€ for the first place and additional smaller prizes for others!

## Goal
> *The goal of the challenge is to make a good-looking montage of images from videos from a database of celebrity videos to match an unknown target video! We will also look through best scoring videos and add points for creativity!*

Good-looking face-alignement has been defined as an cost function that incentivises to keep frame differences low between frame and face alignement between target and produced outputs low.

For creativity, do image warping, have videos of only a single celebrity, add a beat, do face swaps... let your creativity flow and make us laugh! :)


## Running the code TODO

### Pull the github repository
In order to submit an answer, clone current github repository for solution interface
```
git clone git@github.com:Jonksar/face_alignement_challenge.git
cd face_alignement_challenge

# You can set up an virtual environment here
pip install -r requirements.txt
```
### Download the data
```
# Download only 10% of the data, to get started faster ~1GB 
aws s3 cp s3://veriff-face-alignment-challenge/youtube_faces_with_keypoints_small.zip .
aws s3 cp s3://veriff-face-alignment-challenge/youtube_faces_with_keypoints_small.csv .

# Download the remaining data ~10GB
aws s3 cp s3://veriff-face-alignment-challenge/youtube_faces_with_keypoints_big.csv .
aws s3 cp s3://veriff-face-alignment-challenge/youtube_faces_with_keypoints_large.csv .

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

## Participating
In TODO, you can find a Processor class that is abstraction for your solution. TODO comments in the code. You are expected to fill this in.
```
class Processor(ProcessorBase):
    def __init__(self, videos_csv_filename: str) -> None:
        super().__init__(videos_csv_filename)

    def build_index(self, filename: str = None) -> None:
        pass
    
    def reset(self) -> None:
        pass

    def process_frame(self, frame: np.ndarray, landmarks) -> ProcessorResult:
        raise NotImplementedError()
```

You can also find a baseline approach on solving the problem as BaselineProcessor, that does OK in face-alignement, but does not work well for frame difference part of the cost function. It is OK to use baseline approach as a starting point and improve upon it.

## Submitting
TODO 

Challenge to be used in PyCon Estonia 2019
