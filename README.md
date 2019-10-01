# Face alignment challenge

## There is a grand prize of 500€ for the first place and additional smaller prizes for others!

## Goal
> *The goal of the challenge is to make a good-looking montage of images from videos from a database of celebrity videos to match an unknown target video! We will also look through best scoring videos and add points for creativity!*

Good-looking face-alignement has been defined as an cost function that incentivises to keep frame differences low between frame and face alignement between target and produced outputs low.

For creativity, do image warping, have videos of only a single celebrity, add a beat, do face swaps... let your creativity flow and make us laugh! :)


## Running the code TODO
1. Download the data
2. Pull the github repository
3. Command line interface

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
