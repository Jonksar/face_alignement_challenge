import numpy as np

from base import ProcessorBase, ProcessorResult, load_data_by_id


class Processor(ProcessorBase):
    def __init__(self, videos_csv_filename: str) -> None:
        super().__init__(videos_csv_filename)

    def build_index(self, filename: str = None):
        pass

    def reset(self) -> None:
        pass

    def process_frame(self, frame: np.ndarray, landmarks) -> ProcessorResult:
        raise NotImplementedError()
