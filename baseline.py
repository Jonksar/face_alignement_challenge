import numpy as np

from base import ProcessorBase, load_data_by_id, ProcessorResult, get_annoy_index

import logging

import attr
import pandas as pd
import matplotlib.pyplot as plt
import glob
from glob import glob
from tqdm import tqdm
import cv2
from annoy import AnnoyIndex
import click
logger = logging.getLogger(__name__)


class BaselineProcessor(ProcessorBase):
    def __init__(self, videos_csv_filename: str) -> None:
        super().__init__(videos_csv_filename)

        self.landmarks_index = get_annoy_index(self.embedding_maker)

    def build_index(self, filename: str = None):
        if filename is None:
            filename = self.default_index_filename

        face_counter = 0
        for video_i, row in tqdm(self.video_df.iterrows(), total=len(self.video_df)):
            db_paths = glob(
                "./data/*/{videoID}.npz".format(videoID=self.video_df.loc[video_i].videoID)
            )
            if len(db_paths) == 0:
                continue

            db_path = db_paths[0]
            db_colorImages, db_boundingBox, db_landmarks2D, db_landmarks3D = load_data(
                db_path
            )

            start_index = face_counter
            for frame_i in range(db_colorImages.shape[-1]):
                face_counter += 1
                self.landmarks_index.add_item(
                    face_counter,
                    self.embedding_maker.make_embedding(db_landmarks2D[..., frame_i]),
                )
            end_index = face_counter

            self.video_df.at[video_i, "start"] = start_index
            self.video_df.at[video_i, "end"] = end_index

        logger.info("Building index...")
        self.landmarks_index.build(10)  # 10 trees

        index_filename = f"{filename}.landmarks.ann"
        logger.info("Saving index to %s", filename)
        self.landmarks_index.save(filename)

        # TODO
        csv_filename = f"{filename}.landmarks.ann"
        print("Saving csv alongside index...")
        self.video_df.to_csv(filename + ".csv", index=False)

    def load_index(self, filename: str = None):
        self.landmarks_index.load(filename)  # super fast, will just mmap the file

    def reset(self) -> None:
        pass

    def process_frame(self, frame: np.ndarray, landmarks) -> ProcessorResult:
        nns, dists = self.landmarks_index.get_nns_by_vector(
            self.embedding_maker.make_embedding(landmarks),
            10,
            include_distances=True,
        )

        best_matches = [(image_i, dist) for image_i, dist in zip(nns, dists)]
        image_diffs = sorted(
            best_matches, key=lambda x: x[1], reverse=True
        )  # sort by distance

        best_match_idx = image_diffs[0][0]
        best_image, _, best_landmarks2D, best_landmarks3D = load_data_by_id(
            best_match_idx, self.video_df
        )

        return ProcessorResult(frame=best_image, frame_idx=best_match_idx)
