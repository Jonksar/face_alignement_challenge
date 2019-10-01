import logging

import attr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from glob import glob
from tqdm import tqdm
import cv2
from annoy import AnnoyIndex
import click
logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class ProcessorResult:
    # The resulting color image (one frame of video)
    frame: np.ndarray
    # Index of the frame in the original data
    frame_idx: int

    # Face landmarks which could be transformed the same way as `frame`
    landmarks: np.array


def overlay_landmarks_on_frame(landmarks, frame):
    """
    :param landmarks: 2d landmarks for the face
    :param frame: corresponding image
    :return: frame with face drawn on top
    """
    frame = frame.astype(np.uint8)
    landmarks = landmarks.astype(np.int32)
    # define which points need to be connected with a line
    jawPoints = [0, 17]
    rigthEyebrowPoints = [17, 22]
    leftEyebrowPoints = [22, 27]
    noseRidgePoints = [27, 31]
    noseBasePoints = [31, 36]
    rightEyePoints = [36, 42]
    leftEyePoints = [42, 48]
    outerMouthPoints = [48, 60]
    innerMouthPoints = [60, 68]

    connectedPoints = [
        rigthEyebrowPoints,
        leftEyebrowPoints,
        noseRidgePoints,
        noseBasePoints,
        rightEyePoints,
        leftEyePoints,
        outerMouthPoints,
        innerMouthPoints,
    ]

    unconnectedPoints = [jawPoints]

    for conPts in connectedPoints:
        frame = cv2.polylines(
            frame,
            [landmarks[conPts[0] : conPts[1]]],
            isClosed=True,
            color=[255, 255, 255],
            thickness=1,
        )

    for conPts in unconnectedPoints:
        frame = cv2.polylines(
            frame,
            [landmarks[conPts[0] : conPts[1]]],
            isClosed=False,
            color=[255, 255, 255],
            thickness=1,
        )

    return frame


def load_data(npz_filepath):
    """
    :param npz_filepath: .npz file from youtube-faces-with-keypoints dataset
    :return: colorImages, boundingBox, landmarks2D, landmarks3D
    """
    with np.load(npz_filepath) as face_landmark_data:
        colorImages = face_landmark_data["colorImages"]
        boundingBox = face_landmark_data["boundingBox"]
        landmarks2D = face_landmark_data["landmarks2D"]
        landmarks3D = face_landmark_data["landmarks3D"]

    return colorImages, boundingBox, landmarks2D, landmarks3D


def plot_images_in_row(*images, size=3, titles=None):
    """
    param: images to plot in a row
    :param size: inches size for the plot
    :param titles: subplot titles

    return: matplotlib figure
    """
    fig = plt.figure(figsize=(size * len(images), size))

    if titles is None:
        titles = ["" for _ in images]

    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.title(title)
        plt.imshow(image)

    return fig


def debug_landmark_images(npz_path):
    """
    :param npz_path: filepath to a single preprocessed video
    :return: matplotlib figure
    """
    colorImages, boundingBox, landmarks2D, landmarks3D = load_data(npz_path)
    print(
        list(
            map(lambda x: x.shape, [colorImages, boundingBox, landmarks2D, landmarks3D])
        )
    )

    viz_images = []

    num_images = colorImages.shape[-1]  # 240

    for i in range(0, num_images, 40):
        img = colorImages[..., i].astype(np.uint8)
        landmarks = landmarks2D[..., i].astype(np.int32)
        img = overlay_landmarks_on_frame(landmarks, img)
        viz_images.append(img)

    return plot_images_in_row(*viz_images)


def full_cost_function(
    query_video, predicted_video, query_keypoints, predicted_keypoints
):
    pass


def frame_cost_function(
    last_frame, current_frame, keypoints_query, keypoints_current, query_image
):
    """
    :param last_frame: np.array of the last frame in predicted sequence
    :param current_frame: np.array of the current frame in predicted sequence
    :param keypoints_query: np.array of 2D keypoints on the current frame of query sequence
    :param keypoints_current: np.array of 2D keypoints on the current frame of predicted sequence

    :return: float, a weighted sum of different costs
    """
    image_diff_W = 0.1
    keypoint_diff_W = 0.9

    img_diff = np.mean(
        np.abs(cv2.resize(last_frame, current_frame.shape[:2][::-1]) - current_frame)
        > 25
    )

    # normalizing keypoints to be [0; 1]
    qshape = np.array(query_image.shape[:2])
    cshape = np.array(current_frame.shape[:2])

    keypoints_query_norm = keypoints_query / qshape
    keypoints_current_norm = keypoints_current / cshape

    keypoint_diff = np.mean(np.abs(keypoints_query_norm - keypoints_current_norm)) * 10

    return float(img_diff) * image_diff_W, float(keypoint_diff) * keypoint_diff_W


class FaceEmbeddingGenerator:
    """
    Base class of embedding generators
    """

    dim: int

    @staticmethod
    def make_embedding(embedding):
        return embedding.flatten()


class FaceEmbeddingGenerator2D(FaceEmbeddingGenerator):
    """ Embedding generator for 3D keypoints, which holds statically the input dimensions"""

    dim = 68 * 2


class FaceEmbeddingGenerator3D(FaceEmbeddingGenerator):
    """ Embedding generator for 2D keypoints, which holds statically the input dimensions"""

    dim = 68 * 3


def get_embedding_maker() -> FaceEmbeddingGenerator:
    return FaceEmbeddingGenerator2D()


def get_annoy_index(embeddingMaker: FaceEmbeddingGenerator) -> AnnoyIndex:
    landmarks_index_args = [embeddingMaker.dim, "euclidean"]
    return AnnoyIndex(*landmarks_index_args)  # Approximate search index



def load_data_by_id(id: int, videoDF):
    """Given an frame ID, and a dataset description"""
    video = videoDF[(videoDF.start < id) & (videoDF.end > id)]
    if len(video) == 0:
        return None, None, None, None

    paths = glob("./data/*/{videoID}.npz".format(videoID=video.iloc[0].videoID))
    if len(paths) == 0:
        return None, None, None, None

    path = paths[0]
    frame_id = int(id - video.start)
    q_colorImages, q_boundingBox, q_landmarks2D, q_landmarks3D = load_data(path)
    return (
        q_colorImages[..., frame_id],
        q_boundingBox[..., frame_id],
        q_landmarks2D[..., frame_id],
        q_landmarks3D[..., frame_id],
    )


def load_image_by_id(id, videoDF):
    """Utility function to get a single frame by ID"""
    return load_data_by_id(id, videoDF)[0]



class ProcessorBase:
    default_index_filename = "data/index"

    def __init__(self, videos_csv_filename: str) -> None:
        self.embedding_maker = get_embedding_maker()
        self.video_df = pd.read_csv(videos_csv_filename)

    def build_index(self, filename: str):
        pass

    def load_index(self, filename: str):
        pass

    def reset(self) -> None:
        pass

    def process_frame(self, frame: np.ndarray, landmarks) -> ProcessorResult:
        raise NotImplementedError()

    def process_video(self, video_filename):
        self.reset()

        q_colorImages, q_boundingBox, q_landmarks2D, q_landmarks3D = load_data(video_filename)

        last_predicted_image = q_colorImages[..., 0]
        for i in tqdm(range(q_colorImages.shape[-1])):
            query_image = q_colorImages[..., i]
            landmarks = q_landmarks2D[..., i]
            result = self.process_frame(frame=query_image, landmarks=landmarks)

            if not (last_predicted_image is None or result.frame is None):
                plot_images_in_row(
                    last_predicted_image,
                    overlay_landmarks_on_frame(result.landmarks, result.frame),
                    overlay_landmarks_on_frame(q_landmarks2D[..., i], query_image),
                    titles=["Last frame", f"Query {i}", f"Target {result.frame_idx}"],
                )

            # Something went badly wrong.
            if last_predicted_image is None or result.frame is None:
                print(
                    f"last_predicted_image is None: {last_predicted_image is None}; best_image is None: {result.frame is None}"
                )
                continue
            else:
                frame_cost = frame_cost_function(
                    last_predicted_image,
                    result.frame,
                    q_landmarks2D[..., i],
                    result.landmarks,
                    query_image,
                )
                last_predicted_image = result.frame

            plt.suptitle(f"Cost: {frame_cost}")
            dbg_name = f"debug/debug_{i:03d}.png"
            print(f"Saving debug image: {dbg_name}")
            plt.savefig(dbg_name)

            plt.close(fig="all")
