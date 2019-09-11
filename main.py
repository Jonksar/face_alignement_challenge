import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from glob import glob
from tqdm import tqdm
import cv2


def overlay_landmarks_on_frame(landmarks, frame):
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

    connectedPoints = [rigthEyebrowPoints, leftEyebrowPoints,
                       noseRidgePoints, noseBasePoints,
                       rightEyePoints, leftEyePoints, outerMouthPoints, innerMouthPoints]

    unconnectedPoints = [jawPoints]

    for conPts in connectedPoints:
        frame = cv2.polylines(frame,
                              [landmarks[conPts[0]:conPts[-1]]],
                              isClosed=True,
                              color=[255, 255, 255],
                              thickness=1)

    for conPts in unconnectedPoints:
        frame = cv2.polylines(frame,
                              [landmarks[conPts[0]:conPts[-1]]],
                              isClosed=False,
                              color=[255, 255, 255],
                              thickness=1)

    return frame


def load_data(npz_filepath):
    with np.load(npz_filepath) as face_landmark_data:
        colorImages = face_landmark_data['colorImages']
        boundingBox = face_landmark_data['boundingBox']
        landmarks2D = face_landmark_data['landmarks2D']
        landmarks3D = face_landmark_data['landmarks3D']

    return colorImages, boundingBox, landmarks2D, landmarks3D


def plot_images_in_row(*images, size=3, titles=None):
    plt.figure(figsize=(size * len(images), size))

    if titles is None:
        titles = ["" for _ in images]

    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.title(title)
        plt.imshow(image)


def debug_landmark_images(npz_path):
    colorImages, boundingBox, landmarks2D, landmarks3D = load_data(npz_path)
    print(list(map(lambda x: x.shape, [colorImages, boundingBox, landmarks2D, landmarks3D])))

    viz_images = []

    num_images = colorImages.shape[-1]  # 240

    for i in range(0, num_images, 40):
        img = colorImages[..., i].astype(np.uint8)
        landmarks = landmarks2D[..., i].astype(np.int32)
        img = overlay_landmarks_on_frame(landmarks, img)
        viz_images.append(img)

    plot_images_in_row(*viz_images)


def cost_function(query_video, predicted_video):
    # Frame diff sum
    # Number of videos
    # Number of transitions between videos
    # Face alignement diff
    pass

def frame_cost_function(last_frame, current_frame, keypoints_query, keypoints_current):
    pass

from annoy import AnnoyIndex

class FaceEmbeddingGenerator:
    @staticmethod
    def make_embedding(embedding):
        return embedding.flatten()


class FaceEmbeddingGenerator2D(FaceEmbeddingGenerator):
    dim = 68 * 2


class FaceEmbeddingGenerator3D(FaceEmbeddingGenerator):
    dim = 68 * 3


videoDF = pd.read_csv('./youtube-faces-with-facial-keypoints/youtube_faces_with_keypoints_large.csv')
print(videoDF.head(15))

query_loc = 10
query_path = glob("./youtube-faces-with-facial-keypoints/*/{videoID}.npz".format(videoID=videoDF.loc[query_loc].videoID))[0]  # To face align with this
q_colorImages, q_boundingBox, q_landmarks2D, q_landmarks3D = load_data(query_path)

embeddingMaker = FaceEmbeddingGenerator2D()                    # Embedding generator for 3D face keypoints
landmarks_index_args = [embeddingMaker.dim, 'euclidean']
"""
landmarks_index = AnnoyIndex(*landmarks_index_args)  # Approximate search index
face_counter = 0
for video_i, row in tqdm(videoDF.iterrows(), total=len(videoDF)):
    # Dont add video to the index
    if video_i == query_loc:
        continue

    db_paths = glob("./youtube-faces-with-facial-keypoints/*/{videoID}.npz".format(videoID=videoDF.loc[video_i].videoID))  # To face align with this
    if len(db_paths) == 0:
        continue

    db_path = db_paths[0]
    db_colorImages, db_boundingBox, db_landmarks2D, db_landmarks3D = load_data(db_path)

    start_index = face_counter
    for frame_i in range(db_colorImages.shape[-1]):
        face_counter += 1
        landmarks_index.add_item(face_counter, embeddingMaker.make_embedding(db_landmarks2D[..., frame_i]))
    end_index = face_counter

    videoDF.at[video_i, 'start'] = start_index
    videoDF.at[video_i, 'end']   = end_index

videoDF.to_csv('youtube-faces-with-facial-keypoints/youtube_faces_with_keypoints_large.csv', index=False)
landmarks_index.build(10) # 10 trees
landmarks_index.save('landmarks.ann')
"""
landmarks_index = AnnoyIndex(*landmarks_index_args)
landmarks_index.load('landmarks.ann') # super fast, will just mmap the file


def load_image_by_id(id, videoDF):
    video = videoDF[(videoDF.start < id) & (videoDF.end > id)]
    if len(video) == 0:
        return np.zeros((100, 100, 3))

    paths = glob("./youtube-faces-with-facial-keypoints/*/{videoID}.npz".format(videoID=video.iloc[0].videoID))
    if len(paths) == 0:
        return np.zeros((100, 100, 3))
    path = paths[0]
    frame_id = id - video.start
    q_colorImages, q_boundingBox, q_landmarks2D, q_landmarks3D = load_data(path)
    return q_colorImages[..., int(frame_id)]


for i in tqdm(range(q_colorImages.shape[-1])):
    query_image = q_colorImages[..., i]
    nns, dists = landmarks_index.get_nns_by_vector(
        embeddingMaker.make_embedding(q_landmarks2D[..., i]),
        10, include_distances=True)
    best_matches = [(image_i, load_image_by_id(image_i, videoDF)) for image_i in nns]
    image_diffs = [(i, np.linalg.norm(cv2.resize(match, query_image.shape[:2][::-1]) - query_image)) for i, match in best_matches]
    image_diffs = sorted(image_diffs, key=lambda x: x[1], reverse=True)

    best_match_idx = image_diffs[0][0]

    plot_images_in_row(
        load_image_by_id(best_match_idx, videoDF),
        query_image,
        titles=[f"Query {i}", f"Target {nns[0]}"])

    plt.suptitle(f"Cost: {dists[0]}")
    plt.savefig(f"debug/debug_{i:03d}.jpg")
    plt.close(fig='all')


# TODO: Documentation
# TODO: Command line utility that takes video in, and returns generated video

# 