import logging

import click

from baseline import BaselineProcessor
from processor import Processor
from utils.logging import setup_logging

logger = logging.getLogger(__name__)
videos_csv_filename = "./data/youtube_faces_with_keypoints_large.csv"

processor_cls = Processor


@click.group()
@click.option("--debug", "-d", is_flag=True)
@click.option("--baseline", "-b", is_flag=True)
def cli(debug: bool, baseline: bool):
    global processor_cls

    setup_logging(debug=debug)

    if baseline:
        processor_cls = BaselineProcessor


@cli.command()
@click.option("--videos", "-v", type=click.Path(exists=True), default=videos_csv_filename)
def index(videos: str):
    processor = processor_cls(videos)
    processor.build_index()


@cli.command()
@click.option("--videos", "-v", type=click.Path(exists=True), default=videos_csv_filename)
@click.argument('filename', type=click.Path(exists=True, dir_okay=False), required=False)
def process_video(videos: str, filename: str = None):
    processor = processor_cls(videos)

    processor.load_index()
    logger.info("Processing video from %s", filename)
    processor.process_video(filename)


if __name__ == "__main__":
    cli()
