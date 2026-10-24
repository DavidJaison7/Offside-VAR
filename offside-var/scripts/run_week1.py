"""Week 1: detect + track + team colouring on a short clip.

Usage:
    python scripts/run_week1.py --source data/clip.mp4 --output outputs/week1.mp4
"""
import argparse
from pathlib import Path

import numpy as np
import supervision as sv
from tqdm import tqdm

from offside.detect import Detector
from offside.teams import TeamClassifier, torso_color
from offside.track import make_tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default="outputs/week1.mp4")
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--max-frames", type=int, default=250, help="keep runs short on a fanless Air")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    info = sv.VideoInfo.from_video_path(args.source)
    detector = Detector(args.weights, imgsz=args.imgsz)
    tracker = make_tracker(info.fps)

    # Pass 1: sample frames to learn the two team colours.
    colors = []
    for frame in sv.get_video_frames_generator(args.source, stride=15, end=args.max_frames):
        players, _ = detector(frame)
        for box in players.xyxy:
            c = torso_color(frame, box)
            if c is not None:
                colors.append(c)
    teams = TeamClassifier().fit(colors)

    # Pass 2: track, classify, annotate, write.
    palette = sv.ColorPalette.from_hex(["#00BFFF", "#FF1493"])
    ellipse = sv.EllipseAnnotator(color=palette, thickness=2)
    label = sv.LabelAnnotator(color=palette, text_position=sv.Position.BOTTOM_CENTER)
    ball_ann = sv.TriangleAnnotator(color=sv.Color.YELLOW, base=20, height=17)

    with sv.VideoSink(args.output, info) as sink:
        frames = sv.get_video_frames_generator(args.source, end=args.max_frames)
        for frame in tqdm(frames, total=min(info.total_frames, args.max_frames)):
            players, ball = detector(frame)
            players = tracker.update_with_detections(players)
            if len(players):
                players.class_id = teams.predict(frame, players.xyxy)
            labels = [f"#{t}" for t in players.tracker_id]

            out = ellipse.annotate(frame.copy(), players)
            out = label.annotate(out, players, labels=labels)
            if len(ball):
                out = ball_ann.annotate(out, ball)
            sink.write_frame(out)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
