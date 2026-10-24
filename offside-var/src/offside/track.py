import supervision as sv


def make_tracker(fps: int = 25) -> sv.ByteTrack:
    """ByteTrack gives each player a stable ID across frames."""
    return sv.ByteTrack(frame_rate=int(fps))
