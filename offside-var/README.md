# Offside VAR (portfolio project)

Computer-vision pipeline that estimates offside decisions from football broadcast clips.

## Setup (macOS, Apple Silicon)
```bash
brew install python@3.11 ffmpeg git
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

## Run Week 1
Put a short clip (5-10 s) in `data/`, then:
```bash
python scripts/run_week1.py --source data/clip.mp4
```

## Roadmap
- [x] Week 1: detection, tracking, team colours
- [ ] Week 2: pitch keypoints, homography, minimap
- [ ] Week 3: kick frame, offside rule, overlay
- [ ] Week 4: pose keypoints, evaluation, demo app

## Limitations
(Fill in as you go: occlusion, calibration error, frame rate.)
