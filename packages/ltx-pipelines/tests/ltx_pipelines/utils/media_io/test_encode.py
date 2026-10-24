from fractions import Fraction
from pathlib import Path

import av
import torch

from ltx_pipelines.utils.media_io.encode import encode_video


def test_video_keeps_ntsc_frame_rate(tmp_path: Path) -> None:
    out = tmp_path / "clip.mp4"
    encode_video(torch.rand(9, 64, 64, 3), 24000 / 1001, None, str(out), 1)
    with av.open(str(out)) as container:
        assert container.streams.video[0].average_rate == Fraction(24000, 1001)

