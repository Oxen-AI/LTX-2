from fractions import Fraction
from pathlib import Path

import av
import torch

from ltx_core.color.hlg import encode_linear_hdr_frames_to_hlg_mp4


def test_master_keeps_ntsc_frame_rate(tmp_path: Path) -> None:
    out = tmp_path / "master.mp4"
    encode_linear_hdr_frames_to_hlg_mp4([torch.rand(9, 64, 64, 3)], out, fps=24000 / 1001, device=torch.device("cpu"))
    with av.open(str(out)) as container:
        assert container.streams.video[0].average_rate == Fraction(24000, 1001)
