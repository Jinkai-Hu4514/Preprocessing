"""Train on the valid-frame and padding boundary alone."""

from elder_common import INPUT_SHAPE, run_experiment


def make_inputs(_elder, _samples, frame_counts):
    import torch

    time_steps, subcarriers, antennas = INPUT_SHAPE
    valid = torch.arange(time_steps)[None, :] < torch.tensor(frame_counts)[:, None]
    return valid.float()[:, :, None, None].expand(-1, -1, subcarriers, antennas)


if __name__ == "__main__":
    run_experiment(make_inputs, "frame_count_only.csv")
