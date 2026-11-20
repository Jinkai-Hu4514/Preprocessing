from common import TIME_COLUMNS, run_training


if __name__ == "__main__":
    run_training("frame_time", ("frame_count", *TIME_COLUMNS))
