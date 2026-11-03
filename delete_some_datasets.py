from pathlib import Path
from collections import Counter
import argparse
import re

# Widar 数据集根目录
DATASET_ROOT = Path(__file__).resolve().parent / "datasets" / "widar3.0"

# 需要处理的用户
USERS = ["user1", "user2", "user3"]

# 需要删除的重复次数
DELETE_REPETITIONS = {7, 8, 9, 10}

# Widar 文件名格式：
# user-gesture-position-orientation-repetition-receiver.dat
# 例如：user1-1-1-1-7-r1.dat
PATTERN = re.compile(
    r"^(user[123])-\d+-\d+-\d+-(\d+)-r\d+\.dat$",
    re.IGNORECASE
)


def main():
    parser = argparse.ArgumentParser(
        description="删除 Widar 三个用户指定重复次数的数据文件"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="实际删除文件；不添加该参数时仅预览"
    )
    args = parser.parse_args()

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"数据集目录不存在: {DATASET_ROOT}"
        )

    total_matched = 0
    total_deleted = 0

    for user in USERS:
        user_dir = DATASET_ROOT / user

        if not user_dir.is_dir():
            print(f"[跳过] 目录不存在: {user_dir}")
            continue

        target_files = []
        repetition_counter = Counter()

        for file in user_dir.rglob("*.dat"):
            if not file.is_file():
                continue

            match = PATTERN.fullmatch(file.name)

            if match is None:
                continue

            filename_user = match.group(1).lower()
            repetition = int(match.group(2))

            # 双重检查用户，避免误删
            if filename_user != user:
                continue

            if repetition in DELETE_REPETITIONS:
                target_files.append(file)
                repetition_counter[repetition] += 1

        target_files.sort()

        print(f"\n========== {user} ==========")

        for repetition in sorted(DELETE_REPETITIONS):
            print(
                f"重复次数 {repetition}: "
                f"{repetition_counter[repetition]} 个文件"
            )

        print(f"待删除文件总数: {len(target_files)}")

        for file in target_files:
            print(file)

            if args.delete:
                file.unlink()
                total_deleted += 1

        total_matched += len(target_files)

    print("\n========== 处理完成 ==========")
    print(f"匹配文件总数: {total_matched}")

    if args.delete:
        print(f"实际删除文件: {total_deleted}")
    else:
        print("当前为预览模式，没有删除任何文件")
        print("确认无误后，添加 --delete 参数执行删除")


if __name__ == "__main__":
    main()