"""Download the DROID 1.0.1 dataset from HuggingFace into a local directory.

https://huggingface.co/datasets/cadene/droid_1.0.1

The dataset is in LeRobot v2.1 format: `meta/info.json` holds the episode count
and the path templates, so every file path is derived arithmetically instead of
by listing the repo (the repo has ~380k files, which neither `snapshot_download`
nor a directory walk can enumerate without blowing the HF API rate limit).

Downloads meta, one parquet per episode and the 3 camera views:
    meta/*
    data/chunk-XXX/episode_XXXXXX.parquet
    videos/chunk-XXX/observation.images.{exterior_1_left,exterior_2_left,wrist_left}/*.mp4

Already downloaded files are skipped, so re-run the script to resume.

Examples:
    # full dataset (~370G, 95600 episodes)
    python scripts/download_droid.py --local_dir ${path to droid}

    # first 2 chunks, 5 episodes each, for a quick test
    python scripts/download_droid.py --local_dir dataset_example/droid_hf \
        --num_chunks 2 --max_episodes_per_chunk 5

    # then
    accelerate launch dataset_example/extract_latent.py \
        --droid_hf_path ${path to droid} \
        --droid_output_path dataset_example/droid --svd_path ${path to svd}
"""

import argparse
import json
import os
import random
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, disable_progress_bars
from tqdm import tqdm

REPO_ID = 'cadene/droid_1.0.1'
META_FILES = ['meta/info.json', 'meta/episodes.jsonl', 'meta/episodes_stats.jsonl', 'meta/tasks.jsonl']

# set on ctrl-C / SIGTERM so queued work stops instead of running to completion
STOP = threading.Event()


def fetch(path, args, retries=6):
    """Download one repo file, retrying with backoff on rate limits."""
    for attempt in range(retries):
        if STOP.is_set():
            return None
        try:
            return hf_hub_download(repo_id=args.repo_id, repo_type='dataset',
                                   revision=args.revision, filename=path,
                                   local_dir=args.local_dir, token=args.token)
        except HfHubHTTPError as e:
            status = getattr(e.response, 'status_code', None)
            if status not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            # 429 quota is per 5 min window; back off well past it
            delay = min(60 * 2 ** attempt, 600) + random.uniform(0, 10)
            STOP.wait(delay)
    return None


def collect_files(args):
    """Derive every repo path to download from meta/info.json."""
    info = json.load(open(fetch('meta/info.json', args)))
    cameras = [k for k, v in info['features'].items() if v.get('dtype') == 'video']
    chunk_size = info['chunks_size']
    total = info['total_episodes']
    num_chunks = -(-total // chunk_size)
    if args.num_chunks is not None:
        num_chunks = min(num_chunks, args.num_chunks)
    print(f'{total} episodes / {num_chunks} chunk(s), cameras: {cameras}')

    files = list(META_FILES)
    for chunk in range(num_chunks):
        start = chunk * chunk_size
        end = min(start + chunk_size, total)
        if args.max_episodes_per_chunk is not None:
            end = min(end, start + args.max_episodes_per_chunk)
        for episode in range(start, end):
            files.append(info['data_path'].format(episode_chunk=chunk, episode_index=episode))
            for camera in cameras:
                files.append(info['video_path'].format(
                    episode_chunk=chunk, video_key=camera, episode_index=episode))
    return files


def download_one(path, args):
    if STOP.is_set():
        return None
    if os.path.exists(os.path.join(args.local_dir, path)):
        return None
    return fetch(path, args)


def main():
    parser = argparse.ArgumentParser(description='Download the DROID 1.0.1 HF dataset.')
    parser.add_argument('--local_dir', type=str, required=True,
                        help='directory to store the dataset in (created if missing)')
    parser.add_argument('--repo_id', type=str, default=REPO_ID)
    parser.add_argument('--revision', type=str, default=None,
                        help='branch / tag / commit sha (default: main)')
    parser.add_argument('--num_chunks', type=int, default=None,
                        help='only download the first N chunks of 1000 episodes')
    parser.add_argument('--max_episodes_per_chunk', type=int, default=None,
                        help='only download the first N episodes of each chunk (for testing)')
    parser.add_argument('--workers', type=int, default=4,
                        help='parallel download workers; lower this if you hit rate limits')
    parser.add_argument('--token', type=str, default=None,
                        help='HF token, defaults to $HF_TOKEN / the cached login')
    parser.add_argument('--list_only', action='store_true',
                        help='print how many files would be downloaded and exit')
    args = parser.parse_args()
    args.token = args.token or os.environ.get('HF_TOKEN')

    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    # per-file bars from hf_hub_download would drown out the overall progress bar
    disable_progress_bars()
    os.makedirs(args.local_dir, exist_ok=True)

    print(f'downloading {args.repo_id} -> {args.local_dir}')
    files = collect_files(args)
    todo = [f for f in files if not os.path.exists(os.path.join(args.local_dir, f))]
    print(f'{len(files)} file(s) selected, {len(files) - len(todo)} already present')
    if args.list_only:
        for f in todo[:20]:
            print(' ', f)
        return

    failed = []
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(download_one, f, args): f for f in todo}
    try:
        for future in tqdm(as_completed(futures), total=len(futures), desc='downloading'):
            if STOP.is_set():
                break
            try:
                future.result()
            except (HfHubHTTPError, OSError) as e:
                failed.append((futures[future], repr(e)))
    except KeyboardInterrupt:
        STOP.set()
    if STOP.is_set():
        # in-flight downloads cannot be interrupted, so drop them rather than
        # waiting: partial files live in .cache/ and are redone on the next run
        pool.shutdown(wait=False, cancel_futures=True)
        print('\ninterrupted, re-run the same command to resume')
        sys.stdout.flush()
        os._exit(130)
    pool.shutdown()

    if failed:
        print(f'{len(failed)} file(s) failed, re-run the script to retry:')
        for path, err in failed[:10]:
            print(f'  {path}: {err}')
    else:
        print(f'done: {args.local_dir}')


if __name__ == '__main__':
    main()
