"""Download the DROID 1.0.1 dataset from HuggingFace into a local directory.

https://huggingface.co/datasets/cadene/droid_1.0.1

By default only the files needed by `dataset_example/extract_latent.py` are
fetched (meta, the episode parquets and the 3 camera views actually used):
    meta/*
    data/chunk-XXX/episode_XXXXXX.parquet
    videos/chunk-XXX/observation.images.{exterior_1_left,exterior_2_left,wrist_left}/*.mp4
Use --full to also pull every other camera view (~370G+ in total).

The repo holds far too many files for `snapshot_download` to list, so the file
list is built by walking the repo tree directory by directory. Already
downloaded files are skipped, so the script can be re-run to resume.

Examples:
    # everything the training pipeline needs
    python scripts/download_droid.py --local_dir /path/to/droid_hf/droid_1.0.1

    # just the first 2 chunks, 5 episodes each, for a quick test
    python scripts/download_droid.py --local_dir dataset_example/droid_hf \
        --num_chunks 2 --max_episodes_per_chunk 5

    # then
    accelerate launch dataset_example/extract_latent.py \
        --droid_hf_path /path/to/droid_hf/droid_1.0.1 \
        --droid_output_path dataset_example/droid --svd_path ${path to svd}
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm

REPO_ID = 'cadene/droid_1.0.1'

# camera views read by dataset_example/extract_latent.py
CAMERAS = [
    'observation.images.exterior_1_left',
    'observation.images.exterior_2_left',
    'observation.images.wrist_left',
]


def list_dir(api, repo_id, revision, path, dirs=False):
    """List file (or sub-directory) paths directly under `path` in the repo."""
    items = api.list_repo_tree(repo_id, repo_type='dataset', revision=revision,
                               path_in_repo=path, recursive=False)
    return sorted(i.path for i in items if i.tree_id is not None) if dirs else \
        sorted(i.path for i in items if getattr(i, 'size', None) is not None)


def collect_files(api, args):
    """Build the list of repo paths to download."""
    files = list_dir(api, args.repo_id, args.revision, 'meta')

    chunks = list_dir(api, args.repo_id, args.revision, 'data', dirs=True)
    if args.num_chunks is not None:
        chunks = chunks[:args.num_chunks]
    print(f'{len(chunks)} chunk(s) to fetch')

    for chunk in tqdm(chunks, desc='listing chunks'):
        name = os.path.basename(chunk)  # chunk-XXX
        parquets = list_dir(api, args.repo_id, args.revision, f'data/{name}')
        if args.max_episodes_per_chunk is not None:
            parquets = parquets[:args.max_episodes_per_chunk]
        files.extend(parquets)

        episodes = {os.path.basename(p).replace('.parquet', '.mp4') for p in parquets}
        if args.full:
            cameras = [os.path.basename(d) for d in
                       list_dir(api, args.repo_id, args.revision, f'videos/{name}', dirs=True)]
        else:
            cameras = CAMERAS
        for camera in cameras:
            videos = list_dir(api, args.repo_id, args.revision, f'videos/{name}/{camera}')
            files.extend(v for v in videos if os.path.basename(v) in episodes)
    return files


def download_one(path, args):
    local_path = os.path.join(args.local_dir, path)
    if os.path.exists(local_path):
        return None
    hf_hub_download(repo_id=args.repo_id, repo_type='dataset', revision=args.revision,
                    filename=path, local_dir=args.local_dir, token=args.token)
    return path


def main():
    parser = argparse.ArgumentParser(description='Download the DROID 1.0.1 HF dataset.')
    parser.add_argument('--local_dir', type=str, required=True,
                        help='directory to store the dataset in (created if missing)')
    parser.add_argument('--repo_id', type=str, default=REPO_ID)
    parser.add_argument('--revision', type=str, default=None,
                        help='branch / tag / commit sha (default: main)')
    parser.add_argument('--full', action='store_true',
                        help='download every camera view instead of the 3 used for training')
    parser.add_argument('--num_chunks', type=int, default=None,
                        help='only download the first N chunks (chunk-000 ... chunk-N-1)')
    parser.add_argument('--max_episodes_per_chunk', type=int, default=None,
                        help='only download the first N episodes of each chunk (for testing)')
    parser.add_argument('--workers', type=int, default=8,
                        help='parallel download workers')
    parser.add_argument('--token', type=str, default=None,
                        help='HF token, defaults to $HF_TOKEN / the cached login')
    parser.add_argument('--list_only', action='store_true',
                        help='print how many files would be downloaded and exit')
    args = parser.parse_args()
    args.token = args.token or os.environ.get('HF_TOKEN')

    os.makedirs(args.local_dir, exist_ok=True)
    api = HfApi(token=args.token)

    print(f'downloading {args.repo_id} -> {args.local_dir}')
    files = collect_files(api, args)
    print(f'{len(files)} file(s) selected')
    if args.list_only:
        for f in files[:20]:
            print(' ', f)
        return

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, f, args): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc='downloading'):
            try:
                future.result()
            except (HfHubHTTPError, OSError) as e:
                failed.append((futures[future], repr(e)))

    if failed:
        print(f'{len(failed)} file(s) failed, re-run the script to retry:')
        for path, err in failed[:10]:
            print(f'  {path}: {err}')
    else:
        print(f'done: {args.local_dir}')


if __name__ == '__main__':
    main()
