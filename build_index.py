"""Optional CLI: prebuild MP4 embeddings before launching Streamlit.

    python build_index.py
    python build_index.py --model BAAI/bge-small-en-v1.5 --force
"""
import argparse
from pathlib import Path

from file_index_catalog import load_file_index
from semantic_matcher import (DEFAULT_MODEL, catalog_records, build_encoder,
                              load_slide_evidence, prepare_index)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    records = catalog_records(load_file_index(here/'bootcamp_step1_file_index.xlsx'))
    print(f'{len(records)} eligible MP4 paths loaded. Model: {args.model}', flush=True)
    print('Loading embedding model (first run may download model weights)...', flush=True)
    encoder = build_encoder(args.model)
    print('Checking persisted index, then encoding titles and metadata if needed...', flush=True)
    result = prepare_index(records, cache_dir=here/'index', model_name=args.model,
                           encoder=encoder,
                           slide_evidence=load_slide_evidence(here/'index'/'verified_video_slides.json'),
                           force=args.force)
    print(f"Index ready: {len(result['records'])} MP4s; "
          f"loaded existing vectors={result['from_cache']}; "
          f"verified slide sets={sum(bool(r['slide_text']) for r in result['records'])}")


if __name__ == '__main__':
    main()
