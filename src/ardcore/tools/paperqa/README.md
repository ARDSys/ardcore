# PaperQA Tool

Query scientific papers with citations using PaperQA. Includes a search tool for LLM agents and an index builder.

## Components

### `paperqa_manager.py` - Search Tool
LangChain tool that searches indexed papers and returns cited answers.

```python
from ardcore.tools.paperqa import paperqa_search

# Used by LLM agents to search papers
result = paperqa_search("What are the effects of meditation on anxiety?")
```

**Requirements:**
- Index must be available locally at `~/.paperqa/{CORPUS_NAME}/`
- Use `ensure_paperqa_index_available()` to sync from S3 before first use

### `build_index.py` - Index Builder
Builds PaperQA indexes from PDFs stored in S3.

```python
from ardcore.tools.paperqa.build_index import main

# Build index for a corpus
main(corpus_name="my-corpus")
```

**Features:**
- Incremental updates (skips already-indexed papers)
- Error handling (skips corrupted PDFs)
- Checkpointing (saves progress periodically)
- S3 sync (downloads PDFs, uploads index)

## Configuration

Required environment variables:

```bash
PAPERQA_CORPUS_NAME=my-corpus-name
PAPERQA_PAPERS_BUCKET=s3://your-papers-bucket
PAPERQA_INDEX_BUCKET=s3://your-index-bucket

# AWS credentials
AWS_ACCESS_KEY_ID=your_key_id
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

## Storage Paths

- **Papers (S3)**: `{PAPERQA_PAPERS_BUCKET}/{CORPUS_NAME}/`
- **Index (S3)**: `{PAPERQA_INDEX_BUCKET}/{CORPUS_NAME}/`
- **Index (Local)**: `~/.paperqa/{CORPUS_NAME}/`
