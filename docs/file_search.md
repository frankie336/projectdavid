# File Search

## Overview

File search sits on top of the vector‑store lifecycle, but with a few important wrinkles. A vector store is a persistent, embedding‑backed database: every ingested user file is chunked, embedded into a high‑dimensional space, and indexed for approximate‑nearest‑neighbour (ANN) retrieval. File search builds on that foundation, accepting a natural‑language query, converting it into the same embedding space, then returning the *k* most similar chunks—optionally post‑filtered by any Qdrant payload expression (e.g. `{"page": {"$lte": 5}}`). In short, the lifecycle is:

* **Ingest** → chunk, embed, upsert (creates a vector‑store record + file metadata)  
* **Search** → embed the query, run ANN, score + rank  
* **Filter / Facet** → apply payload filters (`page`, `author`, custom tags)  
* **Synthesis (optional)** → feed the top‑N passages into an LLM for abstractive answers  

Because everything is vectorised, file search supports highly flexible, multi‑dimensional semantic retrieval, far beyond simple keyword matching—yet still lets you constrain results with deterministic filters when precision matters.


```python

import os
from pathlib import Path
from dotenv import load_dotenv
from projectdavid import Entity
from projectdavid_common import UtilsInterface   # only for pretty logging


# --------------------------------------------------------------------- #
# environment & client
# --------------------------------------------------------------------- #
load_dotenv()                                        # reads .env in cwd

client = Entity(
    base_url=os.getenv("BASE_URL", "http://localhost:9000"),
    api_key=os.getenv("ENTITIES_API_KEY"),
)
# --------------------------------------------------------------------- #
# 1.  Create a vector store
# --------------------------------------------------------------------- #
store = client.vectors.create_vector_store(
    name="cook‑book‑demo",

# --------------------------------------------------------------------- #
# 2.  Quick diagnostic (optional)
# --------------------------------------------------------------------- #
# sanity = client.vectors.retrieve_vector_store(store.id)
# log.info("Files so far: %d", sanity.file_count)     # should be 0 on first run

file_rec = client.vectors.add_file_to_vector_store(
    vector_store_id=store.id,
    file_path=FILE_PATH,
)
    
# --------------------------------------------------------------------- #
# 3.  Add a file (chunk → embed → upsert → register)
# --------------------------------------------------------------------- #
FILE_PATH = Path("Donoghue_v_Stevenson__1932__UKHL_100__26_May_1932_.pdf")   # any local text file

client.vectors.add_file_to_vector_store(
    vector_store_id=store.id,
    file_path=FILE_PATH,
)

# --------------------------------------------------------------------- #
# 4.  Run a similarity search
# --------------------------------------------------------------------- #
env = client.vectors.search_vector_store_openai(
    vector_store_id=store.id,
    query_text="proof, and in my opinion she is entitled to have an opportunity",
    top_k=3,
)

import json, pprint
pprint.pp(json.dumps(env, indent=2))


env = client.vectors.answer_question(
    vector_store_id=store.id,
    query_text="Explain the neighbour principle in Donoghue v Stevenson.",
)







```