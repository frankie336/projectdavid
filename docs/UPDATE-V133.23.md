> **PLEASE NOTE:**
>
>  As of V1.33.23 Thread creation no longer requires participant_ids to be passed in:
>   

```python
thread = client.threads.create_thread(participant_ids=user.id)

# Can be shortened to: 

thread = client.threads.create_thread() 

```


## Function call error surfacing 

Function call error trace stack messages are now surfaced to the message dialogue. If the assistant 
does not proactively say so, you can force this with a followup prompt like:
```python
"What happened?"
```

Sometimes it receives the error but does not proactively reveal.  Please be aware that your consumers 
will have access to stack trace messages. OpenAI do this, the risk is minimal.

# Data Ingestion and Search Methods

## Vector Store Standard Data Ingestion Pipeline

As of **projectdavid v1.33.23** Json output from function calls are now suppressed by default.  
You can reveal Json output them with:

```python

sync_stream.stream_chunks(
            provider=PROVIDER,
            model=MODEL,
            timeout_per_chunk=60.0,
            suppress_fc=True,
        ):
...        
```

Please see [here](https://github.com/frankie336/projectdavid/blob/master/docs/inference.md) 
for  detailed use example.  


## Vector Store Standard Data Ingestion Pipeline

Our standard public ingestion method, `VectorStoreClient.add_file_to_vector_store`, will:

- Pre-process files.
- Chunk files.
- Generate embeddings.
- Upload processed chunks to the specified vector store.
- Prepare file contents for semantic search.

This method is designed primarily for individual files containing mostly unstructured text. It’s powerful and will cover most Retrieval-Augmented Generation (RAG) use cases.

For detailed instructions, please refer to our [usage documentation](https://github.com/frankie336/projectdavid/blob/master/docs/vector_store.md).

You can also find a real-world usage example in our cookbook here:
[Basic Vector Embeddings Search Example](https://github.com/frankie336/entities_cook_book/blob/master/recipes/vector_store/basic_vector_embeddings_search.py).

## Your Vector Store Custom Data Ingestion Pipeline

We directly leverage our embedding model to craft a customized ingestion pipeline using `FileProcessor.embedding_model`. This pipeline:

- Pre-processes structured datasets (e.g., MovieLens).
- Converts each movie record into its own chunk.
- Manually constructs rich text embeddings from multiple metadata fields (title, genres, release year, etc.).

In summary, this custom pipeline is optimized for granular semantic results from structured datasets such as MovieLens. This approach can easily be adapted to other similar datasets, especially useful in recommendation algorithms.

The custom pipeline example is available [here](#).

## Search Methods

As of **projectdavid v1.33.23** ([PyPI Link](https://pypi.org/project/projectdavid/)), the following search methods are available:

### `VectorStoreClient.vector_file_search_raw`

> **PLEASE NOTE:**
>
> This method was previously named `VectorStoreClient.vector_file_search`. Please update your code accordingly when migrating to v1.33.23.

- Returns raw dictionaries with results ranked by corresponding `K` values in descending order.
- Currently used in our semantic search examples on vectorized instances of the MovieLens dataset.

#### Batch Search Example:

[Batch Search on MovieLens](https://github.com/frankie336/entities_cook_book/blob/master/recipes/reccomender/batch_search_movielens.py)

#### Fuzzy Search App Example:

[Fuzzy Search App](https://github.com/frankie336/entities_cook_book/blob/master/recipes/reccomender/search_movielens-v2.py)



### `VectorStoreClient.simple_vector_file_search`

Returns a data structure optimized for interpretation and synthesis by LLM models, suitable for function call returns with potential citations.

**Example Response:**

```json
{
  "object": "vector_store.file_search_result",
  "data": [
    {
      "object": "vector_store.file_hit",
      "index": 0,
      "text": "Title: Toy Story. Genres: Animation, Children's, Comedy. Released in 1995.",
      "score": 0.92,
      "meta_data": {
        "item_id": 1,
        "title": "Toy Story",
        "genres": ["Animation", "Children's", "Comedy"],
        "release_year": 1995,
        "IMDb_URL": "http://www.imdb.com/title/tt0114709/"
      },
      "vector_id": "vec_abc123",
      "store_id": "vect_mqfWyNlZbacer73PQu4Upy"
    },
    {
      "object": "vector_store.file_hit",
      "index": 1,
      "text": "Title: The Lion King. Genres: Animation, Children's, Musical. Released in 1994.",
      "score": 0.89,
      "meta_data": {
        "item_id": 2,
        "title": "The Lion King",
        "genres": ["Animation", "Children's", "Musical"],
        "release_year": 1994,
        "IMDb_URL": "http://www.imdb.com/title/tt0110357/"
      },
      "vector_id": "vec_def456",
      "store_id": "vect_mqfWyNlZbacer73PQu4Upy"
    }
  ],
  "answer": "Here are 2 fun kids' movies from the 1990s: **Toy Story** (1995, Animation/Comedy) and **The Lion King** (1994, Animation/Musical). Both are highly rated family films.",
  "query": "fun kids movies from the 1990s"
}
```

### `VectorStoreClient.attended_file_search`

Utilizes an integrated AI agent to synthesize analysis and employs a specialized post-processing ranking model to ensure highly precise results. Outputs use a similar envelope as `simple_vector_file_search`. Ideal for quick demonstrations or standalone push-button integrations.

### `VectorStoreClient.unattended_file_search`

Employs the same advanced post-processing ranking model as `attended_file_search`, but without integrated synthesis. Suitable for standard function call implementations.