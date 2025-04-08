
# Vector Store

## Overview

A high-performance vector storage and retrieval system designed for AI/ML workflows. This implementation provides:

✨ **Semantic Search**  
🔍 **Nearest Neighbor Retrieval**  
📈 **Scalable Embedding Storage**  
🤖 **ML Framework Integration**

Associated methods can be used to extend the memory and contextual recall of AI assistants beyond the context window, allowing for Retrieval Augmented Generations (RAG).  

## Basic Vector Store Operations

```python
from projectdavid import Entity

client = Entity()

# create a user
test_user = client.users.create_user(name='test_user')
print(test_user)

# create a vector store
store = client.vectors.create_vector_store(
    name='Test Vector Store1',
    user_id=test_user.id,
)
print(store)
```

Will provide metadata about your new store:

```bash
id='vect_WsdjjLHoQqyMLmCdrvShc6' name='Test Vector Store3' user_id='user_gTv1u0Lb97qRpZbbMa4GxF' collection_name='vect_WsdjjLHoQqyMLmCdrvShc6' vector_size=384 distance_metric='COSINE' created_at=1743991638 updated_at=None status=<StatusEnum.active: 'active'> config={} file_count=0
```

You can get the same information with:

```python
retrieve_vector_store = client.vectors.retrieve_vector_store(vector_store_id='vect_WsdjjLHoQqyMLmCdrvShc6')
```

### Attaching a Store to an Assistant

```python
from projectdavid import Entity

client = Entity()

assistant = client.assistants.create_assistant(
    name='movie_db_drone',
    instructions='You will defer to a vector store search for contextual information before every response'
)

attach = client.vectors.attach_vector_store_to_assistant(
    vector_store_id='vect_WsdjjLHoQqyMLmCdrvShc6',
    assistant_id=assistant.id
)
```

Stores attached to an assistant become available to the assistant 
to make semantic searches using its latent logic. No further coding needed. 

### Saving a File to a Store

```python
from projectdavid import Entity

client = Entity()

save_file_to_store = client.vectors.add_file_to_vector_store(
    vector_store_id='vect_WsdjjLHoQqyMLmCdrvShc6',
    file_path='test_file.txt'
)
```

At this point, your file has been vectorized to your store.

---

### Making Searches Against Files in a Store

- The assistant will self-select appropriate vector store 
searches using its latent logic when responding to a prompt.

![Vector Search](/assets/latent_vector_db.png)
![Vector Search](/assets/latent_vector_db2.png)

- The assistant can be asked to search a store directly:

![Vector Search](/assets/direct_vector_search2.png)

## ✅ Supported File Formats in Vector Store

| File Type | Extensions             | Processing Method         | Chunking Strategy            | Notes / Limitations                                                 |
|-----------|------------------------|----------------------------|------------------------------|----------------------------------------------------------------------|
| PDF       | `.pdf`                 | `_process_pdf()`           | Page-based, line-aware       | Uses `pdfplumber`; includes page + line metadata                    |
| Text      | `.txt`, `.md`, `.rst`  | `_process_text()`          | Sentence-aware + token limit | Fallback encodings (UTF-8 / Latin-1); semantic + token chunking     |
| CSV       | `.csv` (special case)  | `process_csv_dynamic()`    | Row-based                    | Uses column `description` by default; all other fields = metadata   |

---

## 📚 Other Public API (Client Methods)

All methods are accessible via `client.vectors`

### Vector Store Lifecycle

```python
create_vector_store(name, user_id, vector_size=384, distance_metric='Cosine') → VectorStoreRead
retrieve_vector_store(vector_store_id) → VectorStoreRead
retrieve_vector_store_by_collection(collection_name) → VectorStoreRead
retrieve_vector_store_sync(vector_store_id) → VectorStoreRead
delete_vector_store(vector_store_id, permanent=False) → dict
```

### File Operations

```python
add_file_to_vector_store(vector_store_id, file_path, user_metadata=None) → VectorStoreRead
delete_file_from_vector_store(vector_store_id, file_path) → dict
list_store_files(vector_store_id) → List[VectorStoreFileRead]
update_vector_store_file_status(vector_store_id, file_id, status, error_message=None) → VectorStoreFileRead
```

### Search

```python
search_vector_store(vector_store_id, query_text, top_k=5, filters=None) → List[dict]
```

### Assistant Integration

```python
attach_vector_store_to_assistant(vector_store_id, assistant_id) → dict
detach_vector_store_from_assistant(vector_store_id, assistant_id) → dict
get_vector_stores_for_assistant(assistant_id) → List[VectorStoreRead]
get_stores_by_user(user_id) → List[VectorStoreRead]
```

---
