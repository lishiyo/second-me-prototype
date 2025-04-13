# Chat History Storage in the Training Pipeline

## Storage Format

In the training pipeline, chat histories are not stored in a database as a single row. Instead, they are:
- Stored as structured objects in the Chat class (defined in `lpm_kernel/L1/bio.py`)
- Serialized to JSON files during the training preparation process
- Converted to text files for the actual training

The Conversion Flow

Here's the exact process:
1. Collection: Chat histories are collected through the frontend and API interfaces.
2. Structuring: Chat data is stored as Chat objects with the following fields:

```python
class Chat:
       def __init__(
           self,
           sessionId: str = "",
           summary: str = "",
           title: str = "",
           createTime: str = "",
       ) -> None:
           self.session_id = sessionId
           self.summary = summary
           self.title = title
           self.create_time = createTime
```

3. File Generation: During training data preparation, these objects are processed through the L2DataProcessor class, which:
- Splits notes by type (split_notes_by_type method)
- Refines data based on templates (refine_notes_data_subjective and refine_notes_data_objective methods)
- Serializes to JSON files: Each processed note (including chats) is written to JSON files
- Converts JSON to text: The `json_to_txt_each` method converts all data to individual text files:

```python
def json_to_txt_each(
             self, list_processed_notes: List[Note], txt_file_base: str, file_type: str
     ):
         # ...
         for no, item in enumerate(tqdm(list_processed_notes)):
             # Build the txt file path
             txt_file = os.path.join(txt_file_base, f"{file_type}_{no}.txt")
             try:
                 # Ensure the processed field exists in the item
                 if item.processed:
                     with open(txt_file, "w", encoding="utf-8") as tf:
                         tf.write(item.processed)
             # ...
```

4. GraphRAG Indexing: The text data is then indexed using GraphRAG to extract entities and relationships:
```python
def graphrag_indexing(
           self, note_list: List[Note], graph_input_dir: str, output_dir: str, lang: str
   ):
    """Index notes using GraphRAG.
        
        This method configures and runs GraphRAG indexing on the processed notes,
        creating entity and relation extractions.
        
        Args:
            note_list: List of Note objects to index.
            graph_input_dir: Directory containing input files for indexing.
            output_dir: Directory to save indexing results.
            lang: Language for the prompts.
        """
       # Configure and run GraphRAG indexing...
```

5. Training Data Creation: The `create_chat_data` function in `lpm_kernel/L2/utils.py` loads these files and formats them for training:

```python
def create_chat_data(data_args, tokenizer):
       # Load dataset from JSON files
       dataset = load_dataset("json", data_files=data_args.dataset_name, split="train")
       # Process and format each sample
       res_dataset = []
       for case in dataset:
           res_dataset.extend(preprocess(case, data_args.user_name, data_args.is_cot))
       # Create a Dataset object
       res = Dataset.from_list(res_dataset)
       # ...
```

## Directory Structure

The chat data follows this path through the system (IMPORTANT: we need to convert this to Wasabi for storage):
1. Raw chat histories →
2. Chat objects in memory →
3. JSON files in `resources/L2/data_pipeline/raw_data` →
4. Text files in directories like `L1/processed_data/subjective` and `L1/processed_data/objective` →
5. GraphRAG indexed data in `L1/graphrag_indexing_output/[subjective|objective]` →
6. Training dataset loaded by HuggingFace's dataset utilities

## Integration with Memory Systems

Chat data is treated as a "memory" in the system:

```python
class UserInfo:
    def __init__(
        self, cur_time: str, notes: List[Note], todos: List[Todo], chats: List[Chat]
    ):
        # ...
        self.memories = sorted(
            notes + todos + chats,
            key=lambda x: datetime2timestamp(x.create_time),
            reverse=True,
        )
```

This combines chat histories with other forms of memory to create a comprehensive representation of the user's context. Chat histories in the training pipeline should NOT be stored as single rows in a database, but rather as individual files in a structured hierarchy. This approach allows the system to process chat data alongside other types of memories (notes, todos) in a unified way, ensuring the AI self captures the complete context of the user's interactions.

## Current Chat Storage

Currently in the `lpm_kernel`, chat histories are stored in JSON files in several locations. For each of these we need to convert to Wasabi for storage.

`resources/L2/data_pipeline/raw_data/L1/processed_data/` - primary storage path
`resources/L2/data_pipeline/raw_data/L1/processed_data/subjective/` - for subjective data
`resources/L2/data_pipeline/raw_data/L1/processed_data/objective/` - for objective data

The specific code that stores chat histories in these JSON files is in the L2DataProcessor class (in `lpm_kernel/L2/data.py`):

```python
def refine_notes_data_subjective(
        self, note_list: List[Note], user_info: Dict, json_file_remade: str
):
    # Process notes...
    
    json_data_filted = [o.to_json() for o in data_filtered]
    file_dir = os.path.dirname(json_file_remade)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(json_file_remade, "w", encoding="utf-8") as file:
        json.dump(json_data_filted, file, ensure_ascii=False, indent=4)
```

The chat data is converted from Chat objects to JSON using the `to_json()` method. Each chat history includes:
- Session ID
- Summary of the conversation
- Title of the chat
- Creation timestamp

In addition to the raw chat history data, the current `lpm_kernel` implementation creates additional related files:

`resources/L2/data_pipeline/raw_data/id_entity_mapping_subjective_v2.json` - entity mapping
`resources/L1/graphrag_indexing_output/subjective/entities.parquet` - contains structured entity information extracted from chat histories, used for semantic retrieval and training

IMPORTANT: We need to convert all these to Wasabi for storage.