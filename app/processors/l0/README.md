# L0 Processing Layer

The L0 layer provides document processing capabilities for deep understanding and insights. It handles document analysis, chunking, embedding generation, and content extraction.

## Key Components

### Document Analysis

The document analysis pipeline consists of:

1. **DocumentInsightGenerator**: Produces deep insights and detailed analysis of documents
2. **DocumentSummaryGenerator**: Creates concise summaries and extracts keywords
3. **DocumentInsightProcessor**: Coordinates the two-stage analysis process

### Prompts-Based Analysis

We've implemented a prompts-based approach for document analysis using structured prompts:

- **DOCUMENT_INSIGHT_PROMPT**: Used to generate comprehensive document insights with a structured breakdown
- **DOCUMENT_SUMMARY_PROMPT**: Used to extract concise summaries and relevant keywords
- **DOCUMENT_TITLE_SUMMARY_PROMPT**: Alternative approach for direct title and summary generation, includes title, summary, keywords

### Bio-Enhanced Processing

We also support personalized document analysis using biographical context:

- The `process_with_bio` method enhances document insights with personalized context
- User biographical information is captured in the `BioInfo` model
- This allows for more relevant and personalized document insights

## Example Usage

See the `examples/simple_document_analyzer.py` file for examples of how to use the various document analysis approaches:

1. Direct use of `DocumentInsightGenerator`
2. Using the alternative `generate_title_and_summary` method
3. Using the two-stage processor with `DocumentInsightProcessor`
4. Enhanced processing with biographical information
5. Using the `InsighterInput` model for structured input

You can also run the full example script:
```
python scripts/test_l0_pipeline.py --path/to/your/file
```

To ingest all the files under `data`:
```
python scripts/process_all_data.py --skip-existing

// To use a custom directory
python scripts/process_all_data.py --data-dir path/to/your/directory
```

To query the relational database:
```
psql -h $(grep DB_HOST .env | cut -d '=' -f2) -p $(grep DB_PORT .env | cut -d '=' -f2) -U $(grep DB_USER .env | cut -d '=' -f2) -d $(grep DB_NAME .env | cut -d '=' -f2) -c "SELECT id, filename, title, processed, chunk_count, insight IS NOT NULL AS has_insight, summary IS NOT NULL AS has_summary FROM documents ORDER BY uploaded_at DESC;"

// For local postgres
psql -h 127.0.0.1 -U postgres -d second-me-prototype -c "SELECT id, filename, title, processed, chunk_count, insight IS NOT NULL AS has_insight, summary IS NOT NULL AS has_summary FROM documents ORDER BY uploaded_at DESC;"

// Check table structure
psql -h 127.0.0.1 -U postgres -d second-me-prototype -c "\d documents"

// check one file
psql -h 127.0.0.1 -U postgres -d second-me-prototype -c "SELECT * FROM documents WHERE filename = '2012-12-06.md';"

psql -h 127.0.0.1 -U postgres -d second-me-prototype
SELECT * FROM documents ORDER BY uploaded_at DESC;
```

## Implementation Details

The implementation is inspired by the approach used in the `lpm_kernel` L0 layer but simplified and adapted for our needs. The key differences are:

- Streamlined prompt structure
- Added support for structured breakdown in insights
- Enhanced error handling and fallback mechanisms
- Biographical context integration for personalization 