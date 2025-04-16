# Running list of Todos to circle back on

1. `generate_topics_for_shades`: 
- We are modifying `default_outlier_cutoff_distance` from 0.5 in topic_generator to 0.1 to reduce the initial outliers in cold start.
- We reduced the size_threshold in the cold start and update to filter out fewer clusters (from sqrt to sqrt * 0.8)
- TODO: review this and figure out if we need to go back to defaults


## Larger Todos

How to integrate chat histories, voting histories, and other sources of data?
- add to training pipeline
- integrate a RAG approach alongside the fine-tuned model

Chat histories need particular thinking.