// ===================================================================
//  IMPORTS
// ===================================================================
use crate::embedding_generator::EmbeddingRecord;
use anyhow::Result;
use arrow::array::{Array, Float32Array, StringArray, FixedSizeListArray};
use arrow::datatypes::{DataType, Field, Schema, Float32Type};
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use lancedb::{connection::Connection, table::Table, query::{QueryBase, ExecutableQuery}};
use futures::TryStreamExt;
use std::sync::Arc;

// ===================================================================
//  PUBLIC STRUCT
// ===================================================================

/// Manages the LanceDB vector database connection and all related operations.
pub struct VectorDBManager {
    _conn: Connection,
    table: Table,
}

// ===================================================================
//  PRIVATE HELPERS
// ===================================================================

impl VectorDBManager {
    /// Creates the Arrow schema for our embeddings table.
    fn create_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("embedding", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                384 // BERT all-MiniLM-L6-v2 produces 384-dimensional embeddings
            ), false),
            Field::new("text_chunk", DataType::Utf8, false),
            Field::new("document_path", DataType::Utf8, false),
            Field::new("embedding_type", DataType::Utf8, false),
        ]))
    }

    /// Converts EmbeddingRecord structs into an Arrow RecordBatch.
    fn records_to_batch(records: &[EmbeddingRecord]) -> Result<RecordBatch> {
        if records.is_empty() {
            return Err(anyhow::anyhow!("Cannot create batch from empty records"));
        }

        // Convert records to Arrow format
        let embeddings: Vec<Option<Vec<Option<f32>>>> = records.iter()
            .map(|record| Some(record.embedding.iter().map(|&v| Some(v)).collect()))
            .collect();
        
        let text_chunks: Vec<&str> = records.iter()
            .map(|record| record.text_chunk.as_str())
            .collect();
            
        let doc_paths: Vec<&str> = records.iter()
            .map(|record| record.document_path.as_str())
            .collect();
            
        let embedding_types: Vec<&str> = records.iter()
            .map(|record| record.embedding_type.as_str())
            .collect();

        // Create Arrow arrays
        let embedding_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            embeddings,
            384
        );
        let text_chunk_array = StringArray::from(text_chunks);
        let doc_path_array = StringArray::from(doc_paths);
        let embedding_type_array = StringArray::from(embedding_types);

        // Create record batch
        let record_batch = RecordBatch::try_new(
            Self::create_schema(),
            vec![
                Arc::new(embedding_array),
                Arc::new(text_chunk_array),
                Arc::new(doc_path_array),
                Arc::new(embedding_type_array),
            ],
        )?;

        Ok(record_batch)
    }

    /// Creates an empty RecordBatch for table initialization.
    fn create_empty_batch() -> Result<RecordBatch> {
        let empty_embedding = vec![Some(vec![Some(0.0f32); 384])];
        let empty_text = vec![""];
        let empty_path = vec![""];
        let empty_type = vec![""];

        let embedding_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            empty_embedding,
            384
        );
        let text_chunk_array = StringArray::from(empty_text);
        let doc_path_array = StringArray::from(empty_path);
        let embedding_type_array = StringArray::from(empty_type);

        let record_batch = RecordBatch::try_new(
            Self::create_schema(),
            vec![
                Arc::new(embedding_array),
                Arc::new(text_chunk_array),
                Arc::new(doc_path_array),
                Arc::new(embedding_type_array),
            ],
        )?;

        Ok(record_batch)
    }

    /// Executes a vector search with the given filter and returns parsed results.
    async fn execute_search(
        &self,
        query_vector: &[f32],
        filter: &str,
        include_text_chunk: bool,
    ) -> Result<Vec<(String, Option<String>, f32)>> {
        let query_vec: Vec<f32> = query_vector.to_vec();
        
        let mut search_results = self.table
            .query()
            .nearest_to(query_vec)?
            .only_if(filter)
            .limit(10)
            .execute()
            .await?;

        let mut parsed_results = Vec::new();
        
        while let Some(batch) = search_results.try_next().await? {
            for i in 0..batch.num_rows() {
                let doc_path_col = batch.column_by_name("document_path")
                    .ok_or_else(|| anyhow::anyhow!("Missing document_path column"))?;
                let distance_col = batch.column_by_name("_distance")
                    .ok_or_else(|| anyhow::anyhow!("Missing _distance column"))?;

                if let (Some(doc_array), Some(dist_array)) = (
                    doc_path_col.as_any().downcast_ref::<StringArray>(),
                    distance_col.as_any().downcast_ref::<Float32Array>()
                ) {
                    let path = doc_array.value(i);
                    let distance = dist_array.value(i);
                    
                    if !doc_array.is_null(i) && !dist_array.is_null(i) {
                        let text_chunk = if include_text_chunk {
                            let text_chunk_col = batch.column_by_name("text_chunk")
                                .ok_or_else(|| anyhow::anyhow!("Missing text_chunk column"))?;
                            
                            if let Some(chunk_array) = text_chunk_col.as_any().downcast_ref::<StringArray>() {
                                if !chunk_array.is_null(i) {
                                    Some(chunk_array.value(i).to_string())
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        
                        parsed_results.push((path.to_string(), text_chunk, distance));
                    }
                }
            }
        }
        
        Ok(parsed_results)
    }

    /// Safely escapes a string for SQL queries.
    /// TODO: Replace with parameterized queries when available in LanceDB.
    fn escape_sql_string(input: &str) -> String {
        // Basic SQL string escaping - replace single quotes with double quotes
        // This is a temporary solution until parameterized queries are available
        input.replace("'", "''")
    }
}

// ===================================================================
//  PUBLIC IMPLEMENTATION
// ===================================================================

impl VectorDBManager {
    /// Creates or opens the LanceDB database and the "embeddings" table.
    /// This is a one-time setup operation.
    pub async fn new() -> Result<Self> {
        // 1. Get the path to the app's data directory
        let data_dir = dirs::data_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find application data directory"))?;
        let db_path = data_dir.join("multi-search").join("vector_store");
        std::fs::create_dir_all(&db_path)?;

        // 2. Connect to the LanceDB database at that path
        let db = lancedb::connect(db_path.to_str().unwrap()).execute().await?;

        // 3. Check if table exists, if not create it
        let table = if db.table_names().execute().await?.contains(&"embeddings".to_string()) {
            // If YES, open existing table
            db.open_table("embeddings").execute().await?
        } else {
            // If NO, create it with empty schema
            let empty_batch = Self::create_empty_batch()?;
            let batch_iterator = RecordBatchIterator::new(
                vec![Ok(empty_batch)].into_iter(),
                Self::create_schema()
            );

            let table = db.create_table("embeddings", Box::new(batch_iterator)).execute().await?;
            
            // Clean up the initialization record
            table.delete("text_chunk = ''").await?;
            
            table
        };

        Ok(VectorDBManager {
            _conn: db,
            table,
        })
    }

    /// Adds a batch of new embedding records to the database.
    pub async fn add_embeddings(&self, records: Vec<EmbeddingRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let record_batch = Self::records_to_batch(&records)?;
        let batch_iterator = RecordBatchIterator::new(
            vec![Ok(record_batch)].into_iter(),
            Self::create_schema()
        );
        
        self.table.add(Box::new(batch_iterator)).execute().await?;
        Ok(())
    }

    /// Deletes all embedding records associated with a specific document path.
    pub async fn delete_document_embeddings(&self, document_path: &str) -> Result<()> {
        let escaped_path = Self::escape_sql_string(document_path);
        let filter_string = format!("document_path = '{}'", escaped_path);
        self.table.delete(&filter_string).await?;
        Ok(())
    }

    /// Updates document embeddings by replacing old records with new ones.
    pub async fn update_document_embeddings(
        &self,
        document_path: &str,
        new_records: Vec<EmbeddingRecord>,
    ) -> Result<()> {
        self.delete_document_embeddings(document_path).await?;
        self.add_embeddings(new_records).await?;
        Ok(())
    }

    // ===================================================================
    //  SEARCH METHODS
    // ===================================================================

    /// Searches for the most similar document titles.
    pub async fn search_titles(&self, query_vector: &[f32]) -> Result<Vec<(String, f32)>> {
        let results = self.execute_search(
            query_vector,
            "embedding_type = 'title'",
            false
        ).await?;
        
        Ok(results.into_iter().map(|(path, _, distance)| (path, distance)).collect())
    }

    /// Searches for the most similar document summaries.
    pub async fn search_summaries(&self, query_vector: &[f32]) -> Result<Vec<(String, f32)>> {
        let results = self.execute_search(
            query_vector,
            "embedding_type = 'summary'",
            false
        ).await?;
        
        Ok(results.into_iter().map(|(path, _, distance)| (path, distance)).collect())
    }

    /// Searches for the most similar text chunks (for finding answers).
    pub async fn search_chunks(&self, query_vector: &[f32]) -> Result<Vec<(String, String, f32)>> {
        let results = self.execute_search(
            query_vector,
            "embedding_type = 'chunk'",
            true
        ).await?;
        
        Ok(results.into_iter()
            .filter_map(|(path, text_chunk, distance)| {
                text_chunk.map(|chunk| (path, chunk, distance))
            })
            .collect())
    }
}