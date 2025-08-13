// ===================================================================
//  IMPORTS
// ===================================================================
// Import all the modules and structs this orchestrator will manage.
use crate::index_manager::{IndexManager, IndexableDocument as KeywordDocument};
use crate::vector_db::VectorDBManager;
use crate::embedding_generator::EmbeddingGenerator;
use anyhow::Result;
use std::sync::Arc; // For sharing state safely across threads
use std::collections::HashMap;
use std::time::{SystemTime, Duration, UNIX_EPOCH};
use sha2::{Sha256, Digest};

// ===================================================================
//  PUBLIC STRUCTS
// ===================================================================

/// The final, rich search result that will be sent to the UI.
#[derive(serde::Serialize)] // So Tauri can convert it to JSON
pub struct HybridSearchResult {
    pub path: String,
    pub title: String,
    pub source_type: String,
    pub modified_date: std::time::SystemTime,
    pub final_score: f32,
    pub best_matching_chunk: Option<String>, // For displaying snippets
}

/// A struct to hold the raw data from a connector before processing.
pub struct RawDocument {
    pub path: String,
    pub title: String,
    pub body: String,
    pub source_type: String,
    pub author: Option<String>,
    pub modified_date: std::time::SystemTime,
}

/// Internal struct to accumulate scores from different search methods.
#[derive(Debug, Clone)]
struct CombinedScore {
    title: String,
    source_type: String,
    modified_date: SystemTime,
    rrf_score: f32,
    best_chunk: Option<String>,
}

/// The central orchestrator that manages all indexing and search operations.
pub struct SearchOrchestrator {
    index_manager: Arc<IndexManager>,
    vector_db: Arc<VectorDBManager>,
    embedding_generator: Arc<EmbeddingGenerator>,
}

// ===================================================================
//  HELPER FUNCTIONS
// ===================================================================

/// Calculates SHA-256 hash of content for deduplication.
fn calculate_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Calculates Reciprocal Rank Fusion (RRF) score for a given rank position.
/// RRF formula: 1 / (k + rank) where k is typically 60.
fn calculate_rrf_score(rank: usize) -> f32 {
    1.0 / (60.0 + rank as f32 + 1.0)
}

/// Calculates a recency score based on how recent a document is.
/// More recent documents get higher scores (0.0 to 1.0).
fn calculate_recency_score(modified_date: SystemTime) -> f32 {
    let now = SystemTime::now();
    let duration_since_epoch = modified_date.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
    let now_since_epoch = now.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
    
    // If document is from the future or same time, give it max score
    if duration_since_epoch >= now_since_epoch {
        return 1.0;
    }
    
    let age = now_since_epoch - duration_since_epoch;
    let age_days = age.as_secs() as f32 / (24.0 * 3600.0);
    
    // Exponential decay: documents older than 365 days get very low scores
    let recency_factor = (-age_days / 365.0).exp();
    recency_factor.max(0.01).min(1.0)
}

// ===================================================================
//  IMPLEMENTATION
// ===================================================================

impl SearchOrchestrator {
    /// Helper method to ensure document metadata exists in combined_scores.
    async fn ensure_metadata_exists(
        &self,
        path: &str,
        combined_scores: &mut HashMap<String, CombinedScore>,
    ) -> Result<()> {
        // If we already have the document metadata, nothing to do
        if combined_scores.contains_key(path) {
            return Ok(());
        }

        // Fetch metadata from IndexManager (using spawn_blocking for synchronous database access)
        let index_manager_clone = Arc::clone(&self.index_manager);
        let path_clone = path.to_string();
        let metadata = tokio::task::spawn_blocking(move || {
            index_manager_clone.get_document_metadata(&path_clone)
                .map_err(|e| anyhow::anyhow!("Failed to fetch document metadata: {}", e))
        }).await
            .map_err(|e| anyhow::anyhow!("Metadata fetch task failed: {}", e))??;

        let score_data = if let Some(metadata) = metadata {
            // Use real metadata from the index
            CombinedScore {
                title: metadata.title,
                source_type: metadata.source_type,
                modified_date: metadata.modified_date,
                rrf_score: 0.0,
                best_chunk: None,
            }
        } else {
            // Document not found in keyword index - this can happen if it was
            // indexed only in vector DB or there's an inconsistency
            CombinedScore {
                title: format!("Document: {}", path.split('/').last().unwrap_or("Unknown")),
                source_type: "Unknown".to_string(),
                modified_date: SystemTime::UNIX_EPOCH,
                rrf_score: 0.0,
                best_chunk: None,
            }
        };

        combined_scores.insert(path.to_string(), score_data);
        Ok(())
    }

    /// Asynchronously creates a new SearchOrchestrator.
    /// This is a heavy, one-time operation that initializes all underlying managers.
    pub async fn new() -> Result<Self> {
        // 1. Initialize each of the core modules. The `await` keyword is used
        //    because the model loading and DB connection are async operations.
        let index_manager = IndexManager::new().map_err(|e| anyhow::anyhow!("Failed to create IndexManager: {}", e))?;
        let embedding_generator = EmbeddingGenerator::new().await?;
        let vector_db = VectorDBManager::new().await?;

        // 2. Wrap each manager in an Arc (Atomic Reference Counter) to allow them
        //    to be shared safely and efficiently across multiple threads.
        Ok(Self {
            index_manager: Arc::new(index_manager),
            vector_db: Arc::new(vector_db),
            embedding_generator: Arc::new(embedding_generator),
        })
    }

    // ===================================================================
    //  DOCUMENT LIFECYCLE METHODS
    // ===================================================================

    /// Processes and indexes a single new document.
    pub async fn index_document(&self, doc: RawDocument) -> Result<()> {
        // 1. Calculate the content hash for deduplication.
        let content_hash = calculate_hash(&doc.body);

        // 2. Create the `KeywordDocument` for the Tantivy index.
        let keyword_doc = KeywordDocument {
            path: doc.path.clone(),
            title: doc.title.clone(),
            body: doc.body.clone(),
            source_type: doc.source_type.clone(),
            author: doc.author,
            modified_date: doc.modified_date,
            content_hash,
        };

        // 3. Generate all the embeddings for the document (using spawn_blocking for CPU-intensive work).
        let embedding_generator_clone = Arc::clone(&self.embedding_generator);
        let title_clone = doc.title.clone();
        let body_clone = doc.body.clone();
        let path_clone = doc.path.clone();
        let embedding_records = tokio::task::spawn_blocking(move || {
            embedding_generator_clone.generate_embeddings_for_document(&title_clone, &body_clone, &path_clone)
        }).await??;

        // 4. Use `tokio::join!` to save to both databases concurrently for performance.
        let (keyword_result, vector_result) = tokio::join!(
            async {
                let index_manager_clone = Arc::clone(&self.index_manager);
                tokio::task::spawn_blocking(move || {
                    index_manager_clone.add_document_batch(vec![keyword_doc])
                        .map_err(|e| anyhow::anyhow!("Keyword indexing failed: {}", e))
                }).await
                    .map_err(|e| anyhow::anyhow!("Keyword indexing task failed: {}", e))?
            },
            async {
                self.vector_db.add_embeddings(embedding_records).await
            }
        );

        // 5. Check for errors and return the result.
        keyword_result?;
        vector_result?;
        Ok(())
    }

    /// Deletes a document from both databases using its unique path.
    pub async fn delete_document(&self, path: &str) -> Result<()> {
        // 1. Use `tokio::join!` to delete from both databases concurrently.
        let (keyword_result, vector_result) = tokio::join!(
            async {
                let index_manager_clone = Arc::clone(&self.index_manager);
                let path_clone = path.to_string();
                tokio::task::spawn_blocking(move || {
                    index_manager_clone.delete_document(&path_clone)
                        .map_err(|e| anyhow::anyhow!("Keyword deletion failed: {}", e))
                }).await
                    .map_err(|e| anyhow::anyhow!("Keyword deletion task failed: {}", e))?
            },
            async {
                self.vector_db.delete_document_embeddings(path).await
            }
        );

        // 2. Check for errors.
        keyword_result?;
        vector_result?;
        Ok(())
    }

    /// Updates a document by deleting the old versions and indexing the new version.
    pub async fn update_document(&self, doc: RawDocument) -> Result<()> {
        // 1. First, delete the old document from both stores to ensure a clean state.
        self.delete_document(&doc.path).await?;
        // 2. Then, index the new version of the document.
        self.index_document(doc).await?;
        Ok(())
    }

    // ===================================================================
    //  HYBRID SEARCH METHOD
    // ===================================================================

    /// Performs a hybrid search and returns an intelligently ranked list of results.
    pub async fn hybrid_search(&self, query: &str) -> Result<Vec<HybridSearchResult>> {
        // Ranking weight constants for easy tuning
        const KEYWORD_BOOST: f32 = 1.2;
        const TITLE_BOOST: f32 = 1.1;
        const RECENCY_WEIGHT: f32 = 0.3;
        const RRF_WEIGHT: f32 = 0.7;
        // --- STAGE 1: PARALLEL RETRIEVAL ---
        // 1. Generate the query embedding once (using spawn_blocking for CPU-intensive work).
        let embedding_generator_clone = Arc::clone(&self.embedding_generator);
        let query_clone = query.to_string();
        let query_embedding = tokio::task::spawn_blocking(move || {
            embedding_generator_clone.generate_single_embedding(&query_clone)
        }).await??;

        // 2. Use `tokio::join!` to run all four searches concurrently.
        let (
            keyword_results,
            title_results,
            summary_results,
            chunk_results
        ) = tokio::join!(
            async {
                let index_manager_clone = Arc::clone(&self.index_manager);
                let query_clone = query.to_string();
                tokio::task::spawn_blocking(move || {
                    index_manager_clone.search(&query_clone)
                        .map_err(|e| anyhow::anyhow!("Keyword search failed: {}", e))
                }).await
                    .map_err(|e| anyhow::anyhow!("Keyword search task failed: {}", e))?
            },
            async {
                self.vector_db.search_titles(&query_embedding).await
            },
            async {
                self.vector_db.search_summaries(&query_embedding).await
            },
            async {
                self.vector_db.search_chunks(&query_embedding).await
            }
        );

        // Handle any errors from the parallel searches
        let keyword_results = keyword_results?;
        let title_results = title_results?;
        let summary_results = summary_results?;
        let chunk_results = chunk_results?;

        // --- STAGE 2: INTELLIGENT RE-RANKING ---
        // 3. Create a HashMap to store the combined scores for each unique document path.
        let mut combined_scores: HashMap<String, CombinedScore> = HashMap::new();

        // 4. Process keyword results and apply Reciprocal Rank Fusion (RRF).
        //    For each result, add its RRF score to the combined score for that path.
        //    Also, store the document's metadata (title, date, etc.).
        for (rank, result) in keyword_results.iter().enumerate() {
            let rrf_score = calculate_rrf_score(rank);
            
            combined_scores.entry(result.path.clone())
                .and_modify(|score| score.rrf_score += rrf_score * KEYWORD_BOOST) // Boost keyword matches
                .or_insert_with(|| CombinedScore {
                    title: result.title.clone(),
                    source_type: result.source_type.clone(),
                    modified_date: result.modified_date,
                    rrf_score: rrf_score * KEYWORD_BOOST,
                    best_chunk: None,
                });
        }

        // 5. Process semantic title results.
        //    For each result, add its RRF score to the combined score for that path.
        for (rank, (path, _distance)) in title_results.iter().enumerate() {
            let rrf_score = calculate_rrf_score(rank);
            
            self.ensure_metadata_exists(path, &mut combined_scores).await?;
            let score_data = combined_scores.get_mut(path).unwrap();
            score_data.rrf_score += rrf_score * TITLE_BOOST; // Boost title matches
        }

        // 6. Process semantic summary results.
        //    For each result, add its RRF score to the combined score.
        for (rank, (path, _distance)) in summary_results.iter().enumerate() {
            let rrf_score = calculate_rrf_score(rank);
            
            self.ensure_metadata_exists(path, &mut combined_scores).await?;
            let score_data = combined_scores.get_mut(path).unwrap();
            score_data.rrf_score += rrf_score;
        }

        // 7. Process semantic chunk results.
        //    For each result, add its RRF score and store the `best_matching_chunk`.
        for (rank, (path, chunk_text, _distance)) in chunk_results.iter().enumerate() {
            let rrf_score = calculate_rrf_score(rank);
            
            self.ensure_metadata_exists(path, &mut combined_scores).await?;
            let score_data = combined_scores.get_mut(path).unwrap();
            score_data.rrf_score += rrf_score;
            // Keep the best chunk (first one found, as results are sorted by relevance)
            if score_data.best_chunk.is_none() {
                score_data.best_chunk = Some(chunk_text.clone());
            }
        }

        // 8. Calculate the final score for every candidate document.
        let mut final_results = Vec::new();
        for (path, score_data) in combined_scores {
            // Calculate a recency score (e.g., from 0.0 to 1.0) based on `modified_date`.
            let recency_score = calculate_recency_score(score_data.modified_date);

            // Apply our final weighted formula.
            let final_score = (RECENCY_WEIGHT * recency_score) + (RRF_WEIGHT * score_data.rrf_score);

            final_results.push(HybridSearchResult {
                path,
                title: score_data.title,
                source_type: score_data.source_type,
                modified_date: score_data.modified_date,
                final_score,
                best_matching_chunk: score_data.best_chunk,
            });
        }

        // 9. Sort the final list by the `final_score` in descending order.
        final_results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());

        // 10. (Future Step) Apply result collapsing for similar documents here.

        // 11. Return the top N results.
        Ok(final_results.into_iter().take(20).collect())
    }
}