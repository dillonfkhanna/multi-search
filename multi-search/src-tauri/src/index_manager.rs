use std::time::{SystemTime, UNIX_EPOCH};
use tantivy::collector::TopDocs;
use tantivy::query::{QueryParser, TermQuery};
use tantivy::schema::{Schema, TEXT, STORED, FAST, Field, Value};
// Import the concrete `TantivyDocument` struct and the `doc!` macro
use tantivy::{doc, Index, IndexWriter, DateTime, TantivyDocument, Term};

/// Represents a document from any source, ready to be indexed.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct IndexableDocument {
    pub path: String,
    pub title: String,
    pub body: String,
    pub source_type: String,
    pub author: Option<String>,
    pub modified_date: SystemTime,
    pub content_hash: String,
}

/// A struct to hold the results of a search query.
#[derive(Debug, Clone, serde::Serialize)]
#[allow(dead_code)]
pub struct SearchResult {
    pub path: String,
    pub title: String,
    pub score: f32,
    pub source_type: String,
    pub modified_date: SystemTime,
}

/// Manages the Tantivy keyword index.
#[allow(dead_code)]
pub struct IndexManager {
    index: Index,
    path_field: Field,
    title_field: Field,
    body_field: Field,
    source_type_field: Field,
    author_field: Field,
    modified_date_field: Field,
    content_hash_field: Field,
}

#[allow(dead_code)]
impl IndexManager {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let data_dir = dirs::data_dir().ok_or("Could not find application data directory")?;
        let index_path = data_dir.join("multi-search").join("keyword_index");
        std::fs::create_dir_all(&index_path)?;

        let mut schema_builder = Schema::builder();

        let path_field = schema_builder.add_text_field("path", TEXT | STORED | FAST);
        let title_field = schema_builder.add_text_field("title", TEXT | STORED);
        let body_field = schema_builder.add_text_field("body", TEXT);
        let source_type_field = schema_builder.add_text_field("source_type", TEXT | STORED | FAST);
        let author_field = schema_builder.add_text_field("author", TEXT | STORED);
        let modified_date_field = schema_builder.add_date_field("modified_date", STORED);
        let content_hash_field = schema_builder.add_text_field("content_hash", TEXT | STORED | FAST);

        let schema = schema_builder.build();

        let index = match Index::open_in_dir(&index_path) {
            Ok(index) => index,
            Err(_) => Index::create_in_dir(&index_path, schema.clone())?,
        };

        Ok(IndexManager {
            index,
            path_field,
            title_field,
            body_field,
            source_type_field,
            author_field,
            modified_date_field,
            content_hash_field,
        })
    }

    pub fn add_document_batch(
        &self,
        docs: Vec<IndexableDocument>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut writer: IndexWriter = self.index.writer(100_000_000)?;
        for doc in docs {
            let timestamp_secs = doc.modified_date.duration_since(UNIX_EPOCH)?.as_secs() as i64;
            let datetime = DateTime::from_timestamp_secs(timestamp_secs);
            
            let mut tantivy_doc = TantivyDocument::new();
            tantivy_doc.add_text(self.path_field, &doc.path);
            tantivy_doc.add_text(self.title_field, &doc.title);
            tantivy_doc.add_text(self.body_field, &doc.body);
            tantivy_doc.add_text(self.source_type_field, &doc.source_type);
            tantivy_doc.add_text(self.content_hash_field, &doc.content_hash);
            tantivy_doc.add_date(self.modified_date_field, datetime);
            
            if let Some(author) = &doc.author {
                tantivy_doc.add_text(self.author_field, author);
            }
            
            writer.add_document(tantivy_doc)?;
        }
        writer.commit()?;
        Ok(())
    }


    pub fn search(&self, query_str: &str) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let reader = self.index.reader()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(
            &self.index,
            vec![self.title_field, self.body_field, self.author_field],
        );

        let query = query_parser.parse_query(query_str)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(20))?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            // Retrieve the concrete `TantivyDocument` struct.
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;

            // Use the correct `.as_str()` method to extract the text.
            let path = retrieved_doc.get_first(self.path_field).and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let title = retrieved_doc.get_first(self.title_field).and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let source_type = retrieved_doc.get_first(self.source_type_field).and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let modified_date = retrieved_doc.get_first(self.modified_date_field)
                .and_then(|v| v.as_datetime())
                .map(|d| {
                    let timestamp_secs = d.into_timestamp_secs();
                    UNIX_EPOCH + std::time::Duration::from_secs(timestamp_secs as u64)
                })
                .unwrap_or(SystemTime::UNIX_EPOCH);

            results.push(SearchResult {
                path,
                title,
                score,
                source_type,
                modified_date,
            });
        }

        Ok(results)
    }

    /// Updates a document in the index by deleting the old version and adding the new one.
    pub fn update_document(&self, doc: IndexableDocument) -> Result<(), Box<dyn std::error::Error>> {
        let mut writer: IndexWriter = self.index.writer(100_000_000)?;

        // First, delete the old document by its unique path
        let path_term = Term::from_field_text(self.path_field, &doc.path);
        writer.delete_term(path_term);

        // Then, add the new version of the document
        let timestamp_secs = doc.modified_date.duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let datetime = DateTime::from_timestamp_secs(timestamp_secs);
        
        let mut tantivy_doc = TantivyDocument::new();
        tantivy_doc.add_text(self.path_field, &doc.path);
        tantivy_doc.add_text(self.title_field, &doc.title);
        tantivy_doc.add_text(self.body_field, &doc.body);
        tantivy_doc.add_text(self.source_type_field, &doc.source_type);
        tantivy_doc.add_text(self.content_hash_field, &doc.content_hash);
        tantivy_doc.add_date(self.modified_date_field, datetime);
        
        if let Some(author) = &doc.author {
            tantivy_doc.add_text(self.author_field, author);
        }
        
        writer.add_document(tantivy_doc)?;

        // Commit both the deletion and addition in one transaction
        writer.commit()?;
        Ok(())
    }

    /// Deletes a document from the index using its unique path.
    pub fn delete_document(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut writer: IndexWriter = self.index.writer(100_000_000)?;
        let path_term = Term::from_field_text(self.path_field, path);
        writer.delete_term(path_term);
        writer.commit()?;
        Ok(())
    }

    /// Looks up document metadata by path. Returns None if document is not found.
    pub fn get_document_metadata(&self, path: &str) -> Result<Option<SearchResult>, Box<dyn std::error::Error>> {
        let reader = self.index.reader()?;
        let searcher = reader.searcher();

        // Create a term query for the exact path
        let path_term = Term::from_field_text(self.path_field, path);
        let query = TermQuery::new(path_term, tantivy::schema::IndexRecordOption::Basic);
        
        let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;

        if let Some((score, doc_address)) = top_docs.first() {
            let retrieved_doc: TantivyDocument = searcher.doc(*doc_address)?;

            let path = retrieved_doc.get_first(self.path_field).and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let title = retrieved_doc.get_first(self.title_field).and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let source_type = retrieved_doc.get_first(self.source_type_field).and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let modified_date = retrieved_doc.get_first(self.modified_date_field)
                .and_then(|v| v.as_datetime())
                .map(|d| {
                    let timestamp_secs = d.into_timestamp_secs();
                    UNIX_EPOCH + std::time::Duration::from_secs(timestamp_secs as u64)
                })
                .unwrap_or(SystemTime::UNIX_EPOCH);

            Ok(Some(SearchResult {
                path,
                title,
                score: *score,
                source_type,
                modified_date,
            }))
        } else {
            Ok(None)
        }
    }
}