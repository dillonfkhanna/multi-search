// ===================================================================
//  IMPORTS
// ===================================================================
// Document parsing crates for different file types
use lopdf::Document;
use pdf_extract::extract_text_from_mem;
use dotext::*;
use docx_rs::{read_docx, DocumentChild, ParagraphChild, RunChild};
use std::path::Path;
use anyhow::Result;

// ===================================================================
//  PUBLIC DISPATCHER FUNCTION
// ===================================================================

/// The single public entry point for the parsers module.
/// It takes a file path, determines the file type, and calls the appropriate parser.
pub fn parse_document(file_path: &Path) -> Result<String> {
    // 1. Get the file extension from the path. If there's no extension, return an error.
    let extension = file_path.extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("File has no extension"))?;

    // 2. Read the entire file into a byte array (`Vec<u8>`).
    let file_bytes = std::fs::read(file_path)
        .map_err(|e| anyhow::anyhow!("Failed to read file {}: {}", file_path.display(), e))?;

    // 3. Use a `match` statement to call the correct private parser based on the extension.
    match extension.to_lowercase().as_str() {
        "txt" | "md" | "log" | "csv" | "json" | "xml" | "html" | "css" | "js" | "ts" | "py" | "rs" | "c" | "cpp" | "h" | "hpp" | "java" | "go" | "php" | "rb" | "swift" | "kt" | "scala" | "sh" | "bat" | "yml" | "yaml" | "toml" | "ini" | "cfg" | "conf" => {
            parse_plain_text(&file_bytes)
        },
        "pdf" => parse_pdf_content(&file_bytes, file_path),
        "docx" => parse_docx_content(&file_bytes, file_path),
        // Add other file types here in the future (xlsx, pptx, odt, etc.)
        _ => Err(anyhow::anyhow!("Unsupported file type: {}", extension)),
    }
}

// ===================================================================
//  PRIVATE PARSER IMPLEMENTATIONS
// ===================================================================

/// Parses plain text files (UTF-8).
fn parse_plain_text(bytes: &[u8]) -> Result<String> {
    // Convert the byte slice to a String, handling potential encoding issues gracefully
    let content = String::from_utf8_lossy(bytes).to_string();
    
    // Return early if content is empty
    if content.trim().is_empty() {
        return Ok(String::new());
    }
    
    // Clean up the content by normalizing line endings and removing excessive whitespace
    let cleaned_content = content
        .lines()
        .map(|line| line.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    
    Ok(cleaned_content)
}

/// Parses PDF files to extract plain text using a hybrid approach for maximum reliability.
fn parse_pdf_content(bytes: &[u8], file_path: &Path) -> Result<String> {
    // Strategy 1: Try lopdf first (more reliable for complex PDFs)
    if let Ok(text) = parse_pdf_with_lopdf(file_path) {
        if !text.trim().is_empty() {
            return Ok(text);
        }
    }
    
    // Strategy 2: Fallback to pdf-extract (simpler, may work for basic PDFs)
    if let Ok(text) = parse_pdf_with_extract(bytes) {
        if !text.trim().is_empty() {
            return Ok(text);
        }
    }
    
    // If both methods fail, return a meaningful error
    Err(anyhow::anyhow!("Failed to extract text from PDF: {}", file_path.display()))
}

/// Primary PDF parsing method using lopdf for maximum reliability.
fn parse_pdf_with_lopdf(file_path: &Path) -> Result<String> {
    let document = Document::load(file_path)
        .map_err(|e| anyhow::anyhow!("Failed to load PDF document: {}", e))?;
    
    let pages = document.get_pages();
    let mut all_text = Vec::new();
    
    for (page_num, _) in pages.iter().enumerate() {
        let page_number = (page_num + 1) as u32;
        match document.extract_text(&[page_number]) {
            Ok(text) => {
                let cleaned_text = text.trim();
                if !cleaned_text.is_empty() {
                    all_text.push(cleaned_text.to_string());
                }
            }
            Err(_) => {
                // Skip pages that can't be processed, but continue with others
                continue;
            }
        }
    }
    
    if all_text.is_empty() {
        return Err(anyhow::anyhow!("No text content found in PDF"));
    }
    
    Ok(all_text.join("\n\n"))
}

/// Fallback PDF parsing method using pdf-extract for simpler cases.
fn parse_pdf_with_extract(bytes: &[u8]) -> Result<String> {
    let text = extract_text_from_mem(bytes)
        .map_err(|e| anyhow::anyhow!("PDF extraction failed: {}", e))?;
    
    if text.trim().is_empty() {
        return Err(anyhow::anyhow!("No text content extracted from PDF"));
    }
    
    Ok(text.trim().to_string())
}

/// Parses DOCX files to extract plain text using a hybrid approach for maximum reliability.
fn parse_docx_content(bytes: &[u8], file_path: &Path) -> Result<String> {
    // Strategy 1: Try dotext first (simple and fast for most cases)
    if let Ok(text) = parse_docx_with_dotext(file_path) {
        if !text.trim().is_empty() {
            return Ok(text);
        }
    }
    
    // Strategy 2: Fallback to docx-rs (more thorough parsing)
    if let Ok(text) = parse_docx_with_docx_rs(bytes) {
        if !text.trim().is_empty() {
            return Ok(text);
        }
    }
    
    // If both methods fail, return a meaningful error
    Err(anyhow::anyhow!("Failed to extract text from DOCX: {}", file_path.display()))
}

/// Primary DOCX parsing method using dotext for speed and simplicity.
fn parse_docx_with_dotext(file_path: &Path) -> Result<String> {
    let mut file = Docx::open(file_path)
        .map_err(|e| anyhow::anyhow!("Failed to open DOCX file: {}", e))?;
    
    let mut content = String::new();
    std::io::Read::read_to_string(&mut file, &mut content)
        .map_err(|e| anyhow::anyhow!("Failed to read DOCX content: {}", e))?;
    
    if content.trim().is_empty() {
        return Err(anyhow::anyhow!("No text content found in DOCX"));
    }
    
    // Clean up the extracted text
    let cleaned_content = content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    
    Ok(cleaned_content)
}

/// Fallback DOCX parsing method using docx-rs for thorough extraction.
fn parse_docx_with_docx_rs(bytes: &[u8]) -> Result<String> {
    let docx = read_docx(bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse DOCX structure: {}", e))?;
    
    let mut text_content = Vec::new();
    
    // Extract text from all paragraphs in the document
    for child in &docx.document.children {
        if let DocumentChild::Paragraph(paragraph) = child {
            let mut paragraph_text = Vec::new();
            
            for para_child in &paragraph.children {
                if let ParagraphChild::Run(run) = para_child {
                    for run_child in &run.children {
                        if let RunChild::Text(text) = run_child {
                            paragraph_text.push(text.text.clone());
                        }
                    }
                }
            }
            
            let paragraph_str = paragraph_text.join("").trim().to_string();
            if !paragraph_str.is_empty() {
                text_content.push(paragraph_str);
            }
        }
    }
    
    if text_content.is_empty() {
        return Err(anyhow::anyhow!("No text content found in DOCX"));
    }
    
    Ok(text_content.join("\n\n"))
}

// ===================================================================
//  UTILITY FUNCTIONS
// ===================================================================

/// Returns true if the given file extension is supported by this parser.
pub fn is_supported_file_type(extension: &str) -> bool {
    matches!(
        extension.to_lowercase().as_str(),
        "txt" | "md" | "log" | "csv" | "json" | "xml" | "html" | "css" | "js" | "ts" | "py" | "rs" | "c" | "cpp" | "h" | "hpp" | "java" | "go" | "php" | "rb" | "swift" | "kt" | "scala" | "sh" | "bat" | "yml" | "yaml" | "toml" | "ini" | "cfg" | "conf" | "pdf" | "docx"
    )
}

/// Returns a list of all supported file extensions.
pub fn supported_extensions() -> Vec<&'static str> {
    vec![
        // Plain text formats
        "txt", "md", "log", "csv", "json", "xml", "html", "css", "js", "ts", "py", "rs", "c", "cpp", "h", "hpp", "java", "go", "php", "rb", "swift", "kt", "scala", "sh", "bat", "yml", "yaml", "toml", "ini", "cfg", "conf",
        // Binary document formats
        "pdf", "docx"
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_is_supported_file_type() {
        assert!(is_supported_file_type("txt"));
        assert!(is_supported_file_type("PDF")); // Test case insensitivity
        assert!(is_supported_file_type("docx"));
        assert!(!is_supported_file_type("unsupported"));
    }

    #[test]
    fn test_supported_extensions_count() {
        let extensions = supported_extensions();
        assert!(!extensions.is_empty());
        assert!(extensions.contains(&"txt"));
        assert!(extensions.contains(&"pdf"));
        assert!(extensions.contains(&"docx"));
    }

    #[test]
    fn test_parse_plain_text() {
        let content = b"Hello\nWorld\n  \nWith spaces  ";
        let result = parse_plain_text(content).unwrap();
        assert_eq!(result, "Hello\nWorld\n\nWith spaces");
    }

    #[test]
    fn test_parse_empty_content() {
        let content = b"   \n\n  \t  ";
        let result = parse_plain_text(content).unwrap();
        assert_eq!(result, "");
    }
}