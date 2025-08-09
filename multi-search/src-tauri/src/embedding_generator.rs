use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use unicode_segmentation::UnicodeSegmentation;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EmbeddingRecord {
    pub embedding: Vec<f32>,
    pub text_chunk: String,
    pub document_path: String,
    pub embedding_type: String,
}

#[allow(dead_code)]
pub struct EmbeddingGenerator {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

#[allow(dead_code)]
impl EmbeddingGenerator {
    pub async fn new() -> Result<Self> {
        let device = Device::Cpu;

        let api = Api::new()?;
        let repo = api.repo(Repo::new(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            RepoType::Model,
        ));

        let config_filename = repo.get("config.json").await?;
        let tokenizer_filename = repo.get("tokenizer.json").await?;
        let weights_filename = repo.get("model.safetensors").await?;

        let config_str = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)? 
        };
        let model = BertModel::load(vb, &config)?;

        println!("EmbeddingGenerator model loaded successfully");
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn generate_embeddings_for_document(
        &self,
        title: &str,
        body: &str,
        document_path: &str,
    ) -> Result<Vec<EmbeddingRecord>> {
        let mut records = Vec::new();

        // Process title if not empty
        if !title.trim().is_empty() {
            let title_embedding = self.generate_single_embedding(title)?;
            records.push(EmbeddingRecord {
                embedding: title_embedding,
                text_chunk: title.to_string(),
                document_path: document_path.to_string(),
                embedding_type: "title".to_string(),
            });
        }

        // Process summary if not empty
        let summary = self.summarize_text(body);
        if !summary.trim().is_empty() {
            let summary_embedding = self.generate_single_embedding(&summary)?;
            records.push(EmbeddingRecord {
                embedding: summary_embedding,
                text_chunk: summary,
                document_path: document_path.to_string(),
                embedding_type: "summary".to_string(),
            });
        }

        // Process chunks, filtering out empty ones
        let chunks = self.chunk_text(body);
        for chunk in chunks {
            if !chunk.trim().is_empty() {
                let chunk_embedding = self.generate_single_embedding(&chunk)?;
                records.push(EmbeddingRecord {
                    embedding: chunk_embedding,
                    text_chunk: chunk,
                    document_path: document_path.to_string(),
                    embedding_type: "chunk".to_string(),
                });
            }
        }

        Ok(records)
    }

    pub fn generate_single_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            return Err(anyhow::anyhow!("Cannot generate embedding for empty text"));
        }

        let tokens = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let input_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(tokens.get_attention_mask(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?.unsqueeze(0)?;

        let token_embeddings = self.model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        let expanded_mask = attention_mask.unsqueeze(2)?.expand(token_embeddings.shape())?;
        let masked_embeddings = (token_embeddings * &expanded_mask)?;
        let sum_embeddings = masked_embeddings.sum(1)?;
        let sum_mask = expanded_mask.sum(1)?;
        let mean_pooled_embedding = (sum_embeddings / sum_mask)?;

        let norm = mean_pooled_embedding.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized_embedding = (mean_pooled_embedding / norm)?;

        Ok(normalized_embedding.squeeze(0)?.to_vec1::<f32>()?)
    }

    fn summarize_text(&self, text: &str) -> String {
        // Comprehensive stop words list
        let stop_words: HashSet<&str> = [
            // Articles
            "a", "an", "the",
            // Prepositions
            "in", "on", "at", "by", "for", "with", "without", "through", "during", "before",
            "after", "above", "below", "up", "down", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
            "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
            "should", "now", "into", "about", "against", "between", "across", "behind",
            "beyond", "beside", "beneath", "around", "among", "along", "within", "throughout",
            // Pronouns
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
            "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            // Common verbs
            "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "having", "do", "does", "did", "doing", "would", "could", "should", "may", "might",
            "must", "shall", "will", "can", "get", "got", "getting", "go", "going", "gone",
            "went", "come", "came", "coming", "take", "took", "taken", "taking", "make", "made",
            "making", "see", "saw", "seen", "seeing", "know", "knew", "known", "knowing",
            "think", "thought", "thinking", "say", "said", "saying", "tell", "told", "telling",
            "ask", "asked", "asking", "work", "worked", "working", "seem", "seemed", "seeming",
            "feel", "felt", "feeling", "try", "tried", "trying", "leave", "left", "leaving",
            "call", "called", "calling", "put", "putting", "give", "gave", "given", "giving",
            "find", "found", "finding", "become", "became", "becoming", "look", "looked", "looking",
            "want", "wanted", "wanting", "use", "used", "using", "keep", "kept", "keeping",
            "let", "letting", "begin", "began", "begun", "beginning", "help", "helped", "helping",
            "talk", "talked", "talking", "turn", "turned", "turning", "start", "started", "starting",
            "show", "showed", "shown", "showing", "hear", "heard", "hearing", "play", "played",
            "playing", "run", "ran", "running", "move", "moved", "moving", "live", "lived", "living",
            "believe", "believed", "believing", "hold", "held", "holding", "bring", "brought",
            "bringing", "happen", "happened", "happening", "write", "wrote", "written", "writing",
            "provide", "provided", "providing", "sit", "sat", "sitting", "stand", "stood", "standing",
            "lose", "lost", "losing", "pay", "paid", "paying", "meet", "met", "meeting",
            "include", "included", "including", "continue", "continued", "continuing", "set", "setting",
            "learn", "learned", "learning", "change", "changed", "changing", "lead", "led", "leading",
            "understand", "understood", "understanding", "watch", "watched", "watching", "follow",
            "followed", "following", "stop", "stopped", "stopping", "create", "created", "creating",
            "speak", "spoke", "spoken", "speaking", "read", "reading", "allow", "allowed", "allowing",
            "add", "added", "adding", "spend", "spent", "spending", "grow", "grew", "grown", "growing",
            "open", "opened", "opening", "walk", "walked", "walking", "win", "won", "winning",
            "offer", "offered", "offering", "remember", "remembered", "remembering", "love", "loved",
            "loving", "consider", "considered", "considering", "appear", "appeared", "appearing",
            "buy", "bought", "buying", "wait", "waited", "waiting", "serve", "served", "serving",
            "die", "died", "dying", "send", "sent", "sending", "expect", "expected", "expecting",
            "build", "built", "building", "stay", "stayed", "staying", "fall", "fell", "fallen",
            "falling", "cut", "cutting", "reach", "reached", "reaching", "kill", "killed", "killing",
            "remain", "remained", "remaining",
            // Conjunctions
            "and", "or", "but", "if", "while", "although", "though", "because", "since", "unless",
            "until", "whether", "either", "neither", "both", "not", "only", "also", "however",
            "therefore", "thus", "hence", "moreover", "furthermore", "nevertheless", "nonetheless",
            // Common adverbs
            "always", "never", "often", "sometimes", "usually", "frequently", "rarely", "seldom",
            "hardly", "barely", "nearly", "almost", "quite", "rather", "pretty", "fairly", "really",
            "truly", "actually", "certainly", "definitely", "probably", "possibly", "maybe", "perhaps",
            "obviously", "clearly", "apparently", "evidently", "surely", "indeed", "naturally",
            "unfortunately", "fortunately", "hopefully", "basically", "generally", "specifically",
            "particularly", "especially", "mainly", "mostly", "largely", "primarily", "essentially",
            "effectively", "significantly", "considerably", "substantially", "relatively", "comparatively",
            "extremely", "incredibly", "remarkably", "surprisingly", "interestingly", "importantly",
            "finally", "eventually", "ultimately", "originally", "initially", "previously", "recently",
            "currently", "presently", "immediately", "directly", "instantly", "suddenly", "quickly",
            "slowly", "gradually", "steadily", "constantly", "continuously", "regularly", "occasionally",
            "frequently", "repeatedly", "consistently", "persistently", "thoroughly", "completely",
            "entirely", "totally", "fully", "partially", "partly", "slightly", "somewhat", "fairly",
            // Time indicators
            "today", "tomorrow", "yesterday", "now", "then", "soon", "later", "early", "late",
            "already", "still", "yet", "ago", "recently", "currently", "presently", "immediately",
            "soon", "eventually", "finally", "first", "last", "next", "previous", "following",
            // Quantifiers
            "many", "much", "few", "little", "several", "enough", "plenty", "lots", "tons",
            "numerous", "countless", "various", "different", "certain", "particular", "specific",
            "general", "common", "usual", "normal", "regular", "standard", "typical", "average",
            "ordinary", "simple", "basic", "main", "primary", "principal", "major", "minor",
            "important", "significant", "relevant", "appropriate", "suitable", "proper", "correct",
            "right", "wrong", "good", "bad", "better", "worse", "best", "worst", "great", "excellent",
            "perfect", "fine", "okay", "alright", "nice", "wonderful", "amazing", "incredible",
            "fantastic", "awesome", "terrible", "awful", "horrible", "bad", "poor", "weak",
            "strong", "powerful", "effective", "successful", "useful", "helpful", "valuable",
            "worthwhile", "meaningful", "important", "significant", "relevant", "interesting",
            "exciting", "boring", "dull", "easy", "difficult", "hard", "simple", "complex",
            "complicated", "clear", "obvious", "evident", "apparent", "visible", "hidden",
            "secret", "private", "public", "open", "closed", "available", "possible", "impossible",
            "likely", "unlikely", "certain", "uncertain", "sure", "unsure", "confident", "doubtful",
            // Miscellaneous common words
            "well", "oh", "yes", "no", "okay", "ok", "please", "thanks", "thank", "welcome",
            "sorry", "excuse", "pardon", "hello", "hi", "bye", "goodbye", "dear", "sir", "madam",
            "mr", "mrs", "ms", "dr", "prof", "etc", "ie", "eg", "vs", "via", "per", "re", "ps"
        ].iter().cloned().collect();

        // Split text into sentences
        let sentences: Vec<&str> = text.unicode_sentences().collect();
        
        if sentences.len() <= 3 {
            return text.to_string();
        }

        // Calculate word frequencies (excluding stop words)
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for sentence in &sentences {
            for word in sentence.split_whitespace() {
                let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();
                if clean_word.len() > 2 && !stop_words.contains(clean_word.as_str()) {
                    *word_freq.entry(clean_word).or_insert(0) += 1;
                }
            }
        }

        // Score each sentence based on word frequencies
        let mut sentence_scores: Vec<(usize, f32)> = Vec::new();
        for (i, sentence) in sentences.iter().enumerate() {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            if words.is_empty() {
                sentence_scores.push((i, 0.0));
                continue;
            }

            let mut score = 0.0;
            let mut word_count = 0;
            
            for word in words {
                let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();
                if clean_word.len() > 2 && !stop_words.contains(clean_word.as_str()) {
                    score += *word_freq.get(&clean_word).unwrap_or(&0) as f32;
                    word_count += 1;
                }
            }
            
            // Normalize score by sentence length to avoid bias toward longer sentences
            if word_count > 0 {
                score /= word_count as f32;
            }
            sentence_scores.push((i, score));
        }

        // Sort sentences by score (descending) and take top 3-5
        sentence_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let num_sentences = std::cmp::min(5, std::cmp::max(3, sentences.len() / 3));
        let mut selected_indices: Vec<usize> = sentence_scores
            .iter()
            .take(num_sentences)
            .map(|(i, _)| *i)
            .collect();

        // Sort selected sentences by their original order in the document
        selected_indices.sort();

        // Join the selected sentences
        selected_indices
            .iter()
            .map(|&i| sentences[i].trim())
            .collect::<Vec<&str>>()
            .join(" ")
    }

    fn chunk_text(&self, text: &str) -> Vec<String> {
        // Define our target chunk size in characters.
        const TARGET_CHUNK_SIZE: usize = 1000; // Approx 200-250 tokens
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        // Split the text into sentences using the unicode-segmentation crate.
        // This is a robust form of semantic chunking.
        for sentence in text.unicode_sentences() {
            // Check if adding the new sentence would exceed the limit.
            // Add 1 for the space we'll add.
            if !current_chunk.is_empty() && current_chunk.len() + sentence.len() + 1 > TARGET_CHUNK_SIZE {
                chunks.push(current_chunk);
                current_chunk = String::new();
            }
            // Add a space before the new sentence if the chunk isn't empty.
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
        }

        // Add the last remaining chunk if it's not empty.
        if !current_chunk.trim().is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }
}