# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Frontend Development:**
- `npm run dev` - Start development server (Vite)
- `npm run build` - Build frontend (TypeScript compilation + Vite build)
- `npm run preview` - Preview production build

**Tauri Development:**
- `npm run tauri dev` - Start Tauri development mode (runs both frontend and backend)
- `npm run tauri build` - Build complete Tauri application
- `npm run tauri` - Access Tauri CLI commands

**Rust Backend:**
- `cargo check` - Quick compile check (in src-tauri/)
- `cargo build` - Build Rust backend (in src-tauri/)
- `cargo test` - Run Rust tests (in src-tauri/)

## Architecture Overview

**Multi-Search** is a Tauri-based desktop application that provides intelligent search capabilities across multiple data sources. The app features a global hotkey launcher (Cmd/Ctrl+Shift+Space) that opens a floating search interface.

### Key Components

**Frontend (React + TypeScript + Vite):**
- `src/App.tsx` - Main router with two routes: `/launcher` (SearchBar) and `/` (MainPage settings)
- `src/SearchBar.tsx` - Core search interface with typing animations and real-time search
- Uses React Router for navigation between launcher and settings windows

**Backend (Rust + Tauri):**
- `src-tauri/src/main.rs` - Main Tauri application with window management and global shortcuts
- `src-tauri/src/index_manager.rs` - Tantivy-based full-text search indexing system
- `src-tauri/src/embedding_generator.rs` - BERT-based semantic search using Candle ML framework

**Search Architecture:**
- **Keyword Search**: Tantivy index with document storage at `~/AppData/multi-search/keyword_index`
- **Semantic Search**: BERT embeddings using `sentence-transformers/all-MiniLM-L6-v2` model
- **Document Processing**: Automatic chunking, summarization, and embedding generation

**Window System:**
- **Main Window**: Settings interface (800x600)
- **Launcher Window**: Floating search bar (640x80) with transparency effects
- Platform-specific visual effects: macOS vibrancy, Windows blur/rounded corners

### Dependencies

**Key Rust Crates:**
- `tauri` - Desktop app framework with macOS private API access
- `tantivy` - Full-text search engine
- `candle-*` - Machine learning framework for BERT embeddings
- `tokenizers` - Text tokenization for ML models
- `window-vibrancy` - Platform-specific window effects
- `tauri-plugin-global-shortcut` - Global hotkey support

**Key Frontend Dependencies:**
- `react` + `react-dom` - UI framework
- `react-router-dom` - Client-side routing
- `@tauri-apps/api` - Tauri frontend bindings

### Data Storage

- Search indexes stored in system data directory (`dirs::data_dir()`)
- Document embeddings and metadata managed by IndexManager
- Content hashing for deduplication and change detection

### Platform-Specific Features

**macOS:**
- Uses Cocoa APIs for advanced window customization
- Vibrancy effects with transparent backgrounds
- Rounded corners with shadow management

**Windows:**
- DWM API integration for rounded corners (Windows 11+)
- Blur effects for transparency simulation
- Graceful fallback for older Windows versions