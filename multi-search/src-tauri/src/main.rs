// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
// Allow warnings from objc crate macros (external dependency issue)
#![allow(unexpected_cfgs)]

use tauri::{Manager, AppHandle};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};
use window_vibrancy::{apply_vibrancy, NSVisualEffectMaterial};

mod index_manager;
mod embedding_generator;
mod vector_db;

#[cfg(target_os = "macos")]
use cocoa::appkit::NSColor;
#[cfg(target_os = "macos")]
use cocoa::base::{id, nil, YES};
#[cfg(target_os = "macos")]
use objc::{msg_send, sel, sel_impl};

#[cfg(target_os = "windows")]
use window_vibrancy::apply_blur;

#[cfg(target_os = "windows")]
use windows_sys::Win32::Foundation::HWND;
#[cfg(target_os = "windows")]
use windows_sys::Win32::Graphics::Dwm::{DwmSetWindowAttribute, DWMWA_WINDOW_CORNER_PREFERENCE};
#[cfg(target_os = "windows")]
use std::mem;

#[cfg(target_os = "macos")]
fn setup_rounded_transparent_window(window: &tauri::WebviewWindow) {
    unsafe {
        let ns_window = window.ns_window().unwrap() as id;
        
        // Make window background completely transparent
        let clear_color = NSColor::clearColor(nil);
        let _: () = msg_send![ns_window, setBackgroundColor: clear_color];
        let _: () = msg_send![ns_window, setOpaque: false];
        
        // Ensure the window doesn't have a background material that causes gray corners
        let _: () = msg_send![ns_window, setHasShadow: false];
        
        // Force the window to ignore key status for visual effects
        let _: () = msg_send![ns_window, setIgnoresMouseEvents: false];
        
        // Get the content view and set up rounded corners
        let content_view: id = msg_send![ns_window, contentView];
        
        // Set layer properties for rounded corners
        let _: () = msg_send![content_view, setWantsLayer: YES];
        let layer: id = msg_send![content_view, layer];
        let _: () = msg_send![layer, setCornerRadius: 24.0f64];
        let _: () = msg_send![layer, setMasksToBounds: YES];
    }
}

#[cfg(target_os = "macos")]
fn force_vibrancy_active(window: &tauri::WebviewWindow) {
    unsafe {
        let ns_window = window.ns_window().unwrap() as id;
        let content_view: id = msg_send![ns_window, contentView];
        
        // Find all NSVisualEffectView subviews and force them to stay active
        let subviews: id = msg_send![content_view, subviews];
        let count: usize = msg_send![subviews, count];
        
        for i in 0..count {
            let subview: id = msg_send![subviews, objectAtIndex: i];
            let class_name: id = msg_send![subview, className];
            let class_string: *const std::os::raw::c_char = msg_send![class_name, UTF8String];
            let class_str = std::ffi::CStr::from_ptr(class_string).to_str().unwrap_or("");
            
            if class_str.contains("NSVisualEffectView") {
                // Force the visual effect view to always stay active
                let _: () = msg_send![subview, setState: 1]; // NSVisualEffectStateActive
            }
        }
    }
}

// Windows-specific constants for corner preferences (Windows 11 Build 22000+)
#[cfg(target_os = "windows")]
const DWMWCP_DEFAULT: u32 = 0;
#[cfg(target_os = "windows")]
const DWMWCP_DONOTROUND: u32 = 1;
#[cfg(target_os = "windows")]
const DWMWCP_ROUND: u32 = 2;
#[cfg(target_os = "windows")]
const DWMWCP_ROUNDSMALL: u32 = 3;

#[cfg(target_os = "windows")]
fn setup_rounded_transparent_window_windows(window: &tauri::WebviewWindow) {
    unsafe {
        if let Ok(hwnd) = window.hwnd() {
            let hwnd = hwnd.0 as HWND;
            
            // Set rounded corners using DWM API (Windows 11 Build 22000+)
            // This will gracefully fail on older Windows versions
            let corner_preference = DWMWCP_ROUND;
            let result = DwmSetWindowAttribute(
                hwnd,
                DWMWA_WINDOW_CORNER_PREFERENCE,
                &corner_preference as *const u32 as *const std::ffi::c_void,
                mem::size_of::<u32>() as u32,
            );
            
            // Note: We don't panic on failure since this feature requires Windows 11
            // On older Windows versions, this will fail gracefully and the window
            // will just not have rounded corners (but will still function normally)
            if result != 0 {
                eprintln!("Warning: Could not set rounded corners on Windows. This feature requires Windows 11 Build 22000+");
            }
        }
    }
}

#[cfg(target_os = "windows")]
fn force_blur_consistency_windows(window: &tauri::WebviewWindow) {
    // Re-apply blur effect to ensure consistency
    // This can be called when the window becomes visible or loses/gains focus
    let _ = apply_blur(window, Some((18, 18, 18, 125)));
}

fn toggle_launcher_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("launcher") {
        if let Ok(true) = window.is_visible() {
            let _ = window.hide();
        } else {
            let _ = window.show();
            let _ = window.set_focus();
            
            #[cfg(target_os = "macos")]
            {
                // Re-force vibrancy to be active when showing the window
                force_vibrancy_active(&window);
            }
            
            #[cfg(target_os = "windows")]
            {
                // Re-force blur consistency when showing the window
                force_blur_consistency_windows(&window);
            }
        }
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            let handle = app.handle().clone();
            let window = app.get_webview_window("launcher").unwrap();

            #[cfg(target_os = "macos")]
            {
                // Set up the transparent window with rounded corners
                setup_rounded_transparent_window(&window);
                
                // Apply vibrancy that should maintain transparency
                apply_vibrancy(&window, NSVisualEffectMaterial::Popover, None, None)
                    .expect("Unsupported platform! 'apply_vibrancy' is only supported on macOS");
                
                // Force the vibrancy to always stay active
                force_vibrancy_active(&window);
            }

            #[cfg(target_os = "windows")]
            {
                // Set up rounded corners using DWM API (Windows 11 Build 22000+)
                setup_rounded_transparent_window_windows(&window);
                
                // Apply blur effect for transparency
                apply_blur(&window, Some((18, 18, 18, 125)))
                    .expect("Unsupported platform! 'apply_blur' is only supported on Windows");
                
                // Force initial blur consistency
                force_blur_consistency_windows(&window);
            }

            app.global_shortcut()
                .on_shortcut("CmdOrCtrl+Shift+Space", move |_app, _shortcut, event| {
                    if event.state == ShortcutState::Pressed {
                        toggle_launcher_window(&handle);
                    }
                })
                .expect("Failed to register global shortcut");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}