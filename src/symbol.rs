// Copyright (C) 2025 Ryan Daum <ryan.daum@gmail.com> This program is free
// software: you can redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software Foundation, version
// 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// this program. If not, see <https://www.gnu.org/licenses/>.
//

//! Simple case-sensitive symbol interning implementation.
//!
//! This module provides efficient string interning with case-sensitive comparison.
//! Symbols are represented by a single u32 ID that can be stored efficiently in NaNBox.
//!
//! The implementation is thread-safe and uses lock-free data structures where possible.

use ahash::AHasher;
use boxcar::Vec as BoxcarVec;
use once_cell::sync::Lazy;
use papaya::HashMap;
use std::fmt::{Debug, Display};
use std::hash::{BuildHasherDefault, Hash};
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

// ============================================================================
// Cache-aligned atomic counter for performance
// ============================================================================

#[cfg_attr(
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
    ),
    repr(align(128))
)]
#[cfg_attr(
    any(
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64",
    ),
    repr(align(32))
)]
#[cfg_attr(target_arch = "s390x", repr(align(256)))]
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64",
        target_arch = "s390x",
    )),
    repr(align(64))
)]
pub struct CachePadded<T> {
    pub value: T,
}

impl<T> CachePadded<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

// ============================================================================
// Global Interner State
// ============================================================================

struct GlobalInternerState {
    /// Case-sensitive string -> ID mapping
    string_to_id: HashMap<String, u32, BuildHasherDefault<AHasher>>,
    /// Fast reverse lookup: ID as index -> string data
    id_to_string: BoxcarVec<Arc<String>>,
    /// Atomic counter for ID generation
    next_id: CachePadded<AtomicU32>,
    /// Lock for atomic reservation of ID + boxcar slot (only used for NEW symbols)
    allocation_lock: Mutex<()>,
}

impl GlobalInternerState {
    fn new() -> Self {
        Self {
            string_to_id: Default::default(),
            id_to_string: BoxcarVec::new(),
            next_id: CachePadded::new(AtomicU32::new(0)),
            allocation_lock: Mutex::new(()),
        }
    }

    /// Interns a string, returning its unique ID.
    fn intern(&self, s: &str) -> u32 {
        // Pin the map to get a guard
        let guard = self.string_to_id.pin();

        // Try to find existing entry
        if let Some(&existing_id) = guard.get(s) {
            return existing_id;
        }

        // Not found, need to create new entry
        let _lock = self.allocation_lock.lock();

        // Double-check after acquiring lock (another thread might have added it)
        if let Some(&existing_id) = guard.get(s) {
            return existing_id;
        }

        // Allocate new ID and store string
        let arc_string = Arc::new(s.to_string());
        let id = self.id_to_string.push(arc_string.clone()) as u32;

        // Insert into string->id map
        guard.insert(s.to_string(), id);

        id
    }

    fn get_string_by_id(&self, id: u32) -> Option<&Arc<String>> {
        // Fast O(1) lookup using direct boxcar indexing
        self.id_to_string.get(id as usize)
    }
}

static GLOBAL_INTERNER: Lazy<GlobalInternerState> = Lazy::new(GlobalInternerState::new);

// ============================================================================
// Symbol Type
// ============================================================================

/// A case-sensitive interned string.
///
/// Symbols provide efficient storage and comparison for frequently used strings.
/// Each unique string gets a unique ID that can be efficiently stored and compared.
///
/// # Examples
///
/// ```
/// use conat::symbol::Symbol;
///
/// let sym1 = Symbol::mk("hello");
/// let sym2 = Symbol::mk("hello");
/// let sym3 = Symbol::mk("Hello");  // Different from "hello"
///
/// assert_eq!(sym1, sym2);  // Same string, same symbol
/// assert_ne!(sym1, sym3);  // Different case, different symbol
///
/// assert_eq!(sym1.as_string(), "hello");
/// assert_eq!(sym3.as_string(), "Hello");
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol {
    id: u32,
}

// ============================================================================
// Core Symbol Implementation
// ============================================================================

impl Symbol {
    /// Create a new symbol from a string slice.
    ///
    /// This method interns the string, making subsequent creations of symbols
    /// with the same content very fast.
    pub fn mk(s: &str) -> Self {
        let id = GLOBAL_INTERNER.intern(s);
        Symbol { id }
    }

    /// Get the symbol's unique ID.
    ///
    /// This ID can be used with NaNBox for efficient storage.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Create a symbol from a pre-existing ID.
    ///
    /// # Safety
    ///
    /// The caller must ensure the ID corresponds to a valid interned string.
    pub fn from_id(id: u32) -> Self {
        Symbol { id }
    }

    /// Get the original string as an owned `String`.
    pub fn as_string(&self) -> String {
        GLOBAL_INTERNER
            .get_string_by_id(self.id)
            .unwrap_or_else(|| {
                panic!(
                    "Symbol: Invalid ID {}. String not found in interner.",
                    self.id
                )
            })
            .as_ref()
            .clone()
    }

    /// Get the original string as an `Arc<String>`.
    ///
    /// This is more efficient than `as_string()` when you need to share
    /// the string data or when the string will be cloned multiple times.
    pub fn as_arc_string(&self) -> Arc<String> {
        GLOBAL_INTERNER
            .get_string_by_id(self.id)
            .unwrap_or_else(|| {
                panic!(
                    "Symbol: Invalid ID {}. String not found in interner.",
                    self.id
                )
            })
            .clone()
    }

    /// Get the original string as an `Arc<str>`.
    ///
    /// This provides a string slice that can be shared efficiently.
    pub fn as_arc_str(&self) -> Arc<str> {
        let arc_string = self.as_arc_string();
        Arc::from(arc_string.as_str())
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_arc_string().as_ref())
    }
}

impl Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match GLOBAL_INTERNER.get_string_by_id(self.id) {
            Some(s) => f
                .debug_struct("Symbol")
                .field("value", s.as_ref())
                .field("id", &self.id)
                .finish(),
            None => f
                .debug_struct("Symbol")
                .field("value", &"<invalid_id>")
                .field("id", &self.id)
                .finish(),
        }
    }
}

// ============================================================================
// Conversion Traits
// ============================================================================

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Symbol::mk(s)
    }
}

impl From<String> for Symbol {
    fn from(s: String) -> Self {
        Symbol::mk(&s)
    }
}

impl From<&String> for Symbol {
    fn from(s: &String) -> Self {
        Symbol::mk(s.as_str())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::thread;

    #[test]
    fn test_basic_symbol_creation() {
        let sym1 = Symbol::mk("test");
        let sym2 = Symbol::mk("test");
        let sym3 = Symbol::mk("TEST");

        // Same string should produce identical symbols
        assert_eq!(sym1, sym2);
        assert_eq!(sym1.id(), sym2.id());

        // Different case should produce different symbols
        assert_ne!(sym1, sym3);
        assert_ne!(sym1.id(), sym3.id());
    }

    #[test]
    fn test_symbol_string_conversion() {
        let original = "TestString";
        let sym = Symbol::mk(original);

        assert_eq!(sym.as_string(), original);
        assert_eq!(sym.as_arc_string().as_ref(), original);
        assert_eq!(sym.as_arc_str().as_ref(), original);
    }

    #[test]
    fn test_symbol_from_implementations() {
        let sym1 = Symbol::from("test");
        let sym2 = Symbol::from(String::from("test"));
        let s = String::from("test");
        let sym3 = Symbol::from(&s);

        assert_eq!(sym1, sym2);
        assert_eq!(sym2, sym3);
        assert_eq!(sym1.id(), sym2.id());
    }

    #[test]
    fn test_symbol_ordering() {
        let sym_a = Symbol::mk("a");
        let sym_b = Symbol::mk("b");
        let sym_a2 = Symbol::mk("a");

        // Same strings should be equal
        assert_eq!(sym_a, sym_a2);
        assert_eq!(sym_a.cmp(&sym_a2), std::cmp::Ordering::Equal);

        // Different strings should have consistent ordering
        assert_ne!(sym_a, sym_b);
        let ordering = sym_a.cmp(&sym_b);
        assert_eq!(sym_a.cmp(&sym_b), ordering); // Consistent
    }

    #[test]
    fn test_symbol_hashing() {
        let sym1 = Symbol::mk("test");
        let sym2 = Symbol::mk("test");
        let sym3 = Symbol::mk("different");

        let mut map = HashMap::new();
        map.insert(sym1, "value1");

        // Same content should hash to same bucket
        assert_eq!(map.get(&sym2), Some(&"value1"));
        assert_eq!(map.get(&sym3), None);
    }

    #[test]
    fn test_many_symbols() {
        let mut symbols = Vec::new();
        let mut strings = HashSet::new();

        // Create many unique symbols
        for i in 0..1000 {
            let s = format!("symbol_{i}");
            strings.insert(s.clone());
            symbols.push(Symbol::mk(&s));
        }

        // All should be unique by ID
        let mut ids = HashSet::new();
        for sym in &symbols {
            assert!(ids.insert(sym.id()), "Duplicate ID found");
        }

        // All should convert back to original strings
        for (i, sym) in symbols.iter().enumerate() {
            let expected = format!("symbol_{i}");
            assert_eq!(sym.as_string(), expected);
        }
    }

    #[test]
    fn test_concurrent_symbol_creation() {
        let num_threads = 10;
        let symbols_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                thread::spawn(move || {
                    let mut symbols = Vec::new();
                    for i in 0..symbols_per_thread {
                        let s = format!("thread_{thread_id}_{i}");
                        symbols.push(Symbol::mk(&s));
                    }
                    symbols
                })
            })
            .collect();

        let mut all_symbols = Vec::new();
        for handle in handles {
            all_symbols.extend(handle.join().unwrap());
        }

        // Check that all symbols are valid and unique
        let mut ids = HashSet::new();
        for sym in &all_symbols {
            assert!(ids.insert(sym.id()), "Duplicate ID found");
            // Ensure we can still retrieve the string
            assert!(!sym.as_string().is_empty());
        }

        assert_eq!(all_symbols.len(), num_threads * symbols_per_thread);
    }

    #[test]
    fn test_concurrent_same_string() {
        let num_threads = 20;
        let test_string = "concurrent_test";

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                thread::spawn(move || {
                    let mut symbols = Vec::new();
                    for _ in 0..50 {
                        symbols.push(Symbol::mk(test_string));
                    }
                    symbols
                })
            })
            .collect();

        let mut all_symbols = Vec::new();
        for handle in handles {
            all_symbols.extend(handle.join().unwrap());
        }

        // All symbols should be identical
        let first_symbol = all_symbols[0];
        for sym in &all_symbols {
            assert_eq!(*sym, first_symbol);
            assert_eq!(sym.id(), first_symbol.id());
        }
    }

    #[test]
    fn test_from_id() {
        let original = Symbol::mk("test_from_id");
        let reconstructed = Symbol::from_id(original.id());

        assert_eq!(original, reconstructed);
        assert_eq!(original.as_string(), reconstructed.as_string());
    }

    #[test]
    fn test_case_sensitivity() {
        let lower = Symbol::mk("hello");
        let upper = Symbol::mk("HELLO");
        let mixed = Symbol::mk("Hello");

        // All should be different (case-sensitive)
        assert_ne!(lower, upper);
        assert_ne!(lower, mixed);
        assert_ne!(upper, mixed);

        // Should preserve original case
        assert_eq!(lower.as_string(), "hello");
        assert_eq!(upper.as_string(), "HELLO");
        assert_eq!(mixed.as_string(), "Hello");
    }
}
