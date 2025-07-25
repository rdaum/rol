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

//! Native string type optimized for JIT access and MMTk garbage collection.

use crate::heap::flexible_utils::{alloc_with_trailing, free_with_trailing, trailing_slice};
use std::ptr;
use std::str;

/// Native string type optimized for JIT access.
/// Layout: [length: u64][bytes: u8...]
/// String data follows immediately after length field.
#[repr(C)]
pub struct LispString {
    /// Length of the string in bytes
    pub length: u64,
    // String data follows immediately after length (flexible array member)
}

impl LispString {
    /// Create a new LispString from a Rust string slice
    pub fn from_str(s: &str) -> *mut LispString {
        let bytes = s.as_bytes();
        let length = bytes.len() as u64;

        unsafe {
            alloc_with_trailing(bytes.len(), |ptr: *mut LispString, data_ptr: *mut u8| {
                // Initialize header
                (*ptr).length = length;

                // Copy string data immediately after header
                ptr::copy_nonoverlapping(bytes.as_ptr(), data_ptr, bytes.len());
            })
        }
    }

    /// Get the string data as a byte slice
    pub unsafe fn as_bytes(&self) -> &[u8] {
        unsafe { trailing_slice(self as *const LispString, self.length as usize) }
    }

    /// Get the string data as a string slice  
    pub unsafe fn as_str(&self) -> &str {
        unsafe { str::from_utf8_unchecked(self.as_bytes()) }
    }

    /// Free the memory for this LispString
    pub unsafe fn free(ptr: *mut LispString) {
        if ptr.is_null() {
            return;
        }

        let element_count = unsafe { (*ptr).length as usize };
        unsafe { free_with_trailing::<LispString, u8>(ptr, element_count) };
    }
}

// For debugging
impl std::fmt::Debug for LispString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe { write!(f, "LispString({:?})", self.as_str()) }
    }
}
