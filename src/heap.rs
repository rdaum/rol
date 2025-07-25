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

//! Native heap-allocated types for Lisp runtime.
//! These types are designed for JIT access and integrate with mmtk GC.

use crate::mmtk_binding::{mmtk_alloc, mmtk_alloc_placeholder, mmtk_dealloc_placeholder};
use crate::var::Var;
use std::ptr;
use std::slice;
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

        // Calculate total size: header + string data
        let header_size = std::mem::size_of::<LispString>();
        let data_size = bytes.len();
        let total_size = header_size + data_size;

        // Round up to 8-byte alignment as required by MMTk
        let aligned_size = (total_size + 7) & !7;

        // Allocate memory using MMTk
        let ptr = if crate::mmtk_binding::is_mmtk_initialized() {
            mmtk_alloc(aligned_size) as *mut LispString
        } else {
            mmtk_alloc_placeholder(aligned_size) as *mut LispString
        };

        if ptr.is_null() {
            panic!("Failed to allocate memory for LispString");
        }

        unsafe {
            // Initialize header
            (*ptr).length = length;

            // Copy string data immediately after header
            let data_ptr = (ptr as *mut u8).add(header_size);
            ptr::copy_nonoverlapping(bytes.as_ptr(), data_ptr, bytes.len());
        }

        ptr
    }

    /// Get the string data as a byte slice
    pub unsafe fn as_bytes(&self) -> &[u8] {
        unsafe {
            let data_ptr =
                (self as *const LispString as *const u8).add(std::mem::size_of::<LispString>());
            slice::from_raw_parts(data_ptr, self.length as usize)
        }
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

        let header_size = std::mem::size_of::<LispString>();
        let total_size = header_size + unsafe { (*ptr).length as usize };

        unsafe {
            mmtk_dealloc_placeholder(ptr as *mut u8, total_size);
        }
    }
}

/// Native vector type optimized for JIT access.
/// Layout: [length: u64][capacity: u64][elements: Var...]
/// Vector data follows immediately after capacity field.
#[repr(C)]
pub struct LispTuple {
    /// Number of elements currently in the vector
    pub length: u64,
    /// Number of elements that can fit without reallocation
    pub capacity: u64,
    // Vector data follows immediately after capacity (flexible array member)
}

impl LispTuple {
    /// Create a new empty LispVector with the given capacity
    pub fn with_capacity(capacity: usize) -> *mut LispTuple {
        let capacity = capacity as u64;

        // Calculate total size: header + element storage
        let header_size = std::mem::size_of::<LispTuple>();
        let elements_size = capacity as usize * std::mem::size_of::<Var>();
        let total_size = header_size + elements_size;

        // Round up to 8-byte alignment as required by MMTk
        let aligned_size = (total_size + 7) & !7;

        // Allocate memory using MMTk
        let ptr = if crate::mmtk_binding::is_mmtk_initialized() {
            mmtk_alloc(aligned_size) as *mut LispTuple
        } else {
            mmtk_alloc_placeholder(aligned_size) as *mut LispTuple
        };

        if ptr.is_null() {
            panic!("Failed to allocate memory for LispVector");
        }

        unsafe {
            // Initialize header
            (*ptr).length = 0;
            (*ptr).capacity = capacity;

            // Zero out element storage
            let data_ptr = (ptr as *mut u8).add(header_size);
            ptr::write_bytes(data_ptr, 0, elements_size);
        }

        ptr
    }

    /// Create a new LispVector from a slice of Vars
    pub fn from_slice(elements: &[Var]) -> *mut LispTuple {
        let ptr = Self::with_capacity(elements.len());

        unsafe {
            (*ptr).length = elements.len() as u64;

            // Copy elements
            let data_ptr = Self::data_ptr(ptr);
            ptr::copy_nonoverlapping(elements.as_ptr(), data_ptr, elements.len());
        }

        ptr
    }

    /// Create a new empty LispVector
    pub fn new() -> *mut LispTuple {
        Self::with_capacity(0)
    }

    /// Get pointer to the element data
    unsafe fn data_ptr(ptr: *mut LispTuple) -> *mut Var {
        unsafe { (ptr as *mut u8).add(std::mem::size_of::<LispTuple>()) as *mut Var }
    }

    /// Get the elements as a slice
    pub unsafe fn as_slice(&self) -> &[Var] {
        unsafe {
            let data_ptr = (self as *const LispTuple as *mut u8)
                .add(std::mem::size_of::<LispTuple>()) as *const Var;
            slice::from_raw_parts(data_ptr, self.length as usize)
        }
    }

    /// Get the elements as a mutable slice
    pub unsafe fn as_mut_slice(&mut self) -> &mut [Var] {
        unsafe {
            let data_ptr = (self as *mut LispTuple as *mut u8).add(std::mem::size_of::<LispTuple>())
                as *mut Var;
            slice::from_raw_parts_mut(data_ptr, self.length as usize)
        }
    }

    /// Push an element to the vector (may reallocate)
    pub unsafe fn push(ptr: *mut LispTuple, element: Var) -> *mut LispTuple {
        unsafe {
            let length = (*ptr).length;
            let capacity = (*ptr).capacity;

            if length < capacity {
                // Space available, just add element
                let data_ptr = Self::data_ptr(ptr);
                let slot_ptr = data_ptr.add(length as usize);

                // Use RAII write barrier for new element
                crate::with_write_barrier!(
                    ptr as *mut u8,
                    slot_ptr,
                    crate::var::Var::none(), // old value is none/uninitialized
                    element => {
                        *slot_ptr = element;
                    }
                );

                (*ptr).length = length + 1;
                ptr
            } else {
                // Need to reallocate
                let new_capacity = if capacity == 0 { 4 } else { capacity * 2 };
                let new_ptr = Self::with_capacity(new_capacity as usize);

                // Copy existing elements with write barriers
                if length > 0 {
                    let old_data = Self::data_ptr(ptr);
                    let new_data = Self::data_ptr(new_ptr);

                    // Copy existing elements with RAII write barriers
                    for i in 0..length as usize {
                        let old_element = *old_data.add(i);
                        let new_slot_ptr = new_data.add(i);

                        crate::with_write_barrier!(
                            new_ptr as *mut u8,
                            new_slot_ptr,
                            crate::var::Var::none(), // new tuple slot starts as none
                            crate::var::Var::from_u64(old_element.as_u64()) => {
                                *new_slot_ptr = old_element;
                            }
                        );
                    }
                }

                // Add new element with RAII write barrier
                (*new_ptr).length = length + 1;
                let new_data = Self::data_ptr(new_ptr);
                let new_slot_ptr = new_data.add(length as usize);

                crate::with_write_barrier!(
                    new_ptr as *mut u8,
                    new_slot_ptr,
                    crate::var::Var::none(),
                    element => {
                        *new_slot_ptr = element;
                    }
                );

                // Free old vector
                Self::free(ptr);

                new_ptr
            }
        }
    }

    /// Free the memory for this LispVector
    pub unsafe fn free(ptr: *mut LispTuple) {
        if ptr.is_null() {
            return;
        }

        let header_size = std::mem::size_of::<LispTuple>();
        let elements_size = unsafe { (*ptr).capacity as usize } * std::mem::size_of::<Var>();
        let total_size = header_size + elements_size;

        unsafe {
            mmtk_dealloc_placeholder(ptr as *mut u8, total_size);
        }
    }
}

// For debugging
impl std::fmt::Debug for LispString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe { write!(f, "LispString({:?})", self.as_str()) }
    }
}

impl std::fmt::Debug for LispTuple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe { write!(f, "LispVector({:?})", self.as_slice()) }
    }
}

/// Native closure type for JIT-compiled functions.
/// Layout: [func_ptr: *const u8][arity: u32][captured_env: u64]
/// A closure contains the compiled function pointer, parameter count, and captured environment.
#[repr(C)]
pub struct LispClosure {
    /// Pointer to the JIT-compiled function
    /// Function signature: fn(args: *const Var, arg_count: u32, captured_env: u64) -> u64
    pub func_ptr: *const u8,
    /// Number of parameters this function expects
    pub arity: u32,
    /// Captured lexical environment (as Var bits)
    pub captured_env: u64,
}

impl LispClosure {
    /// Create a new closure with the given function pointer, arity, and captured environment
    pub fn new(func_ptr: *const u8, arity: u32, captured_env: u64) -> *mut LispClosure {
        let size = std::mem::size_of::<LispClosure>();
        // Round up to 8-byte alignment as required by MMTk
        let aligned_size = (size + 7) & !7;
        let ptr = if crate::mmtk_binding::is_mmtk_initialized() {
            mmtk_alloc(aligned_size) as *mut LispClosure
        } else {
            mmtk_alloc_placeholder(aligned_size) as *mut LispClosure
        };

        if ptr.is_null() {
            panic!("Failed to allocate memory for LispClosure");
        }

        unsafe {
            (*ptr).func_ptr = func_ptr;
            (*ptr).arity = arity;
            (*ptr).captured_env = captured_env;
        }

        ptr
    }

    /// Get the function pointer
    pub unsafe fn get_function_ptr(&self) -> *const u8 {
        self.func_ptr
    }

    /// Get the arity (number of parameters)
    pub unsafe fn get_arity(&self) -> u32 {
        self.arity
    }

    /// Get the captured environment
    pub unsafe fn get_captured_env(&self) -> u64 {
        self.captured_env
    }

    /// Call this closure with the given arguments
    /// Returns the result as a u64 (Var bits)
    pub unsafe fn call(&self, args: &[Var]) -> u64 {
        // Verify arity
        if args.len() != self.arity as usize {
            panic!(
                "Wrong number of arguments: expected {}, got {}",
                self.arity,
                args.len()
            );
        }

        // Cast function pointer to the expected signature
        // Function signature: fn(args: *const Var, arg_count: u32, captured_env: u64) -> u64
        let func: fn(*const Var, u32, u64) -> u64 = unsafe { std::mem::transmute(self.func_ptr) };

        // Call the function
        func(args.as_ptr(), args.len() as u32, self.captured_env)
    }

    /// Free this closure
    pub unsafe fn free(ptr: *mut LispClosure) {
        if ptr.is_null() {
            return;
        }

        let size = std::mem::size_of::<LispClosure>();
        unsafe {
            mmtk_dealloc_placeholder(ptr as *mut u8, size);
        }
    }
}

impl std::fmt::Debug for LispClosure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LispClosure(func_ptr: {:p}, arity: {}, captured_env: 0x{:016x})",
            self.func_ptr, self.arity, self.captured_env
        )
    }
}
