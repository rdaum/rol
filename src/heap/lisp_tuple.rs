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

//! Native immutable tuple type optimized for JIT access and MMTk garbage collection.

use crate::heap::flexible_utils::{
    alloc_with_trailing, free_with_trailing, trailing_ptr, trailing_slice,
};
use crate::var::Var;
use crate::with_write_barrier;
use std::ptr;

/// Native immutable tuple type optimized for JIT access.
/// Layout: [length: u64][capacity: u64][elements: Var...]
/// Tuple data follows immediately after capacity field.
/// Once created, tuples are immutable - operations create new tuples.
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

        unsafe {
            alloc_with_trailing(
                capacity as usize,
                |ptr: *mut LispTuple, data_ptr: *mut Var| {
                    // Initialize header
                    (*ptr).length = 0;
                    (*ptr).capacity = capacity;

                    // Zero out element storage
                    let elements_size = capacity as usize * std::mem::size_of::<Var>();
                    ptr::write_bytes(data_ptr as *mut u8, 0, elements_size);
                },
            )
        }
    }

    /// Create a new LispVector from a slice of Vars
    pub fn from_slice(elements: &[Var]) -> *mut LispTuple {
        unsafe {
            alloc_with_trailing(elements.len(), |ptr: *mut LispTuple, data_ptr: *mut Var| {
                // Initialize header
                (*ptr).length = elements.len() as u64;
                (*ptr).capacity = elements.len() as u64;

                // Copy elements
                ptr::copy_nonoverlapping(elements.as_ptr(), data_ptr, elements.len());
            })
        }
    }

    /// Create a new empty LispVector
    pub fn new() -> *mut LispTuple {
        Self::with_capacity(0)
    }

    /// Get pointer to the element data
    unsafe fn data_ptr(ptr: *mut LispTuple) -> *mut Var {
        unsafe { trailing_ptr(ptr) }
    }

    /// Get the elements as a slice
    pub unsafe fn as_slice(&self) -> &[Var] {
        unsafe { trailing_slice(self as *const LispTuple, self.length as usize) }
    }

    /// Create a new tuple by prepending an element (cons operation)
    /// This creates a new tuple rather than mutating the existing one
    pub unsafe fn cons(element: Var, ptr: *mut LispTuple) -> *mut LispTuple {
        unsafe {
            let old_length = (*ptr).length;
            let new_length = old_length + 1;
            
            // Create new tuple with space for one more element
            let new_ptr = Self::with_capacity(new_length as usize);
            (*new_ptr).length = new_length;
            
            let old_data = Self::data_ptr(ptr);
            let new_data = Self::data_ptr(new_ptr);
            
            // Set first element to the new element
            with_write_barrier!(
                new_ptr as *mut u8,
                new_data,
                crate::var::Var::none(),
                element => {
                    *new_data = element;
                }
            );
            
            // Copy remaining elements from old tuple
            for i in 0..old_length as usize {
                let old_element = *old_data.add(i);
                let new_slot_ptr = new_data.add(i + 1);
                
                with_write_barrier!(
                    new_ptr as *mut u8,
                    new_slot_ptr,
                    crate::var::Var::none(),
                    old_element => {
                        *new_slot_ptr = old_element;
                    }
                );
            }
            
            new_ptr
        }
    }

    /// Create a new tuple by appending an element 
    /// This creates a new tuple rather than mutating the existing one
    pub unsafe fn append(ptr: *mut LispTuple, element: Var) -> *mut LispTuple {
        unsafe {
            let old_length = (*ptr).length;
            let new_length = old_length + 1;
            
            // Create new tuple with space for one more element
            let new_ptr = Self::with_capacity(new_length as usize);
            (*new_ptr).length = new_length;
            
            let old_data = Self::data_ptr(ptr);
            let new_data = Self::data_ptr(new_ptr);
            
            // Copy existing elements first
            for i in 0..old_length as usize {
                let old_element = *old_data.add(i);
                let new_slot_ptr = new_data.add(i);
                
                with_write_barrier!(
                    new_ptr as *mut u8,
                    new_slot_ptr,
                    crate::var::Var::none(),
                    old_element => {
                        *new_slot_ptr = old_element;
                    }
                );
            }
            
            // Add new element at the end
            let new_slot_ptr = new_data.add(old_length as usize);
            with_write_barrier!(
                new_ptr as *mut u8,
                new_slot_ptr,
                crate::var::Var::none(),
                element => {
                    *new_slot_ptr = element;
                }
            );
            
            new_ptr
        }
    }

    /// Free the memory for this LispVector
    pub unsafe fn free(ptr: *mut LispTuple) {
        if ptr.is_null() {
            return;
        }

        let element_count = unsafe { (*ptr).capacity as usize };
        unsafe { free_with_trailing::<LispTuple, Var>(ptr, element_count) };
    }
}

// For debugging
impl std::fmt::Debug for LispTuple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe { write!(f, "LispVector({:?})", self.as_slice()) }
    }
}
