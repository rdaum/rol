//! Utilities for flexible array member patterns to reduce boilerplate.

use std::mem;
use crate::gc::{is_mmtk_initialized, mmtk_alloc, mmtk_alloc_placeholder, mmtk_dealloc_placeholder};

/// Helper for allocating structs with trailing flexible array members using MMTk.
/// Returns a pointer to the allocated struct with trailing elements.
pub unsafe fn alloc_with_trailing<T, U>(
    element_count: usize,
    init_fn: impl FnOnce(*mut T, *mut U),
) -> *mut T {
    let header_size = mem::size_of::<T>();
    let elements_size = element_count * mem::size_of::<U>();
    let total_size = header_size + elements_size;

    // Round up to 8-byte alignment as required by MMTk
    let aligned_size = (total_size + 7) & !7;

    // Allocate memory using MMTk
    let ptr = if is_mmtk_initialized() {
        mmtk_alloc(aligned_size) as *mut T
    } else {
        mmtk_alloc_placeholder(aligned_size) as *mut T
    };

    if ptr.is_null() {
        panic!("Failed to allocate memory for flexible array struct");
    }

    let elements_ptr = unsafe { trailing_ptr(ptr) };
    init_fn(ptr, elements_ptr);

    ptr
}

/// Get pointer to trailing elements for a flexible array struct.
pub unsafe fn trailing_ptr<T, U>(base_ptr: *mut T) -> *mut U {
    let header_size = mem::size_of::<T>();
    unsafe { (base_ptr as *mut u8).add(header_size) as *mut U }
}

/// Get slice view of trailing elements.
/// SAFETY: caller must ensure element_count is correct and ptr is valid.
pub unsafe fn trailing_slice<'a, T, U>(base_ptr: *const T, element_count: usize) -> &'a [U] {
    let elements_ptr = unsafe { trailing_ptr(base_ptr as *mut T) as *const U };
    unsafe { std::slice::from_raw_parts(elements_ptr, element_count) }
}

/// Get mutable slice view of trailing elements.
/// SAFETY: caller must ensure element_count is correct and ptr is valid.
pub unsafe fn trailing_slice_mut<'a, T, U>(base_ptr: *mut T, element_count: usize) -> &'a mut [U] {
    let elements_ptr = unsafe { trailing_ptr(base_ptr) };
    unsafe { std::slice::from_raw_parts_mut(elements_ptr, element_count) }
}

/// Free memory for a flexible array struct using MMTk deallocation.
/// SAFETY: ptr must have been allocated with alloc_with_trailing and element_count must be correct.
pub unsafe fn free_with_trailing<T, U>(ptr: *mut T, element_count: usize) {
    if ptr.is_null() {
        return;
    }

    let header_size = mem::size_of::<T>();
    let elements_size = element_count * mem::size_of::<U>();
    let total_size = header_size + elements_size;

    unsafe {
        mmtk_dealloc_placeholder(ptr as *mut u8, total_size);
    }
}