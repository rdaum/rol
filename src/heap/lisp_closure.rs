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

//! Native closure type for JIT-compiled functions with captured environments.

use crate::gc::{
    is_mmtk_initialized, mmtk_alloc, mmtk_alloc_placeholder, mmtk_dealloc_placeholder,
};
use crate::var::Var;

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
        let ptr = if is_mmtk_initialized() {
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
