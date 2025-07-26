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

//! Type protocol system for uniform operation dispatch across all Var types.
//! Inspired by Janet's JanetAbstractType system.

use crate::var::{Var, VarType};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A unified protocol for type operations, inspired by Janet's JanetAbstractType.
/// This allows both Rust code and JIT code to dispatch operations uniformly.
pub struct TypeProtocol {
    /// Convert value to string representation
    pub to_string: fn(*const ()) -> String,

    /// Hash the value for use in hash tables
    pub hash: fn(*const ()) -> u64,

    /// Check equality between two values of the same type
    pub equals: fn(*const (), *const ()) -> bool,

    /// Compare two values of the same type (-1, 0, 1)
    pub compare: fn(*const (), *const ()) -> i32,

    /// Get the length/size of the value (tuples, strings, etc.)
    pub length: Option<fn(*const ()) -> i32>,

    /// Get an item by index/key (tuples[index], maps[key])
    pub get: Option<fn(*const (), Var) -> Var>,

    /// Set an item by index/key (mutable operations)
    pub put: Option<fn(*mut (), Var, Var)>,

    /// Get next item for iteration
    pub next: Option<fn(*const (), Var) -> Var>,

    /// Call the value as a function
    pub call: Option<fn(*const (), &[Var]) -> Var>,

    /// Check if the value is truthy
    pub is_truthy: fn(*const ()) -> bool,

    /// Clone/copy the value
    pub clone: fn(*const ()) -> *mut (),

    /// Drop/free the value
    pub drop: fn(*mut ()),
}

/// Get the type protocol for a given VarType
pub fn get_protocol(var_type: VarType) -> &'static TypeProtocol {
    match var_type {
        VarType::None => &NONE_PROTOCOL,
        VarType::Bool => &BOOL_PROTOCOL,
        VarType::I32 => &I32_PROTOCOL,
        VarType::F64 => &F64_PROTOCOL,
        VarType::Symbol => &SYMBOL_PROTOCOL,
        VarType::Tuple => &TUPLE_PROTOCOL,
        VarType::String => &STRING_PROTOCOL,
        VarType::Environment => &ENVIRONMENT_PROTOCOL,
        VarType::Pointer => &POINTER_PROTOCOL,
        VarType::Closure => &CLOSURE_PROTOCOL,
        VarType::Task => &TASK_PROTOCOL,
    }
}

// Protocol implementations for each type
static NONE_PROTOCOL: TypeProtocol = TypeProtocol {
    to_string: |_| "none".to_string(),
    hash: |_| 0,                     // None always hashes to 0
    equals: |_, _| true,             // All None values are equal
    compare: |_, _| 0,               // All None values are equal
    length: None,                    // None has no length
    get: None,                       // None is not indexable
    put: None,                       // None is not mutable
    next: None,                      // None is not iterable
    call: None,                      // None is not callable
    is_truthy: |_| false,            // None is always falsy
    clone: |_| std::ptr::null_mut(), // None doesn't need cloning
    drop: |_| {},                    // None doesn't need dropping
};

static BOOL_PROTOCOL: TypeProtocol = TypeProtocol {
    to_string: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        format!("{}", var.as_bool().unwrap())
    },
    hash: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        if var.as_bool().unwrap() { 1 } else { 0 }
    },
    equals: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        lvar.as_bool() == rvar.as_bool()
    },
    compare: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        match (lvar.as_bool().unwrap(), rvar.as_bool().unwrap()) {
            (false, true) => -1,
            (true, false) => 1,
            _ => 0,
        }
    },
    length: None,
    get: None,
    put: None,
    next: None,
    call: None,
    is_truthy: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        var.as_bool().unwrap()
    },
    clone: |ptr| ptr as *mut (),
    drop: |_| {},
};

static I32_PROTOCOL: TypeProtocol = TypeProtocol {
    to_string: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        format!("{}", var.as_int().unwrap())
    },
    hash: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        var.as_int().unwrap() as u64
    },
    equals: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        lvar.as_int() == rvar.as_int()
    },
    compare: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        let l = lvar.as_int().unwrap();
        let r = rvar.as_int().unwrap();
        if l < r {
            -1
        } else if l > r {
            1
        } else {
            0
        }
    },
    length: None,
    get: None,
    put: None,
    next: None,
    call: None,
    is_truthy: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        var.as_int().unwrap() != 0
    },
    clone: |ptr| ptr as *mut (),
    drop: |_| {},
};

static F64_PROTOCOL: TypeProtocol = TypeProtocol {
    to_string: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        format!("{}", var.as_double().unwrap())
    },
    hash: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        var.as_double().unwrap().to_bits()
    },
    equals: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        lvar.as_double() == rvar.as_double()
    },
    compare: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        let l = lvar.as_double().unwrap();
        let r = rvar.as_double().unwrap();
        l.partial_cmp(&r).map_or(0, |ord| match ord {
            Ordering::Less => -1,
            Ordering::Greater => 1,
            Ordering::Equal => 0,
        })
    },
    length: None,
    get: None,
    put: None,
    next: None,
    call: None,
    is_truthy: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        let val = var.as_double().unwrap();
        val != 0.0 && !val.is_nan()
    },
    clone: |ptr| ptr as *mut (),
    drop: |_| {},
};

static SYMBOL_PROTOCOL: TypeProtocol = TypeProtocol {
    to_string: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        if let Some(sym) = var.as_symbol_obj() {
            sym.as_string().to_string()
        } else {
            format!("sym({})", var.as_symbol().unwrap())
        }
    },
    hash: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        var.as_symbol().unwrap() as u64
    },
    equals: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        lvar.as_symbol() == rvar.as_symbol()
    },
    compare: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        let l = lvar.as_symbol().unwrap();
        let r = rvar.as_symbol().unwrap();
        if l < r {
            -1
        } else if l > r {
            1
        } else {
            0
        }
    },
    length: None,
    get: None,
    put: None,
    next: None,
    call: None,
    is_truthy: |_| true, // Symbols are always truthy
    clone: |ptr| ptr as *mut (),
    drop: |_| {},
};

static TUPLE_PROTOCOL: TypeProtocol = TypeProtocol {
    to_string: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        let tuple = var.as_tuple().unwrap();
        let mut result = String::from("[");
        for (i, item) in tuple.iter().enumerate() {
            if i > 0 {
                result.push_str(", ");
            }
            result.push_str(&format!("{item}"));
        }
        result.push(']');
        result
    },
    hash: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        let tuple = var.as_tuple().unwrap();
        // Simple hash combining tuple length and first few elements
        let mut hash = tuple.len() as u64;
        for (i, item) in tuple.iter().take(3).enumerate() {
            hash ^= item.as_u64().wrapping_mul(i as u64 + 1);
        }
        hash
    },
    equals: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        lvar.as_tuple() == rvar.as_tuple()
    },
    compare: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        let l_tuple = lvar.as_tuple().unwrap();
        let r_tuple = rvar.as_tuple().unwrap();
        match l_tuple.len().cmp(&r_tuple.len()) {
            Ordering::Less => -1,
            Ordering::Greater => 1,
            Ordering::Equal => 0, // Could do lexicographic comparison here
        }
    },
    length: Some(|ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        var.as_tuple().unwrap().len() as i32
    }),
    get: Some(|ptr, index| {
        let var = unsafe { &*(ptr as *const Var) };
        let tuple = var.as_tuple().unwrap();
        if let Some(idx) = index.as_int() {
            if idx >= 0 && (idx as usize) < tuple.len() {
                return tuple[idx as usize];
            }
        }
        Var::none()
    }),
    put: None,  // Tuples are immutable in our implementation
    next: None, // TODO: Implement iteration
    call: None, // Tuples are not callable
    is_truthy: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        !var.as_tuple().unwrap().is_empty()
    },
    clone: |ptr| ptr as *mut (), // Tuples are immutable, can share
    drop: |_| {},                // Memory management handled by Rust
};

static STRING_PROTOCOL: TypeProtocol = TypeProtocol {
    to_string: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        var.as_string().unwrap().to_string()
    },
    hash: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        let s = var.as_string().unwrap();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    },
    equals: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        lvar.as_string() == rvar.as_string()
    },
    compare: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        let l_str = lvar.as_string().unwrap();
        let r_str = rvar.as_string().unwrap();
        match l_str.cmp(r_str) {
            Ordering::Less => -1,
            Ordering::Greater => 1,
            Ordering::Equal => 0,
        }
    },
    length: Some(|ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        var.as_string().unwrap().len() as i32
    }),
    get: Some(|ptr, index| {
        let var = unsafe { &*(ptr as *const Var) };
        let s = var.as_string().unwrap();
        if let Some(idx) = index.as_int() {
            if idx >= 0 && (idx as usize) < s.len() {
                if let Some(ch) = s.chars().nth(idx as usize) {
                    return Var::string(&ch.to_string());
                }
            }
        }
        Var::none()
    }),
    put: None,  // Strings are immutable
    next: None, // TODO: Implement iteration
    call: None, // Strings are not callable
    is_truthy: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        !var.as_string().unwrap().is_empty()
    },
    clone: |ptr| ptr as *mut (), // Strings are immutable, can share
    drop: |_| {},                // Memory management handled by Rust
};

static ENVIRONMENT_PROTOCOL: TypeProtocol = TypeProtocol {
    to_string: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        if let Some(env_ptr) = var.as_environment() {
            unsafe { format!("env(size={})", (*env_ptr).size) }
        } else {
            "env(invalid)".to_string()
        }
    },
    hash: |ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        if let Some(env_ptr) = var.as_environment() {
            env_ptr as u64
        } else {
            0
        }
    },
    equals: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        lvar.as_environment() == rvar.as_environment()
    },
    compare: |lhs, rhs| {
        let lvar = unsafe { &*(lhs as *const Var) };
        let rvar = unsafe { &*(rhs as *const Var) };
        let l_ptr = lvar.as_environment().unwrap_or(std::ptr::null_mut()) as u64;
        let r_ptr = rvar.as_environment().unwrap_or(std::ptr::null_mut()) as u64;
        if l_ptr < r_ptr {
            -1
        } else if l_ptr > r_ptr {
            1
        } else {
            0
        }
    },
    length: Some(|ptr| {
        let var = unsafe { &*(ptr as *const Var) };
        if let Some(env_ptr) = var.as_environment() {
            unsafe { (*env_ptr).size as i32 }
        } else {
            0
        }
    }),
    get: None,                   // Environments are not directly indexed by external code
    put: None,                   // Environments are not directly modified by external code
    next: None,                  // Environments are not iterable
    call: None,                  // Environments are not callable
    is_truthy: |_| true,         // Environments are always truthy
    clone: |ptr| ptr as *mut (), // Environments can be shared
    drop: |_| {},                // Memory management handled by Rust
};

static POINTER_PROTOCOL: TypeProtocol = TypeProtocol {
    to_string: |ptr| format!("ptr({ptr:p})"),
    hash: |ptr| ptr as u64,
    equals: |lhs, rhs| lhs == rhs,
    compare: |lhs, rhs| {
        if lhs < rhs {
            -1
        } else if lhs > rhs {
            1
        } else {
            0
        }
    },
    length: None,
    get: None,
    put: None,
    next: None,
    call: None,
    is_truthy: |_| true, // Pointers are always truthy
    clone: |ptr| ptr as *mut (),
    drop: |_| {},
};

static CLOSURE_PROTOCOL: TypeProtocol = TypeProtocol {
    to_string: |ptr| unsafe {
        let closure = ptr as *const crate::heap::LispClosure;
        format!("closure(arity={})", (*closure).arity)
    },

    // Basic hashing and comparison for closures
    hash: |ptr| ptr as u64,
    equals: |a, b| a == b,
    compare: |a, b| {
        if a < b {
            -1
        } else if a > b {
            1
        } else {
            0
        }
    },

    // Closure-specific operations - return arity as length
    length: Some(|ptr| unsafe {
        let closure = ptr as *const crate::heap::LispClosure;
        (*closure).arity as i32
    }),
    get: Some(|_, _| Var::none()),
    put: None,
    next: None,

    // Closures are callable
    call: Some(|ptr, args| unsafe {
        let closure = ptr as *const crate::heap::LispClosure;
        let result_bits = (*closure).call(args);
        Var::from_u64(result_bits)
    }),

    is_truthy: |_| true, // Closures are always truthy

    // Memory management for closures
    clone: |ptr| {
        // For now, just return the same pointer (shared ownership)
        // In a real implementation, we'd properly clone the closure
        ptr as *mut ()
    },
    drop: |ptr| unsafe {
        crate::heap::LispClosure::free(ptr as *mut crate::heap::LispClosure);
    },
};

static TASK_PROTOCOL: TypeProtocol = TypeProtocol {
    to_string: |ptr| unsafe {
        let task = ptr as *const crate::heap::LispTask;
        format!("task(id={})", (*task).task_id)
    },

    // Basic hashing and comparison for tasks (based on task ID)
    hash: |ptr| unsafe {
        let task = ptr as *const crate::heap::LispTask;
        (*task).task_id
    },
    equals: |a, b| unsafe {
        let task_a = a as *const crate::heap::LispTask;
        let task_b = b as *const crate::heap::LispTask;
        (*task_a).task_id == (*task_b).task_id
    },
    compare: |a, b| unsafe {
        let task_a = a as *const crate::heap::LispTask;
        let task_b = b as *const crate::heap::LispTask;
        let id_a = (*task_a).task_id;
        let id_b = (*task_b).task_id;
        if id_a < id_b {
            -1
        } else if id_a > id_b {
            1
        } else {
            0
        }
    },

    // Task-specific operations
    length: Some(|ptr| unsafe {
        let task = ptr as *const crate::heap::LispTask;
        (*task).globals_count() as i32
    }),
    get: Some(|_, _| crate::var::Var::none()), // Tasks don't support indexing
    put: None,                                 // Tasks are not mutable via protocol
    next: None,                                // Tasks are not iterable

    call: None, // Tasks are not directly callable (use scheduler)

    is_truthy: |_| true, // Tasks are always truthy

    // Memory management for tasks
    clone: |ptr| {
        // For now, just return the same pointer (shared ownership)
        // In a real implementation, we'd properly clone the task
        ptr as *mut ()
    },
    drop: |_ptr| {
        // Task cleanup is handled by the scheduler
        // Don't free directly as tasks may be referenced by scheduler
    },
};
