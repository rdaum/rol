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

//! A variant container that does a kind of NaN-boxing holding the following kinds of values all
//! in a single 64-bit value:
//!     - a "none" value
//!     - a boolean true/false value
//!     - a signed 32 bit integer
//!     - a 64-bit float without NaN
//!     - a Symbol
//!     - a pointer somewhere on the heap (from/into Box)
//! TODO: Will not work on BigEndian platforms
//! Based on https://github.com/zuiderkwast/Var/blob/master/Var.h

use crate::gc::{is_mmtk_initialized, register_var_as_root};
use crate::heap::{Environment, LispClosure, LispString, LispTuple};
use crate::protocol::{TypeProtocol, get_protocol};
use crate::symbol::Symbol;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Rem, Sub};

pub const BOOLEAN_FALSE: u64 = 0x00; // Perfect for truth tables!
pub const BOOLEAN_TRUE: u64 = 0x01;
pub const NULL: u64 = 0x02; // None/unbound variable

pub const MIN_NUMBER: u64 = 0x0006000000000000;
pub const HIGH16_TAG: u64 = 0xffff000000000000;

pub const DOUBLE_ENCODE_OFFSET: u64 = 0x0007000000000000;

// Pointers must have low 2 bits = 00 and be > 0x02 to avoid boolean/none collision
pub const MIN_POINTER: u64 = 0x04;

// Pointer type tags (using high bits that are typically unused in user pointers)
pub const GENERIC_POINTER_TAG: u64 = 0x0000000000000000;
pub const TUPLE_POINTER_TAG: u64 = 0x1000000000000000;
pub const STRING_POINTER_TAG: u64 = 0x2000000000000000;
pub const CLOSURE_POINTER_TAG: u64 = 0x3000000000000000;
pub const ENVIRONMENT_POINTER_TAG: u64 = 0x5000000000000000;
pub const POINTER_TAG_MASK: u64 = 0xF000000000000000;

pub const SYMBOL_TAG: u64 = 0x0009000000000000;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Var(ValueUnion);

// SAFETY: Var contains pointers that are managed by our GC system.
// The GC ensures that these pointers remain valid during collection cycles.
// Since all heap objects are immutable after creation (except for reference updates
// during collection), it's safe to share Var values between threads.
unsafe impl Send for Var {}
unsafe impl Sync for Var {}

#[repr(C)]
#[derive(Clone, Copy)]
union ValueUnion {
    ptr: *const (),
    value: u64,
    as_words: [u32; 2],
}

#[repr(u8)]
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Copy)]
pub enum VarType {
    None,
    I32,
    F64,
    Bool,
    Pointer,
    Symbol,
    Tuple,
    String,
    Environment,
    Closure,
}

impl Default for Var {
    fn default() -> Self {
        Self::new()
    }
}

impl Var {
    pub fn get_type(&self) -> VarType {
        // Check in order of value ranges for efficiency - use early returns
        if self.is_boolean() {
            return VarType::Bool;
        }
        if self.is_none() {
            return VarType::None;
        }
        if self.is_symbol() {
            return VarType::Symbol;
        }
        if self.is_tuple() {
            return VarType::Tuple;
        }
        if self.is_string() {
            return VarType::String;
        }
        if self.is_environment() {
            return VarType::Environment;
        }
        if self.is_closure() {
            return VarType::Closure;
        }
        if self.is_int() {
            return VarType::I32;
        }
        if self.is_double() {
            return VarType::F64;
        }
        if self.is_pointer() {
            return VarType::Pointer;
        }

        panic!("Unknown Var type: 0x{:016x}", unsafe { self.0.value });
    }

    pub fn new() -> Self {
        Self(ValueUnion { value: NULL })
    }

    pub fn none() -> Self {
        Self::new()
    }

    pub fn int(value: i32) -> Self {
        value.into()
    }

    pub fn float(value: f64) -> Self {
        value.into()
    }

    pub fn bool(value: bool) -> Self {
        value.into()
    }

    pub fn symbol(id: u32) -> Self {
        let mut value = id as u64;
        value |= SYMBOL_TAG;
        Self(ValueUnion { value })
    }

    pub fn from_symbol(sym: Symbol) -> Self {
        Self::symbol(sym.id())
    }

    pub fn tuple(elements: &[Var]) -> Self {
        let ptr = LispTuple::from_slice(elements);
        let var = unsafe { Self::mk_tagged_pointer(ptr, TUPLE_POINTER_TAG) };

        // Register the newly allocated tuple as a thread root immediately
        // This prevents GC from collecting it if a collection happens during evaluation
        if is_mmtk_initialized() {
            register_var_as_root(var, false); // false = thread root
        }

        var
    }

    pub fn empty_tuple() -> Self {
        let ptr = LispTuple::new() as u64;
        let tagged_ptr = ptr | TUPLE_POINTER_TAG;
        let var = Self(ValueUnion { value: tagged_ptr });

        // Register the newly allocated empty tuple as a thread root immediately
        // This prevents GC from collecting it if a collection happens during evaluation
        if is_mmtk_initialized() {
            register_var_as_root(var, false); // false = thread root
        }

        var
    }

    pub fn string(value: &str) -> Self {
        let ptr = LispString::from_str(value);
        let var = unsafe { Self::mk_tagged_pointer(ptr, STRING_POINTER_TAG) };

        // Register the newly allocated string as a thread root immediately
        // This prevents GC from collecting it if a collection happens during evaluation
        if is_mmtk_initialized() {
            register_var_as_root(var, false); // false = thread root
        }

        var
    }

    pub fn environment(env: *mut Environment) -> Self {
        let ptr = env as u64;
        let tagged_ptr = ptr | ENVIRONMENT_POINTER_TAG;
        let var = Self(ValueUnion { value: tagged_ptr });

        // Register the environment as a thread root if it contains heap objects
        // This prevents GC from collecting the environment during evaluation
        if is_mmtk_initialized() {
            register_var_as_root(var, false); // false = thread root
        }

        var
    }

    /// Create a Var from a closure pointer
    pub fn closure(ptr: *mut LispClosure) -> Self {
        let ptr_bits = ptr as u64;
        let tagged_ptr = ptr_bits | CLOSURE_POINTER_TAG;
        let var = Self(ValueUnion { value: tagged_ptr });

        // Register the closure as a thread root immediately
        // This prevents GC from collecting it during evaluation
        if is_mmtk_initialized() {
            register_var_as_root(var, false); // false = thread root
        }

        var
    }

    pub fn is_none(&self) -> bool {
        unsafe { self.0.value == NULL }
    }

    pub fn is_boolean(&self) -> bool {
        let v = unsafe { self.0.value };
        v == BOOLEAN_FALSE || v == BOOLEAN_TRUE
    }

    pub fn is_number(&self) -> bool {
        let v = unsafe { self.0.value };
        v >= MIN_NUMBER
            && !self.is_symbol()
            && !self.is_tuple()
            && !self.is_string()
            && !self.is_environment()
    }

    pub fn is_int(&self) -> bool {
        (unsafe { self.0.value } & HIGH16_TAG) == MIN_NUMBER
    }

    pub fn is_double(&self) -> bool {
        self.is_number() && !self.is_int()
    }

    pub fn as_double(&self) -> Option<f64> {
        if self.is_double() {
            Some(f64::from_bits(
                unsafe { self.0.value } - DOUBLE_ENCODE_OFFSET,
            ))
        } else {
            None
        }
    }

    pub fn as_int(&self) -> Option<i32> {
        if self.is_int() {
            let val = unsafe { self.0.value } as u32;
            Some(u32::cast_signed(val))
        } else {
            None
        }
    }

    pub fn is_pointer(&self) -> bool {
        let v = unsafe { self.0.value };
        // Generic pointer: low 2 bits = 00, value >= MIN_POINTER, no type tag bits
        (v & 0x03) == 0 && v >= MIN_POINTER && (v & POINTER_TAG_MASK) == GENERIC_POINTER_TAG
    }

    pub fn is_symbol(&self) -> bool {
        (unsafe { self.0.value } & HIGH16_TAG) == SYMBOL_TAG
    }

    pub fn is_tuple(&self) -> bool {
        let v = unsafe { self.0.value };
        (v & 0x03) == 0 && v >= MIN_POINTER && (v & POINTER_TAG_MASK) == TUPLE_POINTER_TAG
    }

    pub fn is_string(&self) -> bool {
        let v = unsafe { self.0.value };
        (v & 0x03) == 0 && v >= MIN_POINTER && (v & POINTER_TAG_MASK) == STRING_POINTER_TAG
    }

    pub fn is_environment(&self) -> bool {
        let v = unsafe { self.0.value };
        (v & 0x03) == 0 && v >= MIN_POINTER && (v & POINTER_TAG_MASK) == ENVIRONMENT_POINTER_TAG
    }

    pub fn is_closure(&self) -> bool {
        let v = unsafe { self.0.value };
        (v & 0x03) == 0 && v >= MIN_POINTER && (v & POINTER_TAG_MASK) == CLOSURE_POINTER_TAG
    }

    pub unsafe fn as_pointer<T>(&self) -> Option<*const T> {
        if self.is_pointer() {
            // For generic pointers, no masking needed since they have no tag bits
            Some(unsafe { self.0.ptr as *const T })
        } else {
            None
        }
    }

    pub unsafe fn as_pointer_mut<T>(&self) -> Option<*mut T> {
        if self.is_pointer() {
            // For generic pointers, no masking needed since they have no tag bits
            Some(unsafe { self.0.ptr as *mut T })
        } else {
            None
        }
    }

    pub unsafe fn mk_pointer<T>(ptr: *const T) -> Self {
        let addr = ptr as usize;
        // Ensure pointer is properly aligned and >= MIN_POINTER
        assert!(
            addr >= MIN_POINTER as usize,
            "Pointer address too low: 0x{addr:x}"
        );
        assert!(addr & 0x03 == 0, "Pointer not 4-byte aligned: 0x{addr:x}");
        Self(ValueUnion {
            ptr: ptr as *const (),
        })
    }

    /// Create a tagged pointer with the specified tag
    unsafe fn mk_tagged_pointer<T>(ptr: *const T, tag: u64) -> Self {
        let addr = ptr as u64;
        // Ensure pointer is properly aligned and >= MIN_POINTER
        assert!(
            addr >= MIN_POINTER,
            "Pointer address too low: 0x{addr:016x}"
        );
        assert!(
            addr & 0x07 == 0,
            "Pointer not 8-byte aligned: 0x{addr:016x}"
        );
        assert!(
            addr & POINTER_TAG_MASK == 0,
            "Pointer conflicts with tag bits: 0x{addr:016x}"
        );

        let tagged_addr = addr | tag;
        Self(ValueUnion { value: tagged_addr })
    }

    pub fn as_bool(&self) -> Option<bool> {
        if self.is_boolean() {
            Some(unsafe { self.0.value } == BOOLEAN_TRUE)
        } else {
            None
        }
    }

    pub fn as_symbol(&self) -> Option<u32> {
        if self.is_symbol() {
            Some(unsafe { self.0.value } as u32)
        } else {
            None
        }
    }

    pub fn as_symbol_obj(&self) -> Option<Symbol> {
        self.as_symbol().map(Symbol::from_id)
    }

    pub fn as_tuple(&self) -> Option<&[Var]> {
        if self.is_tuple() {
            let ptr_bits = unsafe { self.0.value } & !POINTER_TAG_MASK;
            let ptr = ptr_bits as *const LispTuple;
            unsafe { Some((*ptr).as_slice()) }
        } else {
            None
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        if self.is_string() {
            let ptr_bits = unsafe { self.0.value } & !POINTER_TAG_MASK;
            let ptr = ptr_bits as *const LispString;
            unsafe { Some((*ptr).as_str()) }
        } else {
            None
        }
    }

    pub fn as_environment(&self) -> Option<*mut Environment> {
        if self.is_environment() {
            let ptr_bits = unsafe { self.0.value } & !POINTER_TAG_MASK;
            Some(ptr_bits as *mut Environment)
        } else {
            None
        }
    }

    pub fn as_closure(&self) -> Option<*mut LispClosure> {
        if self.is_closure() {
            let ptr_bits = unsafe { self.0.value } & !POINTER_TAG_MASK;
            Some(ptr_bits as *mut LispClosure)
        } else {
            None
        }
    }

    pub fn as_u64(&self) -> u64 {
        unsafe { self.0.value }
    }

    pub fn from_u64(value: u64) -> Self {
        Self(ValueUnion { value })
    }

    /// Extract truth value for matrix operations.
    /// Returns: 0 for false, 1 for true, 2 for none/unbound
    pub fn as_truth_value(&self) -> u32 {
        match self.get_type() {
            VarType::Bool => {
                if self.as_bool().unwrap() {
                    1
                } else {
                    0
                }
            }
            VarType::None => 2,
            _ => 2, // Treat non-boolean values as unbound
        }
    }

    /// Check if the value is explicitly true
    pub fn is_true(&self) -> bool {
        matches!(self.as_bool(), Some(true))
    }

    /// Check if the value is explicitly false
    pub fn is_false(&self) -> bool {
        matches!(self.as_bool(), Some(false))
    }

    /// Check if the value is "truthy" (follows common programming language conventions)
    /// - true is truthy
    /// - false is falsy
    /// - 0 (int or float) is falsy
    /// - non-zero numbers are truthy
    /// - none is falsy
    /// - symbols are truthy
    /// - pointers are truthy
    /// - empty tuples are falsy
    /// - non-empty tuples are truthy
    /// - empty strings are falsy
    /// - non-empty strings are truthy
    pub fn is_truthy(&self) -> bool {
        match self.get_type() {
            VarType::Bool => self.as_bool().unwrap(),
            VarType::I32 => self.as_int().unwrap() != 0,
            VarType::F64 => {
                let val = self.as_double().unwrap();
                val != 0.0 && !val.is_nan()
            }
            VarType::None => false,
            VarType::Symbol => true,
            VarType::Pointer => true,
            VarType::Tuple => !self.as_tuple().unwrap().is_empty(),
            VarType::String => !self.as_string().unwrap().is_empty(),
            VarType::Environment => true, // Environments are always truthy
            VarType::Closure => true,     // Closures are always truthy
        }
    }

    /// Check if the value is "falsy" (the opposite of truthy)
    pub fn is_falsy(&self) -> bool {
        !self.is_truthy()
    }

    /// Get the type protocol for this value's type
    pub fn get_protocol(&self) -> &'static TypeProtocol {
        get_protocol(self.get_type())
    }

    /// Get the length of the value using the type protocol
    pub fn length(&self) -> Option<i32> {
        let protocol = self.get_protocol();
        protocol.length.map(|f| f(self as *const _ as *const ()))
    }

    /// Get an item by index using the type protocol
    pub fn get(&self, index: Var) -> Var {
        let protocol = self.get_protocol();
        if let Some(get_fn) = protocol.get {
            get_fn(self as *const _ as *const (), index)
        } else {
            Var::none()
        }
    }

    /// Convert to string using the type protocol
    pub fn protocol_to_string(&self) -> String {
        let protocol = self.get_protocol();
        (protocol.to_string)(self as *const _ as *const ())
    }

    /// Hash using the type protocol
    pub fn protocol_hash(&self) -> u64 {
        let protocol = self.get_protocol();
        (protocol.hash)(self as *const _ as *const ())
    }

    /// Compare with another value using the type protocol (same types only)
    pub fn protocol_compare(&self, other: &Var) -> Option<i32> {
        if self.get_type() == other.get_type() {
            let protocol = self.get_protocol();
            Some((protocol.compare)(
                self as *const _ as *const (),
                other as *const _ as *const (),
            ))
        } else {
            None
        }
    }

    /// Check if truthy using the type protocol
    pub fn protocol_is_truthy(&self) -> bool {
        let protocol = self.get_protocol();
        (protocol.is_truthy)(self as *const _ as *const ())
    }
}

impl From<f64> for Var {
    fn from(value: f64) -> Self {
        let mut bits = value.to_bits();
        bits += DOUBLE_ENCODE_OFFSET;
        Self(ValueUnion { value: bits })
    }
}

impl From<bool> for Var {
    fn from(value: bool) -> Self {
        let v = if value { BOOLEAN_TRUE } else { BOOLEAN_FALSE };
        Self(ValueUnion { value: v })
    }
}

impl From<i32> for Var {
    fn from(value: i32) -> Self {
        let bits: u32 = i32::cast_unsigned(value);
        let mut bits = bits as u64;
        bits |= MIN_NUMBER;
        Self(ValueUnion { value: bits })
    }
}

impl From<Symbol> for Var {
    fn from(value: Symbol) -> Self {
        Self::from_symbol(value)
    }
}

impl From<&[Var]> for Var {
    fn from(value: &[Var]) -> Self {
        Self::tuple(value)
    }
}

impl From<&str> for Var {
    fn from(value: &str) -> Self {
        Self::string(value)
    }
}

impl From<String> for Var {
    fn from(value: String) -> Self {
        Self::string(&value)
    }
}

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get_type() {
            VarType::None => write!(f, "Var::None"),
            VarType::Bool => write!(f, "Var::Bool({})", self.as_bool().unwrap()),
            VarType::I32 => write!(f, "Var::I32({})", self.as_int().unwrap()),
            VarType::F64 => write!(f, "Var::F64({})", self.as_double().unwrap()),
            VarType::Pointer => write!(f, "Var::Pointer({:p})", unsafe { self.0.ptr }),
            VarType::Symbol => write!(f, "Var::Symbol({})", self.as_symbol().unwrap()),
            VarType::Tuple => write!(f, "Var::Tuple({:?})", self.as_tuple().unwrap()),
            VarType::String => write!(f, "Var::String({:?})", self.as_string().unwrap()),
            VarType::Environment => {
                if let Some(env_ptr) = self.as_environment() {
                    unsafe { write!(f, "Var::Environment(size={})", (*env_ptr).size) }
                } else {
                    write!(f, "Var::Environment(invalid)")
                }
            }
            VarType::Closure => {
                if let Some(closure_ptr) = self.as_closure() {
                    unsafe { write!(f, "Var::Closure(arity={})", (*closure_ptr).arity) }
                } else {
                    write!(f, "Var::Closure(invalid)")
                }
            }
        }
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get_type() {
            VarType::None => write!(f, "none"),
            VarType::Bool => write!(f, "{}", self.as_bool().unwrap()),
            VarType::I32 => write!(f, "{}", self.as_int().unwrap()),
            VarType::F64 => write!(f, "{}", self.as_double().unwrap()),
            VarType::Pointer => write!(f, "ptr({:p})", unsafe { self.0.ptr }),
            VarType::Symbol => {
                if let Some(sym) = self.as_symbol_obj() {
                    write!(f, "{}", sym.as_string())
                } else {
                    write!(f, "sym({})", self.as_symbol().unwrap())
                }
            }
            VarType::Tuple => {
                let tuple = self.as_tuple().unwrap();
                write!(f, "[")?;
                for (i, item) in tuple.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            VarType::String => write!(f, "{}", self.as_string().unwrap()),
            VarType::Environment => {
                if let Some(env_ptr) = self.as_environment() {
                    unsafe { write!(f, "env(size={})", (*env_ptr).size) }
                } else {
                    write!(f, "env(invalid)")
                }
            }
            VarType::Closure => {
                if let Some(closure_ptr) = self.as_closure() {
                    unsafe { write!(f, "closure(arity={})", (*closure_ptr).arity) }
                } else {
                    write!(f, "closure(invalid)")
                }
            }
        }
    }
}

impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        match (self.get_type(), other.get_type()) {
            (VarType::None, VarType::None) => true,
            (VarType::Bool, VarType::Bool) => self.as_bool() == other.as_bool(),
            (VarType::I32, VarType::I32) => self.as_int() == other.as_int(),
            (VarType::F64, VarType::F64) => self.as_double() == other.as_double(),
            (VarType::Pointer, VarType::Pointer) => unsafe { self.0.ptr == other.0.ptr },
            (VarType::Symbol, VarType::Symbol) => self.as_symbol() == other.as_symbol(),
            (VarType::Tuple, VarType::Tuple) => self.as_tuple() == other.as_tuple(),
            (VarType::String, VarType::String) => self.as_string() == other.as_string(),
            _ => false,
        }
    }
}

impl Eq for Var {}

impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash based on the raw 64-bit value for consistent hashing
        unsafe { self.0.value.hash(state) }
    }
}

impl PartialOrd for Var {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Var {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.get_type(), other.get_type()) {
            // Same types - compare values
            (VarType::Bool, VarType::Bool) => self.as_bool().cmp(&other.as_bool()),
            (VarType::I32, VarType::I32) => self.as_int().cmp(&other.as_int()),
            (VarType::F64, VarType::F64) => self
                .as_double()
                .partial_cmp(&other.as_double())
                .unwrap_or(Ordering::Equal),
            (VarType::Symbol, VarType::Symbol) => self.as_symbol().cmp(&other.as_symbol()),

            // Cross-type numeric comparisons
            (VarType::I32, VarType::F64) => {
                let int_val = self.as_int().unwrap() as f64;
                int_val
                    .partial_cmp(&other.as_double().unwrap())
                    .unwrap_or(Ordering::Equal)
            }
            (VarType::F64, VarType::I32) => {
                let int_val = other.as_int().unwrap() as f64;
                self.as_double()
                    .unwrap()
                    .partial_cmp(&int_val)
                    .unwrap_or(Ordering::Equal)
            }

            // None is less than everything except None
            (VarType::None, VarType::None) => Ordering::Equal,
            (VarType::None, _) => Ordering::Less,
            (_, VarType::None) => Ordering::Greater,

            // Other cross-type comparisons use type precedence
            _ => self.get_type().cmp(&other.get_type()),
        }
    }
}

impl Add for Var {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self.get_type(), rhs.get_type()) {
            (VarType::I32, VarType::I32) => {
                let result = self.as_int().unwrap().wrapping_add(rhs.as_int().unwrap());
                Var::int(result)
            }
            (VarType::F64, VarType::F64) => {
                let result = self.as_double().unwrap() + rhs.as_double().unwrap();
                Var::float(result)
            }
            (VarType::I32, VarType::F64) => {
                let result = self.as_int().unwrap() as f64 + rhs.as_double().unwrap();
                Var::float(result)
            }
            (VarType::F64, VarType::I32) => {
                let result = self.as_double().unwrap() + rhs.as_int().unwrap() as f64;
                Var::float(result)
            }
            _ => Var::none(), // Invalid operation
        }
    }
}

impl Sub for Var {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self.get_type(), rhs.get_type()) {
            (VarType::I32, VarType::I32) => {
                let result = self.as_int().unwrap().wrapping_sub(rhs.as_int().unwrap());
                Var::int(result)
            }
            (VarType::F64, VarType::F64) => {
                let result = self.as_double().unwrap() - rhs.as_double().unwrap();
                Var::float(result)
            }
            (VarType::I32, VarType::F64) => {
                let result = self.as_int().unwrap() as f64 - rhs.as_double().unwrap();
                Var::float(result)
            }
            (VarType::F64, VarType::I32) => {
                let result = self.as_double().unwrap() - rhs.as_int().unwrap() as f64;
                Var::float(result)
            }
            _ => Var::none(), // Invalid operation
        }
    }
}

impl Mul for Var {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self.get_type(), rhs.get_type()) {
            (VarType::I32, VarType::I32) => {
                let result = self.as_int().unwrap().wrapping_mul(rhs.as_int().unwrap());
                Var::int(result)
            }
            (VarType::F64, VarType::F64) => {
                let result = self.as_double().unwrap() * rhs.as_double().unwrap();
                Var::float(result)
            }
            (VarType::I32, VarType::F64) => {
                let result = self.as_int().unwrap() as f64 * rhs.as_double().unwrap();
                Var::float(result)
            }
            (VarType::F64, VarType::I32) => {
                let result = self.as_double().unwrap() * rhs.as_int().unwrap() as f64;
                Var::float(result)
            }
            _ => Var::none(), // Invalid operation
        }
    }
}

impl Div for Var {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self.get_type(), rhs.get_type()) {
            (VarType::I32, VarType::I32) => {
                let rhs_val = rhs.as_int().unwrap();
                if rhs_val == 0 {
                    Var::none() // Division by zero
                } else {
                    let result = self.as_int().unwrap() / rhs_val;
                    Var::int(result)
                }
            }
            (VarType::F64, VarType::F64) => {
                let result = self.as_double().unwrap() / rhs.as_double().unwrap();
                Var::float(result)
            }
            (VarType::I32, VarType::F64) => {
                let result = self.as_int().unwrap() as f64 / rhs.as_double().unwrap();
                Var::float(result)
            }
            (VarType::F64, VarType::I32) => {
                let result = self.as_double().unwrap() / rhs.as_int().unwrap() as f64;
                Var::float(result)
            }
            _ => Var::none(), // Invalid operation
        }
    }
}

impl Rem for Var {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        match (self.get_type(), rhs.get_type()) {
            (VarType::I32, VarType::I32) => {
                let rhs_val = rhs.as_int().unwrap();
                if rhs_val == 0 {
                    Var::none() // Division by zero
                } else {
                    let result = self.as_int().unwrap() % rhs_val;
                    Var::int(result)
                }
            }
            (VarType::F64, VarType::F64) => {
                let result = self.as_double().unwrap() % rhs.as_double().unwrap();
                Var::float(result)
            }
            (VarType::I32, VarType::F64) => {
                let result = self.as_int().unwrap() as f64 % rhs.as_double().unwrap();
                Var::float(result)
            }
            (VarType::F64, VarType::I32) => {
                let result = self.as_double().unwrap() % rhs.as_int().unwrap() as f64;
                Var::float(result)
            }
            _ => Var::none(), // Invalid operation
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::symbol::Symbol;
    use crate::var::{Var, VarType};
    use std::cmp::Ordering;
    use std::mem::size_of;

    #[test]
    fn i_am_64_bits_wide_and_thats_all_i_need() {
        assert_eq!(size_of::<Var>(), 8);
    }

    #[test]
    fn encode_int() {
        let x: Var = 123.into();
        assert_eq!(x.get_type(), VarType::I32);
        assert!(!x.is_double());

        assert!(x.is_int());
        assert_eq!(x.as_int(), Some(123));
        assert_eq!(x.as_double(), None);

        let y: Var = (-123).into();
        assert!(!y.is_double());
        assert!(y.is_int());
        assert_eq!(y.as_int(), Some(-123));
        assert_eq!(y.as_double(), None);
    }

    #[test]
    fn encode_double() {
        let x: Var = 123.456.into();
        assert_eq!(x.get_type(), VarType::F64);
        assert!(x.is_double());
        assert!(!x.is_int());

        assert_eq!(x.as_int(), None);
        assert_eq!(x.as_double(), Some(123.456));

        let y: Var = (-123.456).into();
        assert!(y.is_double());
        assert!(!y.is_int());
        assert_eq!(y.as_int(), None);
        assert_eq!(y.as_double(), Some(-123.456));
    }

    #[test]
    fn encode_bool() {
        let x: Var = true.into();
        assert_eq!(x.get_type(), VarType::Bool);
        assert!(x.is_boolean());
        assert_eq!(x.as_bool(), Some(true));

        let y: Var = false.into();
        assert!(y.is_boolean());
        assert_eq!(y.as_bool(), Some(false));
    }

    #[test]
    fn encode_none() {
        let x: Var = Var::new();
        assert_eq!(x.get_type(), VarType::None);
        assert!(x.is_none());
        assert_eq!(x.as_bool(), None);
        assert_eq!(x.as_int(), None);
        assert_eq!(x.as_double(), None);
    }

    #[derive(Debug)]
    struct TestValue {
        _0: String,
    }

    #[test]
    fn encode_ptr() {
        let my_value = Box::new(TestValue {
            _0: "blarg".to_string(),
        });
        let x = unsafe { Var::mk_pointer(my_value.as_ref()) };
        assert_eq!(x.get_type(), VarType::Pointer);
        assert!(x.is_pointer());
        assert!(!x.is_none());
        assert!(!x.is_double());
        assert!(!x.is_number());
        assert!(!x.is_int());
        let p = unsafe { x.as_pointer().unwrap() };
        assert_eq!(p, my_value.as_ref());
    }

    #[test]
    fn test_constructors() {
        assert_eq!(Var::none().get_type(), VarType::None);
        assert_eq!(Var::int(42).get_type(), VarType::I32);
        assert_eq!(Var::float(3.14).get_type(), VarType::F64);
        assert_eq!(Var::bool(true).get_type(), VarType::Bool);

        assert_eq!(Var::symbol(12345).get_type(), VarType::Symbol);

        let sym = Symbol::mk("test");
        assert_eq!(Var::from_symbol(sym).get_type(), VarType::Symbol);
        assert_eq!(Var::from(sym).get_type(), VarType::Symbol);
    }

    #[test]
    fn test_equality() {
        let a = Var::int(42);
        let b = Var::int(42);
        let c = Var::int(43);
        assert_eq!(a, b);
        assert_ne!(a, c);

        let d = Var::float(3.14);
        let e = Var::float(3.14);
        assert_eq!(d, e);

        let f = Var::bool(true);
        let g = Var::bool(true);
        let h = Var::bool(false);
        assert_eq!(f, g);
        assert_ne!(f, h);
    }

    #[test]
    fn test_debug_display() {
        let none = Var::none();
        assert_eq!(format!("{none:?}"), "Var::None");
        assert_eq!(format!("{none}"), "none");

        let int = Var::int(42);
        assert_eq!(format!("{int:?}"), "Var::I32(42)");
        assert_eq!(format!("{int}"), "42");

        let float = Var::float(3.14);
        assert_eq!(format!("{float:?}"), "Var::F64(3.14)");
        assert_eq!(format!("{float}"), "3.14");
    }

    #[test]
    fn test_raw_conversion() {
        let original = Var::int(42);
        let raw = original.as_u64();
        let reconstructed = Var::from_u64(raw);
        assert_eq!(original, reconstructed);
    }

    #[test]
    fn test_edge_cases() {
        // Test min/max values
        let min_int = Var::int(i32::MIN);
        let max_int = Var::int(i32::MAX);
        assert_eq!(min_int.as_int(), Some(i32::MIN));
        assert_eq!(max_int.as_int(), Some(i32::MAX));

        // Test special float values
        let zero = Var::float(0.0);
        let neg_zero = Var::float(-0.0);
        let inf = Var::float(f64::INFINITY);
        let neg_inf = Var::float(f64::NEG_INFINITY);

        assert_eq!(zero.as_double(), Some(0.0));
        assert_eq!(neg_zero.as_double(), Some(-0.0));
        assert_eq!(inf.as_double(), Some(f64::INFINITY));
        assert_eq!(neg_inf.as_double(), Some(f64::NEG_INFINITY));
    }

    #[test]
    fn test_symbols() {
        // Create symbols properly using Symbol::mk
        let test_sym1 = Symbol::mk("test1");
        let test_sym2 = Symbol::mk("test2");

        let sym1 = Var::from(test_sym1);
        let sym2 = Var::from(test_sym1); // Same symbol
        let sym3 = Var::from(test_sym2);

        assert_eq!(sym1.get_type(), VarType::Symbol);
        assert!(sym1.is_symbol());
        assert!(!sym1.is_int());
        assert!(!sym1.is_double());
        assert!(!sym1.is_boolean());
        assert!(!sym1.is_none());
        // Note: Symbol detection takes precedence over pointer detection in get_type()

        assert_eq!(sym1.as_symbol(), Some(test_sym1.id()));
        assert_eq!(sym1.as_int(), None);

        // Test equality
        assert_eq!(sym1, sym2);
        assert_ne!(sym1, sym3);

        // Test Symbol integration
        let test_sym = Symbol::mk("hello");
        let var_sym = Var::from(test_sym);
        assert_eq!(var_sym.as_symbol_obj(), Some(test_sym));
        assert_eq!(var_sym.as_symbol_obj().unwrap().as_string(), "hello");
    }

    #[test]
    fn test_symbol_edge_cases() {
        // Test min/max symbol IDs
        let min_sym = Var::symbol(0);
        let max_sym = Var::symbol(u32::MAX);

        assert_eq!(min_sym.as_symbol(), Some(0));
        assert_eq!(max_sym.as_symbol(), Some(u32::MAX));

        assert_eq!(min_sym.get_type(), VarType::Symbol);
        assert_eq!(max_sym.get_type(), VarType::Symbol);
    }

    #[test]
    fn test_boolean_truth_table_optimization() {
        // Test that false = 0x00 for optimal truth table operations
        let false_val = Var::bool(false);
        let true_val = Var::bool(true);

        assert_eq!(false_val.as_u64(), 0x0000000000000000);
        assert_eq!(true_val.as_u64(), 0x0000000000000001);

        // Verify these don't conflict with other types
        assert_eq!(false_val.get_type(), VarType::Bool);
        assert_eq!(true_val.get_type(), VarType::Bool);

        // None should be 0x02
        let none_val = Var::none();
        assert_eq!(none_val.as_u64(), 0x0000000000000002);
        assert_eq!(none_val.get_type(), VarType::None);

        // Verify no confusion with int(0) or float(0.0)
        let int_zero = Var::int(0);
        let float_zero = Var::float(0.0);

        assert_ne!(false_val.as_u64(), int_zero.as_u64());
        assert_ne!(false_val.as_u64(), float_zero.as_u64());
        assert_eq!(int_zero.get_type(), VarType::I32);
        assert_eq!(float_zero.get_type(), VarType::F64);
    }

    #[test]
    fn test_boolean_helpers() {
        let true_val = Var::bool(true);
        let false_val = Var::bool(false);
        let int_val = Var::int(42);
        let none_val = Var::none();

        // Test is_true/is_false
        assert!(true_val.is_true());
        assert!(!true_val.is_false());
        assert!(false_val.is_false());
        assert!(!false_val.is_true());
        assert!(!int_val.is_true());
        assert!(!int_val.is_false());
        assert!(!none_val.is_true());
        assert!(!none_val.is_false());

        // Test truthy/falsy
        assert!(true_val.is_truthy());
        assert!(!true_val.is_falsy());
        assert!(!false_val.is_truthy());
        assert!(false_val.is_falsy());

        // Numbers
        assert!(Var::int(1).is_truthy());
        assert!(Var::int(-1).is_truthy());
        assert!(!Var::int(0).is_truthy());
        assert!(Var::int(0).is_falsy());

        assert!(Var::float(1.0).is_truthy());
        assert!(Var::float(-1.0).is_truthy());
        assert!(!Var::float(0.0).is_truthy());
        assert!(Var::float(0.0).is_falsy());
        assert!(!Var::float(f64::NAN).is_truthy());
        assert!(Var::float(f64::NAN).is_falsy());

        // None is falsy
        assert!(!none_val.is_truthy());
        assert!(none_val.is_falsy());

        // Symbols are truthy
        let sym = Var::symbol(123);
        assert!(sym.is_truthy());
        assert!(!sym.is_falsy());
    }

    #[test]
    fn test_arithmetic_operations() {
        // Integer arithmetic
        let a = Var::int(10);
        let b = Var::int(5);

        assert_eq!((a + b).as_int(), Some(15));
        assert_eq!((a - b).as_int(), Some(5));
        assert_eq!((a * b).as_int(), Some(50));
        assert_eq!((a / b).as_int(), Some(2));
        assert_eq!((a % b).as_int(), Some(0));

        // Float arithmetic
        let c = Var::float(10.0);
        let d = Var::float(3.0);

        assert_eq!((c + d).as_double(), Some(13.0));
        assert_eq!((c - d).as_double(), Some(7.0));
        assert_eq!((c * d).as_double(), Some(30.0));
        assert!((c / d).as_double().unwrap() - 3.333333333333 < 0.0001);
        assert_eq!((c % d).as_double(), Some(1.0));

        // Mixed arithmetic (int + float = float)
        let mixed_add = a + c; // 10 + 10.0
        assert_eq!(mixed_add.get_type(), VarType::F64);
        assert_eq!(mixed_add.as_double(), Some(20.0));

        let mixed_sub = c - b; // 10.0 - 5
        assert_eq!(mixed_sub.get_type(), VarType::F64);
        assert_eq!(mixed_sub.as_double(), Some(5.0));
    }

    #[test]
    fn test_arithmetic_edge_cases() {
        // Division by zero
        let zero = Var::int(0);
        let ten = Var::int(10);

        let div_by_zero = ten / zero;
        assert_eq!(div_by_zero.get_type(), VarType::None);

        let mod_by_zero = ten % zero;
        assert_eq!(mod_by_zero.get_type(), VarType::None);

        // Float division by zero results in infinity
        let zero_f = Var::float(0.0);
        let ten_f = Var::float(10.0);

        let div_by_zero_f = ten_f / zero_f;
        assert_eq!(div_by_zero_f.get_type(), VarType::F64);
        assert!(div_by_zero_f.as_double().unwrap().is_infinite());

        // Overflow wrapping
        let max_int = Var::int(i32::MAX);
        let one = Var::int(1);
        let overflow = max_int + one;
        assert_eq!(overflow.as_int(), Some(i32::MIN)); // Wrapping behavior

        // Invalid operations return None
        let bool_val = Var::bool(true);
        let invalid = bool_val + ten;
        assert_eq!(invalid.get_type(), VarType::None);
    }

    #[test]
    fn test_comparison_operations() {
        use std::cmp::Ordering;

        // Integer comparisons
        let a = Var::int(5);
        let b = Var::int(10);
        let c = Var::int(5);

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, c);
        assert!(a <= c);
        assert!(a >= c);
        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
        assert_eq!(a.cmp(&c), Ordering::Equal);

        // Float comparisons
        let d = Var::float(5.5);
        let e = Var::float(10.5);

        assert!(d < e);
        assert!(e > d);

        // Mixed numeric comparisons
        let f = Var::int(5);
        let g = Var::float(5.0);
        let h = Var::float(5.1);

        assert_eq!(f.partial_cmp(&g), Some(Ordering::Equal));
        assert_eq!(f.partial_cmp(&h), Some(Ordering::Less));

        // Boolean comparisons
        let t = Var::bool(true);
        let f_bool = Var::bool(false);

        assert!(t > f_bool);
        assert_eq!(t.cmp(&Var::bool(true)), Ordering::Equal);

        // None comparisons
        let none = Var::none();
        assert!(none < a);
        assert!(none < t);
        assert_eq!(none.cmp(&Var::none()), Ordering::Equal);

        // Symbol comparisons
        let sym1 = Var::symbol(100);
        let sym2 = Var::symbol(200);

        assert!(sym1 < sym2);
        assert_eq!(sym1.cmp(&Var::symbol(100)), Ordering::Equal);

        // Cross-type comparisons that have no natural ordering
        // Should fall back to type precedence
        let type_order = a.cmp(&t); // Uses type precedence
        assert!(type_order != Ordering::Equal);
        // partial_cmp now always returns Some since we implement total ordering
        assert!(a.partial_cmp(&t).is_some());
    }

    #[test]
    fn test_type_precedence_ordering() {
        // Test that type precedence works as expected
        let none = Var::none();
        let int_val = Var::int(42);
        let float_val = Var::float(3.14);
        let bool_val = Var::bool(true);
        let ptr_val = unsafe { Var::mk_pointer(&42i32 as *const i32) };
        let sym_val = Var::symbol(123);

        // None should be less than everything
        assert!(none < int_val);
        assert!(none < float_val);
        assert!(none < bool_val);
        assert!(none < ptr_val);
        assert!(none < sym_val);

        // The ordering between non-comparable types should be consistent
        // (based on enum variant order: None, I32, F64, Bool, Pointer, Symbol)
        assert!(int_val.cmp(&bool_val) != Ordering::Equal);
        assert!(bool_val.cmp(&ptr_val) != Ordering::Equal);
        assert!(ptr_val.cmp(&sym_val) != Ordering::Equal);
    }

    #[test]
    fn test_tuple_basic_creation() {
        // Test simple tuple creation
        let tuple_var = Var::empty_tuple();
        assert_eq!(tuple_var.get_type(), VarType::Tuple);
        assert!(tuple_var.is_tuple());

        // Test accessing the tuple (should not crash)
        let retrieved_tuple = tuple_var.as_tuple();
        assert!(retrieved_tuple.is_some());

        // Test length
        let tuple_ref = retrieved_tuple.unwrap();
        assert_eq!(tuple_ref.len(), 0);
    }

    #[test]
    fn test_tuple_and_string_basic() {
        // Test tuple creation and access
        let elements = [Var::int(1), Var::int(2), Var::int(3)];
        let tuple_var = Var::tuple(&elements);
        assert_eq!(tuple_var.get_type(), VarType::Tuple);
        assert!(tuple_var.is_tuple());
        assert!(!tuple_var.is_string());

        let retrieved_tuple = tuple_var.as_tuple().unwrap();
        assert_eq!(retrieved_tuple.len(), 3);
        assert_eq!(retrieved_tuple[0].as_int(), Some(1));
        assert_eq!(retrieved_tuple[1].as_int(), Some(2));
        assert_eq!(retrieved_tuple[2].as_int(), Some(3));

        // Test string creation and access
        let string_var = Var::string("hello world");
        assert_eq!(string_var.get_type(), VarType::String);
        assert!(string_var.is_string());
        assert!(!string_var.is_tuple());

        let retrieved_string = string_var.as_string().unwrap();
        assert_eq!(retrieved_string, "hello world");

        // Test From implementations
        let elements2 = [Var::bool(true)];
        let tuple_var2: Var = (&elements2[..]).into();
        assert_eq!(tuple_var2.get_type(), VarType::Tuple);

        let string_var2: Var = "test".into();
        assert_eq!(string_var2.get_type(), VarType::String);

        // Test Display
        assert_eq!(format!("{tuple_var}"), "[1, 2, 3]");
        assert_eq!(format!("{string_var}"), "hello world");

        // Test truthy/falsy
        assert!(tuple_var.is_truthy()); // Non-empty tuple is truthy
        assert!(string_var.is_truthy()); // Non-empty string is truthy

        let empty_tuple = Var::empty_tuple();
        let empty_string = Var::string("");
        assert!(empty_tuple.is_falsy()); // Empty tuple is falsy
        assert!(empty_string.is_falsy()); // Empty string is falsy
    }

    #[test]
    fn test_protocol_system() {
        // Test tuple protocol
        let elements = [Var::int(1), Var::int(2), Var::int(3)];
        let tuple = Var::tuple(&elements);

        assert_eq!(tuple.length(), Some(3));
        assert_eq!(tuple.get(Var::int(0)).as_int(), Some(1));
        assert_eq!(tuple.get(Var::int(1)).as_int(), Some(2));
        assert_eq!(tuple.get(Var::int(2)).as_int(), Some(3));
        assert_eq!(tuple.get(Var::int(10)), Var::none()); // Out of bounds
        assert_eq!(tuple.protocol_to_string(), "[1, 2, 3]");
        assert!(tuple.protocol_is_truthy());

        // Test empty tuple
        let empty_tuple = Var::empty_tuple();
        assert_eq!(empty_tuple.length(), Some(0));
        assert!(!empty_tuple.protocol_is_truthy());

        // Test string protocol
        let s = Var::string("hello");
        assert_eq!(s.length(), Some(5));
        assert_eq!(s.get(Var::int(0)).as_string().unwrap(), "h");
        assert_eq!(s.get(Var::int(4)).as_string().unwrap(), "o");
        assert_eq!(s.get(Var::int(10)), Var::none()); // Out of bounds
        assert_eq!(s.protocol_to_string(), "hello");
        assert!(s.protocol_is_truthy());

        // Test empty string
        let empty_string = Var::string("");
        assert_eq!(empty_string.length(), Some(0));
        assert!(!empty_string.protocol_is_truthy());

        // Test primitives
        let num = Var::int(42);
        assert_eq!(num.length(), None); // Numbers don't have length
        assert_eq!(num.protocol_to_string(), "42");
        assert!(num.protocol_is_truthy());

        let zero = Var::int(0);
        assert!(!zero.protocol_is_truthy());

        // Test protocol comparison
        let a = Var::int(5);
        let b = Var::int(10);
        let c = Var::int(5);

        assert_eq!(a.protocol_compare(&b), Some(-1)); // 5 < 10
        assert_eq!(b.protocol_compare(&a), Some(1)); // 10 > 5
        assert_eq!(a.protocol_compare(&c), Some(0)); // 5 == 5

        // Different types can't be compared
        assert_eq!(a.protocol_compare(&s), None);
    }

    #[test]
    fn test_comprehensive_arithmetic_matrix() {
        // Test all combinations of arithmetic operations
        let values = [
            Var::int(6),
            Var::float(4.0),
            Var::bool(true),
            Var::none(),
            Var::symbol(42),
        ];

        for &lhs in &values {
            for &rhs in &values {
                let add_result = lhs + rhs;
                let sub_result = lhs - rhs;
                let mul_result = lhs * rhs;
                let div_result = lhs / rhs;
                let rem_result = lhs % rhs;

                // Valid operations should produce numbers or None
                match (lhs.get_type(), rhs.get_type()) {
                    (VarType::I32, VarType::I32)
                    | (VarType::F64, VarType::F64)
                    | (VarType::I32, VarType::F64)
                    | (VarType::F64, VarType::I32) => {
                        // These should produce valid results (unless division by zero)
                        assert!(add_result.is_number() || add_result.is_none());
                        assert!(sub_result.is_number() || sub_result.is_none());
                        assert!(mul_result.is_number() || mul_result.is_none());
                        assert!(div_result.is_number() || div_result.is_none());
                        assert!(rem_result.is_number() || rem_result.is_none());
                    }
                    _ => {
                        // All other combinations should return None
                        assert_eq!(add_result.get_type(), VarType::None);
                        assert_eq!(sub_result.get_type(), VarType::None);
                        assert_eq!(mul_result.get_type(), VarType::None);
                        assert_eq!(div_result.get_type(), VarType::None);
                        assert_eq!(rem_result.get_type(), VarType::None);
                    }
                }
            }
        }
    }
}
