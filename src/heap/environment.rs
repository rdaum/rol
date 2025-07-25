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

//! Environment system for lexical scoping in JIT-compiled functions.
//! Uses offset-based addressing for efficient variable access.

use crate::heap::flexible_utils::{
    alloc_with_trailing, free_with_trailing, trailing_ptr, trailing_slice,
};
use crate::var::Var;

/// Lexical address for variables - avoids symbol lookup at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexicalAddress {
    /// Number of environment frames to walk up (0 = current frame)
    pub depth: u32,
    /// Index within that frame's slots array
    pub offset: u32,
}

/// Native environment type optimized for JIT access.
/// Layout: [parent: u64][size: u32][slots: Var...]
/// Environment slots follow immediately after size field.
#[repr(C)]
pub struct Environment {
    /// Parent environment (as Var bits) or 0 if no parent
    pub parent: u64,
    /// Number of slots in this environment
    pub size: u32,
    // Slots follow immediately after size (flexible array member)
}

impl Environment {
    /// Create a new environment with the given number of slots.
    /// All slots are initialized to Var::none().
    pub fn new(slot_count: u32, parent: Option<Var>) -> *mut Environment {
        unsafe {
            alloc_with_trailing(
                slot_count as usize,
                |ptr: *mut Environment, slots_ptr: *mut u64| {
                    // Initialize header
                    (*ptr).parent = parent.map_or(0, |p| p.as_u64());
                    (*ptr).size = slot_count;

                    // Initialize all slots to none
                    for i in 0..slot_count as usize {
                        *slots_ptr.add(i) = Var::none().as_u64();
                    }
                },
            )
        }
    }

    /// Create an environment from a slice of initial values.
    pub fn from_values(values: &[Var], parent: Option<Var>) -> *mut Environment {
        unsafe {
            alloc_with_trailing(
                values.len(),
                |ptr: *mut Environment, slots_ptr: *mut u64| {
                    // Initialize header
                    (*ptr).parent = parent.map_or(0, |p| p.as_u64());
                    (*ptr).size = values.len() as u32;

                    // Initialize slots with provided values
                    for (i, &value) in values.iter().enumerate() {
                        *slots_ptr.add(i) = value.as_u64();
                    }
                },
            )
        }
    }

    /// Get pointer to the slots array.
    unsafe fn slots_ptr(ptr: *mut Environment) -> *mut u64 {
        unsafe { trailing_ptr(ptr) }
    }

    /// Get a value from this environment's slots.
    pub unsafe fn get(&self, offset: u32) -> Var {
        debug_assert!(offset < self.size, "Environment slot index out of bounds");
        let slots_ptr = unsafe {
            (self as *const Environment as *const u8).add(std::mem::size_of::<Environment>())
                as *const u64
        };
        unsafe { Var::from_u64(*slots_ptr.add(offset as usize)) }
    }

    /// Set a value in this environment's slots.
    pub unsafe fn set(&mut self, offset: u32, value: Var) {
        debug_assert!(offset < self.size, "Environment slot index out of bounds");
        let slots_ptr = unsafe {
            (self as *mut Environment as *mut u8).add(std::mem::size_of::<Environment>())
                as *mut u64
        };
        unsafe {
            // Get old value and slot pointer for write barrier
            let slot_var_ptr = slots_ptr.add(offset as usize) as *mut crate::var::Var;
            let old_value = crate::var::Var::from_u64(*slots_ptr.add(offset as usize));

            // Use RAII write barrier guard
            crate::with_write_barrier!(
                self as *mut Environment as *mut u8,
                slot_var_ptr,
                old_value,
                value => {
                    *slots_ptr.add(offset as usize) = value.as_u64();
                }
            );
        }
    }

    /// Get the parent environment, if any.
    pub fn parent(&self) -> Option<Var> {
        if self.parent != 0 {
            Some(Var::from_u64(self.parent))
        } else {
            None
        }
    }

    /// Resolve a lexical address to a value by walking up the environment chain.
    pub unsafe fn resolve(&self, addr: LexicalAddress) -> Option<Var> {
        let mut current_env = self;

        // Walk up 'depth' parent links
        for _ in 0..addr.depth {
            if let Some(parent_var) = current_env.parent() {
                if let Some(parent_ptr) = parent_var.as_environment() {
                    current_env = unsafe { &*parent_ptr };
                } else {
                    return None; // Invalid parent
                }
            } else {
                return None; // Not enough parents
            }
        }

        // Check bounds and get value
        if addr.offset < current_env.size {
            Some(unsafe { current_env.get(addr.offset) })
        } else {
            None // Offset out of bounds
        }
    }

    /// Set a value using lexical addressing.
    pub unsafe fn assign(&mut self, addr: LexicalAddress, value: Var) -> bool {
        let mut current_env = self;

        // Walk up 'depth' parent links
        for _ in 0..addr.depth {
            if let Some(parent_var) = current_env.parent() {
                if let Some(parent_ptr) = parent_var.as_environment() {
                    current_env = unsafe { &mut *parent_ptr };
                } else {
                    return false; // Invalid parent
                }
            } else {
                return false; // Not enough parents
            }
        }

        // Check bounds and set value
        if addr.offset < current_env.size {
            unsafe {
                current_env.set(addr.offset, value);
            }
            true
        } else {
            false // Offset out of bounds
        }
    }

    /// Get all values as a slice (for debugging/testing).
    pub unsafe fn as_slice(&self) -> &[Var] {
        unsafe { trailing_slice(self as *const Environment, self.size as usize) }
    }

    /// Free the memory for this Environment.
    pub unsafe fn free(ptr: *mut Environment) {
        if ptr.is_null() {
            return;
        }

        let element_count = unsafe { (*ptr).size as usize };
        unsafe { free_with_trailing::<Environment, u64>(ptr, element_count) };
    }
}

/// JIT helper functions - C-compatible interface for cranelift-generated code.
/// These functions provide efficient runtime environment access without Rust calling conventions.
/// Create a new environment with given slot count and optional parent.
/// Returns the environment as a u64 (Var bits) for JIT code.
#[unsafe(no_mangle)]
pub extern "C" fn env_create(slot_count: u32, parent_bits: u64) -> u64 {
    let parent = if parent_bits == 0 {
        None
    } else {
        Some(Var::from_u64(parent_bits))
    };

    let env_ptr = Environment::new(slot_count, parent);
    Var::environment(env_ptr).as_u64()
}

/// Get a value from environment using lexical addressing.
/// Returns the value as u64 (Var bits), or 0 (none) if invalid address.
#[unsafe(no_mangle)]
pub extern "C" fn env_get(env_bits: u64, depth: u32, offset: u32) -> u64 {
    let env_var = Var::from_u64(env_bits);
    if let Some(env_ptr) = env_var.as_environment() {
        let addr = LexicalAddress { depth, offset };
        unsafe {
            if let Some(value) = (*env_ptr).resolve(addr) {
                return value.as_u64();
            }
        }
    }
    Var::none().as_u64()
}

/// Set a value in environment using lexical addressing.
/// Returns the environment (unchanged) as u64.
#[unsafe(no_mangle)]
pub extern "C" fn env_set(env_bits: u64, depth: u32, offset: u32, value_bits: u64) -> u64 {
    let env_var = Var::from_u64(env_bits);
    if let Some(env_ptr) = env_var.as_environment() {
        let addr = LexicalAddress { depth, offset };
        let value = Var::from_u64(value_bits);
        unsafe {
            (*env_ptr).assign(addr, value);
        }
    }
    env_bits // Return the environment unchanged
}

/// Get direct slot value from current environment frame (depth=0 optimized path).
/// Returns the value as u64 (Var bits), or 0 (none) if invalid offset.
#[unsafe(no_mangle)]
pub extern "C" fn env_get_local(env_bits: u64, offset: u32) -> u64 {
    let env_var = Var::from_u64(env_bits);
    if let Some(env_ptr) = env_var.as_environment() {
        unsafe {
            if offset < (*env_ptr).size {
                return (*env_ptr).get(offset).as_u64();
            }
        }
    }
    Var::none().as_u64()
}

/// Set direct slot value in current environment frame (depth=0 optimized path).
/// Returns 1 if successful, 0 if failed.
#[unsafe(no_mangle)]
pub extern "C" fn env_set_local(env_bits: u64, offset: u32, value_bits: u64) -> u32 {
    let env_var = Var::from_u64(env_bits);
    if let Some(env_ptr) = env_var.as_environment() {
        unsafe {
            if offset < (*env_ptr).size {
                let value = Var::from_u64(value_bits);
                (*env_ptr).set(offset, value);
                return 1;
            }
        }
    }
    0
}

/// Get the parent environment, or 0 if none.
/// Returns parent environment as u64 (Var bits).
#[unsafe(no_mangle)]
pub extern "C" fn env_get_parent(env_bits: u64) -> u64 {
    let env_var = Var::from_u64(env_bits);
    if let Some(env_ptr) = env_var.as_environment() {
        unsafe {
            if let Some(parent) = (*env_ptr).parent() {
                return parent.as_u64();
            }
        }
    }
    0
}

/// Get the size (slot count) of an environment.
#[unsafe(no_mangle)]
pub extern "C" fn env_get_size(env_bits: u64) -> u32 {
    let env_var = Var::from_u64(env_bits);
    if let Some(env_ptr) = env_var.as_environment() {
        unsafe {
            return (*env_ptr).size;
        }
    }
    0
}

/// Check if a Var is an environment (useful for JIT type checks).
/// Returns 1 if environment, 0 if not.
#[unsafe(no_mangle)]
pub extern "C" fn var_is_environment(var_bits: u64) -> u32 {
    let var = Var::from_u64(var_bits);
    if var.is_environment() { 1 } else { 0 }
}

// For debugging
impl std::fmt::Debug for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let parent_info = if self.parent != 0 {
                format!("Some({:x})", self.parent)
            } else {
                "None".to_string()
            };

            write!(
                f,
                "Environment {{ parent: {}, size: {}, slots: {:?} }}",
                parent_info,
                self.size,
                self.as_slice()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_creation() {
        crate::gc::ensure_mmtk_initialized_for_tests();
        unsafe {
            let env = Environment::new(3, None);
            assert_eq!((*env).size, 3);
            assert_eq!((*env).parent, 0);

            // All slots should be initialized to none
            assert_eq!((*env).get(0), Var::none());
            assert_eq!((*env).get(1), Var::none());
            assert_eq!((*env).get(2), Var::none());

            // mmtk handles cleanup automatically
        }
    }

    #[test]
    fn test_environment_from_values() {
        crate::gc::ensure_mmtk_initialized_for_tests();
        let values = [Var::int(42), Var::bool(true), Var::string("hello")];

        unsafe {
            let env = Environment::from_values(&values, None);
            assert_eq!((*env).size, 3);

            assert_eq!((*env).get(0), Var::int(42));
            assert_eq!((*env).get(1), Var::bool(true));
            assert_eq!((*env).get(2).as_string(), Some("hello"));

            // mmtk handles cleanup automatically
        }
    }

    #[test]
    fn test_environment_get_set() {
        crate::gc::ensure_mmtk_initialized_for_tests();
        unsafe {
            let env = Environment::new(2, None);

            // Set values
            (*env).set(0, Var::int(100));
            (*env).set(1, Var::bool(false));

            // Get values
            assert_eq!((*env).get(0), Var::int(100));
            assert_eq!((*env).get(1), Var::bool(false));

            // mmtk handles cleanup automatically
        }
    }

    #[test]
    fn test_environment_parent_chain() {
        crate::gc::ensure_mmtk_initialized_for_tests();
        unsafe {
            // Create parent environment
            let parent_values = [Var::int(1), Var::int(2)];
            let parent = Environment::from_values(&parent_values, None);
            let parent_var = Var::environment(parent);

            // Create child environment
            let child_values = [Var::int(10), Var::int(20)];
            let child = Environment::from_values(&child_values, Some(parent_var));

            // Test parent access
            assert!((*child).parent().is_some());
            let retrieved_parent = (*child).parent().unwrap().as_environment().unwrap();
            assert_eq!((*retrieved_parent).get(0), Var::int(1));
            assert_eq!((*retrieved_parent).get(1), Var::int(2));

            // mmtk handles cleanup automatically
        }
    }

    #[test]
    fn test_lexical_addressing() {
        crate::gc::ensure_mmtk_initialized_for_tests();
        unsafe {
            // Create nested environments:
            // parent: [100, 200]
            // child:  [10, 20]

            let parent = Environment::from_values(&[Var::int(100), Var::int(200)], None);
            let parent_var = Var::environment(parent);

            let child = Environment::from_values(&[Var::int(10), Var::int(20)], Some(parent_var));

            // Test current frame access (depth 0)
            let addr = LexicalAddress {
                depth: 0,
                offset: 1,
            };
            assert_eq!((*child).resolve(addr), Some(Var::int(20)));

            // Test parent frame access (depth 1)
            let addr = LexicalAddress {
                depth: 1,
                offset: 0,
            };
            assert_eq!((*child).resolve(addr), Some(Var::int(100)));

            let addr = LexicalAddress {
                depth: 1,
                offset: 1,
            };
            assert_eq!((*child).resolve(addr), Some(Var::int(200)));

            // Test out of bounds
            let addr = LexicalAddress {
                depth: 0,
                offset: 5,
            };
            assert_eq!((*child).resolve(addr), None);

            let addr = LexicalAddress {
                depth: 2,
                offset: 0,
            };
            assert_eq!((*child).resolve(addr), None);

            // mmtk handles cleanup automatically
        }
    }

    #[test]
    fn test_lexical_assignment() {
        crate::gc::ensure_mmtk_initialized_for_tests();
        unsafe {
            let parent = Environment::from_values(&[Var::int(100), Var::int(200)], None);
            let parent_var = Var::environment(parent);

            let child = Environment::from_values(&[Var::int(10), Var::int(20)], Some(parent_var));

            // Assign to current frame
            let addr = LexicalAddress {
                depth: 0,
                offset: 0,
            };
            assert!((*child).assign(addr, Var::int(999)));
            assert_eq!((*child).get(0), Var::int(999));

            // Assign to parent frame
            let addr = LexicalAddress {
                depth: 1,
                offset: 1,
            };
            assert!((*child).assign(addr, Var::int(888)));
            assert_eq!((*parent).get(1), Var::int(888));

            // mmtk handles cleanup automatically
        }
    }

    #[test]
    fn test_var_environment_integration() {
        crate::gc::ensure_mmtk_initialized_for_tests();
        unsafe {
            // Test that Environment integrates properly with Var system
            let values = [Var::int(42), Var::string("test"), Var::bool(true)];
            let env = Environment::from_values(&values, None);
            let env_var = Var::environment(env);

            // Test type checking
            assert!(env_var.is_environment());
            assert!(!env_var.is_tuple());
            assert!(!env_var.is_string());
            assert!(!env_var.is_number());
            assert_eq!(env_var.get_type(), crate::var::VarType::Environment);

            // Test truthiness
            assert!(env_var.is_truthy());
            assert!(!env_var.is_falsy());

            // Test pointer extraction
            assert_eq!(env_var.as_environment(), Some(env));

            // Test environment operations through Var
            if let Some(retrieved_env) = env_var.as_environment() {
                assert_eq!((*retrieved_env).size, 3);
                assert_eq!((*retrieved_env).get(0), Var::int(42));
                assert_eq!((*retrieved_env).get(1).as_string(), Some("test"));
                assert_eq!((*retrieved_env).get(2), Var::bool(true));
            }

            // Test Display/Debug formatting
            let debug_str = format!("{env_var:?}");
            assert!(debug_str.contains("Environment(size=3)"));
            let display_str = format!("{env_var}");
            assert!(display_str.contains("env(size=3)"));

            // mmtk handles cleanup automatically
        }
    }

    #[test]
    fn test_jit_helper_env_create() {
        crate::gc::ensure_mmtk_initialized_for_tests();
        unsafe {
            // Test creating environment without parent
            let env_bits = env_create(3, 0);
            let env_var = Var::from_u64(env_bits);
            assert!(env_var.is_environment());

            if let Some(env_ptr) = env_var.as_environment() {
                assert_eq!((*env_ptr).size, 3);
                assert_eq!((*env_ptr).parent, 0);
                // mmtk handles cleanup automatically
            }

            // Test creating environment with parent
            let parent_bits = env_create(2, 0);
            let child_bits = env_create(1, parent_bits);

            let child_var = Var::from_u64(child_bits);
            if let Some(child_ptr) = child_var.as_environment() {
                assert_eq!((*child_ptr).size, 1);
                assert_ne!((*child_ptr).parent, 0);
            }
        }
    }

    #[test]
    fn test_jit_helper_local_access() {
        crate::gc::ensure_mmtk_initialized_for_tests();
        unsafe {
            let env_bits = env_create(2, 0);

            // Test local set/get
            assert_eq!(env_set_local(env_bits, 0, Var::int(42).as_u64()), 1);
            assert_eq!(env_set_local(env_bits, 1, Var::bool(true).as_u64()), 1);

            let val0 = Var::from_u64(env_get_local(env_bits, 0));
            let val1 = Var::from_u64(env_get_local(env_bits, 1));

            assert_eq!(val0.as_int(), Some(42));
            assert_eq!(val1.as_bool(), Some(true));

            // Test bounds checking
            assert_eq!(env_get_local(env_bits, 5), Var::none().as_u64());
            assert_eq!(env_set_local(env_bits, 5, Var::int(999).as_u64()), 0);

            // Clean up
            let env_var = Var::from_u64(env_bits);
            if let Some(env_ptr) = env_var.as_environment() {
                // mmtk handles cleanup automatically
            }
        }
    }

    #[test]
    fn test_jit_helper_lexical_access() {
        crate::gc::ensure_mmtk_initialized_for_tests();
        unsafe {
            // Create parent environment
            let parent_bits = env_create(2, 0);
            env_set_local(parent_bits, 0, Var::int(100).as_u64());
            env_set_local(parent_bits, 1, Var::int(200).as_u64());

            // Create child environment
            let child_bits = env_create(1, parent_bits);
            env_set_local(child_bits, 0, Var::int(10).as_u64());

            // Test lexical get - current frame (depth 0)
            let val = Var::from_u64(env_get(child_bits, 0, 0));
            assert_eq!(val.as_int(), Some(10));

            // Test lexical get - parent frame (depth 1)
            let val = Var::from_u64(env_get(child_bits, 1, 0));
            assert_eq!(val.as_int(), Some(100));

            let val = Var::from_u64(env_get(child_bits, 1, 1));
            assert_eq!(val.as_int(), Some(200));

            // Test lexical set - parent frame
            let result = env_set(child_bits, 1, 1, Var::int(999).as_u64());
            assert_eq!(result, child_bits); // env_set returns the environment unchanged
            let val = Var::from_u64(env_get(child_bits, 1, 1));
            assert_eq!(val.as_int(), Some(999));

            // Test out of bounds
            assert_eq!(env_get(child_bits, 0, 5), Var::none().as_u64());
            assert_eq!(env_get(child_bits, 5, 0), Var::none().as_u64());
            let result = env_set(child_bits, 0, 5, Var::int(1).as_u64());
            assert_eq!(result, child_bits); // env_set always returns the environment even if out of bounds

            // Clean up
            let child_var = Var::from_u64(child_bits);
            let parent_var = Var::from_u64(parent_bits);
            if let Some(child_ptr) = child_var.as_environment() {
                // mmtk handles cleanup automatically
            }
            if let Some(parent_ptr) = parent_var.as_environment() {
                // mmtk handles cleanup automatically
            }
        }
    }

    #[test]
    fn test_jit_helper_utilities() {
        crate::gc::ensure_mmtk_initialized_for_tests();
        unsafe {
            let env_bits = env_create(5, 0);

            // Test size function
            assert_eq!(env_get_size(env_bits), 5);

            // Test parent function (should be 0 for no parent)
            assert_eq!(env_get_parent(env_bits), 0);

            // Test type checking
            assert_eq!(var_is_environment(env_bits), 1);
            assert_eq!(var_is_environment(Var::int(42).as_u64()), 0);
            assert_eq!(var_is_environment(Var::none().as_u64()), 0);

            // Create child with parent and test parent function
            let child_bits = env_create(1, env_bits);
            assert_eq!(env_get_parent(child_bits), env_bits);

            // Clean up
            let env_var = Var::from_u64(env_bits);
            let child_var = Var::from_u64(child_bits);
            if let Some(env_ptr) = env_var.as_environment() {
                // mmtk handles cleanup automatically
            }
            if let Some(child_ptr) = child_var.as_environment() {
                // mmtk handles cleanup automatically
            }
        }
    }

    #[test]
    fn test_jit_helper_error_handling() {
        // Test invalid environment bits
        assert_eq!(env_get_size(0), 0);
        assert_eq!(env_get_parent(0), 0);
        assert_eq!(env_get_local(0, 0), Var::none().as_u64());
        assert_eq!(env_set_local(0, 0, Var::int(1).as_u64()), 0);
        assert_eq!(env_get(0, 0, 0), Var::none().as_u64());
        assert_eq!(env_set(0, 0, 0, Var::int(1).as_u64()), 0); // returns 0 (unchanged env) for invalid env

        // Test invalid Var type (not environment)
        let int_bits = Var::int(42).as_u64();
        assert_eq!(env_get_size(int_bits), 0);
        assert_eq!(env_get_parent(int_bits), 0);
        assert_eq!(env_get_local(int_bits, 0), Var::none().as_u64());
        assert_eq!(env_set_local(int_bits, 0, Var::int(1).as_u64()), 0);
    }
}
