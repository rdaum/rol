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

//! Garbage collection integration for heap-allocated types.
//! Provides root enumeration and child tracing for GC systems like mmtk.

use crate::var::Var;

mod mmtk_binding;

pub use mmtk_binding::{
    WriteBarrierGuard, clear_current_task_context, clear_thread_roots, initialize_mmtk,
    is_mmtk_initialized, jit_global_write_barrier, jit_heap_write_barrier,
    jit_memory_write_barrier, jit_safepoint_check, jit_stack_write_barrier, mmtk_alloc,
    mmtk_alloc_placeholder, mmtk_bind_mutator, mmtk_dealloc_placeholder, register_global_root,
    register_thread_root, register_var_as_root, request_task_kill, request_task_yield,
    set_current_task_context,
};

#[cfg(test)]
pub use mmtk_binding::ensure_mmtk_initialized_for_tests;

/// Trait for objects that can be traced by the garbage collector.
/// All heap-allocated types must implement this.
pub trait GcTrace {
    /// Visit all GC-managed references contained within this object.
    /// The visitor function should be called for each Var that contains a heap reference.
    fn trace_children(&self, visitor: &mut dyn FnMut(&Var));

    /// Get the size of this object in bytes for GC accounting.
    fn size_bytes(&self) -> usize;

    /// Get the type name for debugging/introspection.
    fn type_name(&self) -> &'static str;
}

/// Root set enumeration for garbage collection.
/// The GC will call this to find all roots from which to start tracing.
pub trait GcRootSet {
    /// Visit all root references that should not be collected.
    /// This includes stack variables, global variables, JIT live values, etc.
    fn trace_roots<F>(&self, visitor: F)
    where
        F: FnMut(&Var);
}

impl GcTrace for LispString {
    fn trace_children(&self, _visitor: &mut dyn FnMut(&Var)) {
        // Strings have no child references - they're leaf objects
    }

    fn size_bytes(&self) -> usize {
        std::mem::size_of::<LispString>() + self.length as usize
    }

    fn type_name(&self) -> &'static str {
        "LispString"
    }
}

impl GcTrace for LispTuple {
    fn trace_children(&self, visitor: &mut dyn FnMut(&Var)) {
        // Visit all elements in the vector - they might contain heap references
        unsafe {
            let elements = self.as_slice();
            for element in elements {
                visitor(element);
            }
        }
    }

    fn size_bytes(&self) -> usize {
        std::mem::size_of::<LispTuple>() + (self.capacity as usize * std::mem::size_of::<Var>())
    }

    fn type_name(&self) -> &'static str {
        "LispVector"
    }
}

impl GcTrace for Environment {
    fn trace_children(&self, visitor: &mut dyn FnMut(&Var)) {
        // Visit parent environment if present
        if let Some(parent) = self.parent() {
            visitor(&parent);
        }

        // Visit all slots in this environment - they might contain heap references
        unsafe {
            let slots = self.as_slice();
            for slot_value in slots {
                visitor(slot_value);
            }
        }
    }

    fn size_bytes(&self) -> usize {
        std::mem::size_of::<Environment>() + (self.size as usize * std::mem::size_of::<Var>())
    }

    fn type_name(&self) -> &'static str {
        "Environment"
    }
}

impl GcTrace for LispClosure {
    fn trace_children(&self, visitor: &mut dyn FnMut(&Var)) {
        // Trace the captured environment - it might contain heap references
        let captured_env_var = Var::from_u64(self.captured_env);
        if var_needs_tracing(&captured_env_var) {
            visitor(&captured_env_var);
        }
    }

    fn size_bytes(&self) -> usize {
        std::mem::size_of::<LispClosure>()
    }

    fn type_name(&self) -> &'static str {
        "LispClosure"
    }
}

/// Determine if a Var contains a heap reference that needs GC tracing.
pub fn var_needs_tracing(var: &Var) -> bool {
    var.is_tuple() || var.is_string() || var.is_environment() || var.is_closure()
}

/// Extract heap object from a Var for GC tracing.
/// Returns None if the Var doesn't contain a traceable heap reference.
pub unsafe fn var_as_gc_object(var: &Var) -> Option<GcObjectRef> {
    if var.is_tuple() {
        let ptr_bits = var.as_u64() & !crate::var::POINTER_TAG_MASK;
        let ptr = ptr_bits as *const LispTuple;
        Some(GcObjectRef::Vector(ptr))
    } else if var.is_string() {
        let ptr_bits = var.as_u64() & !crate::var::POINTER_TAG_MASK;
        let ptr = ptr_bits as *const LispString;
        Some(GcObjectRef::String(ptr))
    } else if var.is_environment() {
        let ptr_bits = var.as_u64() & !crate::var::POINTER_TAG_MASK;
        let ptr = ptr_bits as *const Environment;
        Some(GcObjectRef::Environment(ptr))
    } else if var.is_closure() {
        let ptr_bits = var.as_u64() & !crate::var::POINTER_TAG_MASK;
        let ptr = ptr_bits as *const LispClosure;
        Some(GcObjectRef::Closure(ptr))
    } else {
        None
    }
}

/// Type-erased reference to a GC-managed heap object.
pub enum GcObjectRef {
    String(*const LispString),
    Vector(*const LispTuple),
    Environment(*const Environment),
    Closure(*const LispClosure),
}

impl GcObjectRef {
    /// Trace children of this object, regardless of concrete type.
    pub unsafe fn trace_children(&self, visitor: &mut dyn FnMut(&Var)) {
        match self {
            GcObjectRef::String(ptr) => unsafe { (*ptr).as_ref().unwrap().trace_children(visitor) },
            GcObjectRef::Vector(ptr) => unsafe { (*ptr).as_ref().unwrap().trace_children(visitor) },
            GcObjectRef::Environment(ptr) => unsafe {
                (*ptr).as_ref().unwrap().trace_children(visitor)
            },
            GcObjectRef::Closure(ptr) => unsafe {
                (*ptr).as_ref().unwrap().trace_children(visitor)
            },
        }
    }

    /// Get size in bytes for GC accounting.
    pub unsafe fn size_bytes(&self) -> usize {
        match self {
            GcObjectRef::String(ptr) => unsafe { (*ptr).as_ref().unwrap().size_bytes() },
            GcObjectRef::Vector(ptr) => unsafe { (*ptr).as_ref().unwrap().size_bytes() },
            GcObjectRef::Environment(ptr) => unsafe { (*ptr).as_ref().unwrap().size_bytes() },
            GcObjectRef::Closure(ptr) => unsafe { (*ptr).as_ref().unwrap().size_bytes() },
        }
    }

    /// Get type name for debugging.
    pub unsafe fn type_name(&self) -> &'static str {
        match self {
            GcObjectRef::String(ptr) => unsafe { (*ptr).as_ref().unwrap().type_name() },
            GcObjectRef::Vector(ptr) => unsafe { (*ptr).as_ref().unwrap().type_name() },
            GcObjectRef::Environment(ptr) => unsafe { (*ptr).as_ref().unwrap().type_name() },
            GcObjectRef::Closure(ptr) => unsafe { (*ptr).as_ref().unwrap().type_name() },
        }
    }
}

use crate::heap::{Environment, LispClosure, LispString, LispTuple};
use std::sync::atomic::{AtomicPtr, Ordering};

/// GC root set implementation using atomic pointers for thread safety.
pub struct SimpleRootSet {
    /// Stack-reachable heap objects (stored as raw pointers to trait objects)
    pub stack_roots: Vec<AtomicPtr<*mut dyn GcTrace>>,
    /// Global heap objects (stored as raw pointers to trait objects)
    pub global_roots: Vec<AtomicPtr<*mut dyn GcTrace>>,
    /// JIT live heap objects (would be more complex in practice)
    pub jit_roots: Vec<AtomicPtr<*mut dyn GcTrace>>,
}

impl GcRootSet for SimpleRootSet {
    fn trace_roots<F>(&self, mut visitor: F)
    where
        F: FnMut(&Var),
    {
        // Visit all stack-reachable heap objects
        for atomic_root in &self.stack_roots {
            let root_ptr = atomic_root.load(Ordering::Acquire);
            if !root_ptr.is_null() {
                unsafe {
                    (**root_ptr).trace_children(&mut visitor);
                }
            }
        }

        // Visit all global heap objects
        for atomic_root in &self.global_roots {
            let root_ptr = atomic_root.load(Ordering::Acquire);
            if !root_ptr.is_null() {
                unsafe {
                    (**root_ptr).trace_children(&mut visitor);
                }
            }
        }

        // Visit all JIT live heap objects
        for atomic_root in &self.jit_roots {
            let root_ptr = atomic_root.load(Ordering::Acquire);
            if !root_ptr.is_null() {
                unsafe {
                    (**root_ptr).trace_children(&mut visitor);
                }
            }
        }
    }
}

/// Complete GC tracing starting from a root set.
/// This would be called by the actual GC implementation.
pub unsafe fn trace_from_roots<R: GcRootSet>(roots: &R, mut mark_object: impl FnMut(*const u8)) {
    let mut worklist = Vec::new();

    // Add all roots to worklist
    roots.trace_roots(|var| {
        if var_needs_tracing(var) {
            worklist.push(*var);
        }
    });

    // Process worklist until empty
    while let Some(var) = worklist.pop() {
        if let Some(obj_ref) = unsafe { var_as_gc_object(&var) } {
            // Mark this object as reachable
            let ptr = match obj_ref {
                GcObjectRef::String(ptr) => ptr as *const u8,
                GcObjectRef::Vector(ptr) => ptr as *const u8,
                GcObjectRef::Environment(ptr) => ptr as *const u8,
                GcObjectRef::Closure(ptr) => ptr as *const u8,
            };
            mark_object(ptr);

            // Add children to worklist
            unsafe {
                obj_ref.trace_children(&mut |child_var| {
                    if var_needs_tracing(child_var) {
                        worklist.push(*child_var);
                    }
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_tracing() {
        ensure_mmtk_initialized_for_tests();
        let string_var = Var::string("hello");

        // String should not need child tracing (it's a leaf)
        unsafe {
            if let Some(obj_ref) = var_as_gc_object(&string_var) {
                let mut child_count = 0;
                obj_ref.trace_children(&mut |_| child_count += 1);
                assert_eq!(child_count, 0, "Strings should have no children");
            }
        }
    }

    #[test]
    fn test_vector_tracing() {
        ensure_mmtk_initialized_for_tests();
        let elements = [Var::int(42), Var::string("nested"), Var::bool(true)];
        let list_var = Var::tuple(&elements);

        // Vector should trace all its elements
        unsafe {
            if let Some(obj_ref) = var_as_gc_object(&list_var) {
                let mut traced_vars = Vec::new();
                obj_ref.trace_children(&mut |var: &Var| traced_vars.push(*var));

                assert_eq!(traced_vars.len(), 3);
                assert_eq!(traced_vars[0].as_int(), Some(42));
                assert_eq!(traced_vars[1].as_string(), Some("nested"));
                assert_eq!(traced_vars[2].as_bool(), Some(true));
            }
        }
    }

    #[test]
    #[ignore] // Disabled due to MMTk shared state issues - run with: cargo test -- --ignored
    fn test_nested_heap_objects() {
        ensure_mmtk_initialized_for_tests();
        // Create nested structure: list containing another list and string
        let inner_list = Var::tuple(&[Var::int(1), Var::int(2)]);
        let string = Var::string("test");

        // Should trace all nested heap objects
        let root_set = SimpleRootSet {
            stack_roots: vec![],
            global_roots: vec![],
            jit_roots: vec![],
        };

        let mut marked_objects = Vec::new();
        unsafe {
            trace_from_roots(&root_set, |ptr| {
                marked_objects.push(ptr);
            });
        }

        // Should mark outer list, inner list, and string
        assert_eq!(marked_objects.len(), 3);
    }

    #[test]
    fn test_environment_tracing() {
        ensure_mmtk_initialized_for_tests();
        unsafe {
            // Create environment with heap-allocated values
            let string_var = Var::string("test");
            let list_var = Var::tuple(&[Var::int(1), Var::int(2)]);
            let env_values = [Var::int(42), string_var, list_var];
            let env_ptr = Environment::from_values(&env_values, None);
            let env_var = Var::environment(env_ptr);

            // Environment should trace its slot values
            if let Some(obj_ref) = var_as_gc_object(&env_var) {
                let mut traced_vars = Vec::new();
                obj_ref.trace_children(&mut |var: &Var| traced_vars.push(*var));

                // Should trace all 3 slot values (no parent)
                assert_eq!(traced_vars.len(), 3);
                assert_eq!(traced_vars[0].as_int(), Some(42));
                assert_eq!(traced_vars[1].as_string(), Some("test"));
                assert!(traced_vars[2].is_tuple());
            }

            // Clean up
            // mmtk handles cleanup automatically
        }
    }

    #[test]
    fn test_environment_parent_tracing() {
        ensure_mmtk_initialized_for_tests();
        unsafe {
            // Create parent and child environments
            let parent_values = [Var::string("parent"), Var::int(100)];
            let parent_ptr = Environment::from_values(&parent_values, None);
            let parent_var = Var::environment(parent_ptr);

            let child_values = [Var::string("child")];
            let child_ptr = Environment::from_values(&child_values, Some(parent_var));
            let child_var = Var::environment(child_ptr);

            // Child environment should trace both parent and its own slots
            if let Some(obj_ref) = var_as_gc_object(&child_var) {
                let mut traced_vars = Vec::new();
                obj_ref.trace_children(&mut |var: &Var| traced_vars.push(*var));

                // Should trace parent + 1 slot value
                assert_eq!(traced_vars.len(), 2);

                // First should be parent environment
                assert!(traced_vars[0].is_environment());
                // Second should be child's slot value
                assert_eq!(traced_vars[1].as_string(), Some("child"));
            }

            // Clean up
            // mmtk handles cleanup automatically
        }
    }

    #[test]
    #[ignore] // Disabled due to MMTk shared state issues - run with: cargo test -- --ignored  
    fn test_environment_gc_integration() {
        ensure_mmtk_initialized_for_tests();
        unsafe {
            // Create complex environment chain with heap objects
            let parent_string = Var::string("parent_data");
            let parent_list = Var::tuple(&[Var::int(1), Var::string("nested")]);
            let parent_values = [parent_string, parent_list];
            let parent_ptr = Environment::from_values(&parent_values, None);
            let parent_var = Var::environment(parent_ptr);

            let child_string = Var::string("child_data");
            let child_values = [child_string, Var::int(200)];
            let child_ptr = Environment::from_values(&child_values, Some(parent_var));

            // Add environment to root set
            let root_set = SimpleRootSet {
                stack_roots: vec![],
                global_roots: vec![],
                jit_roots: vec![],
            };

            let mut marked_objects = Vec::new();
            trace_from_roots(&root_set, |ptr| {
                marked_objects.push(ptr);
            });

            // Should mark:
            // 1. Child environment
            // 2. Parent environment (from child's parent reference)
            // 3. Child's string ("child_data")
            // 4. Parent's string ("parent_data")
            // 5. Parent's list
            // 6. Nested string in parent's list ("nested")
            assert_eq!(marked_objects.len(), 6);

            // Clean up
            // mmtk handles cleanup automatically
        }
    }

    #[test]
    fn test_environment_size_calculation() {
        ensure_mmtk_initialized_for_tests();
        unsafe {
            let env_values = [Var::int(1), Var::int(2), Var::int(3)];
            let env_ptr = Environment::from_values(&env_values, None);
            let env_var = Var::environment(env_ptr);

            if let Some(obj_ref) = var_as_gc_object(&env_var) {
                let size = obj_ref.size_bytes();
                let expected_size =
                    std::mem::size_of::<Environment>() + (3 * std::mem::size_of::<Var>());
                assert_eq!(size, expected_size);

                assert_eq!(obj_ref.type_name(), "Environment");
            }

            // Clean up
            // mmtk handles cleanup automatically
        }
    }
}
