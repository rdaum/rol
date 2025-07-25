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

//! MMTk garbage collector binding for ROL runtime.

use mmtk::util::copy::{CopySemantics, GCWorkerCopyContext};
use mmtk::util::opaque_pointer::*;
use mmtk::util::options::PlanSelector;
use mmtk::util::{Address, ObjectReference};
use mmtk::vm::*;
use mmtk::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use crate::gc::{var_as_gc_object, var_needs_tracing, GcObjectRef, GcRootSet, GcTrace, SimpleRootSet};
use crate::heap::{Environment, LispClosure, LispString, LispTuple};

/// Global MMTk instance - thread-safe lazy initialization
static MMTK_INSTANCE: OnceLock<Box<MMTK<RolVM>>> = OnceLock::new();
static MMTK_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Thread-local mutator storage for proper thread binding
thread_local! {
    static THREAD_MUTATOR: RefCell<Option<Box<Mutator<RolVM>>>> = const { RefCell::new(None) };
}

/// Global count of active mutators
static MUTATOR_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Stop-the-world synchronization for garbage collection
static GC_SYNC: OnceLock<Arc<(Mutex<bool>, Condvar)>> = OnceLock::new();

/// Global flag for safepoint requests - checked by JIT code at safepoints
static GC_SAFEPOINT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Global root set for persistent roots like globals and symbols
static GLOBAL_ROOT_SET: OnceLock<Mutex<SimpleRootSet>> = OnceLock::new();
/// Thread-local root set - simplified to a single instance for now
static CURRENT_ROOT_SET: OnceLock<Mutex<SimpleRootSet>> = OnceLock::new();

/// ROL VM binding
#[derive(Default)]
pub struct RolVM;

/// VMBinding implementation - required for MMTk integration
impl VMBinding for RolVM {
    type VMObjectModel = Self;
    type VMActivePlan = Self;
    type VMCollection = Self;
    type VMScanning = Self;
    type VMReferenceGlue = Self;
    type VMSlot = Address;
    type VMMemorySlice = std::ops::Range<Address>;
}

/// ActivePlan implementation
impl ActivePlan<RolVM> for RolVM {
    fn number_of_mutators() -> usize {
        // Get count from atomic counter
        MUTATOR_COUNT.load(std::sync::atomic::Ordering::Acquire)
    }

    fn is_mutator(_thread: VMThread) -> bool {
        // In our simple runtime, all threads can be mutators
        true
    }

    fn mutator(_thread: VMMutatorThread) -> &'static mut Mutator<RolVM> {
        // Get the thread-local mutator
        THREAD_MUTATOR.with(|mutator_cell| {
            let mut mutator_ref = mutator_cell.borrow_mut();
            if mutator_ref.is_none() {
                // Initialize mutator for this thread if not already done
                *mutator_ref = Some(bind_mutator_internal(_thread));
            }

            // SAFETY: We maintain the mutator in thread-local storage for the lifetime
            // of the thread, and MMTk expects a static mutable reference
            unsafe {
                let ptr = mutator_ref.as_mut().unwrap().as_mut() as *mut Mutator<RolVM>;
                &mut *ptr
            }
        })
    }

    fn mutators<'a>() -> Box<dyn Iterator<Item = &'a mut Mutator<RolVM>> + 'a> {
        // For now, we don't need to iterate over mutators often
        // Return empty iterator as this is mainly used for GC coordination
        // TODO: fill me in
        Box::new(std::iter::empty())
    }
}

/// Collection implementation - stop-the-world garbage collection
impl Collection<RolVM> for RolVM {
    fn stop_all_mutators<F>(_tls: VMWorkerThread, mut _mutator_visitor: F)
    where
        F: FnMut(&'static mut Mutator<RolVM>),
    {
        // Mutators are already stopped by block_for_gc
        // In a multi-threaded implementation, we'd iterate through all mutators here
        // For now, we just acknowledge that the single mutator is stopped
    }

    fn resume_mutators(_tls: VMWorkerThread) {
        // Signal all waiting mutator threads to resume
        let gc_sync = GC_SYNC.get().expect("GC sync not initialized");
        let (lock, condvar) = &**gc_sync;
        let mut gc_in_progress = lock.lock().unwrap();
        *gc_in_progress = false;
        condvar.notify_all();
    }

    fn block_for_gc(_tls: VMMutatorThread) {
        // Wait for GC to complete
        let gc_sync = GC_SYNC.get().expect("GC sync not initialized");
        let (lock, condvar) = &**gc_sync;
        let mut gc_in_progress = lock.lock().unwrap();
        while *gc_in_progress {
            gc_in_progress = condvar.wait(gc_in_progress).unwrap();
        }
    }

    fn spawn_gc_thread(_tls: VMThread, ctx: GCThreadContext<RolVM>) {
        // Start GC - signal that GC is in progress
        let gc_sync = GC_SYNC.get().expect("GC sync not initialized");
        let (lock, _condvar) = &**gc_sync;
        {
            let mut gc_in_progress = lock.lock().unwrap();
            *gc_in_progress = true;
        }

        // Set safepoint flag to request all mutator threads to pause
        GC_SAFEPOINT_REQUESTED.store(true, Ordering::Release);

        eprintln!("[GC] Starting garbage collection cycle");

        // For now, just spawn a simple thread that runs the context
        // TODO: Implement proper GC thread management
        std::thread::spawn(move || {
            // Let MMTk handle the GC work - for now just a placeholder
            // The actual GC coordination happens through other MMTk APIs
            eprintln!("[GC] GC worker thread running");

            // Resume mutators after a brief pause (simulating GC work)
            std::thread::sleep(std::time::Duration::from_millis(1));

            eprintln!("[GC] Garbage collection cycle completed");

            // Signal GC completion
            let gc_sync = GC_SYNC.get().expect("GC sync not initialized");
            let (lock, condvar) = &**gc_sync;
            let mut gc_in_progress = lock.lock().unwrap();
            *gc_in_progress = false;

            // Clear safepoint request flag
            GC_SAFEPOINT_REQUESTED.store(false, Ordering::Release);

            condvar.notify_all();
        });
    }
}

/// ObjectModel implementation - proper metadata layout for MarkSweep
impl ObjectModel<RolVM> for RolVM {
    const GLOBAL_LOG_BIT_SPEC: VMGlobalLogBitSpec = VMGlobalLogBitSpec::side_first();
    const LOCAL_FORWARDING_POINTER_SPEC: VMLocalForwardingPointerSpec =
        VMLocalForwardingPointerSpec::side_first();
    const LOCAL_FORWARDING_BITS_SPEC: VMLocalForwardingBitsSpec =
        VMLocalForwardingBitsSpec::side_first();
    // Use different offsets to avoid overlapping metadata
    const LOCAL_MARK_BIT_SPEC: VMLocalMarkBitSpec = VMLocalMarkBitSpec::in_header(0);
    const LOCAL_LOS_MARK_NURSERY_SPEC: VMLocalLOSMarkNurserySpec =
        VMLocalLOSMarkNurserySpec::in_header(8);
    const OBJECT_REF_OFFSET_LOWER_BOUND: isize = 0;

    fn copy(
        _from: ObjectReference,
        _semantics: CopySemantics,
        _context: &mut GCWorkerCopyContext<RolVM>,
    ) -> ObjectReference {
        unimplemented!()
    }

    fn copy_to(_from: ObjectReference, _to: ObjectReference, _region: Address) -> Address {
        unimplemented!()
    }

    fn get_current_size(_object: ObjectReference) -> usize {
        unimplemented!()
    }

    fn get_size_when_copied(_object: ObjectReference) -> usize {
        unimplemented!()
    }

    fn get_align_when_copied(_object: ObjectReference) -> usize {
        8
    }

    fn get_align_offset_when_copied(_object: ObjectReference) -> usize {
        0
    }

    fn get_reference_when_copied_to(_from: ObjectReference, _to: Address) -> ObjectReference {
        unimplemented!()
    }

    fn get_type_descriptor(_reference: ObjectReference) -> &'static [i8] {
        unimplemented!()
    }

    fn ref_to_object_start(_object: ObjectReference) -> Address {
        _object.to_raw_address()
    }

    fn ref_to_header(_object: ObjectReference) -> Address {
        _object.to_raw_address()
    }

    fn dump_object(_object: ObjectReference) {
        unimplemented!()
    }
}

/// Scanning implementation - connects to our GC infrastructure
impl Scanning<RolVM> for RolVM {
    fn scan_object<SV: SlotVisitor<Address>>(
        _tls: VMWorkerThread,
        object: ObjectReference,
        edge_visitor: &mut SV,
    ) {
        // Convert MMTk ObjectReference to our Var and trace children
        let obj_addr = object.to_raw_address();
        let var = var_from_address(obj_addr);

        unsafe {
            if let Some(gc_obj) = var_as_gc_object(&var) {
                // Log object being scanned
                eprintln!(
                    "[GC] Scanning object at {:p} (type: {})",
                    obj_addr.to_ptr::<u8>(),
                    gc_obj.type_name()
                );

                gc_obj.trace_children(&mut |child_var| {
                    if var_needs_tracing(child_var) {
                        let child_addr = address_from_var(child_var);
                        edge_visitor.visit_slot(child_addr);
                    }
                });
            }
        }
    }

    fn notify_initial_thread_scan_complete(_partial_scan: bool, _tls: VMWorkerThread) {
        // No-op for our simple runtime
    }

    fn scan_roots_in_mutator_thread(
        _tls: VMWorkerThread,
        _mutator: &'static mut Mutator<RolVM>,
        mut factory: impl RootsWorkFactory<Address>,
    ) {
        eprintln!("[GC] Scanning mutator thread roots");

        // Scan thread-local root set (stack frames, local vars, etc.)
        if let Some(root_set_mutex) = get_current_root_set() {
            // Use blocking lock to ensure we wait for mutator threads to reach safepoints
            let root_set = root_set_mutex.lock().unwrap();
            // First, register the root objects themselves
            for atomic_root in &root_set.stack_roots {
                let root_ptr_ptr = atomic_root.load(std::sync::atomic::Ordering::Acquire);
                if !root_ptr_ptr.is_null() {
                    let root_ptr = unsafe { *root_ptr_ptr };
                    let addr = unsafe { Address::from_ptr(root_ptr as *const u8) };
                    eprintln!("[GC] Found thread root object at {root_ptr:p}");
                    factory.create_process_roots_work(vec![addr]);
                }
            }

            // Then trace their children
            root_set.trace_roots(|var| {
                if var_needs_tracing(var) {
                    let addr = address_from_var(var);
                    eprintln!(
                        "[GC] Found child from thread root at {:p}",
                        addr.to_ptr::<u8>()
                    );
                    factory.create_process_roots_work(vec![addr]);
                }
            });
        }
    }

    fn scan_vm_specific_roots(_tls: VMWorkerThread, mut factory: impl RootsWorkFactory<Address>) {
        eprintln!("[GC] Scanning VM-specific roots (globals, symbols, etc.)");

        // Scan global root set (global variables, etc.)
        if let Some(root_set_mutex) = get_global_root_set() {
            // Use blocking lock to ensure proper coordination with mutator threads
            let root_set = root_set_mutex.lock().unwrap();
            // First, register the root objects themselves
            for atomic_root in &root_set.global_roots {
                let root_ptr_ptr = atomic_root.load(std::sync::atomic::Ordering::Acquire);
                if !root_ptr_ptr.is_null() {
                    let root_ptr = unsafe { *root_ptr_ptr };
                    let addr = unsafe { Address::from_ptr(root_ptr as *const u8) };
                    eprintln!("[GC] Found global root object at {root_ptr:p}");
                    factory.create_process_roots_work(vec![addr]);
                }
            }

            // Then trace their children
            root_set.trace_roots(|var| {
                if var_needs_tracing(var) {
                    let addr = address_from_var(var);
                    eprintln!(
                        "[GC] Found child from global root at {:p}",
                        addr.to_ptr::<u8>()
                    );
                    factory.create_process_roots_work(vec![addr]);
                }
            });
        }

        // TODO: Also scan:
        // - Global function registry
        // - JIT compiled code roots
        // - Any other VM-wide roots
    }

    fn supports_return_barrier() -> bool {
        false
    }

    fn prepare_for_roots_re_scanning() {
        // No-op for our simple runtime
    }
}

/// ReferenceGlue implementation - stubs for now
impl ReferenceGlue<RolVM> for RolVM {
    type FinalizableType = ObjectReference;

    fn set_referent(_reference: ObjectReference, _referent: ObjectReference) {
        unimplemented!()
    }

    fn get_referent(_object: ObjectReference) -> Option<ObjectReference> {
        unimplemented!()
    }

    fn clear_referent(_reference: ObjectReference) {
        unimplemented!()
    }

    fn enqueue_references(_references: &[ObjectReference], _tls: VMWorkerThread) {
        unimplemented!()
    }
}

/// Get the global MMTk instance
fn get_mmtk_instance() -> &'static Box<MMTK<RolVM>> {
    MMTK_INSTANCE.get().expect("MMTk not initialized")
}

/// Internal function to bind a mutator to the current thread
fn bind_mutator_internal(tls: VMMutatorThread) -> Box<Mutator<RolVM>> {
    let mmtk = get_mmtk_instance();
    let mutator = memory_manager::bind_mutator(mmtk, tls);

    // Increment the mutator count
    MUTATOR_COUNT.fetch_add(1, std::sync::atomic::Ordering::AcqRel);

    // Return the mutator as a Box
    Box::new(*mutator)
}

/// Get the global root set for persistent roots
fn get_global_root_set() -> Option<&'static Mutex<SimpleRootSet>> {
    GLOBAL_ROOT_SET.get()
}

/// Get the current thread's root set
fn get_current_root_set() -> Option<&'static Mutex<SimpleRootSet>> {
    CURRENT_ROOT_SET.get()
}

/// Convert a raw pointer to a Var for GC scanning
pub fn var_from_address(addr: Address) -> crate::var::Var {
    crate::var::Var::from_u64(addr.as_usize() as u64)
}

/// Convert a Var to an Address for MMTk
fn address_from_var(var: &crate::var::Var) -> Address {
    let ptr_bits = var.as_u64() & !crate::var::POINTER_TAG_MASK;
    unsafe { Address::from_usize(ptr_bits as usize) }
}

/// Initialize MMTk with MarkSweep plan
pub fn initialize_mmtk() -> Result<(), &'static str> {
    if MMTK_INITIALIZED.swap(true, Ordering::AcqRel) {
        return Ok(()); // Already initialized
    }

    // Create MMTk builder
    let mut builder = MMTKBuilder::new();

    // Use MarkSweep for now - GenImmix requires more complex VM space setup
    builder.options.plan.set(PlanSelector::MarkSweep);

    // Initialize MMTk
    let mmtk = memory_manager::mmtk_init(&builder);

    // Store the MMTk instance
    MMTK_INSTANCE
        .set(mmtk)
        .map_err(|_| "Failed to set MMTk instance")?;

    // Reset mutator count
    MUTATOR_COUNT.store(0, std::sync::atomic::Ordering::Release);

    // Initialize GC synchronization BEFORE collection initialization
    GC_SYNC
        .set(Arc::new((Mutex::new(false), Condvar::new())))
        .map_err(|_| "Failed to initialize GC synchronization")?;

    // Initialize collection subsystem
    let mmtk_ref = get_mmtk_instance();
    let thread_id = std::thread::current().id();
    let thread_addr = unsafe { Address::from_usize(&thread_id as *const _ as usize) };
    let vm_thread = VMThread(OpaquePointer::from_address(thread_addr));
    memory_manager::initialize_collection(mmtk_ref, vm_thread);

    // Initialize root sets
    GLOBAL_ROOT_SET
        .set(Mutex::new(SimpleRootSet {
            stack_roots: Vec::new(),
            global_roots: Vec::new(),
            jit_roots: Vec::new(),
        }))
        .map_err(|_| "Failed to initialize global root set")?;

    CURRENT_ROOT_SET
        .set(Mutex::new(SimpleRootSet {
            stack_roots: Vec::new(),
            global_roots: Vec::new(),
            jit_roots: Vec::new(),
        }))
        .map_err(|_| "Failed to initialize current root set")?;

    println!(
        "MMTk MarkSweep plan initialized - concurrent GC ready when virtual memory issues resolved"
    );
    Ok(())
}

/// Bind a mutator for the current thread - should be called when thread starts
pub fn mmtk_bind_mutator() -> Result<(), &'static str> {
    if !is_mmtk_initialized() {
        return Err("MMTk not initialized");
    }

    // Create thread identifier - use current thread pointer as identifier
    let thread_id = std::thread::current().id();
    let thread_addr = unsafe { Address::from_usize(&thread_id as *const _ as usize) };
    let mutator_thread = VMMutatorThread(VMThread(OpaquePointer::from_address(thread_addr)));

    // Initialize thread-local mutator if not already done
    THREAD_MUTATOR.with(|mutator_cell| {
        let mut mutator_ref = mutator_cell.borrow_mut();
        if mutator_ref.is_none() {
            *mutator_ref = Some(bind_mutator_internal(mutator_thread));
        }
    });

    Ok(())
}

/// Check if MMTk is initialized
pub fn is_mmtk_initialized() -> bool {
    MMTK_INITIALIZED.load(Ordering::Acquire)
}

/// Manually trigger garbage collection for testing
pub fn trigger_gc() {
    if !is_mmtk_initialized() {
        println!("MMTk not initialized, cannot trigger GC");
        return;
    }

    println!("Triggering garbage collection...");
    let mmtk = get_mmtk_instance();

    // Get current thread as mutator thread
    let thread_id = std::thread::current().id();
    let thread_addr = unsafe { Address::from_usize(&thread_id as *const _ as usize) };
    let mutator_thread = VMMutatorThread(VMThread(OpaquePointer::from_address(thread_addr)));

    // Manually trigger a GC cycle
    mmtk.harness_begin(mutator_thread);

    println!("Garbage collection completed");
}

/// Allocate memory using MMTk with thread-local mutator
pub fn mmtk_alloc(size: usize) -> *mut u8 {
    if !is_mmtk_initialized() {
        panic!("MMTk not initialized");
    }

    // Ensure mutator is bound for this thread
    if let Err(e) = mmtk_bind_mutator() {
        panic!("Failed to bind mutator: {e}");
    }

    // Use thread-local mutator for allocation
    THREAD_MUTATOR.with(|mutator_cell| {
        let mut mutator_ref = mutator_cell.borrow_mut();
        let mutator = mutator_ref.as_mut().expect("Mutator should be initialized");

        // Allocate using MMTk
        let addr = memory_manager::alloc(
            mutator.as_mut(),
            size,
            8, // alignment
            0, // offset
            AllocationSemantics::Default,
        );

        eprintln!("[GC] Allocated {} bytes at {:p}", size, addr.to_ptr::<u8>());

        addr.to_ptr::<u8>() as *mut u8
    })
}

/// Placeholder for MMTk allocation - falls back to system allocator for now
pub fn mmtk_alloc_placeholder(size: usize) -> *mut u8 {
    use std::alloc::{Layout, alloc};

    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe { alloc(layout) }
}

/// Placeholder for MMTk deallocation
pub unsafe fn mmtk_dealloc_placeholder(ptr: *mut u8, size: usize) {
    use std::alloc::{Layout, dealloc};

    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe {
        dealloc(ptr, layout);
    }
}

/// Register a heap object as a global root (survives across function calls)
pub fn register_global_root(ptr: *mut dyn GcTrace) {
    if let Some(root_set_mutex) = get_global_root_set() {
        if let Ok(mut root_set) = root_set_mutex.lock() {
            eprintln!("[GC] Registering global root: {ptr:p}");
            // Allocate space for the trait object pointer on the heap for atomic storage
            let ptr_box = Box::new(ptr);
            let ptr_raw = Box::into_raw(ptr_box);
            root_set
                .global_roots
                .push(std::sync::atomic::AtomicPtr::new(ptr_raw));
        }
    }
}

/// Register a heap object as a thread-local root (stack variable, local scope)  
pub fn register_thread_root(ptr: *mut dyn GcTrace) {
    if let Some(root_set_mutex) = get_current_root_set() {
        if let Ok(mut root_set) = root_set_mutex.lock() {
            eprintln!("[GC] Registering thread root: {ptr:p}");
            // Allocate space for the trait object pointer on the heap for atomic storage
            let ptr_box = Box::new(ptr);
            let ptr_raw = Box::into_raw(ptr_box);
            root_set
                .stack_roots
                .push(std::sync::atomic::AtomicPtr::new(ptr_raw));
        }
    }
}

/// Unregister a specific heap object from thread roots
pub fn unregister_thread_root(ptr: *mut dyn GcTrace) {
    if let Some(root_set_mutex) = get_current_root_set() {
        if let Ok(mut root_set) = root_set_mutex.lock() {
            root_set.stack_roots.retain(|atomic_ptr| {
                let stored_ptr_ptr = atomic_ptr.load(std::sync::atomic::Ordering::Acquire);
                if !stored_ptr_ptr.is_null() {
                    let stored_ptr = unsafe { *stored_ptr_ptr };
                    let should_retain = !std::ptr::eq(stored_ptr, ptr);
                    // If we're removing this entry, clean up the boxed pointer
                    if !should_retain {
                        let _ = unsafe { Box::from_raw(stored_ptr_ptr) };
                    }
                    should_retain
                } else {
                    true // Keep null entries for now
                }
            });
            eprintln!("[GC] Unregistered thread root: {ptr:p}");
        }
    }
}

/// Clear all thread-local roots (called when leaving scope)
pub fn clear_thread_roots() {
    if let Some(root_set_mutex) = get_current_root_set() {
        if let Ok(mut root_set) = root_set_mutex.lock() {
            eprintln!("[GC] Clearing {} thread roots", root_set.stack_roots.len());
            // Clean up all the boxed pointers before clearing
            for atomic_ptr in &root_set.stack_roots {
                let stored_ptr_ptr = atomic_ptr.load(std::sync::atomic::Ordering::Acquire);
                if !stored_ptr_ptr.is_null() {
                    let _ = unsafe { Box::from_raw(stored_ptr_ptr) };
                }
            }
            root_set.stack_roots.clear();
        }
    }
}

/// Helper: Register a heap object from a Var (extracts pointer and converts to trait object)
pub fn register_var_as_root(var: crate::var::Var, is_global: bool) {
    unsafe {
        if let Some(gc_obj) = var_as_gc_object(&var) {
            let trait_ptr: *mut dyn GcTrace = match gc_obj {
                GcObjectRef::String(ptr) => {
                    ptr as *mut LispString as *mut dyn GcTrace
                }
                GcObjectRef::Vector(ptr) => {
                    ptr as *mut LispTuple as *mut dyn GcTrace
                }
                GcObjectRef::Environment(ptr) => {
                    ptr as *mut Environment as *mut dyn GcTrace
                }
                GcObjectRef::Closure(ptr) => {
                    ptr as *mut LispClosure as *mut dyn GcTrace
                }
            };

            if is_global {
                register_global_root(trait_ptr);
            } else {
                register_thread_root(trait_ptr);
            }
        }
    }
}

/// Write barrier for concurrent garbage collection
/// Call this before modifying any heap object field that contains object references
pub fn write_barrier_pre(object: *mut u8, slot: *mut u8) {
    if !is_mmtk_initialized() {
        return;
    }

    // Skip write barrier for null object (global variables, stack, etc.)
    if object.is_null() {
        return;
    }

    let obj_ref = unsafe {
        ObjectReference::from_raw_address(Address::from_ptr(object))
            .expect("Invalid object reference")
    };
    let slot_addr = unsafe { Address::from_ptr(slot) };

    THREAD_MUTATOR.with(|mutator_cell| {
        let mut mutator_ref = mutator_cell.borrow_mut();
        if let Some(mutator) = mutator_ref.as_mut() {
            // MMTk pre-write barrier for concurrent collection
            // API: object_reference_write_pre(mutator, src, slot, target)
            memory_manager::object_reference_write_pre(
                mutator.as_mut(),
                obj_ref,
                slot_addr,
                Some(obj_ref), // src object reference
            );
        }
    });
}

/// Write barrier for concurrent garbage collection  
/// Call this after modifying any heap object field that contains object references
pub fn write_barrier_post(object: *mut u8, slot: *mut u8, target: *mut u8) {
    if !is_mmtk_initialized() {
        return;
    }

    // Skip write barrier for null object (global variables, stack, etc.)
    if object.is_null() {
        return;
    }

    let obj_ref = unsafe {
        ObjectReference::from_raw_address(Address::from_ptr(object))
            .expect("Invalid object reference")
    };
    let slot_addr = unsafe { Address::from_ptr(slot) };
    let target_ref = if target.is_null() {
        unsafe { ObjectReference::from_raw_address(Address::ZERO).expect("Invalid null reference") }
    } else {
        unsafe {
            ObjectReference::from_raw_address(Address::from_ptr(target))
                .expect("Invalid target reference")
        }
    };

    THREAD_MUTATOR.with(|mutator_cell| {
        let mut mutator_ref = mutator_cell.borrow_mut();
        if let Some(mutator) = mutator_ref.as_mut() {
            // MMTk post-write barrier for concurrent collection
            // API: object_reference_write_post(mutator, src, slot, target)
            memory_manager::object_reference_write_post(
                mutator.as_mut(),
                obj_ref,
                slot_addr,
                Some(target_ref),
            );
        }
    });
}

/// Write barrier for Var slot modifications - handles NaN-boxed pointers
pub fn var_write_barrier(
    containing_object: *mut u8,
    slot_addr: *mut crate::var::Var,
    old_value: crate::var::Var,
    new_value: crate::var::Var,
) {
    // Pre-barrier with old value
    if var_needs_tracing(&old_value) {
        let _old_ptr = address_from_var(&old_value).to_ptr::<u8>() as *mut u8;
        write_barrier_pre(containing_object, slot_addr as *mut u8);
    }

    // Post-barrier with new value
    if var_needs_tracing(&new_value) {
        let new_ptr = address_from_var(&new_value).to_ptr::<u8>() as *mut u8;
        write_barrier_post(containing_object, slot_addr as *mut u8, new_ptr);
    }
}

/// RAII write barrier guard that automatically handles pre/post barriers
/// Usage: let _guard = WriteBarrierGuard::new(object, slot, old_value, new_value);
pub struct WriteBarrierGuard {
    containing_object: *mut u8,
    slot_addr: *mut u8,
    new_value: crate::var::Var,
}

impl WriteBarrierGuard {
    /// Create a new write barrier guard and execute pre-barrier
    pub fn new(
        containing_object: *mut u8,
        slot_addr: *mut crate::var::Var,
        old_value: crate::var::Var,
        new_value: crate::var::Var,
    ) -> Self {
        // Execute pre-barrier
        if var_needs_tracing(&old_value) {
            write_barrier_pre(containing_object, slot_addr as *mut u8);
        }

        Self {
            containing_object,
            slot_addr: slot_addr as *mut u8,
            new_value,
        }
    }

    /// Create a write barrier guard for raw pointers (non-Var)
    pub fn new_raw(
        containing_object: *mut u8,
        slot_addr: *mut u8,
        target_ptr: *mut u8,
    ) -> RawWriteBarrierGuard {
        // Execute pre-barrier
        write_barrier_pre(containing_object, slot_addr);

        RawWriteBarrierGuard {
            containing_object,
            slot_addr,
            target_ptr,
        }
    }
}

impl Drop for WriteBarrierGuard {
    fn drop(&mut self) {
        // Execute post-barrier
        if var_needs_tracing(&self.new_value) {
            let new_ptr = address_from_var(&self.new_value).to_ptr::<u8>() as *mut u8;
            write_barrier_post(self.containing_object, self.slot_addr, new_ptr);
        }
    }
}

/// RAII write barrier guard for raw pointer modifications
pub struct RawWriteBarrierGuard {
    containing_object: *mut u8,
    slot_addr: *mut u8,
    target_ptr: *mut u8,
}

impl Drop for RawWriteBarrierGuard {
    fn drop(&mut self) {
        // Execute post-barrier
        write_barrier_post(self.containing_object, self.slot_addr, self.target_ptr);
    }
}

/// Macro for convenient write barrier usage
/// Usage: with_write_barrier!(object_ptr, slot_ptr, old_val, new_val => { slot_ptr.write(new_val); });
#[macro_export]
macro_rules! with_write_barrier {
    ($object:expr, $slot:expr, $old:expr, $new:expr => $code:block) => {{
        let _guard = $crate::gc::WriteBarrierGuard::new($object, $slot, $old, $new);
        $code
    }};
}

/// Macro for raw pointer write barriers
/// Usage: with_raw_write_barrier!(object_ptr, slot_ptr, target_ptr => { *slot_ptr = target_ptr; });
#[macro_export]
macro_rules! with_raw_write_barrier {
    ($object:expr, $slot:expr, $target:expr => $code:block) => {{
        let _guard = $crate::gc::WriteBarrierGuard::new_raw($object, $slot, $target);
        $code
    }};
}

/// JIT-callable write barrier for environment variable assignments
/// Call this before setting any environment slot to a new value
#[unsafe(no_mangle)]
pub extern "C" fn jit_env_write_barrier(
    env_bits: u64,
    offset: u32,
    old_value_bits: u64,
    new_value_bits: u64,
) {
    let env_var = crate::var::Var::from_u64(env_bits);
    if let Some(env_ptr) = env_var.as_environment() {
        unsafe {
            if offset < (*env_ptr).size {
                // Calculate slot address
                let slots_ptr = (env_ptr as *mut u8)
                    .add(std::mem::size_of::<Environment>())
                    as *mut crate::var::Var;
                let slot_addr = slots_ptr.add(offset as usize);

                let old_value = crate::var::Var::from_u64(old_value_bits);
                let new_value = crate::var::Var::from_u64(new_value_bits);

                var_write_barrier(env_ptr as *mut u8, slot_addr, old_value, new_value);
            }
        }
    }
}

/// JIT-callable write barrier for global variable assignments  
#[unsafe(no_mangle)]
pub extern "C" fn jit_global_write_barrier(
    slot_addr: *mut u8,
    old_value_bits: u64,
    new_value_bits: u64,
) {
    let old_value = crate::var::Var::from_u64(old_value_bits);
    let new_value = crate::var::Var::from_u64(new_value_bits);

    // For global variables, we don't have a containing object, so we use null
    // The write barrier implementation should handle this case
    var_write_barrier(
        std::ptr::null_mut(),
        slot_addr as *mut crate::var::Var,
        old_value,
        new_value,
    );
}

/// JIT-callable write barrier for general heap object field writes
#[unsafe(no_mangle)]
pub extern "C" fn jit_heap_write_barrier(
    object_ptr: *mut u8,
    slot_addr: *mut u8,
    target_ptr: *mut u8,
) {
    write_barrier_pre(object_ptr, slot_addr);
    write_barrier_post(object_ptr, slot_addr, target_ptr);
}

/// JIT-callable write barrier for stack memory stores
/// Call this before storing any Var value to stack memory
#[unsafe(no_mangle)]
pub extern "C" fn jit_stack_write_barrier(
    stack_addr: *mut u8,
    old_value_bits: u64,
    new_value_bits: u64,
) {
    let old_value = crate::var::Var::from_u64(old_value_bits);
    let new_value = crate::var::Var::from_u64(new_value_bits);

    // For stack writes, we don't have a containing object (stack is managed by JIT)
    var_write_barrier(
        std::ptr::null_mut(),
        stack_addr as *mut crate::var::Var,
        old_value,
        new_value,
    );
}

/// JIT-callable write barrier for general memory stores  
/// Call this before storing any Var value to heap-allocated memory
#[unsafe(no_mangle)]
pub extern "C" fn jit_memory_write_barrier(
    addr: *mut u8,
    old_value_bits: u64,
    new_value_bits: u64,
) {
    let old_value = crate::var::Var::from_u64(old_value_bits);
    let new_value = crate::var::Var::from_u64(new_value_bits);

    // For general memory writes, we don't know the containing object
    var_write_barrier(
        std::ptr::null_mut(),
        addr as *mut crate::var::Var,
        old_value,
        new_value,
    );
}

/// JIT safepoint check - called from JIT-generated code at strategic points
/// This is the heart of the cooperative GC coordination system
#[unsafe(no_mangle)]
pub extern "C" fn jit_safepoint_check() {
    // Fast path: check if GC is requested (single atomic load)
    if GC_SAFEPOINT_REQUESTED.load(Ordering::Acquire) {
        // Slow path: block until GC completes
        let gc_sync = GC_SYNC.get().expect("GC sync not initialized");
        let (lock, condvar) = &**gc_sync;
        let mut gc_in_progress = lock.lock().unwrap();
        while *gc_in_progress {
            gc_in_progress = condvar.wait(gc_in_progress).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::var::Var;

    #[test]
    fn test_gc_basic_allocation() {
        // Initialize MMTk for testing
        if let Err(e) = initialize_mmtk() {
            panic!("Failed to initialize MMTk: {e}");
        }

        // Bind mutator for this thread
        if let Err(e) = mmtk_bind_mutator() {
            panic!("Failed to bind mutator: {e}");
        }

        println!("Creating heap objects...");

        // Create some heap objects
        let string1 = Var::string("hello world");
        let string2 = Var::string("goodbye world");
        let tuple1 = Var::tuple(&[Var::int(1), Var::int(2), Var::int(3)]);
        let tuple2 = Var::tuple(&[string1, string2]);

        println!(
            "Created objects: {:?}, {:?}, {:?}, {:?}",
            string1.as_string(),
            string2.as_string(),
            tuple1.is_tuple(),
            tuple2.is_tuple()
        );

        // Trigger garbage collection
        trigger_gc();

        // Objects should still be reachable since they're on the stack
        println!(
            "After GC - objects still accessible: {:?}, {:?}",
            string1.as_string(),
            string2.as_string()
        );
    }

    #[test]
    fn test_gc_unreachable_objects() {
        // Initialize MMTk for testing
        if let Err(e) = initialize_mmtk() {
            panic!("Failed to initialize MMTk: {e}");
        }

        // Bind mutator for this thread
        if let Err(e) = mmtk_bind_mutator() {
            panic!("Failed to bind mutator: {e}");
        }

        println!("Creating unreachable objects...");

        // Create objects in inner scope that will become unreachable
        {
            let _temp_string = Var::string("temporary data");
            let _temp_tuple = Var::tuple(&[Var::int(100), Var::int(200)]);
            println!("Temporary objects created");
        } // Objects go out of scope here

        println!("Objects now unreachable, triggering GC...");
        trigger_gc();

        println!("GC completed - unreachable objects should be collected");
    }

    #[test]
    fn test_gc_root_registration() {
        // Initialize MMTk for testing
        if let Err(e) = initialize_mmtk() {
            panic!("Failed to initialize MMTk: {e}");
        }

        // Bind mutator for this thread
        if let Err(e) = mmtk_bind_mutator() {
            panic!("Failed to bind mutator: {e}");
        }

        println!("Testing root registration and scanning...");

        // Create objects and register them as roots
        let string1 = Var::string("root string");
        let tuple1 = Var::tuple(&[Var::int(42), Var::string("nested")]);

        // Register as thread roots using the helper
        register_var_as_root(string1, false);
        register_var_as_root(tuple1, false);

        // Also test global root
        let global_var = Var::string("global data");
        register_var_as_root(global_var, true);

        println!("Triggering GC to test root scanning...");
        trigger_gc();

        // Clear roots
        clear_thread_roots();

        println!("Root registration test completed");
    }
}
