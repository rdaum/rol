//! Central runtime scheduler - the "top dog" of the system.
//! Manages global namespace and task execution coordination.

use crate::bytecode::BytecodeJIT;
use crate::gc::{clear_current_task_context, request_task_kill, set_current_task_context};
use crate::heap::{LispTask, TaskResult, TaskState};
use crate::symbol::Symbol;
use crate::var::Var;
use once_cell::sync::OnceCell;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock, atomic::AtomicU64};
use std::thread::JoinHandle;

/// Result of executing a function that can be suspended/resumed
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionResult {
    /// Normal completion with result value
    Completed(Var),
    /// Task yielded at safepoint - can resume by calling again
    Yielded,
    /// Task was killed at safepoint
    Killed,
    /// Runtime error occurred
    Error(String),
}

/// Task execution handle combining JoinHandle with task metadata
pub struct TaskHandle {
    pub task_var: Var,
    pub join_handle: JoinHandle<TaskResult>,
}

/// Central runtime scheduler that manages all shared state and task execution.
/// This is the "top dog" of the system - everything flows through here.
pub struct Scheduler {
    /// Global namespace shared by REPL and all tasks using copy-on-write semantics
    globals: RwLock<im::HashMap<Symbol, Var>>,

    /// Version counter for COW optimization - incremented on every global change
    global_version: AtomicU64,

    /// Running tasks mapped by task ID to task handles
    tasks: RwLock<HashMap<u64, TaskHandle>>,
    next_task_id: AtomicU64,

    /// Thread pool for task execution (thread-per-task model)
    thread_pool: ThreadPool,
}

impl Scheduler {
    /// Create a new scheduler instance
    fn new() -> Self {
        // Create a thread pool with reasonable defaults
        // Each task gets its own OS thread for true isolation
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|index| format!("rol-task-{}", index))
            .build()
            .expect("Failed to create task thread pool");

        Self {
            globals: RwLock::new(im::HashMap::new()),
            global_version: AtomicU64::new(0),
            tasks: RwLock::new(HashMap::new()),
            next_task_id: AtomicU64::new(1),
            thread_pool,
        }
    }

    /// Get a reference to the global scheduler instance
    pub fn instance() -> &'static Scheduler {
        static SCHEDULER: OnceCell<Scheduler> = OnceCell::new();
        SCHEDULER.get_or_init(|| Scheduler::new())
    }

    /// Set a global variable (used by REPL, builtin functions, etc.)
    pub fn set_global(symbol: Symbol, value: Var) {
        let scheduler = Self::instance();

        // Update the global namespace
        let mut globals = scheduler.globals.write().unwrap();
        *globals = globals.update(symbol, value);

        // Increment version for COW tracking
        scheduler.global_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Get a global variable value
    pub fn get_global(symbol: Symbol) -> Option<Var> {
        let scheduler = Self::instance();
        scheduler.globals.read().unwrap().get(&symbol).cloned()
    }

    /// Get a copy-on-write snapshot of all globals (used when spawning tasks)
    pub fn get_global_snapshot() -> (im::HashMap<Symbol, Var>, u64) {
        let scheduler = Self::instance();
        let globals = scheduler.globals.read().unwrap().clone(); // O(1) COW clone
        let version = scheduler.global_version.load(Ordering::SeqCst);
        (globals, version)
    }

    /// Get current global version (for checking if globals have changed)
    pub fn get_global_version() -> u64 {
        Self::instance().global_version.load(Ordering::SeqCst)
    }

    /// Get all globals as a regular HashMap (for compatibility with existing BytecodeJIT)
    /// This creates a snapshot for compilation - changes won't be reflected
    pub fn get_globals_for_compilation() -> HashMap<Symbol, Var> {
        let scheduler = Self::instance();
        let im_globals = scheduler.globals.read().unwrap();

        // Convert im::HashMap to std::HashMap for existing compilation code
        let mut std_globals = HashMap::new();
        for (symbol, var) in im_globals.iter() {
            std_globals.insert(*symbol, *var);
        }
        std_globals
    }

    /// Initialize globals from existing std::HashMap (for migration from BytecodeJIT)
    pub fn initialize_globals_from_std(std_globals: HashMap<Symbol, Var>) {
        let scheduler = Self::instance();
        let mut globals = scheduler.globals.write().unwrap();

        // Convert and insert all globals
        for (symbol, var) in std_globals {
            *globals = globals.update(symbol, var);
        }

        // Increment version
        scheduler.global_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Clear all globals (for testing)
    #[cfg(test)]
    pub fn clear_globals() {
        let scheduler = Self::instance();
        *scheduler.globals.write().unwrap() = im::HashMap::new();
        scheduler.global_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Clear all tasks (for testing) - forces cleanup of any remaining tasks
    #[cfg(test)]
    pub fn clear_tasks() {
        let scheduler = Self::instance();
        scheduler.tasks.write().unwrap().clear();
    }

    /// Spawn a new task with the given closure
    /// Returns a Var containing the Task handle for joining/killing
    pub fn spawn_task(closure: Var) -> Result<Var, String> {
        let scheduler = Self::instance();

        // Generate unique task ID
        let task_id = scheduler.next_task_id.fetch_add(1, Ordering::SeqCst);

        // Create copy-on-write snapshot of current globals
        let (globals_snapshot, _version) = Self::get_global_snapshot();

        // Create task heap object
        let task_ptr = LispTask::new(task_id, closure, globals_snapshot);
        let task_var = Var::task(task_ptr);

        // Clone task_var for the thread (since we need to move it)
        let task_var_for_thread = task_var;

        // Spawn task on thread pool
        let join_handle =
            std::thread::spawn(move || Self::execute_task_isolated(task_var_for_thread));

        // Store task handle in scheduler
        let handle = TaskHandle {
            task_var,
            join_handle,
        };

        scheduler.tasks.write().unwrap().insert(task_id, handle);

        Ok(task_var)
    }

    /// Execute a task in isolation with its own JIT instance
    fn execute_task_isolated(task_var: Var) -> TaskResult {
        let task_ptr = match task_var.as_task() {
            Some(ptr) => ptr,
            None => return TaskResult::Error("Invalid task handle".to_string()),
        };

        unsafe {
            let task = &mut *task_ptr;
            let task_id = task.task_id;

            // Set task context for this thread
            set_current_task_context(task_id);

            // Set task state to Running
            task.set_state(TaskState::Running);

            // Create isolated JIT instance for this task
            let mut jit = BytecodeJIT::new();

            // TODO: Extract closure and execute it using resumable execution
            // For now, just return a placeholder result
            let result = Var::int(42);

            // Clear task context when done
            clear_current_task_context();

            task.complete_with_result(result);
            TaskResult::Completed(result)
        }
    }

    /// Join a task and wait for its completion
    pub fn join_task(task_var: Var) -> Result<Var, String> {
        let task_ptr = task_var
            .as_task()
            .ok_or_else(|| "Invalid task handle".to_string())?;

        let task_id = unsafe { (*task_ptr).task_id };

        let scheduler = Self::instance();

        // Remove task from active tasks and get its handle
        let handle = scheduler
            .tasks
            .write()
            .unwrap()
            .remove(&task_id)
            .ok_or_else(|| "Task not found or already completed".to_string())?;

        // Wait for task completion
        match handle.join_handle.join() {
            Ok(TaskResult::Completed(result)) => Ok(result),
            Ok(TaskResult::Error(msg)) => Err(msg),
            Ok(TaskResult::Killed) => Err("Task was killed".to_string()),
            Err(_) => Err("Task panicked".to_string()),
        }
    }

    /// Kill a running task
    pub fn kill_task(task_var: Var) -> Result<(), String> {
        let task_ptr = task_var
            .as_task()
            .ok_or_else(|| "Invalid task handle".to_string())?;

        unsafe {
            let task = &mut *task_ptr;
            task.kill();
        }

        // Request that the task be killed at its next safepoint
        // The task will check this flag when it hits a safepoint in JIT code
        request_task_kill();

        Ok(())
    }

    /// Get the number of active tasks
    pub fn active_task_count() -> usize {
        let scheduler = Self::instance();
        scheduler.tasks.read().unwrap().len()
    }

    /// Clean up completed tasks (remove finished tasks from the registry)
    pub fn cleanup_completed_tasks() {
        let scheduler = Self::instance();
        let mut tasks = scheduler.tasks.write().unwrap();

        // Remove tasks that are finished (their join handles would return immediately)
        tasks.retain(|_task_id, handle| {
            // Check if task is still running by examining the task state
            if let Some(task_ptr) = handle.task_var.as_task() {
                unsafe {
                    let task = &*task_ptr;
                    !task.is_terminal()
                }
            } else {
                false // Invalid task, remove it
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_scheduler_singleton() {
        let s1 = Scheduler::instance();
        let s2 = Scheduler::instance();
        assert!(std::ptr::eq(s1, s2));
    }

    #[test]
    fn test_global_operations() {
        Scheduler::clear_globals();
        Scheduler::clear_tasks();

        let test_symbol = Symbol::from("test");
        let test_value = Var::int(42);

        // Test set/get
        Scheduler::set_global(test_symbol, test_value);
        assert_eq!(Scheduler::get_global(test_symbol), Some(test_value));

        // Test version increment
        let version1 = Scheduler::get_global_version();
        Scheduler::set_global(test_symbol, Var::int(99));
        let version2 = Scheduler::get_global_version();
        assert!(version2 > version1);
    }

    #[test]
    fn test_global_snapshot() {
        use crate::gc::ensure_mmtk_initialized_for_tests;

        // Initialize MMTk for testing
        ensure_mmtk_initialized_for_tests();

        Scheduler::clear_globals();
        Scheduler::clear_tasks();

        let sym1 = Symbol::from("var1");
        let sym2 = Symbol::from("var2");

        Scheduler::set_global(sym1, Var::int(10));
        Scheduler::set_global(sym2, Var::string("hello"));

        let (snapshot, version) = Scheduler::get_global_snapshot();

        // Snapshot should contain our values
        assert_eq!(snapshot.get(&sym1), Some(&Var::int(10)));
        assert_eq!(snapshot.get(&sym2), Some(&Var::string("hello")));
        assert_eq!(version, Scheduler::get_global_version());

        // Modifying original shouldn't affect snapshot (COW)
        Scheduler::set_global(sym1, Var::int(999));
        assert_eq!(snapshot.get(&sym1), Some(&Var::int(10))); // Unchanged
    }

    #[test]
    fn test_std_hashmap_compatibility() {
        Scheduler::clear_globals();
        Scheduler::clear_tasks();

        let mut std_globals = HashMap::new();
        std_globals.insert(Symbol::from("x"), Var::int(1));
        std_globals.insert(Symbol::from("y"), Var::int(2));

        // Initialize from std::HashMap
        Scheduler::initialize_globals_from_std(std_globals);

        // Convert back to std::HashMap
        let converted = Scheduler::get_globals_for_compilation();

        assert_eq!(converted.get(&Symbol::from("x")), Some(&Var::int(1)));
        assert_eq!(converted.get(&Symbol::from("y")), Some(&Var::int(2)));
    }

    #[test]
    fn test_task_management() {
        use crate::gc::ensure_mmtk_initialized_for_tests;

        // Initialize MMTk for testing
        ensure_mmtk_initialized_for_tests();

        Scheduler::clear_globals();
        Scheduler::clear_tasks();

        // Test initial state
        assert_eq!(Scheduler::active_task_count(), 0);

        // Spawn a task with a dummy closure
        let closure = Var::none(); // Placeholder closure
        let task_result = Scheduler::spawn_task(closure);
        assert!(task_result.is_ok());

        let task_var = task_result.unwrap();
        assert!(task_var.is_task());

        // Check that we now have an active task
        assert_eq!(Scheduler::active_task_count(), 1);

        // Join the task (should complete immediately with our placeholder implementation)
        let join_result = Scheduler::join_task(task_var);
        assert!(join_result.is_ok());

        let result = join_result.unwrap();
        assert_eq!(result, Var::int(42)); // Our placeholder result

        // Task should be cleaned up after joining
        assert_eq!(Scheduler::active_task_count(), 0);
    }

    #[test]
    fn test_task_kill() {
        use crate::gc::ensure_mmtk_initialized_for_tests;

        // Initialize MMTk for testing
        ensure_mmtk_initialized_for_tests();

        Scheduler::clear_globals();
        Scheduler::clear_tasks();

        // Spawn a task
        let closure = Var::none();
        let task_result = Scheduler::spawn_task(closure);
        assert!(task_result.is_ok());

        let task_var = task_result.unwrap();

        // Kill the task
        let kill_result = Scheduler::kill_task(task_var);
        assert!(kill_result.is_ok());

        // Verify task is marked as killed
        if let Some(task_ptr) = task_var.as_task() {
            unsafe {
                let task = &*task_ptr;
                assert_eq!(task.get_state(), TaskState::Killed);
            }
        } else {
            panic!("Failed to extract task pointer");
        }
    }

    #[test]
    fn test_task_global_isolation() {
        use crate::gc::ensure_mmtk_initialized_for_tests;

        // Initialize MMTk for testing
        ensure_mmtk_initialized_for_tests();

        Scheduler::clear_globals();
        Scheduler::clear_tasks();

        // Set some globals
        let test_symbol = Symbol::from("test_var");
        Scheduler::set_global(test_symbol, Var::int(100));

        // Spawn a task (should get snapshot of current globals)
        let closure = Var::none();
        let task_result = Scheduler::spawn_task(closure);
        assert!(task_result.is_ok());

        let task_var = task_result.unwrap();

        // Verify task has isolated global snapshot
        if let Some(task_ptr) = task_var.as_task() {
            unsafe {
                let task = &*task_ptr;
                assert_eq!(task.get_global(test_symbol), Some(Var::int(100)));
                assert_eq!(task.globals_count(), 1);
            }
        } else {
            panic!("Failed to extract task pointer");
        }

        // Modify globals after task creation
        Scheduler::set_global(test_symbol, Var::int(200));

        // Task should still have its isolated snapshot
        if let Some(task_ptr) = task_var.as_task() {
            unsafe {
                let task = &*task_ptr;
                assert_eq!(task.get_global(test_symbol), Some(Var::int(100))); // Still old value
            }
        }

        // Clean up
        let _ = Scheduler::join_task(task_var);
    }

    #[test]
    fn test_execution_result_enum() {
        use crate::bytecode::BytecodeJIT;
        use crate::gc::ensure_mmtk_initialized_for_tests;

        // Initialize MMTk for testing
        ensure_mmtk_initialized_for_tests();

        Scheduler::clear_globals();
        Scheduler::clear_tasks();

        // Test that ExecutionResult enum works correctly
        let mut jit = BytecodeJIT::new();

        // For now, just test that the new API compiles and works
        // In the future, this would test actual yielding/killing behavior
        // when we have real closures and safepoints

        // Test ExecutionResult::Completed variant
        let completed = ExecutionResult::Completed(Var::int(42));
        assert_eq!(completed, ExecutionResult::Completed(Var::int(42)));

        // Test ExecutionResult::Yielded variant
        let yielded = ExecutionResult::Yielded;
        assert_eq!(yielded, ExecutionResult::Yielded);

        // Test ExecutionResult::Killed variant
        let killed = ExecutionResult::Killed;
        assert_eq!(killed, ExecutionResult::Killed);

        // Test ExecutionResult::Error variant
        let error = ExecutionResult::Error("test error".to_string());
        assert_eq!(error, ExecutionResult::Error("test error".to_string()));
    }
}
