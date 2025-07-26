//! Task heap object for cooperative multithreading.
//! Contains task metadata, execution state, and isolated global namespace.

use crate::gc::{
    is_mmtk_initialized, mmtk_alloc, mmtk_alloc_placeholder, mmtk_dealloc_placeholder,
};
use crate::symbol::Symbol;
use crate::var::Var;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};

/// Task execution states for cooperative scheduling
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    Ready = 0,     // Task is ready to run
    Running = 1,   // Task is currently executing
    Suspended = 2, // Task is suspended (waiting for resume)
    Yielded = 3,   // Task yielded voluntarily
    Completed = 4, // Task finished successfully
    Killed = 5,    // Task was terminated
    Error = 6,     // Task encountered an error
}

impl From<u8> for TaskState {
    fn from(value: u8) -> Self {
        match value {
            0 => TaskState::Ready,
            1 => TaskState::Running,
            2 => TaskState::Suspended,
            3 => TaskState::Yielded,
            4 => TaskState::Completed,
            5 => TaskState::Killed,
            6 => TaskState::Error,
            _ => TaskState::Error, // Default to error for invalid values
        }
    }
}

/// Result of task execution
#[derive(Debug, Clone)]
pub enum TaskResult {
    Completed(Var), // Task completed successfully with result
    Error(String),  // Task encountered an error
    Killed,         // Task was terminated
}

/// Task heap object for cooperative multithreading.
/// Layout: [task_id: u64][state: AtomicU8][closure: Var][globals_snapshot: HashMap][result: Option<Var>]
/// Contains all state needed for task-local execution with copy semantics.
#[repr(C)]
pub struct LispTask {
    /// Unique task identifier
    pub task_id: u64,

    /// Current task state (atomic for thread-safe access)
    pub state: AtomicU8,

    /// Closure to execute (copied, not shared)
    pub closure: Var,

    /// Copy-on-write snapshot of globals at task creation time
    /// Using std::HashMap for now, may optimize later
    pub globals_snapshot: HashMap<Symbol, Var>,

    /// Task execution result (set when task completes)
    pub result: Option<TaskResult>,
}

impl LispTask {
    /// Create a new task with the given closure and global snapshot
    pub fn new(
        task_id: u64,
        closure: Var,
        globals_snapshot: HashMap<Symbol, Var>,
    ) -> *mut LispTask {
        let size = std::mem::size_of::<LispTask>();
        // Round up to 8-byte alignment as required by MMTk
        let aligned_size = (size + 7) & !7;

        let ptr = if is_mmtk_initialized() {
            mmtk_alloc(aligned_size) as *mut LispTask
        } else {
            mmtk_alloc_placeholder(aligned_size) as *mut LispTask
        };

        if ptr.is_null() {
            panic!("Failed to allocate memory for LispTask");
        }

        unsafe {
            std::ptr::write(
                ptr,
                LispTask {
                    task_id,
                    state: AtomicU8::new(TaskState::Ready as u8),
                    closure,
                    globals_snapshot,
                    result: None,
                },
            );
        }

        ptr
    }

    /// Get the current task state
    pub fn get_state(&self) -> TaskState {
        TaskState::from(self.state.load(Ordering::SeqCst))
    }

    /// Set the task state
    pub fn set_state(&self, new_state: TaskState) {
        self.state.store(new_state as u8, Ordering::SeqCst);
    }

    /// Check if task is in a terminal state (completed, killed, or error)
    pub fn is_terminal(&self) -> bool {
        match self.get_state() {
            TaskState::Completed | TaskState::Killed | TaskState::Error => true,
            _ => false,
        }
    }

    /// Check if task can be scheduled (ready or yielded)
    pub fn is_schedulable(&self) -> bool {
        match self.get_state() {
            TaskState::Ready | TaskState::Yielded => true,
            _ => false,
        }
    }

    /// Mark task as completed with result
    pub fn complete_with_result(&mut self, result: Var) {
        self.result = Some(TaskResult::Completed(result));
        self.set_state(TaskState::Completed);
    }

    /// Mark task as completed with error
    pub fn complete_with_error(&mut self, error: String) {
        self.result = Some(TaskResult::Error(error));
        self.set_state(TaskState::Error);
    }

    /// Mark task as killed
    pub fn kill(&mut self) {
        self.result = Some(TaskResult::Killed);
        self.set_state(TaskState::Killed);
    }

    /// Get task result (if completed)
    pub fn get_result(&self) -> Option<&TaskResult> {
        self.result.as_ref()
    }

    /// Get a global variable from the task's snapshot
    pub fn get_global(&self, symbol: Symbol) -> Option<Var> {
        self.globals_snapshot.get(&symbol).copied()
    }

    /// Check if a global variable exists in the task's snapshot
    pub fn has_global(&self, symbol: Symbol) -> bool {
        self.globals_snapshot.contains_key(&symbol)
    }

    /// Get the number of globals in the task's snapshot
    pub fn globals_count(&self) -> usize {
        self.globals_snapshot.len()
    }
}

impl Drop for LispTask {
    fn drop(&mut self) {
        // Task cleanup when dropped
        if !self.is_terminal() {
            self.kill();
        }
    }
}

// SAFETY: Tasks are designed for multi-threaded execution with atomic state management
unsafe impl Send for LispTask {}
unsafe impl Sync for LispTask {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::ensure_mmtk_initialized_for_tests;
    use crate::symbol::Symbol;

    fn setup_test() {
        ensure_mmtk_initialized_for_tests();
    }

    #[test]
    fn test_task_creation() {
        setup_test();

        let closure = Var::none(); // Placeholder closure
        let globals = HashMap::new();

        let task_ptr = LispTask::new(42, closure, globals);

        unsafe {
            let task = &*task_ptr;
            assert_eq!(task.task_id, 42);
            assert_eq!(task.get_state(), TaskState::Ready);
            assert_eq!(task.closure, Var::none());
            assert_eq!(task.globals_count(), 0);
            assert!(task.get_result().is_none());
        }
    }

    #[test]
    fn test_task_state_transitions() {
        setup_test();

        let closure = Var::none();
        let globals = HashMap::new();
        let task_ptr = LispTask::new(1, closure, globals);

        unsafe {
            let task = &*task_ptr;

            // Initial state should be Ready
            assert_eq!(task.get_state(), TaskState::Ready);
            assert!(task.is_schedulable());
            assert!(!task.is_terminal());

            // Transition to Running
            task.set_state(TaskState::Running);
            assert_eq!(task.get_state(), TaskState::Running);
            assert!(!task.is_schedulable());
            assert!(!task.is_terminal());

            // Transition to Completed
            task.set_state(TaskState::Completed);
            assert_eq!(task.get_state(), TaskState::Completed);
            assert!(!task.is_schedulable());
            assert!(task.is_terminal());
        }
    }

    #[test]
    fn test_task_completion() {
        setup_test();

        let closure = Var::none();
        let globals = HashMap::new();
        let task_ptr = LispTask::new(1, closure, globals);

        unsafe {
            let task = &mut *task_ptr;

            // Complete with result
            let result_var = Var::int(42);
            task.complete_with_result(result_var);

            assert_eq!(task.get_state(), TaskState::Completed);
            assert!(task.is_terminal());

            if let Some(TaskResult::Completed(result)) = task.get_result() {
                assert_eq!(*result, Var::int(42));
            } else {
                panic!("Expected completed result");
            }
        }
    }

    #[test]
    fn test_task_globals_snapshot() {
        setup_test();

        let closure = Var::none();
        let mut globals = HashMap::new();

        let sym1 = Symbol::from("x");
        let sym2 = Symbol::from("y");
        globals.insert(sym1, Var::int(10));
        globals.insert(sym2, Var::int(20));

        let task_ptr = LispTask::new(1, closure, globals);

        unsafe {
            let task = &*task_ptr;

            assert_eq!(task.globals_count(), 2);
            assert_eq!(task.get_global(sym1), Some(Var::int(10)));
            assert_eq!(task.get_global(sym2), Some(Var::int(20)));
            assert!(task.has_global(sym1));
            assert!(!task.has_global(Symbol::from("z")));
        }
    }

    #[test]
    fn test_task_error_handling() {
        setup_test();

        let closure = Var::none();
        let globals = HashMap::new();
        let task_ptr = LispTask::new(1, closure, globals);

        unsafe {
            let task = &mut *task_ptr;

            // Complete with error
            task.complete_with_error("Division by zero".to_string());

            assert_eq!(task.get_state(), TaskState::Error);
            assert!(task.is_terminal());

            if let Some(TaskResult::Error(msg)) = task.get_result() {
                assert_eq!(msg, "Division by zero");
            } else {
                panic!("Expected error result");
            }
        }
    }

    #[test]
    fn test_task_kill() {
        setup_test();

        let closure = Var::none();
        let globals = HashMap::new();
        let task_ptr = LispTask::new(1, closure, globals);

        unsafe {
            let task = &mut *task_ptr;

            task.kill();

            assert_eq!(task.get_state(), TaskState::Killed);
            assert!(task.is_terminal());

            if let Some(TaskResult::Killed) = task.get_result() {
                // Expected
            } else {
                panic!("Expected killed result");
            }
        }
    }
}
