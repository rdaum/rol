//! Central runtime scheduler - the "top dog" of the system.
//! Manages global namespace and task execution coordination.

use crate::symbol::Symbol;
use crate::var::Var;
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{RwLock, atomic::AtomicU64};

/// Central runtime scheduler that manages all shared state and task execution.
/// This is the "top dog" of the system - everything flows through here.
pub struct Scheduler {
    /// Global namespace shared by REPL and all tasks using copy-on-write semantics
    globals: RwLock<im::HashMap<Symbol, Var>>,

    /// Version counter for COW optimization - incremented on every global change
    global_version: AtomicU64,

    /// Running tasks (will be added when we implement task management)
    tasks: RwLock<HashMap<u64, ()>>, // Placeholder for now
    next_task_id: AtomicU64,
}

impl Scheduler {
    /// Create a new scheduler instance
    fn new() -> Self {
        Self {
            globals: RwLock::new(im::HashMap::new()),
            global_version: AtomicU64::new(0),
            tasks: RwLock::new(HashMap::new()),
            next_task_id: AtomicU64::new(1),
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
        Scheduler::clear_globals();

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
}
