# Multithreaded Cooperative Scheduler Design

## Current Architecture Analysis

### Safepoint System
- **Implementation**: `jit_safepoint_check()` in `src/gc/mmtk_binding.rs:951`
- **Mechanism**: Called from JIT-generated code at strategic points
- **Coordination**: Uses `GC_SAFEPOINT_REQUESTED` atomic flag and condition variables
- **Integration**: Already wired into bytecode compilation in `src/bytecode.rs` with `safepoint_ref` function references

### GC Integration
- **System**: MMTk-based garbage collection with concurrent collection support
- **Root Management**: Global and thread-local root registration system
- **Current State**: GC can request safepoints to coordinate collection cycles

### Current Execution Model
- **Function**: `execute_function()` in `src/bytecode.rs:1115`
- **Type**: Synchronous execution - transmutes function pointer to native function and executes directly
- **Context**: Takes `*mut BytecodeJIT` as context parameter
- **Result**: Returns `Var` result immediately

### Var System
- **Design**: NaN-boxed 64-bit values with type tagging
- **Heap Objects**: Supports pointers with type-specific tags (TUPLE_POINTER_TAG, STRING_POINTER_TAG, etc.)
- **Current Types**: None, I32, F64, Bool, Pointer, Symbol, Tuple, String, Environment, Closure

## Design Questions for Multithreaded Cooperative Scheduler

### 1. Task Var Type Implementation
**Question**: Should the new Task be implemented as:
- [ ] A. New `VarType::Task` variant in the enum
- [x] B. Extend existing pointer system with `TASK_POINTER_TAG`
  - Like Closure or Tuple etc
- [ ] C. Other approach?

**Context**: Need to make tasks heap-manageable objects that participate in GC.

### 2. Task State Management
**Question**: What task states should we support and how should they be checked?
- [x] Suggested states: Ready, Running, Suspended, Killed, Yielded
  - We may want a state for "blocked while collecting" purely for debugging  
- [x] Should these be per-task atomic flags checked at safepoints?
- [ ] Other state model?

**Context**: Since there's no bytecode loop, we need flags that JIT code can check at safepoints.

### 3. Scheduler Architecture
**Question**: How should the Scheduler be integrated?
- [ ] A. Singleton global instance
- [ ] B. Part of the `BytecodeJIT` context
  - You will need a new bytecode jit per task
- [x] C. Separate component that owns/manages JIT instances
- [ ] D. Other architecture?

**Context**: Scheduler needs to be a GC root source and manage multiple tasks.

### 4. Task Context Structure
**Question**: What should each Task object contain?
- [x] Execution state (program counter equivalent?)
- [x] Local variable environment
- [x] Stack state
- [x] Task ID/metadata
- [x] What else?
  - There will need to be some way to pass globals to it and have them returned somehow. This was 
    a sticky point in the previous design. Global functions and variables, and the resolution of them!

**Context**: Tasks need enough context to be suspended and resumed.

### 5. Safepoint Extensions
**Question**: Beyond GC coordination, what should safepoints check?
- [x] Task yield requests
- [x] Task kill signals
- [x] Task suspend/resume requests
- [ ] Scheduler time slice management
  - This can come later
- [x] Other coordination needs?
  - Right now we have a global safepoint for GC collection for all the system, but we will actually
    want to add a per-task/thread flag eventually as well, and take advantage of MMTK's ability to
    iterate mutators and pause only one at a time.

**Context**: Safepoints are our main coordination mechanism in JIT code.

### 6. Execute Function Evolution
**Question**: How should execution change to support tasks?
- [x] A. Make `execute_function` async/resumable
  - when a flag is noticed during safe point, execute function should return its state as an enum, and then
    the scheduler could resume it later
- [ ] B. Add new `execute_task` method alongside existing
  - no
- [ ] C. Replace with task-oriented execution model
- [ ] D. Other approach?

**Context**: Need to support interruption and resumption of execution.

### 7. Task Creation and Management
**Question**: How should tasks be created and managed?
- [x] Who creates tasks? (Scheduler, user code, system?)
  - we'll have a new builtin which takes a closure.
- [x] How are tasks scheduled? (Round-robin, priority-based, etc.)
  - CPU thread per task, runs on a separate thread, no need to slice it
- [ ] Task lifetime management?

### 8. Concurrency Model
**Question**: What's the target concurrency model?
- [ ] Single-threaded cooperative (like JavaScript event loop)
- [  Multi-threaded with work stealing
- [ ] Hybrid approach
- [x] Other model?
  - thread per task. can use rayon to pool threads

**Context**: README mentions "no support for threads or any concurrency at all" currently.

## Detailed Design Specification

### Task Heap Object Structure

```rust
// In src/heap.rs  
pub struct LispTask {
    pub task_id: u64,
    pub state: AtomicU8, // TaskState as u8
    pub closure: LispClosure, // COPIED closure to execute (not pointer)
    pub globals_snapshot: im::HashMap<Symbol, Var>, // COW globals at task creation
    pub result: Option<Var>, // Result when task completes (copied back)
    pub join_handle: Option<JoinHandle<TaskResult>>, // Rayon thread handle
}

pub enum TaskResult {
    Completed(Var), // Copied result value
    Error(String),  // Error message
    Killed,
}

// In src/var.rs
pub const TASK_POINTER_TAG: u64 = 0x6000000000000000;

impl Var {
    pub fn task(task_ptr: *mut LispTask) -> Self {
        // Similar to closure(), string(), etc.
    }
    
    pub fn as_task(&self) -> Option<*mut LispTask> {
        // Extract task pointer
    }
    
    pub fn is_task(&self) -> bool {
        // Check if this Var is a task
    }
}
```

### Task State Management

```rust
// In src/scheduler/mod.rs (new module)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskState {
    Ready = 0,
    Running = 1,
    Suspended = 2,
    Killed = 3,
    Yielded = 4,
    BlockedOnGC = 5, // For debugging
    Completed = 6,
}

// Per-task atomic flags checked at safepoints
pub struct TaskFlags {
    pub yield_requested: AtomicBool,
    pub kill_requested: AtomicBool,
    pub suspend_requested: AtomicBool,
}
```

### Scheduler Architecture

```rust
// In src/scheduler/scheduler.rs
pub struct Scheduler {
    tasks: RwLock<HashMap<u64, Arc<LispTask>>>,
    next_task_id: AtomicU64,
    thread_pool: ThreadPool, // Rayon thread pool
    global_snapshot: RwLock<im::HashMap<Symbol, Var>>, // COW global state
    global_version: AtomicU64, // Version counter for COW
}

// Global singleton instance
pub static SCHEDULER: OnceCell<Scheduler> = OnceCell::new();

impl Scheduler {
    pub fn spawn_task(&self, closure: Var) -> Result<Var, String> {
        // 1. Copy closure and current global snapshot
        // 2. Create task with copied data
        // 3. Spawn on Rayon thread with isolated heap
        // 4. Return task Var handle
    }
    
    pub fn join_task(&self, task: Var) -> Result<Var, String> {
        // Wait for task completion and copy result back to current heap
    }
    
    pub fn kill_task(&self, task: Var) -> Result<(), String> {
        // Set kill flag for task (checked at safepoints)
    }
    
    pub fn update_globals(&self, new_globals: im::HashMap<Symbol, Var>) {
        // Update global snapshot with COW semantics
        // Increment version for running tasks to detect changes
    }
}

// Note: No GC root source needed - tasks are isolated with their own heaps!
```

### Execute Function Evolution

```rust
// In src/bytecode.rs
pub enum ExecutionResult {
    Completed(Var), // Normal completion
    Yielded,        // Yielded at safepoint - can resume by calling again
    Killed,         // Task was killed
    Error(String),  // Runtime error
}

// No complex state capture needed - yielding only at safepoints means
// we can resume by calling the same function again with fresh stack

impl BytecodeJIT {
    pub fn execute_function_resumable(&mut self, func_ptr: *const u8) -> ExecutionResult {
        // Each task gets its own JIT instance with isolated heap
        // Safepoints check task flags and return Yielded/Killed as needed
        // No stack/register state to preserve - fresh call stack each resume
    }
}
```

### Extended Safepoint System

```rust
// In src/gc/mmtk_binding.rs  
// Global safepoint (existing)
pub static GC_SAFEPOINT_REQUESTED: AtomicBool = AtomicBool::new(false);

// Per-task coordination - much simpler with isolated heaps
thread_local! {
    static CURRENT_TASK_ID: Cell<Option<u64>> = Cell::new(None);
    static TASK_KILL_REQUESTED: Cell<bool> = Cell::new(false);
    static TASK_YIELD_REQUESTED: Cell<bool> = Cell::new(false);
}

pub extern "C" fn jit_safepoint_check() -> u32 {
    // Check global GC safepoint (existing logic) 
    if GC_SAFEPOINT_REQUESTED.load(Ordering::Acquire) {
        // Each task has isolated heap, so GC per-task
        // ... existing GC coordination for this task's heap
    }
    
    // Check task control flags
    let should_yield = TASK_YIELD_REQUESTED.with(|f| f.get());
    let should_kill = TASK_KILL_REQUESTED.with(|f| f.get());
    
    if should_kill {
        return 2; // KILLED signal
    }
    if should_yield {
        TASK_YIELD_REQUESTED.with(|f| f.set(false)); // Clear flag
        return 1; // YIELDED signal  
    }
    
    0 // CONTINUE
}

// Task execution wrapper
pub fn set_current_task_context(task_id: u64) {
    CURRENT_TASK_ID.with(|id| id.set(Some(task_id)));
}

pub fn request_task_kill() {
    TASK_KILL_REQUESTED.with(|f| f.set(true));
}
```

### Task Creation Builtin

```rust
// In src/builtins/ (new module)
pub fn builtin_spawn_task(args: &[Var]) -> Result<Var, String> {
    // Takes a closure, creates task, returns task Var
    // (spawn-task (lambda () (+ 1 2)))
}

pub fn builtin_join_task(args: &[Var]) -> Result<Var, String> {
    // Takes a task, waits for completion, returns result
    // (join-task my-task)
}

pub fn builtin_kill_task(args: &[Var]) -> Result<Var, String> {
    // Takes a task, signals kill
    // (kill-task my-task)
}
```

## Critical Discovery: Global Namespace Challenge

### Current Global System Analysis

The current global management system reveals why this is a "sticky point":

```rust
// In BytecodeJIT::compile_function() - globals are compiled as CONSTANTS
for (symbol, var) in &self.global_variables {
    let const_value = builder.ins().iconst(types::I64, var.as_u64() as i64);
    analyzer.variables.insert(*symbol, const_value);
}
```

**Key Issues:**
1. **Compiled Constants**: Globals are baked into machine code as `iconst` values
2. **Per-JIT Isolation**: Each `BytecodeJIT` has its own `global_variables: HashMap<Symbol, Var>`  
3. **Two-tier System**: Fast offsets for immutable globals, HashMap lookup for mutable ones
4. **GC Write Barriers**: Complex write barrier handling for global updates
5. **No Synchronization**: No mechanism to share global changes between JIT instances

**Implications for Multi-Task System:**
- Each task's JIT will compile different constant values for globals
- Global updates in one task are invisible to other tasks
- Function definitions in one task won't be available to others
- REPL-defined globals won't propagate to spawned tasks

### Potential Solutions Architecture

#### Option 1: Runtime Global Resolution
Replace compiled constants with runtime function calls:

```rust
// Instead of: iconst(global_value)
// Generate: call(get_global_runtime, symbol_id)

pub extern "C" fn get_global_runtime(scheduler: *mut Scheduler, symbol: u64) -> u64 {
    // Thread-safe global lookup with proper locking
}
```

**Pros**: True shared state, updates visible immediately
**Cons**: Performance cost on every global access, complex synchronization

#### Option 2: Copy-on-Write Global Snapshots  
Tasks get immutable snapshots, updates create new versions:

```rust
pub struct GlobalSnapshot {
    version: u64,
    globals: Arc<HashMap<Symbol, Var>>,
}

// Tasks check version at safepoints, get new snapshot if changed
```

**Pros**: Fast reads, eventual consistency, simpler than locking
**Cons**: Memory overhead, delayed visibility, complex versioning

#### Option 3: Hybrid Function vs Variable Semantics
Different handling for functions (immutable) vs variables (mutable):

```rust
// Functions: Compiled as constants (fast), shared via code cache
// Variables: Runtime lookups (slower), synchronized access
```

**Pros**: Optimizes common case (functions), manageable complexity  
**Cons**: Inconsistent semantics, still need synchronization for variables

#### Option 4: Task-Local Globals with Explicit Sharing
Each task has isolated globals, explicit mechanisms for sharing:

```rust
// (set-global! 'x 42)      ; Local to task
// (share-global! 'x)       ; Make visible to other tasks  
// (import-global! 'x)      ; Import from shared namespace
```

**Pros**: Clear semantics, no accidental sharing, good isolation
**Cons**: Different programming model, requires explicit coordination

#### Option 5: Message-Passing Global Updates
Tasks send global updates through channels:

```rust
enum GlobalUpdate {
    SetVariable(Symbol, Var),
    DefineFunction(Symbol, Var),
    RemoveGlobal(Symbol),
}

// Scheduler processes updates and broadcasts to all tasks
```

**Pros**: No shared mutable state, clear event ordering
**Cons**: Async complexity, potential message queue bottlenecks

## Copy Semantics Challenge & Solutions

### The Problem
Task-local heaps + copy semantics creates a fundamental challenge:
- Current heap objects use raw pointers (`*mut LispClosure`, `*mut Environment`)
- These pointers are invalid across different heaps
- Need to copy complex object graphs between isolated heaps
- **No error/exception system exists yet** for handling copy failures

### Current Var Types & Copy Complexity

| Type | Copy Complexity | Notes |
|------|----------------|--------|
| `I32`, `F64`, `Bool`, `None` | ✅ Trivial | Just copy the 64-bit value |
| `Symbol` | ✅ Simple | Interned symbols, just copy ID |
| `String` | 🟡 Medium | Need deep copy of heap string |
| `Tuple` | 🔴 Complex | Recursive copy of all elements |
| `Environment` | 🔴 Complex | Copy bindings + parent chain |
| `Closure` | 🔴 Very Complex | Copy bytecode + captured env |

### Proposed Solutions

#### Option 1: Restricted Task Arguments (Your Suggestion)
```rust
// Only allow copyable types as task arguments
fn validate_task_copyable(var: &Var) -> Result<(), &'static str> {
    match var.get_type() {
        VarType::I32 | VarType::F64 | VarType::Bool | 
        VarType::None | VarType::Symbol => Ok(()),
        VarType::String => Ok(()), // Can deep copy
        VarType::Tuple => {
            // Recursively check all elements
            let elements = var.as_tuple().ok_or("Invalid tuple")?;
            for element in elements {
                validate_task_copyable(element)?;
            }
            Ok(())
        },
        VarType::Closure | VarType::Environment => {
            Err("Cannot pass closures/environments to tasks")
        },
        _ => Err("Unknown type cannot be copied"),
    }
}

// (spawn-task (lambda () (+ 1 2)) 42 "hello")  ; ✅ OK
// (spawn-task (lambda () (+ 1 2)) some-closure) ; ❌ Error
```

**Pros**: Simple, clear semantics, prevents complex copying issues
**Cons**: Restrictive, prevents passing closures as task arguments

#### Option 2: Deep Copy with Serialization
```rust
// Serialize to intermediate representation, then reconstruct
trait DeepCopyable {
    fn serialize(&self) -> Vec<u8>;
    fn deserialize(data: &[u8], target_heap: &mut TaskHeap) -> Result<Var, String>;
}

// Each copyable type implements custom serialization
impl DeepCopyable for LispString { /* ... */ }
impl DeepCopyable for LispTuple { /* ... */ }
// Closures could serialize their bytecode + captured environment
```

**Pros**: Supports all types, clean isolation
**Cons**: Performance overhead, complex implementation

#### Option 3: Closure-Specific Handling
```rust
// Special case for closures - copy the function but not captured environment
pub struct TaskClosure {
    pub function_id: FunctionId,  // Reference to shared bytecode
    pub arity: u32,
    // No captured environment - must use globals or parameters
}

// (spawn-task (lambda (x y) (+ x y)) 10 20)  ; ✅ Pure function
// (spawn-task (lambda () captured-var))      ; ❌ Error - uses capture
```

**Pros**: Allows pure closures, simpler than full deep copy
**Cons**: Still restrictive, complex to validate purity

#### Option 4: Hybrid Approach - Copy Budget
```rust
pub struct CopyBudget {
    max_depth: u32,
    max_objects: u32,
    copied_objects: u32,
}

// Allow copying up to a certain complexity limit
fn deep_copy_with_budget(var: &Var, budget: &mut CopyBudget) -> Result<Var, String> {
    if budget.copied_objects >= budget.max_objects {
        return Err("Copy budget exceeded - object graph too complex");
    }
    // ... recursive copying with budget tracking
}
```

**Pros**: Prevents runaway copying, allows reasonable complexity
**Cons**: Arbitrary limits, still complex to implement

### Recommended Approach: Phased Implementation

**Phase 1: Restricted Arguments (Immediate)**
- Start with Option 1 - only allow simple copyable types
- Build the basic task system without complex copying
- Error handling returns `Result<Var, String>` for now

**Phase 2: Add Error System** 
- Implement proper error/exception handling
- Better error messages for unsupported types

**Phase 3: Expand Copying**
- Add string and simple tuple copying
- Consider closure copying for pure functions

This gives us a working system quickly while leaving room for expansion.

### Task Creation API Design

```rust
// (spawn-task (lambda () (+ 1 2)))           ; ✅ Pure closure
// (spawn-task (lambda (x) (* x x)) 42)       ; ✅ Pure closure + copyable arg  
// (spawn-task (lambda () captured-var))      ; ❌ Error - uses captured variable
// (spawn-task (lambda (f) (f 10)) +)         ; ❌ Error - closure argument

pub fn builtin_spawn_task(args: &[Var]) -> Result<Var, String> {
    let closure = args.get(0).ok_or("spawn-task requires closure argument")?;
    
    if !closure.is_closure() {
        return Err("First argument must be a closure");
    }
    
    // Validate closure is "pure" (no captured environment)
    let closure_ptr = closure.as_closure().unwrap();
    unsafe {
        if !(*closure_ptr).captured_env.is_null() {
            return Err("Cannot spawn task with closure that captures variables");
        }
    }
    
    // Validate all other arguments are copyable
    for arg in &args[1..] {
        validate_task_copyable(arg)
            .map_err(|e| format!("Task argument not copyable: {}", e))?;
    }
    
    // Proceed with task creation...
}
```

## Design Questions Needing Clarification

### 0. Global Namespace Management (CRITICAL)
**Question**: Which approach should we take for handling globals across tasks?

The current system compiles globals as constants, making them per-JIT and immutable. For multitasking, we need to choose:

**A. Runtime Global Resolution** - Replace constants with runtime calls
- Every global access becomes `call(get_global_runtime, symbol)`
- True shared state, immediate visibility of updates
- Performance cost on every access, complex locking

**B. Copy-on-Write Snapshots** - Tasks get versioned global snapshots  
- Fast reads from immutable snapshots
- Check version at safepoints, update snapshot if changed
- Memory overhead, eventual consistency model

**C. Hybrid Function/Variable Handling** - Different semantics by type
- Functions: Compiled constants (fast, immutable, shared code cache)
- Variables: Runtime lookups (slow, mutable, synchronized)
- Optimizes common case but inconsistent semantics

**D. Task-Local with Explicit Sharing** - Isolated by default
- Each task has private globals
- Explicit primitives: `(share-global! 'x)`, `(import-global! 'x)`
- Clear semantics but different programming model

**E. Message-Passing Updates** - No shared mutable state
- Tasks send global updates through channels to scheduler
- Scheduler broadcasts changes to all tasks
- Actor-model style, no locking, but async complexity

**Context**: This decision affects performance, semantics, and implementation complexity of the entire system.

**Specific Sub-Questions:**
- Should REPL globals be automatically visible to spawned tasks?
- Should tasks be able to define globals visible to other tasks?
- What happens to running tasks when globals change?
- How do we handle global functions vs global variables differently (if at all)?

### 1. JIT State Capture for Resumption
**Question**: How do we capture and restore JIT execution state when a task yields/suspends?
- JIT functions are native machine code - we can't easily "pause" them mid-execution
- Do we need to modify the JIT to support resumption points?
- Or do we only allow yielding at specific safepoint locations?

### 2. Global Functions/Variables Resolution
**Question**: How should tasks access and modify globals?
- **Option A**: Snapshot globals at task creation (immutable view)
- **Option B**: Shared concurrent access with locks
- **Option C**: Copy-on-write semantics  
- **Option D**: Message passing for global updates

**Context**: You mentioned this was a "sticky point" - what approach do you prefer?

### 3. Task Result Communication
**Question**: How should tasks communicate results back?
- Store result in task object (current design)?
- Return through channels/queues?
- Callback mechanisms?

### 4. Error Handling in Tasks
**Question**: How should task errors be handled?
- Panic the task thread?
- Store error in task result?
- Propagate to scheduler?

### 5. Task-to-Task Communication
**Question**: Do tasks need to communicate with each other?
- Shared variables? 
  - No
- Message queues? 
  - No
- Channels? 
  - Yes eventually, we will have a new LispChannel type wrapping flume channels. Not today.
- Or is this out of scope for initial implementation?

### 6. Scheduler Lifecycle
**Question**: How is the scheduler created and managed?
    - Should be created at start  
- Global singleton? Yes.
- Part of REPL context? Implicit.
- Explicit scheduler objects? More than one would confuse.

### 7. Memory Model
**Question**: What's the memory visibility model between tasks?
- All heap objects shared (current GC model)?
  - No, but need to think about globals
  - **Copy-on-write semantics?**  not a bad idea.  im::HashMap or similar... somehow?
- Task-local heaps?
  - This is preferred
- Copy semantics for task arguments? -- Yes

## Implementation Considerations

### GC Integration Requirements
- Scheduler must be registered as GC root source
- Tasks must be properly tracked during collection
- Safepoint coordination with existing GC system
- Each task's JIT instance needs GC coordination

### Thread Safety
- Multiple threads accessing shared heap through GC
- Task state coordination through atomics
- Global variable access synchronization

### Performance Considerations
- Thread creation overhead (mitigated by Rayon pool)
- Safepoint check overhead in JIT code
- Global variable access overhead
- Task state synchronization overhead

### Current Limitations to Address
- No existing concurrency support
- Synchronous execution model  
- Single-threaded REPL environment
- Global state management