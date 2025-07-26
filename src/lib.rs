//! Rol programming language implementation
//!
//! A dynamic functional programming language with garbage collection,
//! bytecode compilation, and JIT compilation.

pub mod ast;
pub mod bytecode;
pub mod gc;
pub mod heap;
pub mod jit;
pub mod lexer;
pub mod parser;
pub mod protocol;
pub mod symbol;
pub mod var;

// Re-export main public APIs
pub use gc::{initialize_mmtk, mmtk_bind_mutator};
pub use var::{Var};