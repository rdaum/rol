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

//! Read-Eval-Print Loop for the Lisp interpreter.
//! Provides an interactive shell with readline support, history, and error handling.

use rol::bytecode::{BytecodeCompiler, BytecodeJIT};
use rol::gc::{clear_thread_roots, register_var_as_root};
use rol::heap::Environment;
use rol::parser::parse_expr_string;
use rol::var::{Var, VarType};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

/// REPL state that maintains the bytecode compiler and JIT
pub struct Repl {
    bytecode_compiler: BytecodeCompiler,
    jit: BytecodeJIT,
    editor: DefaultEditor,
    global_env_ptr: *mut Environment,
}

impl Repl {
    /// Create a new REPL instance
    pub fn new() -> std::result::Result<Self, ReadlineError> {
        let bytecode_compiler = BytecodeCompiler::new();
        let jit = BytecodeJIT::new();
        let editor = DefaultEditor::new()?;

        // Create a global environment (empty for now)
        let global_env_ptr = Environment::from_values(&[], None);

        // Register the global environment as a global root
        let global_env_var = Var::environment(global_env_ptr);
        register_var_as_root(global_env_var, true); // true = global root

        Ok(Self {
            bytecode_compiler,
            jit,
            editor,
            global_env_ptr,
        })
    }

    /// Evaluate a Lisp expression string and return the result
    pub fn eval(&mut self, input: &str) -> std::result::Result<Var, Box<dyn std::error::Error>> {
        // Parse the expression
        let expr = parse_expr_string(input).map_err(|e| e as Box<dyn std::error::Error>)?;

        // Compile to bytecode
        let function = self.bytecode_compiler.compile_expr(&expr).map_err(|e| {
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                as Box<dyn std::error::Error>
        })?;

        // JIT compile bytecode to machine code with lambda registry and recursive call info
        let recursive_calls = self.bytecode_compiler.get_recursive_calls();
        let global_symbol_table = self.bytecode_compiler.get_global_symbol_table();
        let func_ptr = self
            .jit
            .compile_function(
                &function,
                &self.bytecode_compiler.lambda_registry,
                recursive_calls,
                global_symbol_table,
            )
            .map_err(|e| {
                Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                    as Box<dyn std::error::Error>
            })?;

        // Execute the compiled function with JIT context
        let result = self.jit.execute_function(func_ptr);

        // NOTE: We don't need to register the result here anymore because
        // heap objects are now registered immediately when they're allocated
        // in Var::string(), Var::tuple(), etc. This prevents double registration.

        // Return result
        Ok(result)
    }

    /// Format a Var for display in the REPL
    pub fn format_result(&self, var: &Var) -> String {
        match var.get_type() {
            VarType::I32 => {
                if let Some(n) = var.as_int() {
                    format!("{n}")
                } else {
                    format!("{var:?}")
                }
            }
            VarType::F64 => {
                if let Some(n) = var.as_double() {
                    format!("{n}")
                } else {
                    format!("{var:?}")
                }
            }
            VarType::String => {
                if let Some(s) = var.as_string() {
                    format!("\"{s}\"")
                } else {
                    format!("{var:?}")
                }
            }
            VarType::Bool => {
                if let Some(b) = var.as_bool() {
                    if b {
                        "true".to_string()
                    } else {
                        "false".to_string()
                    }
                } else {
                    format!("{var:?}")
                }
            }
            VarType::Tuple => {
                // TODO: Format lists nicely
                format!("{var:?}")
            }
            VarType::Environment => "#<environment>".to_string(),
            VarType::Symbol => {
                format!("{var:?}")
            }
            VarType::None => "nil".to_string(),
            VarType::Pointer => "#<pointer>".to_string(),
            VarType::Closure => {
                if let Some(closure_ptr) = var.as_closure() {
                    unsafe { format!("#<closure:{}>", (*closure_ptr).arity) }
                } else {
                    "#<closure:invalid>".to_string()
                }
            }
            VarType::Task => {
                if let Some(task_ptr) = var.as_task() {
                    unsafe { format!("#<task:{}>", (*task_ptr).task_id) }
                } else {
                    "#<task:invalid>".to_string()
                }
            }
        }
    }

    /// Run the main REPL loop
    pub fn run(&mut self) -> std::result::Result<(), ReadlineError> {
        println!("Welcome to ROL - Ryan's Own Lisp!");
        println!("A JIT-compiled Lisp interpreter with lexical scoping.");
        println!("Type expressions to evaluate them, or 'quit' to exit.");
        println!();

        loop {
            match self.editor.readline("rol> ") {
                Ok(line) => {
                    let line = line.trim();

                    // Handle special commands
                    if line.is_empty() {
                        continue;
                    }

                    if line == "quit" || line == "exit" || line == ":q" {
                        println!("Goodbye!");
                        break;
                    }

                    if line == "help" || line == ":help" {
                        self.print_help();
                        continue;
                    }

                    if line == ":fib" {
                        println!("Creating fibonacci function...");
                        println!("For now, fibonacci must be implemented as lambda expressions.");
                        println!("Try this when lambda support is complete:");
                        println!(
                            "  (let ((fib (lambda (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))) (fib 10))"
                        );
                        continue;
                    }

                    // Add to history
                    self.editor.add_history_entry(line)?;

                    // Evaluate the expression
                    match self.eval(line) {
                        Ok(result) => {
                            println!("{}", self.format_result(&result));
                            // Clear thread roots after displaying the result
                            // This allows GC to collect temporary objects from this evaluation
                            clear_thread_roots();
                        }
                        Err(err) => {
                            println!("Error: {err}");
                            // Clear thread roots even on error to prevent accumulation
                            clear_thread_roots();
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!("^D");
                    break;
                }
                Err(err) => {
                    println!("Error: {err:?}");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Print help information
    fn print_help(&self) {
        println!("ROL - Ryan's Own Lisp Help");
        println!("==========================");
        println!();
        println!("Basic Syntax:");
        println!("  42                    ; integers");
        println!("  3.14                  ; floats");
        println!("  \"hello\"               ; strings");
        println!("  :keyword              ; keywords");
        println!();
        println!("Arithmetic:");
        println!("  (+ 2 3)               ; addition → 5");
        println!("  (+ 2.5 3)             ; mixed types → 5.5 (float)");
        println!("  (+ (+ 1 2) 3)         ; nested → 6");
        println!();
        println!("Variables:");
        println!("  (let ((x 5)) x)       ; let binding → 5");
        println!("  (let ((x 5)) (+ x 2)) ; using variables → 7");
        println!("  (let ((x 2) (y 3)) (+ x y))  ; multiple bindings → 5");
        println!();
        println!("Conditionals:");
        println!("  (if 1 42 24)          ; truthy condition → 42");
        println!("  (if 0 42 24)          ; falsy condition → 24");
        println!("  (if (+ 1 2) \"yes\" \"no\")  ; with expressions → \"yes\"");
        println!();
        println!("Nested Expressions:");
        println!("  (let ((x 3)) (let ((y 4)) (+ x y)))  ; nested scoping → 7");
        println!("  (+ (let ((x 1) (y 2)) (+ x y)) (let ((a 3) (b 4)) (+ a b)))");
        println!();
        println!("Commands:");
        println!("  help, :help           ; show this help");
        println!("  quit, exit, :q        ; exit the REPL");
        println!("  Ctrl+C                ; interrupt current input");
        println!("  Ctrl+D                ; exit the REPL");
        println!();
        println!("Features:");
        println!("  • JIT compilation to native machine code");
        println!("  • Smart type coercion (int + int = int, mixed → float)");
        println!("  • Lexical scoping with environment chains");
        println!("  • Proper Lisp truthiness (0, 0.0, false are falsy)");
        println!("  • Readline support with history and line editing");
        println!();
    }
}

impl Drop for Repl {
    fn drop(&mut self) {
        // mmtk handles cleanup automatically - no manual free needed
    }
}

/// Create and run a REPL
pub fn start_repl() -> std::result::Result<(), ReadlineError> {
    let mut repl = Repl::new()?;
    repl.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repl_creation() {
        let repl = Repl::new();
        assert!(repl.is_ok());
    }

    #[test]
    #[ignore] // Disabled due to MMTk shared state issues - run with: cargo test -- --ignored
    fn test_gc_integration_string_evaluation() {
        // Test that string evaluation works with GC root registration
        let mut repl = Repl::new().expect("Failed to create REPL");

        // Evaluate a string literal - should register as thread root and work
        let result = repl
            .eval(r#""hello world""#)
            .expect("Failed to evaluate string");

        // Verify the result is a string
        assert!(result.is_string());
        if let Some(s) = result.as_string() {
            assert_eq!(s, "hello world");
        }

        println!("String evaluation with GC integration: PASSED");
    }

    #[test]
    #[ignore] // Disabled due to MMTk shared state issues - run with: cargo test -- --ignored
    fn test_gc_integration_tuple_evaluation() {
        // Test that tuple evaluation works with GC root registration
        let mut repl = Repl::new().expect("Failed to create REPL");

        // Evaluate an empty tuple - should register as thread root and work
        let result = repl.eval("()").expect("Failed to evaluate empty tuple");

        // Verify the result is a tuple
        assert!(result.is_tuple());
        if let Some(elements) = result.as_tuple() {
            assert_eq!(elements.len(), 0);
        }

        println!("Tuple evaluation with GC integration: PASSED");
    }

    #[test]
    fn test_repl_eval() {
        let mut repl = Repl::new().unwrap();

        // Test basic arithmetic
        let result = repl.eval("(+ 2 3)").unwrap();
        assert_eq!(result.as_int(), Some(5));

        // Test let binding
        let result = repl.eval("(let ((x 10)) (+ x 5))").unwrap();
        assert_eq!(result.as_int(), Some(15));

        // Test if expression
        let result = repl.eval("(if 1 42 0)").unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_result_formatting() {
        let repl = Repl::new().unwrap();

        assert_eq!(repl.format_result(&Var::int(42)), "42");
        assert_eq!(repl.format_result(&Var::float(3.14)), "3.14");
        assert_eq!(repl.format_result(&Var::string("hello")), "\"hello\"");
        assert_eq!(repl.format_result(&Var::bool(true)), "true");
        assert_eq!(repl.format_result(&Var::bool(false)), "false");
        assert_eq!(repl.format_result(&Var::none()), "nil");
    }
}
