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

//! Integration tests for the complete Lisp interpreter pipeline.
//! Tests: source code -> lexer -> parser -> compiler -> execution

use crate::compiler::Compiler;
use crate::environment::Environment;
use crate::parser::parse_expr_string;
use crate::var::Var;

/// Helper function to compile and execute a Lisp expression string
fn eval_lisp(source: &str) -> Result<Var, Box<dyn std::error::Error>> {
    // Parse the source code
    let expr = parse_expr_string(source)?;

    // Compile to machine code
    let mut compiler = Compiler::new();
    let func_ptr = compiler.compile_expr(&expr)?;

    // Execute with empty environment
    let func: fn(u64) -> u64 = unsafe { std::mem::transmute(func_ptr) };
    let empty_env_ptr = Environment::from_values(&[], None);
    let empty_env = Var::environment(empty_env_ptr).as_u64();
    let result_bits = func(empty_env);

    // Clean up
    unsafe { Environment::free(empty_env_ptr) };

    // Return result
    Ok(Var::from_u64(result_bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_numbers() {
        // Test integer literal
        let result = eval_lisp("42").unwrap();
        assert_eq!(result.as_int(), Some(42));

        // Test float literal
        let result = eval_lisp("3.14").unwrap();
        assert_eq!(result.as_double(), Some(3.14));

        // Test negative numbers
        let result = eval_lisp("-17").unwrap();
        assert_eq!(result.as_int(), Some(-17));
    }

    #[test]
    fn test_simple_arithmetic() {
        // Test integer addition (int + int = int)
        let result = eval_lisp("(+ 2 3)").unwrap();
        assert_eq!(result.as_int(), Some(5));

        // Test float addition (float + float = float)
        let result = eval_lisp("(+ 2.5 1.5)").unwrap();
        assert_eq!(result.as_double(), Some(4.0));

        // Test mixed int/float (coerced to float)
        let result = eval_lisp("(+ 2 3.5)").unwrap();
        assert_eq!(result.as_double(), Some(5.5));

        // Test mixed float/int (coerced to float)
        let result = eval_lisp("(+ 2.5 3)").unwrap();
        assert_eq!(result.as_double(), Some(5.5));
    }

    #[test]
    fn test_nested_arithmetic() {
        // Test nested addition (all integers)
        let result = eval_lisp("(+ (+ 1 2) 3)").unwrap();
        assert_eq!(result.as_int(), Some(6));

        // Test more complex nesting (all integers)
        let result = eval_lisp("(+ (+ 1 2) (+ 3 4))").unwrap();
        assert_eq!(result.as_int(), Some(10));

        // Test nested with mixed types (should coerce to float)
        let result = eval_lisp("(+ (+ 1 2.0) 3)").unwrap();
        assert_eq!(result.as_double(), Some(6.0));
    }

    #[test]
    fn test_let_bindings() {
        // Simple let binding
        let result = eval_lisp("(let ((x 5)) x)").unwrap();
        assert_eq!(result.as_int(), Some(5));

        // Let binding with arithmetic (int + int = int)
        let result = eval_lisp("(let ((x 5)) (+ x 2))").unwrap();
        assert_eq!(result.as_int(), Some(7));

        // Multiple bindings (int + int = int)
        let result = eval_lisp("(let ((x 3) (y 4)) (+ x y))").unwrap();
        assert_eq!(result.as_int(), Some(7));
    }

    #[test]
    fn test_nested_let_bindings() {
        // Nested let bindings (int + int = int)
        let result = eval_lisp("(let ((x 2)) (let ((y 3)) (+ x y)))").unwrap();
        assert_eq!(result.as_int(), Some(5));

        // More complex nesting with shadowing-like behavior (int + int = int)
        let result = eval_lisp("(let ((x 1)) (let ((x 2) (y 3)) (+ x y)))").unwrap();
        assert_eq!(result.as_int(), Some(5));
    }

    #[test]
    fn test_if_expressions() {
        // Simple if with truthy condition
        let result = eval_lisp("(if 1 42 24)").unwrap();
        assert_eq!(result.as_int(), Some(42));

        // Simple if with falsy condition
        let result = eval_lisp("(if 0 42 24)").unwrap();
        assert_eq!(result.as_int(), Some(24));

        // If with arithmetic in branches (int + int = int)
        let result = eval_lisp("(if 1 (+ 2 3) (+ 4 5))").unwrap();
        assert_eq!(result.as_int(), Some(5));
    }

    #[test]
    fn test_complex_expressions() {
        // Let with if (int + int = int)
        let result = eval_lisp("(let ((x 5)) (if (+ x 0) (+ x 10) (+ x 20)))").unwrap();
        assert_eq!(result.as_int(), Some(15));

        // Nested let with arithmetic (int + int = int)
        let result = eval_lisp("(let ((a 2)) (let ((b 3) (c 4)) (+ (+ a b) c)))").unwrap();
        assert_eq!(result.as_int(), Some(9));

        // Complex nested expression (int + int = int)
        let result =
            eval_lisp("(+ (let ((x 1) (y 2)) (+ x y)) (let ((a 3) (b 4)) (+ a b)))").unwrap();
        assert_eq!(result.as_int(), Some(10));
    }

    #[test]
    fn test_string_literals() {
        let result = eval_lisp(r#""hello world""#).unwrap();
        assert_eq!(result.as_string(), Some("hello world"));

        // Test strings with escapes
        let result = eval_lisp(r#""hello\nworld""#).unwrap();
        assert_eq!(result.as_string(), Some("hello\nworld"));
    }

    #[test]
    fn test_error_handling() {
        // Test unbound variable
        let result = eval_lisp("undefined_var");
        assert!(result.is_err());

        // Test invalid syntax
        let result = eval_lisp("(+ 1 2");
        assert!(result.is_err());

        // Test wrong number of arguments
        let result = eval_lisp("(+ 1)");
        assert!(result.is_err());
    }

    #[test]
    fn test_whitespace_and_comments() {
        // Test with lots of whitespace (int + int = int)
        let result = eval_lisp("  (  +   1    2  )  ").unwrap();
        assert_eq!(result.as_int(), Some(3));

        // Test with comments (int + int = int)
        let result = eval_lisp("(+ 1 2) ; this is a comment").unwrap();
        assert_eq!(result.as_int(), Some(3));

        // Test multiline with comments (int + int = int)
        let result = eval_lisp("(let ((x 5)) ; bind x to 5\n  (+ x 2))   ; add 2 to x").unwrap();
        assert_eq!(result.as_int(), Some(7));
    }

    #[test]
    fn test_keywords() {
        // For now, keywords are treated as string literals
        let result = eval_lisp(":foo").unwrap();
        assert_eq!(result.as_string(), Some(":foo"));
    }
}
