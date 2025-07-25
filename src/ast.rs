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

//! Abstract Syntax Tree for Lisp expressions.
//! Simple but extensible design for JIT compilation.

use crate::symbol::Symbol;
use crate::var::Var;

/// A Lisp expression that can be compiled to machine code.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal values: numbers, booleans, strings
    Literal(Var),

    /// Variable reference by symbol
    Variable(Symbol),

    /// Function call: (func arg1 arg2 ...)
    Call { func: Box<Expr>, args: Vec<Expr> },

    /// Let binding: (let ((var1 val1) (var2 val2) ...) body)
    Let {
        bindings: Vec<(Symbol, Expr)>,
        body: Box<Expr>,
    },

    /// Lambda expression: (lambda (param1 param2 ...) body)
    Lambda {
        params: Vec<Symbol>,
        body: Box<Expr>,
    },

    /// If conditional: (if condition then-expr else-expr)  
    If {
        condition: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },

    /// While loop: (while condition body)
    While {
        condition: Box<Expr>,
        body: Box<Expr>,
    },

    /// For loop: (for var start end body)
    For {
        var: Symbol,
        start: Box<Expr>,
        end: Box<Expr>,
        body: Box<Expr>,
    },

    /// Global definition: (def var value)
    Def { var: Symbol, value: Box<Expr> },

    /// Mutable global variable: (var name value)
    VarDef { var: Symbol, value: Box<Expr> },

    /// Function definition: (defn name [params...] body)
    Defn {
        name: Symbol,
        params: Vec<Symbol>,
        body: Box<Expr>,
    },
}

impl Expr {
    /// Create a literal number expression
    pub fn number(value: f64) -> Self {
        Expr::Literal(Var::float(value))
    }

    /// Create a literal integer expression  
    pub fn int(value: i32) -> Self {
        Expr::Literal(Var::int(value))
    }

    /// Create a boolean literal
    pub fn boolean(value: bool) -> Self {
        Expr::Literal(Var::bool(value))
    }

    /// Create a string literal
    pub fn string(value: &str) -> Self {
        Expr::Literal(Var::string(value))
    }

    /// Create a variable reference
    pub fn variable(name: &str) -> Self {
        Expr::Variable(Symbol::mk(name))
    }

    /// Create a function call
    pub fn call(func: Expr, args: Vec<Expr>) -> Self {
        Expr::Call {
            func: Box::new(func),
            args,
        }
    }

    /// Create a let binding
    pub fn let_binding(bindings: Vec<(Symbol, Expr)>, body: Expr) -> Self {
        Expr::Let {
            bindings,
            body: Box::new(body),
        }
    }

    /// Create a lambda expression
    pub fn lambda(params: Vec<Symbol>, body: Expr) -> Self {
        Expr::Lambda {
            params,
            body: Box::new(body),
        }
    }

    /// Create an if expression
    pub fn if_expr(condition: Expr, then_expr: Expr, else_expr: Expr) -> Self {
        Expr::If {
            condition: Box::new(condition),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        }
    }
}

/// Built-in functions that the compiler knows how to handle directly
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logic
    And,
    Or,
    Not,
}

impl BuiltinOp {
    /// Get the builtin operation for a symbol name
    pub fn from_symbol(sym: Symbol) -> Option<Self> {
        let name = sym.as_string();
        match name.as_str() {
            "+" => Some(BuiltinOp::Add),
            "-" => Some(BuiltinOp::Sub),
            "*" => Some(BuiltinOp::Mul),
            "/" => Some(BuiltinOp::Div),
            "%" => Some(BuiltinOp::Mod),
            "=" => Some(BuiltinOp::Eq),
            "!=" => Some(BuiltinOp::Ne),
            "<" => Some(BuiltinOp::Lt),
            "<=" => Some(BuiltinOp::Le),
            ">" => Some(BuiltinOp::Gt),
            ">=" => Some(BuiltinOp::Ge),
            "and" => Some(BuiltinOp::And),
            "or" => Some(BuiltinOp::Or),
            "not" => Some(BuiltinOp::Not),
            _ => None,
        }
    }

    /// Get the expected number of arguments for this builtin
    pub fn arity(self) -> Option<usize> {
        match self {
            BuiltinOp::Not => Some(1),
            BuiltinOp::Add
            | BuiltinOp::Sub
            | BuiltinOp::Mul
            | BuiltinOp::Div
            | BuiltinOp::Mod
            | BuiltinOp::Eq
            | BuiltinOp::Ne
            | BuiltinOp::Lt
            | BuiltinOp::Le
            | BuiltinOp::Gt
            | BuiltinOp::Ge
            | BuiltinOp::And
            | BuiltinOp::Or => Some(2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_constructors() {
        // Test literal constructors
        let num_expr = Expr::number(42.0);
        assert!(matches!(num_expr, Expr::Literal(_)));

        let bool_expr = Expr::boolean(true);
        assert!(matches!(bool_expr, Expr::Literal(_)));

        // Test variable reference
        let var_expr = Expr::variable("x");
        assert!(matches!(var_expr, Expr::Variable(_)));

        // Test function call
        let call_expr = Expr::call(
            Expr::variable("+"),
            vec![Expr::number(1.0), Expr::number(2.0)],
        );
        assert!(matches!(call_expr, Expr::Call { .. }));
    }

    #[test]
    fn test_builtin_operations() {
        // Test builtin operation recognition
        let add_sym = Symbol::mk("+");
        assert_eq!(BuiltinOp::from_symbol(add_sym), Some(BuiltinOp::Add));

        let unknown_sym = Symbol::mk("unknown");
        assert_eq!(BuiltinOp::from_symbol(unknown_sym), None);

        // Test arity
        assert_eq!(BuiltinOp::Add.arity(), Some(2));
        assert_eq!(BuiltinOp::Not.arity(), Some(1));
    }

    #[test]
    fn test_complex_expressions() {
        // Test let binding: (let ((x 5)) (+ x 2))
        let let_expr = Expr::let_binding(
            vec![(Symbol::mk("x"), Expr::number(5.0))],
            Expr::call(
                Expr::variable("+"),
                vec![Expr::variable("x"), Expr::number(2.0)],
            ),
        );

        if let Expr::Let { bindings, body } = let_expr {
            assert_eq!(bindings.len(), 1);
            assert_eq!(bindings[0].0, Symbol::mk("x"));
            assert!(matches!(body.as_ref(), Expr::Call { .. }));
        } else {
            panic!("Expected Let expression");
        }
    }

    #[test]
    fn test_lambda_expression() {
        // Test lambda: (lambda (x y) (+ x y))
        let lambda_expr = Expr::lambda(
            vec![Symbol::mk("x"), Symbol::mk("y")],
            Expr::call(
                Expr::variable("+"),
                vec![Expr::variable("x"), Expr::variable("y")],
            ),
        );

        if let Expr::Lambda { params, body } = lambda_expr {
            assert_eq!(params.len(), 2);
            assert_eq!(params[0], Symbol::mk("x"));
            assert_eq!(params[1], Symbol::mk("y"));
            assert!(matches!(body.as_ref(), Expr::Call { .. }));
        } else {
            panic!("Expected Lambda expression");
        }
    }
}
