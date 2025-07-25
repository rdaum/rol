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

mod ast;
mod bench;
mod bytecode;
mod compiler;
mod environment;
mod gc;
mod heap;
mod integration_tests;
mod jit;
mod lexer;
mod mmtk_binding;
mod parser;
mod protocol;
mod repl;
mod symbol;
mod var;

fn main() {
    // Set up logging to suppress verbose Cranelift output
    if std::env::var("RUST_LOG").is_err() {
        unsafe {
            std::env::set_var("RUST_LOG", "mmtk=info,cranelift_jit=error");
        }
    }

    // Initialize MMTk garbage collector
    if let Err(err) = mmtk_binding::initialize_mmtk() {
        eprintln!("Failed to initialize MMTk: {err}");
        std::process::exit(1);
    }

    // Bind mutator for the main thread
    if let Err(err) = mmtk_binding::mmtk_bind_mutator() {
        eprintln!("Failed to bind mutator for main thread: {err}");
        std::process::exit(1);
    }

    if let Err(err) = repl::start_repl() {
        eprintln!("REPL error: {err}");
        std::process::exit(1);
    }
}
