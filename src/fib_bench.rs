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

//! Standalone fibonacci benchmark for profiling.
//! Runs fibonacci computation in a loop for detailed performance analysis.

mod ast;
mod bytecode;
mod compiler;
mod gc;
mod heap;
mod jit;
mod lexer;
mod parser;
mod protocol;
mod repl;
mod symbol;
mod var;

use repl::Repl;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ROL Fibonacci Standalone Benchmark");
    println!("==================================");

    // Create REPL instance
    let mut repl = Repl::new()?;

    // Define fibonacci function
    println!("Compiling fibonacci function...");
    repl.eval("(defn fib [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))")?;
    println!("Function compiled successfully!");

    // Warm up JIT with a small fibonacci call
    println!("Warming up JIT...");
    repl.eval("(fib 5)")?;
    println!("JIT warmed up!");

    // Parameters for the benchmark
    let fib_n = 20; // Fibonacci number to compute
    let iterations = 5000; // Number of times to run it - increased for better profiling

    println!("Running fibonacci({fib_n}) {iterations} times...");
    println!("This will take a few seconds - perfect for profiling!");

    let start_time = Instant::now();

    // Run fibonacci computation in a loop
    for i in 0..iterations {
        let result = repl.eval(&format!("(fib {fib_n})"))?;

        // Verify result is correct (fib(20) = 6765)
        if let Some(val) = result.as_int() {
            if val != 6765 {
                return Err(format!("Wrong result: expected 6765, got {val}").into());
            }
        } else {
            return Err(format!("Non-integer result: {result:?}").into());
        }
    }

    let total_time = start_time.elapsed();

    println!("Benchmark completed!");
    println!("Total time: {total_time:?}");
    println!("Average per iteration: {:?}", total_time / iterations);
    println!("Result: fib({fib_n}) = 6765");

    Ok(())
}
