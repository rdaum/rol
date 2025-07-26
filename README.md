# "Ryan's Own Lisp"

A dynamically typed S-expr language designed in the Lisp family for embedding in other applications.

A "New Lisp" thing in the vague style of Janet and maybe a sprinkle of Clojure.

Is currently missing a pile of features still, but has Janet-like tuples (immutable lists),
closures, simple loops, global functions, variables, etc.

Compiles from text to AST to a stack-based bytecode, which is then lazily jitted to native by
[Cranelift](https://cranelift.dev/) before executing. So a bit of a hybrid JIT / VM approach. A lot of room for future
optimization. But actually currently pretty fast.

Uses [MMTk](https://www.mmtk.io/) to support garbage collection. In theory collection is or can be concurrent. Not well
tested. Currently configured with mark & sweep, but more sophisticated algorithms are possible in
the future.

Intended language features missing include quoting, vectors, and a pile of operations on strings,
tuples, vectors, etc.

Simple recursive fibonacci test is currently about 1/3rd the performance of Guile scheme, but about
twice the speed of Python

Features a central runtime scheduler that manages global state and cooperative multitasking.
Tasks can be spawned, joined, and killed with isolated execution contexts.

Extremely rough around the edges and will almost certainly crash.

More of an attempt to see how far I could get with this model of compilation and execution. Coded up
in a frenzy over a few days.

## If you're crazy...

`cargo run --bin rol`

GPL3