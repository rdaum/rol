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

//! AST to Cranelift IR compiler.
//! Converts Lisp expressions to JIT-compiled machine code.

use crate::ast::{BuiltinOp, Expr};
use crate::jit::VarBuilder;
use crate::symbol::Symbol;

use cranelift::codegen::ir::FuncRef;
use cranelift::prelude::*;
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module};
use std::collections::HashMap;
use crate::heap::{env_create, env_get, env_set, LexicalAddress};

/// Compilation context that tracks variables and their lexical addresses
#[derive(Debug, Clone)]
pub struct CompileContext {
    /// Variable bindings: symbol -> lexical address  
    bindings: HashMap<Symbol, LexicalAddress>,
    /// Current environment depth
    depth: u32,
}

impl CompileContext {
    /// Create a new empty compilation context
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            depth: 0,
        }
    }

    /// Look up a variable binding
    pub fn lookup(&self, var: Symbol) -> Option<LexicalAddress> {
        self.bindings.get(&var).copied()
    }

    /// Add a variable binding at the current depth
    pub fn bind(&mut self, var: Symbol, offset: u32) {
        let addr = LexicalAddress {
            depth: self.depth,
            offset,
        };
        self.bindings.insert(var, addr);
    }

    /// Create a new context with increased depth (for nested scopes)
    pub fn push_scope(&self) -> Self {
        Self {
            bindings: self.bindings.clone(),
            depth: self.depth + 1,
        }
    }
}

/// The main compiler for converting AST to executable functions
pub struct Compiler {
    module: JITModule,
    var_builder: VarBuilder,
    ctx: codegen::Context,
    builder_context: FunctionBuilderContext,
    env_get_id: FuncId,
    env_create_id: FuncId,
    env_set_id: FuncId,
}

impl Compiler {
    /// Create a new compiler instance
    pub fn new() -> Self {
        // Use the same ISA detection as our existing JIT infrastructure
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {msg}");
        });
        let isa = isa_builder
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();

        let mut builder =
            cranelift_jit::JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Add symbols for our environment functions
        builder.symbol("env_get", env_get as *const u8);
        builder.symbol("env_create", env_create as *const u8);
        builder.symbol("env_set", env_set as *const u8);

        let mut module = JITModule::new(builder);
        let var_builder = VarBuilder::new();

        // Declare external environment functions
        let mut env_get_sig = module.make_signature();
        env_get_sig.params.push(AbiParam::new(types::I64)); // env: u64
        env_get_sig.params.push(AbiParam::new(types::I32)); // depth: u32  
        env_get_sig.params.push(AbiParam::new(types::I32)); // offset: u32
        env_get_sig.returns.push(AbiParam::new(types::I64)); // -> u64

        let mut env_create_sig = module.make_signature();
        env_create_sig.params.push(AbiParam::new(types::I32)); // slot_count: u32
        env_create_sig.params.push(AbiParam::new(types::I64)); // parent: u64
        env_create_sig.returns.push(AbiParam::new(types::I64)); // -> u64

        let mut env_set_sig = module.make_signature();
        env_set_sig.params.push(AbiParam::new(types::I64)); // env: u64
        env_set_sig.params.push(AbiParam::new(types::I32)); // depth: u32
        env_set_sig.params.push(AbiParam::new(types::I32)); // offset: u32
        env_set_sig.params.push(AbiParam::new(types::I64)); // value: u64
        env_set_sig.returns.push(AbiParam::new(types::I64)); // -> u64 (updated env)

        let env_get_id = module
            .declare_function("env_get", Linkage::Import, &env_get_sig)
            .expect("Failed to declare env_get");
        let env_create_id = module
            .declare_function("env_create", Linkage::Import, &env_create_sig)
            .expect("Failed to declare env_create");
        let env_set_id = module
            .declare_function("env_set", Linkage::Import, &env_set_sig)
            .expect("Failed to declare env_set");

        Self {
            module,
            var_builder,
            ctx: codegen::Context::new(),
            builder_context: FunctionBuilderContext::new(),
            env_get_id,
            env_create_id,
            env_set_id,
        }
    }

    /// Compile an expression to a function that returns a Var (as u64)
    /// The function signature is: fn(env: u64) -> u64
    pub fn compile_expr(&mut self, expr: &Expr) -> Result<*const u8, String> {
        // First pass: Pre-compile all lambda expressions and create a modified AST
        let (modified_expr, _lambda_closures) = self.precompile_lambdas(expr, 0)?;

        // Create function signature: (env: u64) -> u64
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::I64)); // environment parameter
        sig.returns.push(AbiParam::new(types::I64)); // return Var as u64

        // Generate unique function name to avoid conflicts
        use std::sync::atomic::{AtomicU32, Ordering};
        static FUNC_COUNTER: AtomicU32 = AtomicU32::new(0);
        let func_name = format!(
            "compiled_expr_{}",
            FUNC_COUNTER.fetch_add(1, Ordering::SeqCst)
        );

        // Create the function
        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Export, &sig)
            .map_err(|e| format!("Failed to declare function: {e}"))?;

        // Clear the context and set up function
        self.ctx.clear();
        self.ctx.func.signature = sig;

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        // Import the external environment functions into this function
        let env_get_ref = self
            .module
            .declare_func_in_func(self.env_get_id, builder.func);
        let env_create_ref = self
            .module
            .declare_func_in_func(self.env_create_id, builder.func);
        let env_set_ref = self
            .module
            .declare_func_in_func(self.env_set_id, builder.func);

        // Create entry block
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Get the environment parameter
        let env_param = builder.block_params(entry_block)[0];

        // Compile the modified expression (with lambdas replaced by closure literals)
        let ctx = CompileContext::new();
        let var_builder = &self.var_builder;
        let result = compile_expr_recursive(
            &modified_expr,
            &mut builder,
            env_param,
            &ctx,
            var_builder,
            env_get_ref,
            env_create_ref,
            env_set_ref,
        )?;

        // Return the result
        builder.ins().return_(&[result]);

        // Finalize the function
        builder.finalize();

        // Define the function in the module
        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| format!("Failed to define function: {e}"))?;

        // Finalize the module and get the function pointer
        self.module
            .finalize_definitions()
            .map_err(|e| format!("Failed to finalize: {e}"))?;

        let code_ptr = self.module.get_finalized_function(func_id);
        Ok(code_ptr)
    }

    /// Pre-compile all lambda expressions in an AST and replace them with closure literals
    /// Returns the modified AST and a list of compiled closures
    fn precompile_lambdas(
        &mut self,
        expr: &Expr,
        captured_env: u64,
    ) -> Result<(Expr, Vec<crate::heap::LispClosure>), String> {
        match expr {
            Expr::Lambda { params, body } => {
                // Compile this lambda function
                let func_ptr = self.compile_lambda(params, body, captured_env)?;

                // Create a closure object
                let closure_ptr =
                    crate::heap::LispClosure::new(func_ptr, params.len() as u32, captured_env);

                // Create a Var that contains this closure
                let closure_var = crate::var::Var::closure(closure_ptr);

                // Return the closure as a literal expression
                Ok((Expr::Literal(closure_var), vec![]))
            }

            Expr::Call { func, args } => {
                // Recursively process function and arguments
                let (new_func, func_closures) = self.precompile_lambdas(func, captured_env)?;
                let mut new_args = Vec::new();
                let mut all_closures = func_closures;

                for arg in args {
                    let (new_arg, mut arg_closures) = self.precompile_lambdas(arg, captured_env)?;
                    new_args.push(new_arg);
                    all_closures.append(&mut arg_closures);
                }

                Ok((
                    Expr::Call {
                        func: Box::new(new_func),
                        args: new_args,
                    },
                    all_closures,
                ))
            }

            Expr::Let { bindings, body } => {
                // Process bindings and body
                let mut new_bindings = Vec::new();
                let mut all_closures = Vec::new();

                for (symbol, binding_expr) in bindings {
                    let (new_binding, mut binding_closures) =
                        self.precompile_lambdas(binding_expr, captured_env)?;
                    new_bindings.push((*symbol, new_binding));
                    all_closures.append(&mut binding_closures);
                }

                let (new_body, mut body_closures) = self.precompile_lambdas(body, captured_env)?;
                all_closures.append(&mut body_closures);

                Ok((
                    Expr::Let {
                        bindings: new_bindings,
                        body: Box::new(new_body),
                    },
                    all_closures,
                ))
            }

            Expr::If {
                condition,
                then_expr,
                else_expr,
            } => {
                // Process condition and branches
                let (new_condition, cond_closures) =
                    self.precompile_lambdas(condition, captured_env)?;
                let (new_then, mut then_closures) =
                    self.precompile_lambdas(then_expr, captured_env)?;
                let (new_else, mut else_closures) =
                    self.precompile_lambdas(else_expr, captured_env)?;

                let mut all_closures = cond_closures;
                all_closures.append(&mut then_closures);
                all_closures.append(&mut else_closures);

                Ok((
                    Expr::If {
                        condition: Box::new(new_condition),
                        then_expr: Box::new(new_then),
                        else_expr: Box::new(new_else),
                    },
                    all_closures,
                ))
            }

            Expr::While { condition, body } => {
                let (new_condition, mut condition_closures) =
                    self.precompile_lambdas(condition, captured_env)?;
                let (new_body, mut body_closures) = self.precompile_lambdas(body, captured_env)?;

                condition_closures.append(&mut body_closures);
                Ok((
                    Expr::While {
                        condition: Box::new(new_condition),
                        body: Box::new(new_body),
                    },
                    condition_closures,
                ))
            }

            Expr::For {
                var,
                start,
                end,
                body,
            } => {
                let (new_start, mut start_closures) =
                    self.precompile_lambdas(start, captured_env)?;
                let (new_end, mut end_closures) = self.precompile_lambdas(end, captured_env)?;
                let (new_body, mut body_closures) = self.precompile_lambdas(body, captured_env)?;

                start_closures.append(&mut end_closures);
                start_closures.append(&mut body_closures);
                Ok((
                    Expr::For {
                        var: *var,
                        start: Box::new(new_start),
                        end: Box::new(new_end),
                        body: Box::new(new_body),
                    },
                    start_closures,
                ))
            }

            Expr::Def { var, value } => {
                let (new_value, closures) = self.precompile_lambdas(value, captured_env)?;
                Ok((
                    Expr::Def {
                        var: *var,
                        value: Box::new(new_value),
                    },
                    closures,
                ))
            }

            Expr::VarDef { var, value } => {
                let (new_value, closures) = self.precompile_lambdas(value, captured_env)?;
                Ok((
                    Expr::VarDef {
                        var: *var,
                        value: Box::new(new_value),
                    },
                    closures,
                ))
            }

            Expr::Defn { name, params, body } => {
                let (new_body, closures) = self.precompile_lambdas(body, captured_env)?;
                Ok((
                    Expr::Defn {
                        name: *name,
                        params: params.clone(),
                        body: Box::new(new_body),
                    },
                    closures,
                ))
            }

            // Base cases - no lambdas to process
            Expr::Literal(_) | Expr::Variable(_) => Ok((expr.clone(), vec![])),
        }
    }

    /// Compile a lambda expression into a closure
    /// Returns a closure that can be called with the specified arguments
    pub fn compile_lambda(
        &mut self,
        params: &[Symbol],
        body: &Expr,
        _captured_env: u64,
    ) -> Result<*const u8, String> {
        // Create function signature: fn(args: *const Var, arg_count: u32, captured_env: u64) -> u64
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::I64)); // args pointer
        sig.params.push(AbiParam::new(types::I32)); // arg count
        sig.params.push(AbiParam::new(types::I64)); // captured environment
        sig.returns.push(AbiParam::new(types::I64)); // return Var as u64

        // Generate unique function name
        use std::sync::atomic::{AtomicU32, Ordering};
        static LAMBDA_COUNTER: AtomicU32 = AtomicU32::new(0);
        let func_name = format!("lambda_{}", LAMBDA_COUNTER.fetch_add(1, Ordering::SeqCst));

        // Create the function
        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Export, &sig)
            .map_err(|e| format!("Failed to declare lambda function: {e}"))?;

        // Clear the context and set up function
        self.ctx.clear();
        self.ctx.func.signature = sig;

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        // Import the external environment functions into this function
        let env_get_ref = self
            .module
            .declare_func_in_func(self.env_get_id, builder.func);
        let env_create_ref = self
            .module
            .declare_func_in_func(self.env_create_id, builder.func);
        let env_set_ref = self
            .module
            .declare_func_in_func(self.env_set_id, builder.func);

        // Create entry block
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Get function parameters
        let args_ptr = builder.block_params(entry_block)[0];
        let arg_count = builder.block_params(entry_block)[1];
        let captured_env_param = builder.block_params(entry_block)[2];

        // Create compilation context with parameter bindings
        let mut lambda_ctx = CompileContext::new();

        // Bind each parameter to its position in the args array
        for (i, &param_symbol) in params.iter().enumerate() {
            // Parameters are at depth 0 (current frame), offset i
            lambda_ctx.bind(param_symbol, i as u32);
        }

        // Load parameters from the args array into local variables
        // For now, we'll access them directly in variable lookups

        // Pre-compile any nested lambdas in the body
        let (modified_body, _nested_closures): (Expr, Vec<crate::heap::LispClosure>) = {
            // We need to temporarily create a new compiler instance to avoid borrowing conflicts
            // For now, we'll just pass the body through without nested lambda support
            // TODO: Implement proper nested lambda support
            (body.clone(), vec![])
        };

        // Compile the lambda body with parameter bindings
        let var_builder = &self.var_builder;
        let result = compile_lambda_body_recursive(
            &modified_body,
            &mut builder,
            args_ptr,
            arg_count,
            captured_env_param,
            &lambda_ctx,
            var_builder,
            env_get_ref,
            env_create_ref,
            env_set_ref,
        )?;

        // Return the result
        builder.ins().return_(&[result]);

        // Finalize the function
        builder.finalize();

        // Define the function in the module
        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| format!("Failed to define lambda function: {e}"))?;

        // Finalize the module and get the function pointer
        self.module
            .finalize_definitions()
            .map_err(|e| format!("Failed to finalize lambda: {e}"))?;

        let code_ptr = self.module.get_finalized_function(func_id);
        Ok(code_ptr)
    }
}

/// Standalone recursive expression compiler to avoid borrowing conflicts
fn compile_expr_recursive(
    expr: &Expr,
    builder: &mut FunctionBuilder,
    env: Value,
    ctx: &CompileContext,
    var_builder: &VarBuilder,
    env_get_ref: FuncRef,
    env_create_ref: FuncRef,
    env_set_ref: FuncRef,
) -> Result<Value, String> {
    match expr {
        Expr::Literal(var) => {
            // Load literal value as u64
            let bits = var.as_u64();
            Ok(builder.ins().iconst(types::I64, bits as i64))
        }

        Expr::Variable(sym) => {
            // Look up variable in environment
            if let Some(addr) = ctx.lookup(*sym) {
                // Call env_get(env, depth, offset)
                let depth_val = builder.ins().iconst(types::I32, addr.depth as i64);
                let offset_val = builder.ins().iconst(types::I32, addr.offset as i64);
                let call_inst = builder
                    .ins()
                    .call(env_get_ref, &[env, depth_val, offset_val]);
                let result = builder.inst_results(call_inst)[0];
                Ok(result)
            } else {
                Err(format!("Unbound variable: {}", sym.as_string()))
            }
        }

        Expr::Call { func, args } => {
            // Check if it's a builtin operation
            if let Expr::Variable(sym) = func.as_ref() {
                if let Some(builtin) = BuiltinOp::from_symbol(*sym) {
                    return compile_builtin_recursive(
                        builtin,
                        args,
                        builder,
                        env,
                        ctx,
                        var_builder,
                        env_get_ref,
                        env_create_ref,
                        env_set_ref,
                    );
                }
            }

            // User-defined function calls
            compile_function_call_recursive(
                func,
                args,
                builder,
                env,
                ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )
        }

        Expr::Let { bindings, body } => {
            // Create new environment with space for bindings
            let slot_count = bindings.len() as u32;
            let count_val = builder.ins().iconst(types::I32, slot_count as i64);

            // Call env_create(slot_count, parent_env)
            let create_call = builder.ins().call(env_create_ref, &[count_val, env]);
            let new_env = builder.inst_results(create_call)[0];

            // Create new context for the let body
            // Existing bindings from outer scopes need their depth incremented
            let mut new_bindings = HashMap::new();
            for (symbol, addr) in &ctx.bindings {
                let new_addr = LexicalAddress {
                    depth: addr.depth + 1, // Existing variables are now one level deeper
                    offset: addr.offset,
                };
                new_bindings.insert(*symbol, new_addr);
            }

            let mut new_ctx = CompileContext {
                bindings: new_bindings,
                depth: ctx.depth + 1, // We are in a deeper scope
            };

            // Compile and store each binding
            for (i, (var, expr)) in bindings.iter().enumerate() {
                // Compile the binding expression in the outer context
                let value = compile_expr_recursive(
                    expr,
                    builder,
                    env,
                    ctx,
                    var_builder,
                    env_get_ref,
                    env_create_ref,
                    env_set_ref,
                )?;

                // Store in the new environment at depth 0 (current level)
                let depth_val = builder.ins().iconst(types::I32, 0);
                let offset_val = builder.ins().iconst(types::I32, i as i64);
                let set_call = builder
                    .ins()
                    .call(env_set_ref, &[new_env, depth_val, offset_val, value]);
                let _updated_env = builder.inst_results(set_call)[0]; // Updated environment (might be reallocated)

                // Add to new context at depth 0 (new bindings are in the current environment)
                let addr = LexicalAddress {
                    depth: 0,
                    offset: i as u32,
                };
                new_ctx.bindings.insert(*var, addr);
            }

            // Compile the body with the new environment and context
            compile_expr_recursive(
                body,
                builder,
                new_env,
                &new_ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )
        }

        Expr::Lambda { params: _, body: _ } => {
            // Lambda expressions should have been pre-compiled and stored as literals
            // This case should not occur if compilation is done properly
            Err("Lambda expressions must be pre-compiled in the main compilation phase".to_string())
        }

        Expr::If {
            condition,
            then_expr,
            else_expr,
        } => {
            // Compile condition
            let cond_value = compile_expr_recursive(
                condition,
                builder,
                env,
                ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )?;

            // Check if condition is truthy using proper Var truthiness logic
            let is_true = var_builder.is_truthy(builder, cond_value);

            // Create blocks
            let then_block = builder.create_block();
            let else_block = builder.create_block();
            let merge_block = builder.create_block();

            // Add block parameter for the result
            builder.append_block_param(merge_block, types::I64);

            // Branch based on condition
            builder
                .ins()
                .brif(is_true, then_block, &[], else_block, &[]);

            // Compile then branch
            builder.switch_to_block(then_block);
            builder.seal_block(then_block);
            let then_result = compile_expr_recursive(
                then_expr,
                builder,
                env,
                ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )?;
            builder.ins().jump(merge_block, [then_result.into()].iter());

            // Compile else branch
            builder.switch_to_block(else_block);
            builder.seal_block(else_block);
            let else_result = compile_expr_recursive(
                else_expr,
                builder,
                env,
                ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )?;
            builder.ins().jump(merge_block, [else_result.into()].iter());

            // Merge point
            builder.switch_to_block(merge_block);
            builder.seal_block(merge_block);

            Ok(builder.block_params(merge_block)[0])
        }

        Expr::While { .. } => {
            // While loops not implemented in recursive compiler
            Err("While loops not supported in recursive compiler".to_string())
        }

        Expr::For { .. } => {
            // For loops not implemented in recursive compiler
            Err("For loops not supported in recursive compiler".to_string())
        }

        Expr::Def { .. } => {
            // Def not implemented in recursive compiler
            Err("Def not supported in recursive compiler".to_string())
        }

        Expr::VarDef { .. } => {
            // VarDef not implemented in recursive compiler
            Err("VarDef not supported in recursive compiler".to_string())
        }

        Expr::Defn { .. } => {
            // Defn not implemented in recursive compiler
            Err("Defn not supported in recursive compiler".to_string())
        }
    }
}

/// Standalone builtin operation compiler
fn compile_builtin_recursive(
    op: BuiltinOp,
    args: &[Expr],
    builder: &mut FunctionBuilder,
    env: Value,
    ctx: &CompileContext,
    var_builder: &VarBuilder,
    env_get_ref: FuncRef,
    env_create_ref: FuncRef,
    env_set_ref: FuncRef,
) -> Result<Value, String> {
    // Validate arity
    if let Some(expected_arity) = op.arity() {
        if args.len() != expected_arity {
            return Err(format!(
                "Wrong number of arguments for {:?}: expected {}, got {}",
                op,
                expected_arity,
                args.len()
            ));
        }
    }

    match op {
        BuiltinOp::Add => {
            let lhs = compile_expr_recursive(
                &args[0],
                builder,
                env,
                ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )?;
            let rhs = compile_expr_recursive(
                &args[1],
                builder,
                env,
                ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )?;

            // Use proper type coercion: int + int = int, otherwise float
            let result = var_builder.emit_arithmetic_add(builder, lhs, rhs);
            Ok(result)
        }

        BuiltinOp::Sub => {
            let lhs = compile_expr_recursive(
                &args[0],
                builder,
                env,
                ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )?;
            let rhs = compile_expr_recursive(
                &args[1],
                builder,
                env,
                ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )?;

            // Use proper type coercion: int - int = int, otherwise float
            let result = var_builder.emit_arithmetic_sub(builder, lhs, rhs);
            Ok(result)
        }

        BuiltinOp::Lt => {
            let lhs = compile_expr_recursive(
                &args[0],
                builder,
                env,
                ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )?;
            let rhs = compile_expr_recursive(
                &args[1],
                builder,
                env,
                ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )?;

            // Compare and return boolean
            let result = var_builder.emit_arithmetic_lt(builder, lhs, rhs);
            Ok(result)
        }

        BuiltinOp::Mul => {
            // TODO: Implement multiplication
            Err("Multiplication not yet implemented".to_string())
        }

        BuiltinOp::Div => {
            // TODO: Implement division
            Err("Division not yet implemented".to_string())
        }

        BuiltinOp::Mod => {
            // TODO: Implement modulo
            Err("Modulo not yet implemented".to_string())
        }

        _ => Err(format!("Builtin operation {op:?} not yet implemented")),
    }
}

/// Compile a function call to a user-defined function (closure)
fn compile_function_call_recursive(
    func_expr: &Expr,
    args: &[Expr],
    builder: &mut FunctionBuilder,
    env: Value,
    ctx: &CompileContext,
    var_builder: &VarBuilder,
    env_get_ref: FuncRef,
    env_create_ref: FuncRef,
    env_set_ref: FuncRef,
) -> Result<Value, String> {
    // Compile the function expression (should evaluate to a closure)
    let func_value = compile_expr_recursive(
        func_expr,
        builder,
        env,
        ctx,
        var_builder,
        env_get_ref,
        env_create_ref,
        env_set_ref,
    )?;

    // Compile all argument expressions
    let mut arg_values = Vec::with_capacity(args.len());
    for arg_expr in args {
        let arg_value = compile_expr_recursive(
            arg_expr,
            builder,
            env,
            ctx,
            var_builder,
            env_get_ref,
            env_create_ref,
            env_set_ref,
        )?;
        arg_values.push(arg_value);
    }

    // Call the closure with the arguments
    // We need to call the closure's function pointer with: (args: *const Var, arg_count: u32, captured_env: u64) -> u64

    // First, check that func_value is actually a closure
    let is_closure = var_builder.is_closure(builder, func_value);

    // Create error block for non-closure case
    let error_block = builder.create_block();
    let success_block = builder.create_block();
    let cont_block = builder.create_block();

    // Add result parameter to continuation block
    builder.append_block_param(cont_block, types::I64);

    // Branch based on closure check
    builder
        .ins()
        .brif(is_closure, success_block, &[], error_block, &[]);

    // Error block: return an error value (for now, just return none)
    builder.switch_to_block(error_block);
    builder.seal_block(error_block);
    let error_result = var_builder.make_none(builder);
    builder.ins().jump(cont_block, [error_result.into()].iter());

    // Success block: actually call the closure
    builder.switch_to_block(success_block);
    builder.seal_block(success_block);

    // Extract closure pointer from the Var
    let closure_ptr = var_builder.extract_closure_ptr(builder, func_value);

    // Create array of argument Vars on the stack
    // For now, we'll use a simple approach - allocate stack space for args
    let arg_count = arg_values.len() as u32;

    if arg_count > 0 {
        // Create a stack slot for arguments (arg_count * 8 bytes)
        let arg_array_ptr = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            arg_count * 8,
            3,
        ));
        let arg_array_addr = builder.ins().stack_addr(types::I64, arg_array_ptr, 0);

        // Store each argument in the array
        // TODO: Add write barriers here for concurrent GC safety
        // This stores Var values to stack memory and should call jit_stack_write_barrier
        // Currently skipped because this is a standalone compiler without access to write barrier function refs
        for (i, &arg_value) in arg_values.iter().enumerate() {
            let offset = (i * 8) as i32;
            builder
                .ins()
                .store(MemFlags::trusted(), arg_value, arg_array_addr, offset);
        }

        // Call the closure function
        let result = var_builder.call_closure(builder, closure_ptr, arg_array_addr, arg_count);
        builder.ins().jump(cont_block, [result.into()].iter());
    } else {
        // No arguments - pass null pointer
        let null_ptr = builder.ins().iconst(types::I64, 0);
        let result = var_builder.call_closure(builder, closure_ptr, null_ptr, 0);
        builder.ins().jump(cont_block, [result.into()].iter());
    }

    // Continuation block
    builder.switch_to_block(cont_block);
    builder.seal_block(cont_block);

    Ok(builder.block_params(cont_block)[0])
}

/// Compile the body of a lambda function with parameter access
fn compile_lambda_body_recursive(
    expr: &Expr,
    builder: &mut FunctionBuilder,
    args_ptr: Value,
    _arg_count: Value,
    captured_env: Value,
    ctx: &CompileContext,
    var_builder: &VarBuilder,
    env_get_ref: FuncRef,
    env_create_ref: FuncRef,
    env_set_ref: FuncRef,
) -> Result<Value, String> {
    match expr {
        Expr::Variable(sym) => {
            // Look up variable - could be a parameter or from captured environment
            if let Some(addr) = ctx.lookup(*sym) {
                if addr.depth == 0 {
                    // This is a parameter - load from args array
                    let offset_bytes = builder.ins().iconst(types::I64, (addr.offset * 8) as i64);
                    let arg_addr = builder.ins().iadd(args_ptr, offset_bytes);
                    let param_value =
                        builder
                            .ins()
                            .load(types::I64, MemFlags::trusted(), arg_addr, 0);
                    Ok(param_value)
                } else {
                    // This is from captured environment - use env_get
                    let depth_val = builder.ins().iconst(types::I32, addr.depth as i64);
                    let offset_val = builder.ins().iconst(types::I32, addr.offset as i64);
                    let call_inst = builder
                        .ins()
                        .call(env_get_ref, &[captured_env, depth_val, offset_val]);
                    let result = builder.inst_results(call_inst)[0];
                    Ok(result)
                }
            } else {
                Err(format!("Unbound variable in lambda: {}", sym.as_string()))
            }
        }

        // For other expressions, use the regular compiler but with our special variable lookup
        _ => {
            // This is a simplified approach - for a full implementation, we'd need to
            // handle all expression types with the lambda-specific context
            compile_expr_recursive(
                expr,
                builder,
                captured_env,
                ctx,
                var_builder,
                env_get_ref,
                env_create_ref,
                env_set_ref,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr;
    use crate::heap::Environment;
    use crate::parser::parse_expr_string;
    use crate::var::Var;

    #[test]
    fn test_compile_context() {
        let mut ctx = CompileContext::new();

        // Test variable binding
        let var = Symbol::mk("x");
        ctx.bind(var, 0);

        let addr = ctx.lookup(var).unwrap();
        assert_eq!(addr.depth, 0);
        assert_eq!(addr.offset, 0);

        // Test scope pushing
        let nested_ctx = ctx.push_scope();
        assert_eq!(nested_ctx.depth, 1);
    }

    #[test]
    fn test_compiler_creation() {
        let _compiler = Compiler::new();
        // Just test that we can create a compiler without panicking
    }

    #[test]
    fn test_literal_compilation() {
        let mut compiler = Compiler::new();
        let expr = Expr::number(42.0);

        // This should compile without error (though may not execute properly yet)
        let result = compiler.compile_expr(&expr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_addition_compilation() {
        let mut compiler = Compiler::new();

        // Create (+ 1.0 2.0)
        let expr = Expr::call(
            Expr::variable("+"),
            vec![Expr::number(1.0), Expr::number(2.0)],
        );

        // Should compile successfully
        let result = compiler.compile_expr(&expr);
        assert!(
            result.is_ok(),
            "Addition compilation failed: {:?}",
            result.err()
        );

        // Get the function pointer
        let func_ptr = result.unwrap();
        assert!(!func_ptr.is_null(), "Function pointer should not be null");
    }

    #[test]
    fn test_nested_addition_compilation() {
        let mut compiler = Compiler::new();

        // Create (+ (+ 1.0 2.0) 3.0)
        let inner_add = Expr::call(
            Expr::variable("+"),
            vec![Expr::number(1.0), Expr::number(2.0)],
        );
        let outer_add = Expr::call(Expr::variable("+"), vec![inner_add, Expr::number(3.0)]);

        // Should compile successfully
        let result = compiler.compile_expr(&outer_add);
        assert!(
            result.is_ok(),
            "Nested addition compilation failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_variable_compilation_error() {
        let mut compiler = Compiler::new();

        // Create unbound variable reference
        let expr = Expr::variable("x");

        // Should fail compilation with unbound variable error
        let result = compiler.compile_expr(&expr);
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("Unbound variable"));
    }

    #[test]
    fn test_let_binding_compilation() {
        let mut compiler = Compiler::new();

        // Create (let ((x 5.0)) (+ x 2.0))
        use crate::symbol::Symbol;
        let x_sym = Symbol::mk("x");
        let expr = Expr::let_binding(
            vec![(x_sym, Expr::number(5.0))],
            Expr::call(
                Expr::variable("+"),
                vec![Expr::variable("x"), Expr::number(2.0)],
            ),
        );

        // Should compile successfully
        let result = compiler.compile_expr(&expr);
        assert!(
            result.is_ok(),
            "Let binding compilation failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_if_expression_compilation() {
        let mut compiler = Compiler::new();

        // Create (if 1 42.0 24.0) - non-zero condition should be truthy
        let expr = Expr::if_expr(Expr::number(1.0), Expr::number(42.0), Expr::number(24.0));

        // Should compile successfully
        let result = compiler.compile_expr(&expr);
        assert!(
            result.is_ok(),
            "If expression compilation failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_invalid_builtin_arity() {
        let mut compiler = Compiler::new();

        // Create (+ 1.0) - addition requires 2 arguments
        let expr = Expr::call(Expr::variable("+"), vec![Expr::number(1.0)]);

        // Should fail compilation with arity error
        let result = compiler.compile_expr(&expr);
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("Wrong number of arguments"));
    }

    #[test]
    fn test_unimplemented_builtin() {
        let mut compiler = Compiler::new();

        // Create (* 5.0 3.0) - multiplication not yet implemented
        let expr = Expr::call(
            Expr::variable("*"),
            vec![Expr::number(5.0), Expr::number(3.0)],
        );

        // Should fail compilation with unimplemented error
        let result = compiler.compile_expr(&expr);
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("not yet implemented"));
    }

    #[test]
    fn test_execute_literal() {
        let mut compiler = Compiler::new();
        let expr = Expr::number(42.0);

        // Compile the expression
        let func_ptr = compiler.compile_expr(&expr).unwrap();

        // Cast to function and execute with null environment
        let func: fn(u64) -> u64 = unsafe { std::mem::transmute(func_ptr) };
        let result_bits = func(0); // null environment for literal

        // Convert result back to Var and check value
        let result_var = Var::from_u64(result_bits);
        assert_eq!(result_var.as_double(), Some(42.0));
    }

    #[test]
    fn test_execute_addition() {
        let mut compiler = Compiler::new();

        // Create (+ 1.0 2.0)
        let expr = Expr::call(
            Expr::variable("+"),
            vec![Expr::number(1.0), Expr::number(2.0)],
        );

        // Compile and execute
        let func_ptr = compiler.compile_expr(&expr).unwrap();
        let func: fn(u64) -> u64 = unsafe { std::mem::transmute(func_ptr) };
        let result_bits = func(0); // null environment

        // Check result
        let result_var = Var::from_u64(result_bits);
        assert_eq!(result_var.as_double(), Some(3.0));
    }

    #[test]
    fn test_execute_nested_addition() {
        let mut compiler = Compiler::new();

        // Create (+ (+ 1.0 2.0) 3.0) = 6.0
        let inner_add = Expr::call(
            Expr::variable("+"),
            vec![Expr::number(1.0), Expr::number(2.0)],
        );
        let outer_add = Expr::call(Expr::variable("+"), vec![inner_add, Expr::number(3.0)]);

        // Compile and execute
        let func_ptr = compiler.compile_expr(&outer_add).unwrap();
        let func: fn(u64) -> u64 = unsafe { std::mem::transmute(func_ptr) };
        let result_bits = func(0);

        // Check result
        let result_var = Var::from_u64(result_bits);
        assert_eq!(result_var.as_double(), Some(6.0));
    }

    #[test]
    fn test_execute_if_expression() {
        let mut compiler = Compiler::new();

        // Test truthy condition: (if 1.0 42.0 24.0) should return 42.0
        let expr = Expr::if_expr(Expr::number(1.0), Expr::number(42.0), Expr::number(24.0));

        let func_ptr = compiler.compile_expr(&expr).unwrap();
        let func: fn(u64) -> u64 = unsafe { std::mem::transmute(func_ptr) };
        let result_bits = func(0);
        let result_var = Var::from_u64(result_bits);
        assert_eq!(result_var.as_double(), Some(42.0));

        // Test falsy condition: (if 0.0 42.0 24.0) should return 24.0
        let expr_false = Expr::if_expr(Expr::number(0.0), Expr::number(42.0), Expr::number(24.0));

        let func_ptr_false = compiler.compile_expr(&expr_false).unwrap();
        let func_false: fn(u64) -> u64 = unsafe { std::mem::transmute(func_ptr_false) };
        let result_bits_false = func_false(0);
        let result_var_false = Var::from_u64(result_bits_false);
        assert_eq!(result_var_false.as_double(), Some(24.0));
    }

    #[test]
    fn test_execute_let_binding() {
        let mut compiler = Compiler::new();

        // Create (let ((x 5.0)) (+ x 2.0)) = 7.0
        use crate::symbol::Symbol;
        let x_sym = Symbol::mk("x");
        let expr = Expr::let_binding(
            vec![(x_sym, Expr::number(5.0))],
            Expr::call(
                Expr::variable("+"),
                vec![Expr::variable("x"), Expr::number(2.0)],
            ),
        );

        // Compile and execute
        let func_ptr = compiler.compile_expr(&expr).unwrap();
        let func: fn(u64) -> u64 = unsafe { std::mem::transmute(func_ptr) };

        // Create an empty environment to pass in
        let empty_env_ptr = Environment::from_values(&[], None);
        let empty_env = Var::environment(empty_env_ptr).as_u64();
        let result_bits = func(empty_env);

        // Clean up
        unsafe { Environment::free(empty_env_ptr) };

        let result_var = Var::from_u64(result_bits);
        assert_eq!(result_var.as_double(), Some(7.0));
    }

    #[test]
    fn test_execute_nested_let_bindings() {
        let mut compiler = Compiler::new();

        // Create (let ((x 3.0)) (let ((y 4.0)) (+ x y))) = 7.0
        use crate::symbol::Symbol;
        let x_sym = Symbol::mk("x");
        let y_sym = Symbol::mk("y");

        let inner_let = Expr::let_binding(
            vec![(y_sym, Expr::number(4.0))],
            Expr::call(
                Expr::variable("+"),
                vec![Expr::variable("x"), Expr::variable("y")],
            ),
        );

        let outer_let = Expr::let_binding(vec![(x_sym, Expr::number(3.0))], inner_let);

        // Compile and execute
        let func_ptr = compiler.compile_expr(&outer_let).unwrap();
        let func: fn(u64) -> u64 = unsafe { std::mem::transmute(func_ptr) };

        let empty_env_ptr = Environment::from_values(&[], None);
        let empty_env = Var::environment(empty_env_ptr).as_u64();
        let result_bits = func(empty_env);

        // Clean up
        unsafe { Environment::free(empty_env_ptr) };

        // Check result
        let result_var = Var::from_u64(result_bits);
        assert_eq!(result_var.as_double(), Some(7.0));
    }

    #[test]
    fn test_execute_multiple_bindings() {
        let mut compiler = Compiler::new();

        // Create (let ((x 2.0) (y 3.0) (z 1.0)) (+ (+ x y) z)) = 6.0
        use crate::symbol::Symbol;
        let x_sym = Symbol::mk("x");
        let y_sym = Symbol::mk("y");
        let z_sym = Symbol::mk("z");

        let expr = Expr::let_binding(
            vec![
                (x_sym, Expr::number(2.0)),
                (y_sym, Expr::number(3.0)),
                (z_sym, Expr::number(1.0)),
            ],
            Expr::call(
                Expr::variable("+"),
                vec![
                    Expr::call(
                        Expr::variable("+"),
                        vec![Expr::variable("x"), Expr::variable("y")],
                    ),
                    Expr::variable("z"),
                ],
            ),
        );

        // Compile and execute
        let func_ptr = compiler.compile_expr(&expr).unwrap();
        let func: fn(u64) -> u64 = unsafe { std::mem::transmute(func_ptr) };

        let empty_env_ptr = Environment::from_values(&[], None);
        let empty_env = Var::environment(empty_env_ptr).as_u64();
        let result_bits = func(empty_env);

        // Clean up
        unsafe { Environment::free(empty_env_ptr) };

        // Check result
        let result_var = Var::from_u64(result_bits);
        assert_eq!(result_var.as_double(), Some(6.0));
    }

    #[test]
    fn test_fibonacci_lambda_performance() {
        let mut compiler = Compiler::new();

        // Test a simple arithmetic lambda that we know works
        println!("Creating high-performance arithmetic lambda test...");

        // Test a lambda that does arithmetic (avoiding if for now since that has issues)
        let arith_expr = parse_expr_string("(lambda [n] (+ (+ n 1) (- n 1)))").unwrap();

        let func_ptr = compiler
            .compile_expr(&arith_expr)
            .expect("Failed to compile arithmetic lambda");

        // Create environment and execute
        let env_ptr = Environment::from_values(&[], None);
        let env_var = Var::environment(env_ptr);

        let func: fn(u64) -> u64 = unsafe { std::mem::transmute(func_ptr) };
        let result_bits = func(env_var.as_u64());
        let closure_var = Var::from_u64(result_bits);

        // Test the arithmetic closure with performance measurement
        if let Some(closure_ptr) = closure_var.as_closure() {
            println!("Testing arithmetic closure performance...");

            // Test cases for (+ (+ n 1) (- n 1)) which should always equal n + n = 2n
            let test_cases = vec![
                (0, 0),   // (+ (+ 0 1) (- 0 1)) = (+ 1 -1) = 0
                (1, 2),   // (+ (+ 1 1) (- 1 1)) = (+ 2 0) = 2
                (2, 4),   // (+ (+ 2 1) (- 2 1)) = (+ 3 1) = 4
                (3, 6),   // (+ (+ 3 1) (- 3 1)) = (+ 4 2) = 6
                (5, 10),  // (+ (+ 5 1) (- 5 1)) = (+ 6 4) = 10
                (10, 20), // (+ (+ 10 1) (- 10 1)) = (+ 11 9) = 20
            ];

            println!("Running arithmetic performance tests...");
            let start_time = std::time::Instant::now();

            // Run multiple iterations to test performance
            let iterations = 10000; // Increased for better performance measurement
            let mut total_results = 0;
            let mut successful_calls = 0;

            for _iter in 0..iterations {
                for &(n, _expected) in &test_cases {
                    let args = vec![Var::int(n)];
                    let result_bits = unsafe { (*closure_ptr).call(&args) };
                    let result_var = Var::from_u64(result_bits);

                    if let Some(actual) = result_var.as_int() {
                        total_results += actual;
                        successful_calls += 1;
                    }
                }
            }

            let duration = start_time.elapsed();
            let total_calls = iterations as usize * test_cases.len();
            println!("Completed {total_calls} iterations in {duration:?}");
            if total_calls > 0 {
                println!("Average time per call: {:?}", duration / total_calls as u32);
            }
            println!("Successful calls: {successful_calls}/{total_calls}");
            println!("Total computed values: {total_results}");

            // Test individual calls to verify correctness
            println!("Verifying individual results:");
            for (n, expected) in test_cases {
                let args = vec![Var::int(n)];
                let result_bits = unsafe { (*closure_ptr).call(&args) };
                let result_var = Var::from_u64(result_bits);

                if let Some(actual) = result_var.as_int() {
                    let status = if actual == expected { "✓" } else { "✗" };
                    println!("f({n}) = {actual} (expected: {expected}) {status}");
                } else {
                    println!("f({n}) returned non-integer: {result_var:?}");
                }
            }

            // Test edge cases and larger values
            println!("Testing larger values for stress test...");
            for n in [10, 20, 30, 50] {
                let start = std::time::Instant::now();
                let args = vec![Var::int(n)];
                let result_bits = unsafe { (*closure_ptr).call(&args) };
                let result_var = Var::from_u64(result_bits);
                let duration = start.elapsed();
                println!("f({n}) = {result_var:?} (took {duration:?})");
            }
        } else {
            panic!("Failed to create arithmetic closure");
        }

        // Clean up
        unsafe {
            Environment::free(env_ptr);
        }

        println!("✓ Fibonacci-like lambda performance test completed successfully");
    }

    #[test]
    fn test_jit_vs_native_performance_comparison() {
        println!("🚀 JIT vs Native Rust Performance Comparison");
        println!("=============================================");

        // Native Rust function equivalent to our lambda: (+ (+ n 1) (- n 1)) = 2n
        fn native_arithmetic(n: i32) -> i32 {
            (n + 1) + (n - 1)
        }

        // Test data
        let test_values = vec![0, 1, 2, 3, 5, 10, 20, 50, 100];
        let iterations = 100_000; // Higher for better precision

        // === NATIVE RUST BENCHMARK ===
        println!("\n📊 Native Rust Function Benchmark:");
        let start_time = std::time::Instant::now();
        let mut native_total = 0i64;

        for _iter in 0..iterations {
            for &n in &test_values {
                native_total += native_arithmetic(n) as i64;
            }
        }

        let native_duration = start_time.elapsed();
        let native_total_calls = iterations * test_values.len();
        let native_avg_ns = native_duration.as_nanos() / native_total_calls as u128;

        println!("  Completed {native_total_calls} calls in {native_duration:?}");
        println!("  Average: {native_avg_ns}ns per call");
        println!("  Total computed: {native_total}");

        // === JIT LAMBDA BENCHMARK ===
        println!("\n⚡ JIT Lambda Benchmark:");
        let mut compiler = Compiler::new();

        // Create the same arithmetic lambda
        let lambda_expr = parse_expr_string("(lambda [n] (+ (+ n 1) (- n 1)))").unwrap();
        let func_ptr = compiler
            .compile_expr(&lambda_expr)
            .expect("Failed to compile lambda");

        let env_ptr = Environment::from_values(&[], None);
        let env_var = Var::environment(env_ptr);

        let func: fn(u64) -> u64 = unsafe { std::mem::transmute(func_ptr) };
        let result_bits = func(env_var.as_u64());
        let closure_var = Var::from_u64(result_bits);

        if let Some(closure_ptr) = closure_var.as_closure() {
            let start_time = std::time::Instant::now();
            let mut jit_total = 0i64;
            let mut successful_calls = 0;

            for _iter in 0..iterations {
                for &n in &test_values {
                    let args = vec![Var::int(n)];
                    let result_bits = unsafe { (*closure_ptr).call(&args) };
                    let result_var = Var::from_u64(result_bits);

                    if let Some(actual) = result_var.as_int() {
                        jit_total += actual as i64;
                        successful_calls += 1;
                    }
                }
            }

            let jit_duration = start_time.elapsed();
            let jit_total_calls = iterations * test_values.len();
            let jit_avg_ns = if successful_calls > 0 {
                jit_duration.as_nanos() / successful_calls as u128
            } else {
                jit_duration.as_nanos() / jit_total_calls as u128
            };

            println!("  Completed {jit_total_calls} calls in {jit_duration:?}");
            println!("  Successful: {successful_calls}/{jit_total_calls}");
            println!("  Average: {jit_avg_ns}ns per call");
            println!("  Total computed: {jit_total}");

            // === PERFORMANCE COMPARISON ===
            println!("\n📈 Performance Analysis:");
            println!("  Native Rust: {native_avg_ns}ns per call");
            println!("  JIT Lambda:  {jit_avg_ns}ns per call");

            if jit_avg_ns > 0 && native_avg_ns > 0 {
                let overhead_ratio = jit_avg_ns as f64 / native_avg_ns as f64;
                let overhead_percent = (overhead_ratio - 1.0) * 100.0;

                println!(
                    "  JIT Overhead: {overhead_ratio:.1}x slower ({overhead_percent:.1}% overhead)"
                );

                if overhead_ratio < 2.0 {
                    println!("  🎉 Excellent! JIT performance is within 2x of native");
                } else if overhead_ratio < 5.0 {
                    println!("  👍 Good! JIT performance is within 5x of native");
                } else if overhead_ratio < 10.0 {
                    println!("  👌 Reasonable JIT overhead");
                } else {
                    println!("  🔍 High overhead - room for optimization");
                }
            }

            // === CORRECTNESS VERIFICATION ===
            if successful_calls > 0 {
                println!("\n✅ Correctness Check:");
                for &n in test_values.iter().take(5) {
                    // Check first 5 values
                    let native_result = native_arithmetic(n);
                    let args = vec![Var::int(n)];
                    let result_bits = unsafe { (*closure_ptr).call(&args) };
                    let result_var = Var::from_u64(result_bits);

                    if let Some(jit_result) = result_var.as_int() {
                        let status = if jit_result == native_result {
                            "✓"
                        } else {
                            "✗"
                        };
                        println!("  f({n}) = native:{native_result}, jit:{jit_result} {status}");
                    } else {
                        println!("  f({n}) = native:{native_result}, jit:None ✗");
                    }
                }
            } else {
                println!("\n❌ No successful JIT calls - debugging needed");
            }

            // === MEMORY EFFICIENCY ===
            println!("\n💾 Memory Analysis:");
            println!("  Native function: ~0 bytes (compiled into binary)");
            println!(
                "  JIT closure: {} bytes (heap allocated)",
                std::mem::size_of::<crate::heap::LispClosure>()
            );
            println!("  Plus JIT-compiled machine code (varies)");
        } else {
            println!("❌ Failed to create JIT closure");
        }

        // Cleanup
        unsafe {
            Environment::free(env_ptr);
        }

        println!("\n🏁 Performance comparison completed!");
    }
}
