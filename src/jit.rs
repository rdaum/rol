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

//! Cranelift JIT integration for NaN-boxed Var operations.
//! Provides helpers for emitting cranelift IR that manipulates our Var type efficiently.

use crate::var::{
    BOOLEAN_FALSE, BOOLEAN_TRUE, DOUBLE_ENCODE_OFFSET, HIGH16_TAG, MIN_NUMBER, MIN_POINTER, NULL,
    POINTER_TAG_MASK, STRING_POINTER_TAG, SYMBOL_TAG, TUPLE_POINTER_TAG, Var,
};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

// Generate arithmetic and comparison methods using macros
macro_rules! impl_arithmetic_binop {
    ($name:ident, $int_op:ident, $float_op:ident) => {
        pub fn $name(&self, builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
            // Check if both operands are integers
            let lhs_is_int = self.is_int(builder, lhs);
            let rhs_is_int = self.is_int(builder, rhs);
            let both_ints = builder.ins().band(lhs_is_int, rhs_is_int);

            // Create blocks for int and float paths
            let int_block = builder.create_block();
            let float_block = builder.create_block();
            let merge_block = builder.create_block();

            // Add block parameter for the result
            builder.append_block_param(merge_block, types::I64);

            // Branch based on types
            builder
                .ins()
                .brif(both_ints, int_block, &[], float_block, &[]);

            // Integer path: both are ints, use int operation
            builder.switch_to_block(int_block);
            builder.seal_block(int_block);
            let lhs_int = self.extract_int(builder, lhs);
            let rhs_int = self.extract_int(builder, rhs);
            let int_result = builder.ins().$int_op(lhs_int, rhs_int);
            let int_var = self.make_int(builder, int_result);
            builder.ins().jump(merge_block, [int_var.into()].iter());

            // Float path: at least one is a double, use float operation
            builder.switch_to_block(float_block);
            builder.seal_block(float_block);
            let lhs_double = self.coerce_to_double(builder, lhs);
            let rhs_double = self.coerce_to_double(builder, rhs);
            let lhs_f64 = self.extract_double(builder, lhs_double);
            let rhs_f64 = self.extract_double(builder, rhs_double);
            let float_result = builder.ins().$float_op(lhs_f64, rhs_f64);
            let float_var = self.make_double(builder, float_result);
            builder.ins().jump(merge_block, [float_var.into()].iter());

            // Merge point
            builder.switch_to_block(merge_block);
            builder.seal_block(merge_block);

            builder.block_params(merge_block)[0]
        }
    };
}

macro_rules! impl_comparison_binop {
    ($name:ident, $int_cond:ident, $float_cond:ident) => {
        pub fn $name(&self, builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
            // Check if both operands are integers
            let lhs_is_int = self.is_int(builder, lhs);
            let rhs_is_int = self.is_int(builder, rhs);
            let both_ints = builder.ins().band(lhs_is_int, rhs_is_int);

            // Create blocks for int and float paths
            let int_block = builder.create_block();
            let float_block = builder.create_block();
            let merge_block = builder.create_block();

            // Add block parameter for the result
            builder.append_block_param(merge_block, types::I64);

            // Branch based on types
            builder
                .ins()
                .brif(both_ints, int_block, &[], float_block, &[]);

            // Integer path: both are ints, use int comparison
            builder.switch_to_block(int_block);
            builder.seal_block(int_block);
            let lhs_int = self.extract_int(builder, lhs);
            let rhs_int = self.extract_int(builder, rhs);
            let int_cmp = builder.ins().icmp(IntCC::$int_cond, lhs_int, rhs_int);
            let int_result = builder.ins().uextend(types::I64, int_cmp);
            let int_var = self.make_bool(builder, int_result);
            builder.ins().jump(merge_block, [int_var.into()].iter());

            // Float path: at least one is a double, use float comparison
            builder.switch_to_block(float_block);
            builder.seal_block(float_block);
            let lhs_double = self.coerce_to_double(builder, lhs);
            let rhs_double = self.coerce_to_double(builder, rhs);
            let lhs_f64 = self.extract_double(builder, lhs_double);
            let rhs_f64 = self.extract_double(builder, rhs_double);
            let float_cmp = builder.ins().fcmp(FloatCC::$float_cond, lhs_f64, rhs_f64);
            let float_result = builder.ins().uextend(types::I64, float_cmp);
            let float_var = self.make_bool(builder, float_result);
            builder.ins().jump(merge_block, [float_var.into()].iter());

            // Merge point
            builder.switch_to_block(merge_block);
            builder.seal_block(merge_block);

            builder.block_params(merge_block)[0]
        }
    };
}

pub struct VarJIT {
    builder_context: FunctionBuilderContext,
    ctx: codegen::Context,
    module: JITModule,
    isa: cranelift::codegen::isa::OwnedTargetIsa,
}

impl VarJIT {
    pub fn new() -> Self {
        // Use cranelift-native to get the appropriate target for this machine
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {msg}");
        });
        let isa = isa_builder
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();

        let builder = JITBuilder::with_isa(isa.clone(), cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);

        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
            isa,
        }
    }

    /// Create a function that takes two Vars and returns a Var
    pub fn create_binary_function(&mut self, name: &str) -> FuncId {
        let int = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(int)); // First Var
        sig.params.push(AbiParam::new(int)); // Second Var  
        sig.returns.push(AbiParam::new(int)); // Result Var

        self.module
            .declare_function(name, Linkage::Export, &sig)
            .unwrap()
    }

    /// Compile a function using the provided compilation closure
    pub fn compile_function<F>(&mut self, func_id: FuncId, compile_fn: F) -> *const u8
    where
        F: FnOnce(&mut FunctionBuilder, &VarBuilder),
    {
        self.ctx.func.signature = self
            .module
            .declarations()
            .get_function_decl(func_id)
            .signature
            .clone();
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
        let var_builder = VarBuilder::new();

        compile_fn(&mut builder, &var_builder);

        builder.finalize();

        self.module.define_function(func_id, &mut self.ctx).unwrap();
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().unwrap();

        self.module.get_finalized_function(func_id)
    }
}

pub struct VarBuilder;

impl VarBuilder {
    pub fn new() -> Self {
        Self
    }

    // Type Testing Helpers
    pub fn is_int(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // Integers have high bits matching MIN_NUMBER's pattern (0x0006...)
        // Test: (var & HIGH16_TAG) == (MIN_NUMBER & HIGH16_TAG)
        let high_mask = builder.ins().iconst(types::I64, HIGH16_TAG as i64);
        let masked = builder.ins().band(var, high_mask);
        let min_num_high = builder
            .ins()
            .iconst(types::I64, (MIN_NUMBER & HIGH16_TAG) as i64);

        builder.ins().icmp(IntCC::Equal, masked, min_num_high)
    }

    pub fn is_double(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // Use the same logic as the Rust implementation:
        // is_double() = is_number() && !is_int()
        // where is_number() = (value >= MIN_NUMBER) && !is_symbol()

        let is_number = self.is_number(builder, var);
        let is_int = self.is_int(builder, var);
        let not_int = builder.ins().bnot(is_int);

        builder.ins().band(is_number, not_int)
    }

    pub fn is_number(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // is_number() = (value >= MIN_NUMBER) && !is_symbol() && !is_list() && !is_string()
        let min_number = builder.ins().iconst(types::I64, MIN_NUMBER as i64);
        let is_ge_min = builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, var, min_number);

        // Check for symbol (high bits match SYMBOL_TAG pattern)
        let symbol_high = builder
            .ins()
            .iconst(types::I64, (SYMBOL_TAG & HIGH16_TAG) as i64);
        let high_mask = builder.ins().iconst(types::I64, HIGH16_TAG as i64);
        let high_bits = builder.ins().band(var, high_mask);
        let is_symbol = builder.ins().icmp(IntCC::Equal, high_bits, symbol_high);
        let not_symbol = builder.ins().bnot(is_symbol);

        // Check it's not a list or string
        let is_list = self.is_list(builder, var);
        let is_string = self.is_string(builder, var);
        let is_heap_type = builder.ins().bor(is_list, is_string);
        let not_heap_type = builder.ins().bnot(is_heap_type);

        // Combine all conditions: >= MIN_NUMBER && !symbol && !list && !string
        let basic_check = builder.ins().band(is_ge_min, not_symbol);
        builder.ins().band(basic_check, not_heap_type)
    }

    pub fn is_bool(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // var == BOOLEAN_TRUE || var == BOOLEAN_FALSE
        let true_val = builder.ins().iconst(types::I64, BOOLEAN_TRUE as i64);
        let false_val = builder.ins().iconst(types::I64, BOOLEAN_FALSE as i64);

        let is_true = builder.ins().icmp(IntCC::Equal, var, true_val);
        let is_false = builder.ins().icmp(IntCC::Equal, var, false_val);
        builder.ins().bor(is_true, is_false)
    }

    pub fn is_list(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // is_list: (v & 0x03) == 0 && v >= MIN_POINTER && (v & POINTER_TAG_MASK) == LIST_POINTER_TAG

        // Check low 2 bits == 0
        let low_mask = builder.ins().iconst(types::I64, 0x03);
        let low_bits = builder.ins().band(var, low_mask);
        let zero = builder.ins().iconst(types::I64, 0);
        let low_bits_zero = builder.ins().icmp(IntCC::Equal, low_bits, zero);

        // Check >= MIN_POINTER
        let min_ptr = builder.ins().iconst(types::I64, MIN_POINTER as i64);
        let is_ge_min = builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, var, min_ptr);

        // Check pointer tag mask == LIST_POINTER_TAG
        let tag_mask = builder.ins().iconst(types::I64, POINTER_TAG_MASK as i64);
        let tag_bits = builder.ins().band(var, tag_mask);
        let list_tag = builder.ins().iconst(types::I64, TUPLE_POINTER_TAG as i64);
        let is_list_tag = builder.ins().icmp(IntCC::Equal, tag_bits, list_tag);

        // Combine all conditions
        let aligned_and_valid = builder.ins().band(low_bits_zero, is_ge_min);
        builder.ins().band(aligned_and_valid, is_list_tag)
    }

    pub fn is_string(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // is_string: (v & 0x03) == 0 && v >= MIN_POINTER && (v & POINTER_TAG_MASK) == STRING_POINTER_TAG

        // Check low 2 bits == 0
        let low_mask = builder.ins().iconst(types::I64, 0x03);
        let low_bits = builder.ins().band(var, low_mask);
        let zero = builder.ins().iconst(types::I64, 0);
        let low_bits_zero = builder.ins().icmp(IntCC::Equal, low_bits, zero);

        // Check >= MIN_POINTER
        let min_ptr = builder.ins().iconst(types::I64, MIN_POINTER as i64);
        let is_ge_min = builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, var, min_ptr);

        // Check pointer tag mask == STRING_POINTER_TAG
        let tag_mask = builder.ins().iconst(types::I64, POINTER_TAG_MASK as i64);
        let tag_bits = builder.ins().band(var, tag_mask);
        let string_tag = builder.ins().iconst(types::I64, STRING_POINTER_TAG as i64);
        let is_string_tag = builder.ins().icmp(IntCC::Equal, tag_bits, string_tag);

        // Combine all conditions
        let aligned_and_valid = builder.ins().band(low_bits_zero, is_ge_min);
        builder.ins().band(aligned_and_valid, is_string_tag)
    }

    // Value Extraction Helpers
    pub fn extract_int(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // Subtract MIN_NUMBER to get the actual int value
        let offset = builder.ins().iconst(types::I64, MIN_NUMBER as i64);
        let raw_int = builder.ins().isub(var, offset);
        // Truncate to i32 and sign extend back to i64
        let as_i32 = builder.ins().ireduce(types::I32, raw_int);
        builder.ins().sextend(types::I64, as_i32)
    }

    pub fn extract_double(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // Subtract DOUBLE_ENCODE_OFFSET and reinterpret as f64
        let offset = builder
            .ins()
            .iconst(types::I64, DOUBLE_ENCODE_OFFSET as i64);
        let raw_bits = builder.ins().isub(var, offset);
        builder.ins().bitcast(types::F64, MemFlags::new(), raw_bits)
    }

    pub fn extract_bool(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // Just check if var == BOOLEAN_TRUE
        let true_val = builder.ins().iconst(types::I64, BOOLEAN_TRUE as i64);
        builder.ins().icmp(IntCC::Equal, var, true_val)
    }

    // Construction Helpers
    pub fn make_int(&self, builder: &mut FunctionBuilder, value: Value) -> Value {
        // Add MIN_NUMBER to the i64 value
        let offset = builder.ins().iconst(types::I64, MIN_NUMBER as i64);
        builder.ins().iadd(value, offset)
    }

    pub fn make_double(&self, builder: &mut FunctionBuilder, value: Value) -> Value {
        // Bitcast f64 to i64, then add DOUBLE_ENCODE_OFFSET
        let as_bits = builder.ins().bitcast(types::I64, MemFlags::new(), value);
        let offset = builder
            .ins()
            .iconst(types::I64, DOUBLE_ENCODE_OFFSET as i64);
        builder.ins().iadd(as_bits, offset)
    }

    pub fn make_bool(&self, builder: &mut FunctionBuilder, value: Value) -> Value {
        // Select between BOOLEAN_TRUE and BOOLEAN_FALSE
        let true_val = builder.ins().iconst(types::I64, BOOLEAN_TRUE as i64);
        let false_val = builder.ins().iconst(types::I64, BOOLEAN_FALSE as i64);
        builder.ins().select(value, true_val, false_val)
    }

    pub fn make_none(&self, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, NULL as i64)
    }

    pub fn make_list(&self, builder: &mut FunctionBuilder, ptr: Value) -> Value {
        // Tag the pointer with LIST_POINTER_TAG: ptr | LIST_POINTER_TAG
        let list_tag = builder.ins().iconst(types::I64, TUPLE_POINTER_TAG as i64);
        builder.ins().bor(ptr, list_tag)
    }

    pub fn make_string(&self, builder: &mut FunctionBuilder, ptr: Value) -> Value {
        // Tag the pointer with STRING_POINTER_TAG: ptr | STRING_POINTER_TAG
        let string_tag = builder.ins().iconst(types::I64, STRING_POINTER_TAG as i64);
        builder.ins().bor(ptr, string_tag)
    }

    pub fn extract_pointer(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // Remove the pointer tag: var & !POINTER_TAG_MASK
        let tag_mask = builder.ins().iconst(types::I64, POINTER_TAG_MASK as i64);
        let inverted_mask = builder.ins().bnot(tag_mask);
        builder.ins().band(var, inverted_mask)
    }

    // Scalar Operations
    pub fn emit_int_add(&self, builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
        let a = self.extract_int(builder, lhs);
        let b = self.extract_int(builder, rhs);
        let sum = builder.ins().iadd(a, b);
        self.make_int(builder, sum)
    }

    pub fn emit_double_add(&self, builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
        let a = self.extract_double(builder, lhs);
        let b = self.extract_double(builder, rhs);
        let sum = builder.ins().fadd(a, b);
        self.make_double(builder, sum)
    }

    pub fn emit_int_sub(&self, builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
        let a = self.extract_int(builder, lhs);
        let b = self.extract_int(builder, rhs);
        let diff = builder.ins().isub(a, b);
        self.make_int(builder, diff)
    }

    pub fn emit_double_sub(&self, builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
        let a = self.extract_double(builder, lhs);
        let b = self.extract_double(builder, rhs);
        let diff = builder.ins().fsub(a, b);
        self.make_double(builder, diff)
    }

    pub fn emit_int_lt(&self, builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
        let a = self.extract_int(builder, lhs);
        let b = self.extract_int(builder, rhs);
        let lt = builder.ins().icmp(IntCC::SignedLessThan, a, b);
        self.make_bool(builder, lt)
    }

    pub fn emit_double_lt(&self, builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
        let a = self.extract_double(builder, lhs);
        let b = self.extract_double(builder, rhs);
        let lt = builder.ins().fcmp(FloatCC::LessThan, a, b);
        self.make_bool(builder, lt)
    }

    /// Emit arithmetic addition with proper type coercion
    /// int + int = int, otherwise coerce to double
    pub fn emit_arithmetic_add(
        &self,
        builder: &mut FunctionBuilder,
        lhs: Value,
        rhs: Value,
    ) -> Value {
        // Check if both operands are integers
        let lhs_is_int = self.is_int(builder, lhs);
        let rhs_is_int = self.is_int(builder, rhs);
        let both_ints = builder.ins().band(lhs_is_int, rhs_is_int);

        // Create blocks for int and float paths
        let int_block = builder.create_block();
        let float_block = builder.create_block();
        let merge_block = builder.create_block();

        // Add block parameter for the result
        builder.append_block_param(merge_block, types::I64);

        // Branch based on types
        builder
            .ins()
            .brif(both_ints, int_block, &[], float_block, &[]);

        // Integer path: both are ints, use int arithmetic
        builder.switch_to_block(int_block);
        builder.seal_block(int_block);
        let int_result = self.emit_int_add(builder, lhs, rhs);
        builder.ins().jump(merge_block, [int_result.into()].iter());

        // Float path: at least one is float, coerce both and use float arithmetic
        builder.switch_to_block(float_block);
        builder.seal_block(float_block);
        let lhs_float = self.coerce_to_double(builder, lhs);
        let rhs_float = self.coerce_to_double(builder, rhs);
        let float_result = self.emit_double_add(builder, lhs_float, rhs_float);
        builder
            .ins()
            .jump(merge_block, [float_result.into()].iter());

        // Merge point
        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);

        builder.block_params(merge_block)[0]
    }

    /// Emit arithmetic subtraction with proper type coercion
    /// int - int = int, otherwise coerce to double
    pub fn emit_arithmetic_sub(
        &self,
        builder: &mut FunctionBuilder,
        lhs: Value,
        rhs: Value,
    ) -> Value {
        // Check if both operands are integers
        let lhs_is_int = self.is_int(builder, lhs);
        let rhs_is_int = self.is_int(builder, rhs);
        let both_ints = builder.ins().band(lhs_is_int, rhs_is_int);

        // Create blocks for int and float paths
        let int_block = builder.create_block();
        let float_block = builder.create_block();
        let merge_block = builder.create_block();

        // Add block parameter for the result
        builder.append_block_param(merge_block, types::I64);

        // Branch based on types
        builder
            .ins()
            .brif(both_ints, int_block, &[], float_block, &[]);

        // Integer path: both are ints, use int arithmetic
        builder.switch_to_block(int_block);
        builder.seal_block(int_block);
        let int_result = self.emit_int_sub(builder, lhs, rhs);
        builder.ins().jump(merge_block, [int_result.into()].iter());

        // Float path: at least one is float, coerce both and use float arithmetic
        builder.switch_to_block(float_block);
        builder.seal_block(float_block);
        let lhs_float = self.coerce_to_double(builder, lhs);
        let rhs_float = self.coerce_to_double(builder, rhs);
        let float_result = self.emit_double_sub(builder, lhs_float, rhs_float);
        builder
            .ins()
            .jump(merge_block, [float_result.into()].iter());

        // Merge point
        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);

        builder.block_params(merge_block)[0]
    }

    /// Emit arithmetic less-than comparison with proper type coercion
    /// Always returns a boolean
    pub fn emit_arithmetic_lt(
        &self,
        builder: &mut FunctionBuilder,
        lhs: Value,
        rhs: Value,
    ) -> Value {
        // Check if both operands are integers
        let lhs_is_int = self.is_int(builder, lhs);
        let rhs_is_int = self.is_int(builder, rhs);
        let both_ints = builder.ins().band(lhs_is_int, rhs_is_int);

        // Create blocks for int and float paths
        let int_block = builder.create_block();
        let float_block = builder.create_block();
        let merge_block = builder.create_block();

        // Add block parameter for the result
        builder.append_block_param(merge_block, types::I64);

        // Branch based on types
        builder
            .ins()
            .brif(both_ints, int_block, &[], float_block, &[]);

        // Integer path: both are ints, use int comparison
        builder.switch_to_block(int_block);
        builder.seal_block(int_block);
        let int_result = self.emit_int_lt(builder, lhs, rhs);
        builder.ins().jump(merge_block, [int_result.into()].iter());

        // Float path: at least one is float, coerce both and use float comparison
        builder.switch_to_block(float_block);
        builder.seal_block(float_block);
        let lhs_float = self.coerce_to_double(builder, lhs);
        let rhs_float = self.coerce_to_double(builder, rhs);
        let float_result = self.emit_double_lt(builder, lhs_float, rhs_float);
        builder
            .ins()
            .jump(merge_block, [float_result.into()].iter());

        // Merge point
        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);

        builder.block_params(merge_block)[0]
    }

    // Generate the missing arithmetic methods using the macros defined above
    impl_arithmetic_binop!(emit_arithmetic_mul, imul, fmul);
    impl_arithmetic_binop!(emit_arithmetic_div, sdiv, fdiv);
    // Manual implementation for modulo since floating-point remainder is special
    pub fn emit_arithmetic_mod(
        &self,
        builder: &mut FunctionBuilder,
        lhs: Value,
        rhs: Value,
    ) -> Value {
        // Check if both operands are integers
        let lhs_is_int = self.is_int(builder, lhs);
        let rhs_is_int = self.is_int(builder, rhs);
        let both_ints = builder.ins().band(lhs_is_int, rhs_is_int);

        // Create blocks for int and float paths
        let int_block = builder.create_block();
        let float_block = builder.create_block();
        let merge_block = builder.create_block();

        // Add block parameter for the result
        builder.append_block_param(merge_block, types::I64);

        // Branch based on types
        builder
            .ins()
            .brif(both_ints, int_block, &[], float_block, &[]);

        // Integer path: both are ints, use integer remainder
        builder.switch_to_block(int_block);
        builder.seal_block(int_block);
        let lhs_int = self.extract_int(builder, lhs);
        let rhs_int = self.extract_int(builder, rhs);
        let int_result = builder.ins().srem(lhs_int, rhs_int);
        let int_var = self.make_int(builder, int_result);
        builder.ins().jump(merge_block, [int_var.into()].iter());

        // Float path: for simplicity, convert to integers for modulo
        // (In a real implementation, you might want floating-point modulo)
        builder.switch_to_block(float_block);
        builder.seal_block(float_block);
        let lhs_double = self.coerce_to_double(builder, lhs);
        let rhs_double = self.coerce_to_double(builder, rhs);
        let lhs_f64 = self.extract_double(builder, lhs_double);
        let rhs_f64 = self.extract_double(builder, rhs_double);
        // Convert to integers for modulo operation
        let lhs_int = builder.ins().fcvt_to_sint(types::I32, lhs_f64);
        let rhs_int = builder.ins().fcvt_to_sint(types::I32, rhs_f64);
        let float_result = builder.ins().srem(lhs_int, rhs_int);
        let float_result_i64 = builder.ins().sextend(types::I64, float_result);
        let float_var = self.make_int(builder, float_result_i64);
        builder.ins().jump(merge_block, [float_var.into()].iter());

        // Merge point
        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);

        builder.block_params(merge_block)[0]
    }

    // Generate the missing comparison methods
    impl_comparison_binop!(emit_arithmetic_le, SignedLessThanOrEqual, LessThanOrEqual);
    impl_comparison_binop!(emit_arithmetic_gt, SignedGreaterThan, GreaterThan);
    impl_comparison_binop!(
        emit_arithmetic_ge,
        SignedGreaterThanOrEqual,
        GreaterThanOrEqual
    );
    impl_comparison_binop!(emit_arithmetic_eq, Equal, Equal);
    impl_comparison_binop!(emit_arithmetic_ne, NotEqual, NotEqual);

    /// Logical AND operation
    pub fn emit_logical_and(&self, builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
        let lhs_truthy = self.emit_is_truthy(builder, lhs);
        let rhs_truthy = self.emit_is_truthy(builder, rhs);
        let result = builder.ins().band(lhs_truthy, rhs_truthy);
        let result_i64 = builder.ins().uextend(types::I64, result);
        self.make_bool(builder, result_i64)
    }

    /// Logical OR operation
    pub fn emit_logical_or(&self, builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
        let lhs_truthy = self.emit_is_truthy(builder, lhs);
        let rhs_truthy = self.emit_is_truthy(builder, rhs);
        let result = builder.ins().bor(lhs_truthy, rhs_truthy);
        let result_i64 = builder.ins().uextend(types::I64, result);
        self.make_bool(builder, result_i64)
    }

    /// Logical NOT operation
    pub fn emit_logical_not(&self, builder: &mut FunctionBuilder, value: Value) -> Value {
        let is_truthy = self.emit_is_truthy(builder, value);
        let result = builder.ins().bxor_imm(is_truthy, 1); // XOR with 1 to flip the bit
        let result_i64 = builder.ins().uextend(types::I64, result);
        self.make_bool(builder, result_i64)
    }

    /// Coerce a Var to double - if it's an int, convert to double; if already double, pass through
    pub fn coerce_to_double(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        let is_int = self.is_int(builder, var);

        // Create blocks for int conversion and passthrough
        let convert_block = builder.create_block();
        let passthrough_block = builder.create_block();
        let merge_block = builder.create_block();

        // Add block parameter for the result
        builder.append_block_param(merge_block, types::I64);

        // Branch based on type
        builder
            .ins()
            .brif(is_int, convert_block, &[], passthrough_block, &[]);

        // Convert int to double
        builder.switch_to_block(convert_block);
        builder.seal_block(convert_block);
        let int_val = self.extract_int(builder, var);
        let float_val = builder.ins().fcvt_from_sint(types::F64, int_val);
        let double_var = self.make_double(builder, float_val);
        builder.ins().jump(merge_block, [double_var.into()].iter());

        // Pass through (assume it's already a double)
        builder.switch_to_block(passthrough_block);
        builder.seal_block(passthrough_block);
        builder.ins().jump(merge_block, [var.into()].iter());

        // Merge point
        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);

        builder.block_params(merge_block)[0]
    }

    /// Check if a Var is truthy (not 0, 0.0, or false)
    pub fn is_truthy(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // Check for integer 0
        let int_zero = builder
            .ins()
            .iconst(types::I64, Var::int(0).as_u64() as i64);
        let is_int_zero = builder.ins().icmp(IntCC::Equal, var, int_zero);

        // Check for float 0.0
        let float_zero = builder
            .ins()
            .iconst(types::I64, Var::float(0.0).as_u64() as i64);
        let is_float_zero = builder.ins().icmp(IntCC::Equal, var, float_zero);

        // Check for false
        let bool_false = builder
            .ins()
            .iconst(types::I64, Var::bool(false).as_u64() as i64);
        let is_bool_false = builder.ins().icmp(IntCC::Equal, var, bool_false);

        // Var is falsy if it's any of these three values
        let is_falsy_1 = builder.ins().bor(is_int_zero, is_float_zero);
        let is_falsy = builder.ins().bor(is_falsy_1, is_bool_false);

        // Return the logical negation (truthy = !falsy)
        // Since is_falsy is already an i1, we can use icmp with 0 to negate it
        let zero_i1 = builder.ins().iconst(types::I8, 0);
        builder.ins().icmp(IntCC::Equal, is_falsy, zero_i1)
    }

    pub fn emit_int_equals(&self, builder: &mut FunctionBuilder, lhs: Value, rhs: Value) -> Value {
        // For integers, we can just compare the raw Var values directly!
        let eq = builder.ins().icmp(IntCC::Equal, lhs, rhs);
        self.make_bool(builder, eq)
    }

    pub fn emit_double_equals(
        &self,
        builder: &mut FunctionBuilder,
        lhs: Value,
        rhs: Value,
    ) -> Value {
        let a = self.extract_double(builder, lhs);
        let b = self.extract_double(builder, rhs);
        let eq = builder.ins().fcmp(FloatCC::Equal, a, b);
        self.make_bool(builder, eq)
    }

    /// Check if a Var is a closure
    pub fn is_closure(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // Check if the tag matches CLOSURE_POINTER_TAG
        let tag_mask = builder
            .ins()
            .iconst(types::I64, crate::var::POINTER_TAG_MASK as i64);
        let closure_tag = builder
            .ins()
            .iconst(types::I64, crate::var::CLOSURE_POINTER_TAG as i64);
        let min_pointer = builder
            .ins()
            .iconst(types::I64, crate::var::MIN_POINTER as i64);

        // Check if low 2 bits are 00 (pointer)
        let low_bits_mask = builder.ins().iconst(types::I64, 0x03);
        let low_bits = builder.ins().band(var, low_bits_mask);
        let zero = builder.ins().iconst(types::I64, 0);
        let is_pointer = builder.ins().icmp(IntCC::Equal, low_bits, zero);

        // Check if >= MIN_POINTER
        let is_valid_ptr = builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, var, min_pointer);

        // Check if tag matches closure tag
        let var_tag = builder.ins().band(var, tag_mask);
        let is_closure_tag = builder.ins().icmp(IntCC::Equal, var_tag, closure_tag);

        // All conditions must be true
        let ptr_and_valid = builder.ins().band(is_pointer, is_valid_ptr);
        builder.ins().band(ptr_and_valid, is_closure_tag)
    }

    /// Extract closure pointer from a Var (assumes it's been checked to be a closure)
    pub fn extract_closure_ptr(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // Remove the closure tag to get the raw pointer
        let tag_mask = builder
            .ins()
            .iconst(types::I64, crate::var::POINTER_TAG_MASK as i64);
        let inverted_mask = builder.ins().bnot(tag_mask);
        builder.ins().band(var, inverted_mask)
    }

    /// Call a closure with the given arguments
    pub fn call_closure(
        &self,
        builder: &mut FunctionBuilder,
        closure_ptr: Value,
        args_ptr: Value,
        arg_count: u32,
    ) -> Value {
        // Load the function pointer from the closure struct
        // LispClosure layout: [func_ptr: *const u8][arity: u32][captured_env: u64]
        let func_ptr = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), closure_ptr, 0);

        // Load the captured environment from offset 12 (8 bytes func_ptr + 4 bytes arity)
        let captured_env = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), closure_ptr, 12);

        // Create the function signature: fn(args: *const Var, arg_count: u32, captured_env: u64) -> u64
        // For now, use the native calling convention on this platform
        let native_call_conv = cranelift_native::builder()
            .unwrap()
            .finish(cranelift::prelude::settings::Flags::new(
                cranelift::prelude::settings::builder(),
            ))
            .unwrap()
            .default_call_conv();
        let mut sig = Signature::new(native_call_conv);
        sig.params.push(AbiParam::new(types::I64)); // args pointer
        sig.params.push(AbiParam::new(types::I32)); // arg count
        sig.params.push(AbiParam::new(types::I64)); // captured environment
        sig.returns.push(AbiParam::new(types::I64)); // return value

        let sig_ref = builder.import_signature(sig);

        // Prepare arguments
        let arg_count_val = builder.ins().iconst(types::I32, arg_count as i64);

        // Make the indirect call
        let call_inst = builder.ins().call_indirect(
            sig_ref,
            func_ptr,
            &[args_ptr, arg_count_val, captured_env],
        );
        builder.inst_results(call_inst)[0]
    }

    /// Convert a Var to a boolean value (0 = false, non-zero = true)
    pub fn emit_is_truthy(&self, builder: &mut FunctionBuilder, var: Value) -> Value {
        // For now, implement a simple truthiness check
        // This is a simplified version - we need to check different Var types

        // Extract type tag from upper 8 bits
        let tag = builder.ins().ushr_imm(var, 56);

        // Create blocks for different types
        let int_block = builder.create_block();
        let bool_block = builder.create_block();
        let merge_block = builder.create_block();
        builder.append_block_param(merge_block, types::I32);

        // Simple type dispatch (just for int and bool for now)
        let is_int = builder.ins().icmp_imm(IntCC::Equal, tag, 0);
        builder.ins().brif(is_int, int_block, &[], bool_block, &[]);

        // Int block: check if non-zero
        builder.switch_to_block(int_block);
        builder.seal_block(int_block);
        let int_val = builder.ins().band_imm(var, 0xFFFFFFFF);
        let int_nonzero = builder.ins().icmp_imm(IntCC::NotEqual, int_val, 0);
        let int_result = builder.ins().uextend(types::I32, int_nonzero);
        builder.ins().jump(merge_block, [int_result.into()].iter());

        // Bool/other block: assume truthy unless zero
        builder.switch_to_block(bool_block);
        builder.seal_block(bool_block);
        let other_nonzero = builder.ins().icmp_imm(IntCC::NotEqual, var, 0);
        let other_result = builder.ins().uextend(types::I32, other_nonzero);
        builder
            .ins()
            .jump(merge_block, [other_result.into()].iter());

        // Merge results
        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);

        builder.block_params(merge_block)[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::var::{Var, VarType};
    use std::mem;

    type BinaryVarFunc = unsafe extern "C" fn(u64, u64) -> u64;

    #[test]
    fn test_jit_int_construction() {
        let mut jit = VarJIT::new();

        // Create a function that takes an i32 and returns it as a Var
        let func_id = jit.create_binary_function("make_int_var");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block); // Seal the block for optimization

            let params = builder.block_params(entry_block);
            let raw_int = params[0]; // First parameter as raw i64

            // Convert the raw i64 to a properly NaN-boxed Var
            let var_result = var_builder.make_int(builder, raw_int);
            builder.ins().return_(&[var_result]);
        });
        let func: unsafe extern "C" fn(i64, i64) -> u64 = unsafe { mem::transmute(code_ptr) };

        // Test: create Var for integer 42
        let result = unsafe { func(42, 0) };
        let var_result = Var::from_u64(result);

        assert_eq!(var_result.get_type(), VarType::I32);
        assert_eq!(var_result.as_int(), Some(42));
    }

    #[test]
    fn test_jit_int_addition() {
        let mut jit = VarJIT::new();

        let func_id = jit.create_binary_function("add_ints");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let params = builder.block_params(entry_block);
            let lhs = params[0];
            let rhs = params[1];

            // Add two Var integers using cranelift
            let result = var_builder.emit_int_add(builder, lhs, rhs);
            builder.ins().return_(&[result]);
        });
        let func: BinaryVarFunc = unsafe { mem::transmute(code_ptr) };

        // Test: 123 + 456 = 579
        let var_a = Var::int(123);
        let var_b = Var::int(456);

        let result = unsafe { func(var_a.as_u64(), var_b.as_u64()) };
        let var_result = Var::from_u64(result);

        assert_eq!(var_result.get_type(), VarType::I32);
        assert_eq!(var_result.as_int(), Some(579));
    }

    #[test]
    fn test_jit_double_addition() {
        let mut jit = VarJIT::new();

        let func_id = jit.create_binary_function("add_doubles");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let params = builder.block_params(entry_block);
            let lhs = params[0];
            let rhs = params[1];

            let result = var_builder.emit_double_add(builder, lhs, rhs);
            builder.ins().return_(&[result]);
        });
        let func: BinaryVarFunc = unsafe { mem::transmute(code_ptr) };

        // Test: 1.5 + 2.5 = 4.0
        let var_a = Var::float(1.5);
        let var_b = Var::float(2.5);

        let result = unsafe { func(var_a.as_u64(), var_b.as_u64()) };
        let var_result = Var::from_u64(result);

        assert_eq!(var_result.get_type(), VarType::F64);
        assert!((var_result.as_double().unwrap() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jit_type_checking() {
        let mut jit = VarJIT::new();

        let func_id = jit.create_binary_function("check_int_type");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let params = builder.block_params(entry_block);
            let var_value = params[0];

            // Check if the value is an integer
            let is_int = var_builder.is_int(builder, var_value);
            let result = var_builder.make_bool(builder, is_int);
            builder.ins().return_(&[result]);
        });
        let func: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { mem::transmute(code_ptr) };

        // Test with integer
        let int_var = Var::int(42);
        let result = unsafe { func(int_var.as_u64(), 0) };
        let bool_result = Var::from_u64(result);
        assert_eq!(bool_result.as_bool(), Some(true));

        // Test with double
        let double_var = Var::float(3.14);
        let result = unsafe { func(double_var.as_u64(), 0) };
        let bool_result = Var::from_u64(result);
        assert_eq!(bool_result.as_bool(), Some(false));
    }

    #[test]
    fn test_round_trip_construction() {
        let mut jit = VarJIT::new();

        // Function that constructs various types and returns them
        let func_id = jit.create_binary_function("construct_types");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Construct different types based on input parameter
            let params = builder.block_params(entry_block);
            let selector = params[0];

            let int_block = builder.create_block();
            let double_block = builder.create_block();
            let bool_block = builder.create_block();
            let default_block = builder.create_block();

            // Use simple conditional branches instead of switch
            let zero = builder.ins().iconst(types::I64, 0);
            let one = builder.ins().iconst(types::I64, 1);
            let two = builder.ins().iconst(types::I64, 2);

            // if (selector == 0) goto int_block else goto check_double
            let check_double_block = builder.create_block();
            let is_zero = builder.ins().icmp(IntCC::Equal, selector, zero);
            builder
                .ins()
                .brif(is_zero, int_block, &[], check_double_block, &[]);

            // check_double: if (selector == 1) goto double_block else goto check_bool
            builder.switch_to_block(check_double_block);
            builder.seal_block(check_double_block);
            let check_bool_block = builder.create_block();
            let is_one = builder.ins().icmp(IntCC::Equal, selector, one);
            builder
                .ins()
                .brif(is_one, double_block, &[], check_bool_block, &[]);

            // check_bool: if (selector == 2) goto bool_block else goto default
            builder.switch_to_block(check_bool_block);
            builder.seal_block(check_bool_block);
            let is_two = builder.ins().icmp(IntCC::Equal, selector, two);
            builder
                .ins()
                .brif(is_two, bool_block, &[], default_block, &[]);

            // Integer case
            builder.switch_to_block(int_block);
            builder.seal_block(int_block);
            let int_val = builder.ins().iconst(types::I64, 123);
            let int_var = var_builder.make_int(builder, int_val);
            builder.ins().return_(&[int_var]);

            // Double case
            builder.switch_to_block(double_block);
            builder.seal_block(double_block);
            let double_val = builder.ins().f64const(3.14159);
            let double_var = var_builder.make_double(builder, double_val);
            builder.ins().return_(&[double_var]);

            // Bool case
            builder.switch_to_block(bool_block);
            builder.seal_block(bool_block);
            let bool_val = builder.ins().iconst(types::I8, 1);
            let bool_var = var_builder.make_bool(builder, bool_val);
            builder.ins().return_(&[bool_var]);

            // Default case - return none
            builder.switch_to_block(default_block);
            builder.seal_block(default_block);
            let none_var = var_builder.make_none(builder);
            builder.ins().return_(&[none_var]);
        });
        let func: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { mem::transmute(code_ptr) };

        // Test integer construction
        let int_result = unsafe { func(0, 0) };
        let int_var = Var::from_u64(int_result);
        assert_eq!(int_var.get_type(), VarType::I32);
        assert_eq!(int_var.as_int(), Some(123));

        // Test double construction
        let double_result = unsafe { func(1, 0) };
        let double_var = Var::from_u64(double_result);
        assert_eq!(double_var.get_type(), VarType::F64);
        assert!((double_var.as_double().unwrap() - 3.14159).abs() < 1e-10);

        // Test bool construction
        let bool_result = unsafe { func(2, 0) };
        let bool_var = Var::from_u64(bool_result);
        assert_eq!(bool_var.get_type(), VarType::Bool);
        assert_eq!(bool_var.as_bool(), Some(true));
    }

    #[test]
    fn test_jit_bool_operations() {
        let mut jit = VarJIT::new();

        // Test bool construction and type checking
        let func_id = jit.create_binary_function("bool_ops");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let params = builder.block_params(entry_block);
            let input_bool = params[0]; // 0 or 1

            // Convert i64 to cranelift bool, then to Var bool
            let zero = builder.ins().iconst(types::I64, 0);
            let as_bool = builder.ins().icmp(IntCC::NotEqual, input_bool, zero);
            let var_bool = var_builder.make_bool(builder, as_bool);

            // Test if it's recognized as a bool type
            let is_bool_result = var_builder.is_bool(builder, var_bool);
            let result = var_builder.make_bool(builder, is_bool_result);

            builder.ins().return_(&[result]);
        });

        let func: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { mem::transmute(code_ptr) };

        // Test with true
        let result = unsafe { func(1, 0) };
        let bool_result = Var::from_u64(result);
        assert_eq!(bool_result.as_bool(), Some(true));

        // Test with false
        let result = unsafe { func(0, 0) };
        let bool_result = Var::from_u64(result);
        assert_eq!(bool_result.as_bool(), Some(true)); // Should recognize false as bool type too
    }

    #[test]
    fn test_jit_bool_extraction() {
        let mut jit = VarJIT::new();

        // Test extracting boolean values from Vars
        let func_id = jit.create_binary_function("extract_bool");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let params = builder.block_params(entry_block);
            let var_value = params[0];

            // Extract the boolean value and return it as an integer (0 or 1)
            let bool_val = var_builder.extract_bool(builder, var_value);
            let zero = builder.ins().iconst(types::I64, 0);
            let one = builder.ins().iconst(types::I64, 1);
            let result = builder.ins().select(bool_val, one, zero);

            builder.ins().return_(&[result]);
        });

        let func: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { mem::transmute(code_ptr) };

        // Test extracting true
        let true_var = Var::bool(true);
        let result = unsafe { func(true_var.as_u64(), 0) };
        assert_eq!(result, 1);

        // Test extracting false
        let false_var = Var::bool(false);
        let result = unsafe { func(false_var.as_u64(), 0) };
        assert_eq!(result, 0);
    }

    #[test]
    fn test_jit_symbol_operations() {
        let mut jit = VarJIT::new();

        // Test symbol construction and round-trip
        let func_id = jit.create_binary_function("symbol_ops");
        let code_ptr = jit.compile_function(func_id, |builder, _var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let params = builder.block_params(entry_block);
            let symbol_id = params[0]; // Raw symbol ID

            // Create a symbol Var by adding the SYMBOL_TAG
            let symbol_tag = builder
                .ins()
                .iconst(types::I64, crate::var::SYMBOL_TAG as i64);
            let symbol_var = builder.ins().bor(symbol_id, symbol_tag);

            builder.ins().return_(&[symbol_var]);
        });

        let func: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { mem::transmute(code_ptr) };

        // Test with symbol ID 12345
        let result = unsafe { func(12345, 0) };
        let symbol_var = Var::from_u64(result);

        assert_eq!(symbol_var.get_type(), VarType::Symbol);
        assert_eq!(symbol_var.as_symbol(), Some(12345));
    }

    #[test]
    fn test_jit_none_handling() {
        let mut jit = VarJIT::new();

        // Test none value creation and detection
        let func_id = jit.create_binary_function("none_ops");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Create a none value
            let none_var = var_builder.make_none(builder);

            builder.ins().return_(&[none_var]);
        });

        let func: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { mem::transmute(code_ptr) };

        let result = unsafe { func(0, 0) };
        let none_var = Var::from_u64(result);

        assert_eq!(none_var.get_type(), VarType::None);
        assert!(none_var.is_none());
    }

    #[test]
    fn test_jit_comprehensive_type_checking() {
        let mut jit = VarJIT::new();

        // Test type checking for all immediate types
        let func_id = jit.create_binary_function("comprehensive_type_check");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let params = builder.block_params(entry_block);
            let var_value = params[0];
            let type_to_check = params[1]; // 0=int, 1=double, 2=bool

            let zero = builder.ins().iconst(types::I64, 0);
            let one = builder.ins().iconst(types::I64, 1);
            let two = builder.ins().iconst(types::I64, 2);

            let int_block = builder.create_block();
            let double_block = builder.create_block();
            let bool_block = builder.create_block();
            let default_block = builder.create_block();

            // Chain of conditionals for type checking
            let check_double = builder.create_block();
            let check_bool = builder.create_block();

            let is_zero = builder.ins().icmp(IntCC::Equal, type_to_check, zero);
            builder
                .ins()
                .brif(is_zero, int_block, &[], check_double, &[]);

            builder.switch_to_block(check_double);
            builder.seal_block(check_double);
            let is_one = builder.ins().icmp(IntCC::Equal, type_to_check, one);
            builder
                .ins()
                .brif(is_one, double_block, &[], check_bool, &[]);

            builder.switch_to_block(check_bool);
            builder.seal_block(check_bool);
            let is_two = builder.ins().icmp(IntCC::Equal, type_to_check, two);
            builder
                .ins()
                .brif(is_two, bool_block, &[], default_block, &[]);

            // Check if it's an int
            builder.switch_to_block(int_block);
            builder.seal_block(int_block);
            let is_int_result = var_builder.is_int(builder, var_value);
            let int_result = var_builder.make_bool(builder, is_int_result);
            builder.ins().return_(&[int_result]);

            // Check if it's a double
            builder.switch_to_block(double_block);
            builder.seal_block(double_block);
            let is_double_result = var_builder.is_double(builder, var_value);
            let double_result = var_builder.make_bool(builder, is_double_result);
            builder.ins().return_(&[double_result]);

            // Check if it's a bool
            builder.switch_to_block(bool_block);
            builder.seal_block(bool_block);
            let is_bool_result = var_builder.is_bool(builder, var_value);
            let bool_result = var_builder.make_bool(builder, is_bool_result);
            builder.ins().return_(&[bool_result]);

            // Default - return false
            builder.switch_to_block(default_block);
            builder.seal_block(default_block);
            let false_val = builder.ins().iconst(types::I8, 0);
            let false_result = var_builder.make_bool(builder, false_val);
            builder.ins().return_(&[false_result]);
        });

        let func: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { mem::transmute(code_ptr) };

        // Test various types
        let int_var = Var::int(42);
        let double_var = Var::float(3.14);
        let bool_var = Var::bool(true);
        let none_var = Var::none();
        let symbol_var = Var::symbol(12345);

        // Test int detection
        let result = unsafe { func(int_var.as_u64(), 0) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(true));

        let result = unsafe { func(double_var.as_u64(), 0) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(false));

        // Test double detection
        let result = unsafe { func(double_var.as_u64(), 1) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(true));

        let result = unsafe { func(int_var.as_u64(), 1) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(false));

        // Test bool detection
        let result = unsafe { func(bool_var.as_u64(), 2) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(true));

        let result = unsafe { func(none_var.as_u64(), 2) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(false));

        let result = unsafe { func(symbol_var.as_u64(), 2) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(false));
    }

    #[test]
    fn test_jit_list_operations() {
        let mut jit = VarJIT::new();

        // Test list construction and type checking
        let func_id = jit.create_binary_function("list_ops");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let params = builder.block_params(entry_block);
            let raw_ptr = params[0]; // Raw pointer as input

            // Create a list Var by tagging the pointer
            let list_var = var_builder.make_list(builder, raw_ptr);

            // Test if it's recognized as a list type
            let is_list_result = var_builder.is_list(builder, list_var);
            let result = var_builder.make_bool(builder, is_list_result);

            builder.ins().return_(&[result]);
        });

        let func: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { mem::transmute(code_ptr) };

        // Create a real list in Rust and extract its pointer
        let rust_list = Var::empty_tuple();
        let raw_ptr_value = rust_list.as_u64() & !POINTER_TAG_MASK;

        // Test with the raw pointer - should be recognized as list
        let result = unsafe { func(raw_ptr_value, 0) };
        let bool_result = Var::from_u64(result);
        assert_eq!(bool_result.as_bool(), Some(true));

        // Test with a non-pointer value - should NOT be recognized as list
        let result = unsafe { func(42, 0) };
        let bool_result = Var::from_u64(result);
        assert_eq!(bool_result.as_bool(), Some(false));
    }

    #[test]
    fn test_jit_string_operations() {
        let mut jit = VarJIT::new();

        // Test string construction and type checking
        let func_id = jit.create_binary_function("string_ops");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let params = builder.block_params(entry_block);
            let raw_ptr = params[0]; // Raw pointer as input

            // Create a string Var by tagging the pointer
            let string_var = var_builder.make_string(builder, raw_ptr);

            // Test if it's recognized as a string type
            let is_string_result = var_builder.is_string(builder, string_var);
            let result = var_builder.make_bool(builder, is_string_result);

            builder.ins().return_(&[result]);
        });

        let func: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { mem::transmute(code_ptr) };

        // Create a real string in Rust and extract its pointer
        let rust_string = Var::string("test");
        let raw_ptr_value = rust_string.as_u64() & !POINTER_TAG_MASK;

        // Test with the raw pointer - should be recognized as string
        let result = unsafe { func(raw_ptr_value, 0) };
        let bool_result = Var::from_u64(result);
        assert_eq!(bool_result.as_bool(), Some(true));

        // Test with a non-pointer value - should NOT be recognized as string
        let result = unsafe { func(123, 0) };
        let bool_result = Var::from_u64(result);
        assert_eq!(bool_result.as_bool(), Some(false));
    }

    #[test]
    fn test_jit_pointer_extraction() {
        let mut jit = VarJIT::new();

        // Test pointer extraction from tagged values
        let func_id = jit.create_binary_function("extract_ptr");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let params = builder.block_params(entry_block);
            let tagged_var = params[0]; // Tagged pointer as input

            // Extract the raw pointer
            let raw_ptr = var_builder.extract_pointer(builder, tagged_var);

            builder.ins().return_(&[raw_ptr]);
        });

        let func: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { mem::transmute(code_ptr) };

        // Create a list and test pointer extraction
        let rust_list = Var::empty_tuple();
        let expected_ptr = rust_list.as_u64() & !POINTER_TAG_MASK;

        let result = unsafe { func(rust_list.as_u64(), 0) };
        assert_eq!(result, expected_ptr);

        // Create a string and test pointer extraction
        let rust_string = Var::string("hello");
        let expected_ptr = rust_string.as_u64() & !POINTER_TAG_MASK;

        let result = unsafe { func(rust_string.as_u64(), 0) };
        assert_eq!(result, expected_ptr);
    }

    #[test]
    fn test_jit_comprehensive_heap_type_checking() {
        let mut jit = VarJIT::new();

        // Test type checking for heap types alongside immediate types
        let func_id = jit.create_binary_function("comprehensive_heap_type_check");
        let code_ptr = jit.compile_function(func_id, |builder, var_builder| {
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let params = builder.block_params(entry_block);
            let var_value = params[0];
            let type_to_check = params[1]; // 0=int, 1=double, 2=bool, 3=list, 4=string

            let zero = builder.ins().iconst(types::I64, 0);
            let one = builder.ins().iconst(types::I64, 1);
            let two = builder.ins().iconst(types::I64, 2);
            let three = builder.ins().iconst(types::I64, 3);
            let four = builder.ins().iconst(types::I64, 4);

            let int_block = builder.create_block();
            let double_block = builder.create_block();
            let bool_block = builder.create_block();
            let list_block = builder.create_block();
            let string_block = builder.create_block();
            let default_block = builder.create_block();

            // Chain of conditionals for type checking
            let check_double = builder.create_block();
            let check_bool = builder.create_block();
            let check_list = builder.create_block();
            let check_string = builder.create_block();

            let is_zero = builder.ins().icmp(IntCC::Equal, type_to_check, zero);
            builder
                .ins()
                .brif(is_zero, int_block, &[], check_double, &[]);

            builder.switch_to_block(check_double);
            builder.seal_block(check_double);
            let is_one = builder.ins().icmp(IntCC::Equal, type_to_check, one);
            builder
                .ins()
                .brif(is_one, double_block, &[], check_bool, &[]);

            builder.switch_to_block(check_bool);
            builder.seal_block(check_bool);
            let is_two = builder.ins().icmp(IntCC::Equal, type_to_check, two);
            builder.ins().brif(is_two, bool_block, &[], check_list, &[]);

            builder.switch_to_block(check_list);
            builder.seal_block(check_list);
            let is_three = builder.ins().icmp(IntCC::Equal, type_to_check, three);
            builder
                .ins()
                .brif(is_three, list_block, &[], check_string, &[]);

            builder.switch_to_block(check_string);
            builder.seal_block(check_string);
            let is_four = builder.ins().icmp(IntCC::Equal, type_to_check, four);
            builder
                .ins()
                .brif(is_four, string_block, &[], default_block, &[]);

            // Check if it's an int
            builder.switch_to_block(int_block);
            builder.seal_block(int_block);
            let is_int_result = var_builder.is_int(builder, var_value);
            let int_result = var_builder.make_bool(builder, is_int_result);
            builder.ins().return_(&[int_result]);

            // Check if it's a double
            builder.switch_to_block(double_block);
            builder.seal_block(double_block);
            let is_double_result = var_builder.is_double(builder, var_value);
            let double_result = var_builder.make_bool(builder, is_double_result);
            builder.ins().return_(&[double_result]);

            // Check if it's a bool
            builder.switch_to_block(bool_block);
            builder.seal_block(bool_block);
            let is_bool_result = var_builder.is_bool(builder, var_value);
            let bool_result = var_builder.make_bool(builder, is_bool_result);
            builder.ins().return_(&[bool_result]);

            // Check if it's a list
            builder.switch_to_block(list_block);
            builder.seal_block(list_block);
            let is_list_result = var_builder.is_list(builder, var_value);
            let list_result = var_builder.make_bool(builder, is_list_result);
            builder.ins().return_(&[list_result]);

            // Check if it's a string
            builder.switch_to_block(string_block);
            builder.seal_block(string_block);
            let is_string_result = var_builder.is_string(builder, var_value);
            let string_result = var_builder.make_bool(builder, is_string_result);
            builder.ins().return_(&[string_result]);

            // Default - return false
            builder.switch_to_block(default_block);
            builder.seal_block(default_block);
            let false_val = builder.ins().iconst(types::I8, 0);
            let false_result = var_builder.make_bool(builder, false_val);
            builder.ins().return_(&[false_result]);
        });

        let func: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { mem::transmute(code_ptr) };

        // Test various types
        let int_var = Var::int(42);
        let double_var = Var::float(3.14);
        let bool_var = Var::bool(true);
        let none_var = Var::none();
        let symbol_var = Var::symbol(12345);

        let list_var = Var::empty_tuple();
        let string_var = Var::string("test");

        // Test int detection
        let result = unsafe { func(int_var.as_u64(), 0) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(true));

        let result = unsafe { func(list_var.as_u64(), 0) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(false));

        // Test double detection
        let result = unsafe { func(double_var.as_u64(), 1) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(true));

        let result = unsafe { func(string_var.as_u64(), 1) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(false));

        // Test bool detection
        let result = unsafe { func(bool_var.as_u64(), 2) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(true));

        let result = unsafe { func(list_var.as_u64(), 2) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(false));

        // Test list detection
        let result = unsafe { func(list_var.as_u64(), 3) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(true));

        let result = unsafe { func(int_var.as_u64(), 3) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(false));

        let result = unsafe { func(string_var.as_u64(), 3) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(false));

        // Test string detection
        let result = unsafe { func(string_var.as_u64(), 4) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(true));

        let result = unsafe { func(list_var.as_u64(), 4) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(false));

        let result = unsafe { func(double_var.as_u64(), 4) };
        assert_eq!(Var::from_u64(result).as_bool(), Some(false));
    }
}
