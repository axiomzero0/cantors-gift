// python/bindings.cpp - Python bindings for cantors-gift via pybind11
//
// Exposes the full compiler stack to Python:
//   - Module / Function / Builder / standard ops
//   - All optimization passes + IterativeDriver
//   - Global Tensor Analysis (GTA) analyses
//   - Schedule / ScheduleSpace / HardwareModel / CostEstimator
//   - Codegen IR / PTX emitter / x86 emitter / backends
//   - Runtime / KernelCache / autotuner
//
// Usage from Python:
//   import cantors_gift as cg
//   m = cg.Module()
//   f = m.create_function("kernel", [cg.tensor_type([16,32], cg.DType.F32)],
//                                   [cg.tensor_type([16,64], cg.DType.F32)])
//   b = cg.Builder(f)
//   mm = b.matmul(f.args[0], ...)
//   ...
//   am = cg.AnalysisManager(m)
//   driver = cg.IterativeDriver(am)
//   driver.run(m)
//   print(cg.to_string(m))

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "cg/analysis/global_analysis.hpp"
#include "cg/analysis/shape_analysis.hpp"
#include "cg/analysis/layout_analysis.hpp"
#include "cg/analysis/dataflow_analysis.hpp"
#include "cg/analysis/lifetime_analysis.hpp"
#include "cg/analysis/arithmetic_intensity.hpp"
#include "cg/analysis/parallelism_analysis.hpp"
#include "cg/analysis/reuse_analysis.hpp"
#include "cg/analysis/global_alias_analysis.hpp"
#include "cg/analysis/global_cost.hpp"

#include "cg/autotuner/bayesian_optimizer.hpp"
#include "cg/backend/amd/amd_backend.hpp"
#include "cg/backend/cpu_backend.hpp"
#include "cg/backend/jit.hpp"
#include "cg/backend/lowering.hpp"
#include "cg/backend/nvidia_backend.hpp"
#include "cg/backend/parallel_executor.hpp"
#include "cg/backend/ptx/ptx_emitter.hpp"
#include "cg/backend/x86/x86_emitter.hpp"
#include "cg/codegen/codegen_ir.hpp"
#include "cg/cost/estimator.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/cost/hardware_profile.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/ir/printer.hpp"
#include "cg/lowering/tensor_to_codegen.hpp"
#include "cg/numerical/semantics.hpp"
#include "cg/optimization/canonicalize/algebraic.hpp"
#include "cg/optimization/canonicalize/canonicalize.hpp"
#include "cg/optimization/const_fold/const_fold.hpp"
#include "cg/optimization/const_fold/sccp.hpp"
#include "cg/optimization/dce/dce.hpp"
#include "cg/optimization/cse/cse.hpp"
#include "cg/optimization/egraph/egraph_superoptimizer.hpp"
#include "cg/optimization/fusion/fusion.hpp"
#include "cg/optimization/global_barrier.hpp"
#include "cg/optimization/iterative_driver.hpp"
#include "cg/optimization/layout/layout_opt.hpp"
#include "cg/optimization/memory/copy_elimination.hpp"
#include "cg/pipeline/matmul_pipeline.hpp"
#include "cg/analysis/unified/abstract_domain.hpp"
#include "cg/analysis/unified/fact_store.hpp"
#include "cg/analysis/unified/tensor_facts.hpp"
#include "cg/analysis/unified/unified_analyzer.hpp"
#include "cg/engine/compile_api.hpp"
#include "cg/optimization/unified/unified_passes.hpp"
#include "cg/optimization/memory/memory_planning.hpp"
#include "cg/optimization/reduction/reduction_opt.hpp"
#include "cg/optimization/shape/shape_opt.hpp"
#include "cg/optimization/specialization/specialization.hpp"
#include "cg/runtime/runtime.hpp"
#include "cg/schedule/domain.hpp"
#include "cg/schedule/schedule.hpp"
#include "cg/shape/dim_expr.hpp"
#include "cg/shape/simplifier.hpp"
#include "cg/shape/solver.hpp"
#include "cg/shape/inference.hpp"
#include "cg/vendor/dispatch.hpp"

namespace py = pybind11;
using namespace cg;

PYBIND11_MODULE(cantors_gift, m) {
    m.doc() = "cantors-gift: multi-level SSA-based tensor compiler";

    // ===================================================================
    // Core types: DType, DeviceId, MemorySpace
    // ===================================================================
    py::enum_<DType>(m, "DType")
        .value("F16", DType::F16)
        .value("BF16", DType::BF16)
        .value("F32", DType::F32)
        .value("F64", DType::F64)
        .value("I8", DType::I8)
        .value("I16", DType::I16)
        .value("I32", DType::I32)
        .value("I64", DType::I64)
        .value("U8", DType::U8)
        .value("U16", DType::U16)
        .value("U32", DType::U32)
        .value("U64", DType::U64)
        .value("BOOL", DType::BOOL)
        .value("INDEX", DType::INDEX)
        .value("CF32", DType::CF32)
        .value("CF64", DType::CF64);

    m.def("dtype_size", &dtype_size);
    m.def("dtype_name", [](DType dt) { return std::string(dtype_name(dt)); });
    m.def("is_float", &is_float);
    m.def("is_int", &is_int);
    m.def("is_signed", &is_signed);
    m.def("promote", &promote);

    py::class_<DeviceId>(m, "DeviceId")
        .def(py::init<>())
        .def_readwrite("kind", &DeviceId::kind)
        .def_readwrite("index", &DeviceId::index)
        .def_static("cpu", &DeviceId::cpu)
        .def_static("cuda", &DeviceId::cuda)
        .def_static("rocm", &DeviceId::rocm)
        .def_static("metal", &DeviceId::metal)
        .def("to_string", &DeviceId::to_string)
        .def("__repr__", [](const DeviceId& d) { return "DeviceId(" + d.to_string() + ")"; });

    py::enum_<DeviceId::Kind>(m, "DeviceKind")
        .value("CPU", DeviceId::Kind::CPU)
        .value("CUDA", DeviceId::Kind::CUDA)
        .value("ROCM", DeviceId::Kind::ROCM)
        .value("METAL", DeviceId::Kind::METAL);

    py::enum_<MemorySpace>(m, "MemorySpace")
        .value("Generic", MemorySpace::Generic)
        .value("Shared", MemorySpace::Shared)
        .value("Constant", MemorySpace::Constant)
        .value("Local", MemorySpace::Local)
        .value("GlobalHost", MemorySpace::GlobalHost);

    // ===================================================================
    // Shape system
    // ===================================================================
    // DimExpr is immutable and held via shared_ptr<const DimExpr>. We bind
    // it as a shared_ptr<DimExpr> wrapper since pybind11 doesn't handle
    // const element types well. The static factory methods return DimExprPtr
    // which we cast to non-const for Python.
    py::class_<DimExpr, std::shared_ptr<DimExpr>>(m, "DimExpr")
        .def("is_constant", py::overload_cast<>(&DimExpr::is_constant, py::const_))
        .def("is_symbol", &DimExpr::is_symbol)
        .def("is_constant_value", [](const DimExpr& e, i64 v) { return e.is_constant(v); })
        .def_static("make_constant", [](i64 v) -> std::shared_ptr<DimExpr> {
            return std::const_pointer_cast<DimExpr>(DimExpr::make_constant(v));
        })
        .def_static("make_symbol", [](u32 id, const std::string& name) -> std::shared_ptr<DimExpr> {
            return std::const_pointer_cast<DimExpr>(DimExpr::make_symbol(id, name));
        })
        .def_static("make_add", [](DimExprPtr a, DimExprPtr b) -> std::shared_ptr<DimExpr> {
            return std::const_pointer_cast<DimExpr>(DimExpr::make_add(a, b));
        })
        .def_static("make_sub", [](DimExprPtr a, DimExprPtr b) -> std::shared_ptr<DimExpr> {
            return std::const_pointer_cast<DimExpr>(DimExpr::make_sub(a, b));
        })
        .def_static("make_mul", [](DimExprPtr a, DimExprPtr b) -> std::shared_ptr<DimExpr> {
            return std::const_pointer_cast<DimExpr>(DimExpr::make_mul(a, b));
        })
        .def_static("make_floor_div", [](DimExprPtr a, DimExprPtr b) -> std::shared_ptr<DimExpr> {
            return std::const_pointer_cast<DimExpr>(DimExpr::make_floor_div(a, b));
        })
        .def_static("make_ceil_div", [](DimExprPtr a, DimExprPtr b) -> std::shared_ptr<DimExpr> {
            return std::const_pointer_cast<DimExpr>(DimExpr::make_ceil_div(a, b));
        })
        .def_static("make_mod", [](DimExprPtr a, DimExprPtr b) -> std::shared_ptr<DimExpr> {
            return std::const_pointer_cast<DimExpr>(DimExpr::make_mod(a, b));
        })
        .def_static("make_min", [](DimExprPtr a, DimExprPtr b) -> std::shared_ptr<DimExpr> {
            return std::const_pointer_cast<DimExpr>(DimExpr::make_min(a, b));
        })
        .def_static("make_max", [](DimExprPtr a, DimExprPtr b) -> std::shared_ptr<DimExpr> {
            return std::const_pointer_cast<DimExpr>(DimExpr::make_max(a, b));
        })
        .def_static("make_neg", [](DimExprPtr a) -> std::shared_ptr<DimExpr> {
            return std::const_pointer_cast<DimExpr>(DimExpr::make_neg(a));
        })
        .def("__repr__", [](const DimExpr& e) {
            if (e.is_constant()) return "DimExpr(" + std::to_string(e.value) + ")";
            if (e.is_symbol()) return "DimExpr(" + std::string(e.symbol.name) + ")";
            return std::string("DimExpr(<expr>)");
        });

    m.def("simplify_dim", &simplify_dim);

    py::class_<Shape>(m, "Shape")
        .def(py::init<>())
        .def(py::init([](std::vector<i64> dims) {
            Shape s;
            for (auto d : dims) s.dims().push_back(DimExpr::make_constant(d));
            return s;
        }))
        .def("rank", &Shape::rank)
        .def("empty", &Shape::empty)
        .def("__len__", &Shape::rank)
        .def("__getitem__", [](const Shape& s, usize i) { return s[i]; })
        .def("num_elements", &Shape::num_elements)
        .def_static("from_constants", [](std::vector<i64> dims) {
            return Shape::from_constants(std::move(dims));
        });

    // Constraint system
    py::class_<Constraint>(m, "Constraint")
        .def(py::init<>())
        .def_readwrite("kind", &Constraint::kind)
        .def_readwrite("modulus", &Constraint::modulus)
        .def_static("eq", &Constraint::eq)
        .def_static("ne", &Constraint::ne)
        .def_static("lt", &Constraint::lt)
        .def_static("le", &Constraint::le)
        .def_static("gt", &Constraint::gt)
        .def_static("ge", &Constraint::ge)
        .def_static("mod_eq", &Constraint::mod_eq)
        .def_static("mod_ne", &Constraint::mod_ne)
        .def("to_string", &Constraint::to_string)
        .def("__repr__", [](const Constraint& c) { return c.to_string(); });

    py::class_<ConstraintSet>(m, "ConstraintSet")
        .def(py::init<>())
        .def("add", &ConstraintSet::add)
        .def("add_eq", &ConstraintSet::add_eq)
        .def("add_mod_eq", &ConstraintSet::add_mod_eq)
        .def("empty", &ConstraintSet::empty)
        .def("size", &ConstraintSet::size);

    py::enum_<CmpKind>(m, "CmpKind")
        .value("EQ", CmpKind::EQ)
        .value("NE", CmpKind::NE)
        .value("LT", CmpKind::LT)
        .value("LE", CmpKind::LE)
        .value("GT", CmpKind::GT)
        .value("GE", CmpKind::GE)
        .value("ModEq", CmpKind::ModEq)
        .value("ModNe", CmpKind::ModNe);

    py::enum_<SolverResult>(m, "SolverResult")
        .value("ProvedTrue", SolverResult::ProvedTrue)
        .value("ProvedFalse", SolverResult::ProvedFalse)
        .value("Unknown", SolverResult::Unknown);

    py::class_<Solver>(m, "Solver")
        .def(py::init<ConstraintSet>())
        .def("prove", &Solver::prove)
        .def("prove_equal", &Solver::prove_equal)
        .def("prove_divisible", &Solver::prove_divisible)
        .def("constraints", &Solver::constraints, py::return_value_policy::reference);

    // Shape inference
    py::class_<InferResult>(m, "InferResult")
        .def_readonly("ok", &InferResult::ok)
        .def_readonly("message", &InferResult::message)
        .def_readonly("shape", &InferResult::shape);

    m.def("infer_elementwise_binary", &infer_elementwise_binary);
    m.def("infer_reduction", [](const Shape& a, std::vector<i32> axes, bool keep_dims) {
        return infer_reduction(a, make_span(axes), keep_dims);
    });
    m.def("infer_broadcast", &infer_broadcast);
    m.def("infer_reshape", &infer_reshape);
    m.def("infer_transpose", [](const Shape& a, std::vector<i32> perm) {
        return infer_transpose(a, make_span(perm));
    });
    m.def("infer_matmul", &infer_matmul);

    // ===================================================================
    // Layout system
    // ===================================================================
    py::class_<Layout, std::shared_ptr<Layout>>(m, "Layout")
        .def("bytes", &Layout::bytes)
        .def("is_row_major_contiguous", &Layout::is_row_major_contiguous)
        .def("byte_offset", [](const Layout& l, std::vector<i64> indices, DType dt) {
            return l.byte_offset(make_span(indices), dt);
        })
        .def("structurally_equal", &Layout::structurally_equal)
        .def_static("make_strided", [](Shape shape, std::vector<i64> strides) -> std::shared_ptr<Layout> {
            return std::const_pointer_cast<Layout>(Layout::make_strided(std::move(shape), SmallVector<i64>(strides.begin(), strides.end())));
        })
        .def_static("make_row_major", [](Shape shape) -> std::shared_ptr<Layout> {
            return std::const_pointer_cast<Layout>(Layout::make_row_major(std::move(shape)));
        })
        .def_static("make_col_major", [](Shape shape) -> std::shared_ptr<Layout> {
            return std::const_pointer_cast<Layout>(Layout::make_col_major(std::move(shape)));
        })
        .def_static("make_symbolic", [](Shape shape, std::string id) -> std::shared_ptr<Layout> {
            return std::const_pointer_cast<Layout>(Layout::make_symbolic(std::move(shape), std::move(id)));
        });

    py::enum_<LayoutKind>(m, "LayoutKind")
        .value("Strided", LayoutKind::Strided)
        .value("Broadcast", LayoutKind::Broadcast)
        .value("Compose", LayoutKind::Compose)
        .value("Transpose", LayoutKind::Transpose)
        .value("Reshape", LayoutKind::Reshape)
        .value("Slice", LayoutKind::Slice)
        .value("Symbolic", LayoutKind::Symbolic);

    // ===================================================================
    // IR types
    // ===================================================================
    py::class_<Type, std::shared_ptr<Type>>(m, "Type")
        .def("to_string", &Type::to_string)
        .def("__repr__", [](const Type& t) { return t.to_string(); });

    py::class_<TensorType, Type, std::shared_ptr<TensorType>>(m, "TensorType")
        .def(py::init<>())
        .def_readwrite("shape", &TensorType::shape)
        .def_readwrite("dtype", &TensorType::dtype)
        .def_readwrite("device", &TensorType::device)
        .def_readwrite("memory_space", &TensorType::memory_space)
        .def_static("make", [](Shape shape, DType dtype, DeviceId device, MemorySpace ms) -> std::shared_ptr<TensorType> {
            return std::const_pointer_cast<TensorType>(TensorType::make(std::move(shape), dtype, nullptr, device, ms));
        });

    py::class_<ScalarType, Type, std::shared_ptr<ScalarType>>(m, "ScalarType")
        .def(py::init<DType>(), py::arg("dtype"))
        .def_static("make", &ScalarType::make);

    py::class_<VoidType, Type, std::shared_ptr<VoidType>>(m, "VoidType")
        .def(py::init<>())
        .def_static("make", &VoidType::make);

    m.def("make_tensor_type", [](Shape shape, DType dtype, DeviceId device, MemorySpace ms) {
        return make_tensor_type(std::move(shape), dtype, device, ms);
    }, py::arg("shape"), py::arg("dtype"),
       py::arg("device") = DeviceId::cpu(),
       py::arg("ms") = MemorySpace::Generic);

    m.def("tensor_type", [](std::vector<i64> dims, DType dtype, DeviceId device, MemorySpace ms) {
        Shape s;
        for (auto d : dims) s.dims().push_back(DimExpr::make_constant(d));
        return make_tensor_type(std::move(s), dtype, device, ms);
    }, py::arg("dims"), py::arg("dtype"),
       py::arg("device") = DeviceId::cpu(),
       py::arg("ms") = MemorySpace::Generic);

    // ===================================================================
    // Value / Operation / Block / Module / Function
    // ===================================================================
    py::class_<Value>(m, "Value")
        .def(py::init<>())
        .def("id", &Value::id)
        .def("type", &Value::type)
        .def("is_null", &Value::is_null)
        .def("as_tensor", &Value::as_tensor)
        .def("__repr__", [](const Value& v) {
            return "Value(%" + std::to_string(v.id()) + ")";
        });

    py::class_<Operation>(m, "Operation")
        .def_readonly("id", &Operation::id)
        .def_readonly("opcode", &Operation::opcode)
        .def_readwrite("name", &Operation::name)
        .def_readonly("operands", &Operation::operands)
        .def_readonly("results", &Operation::results)
        .def_readonly("attributes", &Operation::attributes)
        .def_readonly("effects", &Operation::effects)
        .def("is_pure", &Operation::is_pure)
        .def("has_trait", &Operation::has_trait)
        .def("num_operands", &Operation::num_operands)
        .def("num_results", &Operation::num_results)
        .def("result", &Operation::result)
        .def("operand", &Operation::operand);

    py::class_<Block>(m, "Block")
        .def("name", &Block::name)
        .def("arguments", [](Block& b) -> std::vector<Value>& {
            return b.arguments();
        }, py::return_value_policy::reference)
        .def("size", &Block::size)
        .def("empty", &Block::empty)
        .def("head_id", [](Block& b) -> u32 {
            return b.head() ? b.head()->id : 0;
        })
        .def("ops", [](Block& b) {
            std::vector<Operation*> ops;
            for (auto& op : b) ops.push_back(&op);
            return ops;
        }, py::return_value_policy::reference);

    py::class_<Function>(m, "Function")
        .def("name", &Function::name)
        .def("operand_types", [](Function& f) { return f.operand_types(); })
        .def("result_types", [](Function& f) { return f.result_types(); })
        .def("args", [](Function& f) { return f.args(); })
        .def("entry", &Function::entry, py::return_value_policy::reference);

    // Module is non-copyable (contains unique_ptr<Function>), so we use
    // unique_ptr as the pybind11 holder to prevent copy attempts.
    py::class_<Module, std::unique_ptr<Module>>(m, "Module")
        .def(py::init<>())
        .def("create_function", &Module::create_function,
             py::return_value_policy::reference)
        .def("function_ptrs", [](Module& mod) {
            std::vector<Function*> fns;
            for (usize i = 0; i < mod.num_functions(); ++i) {
                // We don't have an index-based accessor, so we walk via
                // the functions vector but only extract raw pointers.
                auto& fs = mod.functions();
                if (i < fs.size()) fns.push_back(fs[i].get());
            }
            return fns;
        }, py::return_value_policy::reference)
        .def("lookup", &Module::lookup, py::return_value_policy::reference)
        .def("constraints", [](Module& m) -> ConstraintSet& {
            return m.constraints();
        }, py::return_value_policy::reference)
        .def("replace_all_uses", &Module::replace_all_uses)
        .def("num_functions", &Module::num_functions)
        .def("num_operations", &Module::num_operations);

    // ===================================================================
    // Opcodes and op traits
    // ===================================================================
    m.attr("OP_INVALID") = OP_INVALID;
    m.attr("OP_CONSTANT") = OP_CONSTANT;
    m.attr("OP_INPUT") = OP_INPUT;
    m.attr("OP_OUTPUT") = OP_OUTPUT;
    m.attr("OP_ADD") = OP_ADD;
    m.attr("OP_SUB") = OP_SUB;
    m.attr("OP_MUL") = OP_MUL;
    m.attr("OP_DIV") = OP_DIV;
    m.attr("OP_NEG") = OP_NEG;
    m.attr("OP_RELU") = OP_RELU;
    m.attr("OP_GELU") = OP_GELU;
    m.attr("OP_MATMUL") = OP_MATMUL;
    m.attr("OP_BROADCAST") = OP_BROADCAST;
    m.attr("OP_RESHAPE") = OP_RESHAPE;
    m.attr("OP_TRANSPOSE") = OP_TRANSPOSE;
    m.attr("OP_REDUCE_SUM") = OP_REDUCE_SUM;
    m.attr("OP_REDUCE_MAX") = OP_REDUCE_MAX;
    m.attr("OP_REDUCE_MEAN") = OP_REDUCE_MEAN;
    m.attr("OP_CAST") = OP_CAST;
    m.attr("OP_COPY") = OP_COPY;
    m.attr("OP_RETURN") = OP_RETURN;
    m.attr("OP_ALLOC") = OP_ALLOC;
    m.attr("OP_FREE") = OP_FREE;

    py::enum_<OpTrait>(m, "OpTrait", py::arithmetic())
        .value("None", OpTrait::None)
        .value("Pure", OpTrait::Pure)
        .value("Commutative", OpTrait::Commutative)
        .value("Associative", OpTrait::Associative)
        .value("Elementwise", OpTrait::Elementwise)
        .value("Broadcastable", OpTrait::Broadcastable)
        .value("Reduction", OpTrait::Reduction)
        .value("MemoryRead", OpTrait::MemoryRead)
        .value("MemoryWrite", OpTrait::MemoryWrite)
        .value("ShapePreserving", OpTrait::ShapePreserving)
        .value("LayoutPreserving", OpTrait::LayoutPreserving)
        .value("HasSideEffect", OpTrait::HasSideEffect)
        .value("TensorCore", OpTrait::TensorCore);

    py::class_<OpInfo>(m, "OpInfo")
        .def_readonly("opcode", &OpInfo::opcode)
        .def_readonly("name", &OpInfo::name)
        .def_readonly("traits", &OpInfo::traits)
        .def_readonly("effects", &OpInfo::effects);

    py::class_<OpRegistry>(m, "OpRegistry")
        .def_static("instance", &OpRegistry::instance,
                    py::return_value_policy::reference)
        .def("lookup", &OpRegistry::lookup, py::return_value_policy::reference)
        .def("lookup_by_name", &OpRegistry::lookup_by_name,
             py::return_value_policy::reference)
        .def("traits", &OpRegistry::traits)
        .def("effects", &OpRegistry::effects);

    // ===================================================================
    // Builder
    // ===================================================================
    py::class_<Builder>(m, "Builder")
        .def(py::init<Function*>())
        .def("function", &Builder::function, py::return_value_policy::reference)
        .def("block", &Builder::block, py::return_value_policy::reference)
        .def("create", [](Builder& b, Opcode op, std::vector<Value> ops) -> Operation* {
            SmallVector<Value, 4> sv;
            for (auto& v : ops) sv.push_back(v);
            return b.create(op, std::move(sv));
        }, py::return_value_policy::reference)
        .def("constant_tensor", [](Builder& b, std::vector<i64> dims, DType dt, py::bytes bytes) {
            Shape s;
            for (auto d : dims) s.dims().push_back(DimExpr::make_constant(d));
            std::string s_bytes = bytes;
            std::vector<u8> v(s_bytes.begin(), s_bytes.end());
            return b.constant_tensor(s, dt, v);
        })
        .def("input_tensor", [](Builder& b, std::vector<i64> dims, DType dt) {
            Shape s;
            for (auto d : dims) s.dims().push_back(DimExpr::make_constant(d));
            return b.input_tensor(s, dt);
        })
        .def("output_tensor", &Builder::output_tensor)
        .def("add", &Builder::add)
        .def("sub", &Builder::sub)
        .def("mul", &Builder::mul)
        .def("div", &Builder::div)
        .def("neg", &Builder::neg)
        .def("relu", &Builder::relu)
        .def("gelu", &Builder::gelu)
        .def("exp", &Builder::exp)
        .def("sqrt", &Builder::sqrt)
        .def("matmul", &Builder::matmul)
        .def("broadcast", &Builder::broadcast)
        .def("reshape", &Builder::reshape)
        .def("transpose", &Builder::transpose)
        .def("reduce_sum", &Builder::reduce_sum)
        .def("reduce_max", &Builder::reduce_max)
        .def("cast", &Builder::cast)
        .def("conv2d", &Builder::conv2d)
        .def("softmax", &Builder::softmax)
        .def("layernorm", &Builder::layernorm)
        .def("batchnorm", &Builder::batchnorm)
        .def("gather", &Builder::gather)
        .def("concat", [](Builder& b, std::vector<Value> inputs, i64 axis) {
            return b.concat(std::move(inputs), axis);
        })
        .def("slice", &Builder::slice)
        .def("sigmoid", &Builder::sigmoid)
        .def("tanh", &Builder::tanh)
        .def("log", &Builder::log);

    // ===================================================================
    // Printer
    // ===================================================================
    m.def("to_string", py::overload_cast<const Module&>(&to_string));
    m.def("to_string", py::overload_cast<const Function&>(&to_string));
    m.def("to_string", py::overload_cast<const Operation&>(&to_string));

    // ===================================================================
    // Analysis framework
    // ===================================================================
    py::class_<PreservedAnalyses>(m, "PreservedAnalyses")
        .def(py::init<>())
        .def_static("all", &PreservedAnalyses::all)
        .def_static("none", &PreservedAnalyses::none)
        .def("preserves_all", &PreservedAnalyses::preserves_all);

    // AnalysisManager holds a Module& but pybind11 tries to copy Module
    // when returning Module&. We use unique_ptr holder and avoid returning
    // Module& directly; the user already has the Module they created.
    py::class_<AnalysisManager, std::unique_ptr<AnalysisManager>>(m, "AnalysisManager")
        .def(py::init<Module&>())
        .def("invalidate", &AnalysisManager::invalidate)
        .def("clear", &AnalysisManager::clear);

    // GTA analyses (expose via GlobalAnalysisManager)
    py::class_<DataflowAnalysis>(m, "DataflowAnalysis")
        .def("topo_order", &DataflowAnalysis::topo_order, py::return_value_policy::reference)
        .def("critical_path_length", &DataflowAnalysis::critical_path_length)
        .def("fanout", &DataflowAnalysis::fanout);

    py::class_<ShapeAnalysis>(m, "ShapeAnalysis")
        .def("shape_of", &ShapeAnalysis::shape_of, py::return_value_policy::reference);

    py::class_<LayoutAnalysis>(m, "LayoutAnalysis")
        .def("layout_of", &LayoutAnalysis::layout_of);

    py::class_<LifetimeAnalysis>(m, "LifetimeAnalysis")
        .def("peak_live_count", &LifetimeAnalysis::peak_live_count)
        .def("peak_live_bytes", &LifetimeAnalysis::peak_live_bytes);

    py::class_<ArithmeticIntensityAnalysis>(m, "ArithmeticIntensityAnalysis")
        .def("module_bound_class", &ArithmeticIntensityAnalysis::module_bound_class)
        .def("module_intensity", &ArithmeticIntensityAnalysis::module_intensity)
        .def("total_flops", &ArithmeticIntensityAnalysis::total_flops)
        .def("total_bytes", &ArithmeticIntensityAnalysis::total_bytes);

    py::class_<ParallelismAnalysis>(m, "ParallelismAnalysis")
        .def("info_of", &ParallelismAnalysis::info_of, py::return_value_policy::reference);

    py::class_<ReuseAnalysis>(m, "ReuseAnalysis")
        .def("info_of", &ReuseAnalysis::info_of, py::return_value_policy::reference);

    py::class_<GlobalAliasAnalysis>(m, "GlobalAliasAnalysis")
        .def("alias", &GlobalAliasAnalysis::alias);

    py::class_<GlobalCostAnalysis>(m, "GlobalCostAnalysis")
        .def("cost", &GlobalCostAnalysis::cost, py::return_value_policy::reference)
        .def("op_cost", &GlobalCostAnalysis::op_cost)
        .def("set_hardware", &GlobalCostAnalysis::set_hardware);

    py::class_<GlobalAnalysisManager>(m, "GlobalAnalysisManager")
        .def(py::init<AnalysisManager&>())
        .def("dataflow", [](GlobalAnalysisManager& g) -> DataflowAnalysis& {
            return g.dataflow();
        }, py::return_value_policy::reference)
        .def("shapes", [](GlobalAnalysisManager& g) -> ShapeAnalysis& {
            return g.shapes();
        }, py::return_value_policy::reference)
        .def("layouts", [](GlobalAnalysisManager& g) -> LayoutAnalysis& {
            return g.layouts();
        }, py::return_value_policy::reference)
        .def("lifetimes", [](GlobalAnalysisManager& g) -> LifetimeAnalysis& {
            return g.lifetimes();
        }, py::return_value_policy::reference)
        .def("intensity", [](GlobalAnalysisManager& g) -> ArithmeticIntensityAnalysis& {
            return g.intensity();
        }, py::return_value_policy::reference)
        .def("parallelism", [](GlobalAnalysisManager& g) -> ParallelismAnalysis& {
            return g.parallelism();
        }, py::return_value_policy::reference)
        .def("reuse", [](GlobalAnalysisManager& g) -> ReuseAnalysis& {
            return g.reuse();
        }, py::return_value_policy::reference)
        .def("aliases", [](GlobalAnalysisManager& g) -> GlobalAliasAnalysis& {
            return g.aliases();
        }, py::return_value_policy::reference)
        .def("cost", [](GlobalAnalysisManager& g) -> GlobalCostAnalysis& {
            return g.cost();
        }, py::return_value_policy::reference)
        .def("set_hardware", &GlobalAnalysisManager::set_hardware)
        .def("invalidate", &GlobalAnalysisManager::invalidate)
        .def("clear", &GlobalAnalysisManager::clear);

    py::enum_<BoundClass>(m, "BoundClass")
        .value("MemoryBound", BoundClass::MemoryBound)
        .value("ComputeBound", BoundClass::ComputeBound)
        .value("LaunchBound", BoundClass::LaunchBound)
        .value("LatencyBound", BoundClass::LatencyBound)
        .value("Balanced", BoundClass::Balanced);

    py::enum_<ReuseDecision>(m, "ReuseDecision")
        .value("Materialize", ReuseDecision::Materialize)
        .value("Recompute", ReuseDecision::Recompute)
        .value("Fuse", ReuseDecision::Fuse)
        .value("Undecided", ReuseDecision::Undecided);

    py::enum_<TensorAliasKind>(m, "TensorAliasKind")
        .value("NoAlias", TensorAliasKind::NoAlias)
        .value("MayAlias", TensorAliasKind::MayAlias)
        .value("MustAlias", TensorAliasKind::MustAlias)
        .value("ViewOf", TensorAliasKind::ViewOf)
        .value("SliceOf", TensorAliasKind::SliceOf)
        .value("BroadcastOf", TensorAliasKind::BroadcastOf);

    // ===================================================================
    // Pass infrastructure
    // ===================================================================
    py::class_<Pass>(m, "Pass");

    py::class_<PassManager>(m, "PassManager")
        .def(py::init<>())
        .def("add", [](PassManager& pm, std::unique_ptr<Pass> p) {
            pm.add(std::move(p));
        })
        .def("run", &PassManager::run)
        .def("size", &PassManager::size)
        .def("empty", &PassManager::empty);

    // Individual passes
    py::class_<CanonicalizePass, Pass>(m, "CanonicalizePass")
        .def(py::init<>())
        .def("name", &CanonicalizePass::name)
        .def("run", &CanonicalizePass::run);

    py::class_<CSEPass, Pass>(m, "CSEPass")
        .def(py::init<>())
        .def("name", &CSEPass::name)
        .def("run", &CSEPass::run);

    py::class_<ConstantFoldingPass, Pass>(m, "ConstantFoldingPass")
        .def(py::init<u64>(), py::arg("max_fold_bytes") = 1ull << 20)
        .def("name", &ConstantFoldingPass::name)
        .def("run", &ConstantFoldingPass::run);

    py::class_<DCEPass, Pass>(m, "DCEPass")
        .def(py::init<>())
        .def("name", &DCEPass::name)
        .def("run", &DCEPass::run);

    py::class_<AlgebraicSimplificationPass, Pass>(m, "AlgebraicSimplificationPass")
        .def(py::init<>())
        .def("name", &AlgebraicSimplificationPass::name)
        .def("run", &AlgebraicSimplificationPass::run);

    py::class_<SCCPPass, Pass>(m, "SCCPPass")
        .def(py::init<>())
        .def("name", &SCCPPass::name)
        .def("run", &SCCPPass::run);

    py::class_<EGraphSuperoptimizerPass, Pass>(m, "EGraphSuperoptimizerPass")
        .def(py::init<>())
        .def("name", &EGraphSuperoptimizerPass::name)
        .def("run", &EGraphSuperoptimizerPass::run);

    py::class_<FusionPass, Pass>(m, "FusionPass")
        .def(py::init<>())
        .def("name", &FusionPass::name)
        .def("run", &FusionPass::run);

    py::class_<ShapeOptimizationPass, Pass>(m, "ShapeOptimizationPass")
        .def(py::init<>())
        .def("name", &ShapeOptimizationPass::name)
        .def("run", &ShapeOptimizationPass::run);

    py::class_<LayoutOptimizationPass, Pass>(m, "LayoutOptimizationPass")
        .def(py::init<>())
        .def("name", &LayoutOptimizationPass::name)
        .def("run", &LayoutOptimizationPass::run);

    py::class_<ReductionOptimizationPass, Pass>(m, "ReductionOptimizationPass")
        .def(py::init<>())
        .def("name", &ReductionOptimizationPass::name)
        .def("run", &ReductionOptimizationPass::run);

    py::class_<CopyEliminationPass, Pass>(m, "CopyEliminationPass")
        .def(py::init<>())
        .def("name", &CopyEliminationPass::name)
        .def("run", &CopyEliminationPass::run);

    py::class_<MemoryPlanningPass, Pass>(m, "MemoryPlanningPass")
        .def(py::init<>())
        .def("name", &MemoryPlanningPass::name)
        .def("run", &MemoryPlanningPass::run);

    py::class_<SpecializationPass, Pass>(m, "SpecializationPass")
        .def(py::init<>())
        .def("name", &SpecializationPass::name)
        .def("run", &SpecializationPass::run);

    // Global Barrier
    py::class_<GlobalBarrierReport>(m, "GlobalBarrierReport")
        .def_readonly("legal", &GlobalBarrierReport::legal)
        .def_readonly("errors", &GlobalBarrierReport::errors)
        .def_readonly("warnings", &GlobalBarrierReport::warnings);

    py::class_<GlobalBarrier>(m, "GlobalBarrier")
        .def(py::init<GlobalAnalysisManager&>())
        .def("run", &GlobalBarrier::run);

    // Iterative Driver
    py::class_<IterativeDriverOptions>(m, "IterativeDriverOptions")
        .def(py::init<>())
        .def_readwrite("max_iterations", &IterativeDriverOptions::max_iterations)
        .def_readwrite("run_global_barrier", &IterativeDriverOptions::run_global_barrier);

    py::class_<IterativeDriverReport>(m, "IterativeDriverReport")
        .def_readonly("iterations_run", &IterativeDriverReport::iterations_run)
        .def_readonly("converged", &IterativeDriverReport::converged)
        .def_readonly("barrier_report", &IterativeDriverReport::barrier_report);

    py::class_<IterativeDriver>(m, "IterativeDriver")
        .def(py::init<AnalysisManager&, IterativeDriverOptions>(),
             py::arg("am"), py::arg("opts") = IterativeDriverOptions{})
        .def("set_hardware", &IterativeDriver::set_hardware)
        .def("run", &IterativeDriver::run);

    // ===================================================================
    // Schedule IR
    // ===================================================================
    py::enum_<TransformKind>(m, "TransformKind")
        .value("Split", TransformKind::Split)
        .value("Tile", TransformKind::Tile)
        .value("Interchange", TransformKind::Interchange)
        .value("Fuse", TransformKind::Fuse)
        .value("Vectorize", TransformKind::Vectorize)
        .value("Parallelize", TransformKind::Parallelize)
        .value("Unroll", TransformKind::Unroll)
        .value("Cache", TransformKind::Cache)
        .value("Pipeline", TransformKind::Pipeline)
        .value("Prefetch", TransformKind::Prefetch)
        .value("Bind", TransformKind::Bind);

    py::class_<Transform>(m, "Transform")
        .def(py::init<>())
        .def(py::init([](TransformKind kind, std::string dim, i64 factor,
                          i64 factor2, std::string target, MemorySpace mem) {
            Transform t;
            t.kind = kind;
            t.dim = std::move(dim);
            t.factor = factor;
            t.factor2 = factor2;
            t.target = std::move(target);
            t.mem = mem;
            return t;
        }), py::arg("kind"), py::arg("dim") = "", py::arg("factor") = 0,
           py::arg("factor2") = 0, py::arg("target") = "",
           py::arg("mem") = MemorySpace::Generic)
        .def_readwrite("kind", &Transform::kind)
        .def_readwrite("dim", &Transform::dim)
        .def_readwrite("factor", &Transform::factor)
        .def_readwrite("factor2", &Transform::factor2)
        .def_readwrite("target", &Transform::target)
        .def_readwrite("mem", &Transform::mem)
        .def("to_string", &Transform::to_string);

    py::class_<Schedule>(m, "Schedule")
        .def(py::init<>())
        .def("add", &Schedule::add)
        .def("transforms", &Schedule::transforms, py::return_value_policy::reference)
        .def("hash", &Schedule::hash);

    py::class_<ScheduleSpace>(m, "ScheduleSpace")
        .def(py::init<>())
        .def("add", &ScheduleSpace::add)
        .def("schedules", &ScheduleSpace::schedules, py::return_value_policy::reference)
        .def("size", &ScheduleSpace::size)
        .def_static("grid_matmul", [](std::vector<i64> m, std::vector<i64> n,
                                       std::vector<i64> k, std::vector<i64> v) {
            return ScheduleSpace::grid_matmul(std::move(m), std::move(n),
                                               std::move(k), std::move(v));
        });

    py::class_<IterationDomain>(m, "IterationDomain")
        .def(py::init<>())
        .def_readwrite("lo", &IterationDomain::lo)
        .def_readwrite("hi", &IterationDomain::hi)
        .def_readwrite("step", &IterationDomain::step)
        .def("extent", &IterationDomain::extent)
        .def("to_string", &IterationDomain::to_string);

    // ===================================================================
    // Cost model
    // ===================================================================
    py::class_<HardwareModel>(m, "HardwareModel")
        .def(py::init<>())
        .def_readwrite("name", &HardwareModel::name)
        .def_readwrite("device", &HardwareModel::device)
        .def_readwrite("shared_mem_bytes", &HardwareModel::shared_mem_bytes)
        .def_readwrite("l1_cache_bytes", &HardwareModel::l1_cache_bytes)
        .def_readwrite("l2_cache_bytes", &HardwareModel::l2_cache_bytes)
        .def_readwrite("simd_width_bytes", &HardwareModel::simd_width_bytes)
        .def_readwrite("num_cores", &HardwareModel::num_cores)
        .def_readwrite("threads_per_core", &HardwareModel::threads_per_core)
        .def_readwrite("launch_overhead_sec", &HardwareModel::launch_overhead_sec)
        .def_readwrite("register_file_per_sm", &HardwareModel::register_file_per_sm)
        .def_readwrite("base_regs_per_thread", &HardwareModel::base_regs_per_thread)
        .def_readwrite("max_warps_per_sm", &HardwareModel::max_warps_per_sm)
        .def_readwrite("warp_size", &HardwareModel::warp_size)
        .def_readwrite("l2_read_bw", &HardwareModel::l2_read_bw)
        .def_readwrite("l2_hit_rate_estimate", &HardwareModel::l2_hit_rate_estimate)
        .def_readwrite("stall_cycles_per_warp", &HardwareModel::stall_cycles_per_warp)
        .def("peak_flops", &HardwareModel::peak_flops,
             py::arg("dt"), py::arg("use_tensor_core") = false)
        .def("supports_tensor_core", &HardwareModel::supports_tensor_core)
        .def("roofline_ridge", &HardwareModel::roofline_ridge,
             py::arg("dt") = DType::F32, py::arg("use_tensor_core") = false)
        .def("estimate_warps_per_sm", &HardwareModel::estimate_warps_per_sm,
             py::arg("regs_per_thread"), py::arg("shared_mem_per_block"),
             py::arg("threads_per_block"))
        .def_static("generic_cpu", &HardwareModel::generic_cpu)
        .def_static("generic_nvidia_gpu", &HardwareModel::generic_nvidia_gpu)
        .def_static("generic_amd_gpu", &HardwareModel::generic_amd_gpu);

    py::class_<CostEstimate>(m, "CostEstimate")
        .def_readonly("flops", &CostEstimate::flops)
        .def_readonly("bytes_global", &CostEstimate::bytes_global)
        .def_readonly("bytes_shared", &CostEstimate::bytes_shared)
        .def_readonly("kernel_launches", &CostEstimate::kernel_launches)
        .def_readonly("parallel_axes", &CostEstimate::parallel_axes)
        .def_readonly("estimated_runtime_sec", &CostEstimate::estimated_runtime_sec);

    py::class_<CostEstimator>(m, "CostEstimator")
        .def(py::init<HardwareModel>())
        .def("hardware", &CostEstimator::hardware, py::return_value_policy::reference)
        .def("estimate", py::overload_cast<const Module&, const Schedule&>(&CostEstimator::estimate, py::const_))
        .def("estimate_matmul", &CostEstimator::estimate_matmul)
        .def("rank", &CostEstimator::rank);

    // Hardware Profile
    py::class_<HardwareProfile>(m, "HardwareProfile")
        .def(py::init<>())
        .def_readwrite("name", &HardwareProfile::name)
        .def_readwrite("device", &HardwareProfile::device)
        .def_readwrite("vector_width_bytes", &HardwareProfile::vector_width_bytes)
        .def_readwrite("cache_line_bytes", &HardwareProfile::cache_line_bytes)
        .def_readwrite("l1_bytes", &HardwareProfile::l1_bytes)
        .def_readwrite("l2_bytes", &HardwareProfile::l2_bytes)
        .def_readwrite("shared_memory_bytes", &HardwareProfile::shared_memory_bytes)
        .def_readwrite("num_cores", &HardwareProfile::num_cores)
        .def_readwrite("threads_per_core", &HardwareProfile::threads_per_core)
        .def_readwrite("warp_width", &HardwareProfile::warp_width)
        .def_readwrite("peak_flops_f32", &HardwareProfile::peak_flops_f32)
        .def_readwrite("peak_flops_f16", &HardwareProfile::peak_flops_f16)
        .def_readwrite("peak_flops_i8", &HardwareProfile::peak_flops_i8)
        .def("supports_dtype", &HardwareProfile::supports_dtype)
        .def("supports_tensor_core", &HardwareProfile::supports_tensor_core)
        .def("is_gpu", &HardwareProfile::is_gpu)
        .def_static("from_model", &HardwareProfile::from_model);

    // ===================================================================
    // Codegen IR
    // ===================================================================
    py::enum_<CGOpcode>(m, "CGOpcode")
        .value("Load", CGOpcode::Load)
        .value("Store", CGOpcode::Store)
        .value("VectorLoad", CGOpcode::VectorLoad)
        .value("VectorStore", CGOpcode::VectorStore)
        .value("FMA", CGOpcode::FMA)
        .value("Add", CGOpcode::Add)
        .value("Mul", CGOpcode::Mul)
        .value("Reduce", CGOpcode::Reduce)
        .value("Barrier", CGOpcode::Barrier)
        .value("AsyncCopy", CGOpcode::AsyncCopy)
        .value("Prefetch", CGOpcode::Prefetch)
        .value("Broadcast", CGOpcode::Broadcast)
        .value("Shuffle", CGOpcode::Shuffle)
        .value("Cmp", CGOpcode::Cmp)
        .value("Select", CGOpcode::Select)
        .value("Cast", CGOpcode::Cast)
        .value("Const", CGOpcode::Const);

    py::class_<CGValue>(m, "CGValue")
        .def(py::init<>())
        .def_readwrite("id", &CGValue::id)
        .def_readwrite("dtype", &CGValue::dtype)
        .def_readwrite("width", &CGValue::width)
        .def_readwrite("name", &CGValue::name);

    py::class_<CGInstruction>(m, "CGInstruction")
        .def(py::init<>())
        .def_readwrite("opcode", &CGInstruction::opcode)
        .def_readwrite("operands", &CGInstruction::operands)
        .def_readwrite("results", &CGInstruction::results)
        .def_readwrite("mem_space", &CGInstruction::mem_space)
        .def_readwrite("comment", &CGInstruction::comment);

    py::class_<CGFunction>(m, "CGFunction")
        .def(py::init<>())
        .def_readwrite("name", &CGFunction::name)
        .def_readwrite("instructions", &CGFunction::instructions)
        .def_readwrite("args", &CGFunction::args)
        .def_readwrite("returns", &CGFunction::returns)
        .def("allocate", [](CGFunction& f, DType dt, u8 width) {
            return f.allocate(dt, width);
        }, py::arg("dtype"), py::arg("width") = 1)
        .def("emit", &CGFunction::emit);

    py::class_<CGModule>(m, "CGModule")
        .def(py::init<>())
        .def("create_function", &CGModule::create_function, py::return_value_policy::reference)
        .def_readwrite("functions", &CGModule::functions);

    m.def("make_load", &make_load);
    m.def("make_store", &make_store);
    m.def("make_fma", &make_fma);
    m.def("make_vector_load", &make_vector_load);
    m.def("make_vector_store", &make_vector_store);
    m.def("make_barrier", &make_barrier);

    // ===================================================================
    // Backends
    // ===================================================================
    // PTX emitter
    py::class_<PTXEmitter>(m, "PTXEmitter")
        .def(py::init<>())
        .def("emit_kernel", &PTXEmitter::emit_kernel,
             py::arg("fn"), py::arg("kernel_name"),
             py::arg("sm_target") = "sm_80")
        .def("emit_body", &PTXEmitter::emit_body);

    // x86 emitter
    py::enum_<X86Reg>(m, "X86Reg")
        .value("RAX", X86Reg::RAX).value("RCX", X86Reg::RCX)
        .value("RDX", X86Reg::RDX).value("RBX", X86Reg::RBX)
        .value("RSP", X86Reg::RSP).value("RBP", X86Reg::RBP)
        .value("RSI", X86Reg::RSI).value("RDI", X86Reg::RDI)
        .value("R8", X86Reg::R8).value("R9", X86Reg::R9)
        .value("R10", X86Reg::R10).value("R11", X86Reg::R11)
        .value("R12", X86Reg::R12).value("R13", X86Reg::R13)
        .value("R14", X86Reg::R14).value("R15", X86Reg::R15);

    py::enum_<X86VReg>(m, "X86VReg")
        .value("XMM0", X86VReg::XMM0).value("XMM1", X86VReg::XMM1)
        .value("XMM2", X86VReg::XMM2).value("XMM3", X86VReg::XMM3)
        .value("XMM4", X86VReg::XMM4).value("XMM5", X86VReg::XMM5)
        .value("XMM6", X86VReg::XMM6).value("XMM7", X86VReg::XMM7)
        .value("XMM8", X86VReg::XMM8).value("XMM9", X86VReg::XMM9)
        .value("XMM10", X86VReg::XMM10).value("XMM11", X86VReg::XMM11)
        .value("XMM12", X86VReg::XMM12).value("XMM13", X86VReg::XMM13)
        .value("XMM14", X86VReg::XMM14).value("XMM15", X86VReg::XMM15);

    py::enum_<VEXWidth>(m, "VEXWidth")
        .value("XMM", VEXWidth::XMM)
        .value("YMM", VEXWidth::YMM)
        .value("ZMM", VEXWidth::ZMM);

    py::class_<X86Emitter>(m, "X86Emitter")
        .def(py::init<>())
        .def("take_bytes", &X86Emitter::take_bytes)
        .def("size", &X86Emitter::size)
        .def("push", &X86Emitter::push)
        .def("pop", &X86Emitter::pop)
        .def("ret", &X86Emitter::ret)
        .def("nop", &X86Emitter::nop)
        .def("int3", &X86Emitter::int3)
        .def("mov_imm64", &X86Emitter::mov_imm64)
        .def("mov_reg", &X86Emitter::mov_reg)
        .def("xor_reg", &X86Emitter::xor_reg)
        .def("add_reg", &X86Emitter::add_reg)
        .def("sub_reg", &X86Emitter::sub_reg)
        .def("imul_reg", &X86Emitter::imul_reg)
        .def("store_reg", &X86Emitter::store_reg)
        .def("load_reg", &X86Emitter::load_reg)
        .def("vmovaps_load", &X86Emitter::vmovaps_load)
        .def("vmovaps_store", &X86Emitter::vmovaps_store)
        .def("vmovups_load", &X86Emitter::vmovups_load)
        .def("vmovups_store", &X86Emitter::vmovups_store)
        .def("vmovss_load", &X86Emitter::vmovss_load)
        .def("vmovss_store", &X86Emitter::vmovss_store)
        .def("vaddps", &X86Emitter::vaddps)
        .def("vmulps", &X86Emitter::vmulps)
        .def("vfmadd231ps", &X86Emitter::vfmadd231ps)
        .def("vfmadd231ss", &X86Emitter::vfmadd231ss)
        .def("vxorps", &X86Emitter::vxorps)
        .def("vmaxps", &X86Emitter::vmaxps)
        .def("vsubps", &X86Emitter::vsubps)
        .def("label", &X86Emitter::label)
        .def("mark_label", &X86Emitter::mark_label)
        .def("jne", &X86Emitter::jne)
        .def("je", &X86Emitter::je)
        .def("jnz", &X86Emitter::jnz)
        .def("jz", &X86Emitter::jz)
        .def("jmp", &X86Emitter::jmp)
        .def("dec_reg", &X86Emitter::dec_reg)
        .def("cmp_imm32", &X86Emitter::cmp_imm32)
        .def("emit_byte", &X86Emitter::emit_byte);

    // Lowering
    m.def("lower_to_ptx", &lower_to_ptx,
          py::arg("fn"), py::arg("kernel_name"), py::arg("sm_target") = "sm_80");
    m.def("lower_to_x86", [](const CGFunction& fn) {
        return lower_to_x86(fn);
    });

    // Executable
    py::class_<Executable>(m, "Executable")
        .def_readwrite("name", &Executable::name)
        .def_readwrite("target_device", &Executable::target_device)
        .def_readwrite("machine_code", &Executable::machine_code)
        .def_readwrite("ptx_text", &Executable::ptx_text)
        .def_readwrite("entrypoints", &Executable::entrypoints)
        .def_readwrite("shared_mem_bytes", &Executable::shared_mem_bytes)
        .def_readwrite("threads_per_block", &Executable::threads_per_block)
        .def_readwrite("disassembly", &Executable::disassembly)
        .def("machine_code_hex", [](const Executable& e) {
            std::string hex;
            for (u8 b : e.machine_code) {
                char buf[4];
                snprintf(buf, sizeof(buf), "%02x ", b);
                hex += buf;
            }
            return hex;
        });

    // TargetInfo
    py::class_<TargetInfo>(m, "TargetInfo")
        .def_readonly("name", &TargetInfo::name)
        .def_readonly("device", &TargetInfo::device)
        .def_readonly("hardware", &TargetInfo::hardware)
        .def("supports_tensor_core", &TargetInfo::supports_tensor_core);

    // Backends
    py::class_<MachineBackend>(m, "MachineBackend");

    py::class_<NvidiaBackend, MachineBackend>(m, "NvidiaBackend")
        .def(py::init<std::string>(), py::arg("sm_target") = "sm_80")
        .def("name", &NvidiaBackend::name)
        .def("target_info", &NvidiaBackend::target_info,
             py::return_value_policy::reference)
        .def("compile", &NvidiaBackend::compile)
        .def("emit_text", &NvidiaBackend::emit_text);

    py::class_<CpuBackend, MachineBackend>(m, "CpuBackend")
        .def(py::init<>())
        .def("name", &CpuBackend::name)
        .def("target_info", &CpuBackend::target_info,
             py::return_value_policy::reference)
        .def("compile", &CpuBackend::compile)
        .def("emit_text", &CpuBackend::emit_text);

    m.def("disassemble_x86", [](std::vector<u8> bytes) {
        return disassemble_x86(make_span(bytes));
    });

    // ===================================================================
    // Autotuner
    // ===================================================================
    py::class_<ScheduleFeatures>(m, "ScheduleFeatures")
        .def(py::init<>())
        .def_readwrite("m_tile", &ScheduleFeatures::m_tile)
        .def_readwrite("n_tile", &ScheduleFeatures::n_tile)
        .def_readwrite("k_tile", &ScheduleFeatures::k_tile)
        .def_readwrite("vector_width", &ScheduleFeatures::vector_width)
        .def_readwrite("unroll_factor", &ScheduleFeatures::unroll_factor)
        .def_readwrite("num_parallel_axes", &ScheduleFeatures::num_parallel_axes)
        .def_readwrite("uses_shared_memory", &ScheduleFeatures::uses_shared_memory)
        .def_readwrite("uses_tensor_core", &ScheduleFeatures::uses_tensor_core);

    m.def("extract_features", &extract_features);

    py::class_<GaussianProcess>(m, "GaussianProcess")
        .def(py::init<>())
        .def("observe", &GaussianProcess::observe)
        .def("predict", &GaussianProcess::predict)
        .def("expected_improvement", &GaussianProcess::expected_improvement)
        .def("num_observations", &GaussianProcess::num_observations)
        .def_readwrite("length_scale", &GaussianProcess::length_scale)
        .def_readwrite("signal_variance", &GaussianProcess::signal_variance)
        .def_readwrite("noise_variance", &GaussianProcess::noise_variance);

    py::class_<AutotuneResult>(m, "AutotuneResult")
        .def_readonly("best_schedule", &AutotuneResult::best_schedule)
        .def_readonly("best_runtime", &AutotuneResult::best_runtime)
        .def_readonly("runtime_history", &AutotuneResult::runtime_history)
        .def_readonly("total_benchmarks", &AutotuneResult::total_benchmarks);

    m.def("bayesian_autotune", [](
        const ScheduleSpace& space,
        std::function<double(const Schedule&)> benchmark,
        usize max_benchmarks,
        usize initial_random) {
        return bayesian_autotune(space, std::move(benchmark),
                                  max_benchmarks, initial_random);
    }, py::arg("space"), py::arg("benchmark"),
       py::arg("max_benchmarks") = 20, py::arg("initial_random") = 3);

    // ===================================================================
    // AnalyticalCostModelV2 (with proper wave quantization + SM utilization)
    // ===================================================================
    py::class_<CostFeatures>(m, "CostFeatures")
        .def(py::init<>())
        .def_readwrite("flops", &CostFeatures::flops)
        .def_readwrite("useful_flops", &CostFeatures::useful_flops)
        .def_readwrite("uses_tensor_core", &CostFeatures::uses_tensor_core)
        .def_readwrite("bytes_global_load", &CostFeatures::bytes_global_load)
        .def_readwrite("bytes_global_store", &CostFeatures::bytes_global_store)
        .def_readwrite("bytes_shared_load", &CostFeatures::bytes_shared_load)
        .def_readwrite("num_blocks", &CostFeatures::num_blocks)
        .def_readwrite("num_sms", &CostFeatures::num_sms)
        .def_readwrite("threads_per_block", &CostFeatures::threads_per_block)
        .def_readwrite("registers_per_thread", &CostFeatures::registers_per_thread)
        .def_readwrite("m_tile", &CostFeatures::m_tile)
        .def_readwrite("n_tile", &CostFeatures::n_tile)
        .def_readwrite("k_tile", &CostFeatures::k_tile)
        .def_readwrite("num_pipeline_stages", &CostFeatures::num_pipeline_stages);

    py::class_<AnalyticalCostModelV2::CostBreakdown>(m, "CostBreakdown")
        .def_readonly("compute_sec", &AnalyticalCostModelV2::CostBreakdown::compute_sec)
        .def_readonly("memory_global_sec", &AnalyticalCostModelV2::CostBreakdown::memory_global_sec)
        .def_readonly("memory_shared_sec", &AnalyticalCostModelV2::CostBreakdown::memory_shared_sec)
        .def_readonly("overlapped_sec", &AnalyticalCostModelV2::CostBreakdown::overlapped_sec)
        .def_readonly("wave_quant_sec", &AnalyticalCostModelV2::CostBreakdown::wave_quant_sec)
        .def_readonly("bank_conflict_sec", &AnalyticalCostModelV2::CostBreakdown::bank_conflict_sec)
        .def_readonly("stall_sec", &AnalyticalCostModelV2::CostBreakdown::stall_sec)
        .def_readonly("total_sec", &AnalyticalCostModelV2::CostBreakdown::total_sec)
        .def_readonly("estimated_warps_per_sm", &AnalyticalCostModelV2::CostBreakdown::estimated_warps_per_sm)
        .def_readonly("estimated_occupancy_pct", &AnalyticalCostModelV2::CostBreakdown::estimated_occupancy_pct)
        .def_readonly("num_blocks", &AnalyticalCostModelV2::CostBreakdown::num_blocks)
        .def_readonly("num_sms", &AnalyticalCostModelV2::CostBreakdown::num_sms)
        .def_readonly("num_waves", &AnalyticalCostModelV2::CostBreakdown::num_waves)
        .def_readonly("sm_utilization_pct", &AnalyticalCostModelV2::CostBreakdown::sm_utilization_pct)
        .def_readonly("tail_efficiency_pct", &AnalyticalCostModelV2::CostBreakdown::tail_efficiency_pct)
        .def_readonly("idle_sms_in_tail", &AnalyticalCostModelV2::CostBreakdown::idle_sms_in_tail);

    py::class_<AnalyticalCostModelV2>(m, "AnalyticalCostModelV2")
        .def(py::init<HardwareModel>())
        .def("estimate", &AnalyticalCostModelV2::estimate)
        .def("estimate_total", &AnalyticalCostModelV2::estimate_total)
        .def("extract_features", &AnalyticalCostModelV2::extract_features);

    // ===================================================================
    // MatmulPipeline (end-to-end orchestrator)
    // ===================================================================
    py::class_<MatmulPipeline::Config>(m, "MatmulPipelineConfig")
        .def(py::init<>())
        .def_readwrite("M", &MatmulPipeline::Config::M)
        .def_readwrite("K", &MatmulPipeline::Config::K)
        .def_readwrite("N", &MatmulPipeline::Config::N)
        .def_readwrite("dtype", &MatmulPipeline::Config::dtype)
        .def_readwrite("fuse_bias_relu", &MatmulPipeline::Config::fuse_bias_relu)
        .def_readwrite("m_tiles", &MatmulPipeline::Config::m_tiles)
        .def_readwrite("n_tiles", &MatmulPipeline::Config::n_tiles)
        .def_readwrite("k_tiles", &MatmulPipeline::Config::k_tiles)
        .def_readwrite("vector_widths", &MatmulPipeline::Config::vector_widths)
        .def_readwrite("max_benchmarks", &MatmulPipeline::Config::max_benchmarks)
        .def_readwrite("initial_random", &MatmulPipeline::Config::initial_random)
        .def_readwrite("top_k_prune", &MatmulPipeline::Config::top_k_prune)
        .def_readwrite("mode", &MatmulPipeline::Config::mode);

    py::class_<CompileTiming>(m, "CompileTiming")
        .def_readonly("ir_construction_sec", &CompileTiming::ir_construction_sec)
        .def_readonly("gta_sec", &CompileTiming::gta_sec)
        .def_readonly("egraph_saturation_sec", &CompileTiming::egraph_saturation_sec)
        .def_readonly("fusion_sec", &CompileTiming::fusion_sec)
        .def_readonly("scheduling_sec", &CompileTiming::scheduling_sec)
        .def_readonly("autotuning_sec", &CompileTiming::autotuning_sec)
        .def_readonly("codegen_sec", &CompileTiming::codegen_sec)
        .def_readonly("total_sec", &CompileTiming::total_sec);

    py::class_<RooflineBreakdown>(m, "RooflineBreakdown")
        .def_readonly("graph_intensity", &RooflineBreakdown::graph_intensity)
        .def_readonly("kernel_intensity", &RooflineBreakdown::kernel_intensity)
        .def_readonly("effective_intensity", &RooflineBreakdown::effective_intensity)
        .def_readonly("ridge_f32", &RooflineBreakdown::ridge_f32)
        .def_readonly("ridge_f16_tc", &RooflineBreakdown::ridge_f16_tc)
        .def_readonly("effective_bound", &RooflineBreakdown::effective_bound)
        .def_readonly("flops", &RooflineBreakdown::flops)
        .def_readonly("graph_bytes", &RooflineBreakdown::graph_bytes)
        .def_readonly("kernel_bytes", &RooflineBreakdown::kernel_bytes)
        .def_readonly("effective_bytes", &RooflineBreakdown::effective_bytes);

    py::class_<MatmulPipelineResult>(m, "MatmulPipelineResult")
        .def_readonly("best_schedule", &MatmulPipelineResult::best_schedule)
        .def_readonly("best_runtime_sec", &MatmulPipelineResult::best_runtime_sec)
        .def_readonly("autotune_history", &MatmulPipelineResult::autotune_history)
        .def_readonly("autotune_benchmarks", &MatmulPipelineResult::autotune_benchmarks)
        .def_readonly("analytical_estimate_sec", &MatmulPipelineResult::analytical_estimate_sec)
        .def_readonly("cost_breakdown", &MatmulPipelineResult::cost_breakdown)
        .def_readonly("timing", &MatmulPipelineResult::timing)
        .def_readonly("roofline", &MatmulPipelineResult::roofline)
        .def_readonly("schedule_space_size", &MatmulPipelineResult::schedule_space_size)
        .def_readonly("pruned_space_size", &MatmulPipelineResult::pruned_space_size)
        .def_readonly("uses_tensor_core", &MatmulPipelineResult::uses_tensor_core);

    py::class_<MatmulPipeline>(m, "MatmulPipeline")
        .def(py::init<HardwareModel, MatmulPipeline::Config>())
        .def("run_with_analytical_benchmark",
             &MatmulPipeline::run_with_analytical_benchmark);

    // Extended ArithmeticIntensityAnalysis (new methods).
    m.attr("_ArithmeticIntensityAnalysis_extras_loaded") = true;

    // ===================================================================
    // Unified Tensor Analyzer (Tensor Knowledge Graph)
    // ===================================================================
    py::enum_<Confidence>(m, "Confidence")
        .value("Proven", Confidence::Proven)
        .value("Derived", Confidence::Derived)
        .value("Estimated", Confidence::Estimated)
        .value("Profiled", Confidence::Profiled)
        .value("Speculative", Confidence::Speculative);

    py::enum_<TensorProperty>(m, "TensorProperty")
        .value("None", TensorProperty::None)
        .value("Constant", TensorProperty::Constant)
        .value("Zero", TensorProperty::Zero)
        .value("One", TensorProperty::One)
        .value("Identity", TensorProperty::Identity)
        .value("Diagonal", TensorProperty::Diagonal)
        .value("Symmetric", TensorProperty::Symmetric)
        .value("Permutation", TensorProperty::Permutation)
        .value("BroadcastConst", TensorProperty::BroadcastConst)
        .value("Sparse", TensorProperty::Sparse)
        .value("BlockSparse", TensorProperty::BlockSparse)
        .value("TriangularLower", TensorProperty::TriangularLower)
        .value("TriangularUpper", TensorProperty::TriangularUpper)
        .value("Dense", TensorProperty::Dense);

    py::enum_<AliasKind>(m, "AliasKind")
        .value("NoAlias", AliasKind::NoAlias)
        .value("MayAlias", AliasKind::MayAlias)
        .value("MustAlias", AliasKind::MustAlias);

    py::enum_<DependenceKind>(m, "DependenceKind")
        .value("Full", DependenceKind::Full)
        .value("Slice", DependenceKind::Slice)
        .value("Reduction", DependenceKind::Reduction)
        .value("Broadcast", DependenceKind::Broadcast)
        .value("LayoutOnly", DependenceKind::LayoutOnly);

    py::class_<AnalyzerMetrics>(m, "AnalyzerMetrics")
        .def_readonly("iterations", &AnalyzerMetrics::iterations)
        .def_readonly("facts_discovered", &AnalyzerMetrics::facts_discovered)
        .def_readonly("worklist_processed", &AnalyzerMetrics::worklist_processed)
        .def_readonly("latency_sec", &AnalyzerMetrics::latency_sec)
        .def_readonly("contradictions", &AnalyzerMetrics::contradictions)
        .def_readonly("mean_prediction_error", &AnalyzerMetrics::mean_prediction_error)
        .def_readonly("predictions_evaluated", &AnalyzerMetrics::predictions_evaluated);

    py::class_<FusionBenefitReport>(m, "FusionBenefitReport")
        .def_readonly("can_fuse", &FusionBenefitReport::can_fuse)
        .def_readonly("legality_reason", &FusionBenefitReport::legality_reason)
        .def_readonly("saved_bytes", &FusionBenefitReport::saved_bytes)
        .def_readonly("saved_kernel_launches", &FusionBenefitReport::saved_kernel_launches)
        .def_readonly("saved_runtime_sec", &FusionBenefitReport::saved_runtime_sec)
        .def_readonly("added_register_pressure", &FusionBenefitReport::added_register_pressure)
        .def_readonly("occupancy_delta_pct", &FusionBenefitReport::occupancy_delta_pct)
        .def_readonly("critical_path_delta_pct", &FusionBenefitReport::critical_path_delta_pct)
        .def_readonly("net_predicted_improvement", &FusionBenefitReport::net_predicted_improvement)
        .def_readonly("confidence", &FusionBenefitReport::confidence);

    py::class_<UnifiedAnalyzer>(m, "UnifiedAnalyzer")
        .def(py::init<Module&>())
        .def("set_hardware", &UnifiedAnalyzer::set_hardware)
        .def("set_numerical_mode", &UnifiedAnalyzer::set_numerical_mode)
        .def("add_default_propagators", &UnifiedAnalyzer::add_default_propagators)
        .def("run", &UnifiedAnalyzer::run, py::return_value_policy::reference)
        .def("run_one_iteration", &UnifiedAnalyzer::run_one_iteration)
        .def("metrics", &UnifiedAnalyzer::metrics, py::return_value_policy::reference)
        .def("store", [](UnifiedAnalyzer& a) -> FactStore& { return a.store(); },
             py::return_value_policy::reference);

    py::class_<FactStore>(m, "FactStore")
        .def("is_zero", &FactStore::is_zero)
        .def("is_one", &FactStore::is_one)
        .def("is_identity", &FactStore::is_identity)
        .def("is_constant", &FactStore::is_constant)
        .def("is_non_negative", &FactStore::is_non_negative)
        .def("is_strictly_positive", &FactStore::is_strictly_positive)
        .def("is_diagonal", &FactStore::is_diagonal)
        .def("is_sparse", &FactStore::is_sparse)
        .def("constant_value", &FactStore::constant_value)
        .def("static_shape", &FactStore::static_shape)
        .def("dtype_of", &FactStore::dtype_of)
        .def("can_fuse", &FactStore::can_fuse)
        .def("fusion_benefit", &FactStore::fusion_benefit)
        .def("num_tensors", &FactStore::num_tensors)
        .def("facts_discovered", &FactStore::facts_discovered);

    // ===================================================================
    // Unified-driven optimization passes
    // ===================================================================
    py::class_<PropertyDrivenSimplification, Pass>(m, "PropertyDrivenSimplification")
        .def(py::init<>())
        .def("run", &PropertyDrivenSimplification::run)
        .def("stats", &PropertyDrivenSimplification::stats,
             py::return_value_policy::reference);

    py::class_<PropertyDrivenSimplification::Stats>(m, "PropertyDrivenSimplificationStats")
        .def_readonly("mul_zero_eliminated", &PropertyDrivenSimplification::Stats::mul_zero_eliminated)
        .def_readonly("mul_one_eliminated", &PropertyDrivenSimplification::Stats::mul_one_eliminated)
        .def_readonly("add_zero_eliminated", &PropertyDrivenSimplification::Stats::add_zero_eliminated)
        .def_readonly("matmul_identity_eliminated", &PropertyDrivenSimplification::Stats::matmul_identity_eliminated)
        .def_readonly("total_rewrites", &PropertyDrivenSimplification::Stats::total_rewrites);

    py::class_<RangeDrivenStrengthReduction, Pass>(m, "RangeDrivenStrengthReduction")
        .def(py::init<>())
        .def("run", &RangeDrivenStrengthReduction::run)
        .def("stats", &RangeDrivenStrengthReduction::stats,
             py::return_value_policy::reference);

    py::class_<RangeDrivenStrengthReduction::Stats>(m, "RangeDrivenStrengthReductionStats")
        .def_readonly("relu_eliminated", &RangeDrivenStrengthReduction::Stats::relu_eliminated)
        .def_readonly("total_rewrites", &RangeDrivenStrengthReduction::Stats::total_rewrites);

    py::class_<CostGuidedFusion, Pass>(m, "CostGuidedFusion")
        .def(py::init<double, Confidence>(),
             py::arg("min_improvement") = 0.05,
             py::arg("min_confidence") = Confidence::Estimated)
        .def("run", &CostGuidedFusion::run)
        .def("stats", &CostGuidedFusion::stats,
             py::return_value_policy::reference);

    py::class_<CostGuidedFusion::Stats>(m, "CostGuidedFusionStats")
        .def_readonly("fusions_accepted", &CostGuidedFusion::Stats::fusions_accepted)
        .def_readonly("fusions_rejected_cost", &CostGuidedFusion::Stats::fusions_rejected_cost)
        .def_readonly("fusions_rejected_confidence", &CostGuidedFusion::Stats::fusions_rejected_confidence)
        .def_readonly("fusions_rejected_legality", &CostGuidedFusion::Stats::fusions_rejected_legality)
        .def_readonly("total_predicted_improvement", &CostGuidedFusion::Stats::total_predicted_improvement);

    py::class_<LayoutAwareCopyElimination, Pass>(m, "LayoutAwareCopyElimination")
        .def(py::init<>())
        .def("run", &LayoutAwareCopyElimination::run)
        .def("stats", &LayoutAwareCopyElimination::stats,
             py::return_value_policy::reference);

    py::class_<LayoutAwareCopyElimination::Stats>(m, "LayoutAwareCopyEliminationStats")
        .def_readonly("transpose_transpose_eliminated", &LayoutAwareCopyElimination::Stats::transpose_transpose_eliminated)
        .def_readonly("total_rewrites", &LayoutAwareCopyElimination::Stats::total_rewrites);

    py::class_<AliasAwareMemoryPlanning, Pass>(m, "AliasAwareMemoryPlanning")
        .def(py::init<>())
        .def("run", &AliasAwareMemoryPlanning::run)
        .def("stats", &AliasAwareMemoryPlanning::stats,
             py::return_value_policy::reference);

    py::class_<AliasAwareMemoryPlanning::Stats>(m, "AliasAwareMemoryPlanningStats")
        .def_readonly("buffers_merged", &AliasAwareMemoryPlanning::Stats::buffers_merged)
        .def_readonly("bytes_saved", &AliasAwareMemoryPlanning::Stats::bytes_saved);

    py::class_<UnifiedOptimizationPipeline, Pass>(m, "UnifiedOptimizationPipeline")
        .def(py::init<>())
        .def("run", &UnifiedOptimizationPipeline::run)
        .def("stats", &UnifiedOptimizationPipeline::stats,
             py::return_value_policy::reference);

    py::class_<UnifiedOptimizationPipeline::Stats>(m, "UnifiedOptimizationPipelineStats")
        .def_readonly("property", &UnifiedOptimizationPipeline::Stats::property)
        .def_readonly("range", &UnifiedOptimizationPipeline::Stats::range)
        .def_readonly("fusion", &UnifiedOptimizationPipeline::Stats::fusion)
        .def_readonly("layout", &UnifiedOptimizationPipeline::Stats::layout)
        .def_readonly("alias", &UnifiedOptimizationPipeline::Stats::alias);

    // ===================================================================
    // Clean compile API — the user-facing entry point.
    //
    //   task = cg.CompileTask(kind="matmul_bias_relu", M=1024, K=1024, N=1024)
    //   result = cg.compile(task, print_ir=True)
    //   print(result.ir_text)
    //   print(result.predicted_runtime_sec)
    // ===================================================================
    py::class_<CompileTask>(m, "CompileTask")
        .def(py::init<>())
        .def_readwrite("kind", &CompileTask::kind)
        .def_readwrite("M", &CompileTask::M)
        .def_readwrite("K", &CompileTask::K)
        .def_readwrite("N", &CompileTask::N)
        .def_readwrite("chain_depth", &CompileTask::chain_depth)
        .def_readwrite("dtype", &CompileTask::dtype)
        .def_readwrite("numerical_mode", &CompileTask::numerical_mode)
        .def_readwrite("hardware", &CompileTask::hardware)
        .def_readwrite("opt_level", &CompileTask::opt_level);

    py::class_<CompileResult>(m, "CompileResultV2")
        .def_readonly("ir_text", &CompileResult::ir_text)
        .def_readonly("ir_text_before", &CompileResult::ir_text_before)
        .def_readonly("predicted_runtime_sec", &CompileResult::predicted_runtime_sec)
        .def_readonly("ops_before", &CompileResult::ops_before)
        .def_readonly("ops_after", &CompileResult::ops_after)
        .def_readonly("converged", &CompileResult::converged)
        .def_readonly("iterations", &CompileResult::iterations)
        .def_readonly("pass_stats", &CompileResult::pass_stats)
        .def_readonly("analyzer_runs", &CompileResult::analyzer_runs)
        .def_readonly("analyzer_latency_sec", &CompileResult::analyzer_latency_sec)
        .def_readonly("facts_discovered", &CompileResult::facts_discovered)
        .def_readonly("graph_intensity", &CompileResult::graph_intensity)
        .def_readonly("effective_intensity", &CompileResult::effective_intensity)
        .def_readonly("roofline_ridge", &CompileResult::roofline_ridge)
        .def_readonly("sm_utilization_pct", &CompileResult::sm_utilization_pct)
        .def_readonly("num_blocks", &CompileResult::num_blocks)
        .def_readonly("num_waves", &CompileResult::num_waves)
        .def_readonly("error", &CompileResult::error);

    m.def("compile", [](const CompileTask& task, bool print_ir) {
        return compile(task, print_ir);
    }, py::arg("task"), py::arg("print_ir") = false);

    // ===================================================================
    // Runtime
    // ===================================================================
    py::class_<Runtime::CompileResult>(m, "CompileResult")
        .def_readonly("executable", &Runtime::CompileResult::executable)
        .def_readonly("cache_hit", &Runtime::CompileResult::cache_hit);

    py::class_<Runtime::AutotuneResult>(m, "RuntimeAutotuneResult")
        .def_readonly("executable", &Runtime::AutotuneResult::executable)
        .def_readonly("best_runtime", &Runtime::AutotuneResult::best_runtime)
        .def_readonly("total_benchmarks", &Runtime::AutotuneResult::total_benchmarks)
        .def_readonly("cache_hit", &Runtime::AutotuneResult::cache_hit);

    py::class_<KernelCache>(m, "KernelCache")
        .def(py::init<std::string>())
        .def("lookup", &KernelCache::lookup)
        .def("insert", &KernelCache::insert)
        .def_static("compute_key", &KernelCache::compute_key)
        .def("dir", &KernelCache::dir, py::return_value_policy::reference)
        .def("size", &KernelCache::size);

    py::class_<Runtime>(m, "Runtime")
        .def(py::init<>())
        .def("add_device", &Runtime::add_device)
        .def("get_device", &Runtime::get_device, py::return_value_policy::reference)
        .def("num_devices", &Runtime::num_devices)
        .def("kernel_cache", &Runtime::kernel_cache, py::return_value_policy::reference)
        .def("compile_and_cache", &Runtime::compile_and_cache)
        .def("autotune_and_cache", &Runtime::autotune_and_cache);

    // ===================================================================
    // Version
    // ===================================================================
    m.attr("__version__") = "0.5.0";

    // ===================================================================
    // Parallel Executor (multi-threaded JIT)
    // ===================================================================
    py::class_<ParallelExecutor>(m, "ParallelExecutor")
        .def(py::init<u32>(), py::arg("num_threads") = 0)
        .def("num_threads", &ParallelExecutor::num_threads)
        .def("execute", [](ParallelExecutor& pe, uintptr_t jit_entry,
                           py::object a, py::object b, py::object c,
                           u64 total_elements) {
            // Extract raw float pointers from numpy arrays.
            auto get_ptr = [](py::object& arr) -> float* {
                py::capsule cap;
                // Try to get array data via the buffer protocol.
                PyObject* obj = arr.ptr();
                Py_buffer view;
                if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE) == 0) {
                    float* ptr = static_cast<float*>(view.buf);
                    PyBuffer_Release(&view);
                    return ptr;
                }
                return nullptr;
            };
            float* a_ptr = get_ptr(a);
            float* b_ptr = get_ptr(b);
            float* c_ptr = get_ptr(c);
            if (!a_ptr || !b_ptr || !c_ptr) {
                throw std::runtime_error("Failed to get array pointers");
            }
            pe.execute(jit_entry, a_ptr, b_ptr, c_ptr, total_elements);
        }, py::arg("jit_entry"), py::arg("a"), py::arg("b"), py::arg("c"),
           py::arg("total_elements"));

    // ===================================================================
    // Tensor IR -> Codegen IR lowering
    // ===================================================================
    py::class_<LoweringOptions>(m, "LoweringOptions")
        .def(py::init<>())
        .def_readwrite("default_vector_width", &LoweringOptions::default_vector_width)
        .def_readwrite("scalarize_small_tensors", &LoweringOptions::scalarize_small_tensors)
        .def_readwrite("scalarize_threshold", &LoweringOptions::scalarize_threshold)
        .def_readwrite("matmul_m_tile", &LoweringOptions::matmul_m_tile)
        .def_readwrite("matmul_n_tile", &LoweringOptions::matmul_n_tile)
        .def_readwrite("matmul_k_tile", &LoweringOptions::matmul_k_tile);

    py::class_<TensorToCodegenLowering>(m, "TensorToCodegenLowering")
        .def(py::init<LoweringOptions>(),
             py::arg("opts") = LoweringOptions{})
        .def("lower", py::overload_cast<const Module&, const Schedule&>(
            &TensorToCodegenLowering::lower),
            py::arg("m"), py::arg("schedule") = Schedule{})
        .def("options", &TensorToCodegenLowering::options,
             py::return_value_policy::reference);

    // ===================================================================
    // JIT execution
    // ===================================================================
    py::class_<JITMemory>(m, "JITMemory")
        .def(py::init<>())
        .def("allocate", [](JITMemory& j, std::vector<u8> code) {
            return j.allocate(code);
        })
        .def("entry", [](const JITMemory& j) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(j.entry());
        })
        .def("size", &JITMemory::size)
        .def("valid", &JITMemory::valid);

    // ===================================================================
    // AMD backend
    // ===================================================================
    py::class_<GCNEmitter>(m, "GCNEmitter")
        .def(py::init<>())
        .def("emit_kernel", &GCNEmitter::emit_kernel,
             py::arg("fn"), py::arg("kernel_name"),
             py::arg("gcn_target") = "gfx908")
        .def("emit_body", &GCNEmitter::emit_body);

    py::class_<AmdBackend, MachineBackend>(m, "AmdBackend")
        .def(py::init<std::string>(), py::arg("gcn_target") = "gfx908")
        .def("name", &AmdBackend::name)
        .def("target_info", &AmdBackend::target_info,
             py::return_value_policy::reference)
        .def("compile", &AmdBackend::compile)
        .def("emit_text", &AmdBackend::emit_text);

    // ===================================================================
    // Vendor dispatch
    // ===================================================================
    py::enum_<VendorKind>(m, "VendorKind")
        .value("cuBLAS", VendorKind::cuBLAS)
        .value("cuDNN", VendorKind::cuDNN)
        .value("rocBLAS", VendorKind::rocBLAS)
        .value("MIOpen", VendorKind::MIOpen)
        .value("oneDNN", VendorKind::oneDNN)
        .value("CUTLASS", VendorKind::CUTLASS)
        .value("ComposableKernel", VendorKind::ComposableKernel);

    m.def("vendor_name", [](VendorKind k) { return std::string(vendor_name(k)); });

    // VendorKernel and VendorDispatcher are non-copyable (contain
    // unique_ptr). We expose them via pointer-only interfaces.
    py::class_<VendorKernel>(m, "VendorKernel")
        .def("vendor", &VendorKernel::vendor)
        .def("name", &VendorKernel::name)
        .def("supports", &VendorKernel::supports)
        .def("estimated_runtime", &VendorKernel::estimated_runtime);

    // Free functions instead of a class binding for VendorDispatcher.
    m.def("vendor_find_best", [](const Operation& op) -> VendorKernel* {
        return VendorDispatcher::instance().find_best(op);
    }, py::return_value_policy::reference);

    m.def("vendor_has", [](VendorKind k) -> bool {
        return VendorDispatcher::instance().has_vendor(k);
    });

    m.def("vendor_num_kernels", []() -> usize {
        return VendorDispatcher::instance().kernels().size();
    });

    m.def("vendor_kernels", []() -> std::vector<VendorKernel*> {
        std::vector<VendorKernel*> ptrs;
        for (auto& k : VendorDispatcher::instance().kernels())
            ptrs.push_back(k.get());
        return ptrs;
    }, py::return_value_policy::reference);

    // ===================================================================
    // Numerical semantics
    // ===================================================================
    py::enum_<NumericalMode>(m, "NumericalMode")
        .value("Strict", NumericalMode::Strict)
        .value("Relaxed", NumericalMode::Relaxed)
        .value("FastMath", NumericalMode::FastMath);

    m.def("numerical_mode_name", [](NumericalMode m) {
        return std::string(numerical_mode_name(m));
    });
    m.def("allows_reassociation", &allows_reassociation);
    m.def("allows_contraction", &allows_contraction);
    m.def("assumes_no_nan", &assumes_no_nan);
    m.def("allows_reciprocal_approx", &allows_reciprocal_approx);
    m.def("allows_mul_zero_elimination", &allows_mul_zero_elimination);

    // ===================================================================
    // Builder convenience for new ops
    // ===================================================================
    // (Already bound above; these are the new ops)
    // conv2d, softmax, layernorm, batchnorm, gather, concat, slice,
    // sigmoid, tanh, log
}
