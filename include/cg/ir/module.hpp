// ir/module.hpp - top-level IR container
//
// A Module owns Functions, each Function owns Blocks, each Block owns
// Operations. ValueIds are unique within a Module and never reused.
//
// The Module also owns a ConstraintSet of global shape facts (e.g. all
// statically-known divisibility constraints that the front-end emitted).
#pragma once

#include "cg/core/util.hpp"
#include "cg/ir/attributes.hpp"
#include "cg/ir/block.hpp"
#include "cg/ir/operation.hpp"
#include "cg/ir/value.hpp"
#include "cg/shape/constraint.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cg {

class Function;
class Module;

using ModuleId = u32;

// A Function corresponds roughly to a "kernel" or a graph to be compiled as
// a single unit. It has a name, a signature (operand types / result types),
// and one or more Blocks.
class Function {
public:
    Function(std::string name, std::vector<TypePtr> operand_types,
             std::vector<TypePtr> result_types)
        : name_(std::move(name)),
          operand_types_(std::move(operand_types)),
          result_types_(std::move(result_types)) {
        entry_ = std::make_unique<Block>("entry");
        for (usize i = 0; i < operand_types_.size(); ++i) {
            ValueId vid = next_value_id_++;
            args_.emplace_back(operand_types_[i], vid);
        }
        for (auto& v : args_) entry_->add_argument(v);
    }

    const std::string& name() const { return name_; }
    const std::vector<TypePtr>& operand_types() const { return operand_types_; }
    const std::vector<TypePtr>& result_types() const { return result_types_; }
    const std::vector<Value>& args() const { return args_; }

    Block* entry() const { return entry_.get(); }

    ValueId allocate_value_id() { return next_value_id_++; }
    OpId   allocate_op_id()     { return next_op_id_++; }

private:
    friend class Module;
    std::string name_;
    std::vector<TypePtr> operand_types_;
    std::vector<TypePtr> result_types_;
    std::vector<Value> args_;
    std::unique_ptr<Block> entry_;
    ValueId next_value_id_ = 1;
    OpId   next_op_id_     = 1;
};

class Module {
public:
    Module() = default;

    Function* create_function(std::string name,
                              std::vector<TypePtr> operand_types,
                              std::vector<TypePtr> result_types) {
        auto f = std::make_unique<Function>(
            std::move(name), std::move(operand_types), std::move(result_types));
        functions_.push_back(std::move(f));
        return functions_.back().get();
    }

    const std::vector<std::unique_ptr<Function>>& functions() const {
        return functions_;
    }

    Function* lookup(StringRef name) {
        for (auto& f : functions_) if (f->name() == name) return f.get();
        return nullptr;
    }

    // Module-global constraint set (e.g. M % 32 == 0).
    ConstraintSet& constraints() { return constraints_; }
    const ConstraintSet& constraints() const { return constraints_; }

    // Walk every operation in every function. The visitor returns true to
    // continue, false to stop.
    template <typename F>
    void walk(F&& fn) const {
        for (auto& f : functions_) {
            for (auto& op : *f->entry()) fn(op);
        }
    }

    // Replace all uses of `old` with `new_v` across the entire module.
    void replace_all_uses(Value old, Value new_v);

    // Statistics
    usize num_functions() const { return functions_.size(); }
    usize num_operations() const {
        usize n = 0;
        for (auto& f : functions_) n += f->entry()->size();
        return n;
    }

private:
    std::vector<std::unique_ptr<Function>> functions_;
    ConstraintSet constraints_;
};

} // namespace cg
