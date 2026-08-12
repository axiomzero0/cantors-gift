// ir/block.hpp - basic blocks
//
// A Block is an ordered sequence of Operations with a list of block arguments.
// A Function has at least one Block (the entry block); control flow (which we
// will eventually support) extends this to multi-block functions.
//
// Operations are stored as unique_ptrs in an intrusive doubly-linked list.
// This makes insertion / deletion O(1) and lets us iterate without pointer
// invalidation.
#pragma once

#include "cg/core/util.hpp"
#include "cg/ir/operation.hpp"
#include "cg/ir/value.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cg {

class Block {
public:
    explicit Block(std::string name = {}) : name_(std::move(name)) {}
    ~Block() {
        // Operations own themselves; tear down the list.
        Operation* cur = head_;
        while (cur) {
            Operation* n = cur->next;
            delete cur;
            cur = n;
        }
    }

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    Block(Block&&) = delete;
    Block& operator=(Block&&) = delete;

    const std::string& name() const { return name_; }

    // Block arguments.
    const std::vector<Value>& arguments() const { return args_; }
    std::vector<Value>& arguments() { return args_; }

    void add_argument(Value v) { args_.push_back(v); }

    // Append an operation owned by this block. Returns a raw pointer.
    Operation* append(std::unique_ptr<Operation> op) {
        Operation* raw = op.release();
        raw->parent = this;
        if (!tail_) {
            head_ = tail_ = raw;
            raw->prev = raw->next = nullptr;
        } else {
            raw->prev = tail_;
            raw->next = nullptr;
            tail_->next = raw;
            tail_ = raw;
        }
        ++size_;
        return raw;
    }

    // Insert `op` before `before`. If `before` is null, appends.
    Operation* insert_before(Operation* before, std::unique_ptr<Operation> op) {
        if (!before) return append(std::move(op));
        Operation* raw = op.release();
        raw->parent = this;
        raw->prev = before->prev;
        raw->next = before;
        if (before->prev) before->prev->next = raw;
        else              head_ = raw;
        before->prev = raw;
        ++size_;
        return raw;
    }

    // Take ownership of `op` out of this block (does NOT delete it).
    std::unique_ptr<Operation> remove(Operation* op) {
        if (op->prev) op->prev->next = op->next;
        else          head_ = op->next;
        if (op->next) op->next->prev = op->prev;
        else          tail_ = op->prev;
        op->prev = op->next = nullptr;
        op->parent = nullptr;
        --size_;
        return std::unique_ptr<Operation>(op);
    }

    Operation* head() const { return head_; }
    Operation* tail() const { return tail_; }
    usize size() const { return size_; }
    bool empty() const { return size_ == 0; }

    // Iteration helpers.
    class Iterator {
    public:
        explicit Iterator(Operation* o) : op_(o) {}
        Operation& operator*()  const { return *op_; }
        Operation* operator->() const { return op_; }
        Iterator& operator++() { op_ = op_->next; return *this; }
        bool operator==(const Iterator& o) const { return op_ == o.op_; }
        bool operator!=(const Iterator& o) const { return op_ != o.op_; }
    private:
        Operation* op_;
    };
    Iterator begin() const { return Iterator(head_); }
    Iterator end()   const { return Iterator(nullptr); }

private:
    std::string name_;
    std::vector<Value> args_;
    Operation* head_ = nullptr;
    Operation* tail_ = nullptr;
    usize size_ = 0;
};

} // namespace cg
