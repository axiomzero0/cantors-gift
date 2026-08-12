// ir/printer.hpp - textual IR printer
//
// The textual form is round-trippable for development. It is NOT the long-term
// serialization format (we will eventually have a binary form with a strict
// schema), but it is sufficient for testing and debugging.
#pragma once

#include "cg/ir/module.hpp"

#include <ostream>
#include <string>

namespace cg {

class Printer {
public:
    explicit Printer(std::ostream& os) : os_(os) {}

    void print(const Module& m);
    void print(const Function& f);
    void print(const Block& b);
    void print(const Operation& op);

    // Print a value as `%id`.
    void print_value(Value v);

private:
    std::ostream& os_;
    int indent_ = 0;
    void newline();
    void emit_indent();
};

// Convenience: print to string.
std::string to_string(const Module& m);
std::string to_string(const Function& f);
std::string to_string(const Operation& op);

} // namespace cg
