// ir/printer.cpp - implementation of the textual IR printer
#include "cg/ir/printer.hpp"
#include "cg/ir/ops.hpp"

#include <sstream>

namespace cg {

namespace {

const char* opcode_name(Opcode op) {
    auto* info = OpRegistry::instance().lookup(op);
    if (info) return info->name.c_str();
    return "?";
}

} // namespace

void Printer::newline() { os_ << "\n"; }
void Printer::emit_indent() {
    for (int i = 0; i < indent_; ++i) os_ << "  ";
}

void Printer::print_value(Value v) {
    os_ << "%" << v.id();
}

void Printer::print(const Operation& op) {
    emit_indent();
    if (!op.results.empty()) {
        for (usize i = 0; i < op.results.size(); ++i) {
            if (i) os_ << ", ";
            print_value(op.results[i]);
        }
        os_ << " = ";
    }
    os_ << opcode_name(op.opcode);
    if (!op.name.empty()) os_ << "." << op.name;

    if (!op.operands.empty()) {
        os_ << " ";
        for (usize i = 0; i < op.operands.size(); ++i) {
            if (i) os_ << ", ";
            print_value(op.operands[i]);
        }
    }

    // Attributes (inline)
    if (!op.attributes.empty()) {
        os_ << " {";
        bool first = true;
        for (auto& [k, v] : op.attributes) {
            if (!first) os_ << ", ";
            first = false;
            os_ << k << ": " << v->to_string();
        }
        os_ << "}";
    }

    // Type annotations on results.
    if (!op.results.empty()) {
        os_ << " : ";
        if (op.operands.empty()) os_ << "()";
        else {
            os_ << "(";
            for (usize i = 0; i < op.operands.size(); ++i) {
                if (i) os_ << ", ";
                os_ << op.operands[i].type()->to_string();
            }
            os_ << ")";
        }
        os_ << " -> ";
        if (op.results.size() == 1) os_ << op.results[0].type()->to_string();
        else {
            os_ << "(";
            for (usize i = 0; i < op.results.size(); ++i) {
                if (i) os_ << ", ";
                os_ << op.results[i].type()->to_string();
            }
            os_ << ")";
        }
    }

    if (!op.effects.is_pure()) {
        os_ << "  ; effects=0x" << std::hex << static_cast<int>(op.effects.bits()) << std::dec;
    }
}

void Printer::print(const Block& b) {
    if (!b.arguments().empty()) {
        emit_indent();
        os_ << "(";
        for (usize i = 0; i < b.arguments().size(); ++i) {
            if (i) os_ << ", ";
            print_value(b.arguments()[i]);
            os_ << " : " << b.arguments()[i].type()->to_string();
        }
        os_ << ")";
        newline();
    }
    for (auto& op : b) {
        print(op);
        newline();
    }
}

void Printer::print(const Function& f) {
    emit_indent();
    os_ << "func @" << f.name();
    os_ << " : ";
    os_ << "(";
    for (usize i = 0; i < f.operand_types().size(); ++i) {
        if (i) os_ << ", ";
        os_ << f.operand_types()[i]->to_string();
    }
    os_ << ") -> ";
    if (f.result_types().size() == 1) os_ << f.result_types()[0]->to_string();
    else {
        os_ << "(";
        for (usize i = 0; i < f.result_types().size(); ++i) {
            if (i) os_ << ", ";
            os_ << f.result_types()[i]->to_string();
        }
        os_ << ")";
    }
    os_ << " {";
    newline();
    indent_++;
    print(*f.entry());
    indent_--;
    emit_indent();
    os_ << "}";
    newline();
}

void Printer::print(const Module& m) {
    os_ << "// module";
    if (!m.constraints().constraints().empty()) {
        os_ << " constraints:";
        for (auto& c : m.constraints().constraints())
            os_ << " [" << c.to_string() << "]";
    }
    newline();
    for (auto& f : m.functions()) {
        print(*f);
        newline();
    }
}

std::string to_string(const Module& m) {
    std::ostringstream os; Printer p(os); p.print(m); return os.str();
}
std::string to_string(const Function& f) {
    std::ostringstream os; Printer p(os); p.print(f); return os.str();
}
std::string to_string(const Operation& op) {
    std::ostringstream os; Printer p(os); p.print(op); return os.str();
}

} // namespace cg
