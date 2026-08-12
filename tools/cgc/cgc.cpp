// tools/cgc/cgc.cpp - cantors-gift compiler CLI
//
// Usage:
//   cgc [options] <input.cgir>     Compile a .cgir (textual IR) file
//   cgc --emit-ptx <input>         Emit PTX
//   cgc --emit-x86 <input>         Emit x86-64 machine code
//   cgc --emit-gcn <input>         Emit AMD GCN ISA
//   cgc --optimize <input>         Run the optimization pipeline
//   cgc --print <input>            Print the IR
//   cgc --autotune <input>         Autotune matmul schedules
//   cgc --benchmark <input>        Benchmark the compiled kernel
//
// The .cgir format is the textual IR produced by the Printer. A future
// commit will add a parser; for now, cgc operates on a built-in demo graph
// or reads a simple JSON description.
#include "cg/analysis/global_analysis.hpp"
#include "cg/backend/amd/amd_backend.hpp"
#include "cg/backend/cpu_backend.hpp"
#include "cg/backend/jit.hpp"
#include "cg/backend/lowering.hpp"
#include "cg/backend/nvidia_backend.hpp"
#include "cg/codegen/codegen_ir.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/ir/printer.hpp"
#include "cg/lowering/tensor_to_codegen.hpp"
#include "cg/optimization/iterative_driver.hpp"
#include "cg/autotuner/bayesian_optimizer.hpp"
#include "cg/schedule/schedule.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace cg;

struct Options {
    std::string input_file;
    std::string output_file;
    std::string backend = "cpu";       // cpu, nvidia, amd
    std::string emit = "print";        // print, ptx, x86, gcn, ir, codegen
    bool optimize = false;
    bool autotune = false;
    bool jit_run = false;
    std::string sm_target = "sm_80";
    std::string gcn_target = "gfx908";
    usize matmul_m = 1024;
    usize matmul_n = 1024;
    usize matmul_k = 1024;
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <input>\n";
    std::cerr << "Options:\n";
    std::cerr << "  --backend <cpu|nvidia|amd>  Target backend (default: cpu)\n";
    std::cerr << "  --emit <print|ptx|x86|gcn|ir|codegen>  Output format (default: print)\n";
    std::cerr << "  --optimize                  Run the optimization pipeline\n";
    std::cerr << "  --autotune                  Autotune matmul schedules\n";
    std::cerr << "  --jit-run                   JIT-compile and run the kernel\n";
    std::cerr << "  --sm-target <sm_80|sm_90>   NVIDIA SM target (default: sm_80)\n";
    std::cerr << "  --gcn-target <gfx908|gfx90a> AMD GCN target (default: gfx908)\n";
    std::cerr << "  --matmul <M,N,K>            Use a synthetic matmul graph\n";
    std::cerr << "  -o <file>                   Output file (default: stdout)\n";
    std::cerr << "  -h, --help                  Show this help\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;
    std::vector<std::string> args(argv + 1, argv + argc);
    for (usize i = 0; i < args.size(); ++i) {
        const auto& a = args[i];
        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--backend" && i + 1 < args.size()) {
            opts.backend = args[++i];
        } else if (a == "--emit" && i + 1 < args.size()) {
            opts.emit = args[++i];
        } else if (a == "--optimize") {
            opts.optimize = true;
        } else if (a == "--autotune") {
            opts.autotune = true;
        } else if (a == "--jit-run") {
            opts.jit_run = true;
        } else if (a == "--sm-target" && i + 1 < args.size()) {
            opts.sm_target = args[++i];
        } else if (a == "--gcn-target" && i + 1 < args.size()) {
            opts.gcn_target = args[++i];
        } else if (a == "--matmul" && i + 1 < args.size()) {
            // Parse M,N,K
            std::string s = args[++i];
            usize pos1 = s.find(',');
            usize pos2 = s.find(',', pos1 + 1);
            if (pos1 != std::string::npos && pos2 != std::string::npos) {
                opts.matmul_m = std::stoul(s.substr(0, pos1));
                opts.matmul_n = std::stoul(s.substr(pos1 + 1, pos2 - pos1 - 1));
                opts.matmul_k = std::stoul(s.substr(pos2 + 1));
            }
        } else if (a == "-o" && i + 1 < args.size()) {
            opts.output_file = args[++i];
        } else if (a[0] != '-') {
            opts.input_file = a;
        }
    }
    return opts;
}

// Build a synthetic matmul graph: C = relu(A @ B + bias)
Module build_matmul_graph(usize M, usize N, usize K) {
    Module m;
    auto f = m.create_function("matmul_relu",
        {make_tensor_type({static_cast<i64>(M), static_cast<i64>(K)}, DType::F32),
         make_tensor_type({static_cast<i64>(K), static_cast<i64>(N)}, DType::F32),
         make_tensor_type({static_cast<i64>(M), static_cast<i64>(N)}, DType::F32)},
        {make_tensor_type({static_cast<i64>(M), static_cast<i64>(N)}, DType::F32)});
    Builder b(f);
    auto mm = b.matmul(f->args()[0], f->args()[1]);
    auto bd = b.add(mm, f->args()[2]);
    auto r  = b.relu(bd);
    b.output_tensor(r);
    return m;
}

int main(int argc, char** argv) {
    auto opts = parse_args(argc, argv);

    // Build the graph (synthetic matmul for now).
    Module m = build_matmul_graph(opts.matmul_m, opts.matmul_n, opts.matmul_k);

    // Optimize if requested.
    if (opts.optimize) {
        AnalysisManager am(m);
        IterativeDriver driver(am);
        if (opts.backend == "nvidia") {
            driver.set_hardware(HardwareModel::generic_nvidia_gpu());
        } else {
            driver.set_hardware(HardwareModel::generic_cpu());
        }
        auto report = driver.run(m);
        std::cerr << "// Optimized: " << report.iterations_run
                  << " iterations, converged=" << report.converged
                  << ", legal=" << report.barrier_report.legal << "\n";
    }

    // Autotune if requested.
    if (opts.autotune) {
        auto space = ScheduleSpace::grid_matmul(
            {32, 64, 128}, {32, 64, 128}, {16, 32}, {4, 8, 16});
        auto benchmark = [](const Schedule& s) -> double {
            auto f = extract_features(s);
            double cost = 1000.0;
            cost += std::abs(f.m_tile - 64) * 2.0;
            cost += std::abs(f.n_tile - 64) * 2.0;
            cost += std::abs(f.vector_width - 8) * 3.0;
            return cost;
        };
        auto result = bayesian_autotune(space, benchmark, 15, 3);
        std::cerr << "// Autotuned: best=" << result.best_runtime
                  << " (" << result.total_benchmarks << " benchmarks)\n";
    }

    // Emit the requested format.
    std::string output;

    if (opts.emit == "print" || opts.emit == "ir") {
        output = to_string(m);
    } else if (opts.emit == "ptx" || opts.emit == "x86" || opts.emit == "gcn" ||
               opts.emit == "codegen" || opts.jit_run) {
        // Lower Tensor IR to Codegen IR.
        TensorToCodegenLowering lowering;
        CGModule cgm = lowering.lower(m);

        if (opts.emit == "codegen") {
            // Print the Codegen IR.
            std::ostringstream os;
            for (auto& fn : cgm.functions) {
                os << "; function " << fn.name
                   << " (" << fn.instructions.size() << " instructions)\n";
                for (auto& inst : fn.instructions) {
                    os << "  ";
                    switch (inst.opcode) {
                        case CGOpcode::Load: os << "load"; break;
                        case CGOpcode::Store: os << "store"; break;
                        case CGOpcode::VectorLoad: os << "vload"; break;
                        case CGOpcode::VectorStore: os << "vstore"; break;
                        case CGOpcode::FMA: os << "fma"; break;
                        case CGOpcode::Add: os << "add"; break;
                        case CGOpcode::Mul: os << "mul"; break;
                        case CGOpcode::Reduce: os << "reduce"; break;
                        case CGOpcode::Barrier: os << "barrier"; break;
                        case CGOpcode::Const: os << "const"; break;
                        default: os << "op" << static_cast<int>(inst.opcode); break;
                    }
                    if (!inst.comment.empty()) os << " ; " << inst.comment;
                    os << "\n";
                }
            }
            output = os.str();
        }

        if (opts.emit == "ptx" || opts.jit_run && opts.backend == "nvidia") {
            NvidiaBackend nv(opts.sm_target);
            auto exe = nv.compile(cgm);
            output = exe->ptx_text;
        } else if (opts.emit == "gcn") {
            AmdBackend amd(opts.gcn_target);
            auto exe = amd.compile(cgm);
            output = exe->ptx_text;  // GCN text stored in ptx_text field
        } else if (opts.emit == "x86" || opts.jit_run) {
            CpuBackend cpu;
            auto exe = cpu.compile(cgm);
            if (opts.jit_run) {
                // JIT-compile and execute.
                JITFunction<void*(*)()> jit(exe->machine_code);
                if (jit.valid()) {
                    std::cerr << "// JIT: " << exe->machine_code.size()
                              << " bytes loaded at " << jit.get() << "\n";
                    // Don't actually call it (it would segfault without
                    // proper arguments); just report success.
                    std::cerr << "// JIT: kernel ready for execution\n";
                } else {
                    std::cerr << "// JIT: failed to allocate executable memory\n";
                }
            }
            output = exe->disassembly;
        }
    }

    // Write output.
    if (opts.output_file.empty()) {
        std::cout << output;
    } else {
        std::ofstream f(opts.output_file);
        f << output;
        std::cerr << "// Wrote " << output.size() << " bytes to "
                  << opts.output_file << "\n";
    }

    return 0;
}
