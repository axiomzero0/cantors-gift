// optimization/global_barrier.hpp - Global Barrier phase
//
// The Global Barrier is the architectural line between high-level
// optimization and final lowering. Above the barrier, the compiler runs
// local canonicalization, e-graphs, fusion, memory planning, and scheduling.
// At the barrier, the compiler:
//
//   1. Runs Global Tensor Analysis (GTA) end-to-end.
//   2. Verifies global legality (every op has a valid shape, layout, dtype;
//      no dangling operands; every alloc has a matching free; every fused
//      region is well-formed).
//   3. Validates the schedule (every tile size is legal, register pressure
//      is within budget, shared memory fits).
//   4. Makes the final global optimization decisions (which fusion
//      candidates to keep, which layouts to freeze, which kernels to
//      specialize).
//
// Below the barrier, lowering faithfully realizes the decisions.
#pragma once

#include "cg/analysis/global_analysis.hpp"
#include "cg/ir/module.hpp"
#include "cg/schedule/schedule.hpp"

#include <string>
#include <vector>

namespace cg {

struct GlobalBarrierReport {
    bool legal = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    // Final decisions recorded at the barrier.
    struct Decisions {
        // Which op ids are fused together (each inner vector is a fusion cluster).
        std::vector<std::vector<u32>> fusion_clusters;
        // Which buffer ids are reused.
        std::vector<u32> reused_buffers;
        // Which schedules are validated for which ops.
        std::unordered_map<u32, Schedule> validated_schedules;
        // Which ops are specialized and under what predicate.
        std::unordered_map<u32, std::string> specializations;
    } decisions;
};

class GlobalBarrier {
public:
    explicit GlobalBarrier(GlobalAnalysisManager& gta) : gta_(gta) {}

    // Run the full barrier check + decision recording.
    GlobalBarrierReport run(Module& m);

    // Individual checks.
    bool check_legality(Module& m, std::vector<std::string>& errors);
    bool check_schedule(Module& m, std::vector<std::string>& errors);
    void finalize_decisions(Module& m, GlobalBarrierReport::Decisions& out);

private:
    GlobalAnalysisManager& gta_;
};

} // namespace cg
