# cantors_gift - Python bindings for the cantors-gift tensor compiler
#
# This package exposes the full compiler stack to Python:
#   - Module / Function / Builder / standard ops
#   - All optimization passes + IterativeDriver
#   - Global Tensor Analysis (GTA) analyses
#   - Schedule / ScheduleSpace / HardwareModel / CostEstimator
#   - Codegen IR / PTX emitter / x86 emitter / backends
#   - Runtime / KernelCache / autotuner
#
# Usage:
#   import cantors_gift as cg
#   m = cg.Module()
#   f = m.create_function("kernel", [...], [...])
#   b = cg.Builder(f)
#   ...
#   am = cg.AnalysisManager(m)
#   driver = cg.IterativeDriver(am)
#   driver.run(m)
#   print(cg.to_string(m))

from .cantors_gift import *  # noqa: F401,F403
from .cantors_gift import __version__  # noqa: F401
