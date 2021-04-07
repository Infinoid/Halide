#ifndef HALIDE_INJECT_COMMUNICATIONS_H
#define HALIDE_INJECT_COMMUNICATIONS_H

/** \file
 * Defines the lowering pass that injects communication code between MPI nodes.
 */

#include "IR.h"

namespace Halide {
namespace Internal {

/** Find send_to nodes and replace them with actual MPI receive and send code.
 */
Stmt inject_communications(Stmt s);

} // namespace Internal
} // namespace Halide

#endif
