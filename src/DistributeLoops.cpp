#include <mpi.h>

#include "DistributeLoops.h"
#include "Function.h"
#include "IRVisitor.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Simplify.h"

using std::string;
using std::map;

namespace Halide {
namespace Internal {

// Find all distributed loops, mark their names, and record the expressions
struct DistributedLoop {
    Expr min, extent;
};
class MarkDistributedLoops : public IRVisitor {
    Scope<Expr> let_stmts;
    bool inside_distributed_loop = false;
public:
    using IRVisitor::visit;
    map<string, DistributedLoop> distributed_loops;
    bool found_distributed_loop = false;
    
    void visit(const For *op) override {
        if (op->distributed) {
            user_assert(!inside_distributed_loop) << "Nested distributed loops is currently not allowed.";
            std::string loop_min = op->name + ".loop_min";
            std::string loop_extent = op->name + ".loop_extent";
            internal_assert(let_stmts.contains(loop_min));
            internal_assert(let_stmts.contains(loop_extent));
            distributed_loops[op->name] = DistributedLoop{let_stmts.get(loop_min), let_stmts.get(loop_extent)};
            found_distributed_loop = true;
            
            {
                ScopedValue<bool> old_inside_distributed_loop(inside_distributed_loop, true);
                IRVisitor::visit(op);
            }
        } else {
            IRVisitor::visit(op);
        }
    }

    void visit(const LetStmt *op) override {
        let_stmts.push(op->name, op->value);
        IRVisitor::visit(op);
        let_stmts.pop(op->name);
    }
};

// For each distributed for loop, mutate its bounds to be determined
// by processor rank.
// Instead of modifying the for loop extents, we modify the let statements, since the bounds inference
// pass rely on these variables.
class DistributeLoops : public IRMutator {
    const map<string, DistributedLoop> &distributed_loops;
public:
    DistributeLoops(const map<string, DistributedLoop> &distributed_loops) : distributed_loops(distributed_loops) {}

    using IRMutator::visit;
    Stmt visit(const For *op) override {
        if (op->distributed) {
            internal_assert(op->min.as<Variable>() != nullptr && op->min.as<Variable>()->name == op->name + ".loop_min");
            internal_assert(op->extent.as<Variable>() != nullptr && op->extent.as<Variable>()->name == op->name + ".loop_extent");
            Expr rank = Call::make(Int(32), Call::mpi_rank, {}, Call::PureIntrinsic);
            Expr num_processors = Call::make(Int(32), Call::mpi_num_processors, {}, Call::PureIntrinsic);
            DistributedLoop loop_info;
            auto loop_it = distributed_loops.find(op->name);
            if (loop_it != distributed_loops.end()) {
                loop_info = loop_it->second;
            } else {
                internal_assert(false);
            }
            // Number of elements each processor needs to process.
            Expr slice_size = (loop_info.extent + num_processors - 1) / num_processors;
            Expr new_min = simplify(loop_info.min + slice_size * rank);
            Expr new_max = simplify(min(new_min + slice_size - 1, loop_info.min + loop_info.extent - 1));
            Expr new_extent = simplify(new_max - new_min + 1);
            Expr min_var = Variable::make(Int(32), op->name + ".loop_min");
            Expr extent_var = Variable::make(Int(32), op->name + ".loop_extent");
            Stmt for_stmt = For::make(op->name, min_var, extent_var, op->for_type, op->distributed, op->device_api, mutate(op->body));
            for_stmt = LetStmt::make(op->name + ".loop_extent", new_extent, for_stmt);
            for_stmt = LetStmt::make(op->name + ".loop_min", new_min, for_stmt);
            for_stmt = LetStmt::make(op->name + ".loop_max", new_max, for_stmt);
            return for_stmt;
        } else {
            return IRMutator::visit(op);
        }
    }

    Stmt visit(const LetStmt *op) override {
        if (ends_with(op->name, ".loop_max") || ends_with(op->name, ".loop_min") || ends_with(op->name, ".loop_extent")) {
            string loop_name = op->name.substr(0, op->name.rfind('.'));
            if (distributed_loops.find(loop_name) != distributed_loops.end()) {
                // Strip the variable here. We will reintroduce them later in the for loop mutation
                return mutate(op->body);
            } else {
                return IRMutator::visit(op);    
            }
        } else {
            return IRMutator::visit(op);
        }
    }
};

std::pair<Stmt, bool> distribute_loops(Stmt s) {
    MarkDistributedLoops mark_distributed_loops;
    s.accept(&mark_distributed_loops);
    s = DistributeLoops(mark_distributed_loops.distributed_loops).mutate(s);
    return {s, mark_distributed_loops.found_distributed_loop};
}

} // Internal
} // Halide