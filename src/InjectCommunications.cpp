#include "InjectCommunications.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Simplify.h"

using std::string;
using std::to_string;

namespace Halide {
namespace Internal {

class InjectCommunications : public IRMutator {
public:
    InjectCommunications(bool inject_profiling) : inject_profiling(inject_profiling) {}
protected:
    using IRMutator::visit;

    Stmt visit(const Evaluate *op) override {
        if (Call::as_intrinsic(op->value, {Call::send_to_marker})) {
            const Call *marker = op->value.as<Call>();
            internal_assert(marker != nullptr && marker->args.size() == 3);
            const StringImm *producer_name = marker->args[0].as<StringImm>();
            internal_assert(producer_name != nullptr);
            Expr destination_rank = marker->args[1];
            const int64_t *dimensions = as_const_int(marker->args[2]);
            internal_assert(dimensions != nullptr);
            Type type = marker->type;

            Expr rank = Call::make(Int(32), Call::mpi_rank, {}, Call::PureIntrinsic);
            Expr num_processors = Call::make(Int(32), Call::mpi_num_processors, {}, Call::PureIntrinsic);
            string loop_var_name = unique_name('t');
            string min_var = unique_name('t');
            string extent_var = unique_name('t');
            Expr min_var_ptr = Variable::make(type_of<int *>(), min_var);
            Expr extent_var_ptr = Variable::make(type_of<int *>(), extent_var);
            Expr host = Variable::make(type_of<struct halide_buffer_t *>(), producer_name->value);
            Expr loop_var = Variable::make(Int(32), loop_var_name);
            Stmt destination_case = For::make(loop_var_name,
                Expr(0), num_processors, ForType::Serial, false /* distributed */, DeviceAPI::Host, 
                IfThenElse::make(loop_var != destination_rank,
                    Allocate::make(min_var, Int(32),  MemoryType::Register, {}, const_true(),
                    Allocate::make(extent_var, Int(32),  MemoryType::Register, {}, const_true(),
                        Block::make({
                            Evaluate::make(Call::make(Bool(), Call::mpi_recv, {
                                min_var_ptr, 0, Expr(4), loop_var, Expr(0) /* tag */}, Call::Intrinsic)),
                            Evaluate::make(Call::make(Bool(), Call::mpi_recv, {
                                extent_var_ptr, 0, Expr(4), loop_var, Expr(1) /* tag */}, Call::Intrinsic)),
                            Evaluate::make(Call::make(Bool(), Call::mpi_recv, {
                                host,
                                Load::make(Int(32), min_var, 0, Buffer<>(), Parameter(), const_true(), ModulusRemainder()) *
                                type.bytes(), 
                                Load::make(Int(32), extent_var, 0, Buffer<>(), Parameter(), const_true(), ModulusRemainder()) *
                                type.bytes(),
                                loop_var, Expr(2) /* tag */}, Call::Intrinsic)),
                        })
                    ))
                )
            );
            Expr min_val = 0, extent_val = 1;
            for (int i = 0; i < *dimensions; i++) {
                min_val += extent_val * Variable::make(Int(32), producer_name->value + ".min." + to_string(i));
                extent_val *= Variable::make(Int(32), producer_name->value + ".extent." + to_string(i));
            }
            min_val = simplify(min_val);
            extent_val = simplify(extent_val);
            Stmt source_case =
                Allocate::make(min_var, Int(32),  MemoryType::Register, {}, const_true(),
                Allocate::make(extent_var, Int(32),  MemoryType::Register, {}, const_true(),
                    Block::make({
                        Store::make(min_var, min_val, 0, Parameter(), const_true(), ModulusRemainder()),
                        Store::make(extent_var, extent_val, 0, Parameter(), const_true(), ModulusRemainder()),
                        Evaluate::make(Call::make(Bool(), Call::mpi_send, {
                            min_var_ptr, 0, Expr(4), destination_rank, Expr(0) /* tag */}, Call::Intrinsic)),
                        Evaluate::make(Call::make(Bool(), Call::mpi_send, {
                            extent_var_ptr, 0, Expr(4), destination_rank, Expr(1) /* tag */}, Call::Intrinsic)),
                        Evaluate::make(Call::make(Bool(), Call::mpi_send, {
                            host, 0,
                            Load::make(Int(32), extent_var, 0, Buffer<>(), Parameter(), const_true(), ModulusRemainder()) * type.bytes(),
                            destination_rank, Expr(2) /* tag */}, Call::Intrinsic))
                    })
                ));
            Stmt ret = IfThenElse::make(rank != destination_rank, source_case, destination_case);
            if (inject_profiling) {
                ret = Block::make({Evaluate::make(
                        Call::make(Type(), Call::send_to_profiling_marker, {marker->args[0], const_true() /*begin*/}, Call::Intrinsic)),
                    ret,
                    Evaluate::make(
                        Call::make(Type(), Call::send_to_profiling_marker, {marker->args[0], const_false() /*end*/}, Call::Intrinsic))});
            }
            return ret;
        } else {
            return IRMutator::visit(op);
        }
    }
private:
    bool inject_profiling;
};

Stmt inject_communications(Stmt s, bool inject_profiling) {
    return InjectCommunications(inject_profiling).mutate(s);
}

} // namespace Internal
} // namespace Halide