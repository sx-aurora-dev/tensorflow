
#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

//#define TENSORFLOW_USE_VE

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/framework/ve_ops_common.h"
#endif




namespace tensorflow {

#ifdef TENSORFLOW_USE_VE

#if 0
//for CPU
template <typename T, typename Tindices>
class ConvertVESparseTensorCPUOp : public OpKernel {
public:
    explicit ConvertVESparseTensorCPUOp(OpKernelConstruction* ctx)
        : OpKernel(ctx) {

        fprintf(stderr,"test\n");
        fflush(stderr);
        printf("construct test\n");
        fflush(stdin);
        OP_REQUIRES(ctx, 0,
                    errors::InvalidArgument("Tensor test"));


    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* indices;
        const Tensor* values;
        const Tensor* shape;

        fprintf(stderr,"test\n");
        fflush(stderr);
        printf("test compute\n");
        fflush(stdin);
        OP_REQUIRES(ctx, 0,
                    errors::InvalidArgument("Tensor test"));


        OP_REQUIRES_OK(ctx, ctx->input("indices", &indices));
        OP_REQUIRES_OK(ctx, ctx->input("values", &values));
        OP_REQUIRES_OK(ctx, ctx->input("shape", &shape));


        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(shape->shape()),
                    errors::InvalidArgument("Tensor 'shape' is not a vector"));

        OP_REQUIRES(
                    ctx, shape->NumElements() == 2,
                    errors::InvalidArgument("Tensor 'shape' must have 2 elements"));

        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(values->shape()),
                    errors::InvalidArgument("Tensor 'values' is not a vector"));

        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(indices->shape()),
                    errors::InvalidArgument("Tensor 'indices' is not a matrix"));




    }
};


REGISTER_KERNEL_BUILDER(                       \
        Name("ConvertVESparseTensor")            \
        .Device(DEVICE_CPU)                    \
        .TypeConstraint<float>("T") \
        .TypeConstraint<int64>("Tindices")            \
        .HostMemory("shape"),                \
        ConvertVESparseTensorCPUOp<float, int64>);

#endif

//for VE
template <typename T, typename Tindices>
class ConvertVESparseTensorOp : public VEOpKernel {
public:
    explicit ConvertVESparseTensorOp(OpKernelConstruction* ctx)
        : VEOpKernel(ctx) {

        convert_flag = true;
        //        printf("test construct ve\n");
        //        fflush(stdin);



    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* indices;
        const Tensor* values;
        const Tensor* shape;


        //        printf("ConvertVESparseTensorOp compute ve: flag %d\n",convert_flag);
        //        fflush(stdin);



        OP_REQUIRES_OK(ctx, ctx->input("indices", &indices));
        OP_REQUIRES_OK(ctx, ctx->input("values", &values));
        OP_REQUIRES_OK(ctx, ctx->input("shape", &shape));


        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(shape->shape()),
                    errors::InvalidArgument("Tensor 'shape' is not a vector"));

        OP_REQUIRES(
                    ctx, shape->NumElements() == 2,
                    errors::InvalidArgument("Tensor 'shape' must have 2 elements"));

        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(values->shape()),
                    errors::InvalidArgument("Tensor 'values' is not a vector"));

        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(indices->shape()),
                    errors::InvalidArgument("Tensor 'indices' is not a matrix"));

        auto shape_t = shape->vec<int64>();


        const long long int max_val_size = 1000 * 1000 * 1000UL;//800M*4
        const long long int max_idx_size = max_val_size * 1.5;

        double rate = 3;
        long long int out_val_size = values->NumElements() * 3;
        long long int out_idx_size = out_val_size * rate + 3*(shape_t(0) + shape_t(1));

        //        printf("idx size %ldM, %ldMB, max %ldM, %ldMB\n",
        //               out_idx_size/1000000,8*out_idx_size/1000000,
        //               max_idx_size/1000000,8*max_idx_size/1000000);
        out_idx_size = (out_idx_size < max_idx_size) ? out_idx_size:max_idx_size;
        TensorShape out_idx_shape({out_idx_size, 1});
        Tensor* out_idx = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_idx_shape, &out_idx));


        //        printf("val size %ldM, %ldMB, max %ldM, %ldMB\n",
        //               out_val_size/1000000,4*out_val_size/1000000,
        //               max_val_size/1000000,4*max_val_size/1000000);
        out_val_size = (out_val_size < max_val_size) ? out_val_size:max_val_size;
        TensorShape out_val_shape({out_val_size, 1});
        Tensor* out_val = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, out_val_shape, &out_val));




        if(convert_flag){
            if (shape->dim_size(0) > 0) {

                ArgsImpl<> Args = ArgsImpl<>() ;

                Args.addArg<Tensor>(*values) ;

                Args.addArg<Tensor>(*indices) ;

                Args.addArg<int64>(shape_t(1)) ;

                Args.addArg<int64>(shape_t(0)) ;

                Args.addArg<Tensor>(*out_val) ;

                Args.addArg<Tensor>(*out_idx) ;





                Call(ctx, "ConvertVESparseTensor", Args);

            }
        }
        //convert_flag = false;

        //printf("ConvertVESparseTensorOp compute ve: end\n");
    }
private:
    bool convert_flag;
};



REGISTER_KERNEL_BUILDER(                       \
        Name("ConvertVESparseTensor")            \
        .Device(DEVICE_VE)                    \
        .TypeConstraint<float>("T") \
        .TypeConstraint<int64>("Tindices")            \
        .HostMemory("shape"),                \
        ConvertVESparseTensorOp<float, int64>);

#endif



}  // namespace tensorflow


