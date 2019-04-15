
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/framework/ve_ops_common.h"
#endif

using namespace tensorflow;

REGISTER_OP("MLP")
    .Attr("NumLinears: int >= 1")
    .Attr("T: {float}")
    .Attr("batch_size: int >= 1")
    .Attr("epochs: int >= 1")
    .Attr("learning_rate: float = 0.1")
    .Attr("loss: string")
    .Input("layers : string")
    .Input("weights: NumLinears * T")
    .Input("biases:  NumLinears * T")
    .Input("x_train: T")
    .Input("y_train: T")
    .Output("loss_history: T ")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int epochs ;
      TF_RETURN_IF_ERROR(c->GetAttr("epochs", &epochs)) ;
      c->set_output(0, c->Vector(epochs));
      return Status::OK();
    });

enum mlp_layer {
  MLP_LAYER_LINEAR  = 0,
  MLP_LAYER_RELU    = 1,
  MLP_LAYER_SOFTMAX = 2,
  MLP_LAYER_DROPOUT = 3
} ;

enum loss_func {
  LOSS_FUNC_CATEGORICAL_XENT = 0,
} ;

template <typename T>
class VE_MLP_OP : public VEOpKernel {
 public:
  explicit VE_MLP_OP(OpKernelConstruction* context) : VEOpKernel(context) {

    // NumLinears
    OP_REQUIRES_OK(context,
                   context->GetAttr("NumLinears", &NumLinears));

    // batch_size
    OP_REQUIRES_OK(context,
                   context->GetAttr("batch_size", &batch_size));

    OP_REQUIRES(context, batch_size >= 1,
                errors::InvalidArgument("Need batch_size >= 1, got ",
                                        batch_size));

    // epochs
    OP_REQUIRES_OK(context,
                   context->GetAttr("epochs", &epochs));

    OP_REQUIRES(context, epochs >= 1,
                errors::InvalidArgument("Need epochs >= 1, got ",
                                        epochs));

    // learning_rate
    OP_REQUIRES_OK(context,
                   context->GetAttr("learning_rate", &learning_rate));

    OP_REQUIRES(context, learning_rate >= 0.0f,
                errors::InvalidArgument("Need learning_rate >= 0.0f, got ",
                                        learning_rate));

    // loss function
    OP_REQUIRES_OK(context,
                   context->GetAttr("loss", &loss));

  }

  void Compute(OpKernelContext* context) override {

    ArgsImpl<4096> Args = ArgsImpl<4096>() ;
    std::vector<Tensor> TempTensors ;	// for keeping temporal Tensors

    /* Grab Input Tensor (layers) */
    const Tensor& tensor_layers = context->input(0);
    auto layers_flat = tensor_layers.flat<string>() ;

    OP_REQUIRES(
	context, tensor_layers.dims() == 1,
        errors::InvalidArgument("layers is not 1-D Tensor. Instead it has shape ",
                                 tensor_layers.shape().DebugString())) ;

    const int numLayers = tensor_layers.dim_size(0) ;

    std::vector<string> layers ;
    int nLinears = 0 ;

    for(int i=0; i<numLayers; i++) {
      string s{layers_flat(i)} ;
      layers.push_back(s) ;
      if( s.compare("linear") == 0 ) nLinears++ ;
    }

    OP_REQUIRES(context, nLinears == NumLinears,
                errors::InvalidArgument("length of weights and bias [", NumLinears, "]"
                                       " does not match number of 'linear' [", nLinears, "]") ) ;


    /* Grab Input Tensor (weights) */
    std::vector<Tensor> weights ;
    for(int i=0; i<nLinears; i++) {
      const Tensor& w = context->input(i+1) ;
      weights.push_back(w) ;
    }


    /* Grab Input Tensor (biases) */
    std::vector<Tensor> biases ;
    for(int i=0; i<nLinears; i++) {
      const Tensor& b = context->input(i+nLinears+1) ;
      biases.push_back(b) ;
    }


    /* Grab Input Tensor (x_train, y_train) */
    const Tensor& x_train = context->input(2*nLinears+1) ;
    const Tensor& y_train = context->input(2*nLinears+2) ;


    /* check dimension of Tensors */
    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(x_train.shape()),
        errors::InvalidArgument("x_train is not a matrix. Instead it has shape ",
                                 x_train.shape().DebugString()));

    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(y_train.shape()),
        errors::InvalidArgument("y_train is not a matrix. Instead it has shape ",
                                 y_train.shape().DebugString()));
    OP_REQUIRES(
        context, x_train.dim_size(0) == y_train.dim_size(0),
        errors::InvalidArgument("size-incompatible: x_train.dim_size(0) = ", x_train.dim_size(0),
                                ", y_train.dim_size(1) = ", y_train.dim_size(0) )) ;

    const int64_t numData = x_train.dim_size(0) ;
    const int64_t inDim   = x_train.dim_size(1) ;
    const int64_t outDim  = y_train.dim_size(1) ;

    for(int i=0; i<nLinears; i++) {
	OP_REQUIRES(
	    context, TensorShapeUtils::IsMatrix(weights[i].shape()),
	    errors::InvalidArgument("weights[", i, "] is not a matrix. Instead it has shape ",
				     weights[i].shape().DebugString()));
    }

    OP_REQUIRES(
	context, inDim ==  weights[0].dim_size(0),
	errors::InvalidArgument("size-incompatible : x_train.dim_size(1) = ",inDim,
				", weights[0].dim_size(0) = ", weights[0].dim_size(0))) ;
    for(int i=1; i<nLinears; i++) {
      OP_REQUIRES(
  	context, weights[i-1].dim_size(1) ==  weights[i].dim_size(0),
  	errors::InvalidArgument("size-incompatible : weights[", i-1, "].dim_size(1) = ", weights[i-1].dim_size(1),
  				", weights[", i, "].dim_size(0) = ", weights[i].dim_size(0))) ;
    }
    OP_REQUIRES(
	context, weights[nLinears-1].dim_size(1) == outDim,
  	errors::InvalidArgument("size-incompatible : weights[", nLinears-1, "].dim_size(1) = ", weights[nLinears-1].dim_size(1),
  				", y_train.dim_size(1) = ", outDim)) ;

    for(int i=0; i<nLinears; i++) {
      OP_REQUIRES(
	  context, biases[i].dims() == 1,
          errors::InvalidArgument("biases[", i, "] is not 1-D Tensor. Instead it has shape ",
                                   biases[i].shape().DebugString())) ;
      OP_REQUIRES(
	  context, weights[i].dim_size(1) == biases[i].dim_size(0),
	  errors::InvalidArgument("size-incompatible : biases[", i, "].dim_size(0) =", biases[i].dim_size(0),
                             	  ",  weights[", i, "].dim_size(1) = ", weights[i].dim_size(1))) ;
    }

    /* Create an output tensor ( loss_history ) */
    Tensor* loss_history = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({epochs}),
						        &loss_history));

    ///////////////////////////////////////

    /* pack Args : numLayers */
    Args.addArg<int64>(numLayers) ;

    /* pack Args : numData, inDim, outDIm */
    Args.addArg<int64>(numData) ;
    Args.addArg<int64>(inDim) ;
    Args.addArg<int64>(outDim) ;

    /* pack Args : train_data, train_label */
    Args.addArg<Tensor>(x_train) ;
    Args.addArg<Tensor>(y_train) ;

    /* pack Args : each Layers */
    std::vector<int64_t> out_size ;
    std::vector<Tensor>::iterator w_itr = weights.begin() ;
    std::vector<Tensor>::iterator b_itr = biases.begin() ;
    for(int i=0; i<numLayers; i++) {
      std::string& s = layers[i] ;

      if( s.compare("linear") == 0 ) {
	Args.addArg<int64>(MLP_LAYER_LINEAR) ;
	out_size.push_back(w_itr->dim_size(1)) ;

	Args.addArg<Tensor>(*w_itr) ;
	Args.addArg<Tensor>(*b_itr) ;

	Tensor w_grd ;
	OP_REQUIRES_OK(
	    context, context->allocate_temp(DataTypeToEnum<T>::value,
					    w_itr->shape(), &w_grd));
	TempTensors.push_back(w_grd) ;

	Args.addArg<Tensor>(w_grd) ;

	w_itr++ ;
	b_itr++ ;
      }
      else if( s.compare("relu") == 0) {
	Args.addArg<int64>(MLP_LAYER_RELU) ;
	const int64_t dim = out_size.back() ;
	Args.addArg<int64>(dim) ;
	out_size.push_back(dim) ;
      }
      else if( s.compare("softmax") == 0 ) {
	Args.addArg<int64>(MLP_LAYER_SOFTMAX) ;
	const int64_t dim = out_size.back() ;
	Args.addArg<int64>(dim) ;
	out_size.push_back(dim) ;
      }
      else if( s.size() >= 8 && s.substr(0,8).compare("dropout:") == 0  ) {
	Args.addArg<int64>(MLP_LAYER_DROPOUT) ;

	const int64_t dim = out_size.back() ;
	out_size.push_back(dim) ;

	float ratio = std::atof(s.substr(8,s.size()).c_str()) ;
	OP_REQUIRES(
	    context, ratio >= 0.f && ratio < 1.f,
	    errors::InvalidArgument("invalid dropout_ratio : ", ratio )) ;
	Args.addArg<float>(ratio) ;

	Tensor noise ;
	OP_REQUIRES_OK(
	    context, context->allocate_temp(DataTypeToEnum<T>::value,
		                            TensorShape({batch_size, dim}), &noise));
	TempTensors.push_back(noise) ;

	Args.addArg<Tensor>(noise) ;
      }
      else {
	OP_REQUIRES(
	    context, false,
	    errors::InvalidArgument("unknown layer :", s)) ;
      }
    }

    /* pack Args : loss function */
    if( loss.compare("categorical_crossentropy") == 0 ) {
	Args.addArg<int64>(LOSS_FUNC_CATEGORICAL_XENT) ;
	const int64_t dim = out_size.back() ;
	Args.addArg<int64>(dim) ;
	out_size.push_back(dim) ;
    }
    else
    {
      OP_REQUIRES(
	  context, false,
	  errors::InvalidArgument("unknown loss function :", loss)) ;
    }

    /* pack Args : fwd/bwd buffer */
    {
      Tensor fwdBuffer ;
      OP_REQUIRES_OK(
	  context, context->allocate_temp(DataTypeToEnum<T>::value,
	                                  TensorShape({batch_size, inDim}), &fwdBuffer));
      TempTensors.push_back(fwdBuffer) ;

      Args.addArg<Tensor>(fwdBuffer) ;
    }
    for(int i=0; i<numLayers; i++) {
      Tensor fwdBuffer ;
      OP_REQUIRES_OK(
	  context, context->allocate_temp(DataTypeToEnum<T>::value,
	                                  TensorShape({batch_size, out_size[i]}), &fwdBuffer));
      TempTensors.push_back(fwdBuffer) ;

      Args.addArg<Tensor>(fwdBuffer) ;

      Tensor bwdBuffer ;
      OP_REQUIRES_OK(
	  context, context->allocate_temp(DataTypeToEnum<T>::value,
	                                  TensorShape({batch_size, out_size[i]}), &bwdBuffer));
      TempTensors.push_back(bwdBuffer) ;

      Args.addArg<Tensor>(bwdBuffer) ;
    }

    /* pack Args : Training Parameter ( batch_size, learning_rate) */
    Args.addArg<int64>(batch_size) ;
    Args.addArg<float>(learning_rate) ;

    /* pack Args : scratch */
    {
      Tensor scratch ;
      OP_REQUIRES_OK(
	  context, context->allocate_temp(DataTypeToEnum<int64>::value,
	                                  TensorShape({numData}), &scratch));
      TempTensors.push_back(scratch) ;

      Args.addArg<Tensor>(scratch) ;
    }

    /* pack Args : loss history */
    Args.addArg<Tensor>(*loss_history) ;

    Call(context, "MLP", Args) ;

  }

 private :
  int   NumLinears ;
  int   batch_size ;
  int   epochs ;
  float learning_rate ;
  string loss ;
};


#ifdef TENSORFLOW_USE_VE
#define VE_REGISTER_KENREL(type)			\
REGISTER_KERNEL_BUILDER(Name("MLP")			\
                          .Device(DEVICE_VE)		\
			  .TypeConstraint<type>("T") 	\
			  .HostMemory("layers"),        \
			  VE_MLP_OP<type>)

VE_REGISTER_KENREL(float) ;
#undef VE_REGSITER_KERNEL
#endif
