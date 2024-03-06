#include <l4casadi.hpp>

L4CasADi l4casadi("/home/ryan/raigor/train_ws/src/TCN/tcn_with_mlp/l4casadi_mpc_raigor/_l4c_generated", "l4casadi_f", false, "cpu", true, true);

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif


static const casadi_int casadi_s_in0[3] = { 5, 6, 1};
static const casadi_int casadi_s_out0[3] = { 5, 6, 1};


CASADI_SYMBOL_EXPORT int l4casadi_f(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  l4casadi.forward(arg[0], 5, 6, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int jac_l4casadi_f(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  l4casadi.jac(arg[0], 5, 6, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int jac_jac_l4casadi_f(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  l4casadi.hess(arg[0], 5, 6, res[0]);
  return 0;
}

// Only single input, single output is supported at the moment
CASADI_SYMBOL_EXPORT casadi_int l4casadi_f_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int l4casadi_f_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT const casadi_int* l4casadi_f_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s_in0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* l4casadi_f_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s_out0;
    default: return 0;
  }
}

#ifdef __cplusplus
} /* extern "C" */
#endif