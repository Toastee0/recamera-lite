// npu_fwbench.cpp — model-agnostic NPU forward-pass timer (no decode). Registers any
// cvimodel, fills its input buffer from a raw file (content irrelevant for timing),
// runs N CVI_NN_Forward calls, reports ms/frame + FPS. Works for the non-fused INT8
// pose model where bench.cpp (detection decode) and infer_dump.cpp (UINT8 gate) don't.
#include <cviruntime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <sys/time.h>
static double now(){ struct timeval tv; gettimeofday(&tv,NULL); return tv.tv_sec+tv.tv_usec/1e6; }
int main(int argc,char**argv){
  if(argc<3){ printf("usage: %s model.cvimodel input.raw [iters]\n",argv[0]); return 1; }
  int iters = argc>3 ? atoi(argv[3]) : 200;
  CVI_MODEL_HANDLE m=nullptr;
  if(CVI_NN_RegisterModel(argv[1],&m)!=CVI_RC_SUCCESS){ printf("RegisterModel failed\n"); return 2; }
  CVI_TENSOR *in,*out; int nin,nout;
  if(CVI_NN_GetInputOutputTensors(m,&in,&nin,&out,&nout)!=CVI_RC_SUCCESS){ printf("GetIO failed\n"); return 2; }
  // load raw into input[0] (best-effort; timing is content-independent)
  CVI_TENSOR* it=&in[0];
  uint8_t* dst=(uint8_t*)CVI_NN_TensorPtr(it);
  size_t cap=it->mem_size;
  FILE* f=fopen(argv[2],"rb");
  if(f){ size_t n=fread(dst,1,cap,f); fclose(f); if(n<cap) memset(dst+n,0,cap-n); }
  else memset(dst,0,cap);
  printf("model=%s  inputs=%d outputs=%d  iters=%d\n",argv[1],nin,nout,iters);
  for(int i=0;i<5;i++) CVI_NN_Forward(m,in,nin,out,nout);      // warmup
  double t0=now();
  for(int i=0;i<iters;i++) CVI_NN_Forward(m,in,nin,out,nout);
  double dt=now()-t0;
  printf("FORWARD BENCH: %d iters  %.3fs  %.2f ms/frame  %.1f FPS\n",iters,dt,dt/iters*1000.0,iters/dt);
  CVI_NN_CleanupModel(m);
  return 0;
}
