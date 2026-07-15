// npu_posedump.cpp — run the non-fused INT8 pose model on the NPU and dump every
// output tensor (FP32) to <prefix>.out<i>.bin + <prefix>.manifest.txt. Input is the
// daemon's planar INT8 [1,3,640,640] (NOT infer_dump's UINT8 fused input).
#include <cviruntime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
int main(int argc,char**argv){
  if(argc<4){ printf("usage: %s model.cvimodel input_int8.raw out_prefix\n",argv[0]); return 1; }
  CVI_MODEL_HANDLE m=nullptr;
  if(CVI_NN_RegisterModel(argv[1],&m)!=CVI_RC_SUCCESS){ printf("RegisterModel failed\n"); return 2; }
  CVI_TENSOR *in,*out; int nin,nout;
  if(CVI_NN_GetInputOutputTensors(m,&in,&nin,&out,&nout)!=CVI_RC_SUCCESS){ printf("GetIO failed\n"); return 2; }
  CVI_TENSOR* it=&in[0];
  uint8_t* dst=(uint8_t*)CVI_NN_TensorPtr(it);
  size_t cap=it->mem_size;
  FILE* f=fopen(argv[2],"rb");
  if(!f){ printf("cannot open %s\n",argv[2]); return 2; }
  size_t n=fread(dst,1,cap,f); fclose(f);
  printf("input %s: read %zu / %zu bytes\n", argv[2], n, cap);
  if(CVI_NN_Forward(m,in,nin,out,nout)!=CVI_RC_SUCCESS){ printf("Forward failed\n"); return 2; }
  char path[512]; snprintf(path,sizeof path,"%s.manifest.txt",argv[3]);
  FILE* mf=fopen(path,"w");
  for(int i=0;i<nout;i++){
    CVI_TENSOR* o=&out[i];
    snprintf(path,sizeof path,"%s.out%d.bin",argv[3],i);
    FILE* of=fopen(path,"wb"); fwrite(CVI_NN_TensorPtr(o),1,o->count*sizeof(float),of); fclose(of);
    fprintf(mf,"%d %s",i,o->name);
    for(size_t d=0; d<o->shape.dim_size; d++) fprintf(mf," %d",o->shape.dim[d]);
    fprintf(mf,"\n");
    printf("out%d %s count=%zu\n",i,o->name,o->count);
  }
  fclose(mf);
  CVI_NN_CleanupModel(m);
  return 0;
}
