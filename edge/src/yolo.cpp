// yolo.cpp — see yolo.h
#include "yolo.h"
#include <cviruntime.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

const char *COCO80[80] = {
"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
"traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
"dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
"baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
"broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
"bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
"microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
"teddy bear","hair drier","toothbrush"};

static inline float sigmoidf(float x){ return 1.f/(1.f+expf(-x)); }

bool Yolo::load(const char *path){
    CVI_MODEL_HANDLE m=nullptr;
    if(CVI_NN_RegisterModel(path,&m)!=CVI_RC_SUCCESS){ printf("[yolo] RegisterModel failed: %s\n",path); return false; }
    CVI_TENSOR *in,*out; int nin,nout;
    if(CVI_NN_GetInputOutputTensors(m,&in,&nin,&out,&nout)!=CVI_RC_SUCCESS){ printf("[yolo] GetIO failed\n"); CVI_NN_CleanupModel(m); return false; }
    if(nin!=1 || nout!=6){ printf("[yolo] unexpected nin=%d nout=%d\n",nin,nout); CVI_NN_CleanupModel(m); return false; }
    model_=m; in_=in; outs_=out; nout_=nout;
    // input quant: int8 = (pixel/255) * qscale  (zp=0 symmetric)
    float qs = in[0].qscale;            // ~126.999
    for(int p=0;p<256;p++){
        int v = (int)lroundf((p/255.0f)*qs);
        if(v>127)v=127; if(v<-128)v=-128;
        lut_[p]=(signed char)v;
    }
    printf("[yolo] loaded %s in=%s qscale=%.4f\n",path,in[0].name,qs);
    return true;
}

void Yolo::unload(){ if(model_){ CVI_NN_CleanupModel((CVI_MODEL_HANDLE)model_); model_=nullptr; } }

static float iou(const Det&a,const Det&b){
    float xx1=std::max(a.x1,b.x1), yy1=std::max(a.y1,b.y1);
    float xx2=std::min(a.x2,b.x2), yy2=std::min(a.y2,b.y2);
    float w=std::max(0.f,xx2-xx1), h=std::max(0.f,yy2-yy1);
    float inter=w*h;
    float ua=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter;
    return ua>0? inter/ua : 0.f;
}

int Yolo::infer(const uint8_t *r,const uint8_t *g,const uint8_t *b,
                int sr,int sg,int sb,float confThr,float iouThr,std::vector<Det>&out){
    out.clear();
    if(!model_) return 0;
    const int D=640;
    CVI_TENSOR *in=(CVI_TENSOR*)in_;
    signed char *dst=(signed char*)CVI_NN_TensorPtr(in);   // NCHW: R plane, G plane, B plane
    signed char *dR=dst, *dG=dst+D*D, *dB=dst+2*D*D;
    for(int y=0;y<D;y++){
        const uint8_t *pr=r+y*sr,*pg=g+y*sg,*pb=b+y*sb;
        signed char *oR=dR+y*D,*oG=dG+y*D,*oB=dB+y*D;
        for(int x=0;x<D;x++){ oR[x]=lut_[pr[x]]; oG[x]=lut_[pg[x]]; oB[x]=lut_[pb[x]]; }
    }
    if(CVI_NN_Forward((CVI_MODEL_HANDLE)model_,in,1,(CVI_TENSOR*)outs_,nout_)!=CVI_RC_SUCCESS){
        printf("[yolo] Forward failed\n"); return 0;
    }
    CVI_TENSOR *outs=(CVI_TENSOR*)outs_;
    // group the 6 outputs into (box[64ch], cls[80ch]) per grid; stride = 640/G
    std::vector<Det> cand;
    for(int g0=0;g0<3;g0++){
        int G = (g0==0)?80:(g0==1)?40:20;
        int stride = D/G;
        const float *box=nullptr,*cls=nullptr;
        for(int i=0;i<nout_;i++){
            CVI_SHAPE s=outs[i].shape; // [1,C,G,G]
            if(s.dim[2]!=G) continue;
            if(s.dim[1]==64) box=(const float*)CVI_NN_TensorPtr(&outs[i]);
            else if(s.dim[1]==80) cls=(const float*)CVI_NN_TensorPtr(&outs[i]);
        }
        if(!box||!cls) continue;
        const int HW=G*G;
        for(int gy=0;gy<G;gy++) for(int gx=0;gx<G;gx++){
            int cell=gy*G+gx;
            // best class (argmax over logits, then one sigmoid)
            float best=-1e9f; int bc=-1;
            for(int c=0;c<80;c++){ float v=cls[c*HW+cell]; if(v>best){best=v;bc=c;} }
            float conf=sigmoidf(best);
            if(conf<confThr) continue;
            // DFL: 4 sides * 16 bins
            float d[4];
            for(int k=0;k<4;k++){
                const float *bb=box+(k*16)*HW+cell;
                float mx=-1e9f;
                for(int j=0;j<16;j++){ float v=bb[j*HW]; if(v>mx)mx=v; }
                float sum=0,acc=0;
                for(int j=0;j<16;j++){ float e=expf(bb[j*HW]-mx); sum+=e; acc+=e*j; }
                d[k]=acc/sum;
            }
            float ax=gx+0.5f, ay=gy+0.5f;
            Det det;
            det.x1=(ax-d[0])*stride; det.y1=(ay-d[1])*stride;
            det.x2=(ax+d[2])*stride; det.y2=(ay+d[3])*stride;
            det.conf=conf; det.cls=bc;
            cand.push_back(det);
        }
    }
    // class-aware NMS
    std::sort(cand.begin(),cand.end(),[](const Det&a,const Det&b){return a.conf>b.conf;});
    std::vector<char> dead(cand.size(),0);
    for(size_t i=0;i<cand.size();i++){
        if(dead[i])continue; out.push_back(cand[i]);
        for(size_t j=i+1;j<cand.size();j++){
            if(dead[j])continue;
            if(cand[j].cls==cand[i].cls && iou(cand[i],cand[j])>iouThr) dead[j]=1;
        }
    }
    return (int)out.size();
}
