// vision.cpp — unified reCamera vision daemon (Phase 3).
// One VI -> VPSS group with two channels:
//   CHN0  1920x1080 NV12  -> VENC H.264 -> cvi_rtsp  (rtsp://<ip>:8554/<codec>)
//   CHN1   640x640  RGB888-planar (HW letterbox) -> libcviruntime yolo11n NPU
// NPU thread runs on-device salience detection; debounced START/END events are
// logged (Phase 4 will turn these into UDP datagrams; Phase 5 gates yolo11x).
//
// Built on the proven rtspd.cpp pipeline. Sensor selectable (-ov5647 / -gc2053).

#include <iostream>
#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <sys/stat.h>

#include <sample_comm.h>
#include <cvi_rtsp/rtsp.h>
#include "yolo.h"

static bool gRun = true;
static SAMPLE_VI_CONFIG_S gViConfig;
static CVI_RTSP_CTX *gCTX = NULL;
static std::string gCodec = "h264";
static int gPort = 8554;
static bool gNpuOn = true, gRtspOn = true;
static CVI_S16 gLanes[5] = {2, 0, 3, -1, -1};
static SAMPLE_SNS_TYPE_E gSnsType = OV_OV5647_MIPI_2M_30FPS_10BIT;
static CVI_S32 gI2cAddr = 0x36;
static CVI_U8  gOrien   = 1;
static int     gOrienOverride = -1;   // -orien N (0=normal 1=mirror 2=flip 3=mirror+flip); -1 = sensor default
static int     gRbSwap  = -1;         // -rbswap 0|1 (or /userdata/rbswap): swap ISP Bayer R<->B for an
                                      // orientation-INDEPENDENT red/blue channel swap. -1 = per-sensor default.
static const char *gSnsName = "OV5647";
static const char *gPqBin = "/mnt/cfg/param/ov_ov5647_sdr.bin";
static const char *gModel = "/userdata/Models/yolo11n_ours.cvimodel";
static float gSalConf = 0.40f;   // salience confidence threshold
static bool  gSnap = false;      // -snap: dump first person frame + dets JSON, then exit
static double gWatch = 0;        // -watch N: log every frame's detections for N s, then exit
// Phase 4: UDP JSON salience events -> coordinator
static const char *gCamName = "recamera";
static const char *gEvtHost = NULL;   // -evt host:port
static int   gEvtPort = 23906;
static int   gEvtSock = -1;
static struct sockaddr_in gEvtAddr;
#define HB_SEC 2.0                     // heartbeat cadence while active
// Phase 5: threat-driven JPEG snapshots to tmpfs (FPS = threat level)
static bool  gJpegOn = true;
static const char *gJpegDir = "/run/vis";
static int   gJpegQ = 80;              // JPEG quality
static float gFpsMin = 2.0f;           // floor while any salience present
static float gFpsMax = 12.0f;          // cap (wired HQ can raise via -fpsmax)

#define W 1920
#define H 1080
#define NPU_DIM 640
#define JPEG_W 1280
#define JPEG_H 720
#define VENC_CHN0 0
#define VENC_JPEG 1    // JPEG snapshot encoder
#define VPSS_GRP0 0
#define VPSS_CHN0 0    // 1080p NV12 -> VENC (H.264/RTSP)
#define VPSS_CHN1 1    // 640x640 RGB planar -> NPU
#define VPSS_CHN2 2    // 720p NV12 -> JPEG snapshots (threat-gated)

// debounce
#define START_N 2      // consecutive salient frames to fire START
#define END_N   15     // consecutive empty frames to fire END

static chnInputCfg gIc;

// Sensor selection: an explicit -gc2053/-ov5647 flag wins; otherwise read the descriptor
// written by camdetect (/userdata/sensor.conf, SENSOR=GC2053|OV5647) so vision is a
// no-flag drop-in across boards. Defaults to OV5647 if neither is present.
static bool gSensorSet=false;
static void setSensor(const char*name){
    if(!strcmp(name,"GC2053")){ gSnsType=GCORE_GC2053_MIPI_2M_30FPS_10BIT; gI2cAddr=0x3f; gOrien=0; gSnsName="GC2053"; gPqBin="/mnt/cfg/param/gcore_gc2053_sdr.bin"; }
    else if(!strcmp(name,"OV5647")){ gSnsType=OV_OV5647_MIPI_2M_30FPS_10BIT; gI2cAddr=0x36; gOrien=1; gSnsName="OV5647"; gPqBin="/mnt/cfg/param/ov_ov5647_sdr.bin"; }
    else return;
    gSensorSet=true;
}
static bool readSensorConf(const char*path){
    FILE*f=fopen(path,"r"); if(!f) return false;
    char line[160];
    while(fgets(line,sizeof(line),f)){
        char*p=strstr(line,"SENSOR=");
        if(p){ char nm[32]={0}; if(sscanf(p+7,"%31[A-Za-z0-9]",nm)==1) setSensor(nm); }
    }
    fclose(f); return gSensorSet;
}

// --- salience class allowlist (presence/security/catio) ---
static bool isSalient(int c){
    switch(c){ case 0: case 1: case 2: case 3: case 5: case 7: // person,bicycle,car,motorcycle,bus,truck
        case 14: case 15: case 16: case 17: case 18: case 19: case 21: // bird,cat,dog,horse,sheep,cow,bear
        return true; default: return false; }
}

// ---------- pipeline init (from rtspd.cpp) ----------
static void initInputCfg(chnInputCfg *ic){
    memset(ic,0,sizeof(*ic));
    strcpy(ic->codec, gCodec=="h264"?"264":"265");
    ic->rcMode=0; ic->iqp=38; ic->pqp=38; ic->gop=60; ic->bitrate=4000;
    ic->firstFrmstartQp = gCodec=="h264"?51:63;
    ic->num_frames=-1; ic->framerate=30; ic->maxQp=42; ic->minQp=26;
    ic->maxIqp=42; ic->minIqp=26; ic->maxIprop=100; ic->minIprop=1;
    ic->statTime=2; ic->initialDelay=1000; ic->s32MaxReEncodeTimes=0;
}

static void loadPqBin(const char *path){
    FILE *fp=fopen(path,"rb"); if(!fp){ printf("[vision] PQ bin %s missing\n",path); return; }
    fseek(fp,0L,SEEK_END); long sz=ftell(fp); rewind(fp);
    CVI_U8 *buf=(CVI_U8*)malloc(sz);
    if(buf && fread(buf,sz,1,fp)==1){ CVI_BIN_SetBinName(WDR_MODE_NONE,path);
        CVI_S32 r=CVI_BIN_ImportBinData(buf,(CVI_U32)sz);
        printf("[vision] PQ bin %s (%ld) rc=0x%x\n",path,sz,r); }
    free(buf); fclose(fp);
}

// --- Bayer R/B fix via linker --wrap (gRbSwap): the reCamera's OV5647 raw Bayer is
// RG(RGGB) but the SDK's per-sensor default resolves it to BG(BGGR) -> a clean,
// orientation-independent red/blue channel swap (the GC2053 is unaffected). Correct
// it across the R/B diagonal (v -> 3-v) at the SDK's sensor->attr resolvers, so it's
// applied BEFORE the ISP initialises (AE/AWB come up normally; no post-init restart).
// The VI device enBayerFormat is the demosaic determinant; the ISP pub enBayer is
// kept in step. Needs -Wl,--wrap on both resolvers (see the Makefile).
extern "C" CVI_S32 __real_SAMPLE_COMM_VI_GetDevAttrBySns(SAMPLE_SNS_TYPE_E, VI_DEV_ATTR_S*);
extern "C" CVI_S32 __wrap_SAMPLE_COMM_VI_GetDevAttrBySns(SAMPLE_SNS_TYPE_E t, VI_DEV_ATTR_S *a){
    CVI_S32 r=__real_SAMPLE_COMM_VI_GetDevAttrBySns(t,a);
    if(gRbSwap==1 && a){ int o=(int)a->enBayerFormat; a->enBayerFormat=(BAYER_FORMAT_E)(3-o);
        printf("[vision] rbswap: VI dev Bayer %d -> %d\n",o,(int)a->enBayerFormat); }
    return r;
}
extern "C" CVI_S32 __real_SAMPLE_COMM_ISP_GetIspAttrBySns(SAMPLE_SNS_TYPE_E, ISP_PUB_ATTR_S*);
extern "C" CVI_S32 __wrap_SAMPLE_COMM_ISP_GetIspAttrBySns(SAMPLE_SNS_TYPE_E t, ISP_PUB_ATTR_S *a){
    CVI_S32 r=__real_SAMPLE_COMM_ISP_GetIspAttrBySns(t,a);
    if(gRbSwap==1 && a){ int o=(int)a->enBayer; a->enBayer=(ISP_BAYER_FORMAT_E)(3-o); }
    return r;
}

// Custom VB pool config: SAMPLE_PLAT_SYS_INIT makes ONE 3-block pool sized for the
// 1080p channel — too few once a 2nd VPSS channel exists (VPSS_BUF_EMPTY 0xc006800e).
// We make two pools: pool0 = 1080p NV12 (VI raw + CHN0/VENC), pool1 = 640x640 RGB
// planar (NPU CHN1). VPSS auto-selects the smallest pool whose block fits the request.
// Also avoids the SDK's SA_RESETHAND signal handler (keeps our clean shutdown).
static CVI_S32 initSys(SIZE_S stSize){
    VB_CONFIG_S vb; memset(&vb,0,sizeof(vb));
    CVI_U32 nv12 = COMMON_GetPicBufferSize(stSize.u32Width, stSize.u32Height, PIXEL_FORMAT_NV12,
                       DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN);
    CVI_U32 rgbp = COMMON_GetPicBufferSize(NPU_DIM, NPU_DIM, PIXEL_FORMAT_RGB_888_PLANAR,
                       DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN);
    CVI_U32 jpg = COMMON_GetPicBufferSize(JPEG_W, JPEG_H, PIXEL_FORMAT_NV12,
                       DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN);
    // ION carveout is a HARD 60MB; the JPEG VENC's internal jpeg_ion alone is ~9MB and the
    // NPU needs several MB at forward time. Keep VB pools lean or NPU mem_alloc asserts.
    int np = 1;
    vb.astCommPool[0].u32BlkSize = nv12; vb.astCommPool[0].u32BlkCnt = gJpegOn?4:5;
    vb.astCommPool[0].enRemapMode = VB_REMAP_MODE_CACHED;
    if(gNpuOn){
        vb.astCommPool[np].u32BlkSize = rgbp; vb.astCommPool[np].u32BlkCnt = 3;
        vb.astCommPool[np].enRemapMode = VB_REMAP_MODE_CACHED; np++;
    }
    if(gNpuOn && gJpegOn){
        vb.astCommPool[np].u32BlkSize = jpg; vb.astCommPool[np].u32BlkCnt = 2;
        vb.astCommPool[np].enRemapMode = VB_REMAP_MODE_CACHED; np++;
    }
    vb.u32MaxPoolCnt = np;
    printf("[vision] VB pools(%d): nv12=%u x%u  rgbp=%u x%u  jpg720=%u x%u (npu=%d jpeg=%d)\n",
           np, nv12, vb.astCommPool[0].u32BlkCnt, rgbp, gNpuOn?vb.astCommPool[1].u32BlkCnt:0,
           jpg, (gNpuOn&&gJpegOn)?vb.astCommPool[np-1].u32BlkCnt:0, gNpuOn, gJpegOn);
    return SAMPLE_COMM_SYS_Init(&vb);
}

static CVI_S32 initVI(void){
    SAMPLE_INI_CFG_S ini; SIZE_S stSize; PIC_SIZE_E enPicSize; CVI_S32 r;
    memset(&ini,0,sizeof(ini));
    ini.enSource=VI_PIPE_FRAME_SOURCE_DEV; ini.devNum=1; ini.u8UseMultiSns=0;
    ini.enSnsType[0]=gSnsType; ini.enWDRMode[0]=WDR_MODE_NONE; ini.s32BusId[0]=2;
    ini.s32SnsI2cAddr[0]=gI2cAddr; ini.MipiDev[0]=0; ini.u8Orien[0]=gOrien;
    const CVI_S8 pn[5]={0,0,0,0,0};
    memcpy(ini.as16LaneId[0],gLanes,sizeof(gLanes));
    memcpy(ini.as8PNSwap[0],pn,sizeof(pn));
    ini.stMclkAttr[0].bMclkEn=1; ini.stMclkAttr[0].u8Mclk=0;
    printf("[vision] sensor=%s i2c=0x%x lanes=%d,%d,%d,%d,%d\n",gSnsName,gI2cAddr,
           gLanes[0],gLanes[1],gLanes[2],gLanes[3],gLanes[4]);
    if((r=SAMPLE_COMM_VI_IniToViCfg(&ini,&gViConfig))!=CVI_SUCCESS) return r;
    if((r=SAMPLE_COMM_VI_GetSizeBySensor(ini.enSnsType[0],&enPicSize))!=CVI_SUCCESS) return r;
    if((r=SAMPLE_COMM_SYS_GetPicSize(enPicSize,&stSize))!=CVI_SUCCESS) return r;
    printf("[vision] sensor size %ux%u\n",stSize.u32Width,stSize.u32Height);
    if((r=initSys(stSize))!=CVI_SUCCESS){ printf("[vision] SYS_INIT 0x%x\n",r); return r; }
    if((r=SAMPLE_PLAT_VI_INIT(&gViConfig))!=CVI_SUCCESS){ printf("[vision] VI_INIT 0x%x\n",r); return r; }
    loadPqBin(gPqBin);
    return CVI_SUCCESS;
}

static CVI_S32 initVPSS(void){
    VPSS_GRP_ATTR_S grpAttr;
    VPSS_CHN_ATTR_S chnAttr[VPSS_MAX_PHY_CHN_NUM];
    CVI_BOOL chnEnable[VPSS_MAX_PHY_CHN_NUM]={0};
    CVI_S32 r;

    memset(&grpAttr,0,sizeof(grpAttr));
    grpAttr.stFrameRate.s32SrcFrameRate=-1; grpAttr.stFrameRate.s32DstFrameRate=-1;
    grpAttr.enPixelFormat=PIXEL_FORMAT_NV12; grpAttr.u32MaxW=W; grpAttr.u32MaxH=H; grpAttr.u8VpssDev=0;

    CVI_VPSS_StopGrp(VPSS_GRP0); CVI_VPSS_DestroyGrp(VPSS_GRP0);

    memset(chnAttr,0,sizeof(chnAttr));
    // CHN0: 1080p NV12 -> VENC
    chnEnable[VPSS_CHN0]=CVI_TRUE;
    chnAttr[VPSS_CHN0].u32Width=W; chnAttr[VPSS_CHN0].u32Height=H;
    chnAttr[VPSS_CHN0].enVideoFormat=VIDEO_FORMAT_LINEAR;
    chnAttr[VPSS_CHN0].enPixelFormat=PIXEL_FORMAT_NV12;
    chnAttr[VPSS_CHN0].stFrameRate.s32SrcFrameRate=30; chnAttr[VPSS_CHN0].stFrameRate.s32DstFrameRate=30;
    chnAttr[VPSS_CHN0].u32Depth=1;
    chnAttr[VPSS_CHN0].stAspectRatio.enMode=ASPECT_RATIO_AUTO;
    chnAttr[VPSS_CHN0].stAspectRatio.bEnableBgColor=CVI_TRUE;
    chnAttr[VPSS_CHN0].stNormalize.bEnable=CVI_FALSE;

    // CHN1: 640x640 RGB planar, HW letterbox (keep aspect, 114 gray pad) -> NPU
    if(gNpuOn){
        chnEnable[VPSS_CHN1]=CVI_TRUE;
        chnAttr[VPSS_CHN1].u32Width=NPU_DIM; chnAttr[VPSS_CHN1].u32Height=NPU_DIM;
        chnAttr[VPSS_CHN1].enVideoFormat=VIDEO_FORMAT_LINEAR;
        chnAttr[VPSS_CHN1].enPixelFormat=PIXEL_FORMAT_RGB_888_PLANAR;
        chnAttr[VPSS_CHN1].stFrameRate.s32SrcFrameRate=30; chnAttr[VPSS_CHN1].stFrameRate.s32DstFrameRate=15;
        chnAttr[VPSS_CHN1].u32Depth=1;
        chnAttr[VPSS_CHN1].stAspectRatio.enMode=ASPECT_RATIO_AUTO;
        chnAttr[VPSS_CHN1].stAspectRatio.bEnableBgColor=CVI_TRUE;
        chnAttr[VPSS_CHN1].stAspectRatio.u32BgColor=0x00727272;  // 114,114,114
        chnAttr[VPSS_CHN1].stNormalize.bEnable=CVI_FALSE;
    }

    // CHN2: 720p NV12 -> JPEG snapshots (threat-gated; grabbed only when capturing)
    if(gNpuOn && gJpegOn){
        chnEnable[VPSS_CHN2]=CVI_TRUE;
        chnAttr[VPSS_CHN2].u32Width=JPEG_W; chnAttr[VPSS_CHN2].u32Height=JPEG_H;
        chnAttr[VPSS_CHN2].enVideoFormat=VIDEO_FORMAT_LINEAR;
        chnAttr[VPSS_CHN2].enPixelFormat=PIXEL_FORMAT_NV12;
        chnAttr[VPSS_CHN2].stFrameRate.s32SrcFrameRate=30; chnAttr[VPSS_CHN2].stFrameRate.s32DstFrameRate=15;
        chnAttr[VPSS_CHN2].u32Depth=1;
        chnAttr[VPSS_CHN2].stAspectRatio.enMode=ASPECT_RATIO_AUTO;
        chnAttr[VPSS_CHN2].stAspectRatio.bEnableBgColor=CVI_TRUE;
        chnAttr[VPSS_CHN2].stAspectRatio.u32BgColor=0x00000000;
        chnAttr[VPSS_CHN2].stNormalize.bEnable=CVI_FALSE;
    }

    if((r=SAMPLE_COMM_VPSS_Init(VPSS_GRP0,chnEnable,&grpAttr,chnAttr))!=CVI_SUCCESS){ printf("[vision] VPSS_Init 0x%x\n",r); return r; }
    if((r=SAMPLE_COMM_VPSS_Start(VPSS_GRP0,chnEnable,&grpAttr,chnAttr))!=CVI_SUCCESS){ printf("[vision] VPSS_Start 0x%x\n",r); return r; }
    if((r=SAMPLE_COMM_VI_Bind_VPSS(0,0,VPSS_GRP0))!=CVI_SUCCESS){ printf("[vision] VI_Bind 0x%x\n",r); return r; }
    return CVI_SUCCESS;
}

static CVI_S32 initVENC(void){
    PAYLOAD_TYPE_E pl=gCodec=="h264"?PT_H264:PT_H265;
    VENC_GOP_ATTR_S gop; CVI_S32 r;
    initInputCfg(&gIc);
    if((r=SAMPLE_COMM_VENC_GetGopAttr(VENC_GOPMODE_NORMALP,&gop))!=CVI_SUCCESS) return r;
    if((r=SAMPLE_COMM_VENC_Start(&gIc,VENC_CHN0,pl,PIC_1080P,(SAMPLE_RC_E)gIc.rcMode,0,CVI_FALSE,&gop))!=CVI_SUCCESS){ printf("[vision] VENC_Start 0x%x\n",r); return r; }
    return CVI_SUCCESS;
}

static CVI_S32 initJpegVenc(void){
    chnInputCfg ic; memset(&ic,0,sizeof(ic));
    strcpy(ic.codec,"jpg");
    ic.quality   = gJpegQ;     // 1..100
    ic.MCUPerECS = 0;
    ic.num_frames= -1;
    ic.framerate = 30;
    VENC_GOP_ATTR_S gop; CVI_S32 r;
    if((r=SAMPLE_COMM_VENC_GetGopAttr(VENC_GOPMODE_NORMALP,&gop))!=CVI_SUCCESS) return r;
    if((r=SAMPLE_COMM_VENC_Start(&ic,VENC_JPEG,PT_JPEG,PIC_720P,(SAMPLE_RC_E)0,0,CVI_FALSE,&gop))!=CVI_SUCCESS){
        printf("[vision] JPEG VENC_Start 0x%x\n",r); return r; }
    printf("[vision] JPEG snapshot encoder ready (720p q%d) -> %s\n",gJpegQ,gJpegDir);
    return CVI_SUCCESS;
}

// HW-JPEG encode an already-grabbed 720p frame, atomic-write to tmpfs (lighttpd serves it).
static bool encodeJpeg(VIDEO_FRAME_INFO_S *fr, const char *camName){
    if(CVI_VENC_SendFrame(VENC_JPEG,fr,2000)!=CVI_SUCCESS) return false;
    VENC_CHN_STATUS_S st;
    if(CVI_VENC_QueryStatus(VENC_JPEG,&st)!=CVI_SUCCESS || !st.u32CurPacks) return false;
    VENC_STREAM_S stm; memset(&stm,0,sizeof(stm));
    stm.pstPack=(VENC_PACK_S*)malloc(sizeof(VENC_PACK_S)*st.u32CurPacks);
    if(!stm.pstPack) return false;
    bool ok=false;
    if(CVI_VENC_GetStream(VENC_JPEG,&stm,2000)==CVI_SUCCESS){
        char tmp[160],dst[160];
        snprintf(tmp,sizeof(tmp),"%s/.%s.tmp",gJpegDir,camName);
        snprintf(dst,sizeof(dst),"%s/%s.jpg",gJpegDir,camName);
        FILE*f=fopen(tmp,"wb");
        if(f){
            for(unsigned i=0;i<stm.u32PackCount;i++){ VENC_PACK_S*pk=&stm.pstPack[i];
                fwrite(pk->pu8Addr+pk->u32Offset,1,pk->u32Len-pk->u32Offset,f); }
            fclose(f); rename(tmp,dst); ok=true;   // atomic swap
        }
        CVI_VENC_ReleaseStream(VENC_JPEG,&stm);
    }
    free(stm.pstPack);
    return ok;
}

static void finalize(void){
    CVI_BOOL chnEnable[VPSS_MAX_PHY_CHN_NUM]={0};
    chnEnable[VPSS_CHN0]=CVI_TRUE;
    if(gNpuOn) chnEnable[VPSS_CHN1]=CVI_TRUE;
    if(gNpuOn&&gJpegOn) chnEnable[VPSS_CHN2]=CVI_TRUE;
    SAMPLE_COMM_VI_UnBind_VPSS(0,0,VPSS_GRP0);
    SAMPLE_COMM_VPSS_Stop(VPSS_GRP0,chnEnable);
    if(gRtspOn) SAMPLE_COMM_VENC_Stop(VENC_CHN0);
    if(gNpuOn&&gJpegOn) SAMPLE_COMM_VENC_Stop(VENC_JPEG);
    SAMPLE_COMM_ISP_Stop(0);
    SAMPLE_COMM_VI_DestroyVi(&gViConfig);
    SAMPLE_COMM_SYS_Exit();
}

static void handleSig(int s){ (void)s; gRun=false; }

// ---------- phys->virt map cache (VPSS recycles a small buffer ring) ----------
struct MapEnt { uint64_t phy; uint32_t len; void *vir; };
static MapEnt gMaps[24]; static int gNMap=0;
static void* mapPlane(uint64_t phy,uint32_t len){
    for(int i=0;i<gNMap;i++) if(gMaps[i].phy==phy && gMaps[i].len==len) return gMaps[i].vir;
    void *v=CVI_SYS_Mmap(phy,len);
    if(v && gNMap<(int)(sizeof(gMaps)/sizeof(gMaps[0]))){ gMaps[gNMap++]={phy,len,v}; }
    return v;
}

// ---------- NPU salience thread ----------
static Yolo gYolo;

static double nowSec(){ struct timeval tv; gettimeofday(&tv,NULL); return tv.tv_sec + tv.tv_usec/1e6; }

// Emit a UDP JSON salience datagram. type = "start"|"active"|"end".
// box is in source (1920x1080) coords; label/conf/box ignored for "end".
static unsigned gSeq=0;
// `label/conf/box` = top salient (the gating primary). `dets` = full-scene JSON array
// of ALL detections this frame (source coords) so consumers get everything, not just the top.
static void sendEvt(const char*type,const char*label,float conf,const float box[4],
                    double dur,float threat,float fps,const char*dets){
    if(gEvtSock<0) return;
    char buf[1100]; int n;
    if(!strcmp(type,"end"))
        n=snprintf(buf,sizeof(buf),
            "{\"cam\":\"%s\",\"event\":\"end\",\"ts\":%.3f,\"seq\":%u,\"dur\":%.2f}",
            gCamName,nowSec(),++gSeq,dur);
    else
        n=snprintf(buf,sizeof(buf),
            "{\"cam\":\"%s\",\"event\":\"%s\",\"label\":\"%s\",\"conf\":%.3f,"
            "\"box\":[%.0f,%.0f,%.0f,%.0f],\"threat\":%.2f,\"fps\":%.1f,"
            "\"dets\":%s,\"img\":\"%s.jpg\",\"ts\":%.3f,\"seq\":%u}",
            gCamName,type,label,conf,box[0],box[1],box[2],box[3],threat,fps,
            dets&&*dets?dets:"[]",gCamName,nowSec(),++gSeq);
    if(n>0 && n<(int)sizeof(buf)) sendto(gEvtSock,buf,n,0,(struct sockaddr*)&gEvtAddr,sizeof(gEvtAddr));
}

// Edge threat score in [0,1] from the current salient detection set.
// person>animal/vehicle, closer (box area) = scarier, LOOMING (area growth) dominates,
// more targets + higher conf raise it. Maps to capture/process FPS (tachypsychia).
static float gPrevArea=0;
static float threatScore(const std::vector<Det>&dets,int nd,const Det*best){
    if(!best){ gPrevArea=0; return 0; }
    int scount=0; for(int i=0;i<nd;i++) if(isSalient(dets[i].cls)) scount++;
    float area=((best->x2-best->x1)*(best->y2-best->y1))/(float)(NPU_DIM*NPU_DIM);
    if(area<0)area=0; if(area>1)area=1;
    float looming=area-gPrevArea; if(looming<0)looming=0; gPrevArea=area;
    float classW=(best->cls==0)?1.0f:0.6f;     // person vs animal/vehicle
    float t=classW*(0.25f+1.5f*area+10.0f*looming+0.15f*(scount-1))*(0.5f+0.5f*best->conf);
    return t>1.f?1.f:(t<0.f?0.f:t);
}

// Dump the 640x640 planar-RGB frame (de-strided) + detections JSON for overlay.
static void writeSnap(const uint8_t*pr,const uint8_t*pg,const uint8_t*pb,int sr,int sg,int sb,
                      const std::vector<Det>&dets,int nd){
    const int D=NPU_DIM;
    FILE*fr=fopen("/userdata/snap.rgb","wb");
    if(fr){ for(int y=0;y<D;y++) fwrite(pr+y*sr,1,D,fr);
            for(int y=0;y<D;y++) fwrite(pg+y*sg,1,D,fr);
            for(int y=0;y<D;y++) fwrite(pb+y*sb,1,D,fr); fclose(fr); }
    FILE*fj=fopen("/userdata/snap.json","w");
    if(fj){ fprintf(fj,"[");
        for(int i=0;i<nd;i++){ const Det&d=dets[i];
            fprintf(fj,"%s{\"label\":\"%s\",\"conf\":%.3f,\"box\":[%.1f,%.1f,%.1f,%.1f]}",
                    i?",":"",COCO80[d.cls],d.conf,d.x1,d.y1,d.x2,d.y2); }
        fprintf(fj,"]\n"); fclose(fj); }
    printf("[snap] wrote /userdata/snap.rgb + snap.json (%d dets)\n",nd); fflush(stdout);
}

static void *npuLoop(void *arg){
    (void)arg;
    // letterbox params: source WxH -> NPU_DIM (centered, keep aspect)
    float r_lb = (float)NPU_DIM/(W>H?W:H);
    float nw=W*r_lb, nh=H*r_lb;
    float padX=(NPU_DIM-nw)/2.f, padY=(NPU_DIM-nh)/2.f;

    bool active=false; int presN=0, absN=0;
    unsigned long nfr=0, ndet=0;
    double startTs=0, lastHb=0, lastJpeg=0; Det lastBest{}; bool haveLast=false;
    float threat=0, fps=0, lastThreat=0, lastFps=0;

    std::vector<Det> dets;
    while(gRun){
        VIDEO_FRAME_INFO_S f;
        CVI_S32 r=CVI_VPSS_GetChnFrame(VPSS_GRP0,VPSS_CHN1,&f,1000);
        if(r!=CVI_SUCCESS){ usleep(2000); continue; }
        VIDEO_FRAME_S *vf=&f.stVFrame;
        uint8_t *pr=(uint8_t*)mapPlane(vf->u64PhyAddr[0],vf->u32Length[0]);
        uint8_t *pg=(uint8_t*)mapPlane(vf->u64PhyAddr[1],vf->u32Length[1]);
        uint8_t *pb=(uint8_t*)mapPlane(vf->u64PhyAddr[2],vf->u32Length[2]);
        if(pr&&pg&&pb){
            int nd=gYolo.infer(pr,pg,pb,vf->u32Stride[0],vf->u32Stride[1],vf->u32Stride[2],
                               gSalConf,0.45f,dets);
            // salience: best salient detection this frame
            const Det *best=nullptr;
            for(int i=0;i<nd;i++) if(isSalient(dets[i].cls) && (!best||dets[i].conf>best->conf)) best=&dets[i];
            bool sal = best!=nullptr;
            ndet += nd;

            if(gWatch>0){
                static double wStart=0; if(wStart==0) wStart=nowSec();
                char line[256]; int off=snprintf(line,sizeof(line),"[watch] f=%lu:",nfr);
                for(int i=0;i<nd && off<220;i++)
                    off+=snprintf(line+off,sizeof(line)-off," %s/%.2f",COCO80[dets[i].cls],dets[i].conf);
                if(nd==0) snprintf(line+off,sizeof(line)-off," (none)");
                printf("%s\n",line); fflush(stdout);
                if(nowSec()-wStart>=gWatch){ CVI_VPSS_ReleaseChnFrame(VPSS_GRP0,VPSS_CHN1,&f); gRun=false; break; }
            }

            if(gSnap){
                bool hasPerson=false; for(int i=0;i<nd;i++) if(dets[i].cls==0){hasPerson=true;break;}
                if(hasPerson || nfr>=120){   // first person frame, or give up after ~8s
                    writeSnap(pr,pg,pb,vf->u32Stride[0],vf->u32Stride[1],vf->u32Stride[2],dets,nd);
                    CVI_VPSS_ReleaseChnFrame(VPSS_GRP0,VPSS_CHN1,&f);
                    gRun=false; break;
                }
            }

            double now=nowSec();
            if(sal){ presN++; absN=0; lastBest=*best; haveLast=true; }
            else { absN++; presN=0; }

            // threat -> capture/process FPS (tachypsychia); 0 when nothing salient
            threat = threatScore(dets,nd,best);
            fps = sal ? (gFpsMin + (gFpsMax-gFpsMin)*threat) : 0;
            if(sal){ lastThreat=threat; lastFps=fps; }

            // Always DRAIN CHN2 (else an un-consumed 3rd VPSS channel back-pressures the
            // group and starves CHN0). Only ENCODE on the threat-fps schedule.
            if(gJpegOn){
                VIDEO_FRAME_INFO_S jf;
                if(CVI_VPSS_GetChnFrame(VPSS_GRP0,VPSS_CHN2,&jf,0)==CVI_SUCCESS){
                    if(fps>0 && now-lastJpeg >= 1.0/fps){ if(encodeJpeg(&jf,gCamName)) lastJpeg=now; }
                    CVI_VPSS_ReleaseChnFrame(VPSS_GRP0,VPSS_CHN2,&jf);
                }
            }

            // map a 640-coord det to source (1920x1080) coords
            auto srcBox=[&](const Det*d,float o[4]){
                o[0]=(d->x1-padX)/r_lb; o[1]=(d->y1-padY)/r_lb;
                o[2]=(d->x2-padX)/r_lb; o[3]=(d->y2-padY)/r_lb; };

            // full-scene array: ALL detections this frame (source coords), capped to fit the datagram
            char detsB[900]; int dn=snprintf(detsB,sizeof(detsB),"[");
            for(int i=0;i<nd && dn<820;i++){ float bb[4]; srcBox(&dets[i],bb);
                dn+=snprintf(detsB+dn,sizeof(detsB)-dn,"%s{\"l\":\"%s\",\"c\":%.2f,\"b\":[%.0f,%.0f,%.0f,%.0f]}",
                             i?",":"",COCO80[dets[i].cls],dets[i].conf,bb[0],bb[1],bb[2],bb[3]); }
            snprintf(detsB+dn,sizeof(detsB)-dn,"]");

            if(!active && presN>=START_N){
                active=true; startTs=now; lastHb=now;
                float b[4]; srcBox(best,b);
                printf("[EVENT] START %s conf=%.2f threat=%.2f fps=%.1f (%d dets)\n",
                       COCO80[best->cls],best->conf,threat,fps,nd); fflush(stdout);
                sendEvt("start",COCO80[best->cls],best->conf,b,0,threat,fps,detsB);
            } else if(active && absN>=END_N){
                active=false;
                printf("[EVENT] END (idle %d frames, dur %.1fs)\n",absN,now-startTs); fflush(stdout);
                sendEvt("end",NULL,0,NULL,now-startTs,0,0,NULL);
            } else if(active && haveLast && now-lastHb>=HB_SEC){
                lastHb=now;
                float b[4]; srcBox(&lastBest,b);
                sendEvt("active",COCO80[lastBest.cls],lastBest.conf,b,0,lastThreat,lastFps,detsB);
            }
        }
        CVI_VPSS_ReleaseChnFrame(VPSS_GRP0,VPSS_CHN1,&f);
        if(++nfr%150==0)
            printf("[npu] frames=%lu dets/f=%.1f active=%d threat=%.2f fps=%.1f\n",
                   nfr,(double)ndet/nfr,active,lastThreat,active?lastFps:0);
    }
    return NULL;
}

// ---------- RTSP video thread (from rtspd.cpp) ----------
static void *videoLoop(void *arg){
    CVI_RTSP_SESSION *session=(CVI_RTSP_SESSION*)arg; CVI_S32 r;
    unsigned long nGrabErr=0,nSendErr=0,nWrote=0;
    while(gRun){
        VIDEO_FRAME_INFO_S fr;
        r=CVI_VPSS_GetChnFrame(VPSS_GRP0,VPSS_CHN0,&fr,1000);
        if(r!=CVI_SUCCESS){ if(++nGrabErr%30==1) printf("[rtsp] GetChnFrame 0x%x\n",r); continue; }
        r=CVI_VENC_SendFrame(VENC_CHN0,&fr,20000);
        if(r!=CVI_SUCCESS){ if(++nSendErr%30==1) printf("[rtsp] SendFrame 0x%x\n",r);
            CVI_VPSS_ReleaseChnFrame(VPSS_GRP0,VPSS_CHN0,&fr); continue; }
        VENC_CHN_STATUS_S st;
        if(CVI_VENC_QueryStatus(VENC_CHN0,&st)!=CVI_SUCCESS||!st.u32CurPacks){
            CVI_VPSS_ReleaseChnFrame(VPSS_GRP0,VPSS_CHN0,&fr); continue; }
        VENC_STREAM_S stm; memset(&stm,0,sizeof(stm));
        stm.pstPack=(VENC_PACK_S*)malloc(sizeof(VENC_PACK_S)*st.u32CurPacks);
        if(!stm.pstPack){ CVI_VPSS_ReleaseChnFrame(VPSS_GRP0,VPSS_CHN0,&fr); continue; }
        if(CVI_VENC_GetStream(VENC_CHN0,&stm,-1)==CVI_SUCCESS){
            CVI_RTSP_DATA d; memset(&d,0,sizeof(d)); d.blockCnt=stm.u32PackCount;
            for(unsigned i=0;i<stm.u32PackCount;i++){ VENC_PACK_S*pk=&stm.pstPack[i];
                d.dataPtr[i]=pk->pu8Addr+pk->u32Offset; d.dataLen[i]=pk->u32Len-pk->u32Offset; }
            CVI_RTSP_WriteFrame(gCTX,session->video,&d);
            CVI_VENC_ReleaseStream(VENC_CHN0,&stm);
            if(++nWrote%300==1) printf("[rtsp] wrote=%lu\n",nWrote);
        }
        free(stm.pstPack);
        CVI_VPSS_ReleaseChnFrame(VPSS_GRP0,VPSS_CHN0,&fr);
    }
    return NULL;
}

static void onConnect(const char*ip,void*a){ (void)a; std::cout<<"[rtsp] connect: "<<ip<<std::endl; }
static void onDisconnect(const char*ip,void*a){ (void)a; std::cout<<"[rtsp] disconnect: "<<ip<<std::endl; }

int main(int argc,const char*argv[]){
    setvbuf(stdout,NULL,_IOLBF,0);
    signal(SIGPIPE,SIG_IGN); signal(SIGINT,handleSig); signal(SIGTERM,handleSig);

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"h265")) gCodec="h265";
        else if(!strcmp(argv[i],"-p")&&i+1<argc) gPort=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-nonpu")) gNpuOn=false;
        else if(!strcmp(argv[i],"-nortsp")) gRtspOn=false;
        else if(!strcmp(argv[i],"-model")&&i+1<argc) gModel=argv[++i];
        else if(!strcmp(argv[i],"-salconf")&&i+1<argc) gSalConf=atof(argv[++i]);
        else if(!strcmp(argv[i],"-snap")){ gSnap=true; gRtspOn=false; gJpegOn=false; }   // one-shot capture
        else if(!strcmp(argv[i],"-watch")&&i+1<argc){ gWatch=atof(argv[++i]); gRtspOn=false; gJpegOn=false; }
        else if(!strcmp(argv[i],"-name")&&i+1<argc) gCamName=argv[++i];
        else if(!strcmp(argv[i],"-nojpeg")) gJpegOn=false;
        else if(!strcmp(argv[i],"-fpsmax")&&i+1<argc) gFpsMax=atof(argv[++i]);
        else if(!strcmp(argv[i],"-jpegq")&&i+1<argc) gJpegQ=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-jpegdir")&&i+1<argc) gJpegDir=argv[++i];
        else if(!strcmp(argv[i],"-evt")&&i+1<argc){
            char hp[96]; snprintf(hp,sizeof(hp),"%s",argv[++i]);
            char*c=strchr(hp,':'); if(c){ *c=0; gEvtPort=atoi(c+1); }
            static char host[80]; snprintf(host,sizeof(host),"%s",hp); gEvtHost=host;
        }
        else if(!strcmp(argv[i],"-gc2053")) setSensor("GC2053");
        else if(!strcmp(argv[i],"-ov5647")) setSensor("OV5647");
        else if(!strcmp(argv[i],"-orien")&&i+1<argc){ int o=atoi(argv[++i]); if(o>=0&&o<=3) gOrienOverride=o; }
        else if(!strcmp(argv[i],"-rbswap")&&i+1<argc){ gRbSwap=atoi(argv[++i])?1:0; }
        else if(!strcmp(argv[i],"-lanes")&&i+1<argc){ char b[64]; snprintf(b,sizeof(b),"%s",argv[++i]);
            int n=0; for(char*t=strtok(b,",");t&&n<5;t=strtok(NULL,",")) gLanes[n++]=(CVI_S16)atoi(t); }
    }

    if(!gSensorSet){
        if(readSensorConf("/userdata/sensor.conf")) printf("[vision] sensor=%s (from /userdata/sensor.conf)\n",gSnsName);
        else printf("[vision] no -gc2053/-ov5647 flag or sensor.conf; default %s\n",gSnsName);
    }
    // Orientation override: -orien arg wins; else /userdata/orien file (persists across
    // reboots + survives camdetect regenerating sensor.conf). 0=normal 1=mirror 2=flip
    // 3=mirror+flip. Applied after sensor selection so arg order doesn't matter.
    if(gOrienOverride<0){
        FILE*of=fopen("/userdata/orien","r");
        if(of){ int o; if(fscanf(of,"%d",&o)==1 && o>=0 && o<=3) gOrienOverride=o; fclose(of); }
    }
    if(gOrienOverride>=0){ gOrien=(CVI_U8)gOrienOverride; printf("[vision] orientation override -> %d\n",gOrien); }
    // Bayer R/B swap: -rbswap arg > /userdata/rbswap file > per-sensor default (the
    // reCamera OV5647 needs it; the GC2053 does not).
    if(gRbSwap<0){
        FILE*rf=fopen("/userdata/rbswap","r");
        if(rf){ int v; if(fscanf(rf,"%d",&v)==1) gRbSwap=v?1:0; fclose(rf); }
    }
    if(gRbSwap<0) gRbSwap = (gSnsName && !strcmp(gSnsName,"OV5647")) ? 1 : 0;

    if(gNpuOn){
        if(!gYolo.load(gModel)){ printf("[vision] model load failed; NPU disabled\n"); gNpuOn=false; }
    }

    if(gNpuOn && gEvtHost){
        gEvtSock = socket(AF_INET, SOCK_DGRAM, 0);
        memset(&gEvtAddr,0,sizeof(gEvtAddr));
        gEvtAddr.sin_family = AF_INET;
        gEvtAddr.sin_port = htons(gEvtPort);
        if(inet_pton(AF_INET, gEvtHost, &gEvtAddr.sin_addr) != 1 || gEvtSock < 0){
            printf("[vision] bad -evt target %s:%d; events off\n", gEvtHost, gEvtPort);
            if(gEvtSock>=0){ close(gEvtSock); gEvtSock=-1; }
        } else {
            printf("[vision] events -> udp %s:%d as cam=%s\n", gEvtHost, gEvtPort, gCamName);
        }
    }

    if(gNpuOn && gJpegOn){ mkdir(gJpegDir,0755); }   // tmpfs dir for snapshots (served by lighttpd)

    if(initVI()!=CVI_SUCCESS){ printf("[vision] initVI failed\n"); return 1; }
    if(initVPSS()!=CVI_SUCCESS){ printf("[vision] initVPSS failed\n"); return 1; }
    if(gRtspOn && initVENC()!=CVI_SUCCESS){ printf("[vision] initVENC failed\n"); return 1; }
    if(gNpuOn && gJpegOn && initJpegVenc()!=CVI_SUCCESS){ printf("[vision] JPEG disabled\n"); gJpegOn=false; }

    CVI_RTSP_SESSION *session=NULL;
    if(gRtspOn){
        CVI_RTSP_CONFIG cfg; memset(&cfg,0,sizeof(cfg)); cfg.port=gPort;
        if(CVI_RTSP_Create(&gCTX,&cfg)<0){ printf("[vision] RTSP_Create failed\n"); return 1; }
        CVI_RTSP_SESSION_ATTR attr; memset(&attr,0,sizeof(attr));
        attr.video.codec = gCodec=="h264"?RTSP_VIDEO_H264:RTSP_VIDEO_H265;
        // session name = RTSP path. Use "live0" to match the fleet (packaged rtspd/video_demo
        // serves /live0; cameras.json expects it) so vision is a drop-in stream source.
        snprintf(attr.name,sizeof(attr.name),"live0");
        CVI_RTSP_CreateSession(gCTX,&attr,&session);
        CVI_RTSP_STATE_LISTENER lis; memset(&lis,0,sizeof(lis));
        lis.onConnect=onConnect; lis.argConn=gCTX; lis.onDisconnect=onDisconnect;
        CVI_RTSP_SetListener(gCTX,&lis);
        if(CVI_RTSP_Start(gCTX)<0){ printf("[vision] RTSP_Start failed\n"); return 1; }
        printf("[vision] streaming %s on rtsp://<ip>:%d/live0\n",gCodec.c_str(),gPort);
    }

    pthread_t thV=0, thN=0;
    if(gRtspOn) pthread_create(&thV,NULL,videoLoop,session);
    if(gNpuOn){ printf("[vision] NPU salience active (salconf=%.2f)\n",gSalConf); pthread_create(&thN,NULL,npuLoop,NULL); }

    if(thV) pthread_join(thV,NULL);
    if(thN) pthread_join(thN,NULL);

    if(gRtspOn){ CVI_RTSP_Stop(gCTX); CVI_RTSP_DestroySession(gCTX,session); CVI_RTSP_Destroy(&gCTX); }
    if(gNpuOn) gYolo.unload();
    finalize();
    return 0;
}
