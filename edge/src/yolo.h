// yolo.h — libcviruntime YOLO11 (split detect head) inference + DFL/NMS decode.
// Camera-independent: takes a 640x640 letterboxed planar-RGB uint8 frame, returns
// detections in 640-letterbox pixel coords. Map back to source with the letterbox
// scale/pad the caller used to build the frame.
#ifndef YOLO_H_
#define YOLO_H_
#include <cstdint>
#include <vector>

struct Det { float x1, y1, x2, y2, conf; int cls; };

extern const char *COCO80[80];

class Yolo {
public:
    bool load(const char *modelPath);
    void unload();
    // R/G/B planes, each 640 rows of `stride` bytes (stride>=640). Plane order RGB.
    // Runs preprocess(LUT)->forward->decode->NMS. Returns count, fills `out`.
    int infer(const uint8_t *r, const uint8_t *g, const uint8_t *b,
              int strideR, int strideG, int strideB,
              float confThr, float iouThr, std::vector<Det> &out);
    int inDim() const { return 640; }
private:
    void *model_ = nullptr;
    void *in_ = nullptr;          // CVI_TENSOR* input
    void *outs_ = nullptr;        // CVI_TENSOR* outputs array
    int nout_ = 0;
    signed char lut_[256];        // pixel(0..255) -> int8 quantized (pixel/255*qscale)
};

#endif
