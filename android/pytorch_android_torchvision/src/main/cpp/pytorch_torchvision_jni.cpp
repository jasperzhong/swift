#include <cassert>
#include <cmath>
#include <vector>

#include <libyuv.h>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#if defined(__ANDROID__)

#include <android/log.h>
#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "pytorch-vision-jni", __VA_ARGS__)

#endif
#define clamp0255(x) x > 255 ? 255 : x < 0 ? 0 : x

namespace pytorch_vision_jni {
class PytorchVisionJni : public facebook::jni::JavaClass<PytorchVisionJni> {
 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/torchvision/PyTorchVision;";


  static void nativeImageYUV420CenterCropToFloatBuffer(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> yBuffer,
      const int yRowStride,
      const int yPixelStride,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> uBuffer,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> vBuffer,
      const int uRowStride,
      const int uvPixelStride,
      const int imageWidth,
      const int imageHeight,
      const int rotateCWDegrees,
      const int tensorWidth,
      const int tensorHeight,
      facebook::jni::alias_ref<jfloatArray> jnormMeanRGB,
      facebook::jni::alias_ref<jfloatArray> jnormStdRGB,
      facebook::jni::alias_ref<facebook::jni::JBuffer> outBuffer,
      const int outOffset) {
    static JNIEnv* jni = facebook::jni::Environment::current();
    float* outData = (float*)jni->GetDirectBufferAddress(outBuffer.get());

    auto normMeanRGB = jnormMeanRGB->getRegion(0, 3);
    auto normStdRGB = jnormStdRGB->getRegion(0, 3);

    int widthAfterRtn = imageWidth;
    int heightAfterRtn = imageHeight;
    bool oddRotation = rotateCWDegrees == 90 || rotateCWDegrees == 270;
    if (oddRotation) {
      widthAfterRtn = imageHeight;
      heightAfterRtn = imageWidth;
    }

    int cropWidthAfterRtn = widthAfterRtn;
    int cropHeightAfterRtn = heightAfterRtn;

    if (tensorWidth * heightAfterRtn <= tensorHeight * widthAfterRtn) {
      cropWidthAfterRtn = tensorWidth * heightAfterRtn / tensorHeight;
    } else {
      cropHeightAfterRtn = tensorHeight * widthAfterRtn / tensorWidth;
    }

    int cropWidthBeforeRtn = cropWidthAfterRtn;
    int cropHeightBeforeRtn = cropHeightAfterRtn;
    if (oddRotation) {
      cropWidthBeforeRtn = cropHeightAfterRtn;
      cropHeightBeforeRtn = cropWidthAfterRtn;
    }

    const int offsetX = (imageWidth - cropWidthBeforeRtn) / 2.f;
    const int offsetY = (imageHeight - cropHeightBeforeRtn) / 2.f;

    const uint8_t* yData = yBuffer->getDirectBytes();
    const uint8_t* uData = uBuffer->getDirectBytes();
    const uint8_t* vData = vBuffer->getDirectBytes();

    float scale = cropWidthAfterRtn / tensorWidth;
    int uvRowStride = uRowStride >> 1;
    int cropXMult = 1;
    int cropYMult = 1;
    int cropXAdd = offsetX;
    int cropYAdd = offsetY;
    if (rotateCWDegrees == 90) {
      cropYMult = -1;
      cropYAdd = offsetY + (cropHeightBeforeRtn - 1);
    } else if (rotateCWDegrees == 180) {
      cropXMult = -1;
      cropXAdd = offsetX + (cropWidthBeforeRtn - 1);
      cropYMult = -1;
      cropYAdd = offsetY + (cropHeightBeforeRtn - 1);
    } else if (rotateCWDegrees == 270) {
      cropXMult = -1;
      cropXAdd = offsetX + (cropWidthBeforeRtn - 1);
    }

    float normMeanRm255 = 255 * normMeanRGB[0];
    float normMeanGm255 = 255 * normMeanRGB[1];
    float normMeanBm255 = 255 * normMeanRGB[2];
    float normStdRm255 = 255 * normStdRGB[0];
    float normStdGm255 = 255 * normStdRGB[1];
    float normStdBm255 = 255 * normStdRGB[2];

    int cropXAfterRtn, cropYAfterRtn;
    int xBeforeRtn, yBeforeRtn;
    int yIdx, uvIdx;
    int ui, vi;
    int a0;
    int ri, gi, bi;
    int channelSize = tensorWidth * tensorHeight;
    int wr = outOffset;
    int wg = wr + channelSize;
    int wb = wg + channelSize;
    for (int x = 0; x < tensorWidth; x++) {
      for (int y = 0; y < tensorHeight; y++) {
        xBeforeRtn = cropXAdd + cropXMult * (x * scale);
        yBeforeRtn = cropYAdd + cropYMult * (y * scale);
        yIdx = yBeforeRtn * yRowStride + xBeforeRtn * yPixelStride;
        uvIdx = (yBeforeRtn >> 1) * uvRowStride + xBeforeRtn * uvPixelStride;
        ui = uData[uvIdx];
        vi = vData[uvIdx];
        a0 = 1192 * (yData[yIdx] - 16);
        int ri = (a0 + 1634 * (vi - 128)) >> 10;
        int gi = (a0 - 832 * (vi - 128) - 400 * (ui - 128)) >> 10;
        int bi = (a0 + 2066 * (ui - 128)) >> 10;
        outData[wr++] = (clamp0255(ri) - normMeanRm255) / normStdRm255;
        outData[wg++] = (clamp0255(gi) - normMeanGm255) / normStdGm255;
        outData[wb++] = (clamp0255(bi) - normMeanBm255) / normStdBm255;
      }
    }
  }

  static void nativeImageYUV420CenterCropToFloatBufferLibyuv(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> yBuffer,
      const int yRowStride,
      const int yPixelStride,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> uBuffer,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> vBuffer,
      const int uRowStride,
      const int uvPixelStride,
      const int imageWidth,
      const int imageHeight,
      const int rotateCWDegrees,
      const int tensorWidth,
      const int tensorHeight,
      facebook::jni::alias_ref<jfloatArray> jnormMeanRGB,
      facebook::jni::alias_ref<jfloatArray> jnormStdRGB,
      facebook::jni::alias_ref<facebook::jni::JBuffer> outBuffer,
      const int outOffset) {
    const int halfImageWidth = (imageWidth + 1) / 2;
    const int halfImageHeight = (imageHeight + 1) / 2;

    // widthBeforeRtn, heightBeforeRtn{
    const int widthBeforeRtn = imageWidth;
    const int heightBeforeRtn = imageHeight;
    int widthAfterRtn = widthBeforeRtn;
    int heightAfterRtn = heightBeforeRtn;
    const bool oddRtn = rotateCWDegrees == 90 || rotateCWDegrees == 270;
    if (oddRtn) {
      widthAfterRtn = heightBeforeRtn;
      heightAfterRtn = widthBeforeRtn;
    }
    // }widthBeforeRtn, heightBeforeRtn

    int cropWidthAfterRtn = widthAfterRtn;
    int cropHeightAfterRtn = heightAfterRtn;
    if (tensorWidth * heightAfterRtn <= tensorHeight * widthAfterRtn) {
      cropWidthAfterRtn =
          std::floor(tensorWidth * heightAfterRtn / tensorHeight);
    } else {
      cropHeightAfterRtn =
          std::floor(tensorHeight * widthAfterRtn / tensorWidth);
    }
    // }cropWidthAfterRtn, cropHeightAfterRtn
    const int halfCropWidthAfterRtn = (cropWidthAfterRtn + 1) / 2;
    const int halfCropHeightAfterRtn = (cropHeightAfterRtn + 1) / 2;

    const int cropXAfterRtn = (widthAfterRtn - cropWidthAfterRtn) / 2;
    const int cropYAfterRtn = (heightAfterRtn - cropHeightAfterRtn) / 2;

    int cropXBeforeRtn = cropXAfterRtn;
    int cropYBeforeRtn = cropYAfterRtn;
    int cropWidthBeforeRtn = cropWidthAfterRtn;
    int cropHeightBeforeRtn = cropHeightAfterRtn;
    if (oddRtn) {
      std::swap(cropXBeforeRtn, cropYBeforeRtn);
      std::swap(cropWidthBeforeRtn, cropHeightBeforeRtn);
    }
    // }cropXBeforeRtn, cropYBeforeRtn
    // }cropWidthBeforeRtn, cropHeightBeforeRtn
    const int halfCropWidthBeforeRtn = (cropWidthBeforeRtn + 1) / 2;
    const int halfCropHeightBeforeRtn = (cropHeightBeforeRtn + 1) / 2;

    const uint32_t i420CropSize = cropWidthAfterRtn * cropHeightAfterRtn;
    std::vector<uint8_t> i420Crop;
    if (i420Crop.size() != i420CropSize) {
      i420Crop.resize(i420CropSize);
    }

    uint8_t* i420CropY = i420Crop.data();
    uint8_t* i420CropU = i420CropY + cropWidthBeforeRtn * cropWidthBeforeRtn;
    uint8_t* i420CropV =
        i420CropU + halfCropWidthBeforeRtn * halfCropHeightBeforeRtn;

    const auto retAndroid420ToI420 = libyuv::Android420ToI420(
        yBuffer->getDirectBytes() + cropYBeforeRtn * yRowStride +
            cropXBeforeRtn,
        yRowStride,
        uBuffer->getDirectBytes() + cropYBeforeRtn * uRowStride +
            cropXBeforeRtn * uvPixelStride,
        uRowStride,
        vBuffer->getDirectBytes() + cropYBeforeRtn * uRowStride +
            cropXBeforeRtn * uvPixelStride,
        uRowStride,
        uvPixelStride,
        i420CropY,
        imageWidth,
        i420CropU,
        halfImageWidth,
        i420CropV,
        halfImageHeight,
        cropWidthBeforeRtn,
        cropHeightBeforeRtn);
    assert(retAndroid420ToI420 == 0);

    // Rotate{
    const uint32_t i420CropRtdSize = cropWidthAfterRtn * cropHeightAfterRtn;
    std::vector<uint8_t> i420CropRtd;
    if (i420CropRtd.size() != i420CropRtdSize) {
      i420CropRtd.resize(i420CropRtdSize);
    }

    uint8_t* i420CropRtdY = i420CropRtd.data();
    uint8_t* i420CropRtdU =
        i420CropRtdY + cropWidthAfterRtn * cropWidthAfterRtn;
    uint8_t* i420CropRtdV =
        i420CropRtdU + halfCropWidthAfterRtn * halfCropWidthAfterRtn;
    libyuv::RotationMode rMode = libyuv::RotationMode::kRotate0;
    if (rotateCWDegrees == 90) {
      rMode = libyuv::RotationMode::kRotate90;
    } else if (rotateCWDegrees == 180) {
      rMode = libyuv::RotationMode::kRotate180;
    } else if (rotateCWDegrees == 270) {
      rMode = libyuv::RotationMode::kRotate270;
    }

    const auto retI420Rotate = libyuv::I420Rotate(
        i420CropY,
        cropWidthBeforeRtn,
        i420CropU,
        halfCropWidthBeforeRtn,
        i420CropV,
        halfCropHeightBeforeRtn,
        i420CropRtdY,
        cropWidthAfterRtn,
        i420CropRtdU,
        halfCropWidthAfterRtn,
        i420CropRtdV,
        halfCropHeightAfterRtn,
        cropWidthAfterRtn,
        cropHeightAfterRtn,
        rMode);
    assert(retI420Rotate == 0);
    // }Rotate

    // ARGBScale{
    const uint32_t argbTensorSize = 4 * tensorWidth * tensorHeight;
    ALOGI("JJJ argbTensorSize:%d i420CropSize:%d i420CropRtdSize:%d",
        argbTensorSize, i420CropSize, i420CropRtdSize);

    std::vector<uint8_t> argbTensor;
    if (argbTensor.size() != argbTensorSize) {
      argbTensor.resize(argbTensorSize);
    }

    uint8_t* argbData = argbTensor.data();
    const auto retYUVToARGBScaleClip = libyuv::YUVToARGBScaleClip(
        i420CropRtdY,
        cropWidthAfterRtn,
        i420CropRtdU,
        halfCropWidthAfterRtn,
        i420CropRtdV,
        halfCropHeightAfterRtn,
        libyuv::FOURCC_I420,
        cropWidthAfterRtn,
        cropWidthAfterRtn,
        argbData,
        4 * tensorWidth,
        libyuv::FOURCC_ARGB,
        tensorWidth,
        tensorHeight,
        0,
        0,
        tensorWidth,
        tensorHeight,
        libyuv::FilterMode::kFilterNone);
    assert(retYUVToARGBScaleClip == 0);
    // }ARGBScale

    static JNIEnv* jni = facebook::jni::Environment::current();
    float* outData = (float*)jni->GetDirectBufferAddress(outBuffer.get());

    int channelSize = tensorHeight * tensorWidth;
    int tensorInputOffsetG = channelSize;
    int tensorInputOffsetB = 2 * channelSize;
    const auto normMeanRGB = jnormMeanRGB->getRegion(0, 3);
    const auto normStdRGB = jnormStdRGB->getRegion(0, 3);

    for (int x = 0; x < tensorWidth; x++) {
      for (int y = 0; y < tensorHeight; y++) {
        int offset = y * tensorWidth + x;
        const int r = argbData[channelSize + offset];
        const int g = argbData[2 * channelSize + offset];
        const int b = argbData[3 * channelSize + offset];

        const float rf = ((r / 255.f) - normMeanRGB[0]) / normStdRGB[0];
        const float gf = ((g / 255.f) - normMeanRGB[1]) / normStdRGB[1];
        const float bf = ((b / 255.f) - normMeanRGB[2]) / normStdRGB[2];

        const int off = outOffset + offset;
        outData[off] = rf;
        outData[off + tensorInputOffsetG] = gf;
        outData[off + tensorInputOffsetB] = bf;
      }
    }
  }

  static void registerNatives() {
    javaClassStatic()->registerNatives({
        makeNativeMethod(
            "nativeImageYUV420CenterCropToFloatBuffer",
            PytorchVisionJni::nativeImageYUV420CenterCropToFloatBuffer),

        makeNativeMethod(
            "nativeImageYUV420CenterCropToFloatBufferLibyuv",
            PytorchVisionJni::nativeImageYUV420CenterCropToFloatBufferLibyuv),
    });
  }
};
} // namespace pytorch_vision_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { pytorch_vision_jni::PytorchVisionJni::registerNatives(); });
}