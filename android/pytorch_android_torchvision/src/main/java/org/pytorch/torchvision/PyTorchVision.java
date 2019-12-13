package org.pytorch.torchvision;

import android.graphics.ImageFormat;
import android.media.Image;
import android.util.Log;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Locale;

public class PyTorchVision {
  static {
    System.loadLibrary("pytorch_vision_jni");
  }

  public static void imageYUV420CenterCropToFloatBuffer(
      final Image image,
      int rotateCWDegrees,
      final int tensorWidth,
      final int tensorHeight,
      float[] normMeanRGB,
      float[] normStdRGB,
      final Buffer outBuffer,
      final int outBufferOffset) {
    if (image.getFormat() != ImageFormat.YUV_420_888) {
      throw new IllegalArgumentException(
          String.format(
              Locale.US, "Image format %d != ImageFormat.YUV_420_888", image.getFormat()));
    }

    Log.i("AAA", String.format(
        "imageYUV420CenterCropToFloatBuffer image:%s format:%d width:%d height:%d",
        image,
        image.getFormat(),
        image.getWidth(),
        image.getHeight()
    ));

    Image.Plane Y = image.getPlanes()[0];
    Image.Plane U = image.getPlanes()[1];
    Image.Plane V = image.getPlanes()[2];

    nativeImageYUV420CenterCropToFloatBuffer(
        Y.getBuffer(),
        Y.getRowStride(),
        Y.getPixelStride(),
        U.getBuffer(),
        V.getBuffer(),
        U.getRowStride(),
        U.getPixelStride(),
        image.getWidth(),
        image.getHeight(),
        rotateCWDegrees,
        tensorWidth,
        tensorHeight,
        normMeanRGB,
        normStdRGB,
        outBuffer,
        outBufferOffset
    );
  }

  public static void imageYUV420CenterCropToFloatBufferLibyuv(
      final Image image,
      int rotateCWDegrees,
      final int tensorWidth,
      final int tensorHeight,
      float[] normMeanRGB,
      float[] normStdRGB,
      final Buffer outBuffer,
      final int outBufferOffset) {

    if (image.getFormat() != ImageFormat.YUV_420_888) {
      throw new IllegalArgumentException(
          String.format(
              Locale.US, "Image format %d != ImageFormat.YUV_420_888", image.getFormat()));
    }

    Image.Plane Y = image.getPlanes()[0];
    Image.Plane U = image.getPlanes()[1];
    Image.Plane V = image.getPlanes()[2];

    nativeImageYUV420CenterCropToFloatBufferLibyuv(
        Y.getBuffer(),
        Y.getRowStride(),
        Y.getPixelStride(),
        U.getBuffer(),
        V.getBuffer(),
        U.getRowStride(),
        U.getPixelStride(),
        image.getWidth(),
        image.getHeight(),
        rotateCWDegrees,
        tensorWidth,
        tensorHeight,
        normMeanRGB,
        normStdRGB,
        outBuffer,
        outBufferOffset
    );
  }

  private static native void nativeImageYUV420CenterCropToFloatBuffer(
      ByteBuffer yBuffer,
      int yRowStride,
      int yPixelStride,
      ByteBuffer uBuffer,
      ByteBuffer vBuffer,
      int uvRowStride,
      int uvPixelStride,
      int imageWidth,
      int imageHeight,
      int rotateCWDegrees,
      int tensorWidth,
      int tensorHeight,
      float[] normMeanRgb,
      float[] normStdRgb,
      Buffer outBuffer,
      int outBufferOffset
  );

  private static native void nativeImageYUV420CenterCropToFloatBufferLibyuv(
      ByteBuffer yBuffer,
      int yRowStride,
      int yPixelStride,
      ByteBuffer uBuffer,
      ByteBuffer vBuffer,
      int uvRowStride,
      int uvPixelStride,
      int imageWidth,
      int imageHeight,
      int rotateCWDegrees,
      int tensorWidth,
      int tensorHeight,
      float[] normMeanRgb,
      float[] normStdRgb,
      Buffer outBuffer,
      int outBufferOffset
  );

}
