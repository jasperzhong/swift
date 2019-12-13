package org.pytorch.testapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.TextureView;
import android.view.ViewStub;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.PyTorchVision;
import org.pytorch.torchvision.TensorImageUtils;

import java.nio.FloatBuffer;

import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;

public class CameraActivity extends AppCompatActivity {
  private static final String TAG = BuildConfig.LOGCAT_TAG;
  private static final int TEXT_TRIM_SIZE = 4096;

  private static final int UNSET = 0;
  private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;
  private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};

  private long mLastAnalysisResultTime;

  protected HandlerThread mBackgroundThread;
  protected Handler mBackgroundHandler;
  protected Handler mUIHandler;

  private TextView mTextView;
  private StringBuilder mTextViewStringBuilder = new StringBuilder();

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_camera);
    mTextView = findViewById(R.id.text);
    mUIHandler = new Handler(getMainLooper());
    startBackgroundThread();

    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(
          this,
          PERMISSIONS,
          REQUEST_CODE_CAMERA_PERMISSION);
    } else {
      setupCameraX();
    }
  }

  @Override
  protected void onPostCreate(@Nullable Bundle savedInstanceState) {
    super.onPostCreate(savedInstanceState);
    startBackgroundThread();
  }

  protected void startBackgroundThread() {
    mBackgroundThread = new HandlerThread("ModuleActivity");
    mBackgroundThread.start();
    mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
  }

  @Override
  protected void onDestroy() {
    stopBackgroundThread();
    super.onDestroy();
  }

  protected void stopBackgroundThread() {
    mBackgroundThread.quitSafely();
    try {
      mBackgroundThread.join();
      mBackgroundThread = null;
      mBackgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e(TAG, "Error on stopping background thread", e);
    }
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
      if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
        Toast.makeText(
            this,
            "You can't use image classification example without granting CAMERA permission",
            Toast.LENGTH_LONG)
            .show();
        finish();
      } else {
        setupCameraX();
      }
    }
  }

  private static final int TENSOR_WIDTH = 224;
  private static final int TENSOR_HEIGHT = 224;

  private void setupCameraX() {
    final TextureView textureView = ((ViewStub) findViewById(R.id.camera_texture_view_stub))
        .inflate()
        .findViewById(R.id.texture_view);
    final PreviewConfig previewConfig = new PreviewConfig.Builder().build();
    final Preview preview = new Preview(previewConfig);
    preview.setOnPreviewOutputUpdateListener(new Preview.OnPreviewOutputUpdateListener() {
      @Override
      public void onUpdated(Preview.PreviewOutput output) {
        textureView.setSurfaceTexture(output.getSurfaceTexture());
      }
    });

    final ImageAnalysisConfig imageAnalysisConfig =
        new ImageAnalysisConfig.Builder()
            .setTargetResolution(new Size(TENSOR_WIDTH, TENSOR_HEIGHT))
            .setCallbackHandler(mBackgroundHandler)
            .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            .build();
    final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
    imageAnalysis.setAnalyzer(
        new ImageAnalysis.Analyzer() {
          @Override
          public void analyze(ImageProxy image, int rotationDegrees) {
            if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime < 500) {
              return;
            }

            if (BuildConfig.FORWARD_COUNT > 0 && forwardCount >= (BuildConfig.PERF_WARMUP_FORWARD_COUNT + BuildConfig.FORWARD_COUNT)) {
              Log.i(TAG, "===============================");
              for (int i = 0; i < mPerfStats.length; i++) {
                Log.i(TAG, mPerfStats[i].getStatString());
              }
              Log.i(TAG, "===============================");
              return;
            }

            final Result result = CameraActivity.this.analyzeImage(image, rotationDegrees);
            forwardCount++;

            if (result != null) {
              mLastAnalysisResultTime = SystemClock.elapsedRealtime();
              CameraActivity.this.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                  CameraActivity.this.handleResult(result);
                }
              });
            }
          }
        });

    CameraX.bindToLifecycle(this, preview, imageAnalysis);
  }

  private Module mModule;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;

  private PerfStat mPerfStatTensorPrepJ = new PerfStat("Java");
  private PerfStat mPerfStatTensorPrepJOpt = new PerfStat("JavaOpt");
  private PerfStat mPerfStatTensorPrepC = new PerfStat("C");
  private PerfStat mPerfStatTensorPrepL = new PerfStat("LibYUV");

  private PerfStat[] mPerfStats = new PerfStat[]{
      mPerfStatTensorPrepJOpt,
      mPerfStatTensorPrepC,
      mPerfStatTensorPrepJ,
      mPerfStatTensorPrepL,
  };

  private int forwardCount = 0;
  private final int BuildConfigTU_TYPE_COUNT = 4;

  @WorkerThread
  @Nullable
  protected Result analyzeImage(ImageProxy image, int rotationDegrees) {
    Log.i(TAG, String.format("analyzeImage(%s, %d)", image, rotationDegrees));
    if (mModule == null) {
      Log.i(TAG, "Loading module from asset '" + BuildConfig.MODULE_ASSET_NAME + "'");
      mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), BuildConfig.MODULE_ASSET_NAME);
      mInputTensorBuffer = Tensor.allocateFloatBuffer(3 * TENSOR_WIDTH * TENSOR_HEIGHT);
      mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, 3, TENSOR_WIDTH,
          TENSOR_HEIGHT});
    }

    final long startTime = SystemClock.elapsedRealtime();
    final int tensorPrepType = forwardCount % BuildConfigTU_TYPE_COUNT;
    if (tensorPrepType == 2) {
      TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
          image.getImage(), rotationDegrees,
          TENSOR_WIDTH, TENSOR_HEIGHT,
          TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
          TensorImageUtils.TORCHVISION_NORM_STD_RGB,
          mInputTensorBuffer, 0);
    } else if (tensorPrepType == 0) {
      TensorImageUtils.imageYUV420CenterCropToFloatBufferOpt(
          image.getImage(), rotationDegrees,
          TENSOR_WIDTH, TENSOR_HEIGHT,
          TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
          TensorImageUtils.TORCHVISION_NORM_STD_RGB,
          mInputTensorBuffer, 0);
    } else if (tensorPrepType == 1) {
      PyTorchVision.imageYUV420CenterCropToFloatBuffer(
          image.getImage(), rotationDegrees,
          TENSOR_WIDTH, TENSOR_HEIGHT,
          TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
          TensorImageUtils.TORCHVISION_NORM_STD_RGB,
          mInputTensorBuffer, 0);
    } else if (tensorPrepType == 3) {
      PyTorchVision.imageYUV420CenterCropToFloatBufferLibyuv(
          image.getImage(), rotationDegrees,
          TENSOR_WIDTH, TENSOR_HEIGHT,
          TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
          TensorImageUtils.TORCHVISION_NORM_STD_RGB,
          mInputTensorBuffer, 0);
    }
    final long tensorPrepDuration = SystemClock.elapsedRealtime() - startTime;
    if (forwardCount >= BuildConfig.PERF_WARMUP_FORWARD_COUNT) {
      mPerfStats[tensorPrepType].add(tensorPrepDuration);
    }
    Log.i(TAG, String.format("AAA tensorPrepDrtn:%d", tensorPrepDuration));

    final long moduleForwardStartTime = SystemClock.elapsedRealtime();
    final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
    final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;

    final float[] scores = outputTensor.getDataAsFloatArray();
    final long analysisDuration = SystemClock.elapsedRealtime() - startTime;

    return new Result(scores, moduleForwardDuration, analysisDuration);
  }

  @UiThread
  protected void handleResult(Result result) {
    int ixs[] = Utils.topK(result.scores, 1);
    String message = String.format("AAA %d fwdDrtn:%d class:%s",
        forwardCount,
        result.moduleForwardDuration,
        Constants.IMAGENET_CLASSES[ixs[0]]);
    Log.i(TAG, message);
    mTextViewStringBuilder.insert(0, '\n').insert(0, message);
    if (mTextViewStringBuilder.length() > TEXT_TRIM_SIZE) {
      mTextViewStringBuilder.delete(TEXT_TRIM_SIZE, mTextViewStringBuilder.length());
    }
    mTextView.setText(mTextViewStringBuilder.toString());
  }
}
