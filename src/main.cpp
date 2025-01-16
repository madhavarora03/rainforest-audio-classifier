#include <Arduino.h>
#include <driver/i2s.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

#define I2S_WS 40
#define I2S_SD 39
#define I2S_SCK 38

#define I2S_PORT I2S_NUM_0

#define bufferLen 64
#define RECORDING_LENGTH 8000
int16_t sBuffer[bufferLen];
int16_t audioBuffer[RECORDING_LENGTH];

constexpr int kTensorArenaSize = 137 * 1024;
uint8_t tensorArena[kTensorArenaSize];
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input;
TfLiteTensor *output;

void setup()
{
  Serial.begin(115200);
  Serial.println("Starting...");

  const i2s_config_t i2s_config = {
      .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = 8000,
      .bits_per_sample = i2s_bits_per_sample_t(16),
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = 0,
      .dma_buf_count = 8,
      .dma_buf_len = bufferLen,
      .use_apll = false};

  const i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = -1,
      .data_in_num = I2S_SD};

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_start(I2S_PORT);

  static tflite::MicroMutableOpResolver<20> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddDepthwiseConv2D();
  resolver.AddTranspose();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddMaxPool2D();
  resolver.AddConv2D();
  resolver.AddAveragePool2D();
  resolver.AddReduceMax();
  resolver.AddSub();
  resolver.AddExp();
  resolver.AddSum();
  resolver.AddLog();

  static tflite::MicroInterpreter static_interpreter(
      tflite::GetModel(M5_model), resolver, tensorArena, kTensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);

  delay(1000);
}

void recordAudio()
{
  Serial.println("Starting to record...");
  int sampleIndex = 0;
  while (sampleIndex < RECORDING_LENGTH)
  {
    size_t bytesIn = 0;
    i2s_read(I2S_PORT, &sBuffer, bufferLen, &bytesIn, portMAX_DELAY);
    int16_t samples_read = bytesIn / sizeof(int16_t);

    for (int i = 0; i < samples_read && sampleIndex < RECORDING_LENGTH; i++)
    {
      audioBuffer[sampleIndex++] = sBuffer[i];
    }
  }
  Serial.println("Ending recording...");
}

void processAudio()
{
  for (int i = 0; i < RECORDING_LENGTH; i++)
  {
    input->data.f[i] = audioBuffer[i] / 3000.0f;
  }
}

void inferAudio()
{
  if (interpreter->Invoke() != kTfLiteOk)
  {
    Serial.println("Inference failed!");
    return;
  }

  float chainsaw_score = output->data.f[0];
  float environment_score = output->data.f[1];
  int prediction = chainsaw_score > environment_score ? 0 : 1;

  Serial.print("Prediction: ");
  if (prediction == 0)
  {
    Serial.println("Chainsaw");
  }
  else
  {
    Serial.println("Environment");
  }
}

void loop()
{
  recordAudio();
  processAudio();
  inferAudio();
  for (int i = 0; i < RECORDING_LENGTH; i++)
  {
    Serial.print(output->data.f[i]);
    Serial.print(" ");
  }
  Serial.println(" ");
  delay(2000);
}