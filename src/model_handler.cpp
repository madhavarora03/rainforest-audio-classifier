#include "model_handler.h"
#include "model.h"
#include "audio.h"

constexpr int kTensorArenaSize = 137 * 1024;
uint8_t tensorArena[kTensorArenaSize];
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input;
TfLiteTensor *output;

void setupModel()
{
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
