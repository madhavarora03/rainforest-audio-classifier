#include <Arduino.h>
#include "audio.h"
#include "model_handler.h"

void setup()
{
  Serial.begin(115200);
  Serial.println("Starting...");
  setupAudio();
  setupModel();
}

void loop()
{
  recordAudio();
  processAudio();
  inferAudio();
  delay(2000);
}
