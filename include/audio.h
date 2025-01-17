#ifndef AUDIO_H
#define AUDIO_H

#include <Arduino.h>
#include <driver/i2s.h>

#define I2S_WS 40
#define I2S_SD 39
#define I2S_SCK 38
#define I2S_PORT I2S_NUM_0

#define bufferLen 64
#define RECORDING_LENGTH 8000

extern int16_t audioBuffer[RECORDING_LENGTH];

void setupAudio();
void recordAudio();

#endif // AUDIO_H
