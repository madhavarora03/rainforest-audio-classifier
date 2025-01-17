#include "audio.h"

int16_t sBuffer[bufferLen];
int16_t audioBuffer[RECORDING_LENGTH];

void setupAudio()
{
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
