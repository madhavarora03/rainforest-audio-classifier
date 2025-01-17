#ifndef MODEL_HANDLER_H
#define MODEL_HANDLER_H

#include <Arduino.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

void setupModel();
void processAudio();
void inferAudio();

#endif // MODEL_HANDLER_H
